import torch
import torch.nn as nn

class CRF(nn.Module):

    def __init__(self, num_labels, pad_tag=-100):
        super(CRF, self).__init__()

        self.num_labels = num_labels
        self.pad_tag = pad_tag

        self.start_transitions = nn.Parameter(torch.empty(num_labels))
        self.end_transitions = nn.Parameter(torch.empty(num_labels))
        self.transitions = nn.Parameter(torch.empty(num_labels, num_labels))

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)


    def get_loss(self, emissions, labels, mask=None):
        '''
        emissions: (batch_size, max_seq_length, num_labels)
        labels: (batch_size, max_seq_length)
        mask: (batch_size, max_seq_length)  0 or 1
        '''
        emissions = emissions.transpose(0, 1)
        labels = labels.transpose(0, 1)

        if mask is None:
            mask = labels.ne(self.pad_tag).byte().to(b.device)
        else:
            mask = mask.transpose(0, 1)

        if mask.dtype != torch.uint8:
            mask = mask.byte()

        # shape: (batch_size,)
        golden_score = self.sequences_score(emissions, labels, mask)
        # shape: (batch_size,)
        total_score = self.all_sequences_score(emissions, mask)
        # shape: (batch_size,)
        llh = total_score - golden_score
        return llh.sum() / mask.float().sum()

    def forward(self, emissions, mask=None):
        '''
        emissions: (batch_size, max_seq_length, num_labels)
        mask: (batch_size, max_seq_length) 0 or 1

        return: (batch_size, max_seq_length)
        '''
        emissions = emissions.transpose(0, 1)
        
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8,
                              device=emissions.device)
        else:
            mask = mask.transpose(0, 1)

        if mask.dtype != torch.uint8:
            mask = mask.byte()
        return self._viterbi_decode(emissions, mask)


    def sequences_score(self, emissions, labels, mask):
        '''
        emissions: (max_seq_length, batch_size, num_labels)
        labels: (max_seq_length, batch_size)
        mask: (max_seq_length, batch_size)
        '''
        max_seq_length, batch_size = labels.shape
        labels = labels * mask
        mask = mask.float()
        score = self.start_transitions[labels[0]]
        score += emissions[0, torch.arange(batch_size), labels[0]]
        for i in range(1, max_seq_length):
            score += self.transitions[labels[i - 1], labels[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), labels[i]] * mask[i]
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_labels = labels[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_labels]
        return score

    def all_sequences_score(self, emissions, mask):
        '''
        emissions: (max_seq_length, batch_size, num_labels)
        mask: (max_seq_length, batch_size)
        '''
        max_seq_length = emissions.size(0)
        score = self.start_transitions + emissions[0]

        for i in range(1, max_seq_length):
            # shape: (batch_size, num_labels, 1)
            broadcast_score = score.unsqueeze(2)
            # shape: (batch_size, 1, num_labels)
            broadcast_emissions = emissions[i].unsqueeze(1)
            # shape: (batch_size, num_labels, num_labels)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # shape: (batch_size, num_labels)
            next_score = torch.logsumexp(next_score, dim=1)
            # shape: (batch_size, num_labels)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        # shape: (batch_size, num_labels)
        score += self.end_transitions
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, mask):
        '''
        emissions: (max_seq_length, batch_size, num_labels)
        mask: (max_seq_length, batch_size)

        return: (batch_size, max_seq_length)
        '''
        device = emissions.device
        max_seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((max_seq_length, batch_size, self.num_labels),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_labels),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((max_seq_length, batch_size), self.pad_tag,
                             dtype=torch.long, device=device)

        for i in range(1, max_seq_length):
            broadcast_score = score.unsqueeze(2)
            # shape: (batch_size, 1, num_labels)
            broadcast_emission = emissions[i].unsqueeze(1)
            # shape: (batch_size, num_labels, num_labels)
            next_score = broadcast_score + self.transitions + broadcast_emission
            # shape: (batch_size, num_labels)
            next_score, indices = next_score.max(dim=1)
            # shape: (batch_size, num_labels)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices
        # shape: (batch_size, num_labels)
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_labels),
                             end_tag.view(-1, 1, 1).expand(-1, 1, self.num_labels))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_labels_arr = torch.zeros((max_seq_length, batch_size),
                                    dtype=torch.long, device=device)
        best_labels = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(max_seq_length - 1, -1, -1):
            best_labels = torch.gather(history_idx[idx], 1, best_labels)
            best_labels_arr[idx] = best_labels.data.view(batch_size)

        return torch.where(mask, best_labels_arr, oor_tag).transpose(0, 1)