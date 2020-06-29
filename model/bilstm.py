import torch
import torch.nn as nn
from .crf import CRF

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, hidden_dropout_prob=0.2):
        super(BiLSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, batch_first=True,
                            num_layers=1, bidirectional=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2label = nn.Linear(hidden_dim, num_labels)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.word_embeds.weight)
        nn.init.xavier_normal_(self.hidden2label.weight)
        for weights in [self.lstm.weight_hh_l0, self.lstm.weight_ih_l0]:
            nn.init.orthogonal_(weights)

    def forward(self, input_ids, attention_mask=None):
        '''
        input_ids:  (batch_size, max_seq_length)

        return: (batch_size, max_seq_length, num_labels)
        '''
        # (batch_size, max_seq_length, word_embedding_dim)
        embeds = self.word_embeds(input_ids)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2label(lstm_out)
        return lstm_feats


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, hidden_dropout_prob=0.2, use_crf=False):
        super(BiLSTM_CRF, self).__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.bilstm = BiLSTM(vocab_size, embedding_dim, hidden_dim, num_labels, hidden_dropout_prob)

        if self.use_crf:
            self.crf = CRF(num_labels)
    
    def forward(self, input_ids, attention_mask=None, pred_mask=None, input_labels=None):
        '''
        input_ids:  (batch_size, max_seq_length)
        attention_mask:  (batch_size, max_seq_length)
        pred_mask: (batch_size, max_seq_length)
        input_labels: (batch_size, )

        return: (batch_size, max_seq_length)
        '''
        # (batch_size, max_seq_length, num_labels)
        emissions = self.bilstm(input_ids)
        if self.use_crf:
            preds = self.crf.decode(emissions, pred_mask)
            preds = [seq + [-1]*(pred_mask.size(1)-len(seq)) for seq in preds]
            preds = torch.tensor(preds).to(input_ids.device)
        else:
            preds = torch.argmax(emissions, dim=-1)
        
        output = (preds, )

        if input_labels is not None:
            if self.use_crf:
                loss = -1*self.crf(emissions, input_labels, attention_mask)
            else:
                loss_fct = nn.CrossEntropyLoss()
                if pred_mask is not None:
                    pred_pos = pred_mask.view(-1) == 1
                    logits = emissions.view(-1, self.num_labels)[pred_pos]
                    input_labels = input_labels.view(-1)[pred_pos]
                    loss = loss_fct(logits, input_labels)
                else:
                    loss = loss_fct(emissions.view(-1, self.num_labels), input_labels.view(-1))
            output += (loss, )

        return output #(preds, loss)