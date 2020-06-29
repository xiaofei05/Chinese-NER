import torch
import torch.nn as nn
import copy
from transformers import BertModel, BertPreTrainedModel
from .crf import CRF


class BERTforNER_CRF(BertPreTrainedModel):

    def __init__(self, config, use_crf=False):
        super(BERTforNER_CRF, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF(self.config.num_labels)

        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, pred_mask=None, input_labels=None):
        '''
        input_ids:  (batch_size, max_seq_length)
        attention_mask:  (batch_size, max_seq_length)
        token_type_ids:  (batch_size, max_seq_length)
        pred_mask: (batch_size, max_seq_length)
        input_labels: (batch_size, )

        return: (batch_size, max_seq_length), loss
        '''
        # (batch_size, max_seq_length, hidden_size)
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # (batch_size, max_seq_length, num_labels)
        emissions = self.classifier(sequence_output)        

        if self.use_crf:
            # crf model has a problem with mask.
            # to be fixed.
            crf_pred_mask = copy.deepcopy(pred_mask).byte()
            crf_pred_mask[:, 0] = 1
            crf_pred_mask[:, -1] = 1
            crf_seq = self.crf.decode(emissions, crf_pred_mask)
            crf_seq = [seq[1:-1] + [-1]*(pred_mask.size(1)-len(seq)+2) for seq in crf_seq]
            pred = torch.tensor(crf_seq).to(input_ids.device)
        else:
            pred = torch.argmax(emissions, dim=-1)

        
        outputs = (pred, ) # (batch_size, max_seq_length)

        if input_labels is not None:
            if self.use_crf:
                loss = -1*self.crf(emissions, input_labels, crf_pred_mask)
            else:
                loss_fct = nn.CrossEntropyLoss()
                if pred_mask is not None:
                    pred_pos = pred_mask.view(-1) == 1
                    logits = emissions.view(-1, self.config.num_labels)[pred_pos]
                    labels = input_labels.view(-1)[pred_pos]
                    loss = loss_fct(logits, labels)
                else:
                    loss = loss_fct(emissions.view(-1, self.config.num_labels), input_labels.view(-1))
            
            outputs += (loss, )
            
        return outputs # (pred, loss)