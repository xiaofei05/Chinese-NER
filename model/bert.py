import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from .crf import CRF

class BERTforNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BERTforNER, self).__init__(config)
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.name, config=config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        '''
        input_ids:  (batch_size, max_seq_length)
        attention_mask:  (batch_size, max_seq_length)
        token_type_ids:  (batch_size, max_seq_length)

        return: (batch_size, max_seq_length, num_labels)
        '''
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

class BERTforNER_CRF(BertPreTrainedModel):

    def __init__(self, config, use_crf = False):
        super(BERTforNER_CRF, self).__init__(config)
        self.name = "bert"
        self.config = config
        self.use_crf = use_crf

        self.bert4ner = BERTforNER(config)
        if self.use_crf:
            self.crf = CRF(self.config.num_labels, pad_tag=-100)
        else:
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    

    def get_loss(self, input_ids, labels, attention_mask=None, token_type_ids=None):
        '''
        input_ids:  (batch_size, max_seq_length)
        labels: (batch_size, max_seq_length)
        attention_mask:  (batch_size, max_seq_length)
        token_type_ids:  (batch_size, max_seq_length)

        return: (batch_size, max_seq_length, num_labels)
        '''
        emissions = self.bert4ner(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )

        if self.use_crf:
            if attention_mask is not None:
                attention_mask = attention_mask[:, 1:]
            return self.crf.get_loss(emissions[:, 1:, :], labels[:, 1:], attention_mask)
        
        return self.loss_fct(emissions.view(-1, emissions.size()[-1]), labels.view(-1))
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        '''
        input_ids:  (batch_size, max_seq_length)
        attention_mask:  (batch_size, max_seq_length)
        token_type_ids:  (batch_size, max_seq_length)

        return: (batch_size, max_seq_length)
        '''
        # (batch_size, max_seq_length, num_labels)
        emissions = self.bert4ner(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        if self.use_crf:
            if attention_mask is not None:
                attention_mask = attention_mask[:, 1:]
            crf_seq = self.crf(emissions[:, 1:, :], attention_mask)
            cls_labels = torch.LongTensor(emissions.size(0), 1).fill_(-100).to(crf_seq.device)
            return torch.cat([cls_labels, crf_seq], dim=1)
        # (batch_size, max_seq_length)
        return torch.argmax(emissions, dim=-1)
