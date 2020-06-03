import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class BERTforNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        '''
        input_ids:  (bs, max_seq_length)
        attention_mask:  (bs, max_seq_length)
        token_type_ids:  (bs, max_seq_length)

        return (bs, max_seq_length, num_labels)
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