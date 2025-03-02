from torch import nn
from transformers import BertForSequenceClassification, BertConfig

    
class BertSequenceModel(nn.Module):
    def __init__(self, args, device):
        super(BertSequenceModel, self).__init__()

        self.num_labels = args.way
        self.bert_pretrain_path = args.bert_pretrain_path
        self.config = BertConfig.from_pretrained(self.bert_pretrain_path, num_labels=self.num_labels)
        self.bert = BertForSequenceClassification.from_pretrained(self.bert_pretrain_path, num_labels=self.num_labels)  

        self.device = device
        self.bert.to(self.device)

    def forward(self, input_ids, attention_mask, classification_label_id):
 
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=classification_label_id)

        return out[0], out[1]
 
