import numpy as np 
import torch
import torch.nn as nn
import random
import os
import torch.nn.functional as F

class Bert_random(nn.Module):
    def __init__(self, vector_size, sent_hidden_size=50, batch_size=256):
        super(Bert_random, self).__init__() 
        self.m = nn.Sigmoid()
        self.fd = nn.Linear(2*sent_hidden_size, sent_hidden_size)
        self.ff = nn.Linear(sent_hidden_size, 1)
        self.mlp = nn.Linear(vector_size, sent_hidden_size)
        self.r = nn.ReLU()
        self.batch_size = batch_size 
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        #self.bert_model = bert_model
        #self.tokenizer = tokenizer
    
    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.sent_hidden_state = torch.zeros(2, batch_size,10)
        if torch.cuda.is_available():
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def bert_represent(self, text):
        # Tokenized input
        # text = "[CLS] I got restricted because Tom reported my reply [SEP]"
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        input_ids = input_ids.to('cuda')
        with torch.no_grad():
            outputs = self.model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        return last_hidden_states

    def forward(self, output_1, output_2):
        if self.training:
            print(output_1.type())
            output_1_doc = torch.mean(output_1, dim=1, keepdim=False)
            output_2 = torch.mean(output_2, dim=1, keepdim=False)
            output = torch.cat((output_1_doc, output_2), dim=1)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            return self.m(output)
        else:      
            output_1_doc = torch.mean(output_1, dim=1, keepdim=True)
            output_2 = torch.mean(output_2, dim=1, keepdim=False)
            output_1 = output_1.permute(1,0,2)
            output_1_doc = output_1_doc.permute(1,0,2)
            #print(output_1.size())
            #print(output_1_doc.size())
            output_1 = torch.cat((output_1, output_1_doc), 0)
            output_2 = torch.unsqueeze(output_2, 0)
            output = self.cos(output_1, output_2)
            output = torch.rand(output.size(), dtype=output.dtype, layout=output.layout, device=output.device)
            return output

