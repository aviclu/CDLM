import numpy as np 
import torch
import torch.nn as nn
import random
import os

class Bert_cls(nn.Module):
    def __init__(self, vector_size):
        super(Bert_cls, self).__init__()
        self.m = nn.Sigmoid()
        self.fd = nn.Linear(vector_size, vector_size)
        self.ff = nn.Linear(vector_size, 1)
        self.r = nn.ReLU()
        #self.bert_model = bert_model
        #self.tokenizer = tokenizer
    def bert_represent(self, text):
        # Tokenized input
        # text = "[CLS] I got restricted because Tom reported my reply [SEP]"
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        input_ids = input_ids.to('cuda')
        with torch.no_grad():
            outputs = self.model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        return last_hidden_states
    def forward(self, output):
        #print(input.shape)
        output = self.r(self.fd(output))
        output = torch.squeeze(self.ff(output))
        return self.m(output)