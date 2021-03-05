import numpy as np 
import torch
import torch.nn as nn
import random
import os
import torch.nn.functional as F

class Bert_sg_av(nn.Module):
    def __init__(self, vector_size, sent_hidden_size=50, batch_size=256):
        super(Bert_sg_av, self).__init__()
        self.m = nn.Sigmoid()
        self.fd = nn.Linear(2*sent_hidden_size, sent_hidden_size)
        self.mlp_graph = nn.Linear(2*vector_size, sent_hidden_size)
        self.ff = nn.Linear(sent_hidden_size, 1)
        self.r = nn.ReLU()
        self.batch_size = batch_size 
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

    def graph_match(self, input_1, input_2):
        #print(input_1.shape)
        a = torch.matmul(input_1, input_2.permute(0,2,1)) # [128, 6, 100] \times [128, 100, 6] -> [128, 6, 6]
        a_x = F.softmax(a, dim=1)  # i->j [128, 6, 6]
        a_y = F.softmax(a, dim=0)  # j->i [128, 6, 6]
        #print(a_y.shape)
        #print(a_x.shape)
        attention_x = torch.matmul(a_x.transpose(1, 2), input_1)
        attention_y = torch.matmul(a_y, input_2) # [128, 6, 100]
        output_x = torch.cat((input_1, attention_y), dim=2)
        output_y = torch.cat((input_2, attention_x), dim=2)
        #output_x = 2*input_1-attention_y
        #output_y = 2*input_2-attention_x
        output_x = self.mlp_graph(output_x)
        output_y = self.mlp_graph(output_y)
        return output_x, output_y

    def forward(self, output_1, output_2):
        if self.training:
            print(output_1.type())
            output_1_doc = torch.mean(output_1, dim=1, keepdim=True)
            output_2_doc = torch.mean(output_2, dim=1, keepdim=True)
            output_1 = torch.cat((output_1, output_1_doc), dim=1)
            output_2 = torch.cat((output_2, output_2_doc), dim=1)
            output_1, output_2 = self.graph_match(output_1, output_2)
            output_1 = output_1[:,-1,:].squeeze() # [128,6,100] -> [128,1,100]
            output_2 = output_2[:,-1,:].squeeze() 
            output = torch.cat((output_1, output_2), dim=1)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            #print(output.shape)
            #output = output.reshape(-1)
            return self.m(output)     
        else:      
            output_1_doc = torch.mean(output_1, dim=1, keepdim=True)
            output_2_doc = torch.mean(output_2, dim=1, keepdim=True)
            output_1 = torch.cat((output_1, output_1_doc), dim=1)
            output_2 = torch.cat((output_2, output_2_doc), dim=1)
            output_1, output_2 = self.graph_match(output_1, output_2)
            output_1 = output_1.permute(1,0,2)
            output_2 = output_2[:,-1,:].squeeze()
            output_2 = torch.unsqueeze(output_2, 0)
            output_2 = output_2.expand(output_1.size())
            output = torch.cat((output_1, output_2), dim=2)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            return self.m(output)
            

