import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
import torch.nn.functional as F

class DHierGraphNet(nn.Module):
    def __init__(self, vector_size, sent_hidden_size=50, batch_size=256):
        super(DHierGraphNet, self).__init__()
        self.batch_size = batch_size
        self.m = nn.Sigmoid()
        self.fd = nn.Linear(4*sent_hidden_size, sent_hidden_size)
        self.mlp_graph = nn.Linear(4* sent_hidden_size, 2*sent_hidden_size)
        self.ff = nn.Linear(sent_hidden_size, 1)
        self.r = nn.ReLU()
        self.sent_hidden_size = sent_hidden_size
        self.sent_gru = nn.GRU(vector_size, sent_hidden_size, bidirectional=True)
        self.sent_att_net = SentAttNet(sent_hidden_size, sent_hidden_size)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def encode(self, input_1, input_2):
        output_1, hidden = self.sent_gru(input_1, self.sent_hidden_state)
        output_2, hidden = self.sent_gru(input_2, self.sent_hidden_state)
        #print(output.shape)
        sents_1 = self.graph_match_one_direction(output_2, output_1)
        sents_2 = self.graph_match_one_direction(output_1, output_2)
        #print(output.shape)
        doc_1, _ = self.sent_att_net(sents_1, self.sent_hidden_state)
        doc_2, _ = self.sent_att_net(sents_2, self.sent_hidden_state)

        output_1 = torch.cat((sents_1,doc_1.unsqueeze(0)),0)
        output_2 = torch.cat((sents_2,doc_1.unsqueeze(0)),0)
        #output = torch.cat(output_list, 0)
        return output_1.permute(1,0,2), output_2.permute(1,0,2)

    def graph_match_one_direction(self, input_1, input_2):
        input_1 = input_1.permute(1,0,2)
        input_2 = input_2.permute(1,0,2)
        a = torch.matmul(input_1, input_2.permute(0,2,1)) # [128, 6, 100] \times [128, 100, 6] -> [128, 6, 6]
        a_x = F.softmax(a, dim=1)  # i->j [128, 6, 6]
        a_y = F.softmax(a, dim=0)  # j->i [128, 6, 6]
        #print(a_y.shape)
        #print(a_x.shape)
        attention_x = torch.matmul(a_x.transpose(1, 2), input_1)
        output_y = torch.cat((input_2, attention_x), dim=2)
        output_y = self.mlp_graph(output_y)
        output_y = output_y.permute(1,0,2)
        return output_y
        
    def graph_match(self, input_1, input_2):
        #print(input_1.shape)
        a = torch.matmul(input_1, input_2.permute(0,2,1)) # [128, 6, 100] \times [128, 100, 6] -> [128, 6, 6]
        a_x = F.softmax(a, dim=1)  # i->j [128, 6, 6]
        a_y = F.softmax(a, dim=0)  # j->i [128, 6, 6]
        #print(a_y.shape)
        #print(a_x.shape)
        attention_x = torch.matmul(a_x.transpose(1, 2), input_1)
        attention_y = torch.matmul(a_y, input_2) # [128, 6, 100]
        output_x = torch.cat((input_1, input_1-attention_y), dim=2)
        output_y = torch.cat((input_2, input_2-attention_x), dim=2)
        #output_x = 2*input_1-attention_y
        #output_y = 2*input_2-attention_x
        output_x = self.mlp_graph(output_x)
        output_y = self.mlp_graph(output_y)
        return output_x, output_y

    def forward(self, input_1, input_2):
        input_1 = input_1.permute(1, 0, 2)
        input_2 = input_2.permute(1, 0, 2)
        output_1, output_2 = self.encode(input_1, input_2)
        output_1, output_2 = self.graph_match(output_1, output_2) 
        #output = torch.sum(output_1*output_2, 1)
        #self.prediction = torch.exp(-torch.norm((self.encoding_a - self.encoding_b), 1, 1))
        if self.training:
            output_1 = output_1[:,-1,:].squeeze() # [128,6,100] -> [128,1,100]
            output_2 = output_2[:,-1,:].squeeze() 
            output = torch.cat((output_1, output_2), dim=1)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            #print(output.shape)
            #output = output.reshape(-1)
            return self.m(output)
        else:
            output_1 = output_1.permute(1,0,2)
            output_2 = output_2[:,-1,:].squeeze()
            output_2 = torch.unsqueeze(output_2, 0)
            output_2 = output_2.expand(output_1.size())
            output = torch.cat((output_1, output_2), dim=2)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            return self.m(output)

