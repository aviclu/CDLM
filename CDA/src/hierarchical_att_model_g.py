import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet
import torch.nn.functional as F

class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, tune, pretrained_word2vec_path,
                 max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.cos = nn.CosineSimilarity()
        self.m = nn.Sigmoid()
        self.fd = nn.Linear(4 * sent_hidden_size, sent_hidden_size)
        self.ff = nn.Linear(sent_hidden_size, 1)
        self.r = nn.ReLU()
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.word_att_net = WordAttNet(pretrained_word2vec_path, tune, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

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
        return output_x, output_y, a_x[:,:,-1]

    def encode(self, input):
        output_list = []
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0),self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        #print(output.shape)
        output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)
        #print(output.shape)
        return output,torch.cat(output_list, 0)

    def forward(self, input_1, input_2):
        #print(input.shape)
        input_1 = input_1.permute(1, 0, 2)
        input_2 = input_2.permute(1, 0, 2)
        output_1_doc, output_1 = self.encode(input_1)
        #print(input_2.size(1))
        self._init_hidden_state(input_2.size(1))
        #self._init_hidden_state()
        output_2_doc, output_2 = self.encode(input_2)
        #output = torch.sum(output_1*output_2, 1)
        #self.prediction = torch.exp(-torch.norm((self.encoding_a - self.encoding_b), 1, 1))
        if self.training:
            output = torch.cat((output_1_doc, output_2_doc), dim=1)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            #print(output.shape)
            #output = output.reshape(-1)
            return self.m(output)
        else:
            output = torch.cat((output_1_doc, output_2_doc), dim=1)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            #print(output.shape)
            #output = output.reshape(-1)
            output = self.m(output)
            '''
            output_1_list.append(output_1.unsqueeze(0))
            output_1 = torch.cat(output_1_list, 0)
            output_2 = torch.unsqueeze(output_2, 0)
            output_2 = output_2.expand(output_1.size())
            output = torch.cat((output_1, output_2), dim=2)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            return self.m(output)
            '''
            output_1 = output_1.permute(1,0,2)
            output_2 = output_2.permute(1,0,2)
            output_1_doc = output_1_doc.unsqueeze(1)
            output_2_doc = output_2_doc.unsqueeze(1)
            output_1 = torch.cat((output_1, output_1_doc), dim=1)
            output_2 = torch.cat((output_2, output_2_doc), dim=1)
            _, _, Att = self.graph_match(output_1, output_2) 
            output = output.unsqueeze(0)
            Att = Att.permute(1,0)
            Att = Att[:-1]
            Att = torch.cat((Att, output), dim=0)
            return Att


