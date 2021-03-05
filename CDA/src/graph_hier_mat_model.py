import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet
import torch.nn.functional as F

class HierGraphNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, tune, pretrained_word2vec_path,
                 max_sent_length, max_word_length):
        super(HierGraphNet, self).__init__()
        self.batch_size = batch_size
        self.cos = nn.CosineSimilarity()
        self.m = nn.Sigmoid()
        self.fd = nn.Linear(4 * sent_hidden_size, sent_hidden_size)
        self.ff = nn.Linear(sent_hidden_size, 1)
        self.mlp_graph = nn.Linear(4* sent_hidden_size, 2*sent_hidden_size)
        self.r = nn.ReLU()
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
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

    def encode(self, input):
        output_list = []
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0),self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)
        output_list.append(output.unsqueeze(0))
        output = torch.cat(output_list, 0) #[6, 128 ,100]
        return output.permute(1, 0, 2)
    
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
        output_x = self.mlp_graph(output_x)
        output_y = self.mlp_graph(output_y)
        return output_x, output_y


    def forward(self, input_1, input_2, in_test=False):
        #print(input.shape)
        input_1 = input_1.permute(1, 0, 2)
        input_2 = input_2.permute(1, 0, 2)
        output_1 = self.encode(input_1)
        self._init_hidden_state(input_2.size(1))
        #self._init_hidden_state()
        output_2 = self.encode(input_2)
        #output_1 = torch.cat(output_1, 0)
        #output_2 = torch.cat(output_2, 0)
        #output = torch.cat(output_list, 0)
        #print(output_1[0])
        #print(output_2[0])
        output_1, output_2 = self.graph_match(output_1, output_2)
        if in_test:
            output = torch.cat((output_1, output_2), dim=2)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            output = self.m(output) #[128,6]
            return output
        else:
            output_1 = output_1[:,-1,:].squeeze() # [128,6,100] -> [128,1,100]
            output_2 = output_2[:,-1,:].squeeze() 
            output = torch.cat((output_1, output_2), dim=1)
            output = self.r(self.fd(output))
            output = torch.squeeze(self.ff(output))
            return self.m(output)