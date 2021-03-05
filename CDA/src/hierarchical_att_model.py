import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet


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

    def encode(self, input):
        output_list = []
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0),self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        #print(output.shape)
        output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)
        #print(output.shape)
        return output

    def forward(self, input_1, input_2):
        #print(input.shape)
        input_1 = input_1.permute(1, 0, 2)
        input_2 = input_2.permute(1, 0, 2)
        output_1 = self.encode(input_1)
        #print(input_2.size(1))
        self._init_hidden_state(input_2.size(1))
        #self._init_hidden_state()
        output_2 = self.encode(input_2)
        #print(output_1[0])
        #print(output_2[0])
        '''
        output = self.cos(output_1, output_2)
        print(output)
        return output, output_1, output_2
        '''
        #output = torch.sum(output_1*output_2, 1)
        #self.prediction = torch.exp(-torch.norm((self.encoding_a - self.encoding_b), 1, 1))
        output = torch.cat((output_1, output_2), dim=1)
        output = self.r(self.fd(output))
        output = torch.squeeze(self.ff(output))
        #print(output.shape)
        #output = output.reshape(-1)
        return self.m(output)
