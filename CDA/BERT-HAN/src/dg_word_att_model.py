import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul
import pandas as pd
import numpy as np
import csv

class WordAttNet(nn.Module):
    def __init__(self,hidden_size=50):
        super(WordAttNet, self).__init__()
        self.mlp_graph = nn.Linear(768, 2*sent_hidden_size)
        self.r = nn.ReLU()
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_bias.data.normal_(mean,std)
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input):

        f_output = self.mlp_graph(input) 
        f_output = self.r(f_output)
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        #print(output.shape)
        output = matrix_mul(output, self.context_weight).permute(1,0)
        output = F.softmax(output,dim=1)
        output = element_wise_mul(f_output,output.permute(1,0))
        return f_output, output


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
