import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul
import pandas as pd
import numpy as np
import csv

class WordAttNet(nn.Module):
    def __init__(self, vocab, hidden_size=50):
        super(WordAttNet, self).__init__()
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        #self.lookup = nn.Embedding(num_embeddings=len(vocab), embedding_dim=hidden_size, padding_idx=0)
        self.embeddings = torch.FloatTensor(list(vocab.values()))
        # embedding
        self.lookup = nn.Embedding(self.embeddings.size(0), self.embeddings.size(1))
        self.lookup.weight = nn.Parameter(self.embeddings)
        self.lookup.weight.requires_grad = True

        self.gru = nn.GRU(self.embeddings.size(1), hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_bias.data.normal_(mean, std)
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        output = self.lookup(input)
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        #print(output.shape)
        output = matrix_mul(output, self.context_weight).permute(1,0)
        output = F.softmax(output)
        output = element_wise_mul(f_output,output.permute(1,0))

        return output, h_output


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
