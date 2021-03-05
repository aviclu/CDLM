import torch
import random
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul

seed=42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_bias.data.normal_(mean, std)
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        f_output, h_output = self.gru(input, hidden_state)
        #print('------------------>')
        #print(f_output.shape)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        #print(output.shape)
        #print()
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        #output = self.fc(output)

        return output, h_output


if __name__ == "__main__":
    abc = SentAttNet()
