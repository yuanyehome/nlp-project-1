import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from hyperParameters import params
import time
from nn_utils import get_parameters

class_num = 9


class DNN(nn.Module):
    """
    A deeper feedforward neural network with an embedding layer and three fc layers.
    """

    def __init__(self):
        super(DNN, self).__init__()
        self.embed = nn.Embedding(params['vocab_size'], params['embed_dim'])
        embed_dim = params['embed_dim']
        dropout = params['dropout']
        self.fc1 = nn.Linear(embed_dim, 50)
        self.fc2 = nn.Linear(50, 16)
        self.fc3 = nn.Linear(16, class_num)

    def forward(self, x):
        x = self.embed(x)
        x = F.adaptive_avg_pool2d(x.unsqueeze(
            1), (1, params['embed_dim'])).squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LSTM(nn.Module):
    """
    A simple LSTM with an embedding layer and a LSTM layer and two fc layers.
    """

    def __init__(self):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(params['vocab_size'], params['embed_dim'])
        self.embed_dim = params['embed_dim']
        self.hidden_dim = 50
        self.lstm = nn.LSTM(params['embed_dim'], 50)
        self.fc1 = nn.Linear(50, 16)
        self.fc2 = nn.Linear(16, class_num)

    def forward(self, x):
        x = self.embed(x)
        lstm_out, self.hidden = self.lstm(
            x.view(params['pad_len'], -1, self.embed_dim))
        lstm_out = lstm_out.view(-1, params['pad_len'], self.hidden_dim)
        out = F.adaptive_avg_pool2d(lstm_out.unsqueeze(
            1), (1, self.hidden_dim)).squeeze()
        res = F.relu(self.fc1(out))
        res = self.fc2(res)
        return res


if __name__ == "__main__":
    dnn = DNN()
    lstm = LSTM()
    get_parameters(dnn)
    get_parameters(lstm)
