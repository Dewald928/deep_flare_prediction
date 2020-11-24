import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import math


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Simple1DConv(nn.Module):
    def __init__(self, n_inputs, n_filters, num_levels, kernel_size=2,
                 dropout=0.2, dilation=1):
        super(Simple1DConv, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_filters, kernel_size, stride=1,
                               padding=self.padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size, stride=1,
                               padding=self.padding, dilation=dilation)
        self.chomp1 = Chomp1d(self.padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.linear = nn.Linear(n_filters, 2)

        self.downsample = nn.Conv1d(n_inputs, n_filters,
                                    1) if n_inputs != n_filters else None

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in',
                                nonlinearity='relu')
        # self.conv1.weight = []

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)  # for causal convolution
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.linear(out[:, :, -1])  # last value in sequence
        return out
