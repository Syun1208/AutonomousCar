from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 2, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512, 128)  # stride 1: 2304, 2:512
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.conv1(x)  # 48.48.32     #32.32.8
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)  # 24.24.32     #16.16.16
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)  # 12.12.64     #8.8.32
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2)  # 6.6.64  #4.4.32
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)  # 2304
        x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output