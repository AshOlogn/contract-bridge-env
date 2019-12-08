import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, dim=52*7+35+2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(dim, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 13)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class PG(nn.Module):

    def __init__(self, dim=52*7+35+2):
        super(PG, self).__init__()
        self.fc1 = nn.Linear(dim, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 52)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x