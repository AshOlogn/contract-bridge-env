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
        x = F.softmax(self.fc3(x))
        return x

class ReplayMemory(object):

    def __init__(self, cap):
        self.cap = cap
        self.mem = []
        self.pos = 0

    def push(self, *args):
        if len(self.mem) < self.cap:
            self.mem.append(None)
        self.mem[self.pos] = Transition(*args)
        self.pos = (self.pos + 1) % self.cap

    def sample(self, num):
        return random.sample(self.mem, num)

    def __len__(self):
        return len(self.mem)
