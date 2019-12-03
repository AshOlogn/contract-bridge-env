import torch 
import torch.nn as nn
from collections import namedtuple
import random

class DeepQNetwork(nn.Module):
    #need to modify architecture and add network input
    #need to initialize the parameters for the network
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, 256)
        self.l2 = nn.Linear(256, 2)

    def forward(self):
        #need to map from state to action value
        x = F.relu(self.l1(x))
        x = self.l2(x)
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