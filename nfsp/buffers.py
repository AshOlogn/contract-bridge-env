"""
This code is COPIED from https://github.com/belepi93/pytorch-nfsp and is NOT
our original work. It is merely a small piece of the NFSP algorithm that we 
decided to copy in the interest of time.
"""

import numpy as np
import random
import math

from collections import deque, namedtuple
import itertools

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

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

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

class ReservoirBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action):
        state = np.expand_dims(state, 0)
        self.buffer.append((state, action))
    
    def sample(self, batch_size):
        # Efficient Reservoir Sampling
        # http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
        n = len(self.buffer)
        reservoir = list(itertools.islice(self.buffer, 0, batch_size))
        threshold = batch_size * 4
        idx = batch_size
        while (idx < n and idx <= threshold):
            m = random.randint(0, idx)
            if m < batch_size:
                reservoir[m] = self.buffer[idx]
            idx += 1
        
        while (idx < n):
            p = float(batch_size) / idx
            u = random.random()
            g = math.floor(math.log(u) / math.log(1 - p))
            idx = idx + g
            if idx < n:
                k = random.randint(0, batch_size - 1)
                reservoir[k] = self.buffer[idx]
            idx += 1
        state, action = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action
    
    def __len__(self):
        return len(self.buffer)