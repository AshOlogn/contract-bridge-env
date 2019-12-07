import argparse
import sys
import math
import random
import numpy as np
from nfsp.nn import DQN, PG
from contract_bridge.envs.bridge_trick_taking import BridgeEnv

class Agent(object):
    """
    Base class for agents interacting with the Bridge environment
    """
    def __init__(self, pid, env):
        self.pid = pid
        self.env = env
        self.dqn = DQN()
        self.pg = PG()
    
    def act(self):
        raise NotImplementedError()

class SmartGreedyAgent(Agent):
    def __init__(self, pid, env, epsilon=0.9):
        super().__init__(pid, env)
        self.epsilon = epsilon
    
    def act(self):
        #see player's hand and randomly pick a card
        prob = np.random.rand()
        
        hand = self.env.hands[self.pid]
        highest_played = sorted(list(map(lambda x: x[1], self.env.current_trick)))[-1] if len(self.env.current_trick) != 0 else None

        #ids of other players
        teammate = self.env.get_teammate(self.pid)
        left_op = self.env.get_left_opponent(self.pid)
        right_op = self.env.get_right_opponent(self.pid)
        
        #cards that others have played
        played_this_trick = self.env.played_this_trick
        teammate_card = played_this_trick[teammate]
        left_op_card = played_this_trick[left_op]
        right_op_card = played_this_trick[right_op]

        if prob > self.epsilon:
            if teammate_card is not None:
                if left_op_card is not None and right_op_card is not None:
                    if teammate_card > max(left_op_card, right_op_card):
                        #if teammate already won the trick, just burn the worst card
                        return min(hand)
            
            return self._play_worst_winning(hand, left_op_card, right_op_card)
        
        return random.choice(hand)

    def _get_highest_op_card(self, left_op_card, right_op_card):
        if left_op_card is not None and right_op_card is not None:
            return max(left_op_card, right_op_card)
        elif left_op_card is not None:
            return left_op_card
        else:
            return right_op_card
    
    def _play_worst_winning(self, hand, left_op_card, right_op_card):
        #play the smallest winning card, if none exists burn the worst card
        highest_op_card = self._get_highest_op_card(left_op_card, right_op_card)
        hand_sorted = sorted(hand)

        play_card = None
        for card in hand_sorted:
            print(card)
            print(highest_op_card)
            if card > highest_op_card:
                play_card = card
                break
        
        return play_card if play_card is not None else hand_sorted[0]

def RandomAgent(Agent):
    def __init__(self, pid, env):
        super()._init__(pid, env)
    
    def act(self):
        return random.choice(self.env.hands[self.pid])
  
def NFSPAgent(Agent):
    def __init(self, pid, env, dqn=DQN(), pg=PG()):
        super().__init__(pid, env)
        self.dqn = dqn
        self.pg = pg
    
    def act(self):
        pass
    
