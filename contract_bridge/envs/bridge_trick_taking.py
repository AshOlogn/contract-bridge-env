import random
from functools import total_ordering
from itertools import product

import gym
from gym import spaces, utils
from gym.error import InvalidAction

@total_ordering
class Card:
    #12-14 is J, Q, K, A
    ranks = (2,3,4,5,6,7,8,9,10,11,12,13,14)
    suits = ('C','D','H','S')
    
    def __init__(self, rank, suit):
        if rank in Card.ranks:
            self.rank = rank
        else:
            raise Exception('The rank must be an integer between 2 and 14 inclusive.')

        if suit in Card.suits:
            self.suit = suit
        else:
            raise Exception('The suit must be \'C\', \'D\', \'H\', or \'S\'.')
    
    def __str__(self):
        if (self.rank == 14):
            rank = 'A'
        elif (self.rank == 13):
            rank = 'K'
        elif (self.rank == 12):
            rank = 'Q'
        elif (self.rank == 11):
            rank = 'J'
        else:
            rank = str (self.rank)
        return rank + self.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))
    
    def __eq__ (self, other):
        return (self.rank == other.rank) and (self.suit == other.suit)
    
    def __ne__ (self, other):
        return not self.__eq__(other)
    
    def __lt__ (self, other):
        if self.suit != other.suit:
            return self.suit < other.suit
        else:
            return self.rank < other.rank

class Deck:
    def __init__(self):
        self.deck = list(map(lambda x: Card(rank=x[0], suit=x[1]), 
                        product(Card.ranks, Card.suits)))
        self.shuffle()
    
    def shuffle(self):
        random.shuffle(self.deck)
    
    def is_empty(self):
        return len(self.deck) == 0
    
    def deal(self):
        if self.is_empty():
            return None
        else:
            return self.deck.pop(0)
    
    def __str__(self):
        return ', '.join(map(lambda x: str(x), self.deck))

class BridgeEnv(gym.Env):

    def __init__(self, starting_player):
        self.trick_history = []
        self.current_trick = []
        self.hands = {'op1': [], 'op2': [], 't1': [], 't2': []}
        self._deal()
    
    def _deal(self):
        index = 0
        players = ['op1', 'op2', 't1', 't2']
        self.hands = {'op1': [], 'op2': [], 't1': [], 't2': []}

        while not self.deck.is_empty():
            self.hands[players[index]].append(self.deck.deal())
            index = (index+1) % 4
    
    def reset(self):
        self.trick_history = []
        self.current_trick = []
        self._deal()

    def step(self, action):
        """
        Action must be an object with the following attributes:
            - player = 'op1', 'op2', 't1', or 't2'
            - card = some card object, must be in player's hand at that point
        
        Returns a 4-tuple of (observation, reward, termination, info)
        Info field is empty for now
        """
        player = action['player']
        card = action['card']

        self.current_trick.append((player, card))

        #remove the played card from appropriate player's hand
        player_hand = self.hands[player]
        player_hand.pop(player_hand.index(card))

        if len(self.current_trick) == 4:
            #trick done, add it to history
            self.trick_history.append(self.current_trick)

            #determine the winner
            current_trick_sorted = sorted(self.current_trick, key=lambda x: x[1], 
                                        reverse=True)
            trick_winner = current_trick_sorted[0][0]

            #now determine if the round is over
            for p in self.hands:
                if len(self.hands[p]) > 0:
                    return (trick_winner, -1, False, {})
            
            #at this point, the entire round is over
            return (None, -1, None, {})
            
        else:
            return (None, 0, False, {})
