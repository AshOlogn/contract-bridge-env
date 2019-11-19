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
    
    def __eq__ (self, other):
        return (self.rank == other.rank) and (self.suit == other.suit)
    
    def __ne__ (self, other):
        return (self.rank != other.rank) or (self.suit != other.suit)
    
    def __lt__ (self, other):
        if self.suit != other.suit:
            return self.suit < other.suit
        else:
            return self.rank < other.rank

class Deck:
    def __init__(self):
        self.deck = list(map(lambda x: Card(rank=x[0], suit=x[1]), product(Card.ranks, Card.suits)))
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

    def __init__(self, opponent, start=0):
        self.deck = Deck()
        self.player_hands = [[] for i in range(4)]
        self.you, self.opponent = (0,2), (1,3)
        self._deal()
    
    def _deal(self):
        self.player_hands = [[] for i in range(4)]
        
        index = 0
        while not self.deck.is_empty():
            self.player_hands[index].append(self.deck.deal())
            index = (index+1) % 4

    def reset(self):
        self.deck = Deck()
        self._deal()

    def step(self, action):
        pass

    def round(self):
        self.pile = []
        self.game = True
        while game:
            for i in range(0,len(self.all_cards)):
                self.random_int = random.randint(0,len(self.all_cards[i]) + 1)
                self.random_card = self.all_cards[i][self.random_int]
                if self.random_card in self.all_cards[i]:
                    self.pile.append(self.random_card)
                    self.all_cards[i].pop(self.random_int)
            
            base_card = self.pile[0]
            index = self.pile.index(base_card) 
            for j in range(1,len(self.pile)):
                curr_card = self.pile[j]
                #need to write the equalities for this 
                if curr_card < base_card:
                    index = self.pile.index(curr_card) 
            self.all_tricks.append(index)

            self.done = True
            for k in range(0,len(self.all_cards)):
                if len(self.all_cards[i] != 0):
                    self.done = False
            
            if not self.done:
                self.game = False
