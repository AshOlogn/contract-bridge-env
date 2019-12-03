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
    """
    This environment doesn't exactly fit the OpenAI Gym API 
    """
    def __init__(self, players, bid_level, bid_trump, bid_team):
        """
        bid_level - number of tricks ( > 6) that bidder bets on taking
        bid_trump - the trump suit of this round's bid
        bid_maker - which side made the bid (either 0 or 1)
        """
        self.bid_level = bid_level
        self.bid_trump = bid_trump
        self.bid_team = bid_team
        
        self.trick_history = []
        self.current_trick = []
        self.trick_winner = None
        self.round_over = False
        self.team0_num_tricks = 0
        self.team1_num_tricks = 0
        self.team0_score = None 
        self.team1_score = None

        self.hands = {'p_00': [], 'p_01': [], 'p_10': [], 'p_11': []}
        self._deal()
    
    def _deal(self):
        index = 0
        players = ['p_00', 'p_01', 'p_10', 'p_11']
        self.hands = {'p_00': [], 'p_01': [], 'p_10': [], 'p_11': []}

        while not self.deck.is_empty():
            self.hands[players[index]].append(self.deck.deal())
            index = (index+1) % 4

    def _calculate_score(self, team):
        """
        Calculate the score achieved by the input team (0 or 1)
        Output: integer score
        """
        score = 0
        n_tricks = self.team0_num_tricks if team == 0 else self.team1_num_tricks

        if self.bid_team == team:
            if self.bid_trump is None:
                return 0 if n_tricks == 0 else 10 + n_tricks*30
            else:
                return n_tricks*(30 if self.bid_trump in ('H', 'S') else 20)
        else:
            return 50*max(0, self.bid_level-n_tricks)

    def reset(self):
        self.trick_history = []
        self.current_trick = []
        self._deal()

    def play(self, action):
        """
        Action must be an object with the following attributes:
            - player = 'p_00', 'p_01', 'p_10', or 'p_11'
            - card = some card object, must be in player's hand at that point
        
        Updates the appropriate player's hand as well as trick history
        All 4 agents call this method before "step" to get the appropriate reward
        """
        player = action['player']
        card = action['card']

        self.current_trick.append((player, card))

        #remove the played card from appropriate player's hand
        player_hand = self.hands[player]
        player_hand.pop(player_hand.index(card))

        #if the trick is over, calculate the winner and add it to the history
        if len(self.current_trick) == 4:
            current_trick_sorted = sorted(self.current_trick, key=lambda x: x[1], 
                                    reverse=True)
            
            self.trick_winner = int(current_trick_sorted[0][0][2])
            self.trick_history.append(self.current_trick)

            if self.trick_winner == 0:
                self.team0_num_tricks += 1
            else:
                self.team1_num_tricks += 1
        
        #if the round is over, calculate scores for each team based on the bid
        if len(self.trick_history) == 13:
            self.round_over = True
            self.team0_score = self._calculate_score(0)
            self.team1_score = self._calculate_score(1)

    def step(self, player):
        """
        player - one of 'p_00', 'p_01', 'p_10', or 'p_11'
        """
        assert len(self.current_trick) == 4, "Trick not done!"

        if self.round_over:
            return (None, self.team0_score if int(player[2])==0 else self.team1_score, True, None)
        else:
            return (None, 0, False, None)