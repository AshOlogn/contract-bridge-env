from itertools import product
import numpy as np
from torch import Tensor

import gym
from gym import spaces, utils
from gym.error import InvalidAction
from .deck import Card, Deck

class BridgeEnv(gym.Env):
    """
    This environment doesn't exactly fit the OpenAI Gym API
    """
    def __init__(self):
        pass
    
    def initialize(self, bid_level, bid_trump, bid_team):
        """
        bid_level - number of tricks ( > 6) that bidder bets on taking
        bid_trump - the trump suit of this round's bid
        bid_maker - which side made the bid (either 0 or 1)
        """
        self.bid_level = bid_level
        self.bid_trump = bid_trump
        self.bid_team = bid_team

        #calculate the index of the current bid
        self.bid_index = ['C','D','H','S', None].index(bid_trump)*7 + (bid_level-7)
        
        #create a dictionary mapping cards to index
        suits_resorted = ['C','D','H','S']
        if bid_trump is not None:
            suits_resorted.append(suits_resorted.pop(suits_resorted.index(bid_trump)))

        self.card_to_index = {}
        index = 0
        for (rank,suit) in product(Card.ranks, Card.suits):
            self.card_to_index[Card(rank,suit,bid_trump)] = index
            index += 1
        
        self._init_variables()
    
    def reset(self, bid_level, bid_trump, bid_team):
        self.initialize(bid_level, bid_trump, bid_team)
    
    def _init_variables(self):
        self.trick_history = []
        self.current_trick = []
        self.trick_winner = None
        self.round_over = False

        #keep track of each team's score
        self.team0_num_tricks = 0
        self.team1_num_tricks = 0
        self.team0_score = None
        self.team1_score = None

        #some variables to keep track of the state representation given to neural networks
        self.played_cards = {'p_00': [], 'p_01': [], 'p_10': [], 'p_11': []}
        self.played_cards_vector = ({'p_00': np.zeros((52,)), 'p_01': np.zeros((52,)),
            'p_10': np.zeros((52,)), 'p_11': np.zeros((52,))})

        self.played_this_trick = {'p_00': None, 'p_01': None, 'p_10': None, 'p_11': None}
        self.hands = {'p_00': [], 'p_01': [], 'p_10': [], 'p_11': []}
        self.hands_vector = ({'p_00': np.zeros((52,)), 'p_01': np.zeros((52,)),
            'p_10': np.zeros((52,)), 'p_11': np.zeros((52,))})

        self._deal()

    def _deal(self):
        index = 0
        players = ['p_00', 'p_01', 'p_10', 'p_11']
        self.hands = {'p_00': [], 'p_01': [], 'p_10': [], 'p_11': []}

        self.deck = Deck(trump=self.bid_trump)
        while not self.deck.is_empty():
            card = self.deck.deal()
            self.hands[players[index]].append(card)
            self.hands_vector[players[index]][self.card_to_index[card]] = 1
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

    def get_state(self, player):
        """
        Return the tensor representation of the state visible to each player
        This computation is broken up into stages and then concatenated at the end
        """
        teammate = self.get_teammate(player)
        left = self.get_left_opponent(player)
        right = self.get_right_opponent(player)

        #get current hand
        current_hand_vector = self.hands_vector[player]

        #teammate, left, right opponent play history
        teammate_history_vector = self.played_cards_vector[teammate]
        left_history_vector = self.played_cards_vector[left]
        right_history_vector = self.played_cards_vector[right]

        #teammate, left, right opponent plays this trick
        teammate_current_trick = np.zeros((52,))
        left_current_trick = np.zeros((52,))
        right_current_trick = np.zeros((52,))

        if self.played_this_trick[teammate] is not None:
            card = self.played_this_trick[teammate]
            teammate_current_trick[self.card_to_index[card]] = 1

        if self.played_this_trick[left] is not None:
            card = self.played_this_trick[left]
            left_current_trick[self.card_to_index[card]] = 1

        if self.played_this_trick[right] is not None:
            card = self.played_this_trick[right]
            right_current_trick[self.card_to_index[card]] = 1

        #the bid that was made for this round
        bid = np.zeros((35,))
        bid[self.bid_index] = 1

        #whether this team or the opponent made the bid
        #index 0 is this team, 1 is the opponent
        team = int(player[2])
        bid_team = np.array([1,0] if self.bid_team==team else [0,1])

        #concatenate into 1 numpy array and convert into a PyTorch tensor
        concat_tuple = ((current_hand_vector, teammate_history_vector, left_history_vector, 
            right_history_vector, teammate_current_trick, left_current_trick,
            right_current_trick, bid, bid_team))
        
        return Tensor(np.concatenate(concat_tuple))

    def get_teammate(self, player):
        players = ['p_00', 'p_11', 'p_01', 'p_10']
        index = players.index(player)
        return players[(index+2) % len(players)]

    def get_left_opponent(self, player):
        players = ['p_00', 'p_11', 'p_01', 'p_10']
        index = players.index(player)
        return players[(index-1) % len(players)]

    def get_right_opponent(self, player):
        players = ['p_00', 'p_11', 'p_01', 'p_10']
        index = players.index(player)
        return players[(index+1) % len(players)]

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

        #remove the played card from appropriate player's hand and set it
        #as their played card this trick
        player_hand = self.hands[player]
        print(player_hand)
        print(self.played_this_trick)
        self.played_this_trick[player] = player_hand.pop(player_hand.index(card))
        self.hands_vector[player][self.card_to_index[card]] = 0

        #the trick is over
        if len(self.current_trick) == 4:
            current_trick_sorted = sorted(self.current_trick, key=lambda x: x[1],
                                    reverse=True)

            #calculate the winner and add it to the history
            self.trick_winner = int(current_trick_sorted[0][0][2])
            self.trick_history.append(self.current_trick)

            if self.trick_winner == 0:
                self.team0_num_tricks += 1
            else:
                self.team1_num_tricks += 1

            #now add each player's card to their respective histories
            for p in self.played_this_trick:
                self.played_cards[p].append(self.played_this_trick[p])
                self.played_cards_vector[p][self.card_to_index[self.played_this_trick[p]]] = 1
                self.played_this_trick[p] = None


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
            return (None, 1 if int(player[2])==self.trick_winner else 0, False, None)