import random

class Card(object):
    #12-14 is J,Q,K,A
    ranks = (2,3,4,5,6,7,8,9,10,11,12,13,14)
    suits = ('C','D','H','S')

    def __init__(self, rank = 12, suit = 'S'):
        if rank in Card.ranks:
            self.rank = rank
        else:
            self.rank = 12

        if suit in Card.suits:
            self.suit = suit
        else:
            self.suit = 'C'

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
        return self.rank == other.rank

    def __ne__ (self, other):
        return self.rank != other.rank

    def __lt__ (self, other):
        return self.rank < other.rank or (self.rank == other.rank and self.suits.index(self.suit) < other.suits.index(other.suit))

    def __le__ (self, other):
        return self.rank <= other.rank

    def __gt__ (self, other):
        return self.rank > other.rank or (self.rank == other.rank and self.suits.index(self.suit) > other.suits.index(other.suit))

    def __ge__ (self, other):
        return self.rank >= other.rank


class Deck(object):
    #create card deck
    def __init__(self):
        self.deck = []
        for rank in Card.ranks:
            for suit in Card.suits:
                card = Card(rank, suit)
                self.deck.append(card)

    def shuffle(self):
        random.shuffle(self.deck)

    def deal(self):
        if len(self.deck) == 0:
            return None
        else:
            return self.deck.pop(0)

    def __str__(self):
        str1 = ""
        for i in range(0,len(self.deck) - 1):
            str1 += str(self.deck[i]) + ', '
        str1 += str(self.deck[len(self.deck) - 1])
        return str1


class Bridge(object):

    def __init__(self):
        self.deck = Deck()
        self.deck.shuffle()
        self.all_cards = []

        for i in range(0,4):
            player_cards = []
            for j in range(0,13):
                player_cards.append(self.deck.deal())
        
        self.card_ranks = ()
        self.team1, self.team2 = (1,3), (2,4)

        self.all_tricks = []


    def deal(self):
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
            


def main():
    deck = Deck()
    print(deck)

main()

