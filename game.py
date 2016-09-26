import numpy as np

class Tic_Tac_Toe(object):
    """Contains all data representing the game."""
    
    def __init__(self, init_board=None, player=1):
        # empty = 0, 'X' = 1, 'O' = -1
        # board is represented in row-major order in a list
        self.state = [0] * 10
        if init_board:
            self.state[:9] = init_board
            self.state[-1] = player
        else:
            self.reset()

    def print_board(self):
        """Convert from list to 'XO' representation and print board."""

        if self.state[-1]:
            print("X turn")
        else:
            print("O turn")
        num_to_string = lambda num: [' ', 'X', 'O'][num]
        xo_rep = list(map(num_to_string, self.state[:9]))
        print("{0} | {1} | {2}".format(*xo_rep[:3]))
        print("{0} | {1} | {2}".format(*xo_rep[3:6]))
        print("{0} | {1} | {2}\n\n".format(*xo_rep[6:]))
        
    def is_illegal(self, action):
        """Is this action legal?"""
        return self.state[action] != 0
        
    def step(self, action, display=False):
        reward, done = 0, False
        
        if self.state[-1]:
            self.state[action] = 1
            if self.is_winner():
                reward = 1
                done = True
        else:
            self.state[action] = -1
            if self.is_winner():
                reward = -1
                done = True

        if self.is_tied():
            done = True
            
        # switch players
        self.state[-1] = (self.state[-1] + 1) % 2 
        
        if display:
            self.print_board()

        return self.state, reward, done

    def reset(self):
        self.state[-1] = int(np.random.randn() > 0.5)
        # clear board
        self.state[:9] = [0] * 9
        return self.state

    def is_winner(self):
        """Figure out if someone won or game is tied"""

        player = self.state[-1] 
        if player:
            self.check = 1 * 3
        else:
            self.check = -1 * 3
            
        # sum rows, columns, and diagonals
        r1 = sum(self.state[:3])
        r2 = sum(self.state[3:6])
        r3 = sum(self.state[6:9])
        c1 = sum(self.state[0:7:3])
        c2 = sum(self.state[1:8:3])
        c3 = sum(self.state[2:9:3])
        d1 = sum(self.state[0:9:4])
        d2 = sum(self.state[2:7:2])
        return r1 == self.check or r2 == self.check \
            or r3 == self.check or c1 == self.check \
            or c2 == self.check or c3 == self.check \
            or d1 == self.check or d2 == self.check

    def is_tied(self):
        return len(list(filter(lambda pos: pos == 0, self.state[:9]))) == 0
        
    
    def episode(self, agent1, agent2, display=False):
        self.reset()
        while True:
            action = agent1.act(tuple(self.state))
            _, _, done = self.step(action, display=display)
            if done: break
            action = agent2.act(tuple(self.state))
            _, _, done = self.step(action, display=display)
            if done: break


class Human(object):
    def act(self, ob):
        return int(input())

    
class Random(object):

    def __init__(self, board):
        self.board = board
        
    def act(self, ob):
        action = np.random.choice(range(9))
        while self.board.is_illegal(action):
            action = np.random.choice(range(9))
        return action

    
# a few unit tests for gameplay

def test1():
    print("GAME")
    board = Tic_Tac_Toe([0, -1, 1, 1, 0, 1, -1, 1, -1], 0)
    board.print_board()
    _, _, done = board.step(4, display=True)
    assert done == False
    
def test2():
    print("GAME")
    board = Tic_Tac_Toe([0, 1, -1, 1, 0, 1, -1, 1, -1], 1)
    board.print_board()
    _, _, done = board.step(4, display=True)
    assert done == True
    
def test3():
    print("GAME")
    board = Tic_Tac_Toe(player=True)
    board.print_board()
    board.step(0, display=True)
    board.step(4, display=True)
    board.step(1, display=True)
    board.step(2, display=True)
    board.step(6, display=True)
    board.step(3, display=True)
    _, _, done = board.step(5, display=True)
    assert done == False
    assert board.is_tied() == False
    board.step(7, display=True)
    _, _, done = board.step(8, display=True)
    assert board.is_tied() == True
    assert done == True


if __name__== "__main__":
    test1()
    test2()
    test3()
