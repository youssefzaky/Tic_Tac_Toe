class Tic_Tac_Toe(object):
    """Contains all data representing the game."""
    
    def __init__(self, init_state=None, player=True):
        # empty = 0, 'X' = 1, 'O' = -1
        # board is represented in row-major order in a list
        if init_state:
            self.state = init_state
        else:
            self.state = [0] * 9
        self.player = player # player 1 or player 2 turn
        self.winner = None

    def print_board(self):
        """Convert from list to 'XO' representation and print board."""
        
        num_to_string = lambda num: [' ', 'X', 'O'][num]
        xo_rep = list(map(num_to_string, self.state))
        print("{0} | {1} | {2}".format(*xo_rep[:3]))
        print("{0} | {1} | {2}".format(*xo_rep[3:6]))
        print("{0} | {1} | {2}\n\n".format(*xo_rep[6:]))
        
    def is_legal(self, action):
        """Is this action legal?"""
        return self.state[action] == 0
        
    def step(self, action, display=False):
        if self.player:
            self.state[action] = 1 
        else:
            self.state[action] = -1
        self.player = not self.player # switch players
        
        if display:
            self.print_board()
            
    def is_done(self):
        """Figure out if someone won or game is tied"""
        
        if not self.player: # previous player
            self.check = 1 * 3
        else:
            self.check = -1 * 3
        # sum rows, columns, and diagonals
        r1 = sum(self.state[:3])
        r2 = sum(self.state[3:6])
        r3 = sum(self.state[6:])
        c1 = sum(self.state[0:7:3])
        c2 = sum(self.state[1:8:3])
        c3 = sum(self.state[2:9:3])
        d1 = sum(self.state[0:9:4])
        d2 = sum(self.state[2:7:2])
        if r1 == self.check or r2 == self.check \
            or r3 == self.check or c1 == self.check \
            or c2 == self.check or c3 == self.check \
            or d1 == self.check or d2 == self.check:
            self.winner = not self.player
            return True
        else:
            return False

        # if game is tied
        if len(list(filter(self.state, lambda pos: pos > 0))) == 9:
            return True
        
    def episode(self, agent1, agent2, display=False):
        p1_states = [self.state]
        p2_states, p1_actions, p2_actions = [], [], []
        p1_rewards, p2_rewards = [], []
        
        while True:
            action = agent1.act()
            if not self.is_legal(action):
                p1_rewards.append(-10)
            else:
                p1_rewards.append(0)
            self.step(action, display=display)
            p2_states.append[self.state]
            p1_actions.append[action]
            if self.is_done(): break
                
            action = agent2.act()
            if not self.is_legal(action):
                p2_rewards.append(-10)
            else:
                p2_rewards.append(0)
            self.step(action, display=display)
            p1_states.append[self.state]
            p2_actions.append[action]
            if self.is_done(): break
        
        if self.winner is None: # game tied
            p1_rewards.append(0)
            p2_rewards.append(0)
        else: # someone one
            p1_rewards.append(int(self.winner))
            p2_rewards.append(int(not self.winner))
        
        return {"p1_states":p1_states, "p2_states":p2_states,
                "p1_actions:":p1_actions, "p2_actions":p2_actions,
                "p1_rewards":p1_rewards, "p2_rewards":p2_rewards}


# a few unit tests

def test1():
    print("GAME")
    board = Tic_Tac_Toe([0, -1, -1, 1, 0, 1, -1, 1, -1], False)
    board.print_board()
    board.step(4, display=True)
    board.step(0, display=True)
    assert board.is_done() == False
    assert board.winner == None
    
def test2():
    print("GAME")
    board = Tic_Tac_Toe([0, 1, -1, 1, 0, 1, -1, 1, -1], True)
    board.print_board()
    board.step(4, display=True)
    assert board.is_done() == True
    assert board.winner == True
    
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
    board.step(5, display=True)
    assert board.is_done() == False
    assert board.winner == None


if __name__== "__main__":
    test1()
    test2()
    test3()
