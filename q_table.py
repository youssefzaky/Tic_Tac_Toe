import numpy as np

class Q_Table(object):

    def __init__(self, env, discount=1, epsilon=0.2,
                 learning_rate=1):
        self.q_dict = {} # keys are states, values are action-value dicts
        self.discount = discount # discount factor
        self.learning_rate = learning_rate 
        self.env = env
        self.epsilon = epsilon

    def learn(self, max_games):
        state = self.env.reset()
        games = 1
        while games < max_games:
            stateKey = tuple(state)
            if stateKey not in self.q_dict:
                self.initialize(stateKey)
            action = self.act(stateKey)
            nextState, reward, done = self.env.step(action)
            nextKey = tuple(nextState)
            if nextKey not in self.q_dict:
                self.initialize(nextKey)
            self.updateQ(stateKey, action, nextKey, reward, done)
            if done:
                state = self.env.reset()
                games += 1
            else:
                state = nextState
        # when training is done, stop exploration
        self.epsilon = 0
     
    def updateQ(self, stateKey, action, nextKey, reward, done):
        if done:
            expected = reward
        else:
            """ expect opponent to play best move, X tries to 
            maximize reward, O tries to minimize it"""
            if stateKey[-1]: # player X
                lowestQ = min(self.q_dict[nextKey].values())
                expected = reward + (self.discount * lowestQ)
            else: # player O
                highestQ = max(self.q_dict[nextKey].values())
                expected = reward + (self.discount * highestQ)
        change = self.learning_rate * (expected - self.q_dict[stateKey][action])
        self.q_dict[stateKey][action] += change

    def act(self, stateKey):
        actionDict = self.q_dict[stateKey]
        actions = []
        values = []
        for a, v in actionDict.items():
            actions.append(a)
            values.append(v)
        if not stateKey[-1]: # if O player, try to minimize expected reward
            values = [-1 * value for value in values]

        values = np.array(values)
        action = np.argmax(values)
        n = len(values)
        probs = np.zeros(values.shape)
        probs[...] = self.epsilon / n
        probs[action] = 1 - self.epsilon + (self.epsilon / n)
        return np.random.choice(actions, p=probs)

    def initialize(self, stateKey):
        """Set action dict of a new state with small values for legal actions."""
        self.q_dict[stateKey] = {}
        for action in range(9):
            if not self.env.is_illegal(action):
                self.q_dict[stateKey][action] = np.random.uniform(-0.2, 0.2)


if __name__ == "__main__":
    from game import Tic_Tac_Toe, Human
    env = Tic_Tac_Toe()
    agent = Q_Table(env)
    agent.learn(max_games=30000)

    # test learnt policy
    while True:
        env.episode(agent, Human(), display=True)
    
