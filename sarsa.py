from collections import defaultdict
import numpy as np
import tensorflow as tf

def apply_discount(x, gamma):
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    return out


def MLP_4(input_op, input_dim, output_dim, n_hidden1, n_hidden2):
    # weights and biases of a network with one hidden layer
    W0 = tf.Variable(tf.truncated_normal([input_dim, n_hidden1]))
    b0 = tf.Variable(tf.zeros(n_hidden1))
    W1 = tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2]))
    b1 = tf.Variable(tf.zeros(n_hidden2))
    W2 = tf.Variable(tf.truncated_normal([n_hidden2, 1]))
    b2 = tf.Variable(tf.zeros(1))
        
    # state-action values for training
    hidden1 = tf.nn.relu(tf.matmul(input_op, W0) + b0)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W1) + b1)
    value = tf.matmul(hidden2, W2) + b2
    return value

def linear(input_op, input_dim, output_dim):
    W0 = tf.Variable(tf.truncated_normal([input_dim, 1]))
    b0 = tf.Variable(tf.zeros(1))
    value = tf.matmul(input_op, W0 + b0)
    return value
                      

class Sarsa(object):

    def __init__(self, env, discount=1, learning_rate=0.001,
                 epsilon=0.1, n_hidden1=100, n_hidden2=50):
        self.discount = discount 
        self.learning_rate = learning_rate 
        self.epsilon = epsilon
        self.env = env
        self.action_dim = self.env.action_dim
        self.ob_dim = self.env.ob_dim
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.graph, self.ops = self.init_graph()
        self.session = tf.Session(graph=self.graph)

    def init_graph(self):
        
        graph = tf.Graph()
        with graph.as_default():

            input_dim = self.ob_dim + self.action_dim
            ob = tf.placeholder(shape=(1, input_dim), dtype=tf.float32)
            target = tf.placeholder(shape=(1), dtype=tf.float32)

            value = MLP_4(ob, input_dim, 1, self.n_hidden1, self.n_hidden2)
            #value = linear(ob, input_dim, 1)  
        
            # define the loss and optimizer
            loss = tf.square(tf.sub(target, value))
            alg = tf.train.RMSPropOptimizer(self.learning_rate)
            optimizer = alg.minimize(loss)
                                      
            init = tf.initialize_all_variables()

        ops = {"target":target, "value":value, "loss":loss,
               "optimizer":optimizer, "init":init, "ob":ob}

        return graph, ops

    def learn(self, n_episodes):
        """Sarsa algorithm"""
        
        self.session.run(self.ops["init"])
        for episode in range(n_episodes):
            S = self.env.reset()
            A = self.act(S)
            while True:
                S_p, reward, done = self.env.step(A)
                if done:
                    loss = self.update(self.make_ob(S, A), target=reward)
                    break
                A_p = self.act(S_p)
                pred = self.value(self.make_ob(S_p, A_p))
                target = reward + self.discount * pred
                loss = self.update(self.make_ob(S, A), target)
                S = S_p
                A = A_p
            
            print("Episode:", episode, ", Loss:", loss)
            
        # when training is done, stop exploration
        self.epsilon = 0

    def update(self, ob, target):
        _, loss = self.session.run([self.ops["optimizer"],
                                    self.ops["loss"]],
                                   feed_dict={self.ops["ob"]:ob,
                                              self.ops["target"]:[target]})
        return np.asscalar(loss)

    def make_ob(self, state, action):
        """Helper function to make 1-hot code for actions"""
        action_vector = np.zeros(self.action_dim)
        action_vector[action] = 1
        ob = [np.concatenate([state, action_vector])]
        return ob
                
    def value(self, ob):
        """Get value from TF network"""
        return np.asscalar(self.session.run(self.ops["value"],
                                            feed_dict={self.ops["ob"]:ob}))
            
    def act(self, state):
        """Sample actions using e-greedy strategy"""
        actions = []
        values = []

        value_op, ob_op = self.ops["value"], self.ops["ob"]

        # get values of legal actions from approximator
        for action in range(9):
            if not self.env.is_illegal(action):
                actions.append(action)
                ob = self.make_ob(state, action)
                value = self.value(ob)
                values.append(value)
        
        if not state[-1]: # if O player, try to minimize expected reward
            values = [-1 * value for value in values]

        values = np.array(values)
        action = np.argmax(values)
        n = len(values)
        probs = np.zeros(values.shape)
        probs[...] = self.epsilon / n
        probs[action] = 1 - self.epsilon + (self.epsilon / n)
        return np.random.choice(actions, p=probs)


if __name__ == "__main__":
    from game import Tic_Tac_Toe, Human
    env = Tic_Tac_Toe()
    agent = Sarsa(env, learning_rate=0.0001, epsilon=0.3,
                  n_hidden1=100, n_hidden2=20)
    agent.learn(2000)
    while True:
        env.episode(agent, Human(), display=True)
    
