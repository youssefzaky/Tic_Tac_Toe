from collections import defaultdict
import numpy as np
import tensorflow as tf

def apply_discount(x, gamma):
    out = np.zeros(len(x), dtype=np.float64)
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    return out


class Q_Approx(object):

    def __init__(self, env, discount=1, batch_size=1000,
                 learning_rate=0.001, epsilon=0.1, n_hidden1=100,
                 n_hidden2=50):
        self.discount = discount 
        self.learning_rate = learning_rate 
        self.epsilon = epsilon
        self.env = env
        self.batch_size = batch_size
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
            data = tf.placeholder(shape=(self.batch_size,
                                         input_dim),
                                  dtype=tf.float32)
            targets = tf.placeholder(shape=(self.batch_size),
                                     dtype=tf.float32)
        
            # weights and biases of a network with two hidden layers
            W0 = tf.Variable(tf.truncated_normal([input_dim, self.n_hidden1]))
            b0 = tf.Variable(tf.zeros(self.n_hidden1))
            W1 = tf.Variable(tf.truncated_normal([self.n_hidden1, self.n_hidden2]))
            b1 = tf.Variable(tf.zeros(self.n_hidden2))
            W2 = tf.Variable(tf.truncated_normal([self.n_hidden2, 1]))
            b2 = tf.Variable(tf.zeros(1))
        
            # state-action values for training
            hidden1 = tf.nn.relu(tf.matmul(data, W0) + b0)
            hidden2 = tf.nn.relu(tf.matmul(hidden1, W1) + b1)
            values = tf.matmul(hidden2, W2) + b2
        
            # define the loss and optimizer
            loss = tf.reduce_mean(tf.square(tf.sub(targets, values)))    
            alg = tf.train.RMSPropOptimizer(self.learning_rate, momentum=0.2)
            optimizer = alg.minimize(loss)
                          
            # state-action value for control
            ob = tf.placeholder(shape=(1, input_dim), dtype=tf.float32) 
            hidden1 = tf.nn.relu(tf.matmul(ob, W0) + b0)
            hidden2 = tf.nn.relu(tf.matmul(hidden1, W1) + b1)
            value = tf.matmul(hidden2, W2) + b2
            
            init = tf.initialize_all_variables()

        ops = {"data":data, "targets":targets, "value":value,
               "loss":loss, "optimizer":optimizer, "init":init, "ob":ob}

        return graph, ops
    
    def episode(self):
        states = []
        rewards = []
        actions = []
        state = self.env.reset()
        done = False
        while not done:
            states.append(state)
            action = self.act(state)
            actions.append(action)
            nextState, reward, done = self.env.step(action)
            state = nextState
            rewards.append(reward)

        return {"rewards" : np.array(rewards),
                "obs" : np.array(states),
                "actions" : np.array(actions)}

    def make_batch(self):
        size = 0
        trajs = []
        # collect enough trajectories
        while size < self.batch_size:
            traj = self.episode()
            trajs.append(traj)
            size += len(traj["rewards"])
            
        obs = np.concatenate([traj["obs"] for traj in trajs])
        obs = obs[:self.batch_size]
        
        returns = np.concatenate([apply_discount(traj["rewards"], self.discount)
                   for traj in trajs])
        targets = returns[:self.batch_size]

        # turn actions values into a 1-hot code
        actions = [(np.arange(self.action_dim) == traj["actions"][:,None]).astype(np.float32) \
                  for traj in trajs]
        actions = np.concatenate(actions)[:self.batch_size]
        
        data = np.concatenate([obs, actions], axis=1)
        # permute data to help SGD
        permutation = np.random.permutation(self.batch_size)
        data[...] = data[permutation]
        targets[...] = targets[permutation]
        
        return data, targets

    def learn(self, n_iters):
        self.session.run(self.ops["init"])
        for iteration in range(n_iters):
            data, targets = self.make_batch()
            _, loss = self.session.run([self.ops["optimizer"],
                                        self.ops["loss"]],
                                       feed_dict={self.ops["data"]:data,
                                                  self.ops["targets"]:targets})

            print("Iteration:", iteration, ", Loss:", loss)
        # when training is done, stop exploration
        self.epsilon = 0
            
    def act(self, state):
        """Sample actions using e-greedy strategy"""
        actions = []
        values = []

        value_op, ob_op = self.ops["value"], self.ops["ob"]

        # get values of legal actions from approximator
        for action in range(9):
            if not self.env.is_illegal(action):
                actions.append(action)
                action_vector = np.zeros(self.action_dim)
                action_vector[action] = 1
                ob = [np.concatenate([state, action_vector])]
                value = np.asscalar(self.session.run(value_op, feed_dict={ob_op:ob}))
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
    agent = Q_Approx(env, batch_size=200, learning_rate=0.0001,
                     epsilon=0.1, n_hidden1=50, n_hidden2=20)
    agent.learn(1000)
    while True:
        env.episode(agent, Human(), display=True)
