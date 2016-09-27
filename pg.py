import tensorflow as tf
import numpy as np

def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)

class TTT_Agent(object):

    """REINFORCE agent for Tic-Tac-Toe."""

    def __init__(self, board, timesteps_per_batch=1000, ob_dim=10, 
                 action_dim=9, n_hidden=5, learning_rate=0.000, 
                 n_iters=100):
        """Create the tensorflow operations for training."""

        self.board = board
        self.n_iters = n_iters 
        self.timesteps_per_batch = timesteps_per_batch
        
        # operations for optimization
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            # first the input data holders
            self.all_obs = tf.placeholder(shape=(timesteps_per_batch, ob_dim),
                                          dtype=tf.float32) # Observations
            self.all_actions = tf.placeholder(shape=(timesteps_per_batch), 
                                              dtype=tf.int32) # Discrete action
            self.all_advs = tf.placeholder(shape=(timesteps_per_batch), 
                                          dtype=tf.float32) # Advantages
        
            # weights and biases of a network with one hidden layer
            W0 = tf.Variable(tf.truncated_normal([ob_dim, n_hidden]))
            b0 = tf.Variable(tf.zeros(n_hidden))
            W1 = tf.Variable(tf.truncated_normal([n_hidden, action_dim]))
            b1 = tf.Variable(tf.zeros(action_dim))
        
            # Action probabilities for training
            hidden = tf.tanh(tf.matmul(self.all_obs, W0) + b0)
            logits = tf.matmul(hidden, W1) + b1
            self.action_probs = tf.nn.softmax(logits)
        
            # define the loss and optimizer
        
            N = timesteps_per_batch
            """ workaround because of TF indexing, use element-wise multiplication 
            of a binary mask at selected indices to extract required values"""
            indices = tf.transpose(tf.pack([tf.range(N), self.all_actions]))
            binary_mask = tf.sparse_to_dense(indices, self.action_probs.get_shape(),
                                         1.0)
            mult = tf.mul(binary_mask, self.action_probs)
            prob_actions_taken = tf.reduce_sum(mult, 1)
            # we divide by the total number of timesteps because its an expectation
            self.loss = tf.log(tf.mul(prob_actions_taken, self.all_advs)) / N
        
            alg = tf.train.GradientDescentOptimizer(learning_rate)
            self.optimizer = alg.minimize(self.loss)
                          
            # action probability for control
            self.ob = tf.placeholder(shape=(1, ob_dim), dtype=tf.float32) 
            hidden = tf.nn.relu(tf.matmul(self.ob, W0) + b0)
            logits = tf.matmul(hidden, W1) + b1
            self.action_prob = tf.nn.softmax(logits)
            
            init = tf.initialize_all_variables()
        
        self.session = tf.Session(graph=self.graph)
        self.session.run(init)

    def categorical_sample(self, prob_n):
        """
        Sample from categorical distribution,
        specified by a vector of class probabilities
        """
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        action = (csprob_n > np.random.rand()).argmax()
        return action
                                           
    def act(self, ob):
        """Choose an action."""
                          
        prob = self.session.run(self.action_prob, feed_dict={self.ob:ob})
        action = np.random.choice(9, size=1, p=prob.squeeze())[0]
        return action

    def learn(self, env, player1, player2):
        """Run learning algorithm."""
        
        for iteration in range(self.n_iters):
            # Collect trajectories until we get timesteps_per_batch timesteps
            trajs = []
            timesteps_total = 0
            while timesteps_total < self.timesteps_per_batch:
                traj = env.episode(player1, player2)
                trajs.append(traj1)
                timesteps_total += len(traj["rewards"])
                          
            rets = [traj["rewards"] for traj in trajs]
            maxlen = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(maxlen-len(ret))]) for ret in rets]
            # Compute time-dependent baseline
            baseline = np.mean(padded_rets, axis=0)
            # Compute advantage function
            advs = [ret - baseline[:len(ret)] for ret in rets]
            advs = padded_rets
            # collect and clip data
            all_obs = np.concatenate([traj["obs"] for traj in trajs])
            all_obs = all_obs[:self.timesteps_per_batch, :]
            all_actions = np.concatenate([traj["actions"] for traj in trajs])
            all_actions = all_actions[:self.timesteps_per_batch]
            all_advs = np.concatenate(advs)
            all_advs = all_advs[:self.timesteps_per_batch]
            
            #print("obs", all_obs)
            #print("acts", all_actions)
            #print("advs", all_advs)
        
                          
            # Do policy gradient update step
            self.session.run(self.optimizer, feed_dict={self.all_obs:all_obs, 
                                                        self.all_actions:all_actions,
                                                        self.all_advs:all_advs})
            
            eprews = np.array([traj["rewards"].sum() for traj in trajs]) # episode total rewards
            eplens = np.array([len(traj["rewards"]) for traj in trajs]) # episode lengths
                          
            # Print stats
            #"""
            print("-----------------")
            print("Iteration: \t %i"%iteration)
            print("NumTrajs: \t %i"%len(eprews))
            print("NumTimesteps: \t %i"%np.sum(eplens))
            print("MaxRew: \t %s"%eprews.max())
            print("MeanRew: \t %s +- %s"%(eprews.mean(), eprews.std()/np.sqrt(len(eprews))))
            print("MeanLen: \t %s +- %s"%(eplens.mean(), eplens.std()/np.sqrt(len(eplens))))
            print("-----------------")#"""""
            env.episode(self, self, display=False)



from game import Tic_Tac_Toe, Human, Random
env = Tic_Tac_Toe()
agent = TTT_Agent(env)
agent.learn(env, player1=agent, player2=Random(env))
while True:
    env.episode(Human(), agent, display=True)

