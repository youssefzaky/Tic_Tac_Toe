<b>Task Setup (game.py) </b>:  The game is setup so the learning agent plays against itself. If 'X' wins, we get a reward of 1. If 'O' wins,
we get a reward of -1. All other states give 0 reward. 

<b> Action Selection </b>: In all cases, actions are taken according to the epsilon-greedy strategy to keep some exploration going. When its
X's turn, the action is taken to maximize reward. When it's O's turn, the action is taken to minimize reward. This allows
the same agent to play both X and O.

<b> q_table.py </b>: Implementation of tabular Q-learning. Q-values are updated assuming the players are playing optimally.

<b> q_approx.py </b>: Approximates the state-action value function using the Monte Carlo estimates of the returns as targets.
It iterates policy evaluation and policy improvement: it runs the policy for a while to get training data, and performs the 
value function update. The next batch of data is then generated using the policy that uses the new value function. 

<b> sarsa.py </b>: Approximates the state-action value function using the TD(0) backups as targets. Follows the algorithm
on pg. 230 of Sutton's book (draft of second edition).

The tabular method performs the best and is able to learn the best strategy. The approximation methods vary according
to the approximator used and the choice of hyperparameters. The Monte Carlo method seems more stable in learning
and produces decent tic-tac-toe agents.

<b> pg.py </b>: An implementation of the REINFORCE policy gradient method (still buggy).
