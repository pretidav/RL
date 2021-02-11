# Library of RL algorithms
This is a compendium of Reinforcement Learning algorithm for educational purposes. 
## k-armed Bandit
python bandit/bandit.py --replicas=1000 --steps=500 --eps=0.20 --method='ucb','stationary','nonstationary' (chose 1)

This algorithm tests the effectiveness of a simple value function estimation method based on incremental updates in an epsilon-greedy or a Upper Confidence Bound (UCB) policy. 
The code runs over values of epsilon from 0 to the inserted one with steps of 0.05, and provide plots of the average reward vs time steps and the probability of taking the best action (in this example 8th-bandit). 

![alt text](https://github.com/pretidav/RL/raw/main/fig/k-bandit.png)

## Dynamic Programming: Policy-Iteration and Value-Iteration
python policy_and_value_iteration.py --Lx=5 --Ly=6 --gamma=0.1

This algorithm find the optimal policy after cycles of " Evaluation - Improvement - Evaluation ... ". 

For a 5x5 gridworld with final states (0,0) and (5,5) (upper left and bottom right respectively), for a discount factor of 0.1, 
this is the optimal value function: 

-0.5705428314180507     -0.840313796948289      -1.1041497707314407     -1.110931481814008      -1.111101656937537 

-0.840313796948289      -1.097545478835686      -1.1105957797655606     -1.1110763630789737     -1.1109314818140454

-1.1041497707314407     -1.1105957797655608     -1.1110595779765515     -1.1105957797655628     -1.1041497707314814

-1.1109314818140077     -1.111076363078974      -1.1105957797655628     -1.0975454788356906     -0.8403137969483416

-1.111101656937537      -1.1109314818140454     -1.1041497707314811     -0.8403137969483415     -0.570542831418329 

While the optimal policy reads: 
{(0, 0): ['U', 'L'], (0, 1): ['U'], (0, 2): ['U'], (0, 3): ['U'], (0, 4): ['U'], (1, 0): ['L'], (1, 1): ['U', 'L'], (1, 2): ['U'], (1, 3): ['U'], (1, 4): ['R'], (2, 0): ['L'], (2, 1): ['L'], (2, 2): ['U', 'L'], (2, 3): ['R'], (2, 4): ['R'], (3, 0): ['L'], (3, 1): ['L'], (3, 2): ['D'], (3, 3): ['D', 'R'], (3, 4): ['R'], (4, 0): ['L'], (4, 1): ['D'], (4, 2): ['D'], (4, 3): ['D'], (4, 4): ['D', 'R']}
