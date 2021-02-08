# RL
Library of RL algorithms

# k-armed Bandit
python bandit/bandit.py --replicas=1000 --steps=500 --eps=0.20

This algorithm tests the effectiveness of a simple value function estimation method based on incremental updates in an epsilon-greedy policy. 
The code runs over values of epsilon from 0 to the inserted one with steps of 0.05, and provide plots of the average reward vs time steps and the probability of taking the best action (in this example 8th-bandit). 
