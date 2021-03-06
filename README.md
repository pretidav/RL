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
| <!-- -->            | <!-- -->            | <!-- -->            | <!-- -->            | <!-- -->            |  
|---------------------|---------------------|---------------------|---------------------|---------------------|  
| -0.5705428314180507 | -0.840313796948289  | -1.1041497707314407 | -1.110931481814008  | -1.111101656937537  |   
| -0.840313796948289  | -1.097545478835686  | -1.1105957797655606 | -1.1110763630789737 | -1.1109314818140454 |  
| -1.1041497707314407 | -1.1105957797655608 | -1.1110595779765515 | -1.1105957797655628 | -1.1041497707314814 |  
| -1.1109314818140077 | -1.111076363078974  | -1.1105957797655628 | -1.0975454788356906 | -0.8403137969483416 |  
| -1.111101656937537  | -1.1109314818140454 | -1.1041497707314811 | -0.8403137969483415 | -0.570542831418329  |  

While the optimal policy reads: 
| <!-- -->            | <!-- -->            | <!-- -->            | <!-- -->            | <!-- -->            |  
|---------------------|---------------------|---------------------|---------------------|---------------------|  
| ^< | < | < | < | < |   
| ^ | ^< | < | < | v |  
| ^ | ^ | ^< | > | v |  
| ^ | ^ | > | >v | v |  
| ^ | > | > | > | v> |  

## Sampling Methods

python MC_ES.py
or 
python MC_esoft.py

This algorithm implement a Monte Carlo (MC) Exploring Start (ES) or Epsilon-Soft (esoft) algorithm for state-value function estimation and eps-greedy policy improvement. Environment and Agent are managed by the RLglue api. In this implementation the starting state is sampled randomly since we want to estimate the optimal policy in each state. 

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|<!-- -->|<!-- -->|    
|--------|--------|--------|--------|--------|--------|  
| v | v | v | < | < | < |  
| v | < | < | < | < | < |  
| v | x | x | x | x | x |  
| > | > | > | v | v | < |  
| ^ | > | > | > | v | v |  
| ^ | ^ | > | > | $ | < |  

Legenda:   
$: target   
^v<>: actions under the estimated optimal policy in each point.    

## Temporal Difference (tabular) Methods 
Sarsa, Q-learning, Dyna-Q and DynaQ+. 

x: obstacle
+: lava (-10 reward)
@: star
$: target (+1 reward)

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|<!-- -->|<!-- -->|    
|--------|--------|--------|--------|--------|--------|  
| o | o | o | o | o | o |
| o | o | o | x | o | o |
| o | o | o | x | o | o |
| @ | o | o | x | o | $ |
| o | o | o | x | o | o |
| o | o | o | + | o | o |

#: optimal path 

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|<!-- -->|<!-- -->|    
|--------|--------|--------|--------|--------|--------|  
| o | o | # | # | # | # |
| o | # | # | x | o | # |
| o | # | o | x | o | # |
| @ | # | o | o | o | $ |
| o | o | o | x | o | o |
| o | o | o | o | o | o |

with shortcut 

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|<!-- -->|<!-- -->|    
|--------|--------|--------|--------|--------|--------|  
| o | o | o | o | o | o |
| o | o | o | x | o | o |
| o | o | o | x | o | o |
| @ | o | o | o | o | $ |
| o | o | o | x | o | o |
| o | o | o | + | o | o |

|<!-- -->|<!-- -->|<!-- -->|<!-- -->|<!-- -->|<!-- -->|    
|--------|--------|--------|--------|--------|--------|  
| o | o | o | o | o | o |
| o | o | o | x | o | o |
| o | o | o | x | o | o |
| @ | # | # | # | # | $ |
| o | o | o | x | o | o |
| o | o | o | + | o | o |

Changing environment comparison

![alt text](https://github.com/pretidav/RL/raw/main/TD_methods/results/comparison-cum-rew.png)
