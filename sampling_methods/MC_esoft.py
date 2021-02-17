#for relative import
import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from RLglue.rl_glue import RLGlue
from RLglue.agent import BaseAgent
from env.maze2D import MazeEnvironment
import numpy as np
from tqdm import tqdm

class MCagent(BaseAgent):
    def agent_init(self, agent_info):
        self.num_states  = agent_info["num_states"]
        self.num_actions = agent_info["num_actions"]
        self.gamma       = agent_info["discount"]
        self.rand_generator = np.random.RandomState()
        self.epsilon = agent_info["epsilon"]
        self.all_returns = []
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1
        self.model = {} 

    def describe(self):
        print('Agent: number of states  = {}'.format(self.num_states))
        print('Agent: number of actions = {}'.format(self.num_actions))
        print('Agent: discount factor   = {}'.format(self.gamma))
        print('Agent: epsilon           = {}'.format(self.epsilon))
        

    def update_model(self, past_state, past_action, state, reward):
        if past_state in self.model:
            d = {}
            d[past_action] = (state,reward)
            self.model[past_state].update(d)
        else :
            d = {}
            d[past_action] = (state,reward)
            self.model[past_state] = d   
                
    def argmax(self, q_values):
        top = float("-inf")
        ties = []
        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []
            if q_values[i] == top:
                ties.append(i)
        return self.rand_generator.choice(ties)

    def choose_action_egreedy(self, state):
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.actions)
        else:
            values = self.q_values[state]
            action = self.argmax(values)
        return action     

    def choose_action_randomly(self, state):
        action = self.rand_generator.choice(self.actions)
        return action        
        
    def choose_action_esoft(self,state):
        values = self.q_values[state]
        Astar = [i for i in np.flatnonzero(values==np.max(values))]
        if (len(Astar)!=len(self.actions)) and (self.rand_generator.rand() < (self.epsilon/len(self.actions)) ):
            action = self.rand_generator.choice([i for i in self.actions if i not in Astar])
        else:
            action = self.rand_generator.choice(Astar)
        return action     

    def init_random_q(self):
        self.q_values = self.rand_generator.rand(self.num_states, self.num_actions) 

    def agent_start(self, state):
        self.past_state = state
        self.past_action = self.choose_action_esoft(state)  
        self.all_returns = []
        return self.past_action

    def agent_step(self, reward, state):  
        self.update_model(self.past_state, self.past_action, state, reward)
        self.past_action = self.choose_action_esoft(state)
        self.all_returns.append((self.past_state,self.past_action,reward))
        self.past_state = state
        return self.past_action

    def agent_end(self, reward):
        self.update_model(self.past_state, self.past_action, -1, reward)
        self.all_returns.append((self.past_state,self.past_action,reward))

    def estimate_q(self):
        G = 0
        returns = {}
        for S,A,R in self.all_returns[::-1]:
            G *= self.gamma
            G += R
            if (S,A) not in returns.keys():
                returns[(S,A)] = []
            returns[(S,A)].append(G)
            self.q_values[S,A] = np.mean(returns[(S,A)])

        

if __name__=='__main__':
  
    env_info = { 
        "shape": [6,6],
        "start": [0,5],
        "end"  : [[5,4],[5,5]],
        "obstacles":[[1,1],[1,2],[1,3],[1,4],[1,5]]}

    agent_info = {
        "num_states"  : np.prod(env_info["shape"]),   
        "num_actions" : len(env_info["shape"])**2,
        "discount": 0.95,
        "epsilon": 0.15}
    
    rl_glue = RLGlue(
        env_class = MazeEnvironment, 
        agent_class = MCagent
        )

    rl_glue.rl_init(agent_info, env_info)
    rl_glue.agent.init_random_q()
    rl_glue.environment.plot()
    rl_glue.environment.describe()
    rl_glue.agent.describe()

    EPISODES = 50000
    MAX_STEPS = 1000
    tot_steps = []
    for n in tqdm(range(EPISODES)): 
        n_step = 0
        rl_glue.environment.random_init()
        rl_glue.environment.env_start()
        rl_glue.rl_start()
        is_terminal = False               
        converged = False
        update = False
        while not is_terminal:
            n_step +=1
            reward, _, action, is_terminal = rl_glue.rl_step()
            if n_step > MAX_STEPS:
                update=False
                break
            else :
                update = True
        if update==True :                  
            rl_glue.agent.estimate_q()
        tot_steps.append(n_step)
    rl_glue.plot_opt_policy()
