#for relative import
import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from RLglue.rl_glue import RLGlue
from RLglue.agent import BaseAgent
from env.maze2D import MazeEnvironment
import numpy as np

class MCagent(BaseAgent):
    def agent_init(self, agent_info):
        self.num_states  = agent_info["num_states"]
        self.num_actions = agent_info["num_actions"]
        self.gamma       = agent_info["discount"]
        self.rand_generator = np.random.RandomState(42)
        self.epsilon = 0
        self.all_returns = []
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1
        self.model = {} 
        self.pi=np.random.random_integers(4, size=(self.num_states,))-1

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

    def choose_action_pi(self,state):
        return self.pi[state]   

    def choose_action_randomly(self, state):
        action = self.rand_generator.choice(self.actions)
        return action        
        
    def agent_start(self, state):
        self.past_state = state
        self.past_action = self.choose_action_randomly(state)  #Random Starting Exploration (ignore policy)
        return self.past_action

    def agent_step(self, reward, state):  
        
        self.update_model(self.past_state, self.past_action, state, reward)
        self.past_action = self.choose_action_pi(state)
        self.all_returns.append((self.past_state,self.past_action,reward))
        self.past_state = state
        return self.past_action

    def agent_end(self, reward):
        self.update_model(self.past_state, self.past_action, -1, reward)

    def estimate_q(self):
        G = 0
        returns = {}
        for S,A,R in self.all_returns[::-1]:
            G += R
            if (S,A) not in returns.keys():
                returns[(S,A)] = []
            returns[(S,A)].append(G)
            self.q_values[S,A] = np.mean(returns[(S,A)])
            self.pi[S] = self.argmax(self.q_values[S])



if __name__=='__main__':
  
    env_info = { 
        "shape": [10,10],
        "start": [0,0],
        "end"  : [2,2],
        "obstacles":[[1,1],[1,2]]}

    agent_info = {
        "num_states"  : np.prod(env_info["shape"]),   
        "num_actions" : len(env_info["shape"])**2,
        "discount": 0.95}
    
    rl_glue = RLGlue(
        env_class = MazeEnvironment, 
        agent_class = MCagent
        )

    rl_glue.rl_init(agent_info, env_info)
    rl_glue.environment.plot()
    rl_glue.environment.describe()
    rl_glue.agent.describe()

    LIMIT = 2000
    EPISODES = 1000
    for n in range(EPISODES): 
        print(n)
        rl_glue.rl_start()
        is_terminal = False               
        num_steps = 0
        converged = False
        while not is_terminal:
            reward, _, action, is_terminal = rl_glue.rl_step()  
            num_steps += 1                    
            if num_steps > LIMIT:
                break
            else :
                converged = True
        if converged:
            rl_glue.agent.estimate_q()
    
    print(rl_glue.agent.model)
    print(rl_glue.agent.all_returns)
    print(len(rl_glue.agent.all_returns))