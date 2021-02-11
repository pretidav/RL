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
    def __init__(self, num_states, num_actions, discount):
        self.num_states  = num_states
        self.num_actions = num_actions
        self.gamma       = discount
        self.rand_generator = np.random.RandomState(42)
        self.epsilon = 0
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
        
        
    def agent_start(self, state):
        self.past_state = state
        self.past_action = self.choose_action_egreedy(state) 
        return self.past_action

    def agent_step(self, reward, state):
        self.q_values[self.past_state,self.past_action].append(self.q_values[self.past_state,self.past_action])
        self.update_model(self.past_state, self.past_action, state, reward)
        self.past_action = self.choose_action_egreedy(state)
        self.past_state = state
        return self.past_action

    def agent_end(self, reward):
        self.q_values[self.past_state,self.past_action] = self.q_values[self.past_state,self.past_action] + self.step_size* (reward  + self.gamma* np.max(self.q_values[-1,:]) - self.q_values[self.past_state,self.past_action])
        self.update_model(self.past_state, self.past_action, -1, reward)


if __name__=='__main__':

    env = MazeEnvironment(
        shape=[10,10], 
        start=[1,1], 
        end=[8,5],
        obstacles=[[3,1],[3,2],[3,3],[3,5]] )
    env.plot()
    env.describe()

    agent = MCagent(
        num_states  = np.prod(env.maze_dim),
        num_actions = 4,
        discount    = 0.9
    )          
    agent.describe()
