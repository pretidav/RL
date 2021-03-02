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
import matplotlib.pyplot as plt

class DynaQplus(BaseAgent):
    def agent_init(self, agent_info):
        self.num_states  = agent_info["num_states"]
        self.num_actions = agent_info["num_actions"]
        self.gamma       = agent_info["discount"]
        self.planning_steps = agent_info["planning_steps"]
        self.rand_generator = np.random.RandomState()
        self.planning_rand_generator = np.random.RandomState()
        self.epsilon = agent_info["epsilon"]
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1
        self.model = {} 
        self.alpha = agent_info["alpha"]
        self.kappa = agent_info["kappa"]
        self.tau = np.zeros((self.num_states, self.num_actions))

    def describe(self):
        print('Agent: number of states   = {}'.format(self.num_states))
        print('Agent: number of actions  = {}'.format(self.num_actions))
        print('Agent: discount factor    = {}'.format(self.gamma))
        print('Agent: epsilon            = {}'.format(self.epsilon))
        print('Agent: alpha              = {}'.format(self.alpha))
        print('Agent: number of planning = {}'.format(self.planning_steps))
        print('Agent: kappa              = {}'.format(self.kappa))

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

    def planning_step(self):
        for steps in range(self.planning_steps):
            picked_state = self.planning_rand_generator.choice(list(self.model.keys()))
            picked_action = self.planning_rand_generator.choice(list(self.model[picked_state].keys()))

            (sprime, reward) = self.model[picked_state][picked_action]
            reward += self.kappa*np.sqrt(self.tau[picked_state,picked_action])
            if sprime != -1 :
                self.q_values[picked_state,picked_action] = self.q_values[picked_state,picked_action] + self.alpha* (reward  + self.gamma* np.max(self.q_values[sprime,:]) - self.q_values[picked_state,picked_action])
            else : 
                self.q_values[picked_state,picked_action] = self.q_values[picked_state,picked_action] + self.alpha* ( reward - self.q_values[picked_state,picked_action])

    def agent_step(self, reward, state):  
        self.update_model(self.past_state, self.past_action, state, reward)
        action = self.choose_action_egreedy(state)
        self.q_values[self.past_state,self.past_action]=self.q_values[self.past_state,self.past_action] + self.alpha*(reward + self.gamma*np.max(self.q_values[state,:]) - self.q_values[self.past_state,self.past_action])
        self.past_action = action
        self.past_state = state
        return self.past_action

    def agent_end(self, reward):
        self.q_values[self.past_state,self.past_action]=self.q_values[self.past_state,self.past_action] + self.alpha*(reward - self.q_values[self.past_state,self.past_action])
        self.update_model(self.past_state, self.past_action, -1, reward)
        self.planning_step()

if __name__=='__main__':
  
    env_info = { 
        "shape": [6,6],
        "start": [3,0],
        "end"  : [[3,5]],
        "wind":[],
        "obstacles": [[1,3],[2,3],[4,3],[3,3]],
        "lava":[[5,3],[1,3]],
        "change_time": 900}

    agent_info = {
        "num_states"  : np.prod(env_info["shape"]),   
        "num_actions" : len(env_info["shape"])**2,
        "discount": 0.95,
        "epsilon": 0.1,
        "alpha": 0.2,
        "planning_steps": 20,
        "kappa": 0.0005}
    
    rl_glue = RLGlue(
        env_class = MazeEnvironment, 
        agent_class = DynaQplus
        )


    EPISODES  = 1000
    REPLICAS  = 50
    MAX_STEPS = 5000

    tot_steps = np.zeros((REPLICAS,EPISODES))
    cum_reward_all = np.zeros((REPLICAS,EPISODES))
    
    rl_glue.rl_init(agent_info, env_info)
    rl_glue.environment.plot()
    rl_glue.environment.describe()
    rl_glue.agent.describe()
    
    for r in tqdm(range(REPLICAS)):
        rl_glue = RLGlue(
            env_class = MazeEnvironment, 
            agent_class = DynaQplus
            )
        rl_glue.rl_init(agent_info, env_info)
        cum_reward = 0
        for n in range(EPISODES):
            n_step = 0
            rl_glue.rl_start()
            is_terminal = False               
            converged = False
            update = False
            while not is_terminal:
                n_step +=1
                reward, _, action, is_terminal = rl_glue.rl_step()
                cum_reward += reward
                cum_reward_all[r][n] = cum_reward
                if n_step > MAX_STEPS:
                    update=False
                    break
                else: 
                    update=True
                
            if update:
                tot_steps[r,n] = n_step
            
    #from last experiment :
    rl_glue.plot_opt_policy()
    path, ll = rl_glue.get_shortest_path()
    rl_glue.print_shortest_path(path)

    fig = plt.figure()
    plt.plot(np.mean(tot_steps,axis=0))
    plt.legend(['Dyna-Q+'])
    plt.xlabel('episodes')
    plt.ylabel('shortest path lenght')
    plt.axhline(y=7, linestyle='--', color='grey', alpha=0.4)
    plt.savefig('../fig/Dyna-Q+.png')


    fig = plt.figure()
    plt.plot(np.mean(cum_reward_all, axis=0), label='Dyna-Q+')
    plt.axvline(x=env_info["change_time"], linestyle='--', color='grey', alpha=0.4)
    plt.xlabel('episodes')
    plt.ylabel('Cumulative Reward')
    plt.savefig('../fig/Dyna-Q+-cum-rew.png')
