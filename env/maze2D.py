#for relative import
import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from RLglue.environment import BaseEnvironment
import numpy as np
import random

class MazeEnvironment(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """
    def random_init(self):
        while True:
            s = [random.randint(0,self.maze_dim[0]-1),random.randint(0,self.maze_dim[1]-1)]
            if not self.is_obstacle(s[0],s[1]):    
                self.start_state = s
                break
        self.current_state = [None for _ in range(len(self.maze_dim))]
        

    def env_init(self,env_info):

        self.maze_dim = env_info["shape"]
        self.start_state = env_info["start"]
        self.end_state = env_info["end"]
        self.current_state = [None for _ in range(len(self.maze_dim))]
        self.obstacles = env_info.get("obstacles",[])
        self.wind = env_info.get("wind",[]) #upward ^ +1 
        self.lava = env_info.get("lava",[]) #reward -10  
        self.reward_obs_term = [0.0, None, False]
        self.change_at_n = env_info.get("change_time",9999999999)
        self.timesteps = 0


    def describe(self):
        print('Maze - volume   = {}'.format(self.maze_dim))
        print('Maze - Wall     = {}'.format(self.obstacles))
        print('Maze - Start    = {}'.format(self.start_state))
        print('Maze - End      = {}'.format(self.end_state))
        print('Maze - Wind     = {}'.format(self.wind))
        print('Maze - Lava     = {}'.format(self.lava))
        print('Maze - change t = {}'.format(self.change_at_n))
        

    
    def plot(self): 
        Lx, Ly = self.maze_dim[0],self.maze_dim[1] 
        print('-'*((2*Lx)+3))
        for x in range(Lx):
            s = '|'
            for y in range(Ly):
                if self.is_obstacle(x,y):
                    s+=' x'
                elif self.start_state==[x,y]:
                    s+=' @'
                elif [x,y] in self.end_state:
                    s+=' $'
                elif [x,y] in self.wind:
                    s+=' ^'
                elif [x,y] in self.lava:
                    s+=' +'    
                else:
                    s+=' o'
            print(s+' |')
        print('-'*((2*Lx)+3))

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = self.start_state
        self.reward_obs_term[1] = self.get_observation(self.current_state)

        return self.reward_obs_term[1]

    # check if current state is within the gridworld and return bool
    def out_of_bounds(self, row, col):
        if row < 0 or row > self.maze_dim[0]-1 or col < 0 or col > self.maze_dim[1]-1:
            return True
        else:
            return False

    # check if there is an obstacle at (row, col)
    def is_obstacle(self, row, col):
        if [row, col] in self.obstacles:
            return True
        else:
            return False

    # check if there is an wind region at (row, col)
    def is_windy(self, row, col):
        if [row, col] in self.wind:
            return True
        else:
            return False

    def is_lava(self, row, col):
        if [row, col] in self.lava:
            return True
        else:
            return False


    def get_observation(self, state):
        return state[0] * self.maze_dim[1] + state[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        self.timesteps += 1
        if self.timesteps == self.change_at_n:
            self.obstacles = self.obstacles[:-1]  #remove last obstacle

        reward = 0.0
        is_terminal = False

        row = self.current_state[0]
        col = self.current_state[1]

        # update current_state with the action (also check validity of action)
        if action == 0: # up
            if not self.is_windy(row,col):
                if not (self.out_of_bounds(row-1, col) or self.is_obstacle(row-1, col)):
                    self.current_state = [row-1, col]
            else:
                if not (self.out_of_bounds(row-2, col) or self.is_obstacle(row-2, col)):
                    self.current_state = [row-2, col]

        elif action == 1: # right
            if not self.is_windy(row,col):
                if not (self.out_of_bounds(row, col+1) or self.is_obstacle(row, col+1)):
                    self.current_state = [row, col+1]
            else:
                if not (self.out_of_bounds(row-1, col+1) or self.is_obstacle(row-1, col+1)):
                    self.current_state = [row-1, col+1]
            
        elif action == 2: # down
            if not self.is_windy(row,col):
                if not (self.out_of_bounds(row+1, col) or self.is_obstacle(row+1, col)):
                    self.current_state = [row+1, col]
            else:
                if not (self.out_of_bounds(row, col) or self.is_obstacle(row, col)):
                    self.current_state = [row, col]

        elif action == 3: # left
            if not self.is_windy(row,col):
                if not (self.out_of_bounds(row, col-1) or self.is_obstacle(row, col-1)):
                    self.current_state = [row, col-1]
            else:
                if not (self.out_of_bounds(row-1, col-1) or self.is_obstacle(row-1, col-1)):
                    self.current_state = [row-1, col-1]

        if self.current_state in self.lava:
            reward = -10.0
            self.current_state = self.start_state  #if fall in lava restart from start_state

        if self.current_state in self.end_state: # terminate if goal is reached
            reward = 1.0
            is_terminal = True

        self.reward_obs_term = [reward, self.get_observation(self.current_state), is_terminal]

        return self.reward_obs_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        current_state = None

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self.reward_obs_term[0])

        # else
        return "I don't know how to respond to your message"




if __name__=='__main__':
    env_info = { 
    "shape": [6,6],
    "start": [0,5],
    "end"  : [[5,5]],
    "obstacles":[[1,1],[1,2],[1,3]],
    "wind":[[2,3],[3,3],[4,3],[5,3]],
    "lava":[[4,5],[3,5]],
    "change_time":10}

    env = MazeEnvironment()
    env.env_init(env_info=env_info)
    env.describe()
    env.plot()

    