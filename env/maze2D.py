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
        self.obstacles = env_info["obstacles"]
        self.reward_obs_term = [0.0, None, False]

    def describe(self):
        print('Maze - volume = {}'.format(self.maze_dim))
        print('Maze - Wall   = {}'.format(self.obstacles))
        print('Maze - Start  = {}'.format(self.start_state))
        print('Maze - End    = {}'.format(self.end_state))

    
    def plot(self): 
        Lx, Ly = self.maze_dim[0],self.maze_dim[1] 
        print('-'*((2*Lx)+3))
        for x in range(Lx):
            s = '|'
            for y in range(Ly):
                if self.is_obstacle(x,y):
                    s+=' x'
                #elif self.start_state==[x,y]:
                #    s+=' @'
                elif [x,y] in self.end_state:
                    s+=' $'
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

        reward = 0.0
        is_terminal = False

        row = self.current_state[0]
        col = self.current_state[1]

        # update current_state with the action (also check validity of action)
        if action == 0: # up
            if not (self.out_of_bounds(row-1, col) or self.is_obstacle(row-1, col)):
                self.current_state = [row-1, col]

        elif action == 1: # right
            if not (self.out_of_bounds(row, col+1) or self.is_obstacle(row, col+1)):
                self.current_state = [row, col+1]

        elif action == 2: # down
            if not (self.out_of_bounds(row+1, col) or self.is_obstacle(row+1, col)):
                self.current_state = [row+1, col]

        elif action == 3: # left
            if not (self.out_of_bounds(row, col-1) or self.is_obstacle(row, col-1)):
                self.current_state = [row, col-1]

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
    env = MazeEnvironment(
    shape=[10,10], 
    start=[1,1], 
    end=[8,5],
    obstacles=[[3,1],[3,2],[3,3],[3,5]] )

    env.describe()
    env.plot()