#for relative import
import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from RLglue.rl_glue import RLGlue
from RLglue.agent import BaseAgent
from env.maze2D import MazeEnvironment

env = MazeEnvironment(
    shape=[10,10], 
    start=[1,1], 
    end=[8,5],
    obstacles=[[3,1],[3,2],[3,3],[3,5]] )

env.describe()
env.plot()