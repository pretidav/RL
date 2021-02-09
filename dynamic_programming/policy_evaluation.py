import numpy as np
from tqdm import tqdm
import argparse
class grid():
    def __init__(self,size):
        self.maxX           = size[0]
        self.maxY           = size[1]
        self.actions   = ['U','D','L','R']
        self.create_grid()
        self.create_reward()
        self.initialize_state()

    def __str__(self):
        s = '#'*10+'\n'
        for j in range(self.maxX):
            for i in range(self.maxY):
                s += str(self.lattice[j,i])+'\t'
            s += '\n'
        s += '#'*10
        return s
        
    def create_grid(self):    
        self.lattice = np.zeros(shape=(self.maxX,self.maxY)) 
        print('#'*20)
        print('  Grid {}x{} created'.format(self.maxX,self.maxY))
        print('#'*20)

    def create_reward(self):
        self.reward = np.zeros(shape=(self.maxX,self.maxY))

    def initialize_state(self):
        self.state = (0,0)

    def move(self,position,where):
        old_x,old_y = position
        if where=='U':
            self.state = old_x,old_y+1
        elif where=='D':
            self.state = old_x,old_y-1
        elif where=='L':
            self.state = old_x-1,old_y
        elif where=='R':
            self.state = old_x+1,old_y
        
    def get_state(self):
        return self.state
    
    def policy(self):
        exit(1)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Lx', help = "Lx", default=4)
    parser.add_argument('--Ly', help = "Ly", default=10)
    args = parser.parse_args()
    
    grid = grid([int(args.Lx),int(args.Ly)])
    s = grid.get_state()
    print(s)
    grid.move(s,'L')
    s = grid.get_state()
    print(s)