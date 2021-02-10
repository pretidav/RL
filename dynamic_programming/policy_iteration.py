import numpy as np
from tqdm import tqdm
import argparse
import random
class grid():
    def __init__(self,size):
        self.maxX           = size[0]
        self.maxY           = size[1]
        self.create_value_function()
        self.create_reward()
        self.actions   = ['U','D','L','R']
        self.policy = {}
        self.p = {}
        self.create_p()
        self.create_policy()

    def create_policy(self):
        for i in range(self.maxX):
            for j in range(self.maxY):
                self.policy[(i,j)] = self.actions

    def create_p(self):
        for i in range(self.maxX):
            for j in range(self.maxY):
                self.p[(i,j)] = dict(zip(self.actions,[float(1/len(self.actions)) for _ in range(len(self.actions))]))

    def update_p(self):
        for i in range(self.maxX):
            for j in range(self.maxY):
                self.p[(i,j)] = dict(zip(self.actions,[float(1/len(self.actions)) for _ in range(len(self.actions))]))

    def print_value_function(self):
        s = '#'*10+'\n'
        for j in range(self.maxX):
            for i in range(self.maxY):
                s += str(self.V[i,j])+'\t'
            s += '\n'
        s += '#'*10
        print(s)
        
    def print_reward(self):
        s = '#'*10+'\n'
        for j in range(self.maxX):
            for i in range(self.maxY):
                s += str(self.reward[i,j])+'\t'
            s += '\n'
        s += '#'*10
        print(s)
    
    def create_value_function(self):    
        self.V = np.zeros(shape=(self.maxX,self.maxY)) 
        print('[+] V(s)  {}x{} created'.format(self.maxX,self.maxY))


    def create_reward(self):
        self.reward = np.zeros(shape=(self.maxX,self.maxY))
        self.reward[:,:]= -1
        self.reward[0,0] = 0
        self.reward[self.maxX-1,self.maxY-1] = 0
        
        print('[+] R(s)  {}x{} created'.format(self.maxX,self.maxY))
         
    def move(self,action,state):
        x,y = state
        if action=='D':
            if y!=self.maxY-1:
                return (x,y+1)
            else :
                return x,y
        elif action=='U':
            if y!=0:
                return (x,y-1)
            else :
                return x,y
        elif action=='L':
            if x!=0:
                return (x-1,y)
            else :
                return x,y
        elif action=='R':
            if x!=self.maxX-1:
                return (x+1,y)
            else :
                return x,y


def policy_iteration_method(grid,gamma):
    def policy_evaluation():
        epsilon = 0.000000001
        counter = 0
        while True:
            delta = 0.0
            counter += 1
            for i in range(Lx):
                for j in range(Ly):
                    Vs = grid.V[(i,j)] 
                    tmp = 0
                    for a in grid.actions:
                        tmp += grid.p[(i,j)][a]*(grid.reward[grid.move(a,(i,j))] + gamma*grid.V[grid.move(a,(i,j))])
                    grid.V[(i,j)] = tmp
                    delta = max(delta,np.abs(Vs - grid.V[(i,j)]))   
            print('[-] Evaluation swipe: {}'.format(counter))
            if delta < epsilon:
                print('[+] Policy evaluated')
                break

    while True:
        policy_evaluation()
        #policy improvement
        print('[+] Policy Improved')
        policy_stable = True
        for i in range(Lx):
            for j in range(Ly):
                old_a = grid.policy[(i,j)]
                Qs = [grid.p[(i,j)][a]*(grid.reward[grid.move(a,(i,j))] + gamma*grid.V[grid.move(a,(i,j))]) for a in grid.actions]
                grid.policy[(i,j)] = [grid.actions[i] for i in np.flatnonzero(Qs==np.max(Qs))]   
                if old_a!=grid.policy[(i,j)]:
                    policy_stable = False
        if policy_stable == True:
            break        
    print('[+] Optimal Policy:')
    print(grid.policy)
    grid.print_value_function()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Lx', help = "Lx", default=4)
    parser.add_argument('--Ly', help = "Ly", default=10)
    parser.add_argument('--gamma', help = "discount", default=1)
    parser.add_argument('--count', help = "discount", default=1)
    
    args = parser.parse_args()
    
    Lx = int(args.Lx)
    Ly = int(args.Ly)  
    gamma = float(args.gamma)

    grid = grid([Lx,Ly])
    policy_iteration_method(grid,gamma)
    