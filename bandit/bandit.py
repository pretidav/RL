import numpy as np
import random 
import argparse 

class GaussianBandit():
    def __init__(self,mu,sigma,name):
        self.mu     = mu
        self.sigma  = sigma
        self.name   = name
        print("[+] Bandit {}:".format(name))
        print(" mu:{}\n sigma:{}".format(mu,sigma))
    def __str__(self):
        return self.name
        
    def play(self):
        return np.random.normal(self.mu,self.sigma,1)

class GaussianTimeBandit():
    def __init__(self,mu,sigma,k,s,name):
        self.mu     = mu
        self.sigma  = sigma
        self.k      = k
        self.s      = s
        self.name   = name
        print("[+] Bandit {}:".format(name))
        print(" mu:{}\n sigma:{}\n k:{}\n s:{}".format(mu,sigma,k,s))
    def play(self,t):
        return np.random.normal(self.mu+t*k,self.sigma*abs(self.s),1)

def eps_greedy(eps):
    if np.random()<eps:
        return np.random.choice(A)
    if np.random()>eps:
        return np.argmax(Q)
    elif np.random()==eps:
        if np.random<0.5:
            return np.random.choice(A)
        else :
            return np.argmax(Q)

def create_actions(bandits):
    return [i for i in range(0,len(bandits))]

def initialize_Q(bandits):
    return [0 for _ in range(0,len(bandits))]

if __name__=='__main__':
    print('='*20)
    print(' Multi-Armed Bandit ')
    print('='*20)
    
    np.random.seed(1)    #fix random seed

    parser = argparse.ArgumentParser()
    parser.add_argument('--replicas', help = "number replicas", default=100)
    parser.add_argument('--steps', help = "number time steps", default=50)
    parser.add_argument('--eps', help = "epsilon greedy parameter", default=0.1)
    
    args = parser.parse_args()
    replicas = args.replicas
    steps    = args.steps
    eps      = args.eps 

    bandits = [
        GaussianBandit(1.0,0.5,'a'),
        GaussianBandit(2.0,0.5,'b'),
        GaussianBandit(0.5,2  ,'c'),
        GaussianBandit(3  ,3  ,'d')
    ]

    for rep in range(replicas):
        for n in range(steps):

            Actions = create_actions(bandits)
            Q       = initialize_Q(bandits)
            print(Actions)
            print(Q)