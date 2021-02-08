import numpy as np
import random 
import argparse 
from tqdm import tqdm
import matplotlib.pyplot as plt

class GaussianBandit():
    def __init__(self,mu,sigma,name):
        self.mu     = mu
        self.sigma  = sigma
        self.name   = name
        print("[+] Bandit {}:".format(name))
        print(" mu:{}\n sigma:{}\n".format(mu,sigma))

    def __str__(self):
        return self.name
        
    def play(self):
        return np.random.normal(self.mu,self.sigma)

class GaussianTimeBandit():
    def __init__(self,mu,sigma,k,s,name):
        self.mu     = mu
        self.sigma  = sigma
        self.k      = k
        self.s      = s
        self.name   = name
        print("[+] Bandit {}:".format(name))
        print(" mu:{}\n sigma:{}\n k:{}\n s:{}\n".format(mu,sigma,k,s))
    
    def __str__(self):
        return self.name
        
    def play(self,t):
        return np.random.normal(self.mu+t*k,self.sigma*abs(self.s))

class AverageMethodNotStationary():
    def __init__(self,eps,alpha,replicas,steps,bandits):
        self.eps = eps 
        self.alpha = alpha
        self.replicas = replicas
        self.steps = steps
        self.bandits = bandits

        print('[+] AverageMethod\n Parameters:\n replicas:{}\n steps:{}\n eps:{}\n alpha:{}\n bandits:{}\n'.format(
        self.replicas,self.steps,self.eps,self.alpha,len(self.bandits)))

    def __str__(self):
        return 'Parameters:\n replicas:{}\n steps:{}\n eps:{}\n alpha:{}\n bandits:{}\n'.format(
            self.replicas,self.steps,self.eps,self.alpha,len(self.bandits))

    def create_actions(self,bandits):
        return [i for i in range(0,len(bandits))]

    def eps_greedy(self,Q,A):
        if np.random.random()<=self.eps:
            return np.random.choice(A)
        else :
            return np.random.choice(np.flatnonzero(Q==np.max(Q)))
        
    def update_Q(self,Q,n,R):
        return (1-self.alpha)**n*Q + self.alpha*R
        
    def optimize(self,Q_start):
        print('[+] Optimization started')
        Q = np.copy(Q_start)
        hist_R  = np.zeros(shape=(self.replicas,self.steps))
        hist_Ra = np.zeros(shape=(self.replicas,self.steps))
        hist_A = np.zeros(shape=(self.replicas,self.steps,len(Q_start))) 
        Actions = self.create_actions(self.bandits)
        for rep in tqdm(range(self.replicas)):
            R = 0
            Q = np.copy(Q_start)
            for n in range(self.steps):
                a  = self.eps_greedy(Q,Actions)
                hist_A[rep,n,a] += 1
                R *=(1-self.alpha)
                R += bandits[a].play()
                hist_R[rep,n] = R
                hist_Ra[rep,n] = bandits[a].play()
                Q[a] = self.update_Q(Q[a],n,R)
        return hist_R, hist_Ra, hist_A

class AverageMethodStationary():
    def __init__(self,eps,replicas,steps,bandits):
        self.eps = eps 
        self.replicas = replicas
        self.steps = steps
        self.bandits = bandits

        print('[+] AverageMethod\n Parameters:\n replicas:{}\n steps:{}\n eps:{}\n bandits:{}\n'.format(
        self.replicas,self.steps,self.eps,len(self.bandits)))

    def __str__(self):
        return 'Parameters:\n replicas:{}\n steps:{}\n eps:{}\n bandits:{}\n'.format(
            self.replicas,self.steps,self.eps,len(self.bandits))

    def create_actions(self,bandits):
        return [i for i in range(0,len(bandits))]

    def eps_greedy(self,Q,A):
        if np.random.random()<=self.eps:
            return np.random.choice(A)
        else :
            return np.random.choice(np.flatnonzero(Q==np.max(Q)))
               
    def optimize(self,Q_start):
        print('[+] Optimization started')
        Q = np.copy(Q_start)
        hist_R  = np.zeros(shape=(self.replicas,self.steps))
        hist_Ra = np.zeros(shape=(self.replicas,self.steps))
        hist_A = np.zeros(shape=(self.replicas,self.steps,len(Q_start))) 
        Actions = self.create_actions(self.bandits)
        for rep in tqdm(range(self.replicas)):
            R = 0
            Q = np.copy(Q_start)
            N = np.zeros(len(Q_start))
            for n in range(self.steps):
                a  = self.eps_greedy(Q,Actions)
                hist_A[rep,n,a] += 1
                R = bandits[a].play()
                hist_R[rep,n] = R
                hist_Ra[rep,n] = bandits[a].play()
                N[a] +=1
                tmp = Q[a] + float((R-Q[a])/N[a])
                Q[a] = tmp
        return hist_R, hist_Ra, hist_A


if __name__=='__main__':
    print('='*20)
    print(' Multi-Armed Bandit ')
    print('='*20)
    
    #np.random.seed(1)    #fix random seed

    parser = argparse.ArgumentParser()
    parser.add_argument('--replicas', help = "number replicas", default=100)
    parser.add_argument('--steps', help = "number time steps", default=50)
    parser.add_argument('--epsmax', help = "epsilon greedy max parameter", default=0.1)
    parser.add_argument('--stationary', help="if true stationary method", action='store_true')

    args = parser.parse_args()
    replicas = int(args.replicas)
    steps    = int(args.steps)
    epsmax      = float(args.epsmax) 

    bandits = [
        GaussianBandit(1.0,0.1,'a'),
        GaussianBandit(5.0,0.3,'b'),
        GaussianBandit(0.5,2  ,'c'),
        GaussianBandit(2  ,3  ,'d'),
        GaussianBandit(4  ,0.1,'e'),
        GaussianBandit(3  ,2  ,'f'),
        GaussianBandit(6  ,0.3,'g'),
        GaussianBandit(8  ,0.3,'h'),
    ]
    
    if args.stationary:
        leg = []
        for eps in np.arange(0,epsmax,0.05):
            method = AverageMethodStationary(
                eps=eps,
                replicas=replicas,
                steps=steps,
                bandits=bandits)

            Q_start = np.array([0.0 for _ in range(0,len(bandits))]) #Optimism
            R,Ra,A = method.optimize(Q_start)
            A_mean = np.mean(A,axis=0)
            print('Optimal action "h": {}%'.format(100*A_mean[-1,-1]))
            plt.plot(np.mean(Ra,axis=0))
            leg.append(str(eps))
        plt.legend(leg)
        plt.show()
    else :
        leg = []
        for eps in np.arange(0,epsmax,0.05):
            method = AverageMethodNotStationary(
                eps=eps,
                alpha=0.5, 
                replicas=replicas,
                steps=steps,
                bandits=bandits)

            Q_start = np.array([0.0 for _ in range(0,len(bandits))]) #Optimism
            R,Ra,A = method.optimize(Q_start)
            A_mean = np.mean(A,axis=0)
            print('Optimal action "h": {}%'.format(100*A_mean[-1,-1]))
            plt.plot(np.mean(Ra,axis=0))
            leg.append(str(eps))
        plt.legend(leg)
        plt.show()
    
