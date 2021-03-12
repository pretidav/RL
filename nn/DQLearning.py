import gym
import tensorflow as tf
import numpy as np
from collections import deque
import progressbar
import matplotlib.pyplot as plt 
import random
from tqdm import tqdm

class DeepQAgent():
    def __init__(self, enviroment, optimizer):
        
        # Initialize atributes
        self.enviroment = enviroment
        self._state_size = enviroment.observation_space.shape[0]
        self._action_size = enviroment.action_space.n
        self._optimizer = optimizer
        
        self.experience_replay = deque(maxlen=4000)

        # Initialize discount and exploration rate
        self.gamma = 0.8
        self.epsilon = 0.1

        self.rand_generator = np.random.RandomState() 
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.align_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        inp = tf.keras.layers.Input(shape=(self._state_size,))
        d1 = tf.keras.layers.Dense(24, activation='relu')(inp)
        dp1 = tf.keras.layers.Dropout(0.2)(d1)
        d3 = tf.keras.layers.Dense(24, activation='relu')(dp1)
        out = tf.keras.layers.Dense(self._action_size, activation='linear')(d3)
        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

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

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.enviroment.action_space.sample()
        
        q_values = self.q_network.predict(state)
        return self.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)
        
        for state, action, reward, next_state, terminated in minibatch:
            target = self.q_network.predict(state)
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.q_network.fit(state, target, epochs=1, verbose=0)


if __name__=="__main__":

    env = gym.make('CartPole-v0')
    """
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    agent = DeepQAgent(enviroment=env, optimizer=optimizer)
    batch_size = 32
    num_of_episodes = 100
    timesteps_per_episode = 150
    min_epsilon = 0.005
    agent.q_network.summary()

    steps = []
    for i_episode in tqdm(range(num_of_episodes)):
        state = env.reset()
        state = tf.expand_dims(state, axis=0)
        reward = 0
        terminated = False
    
        for t in range(timesteps_per_episode):
            #env.render()
            action = agent.act(state)   #<- random action
            next_state, reward, terminated, info = env.step(action)   #<- step given random action
            next_state = tf.expand_dims(next_state, axis=0)
            agent.store(state, action, reward, next_state, terminated)
            state = next_state   
            
            if terminated or t==timesteps_per_episode-1:
                agent.align_target_model()
                print("Episode {} finished after {} timesteps".format(i_episode,t+1))
                steps.append(t)
                #if agent.epsilon>min_epsilon:
                #    agent.epsilon*=0.99
                break
        
                 
            if len(agent.experience_replay) > batch_size:
                agent.retrain(batch_size)

        
    plt.plot(steps)
    #plt.show()  
    agent.q_network.save('./models/qlearning.hdf5',overwrite=True,include_optimizer=False)
    plt.savefig('./models/qlearning.png')
