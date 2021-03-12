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
        
        self.experience_replay = deque(maxlen=40000)

        # Initialize discount and exploration rate
        self.gamma = 0.999
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
        d1 = tf.keras.layers.Dense(50, activation='relu')(inp)
        d2 = tf.keras.layers.Dense(24, activation='relu')(d1)
        out = tf.keras.layers.Dense(self._action_size, activation='linear')(d2)
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

    def parallel_retrain(self,batch_size):
        # inspired by 
        # https://github.com/Ahmkel/Deep-Learning/blob/master/Reinforcement-Learning/Cart%20Pole%20-%20Deep%20Q-Network%20with%20experience%20replay.ipynb
        
        minibatch = random.sample(self.experience_replay, batch_size)
        states = np.vstack([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.vstack([x[3] for x in minibatch])
        episodes_done = np.array([x[4] for x in minibatch])
        target = self.q_network.predict(next_states)
        target[range(batch_size),actions] = rewards + self.gamma * np.amax(self.target_network.predict(next_states), axis=1) * ~episodes_done
        self.q_network.fit(states, target, epochs=1, verbose=0, batch_size=batch_size)
        target = None


if __name__=="__main__":

    env = gym.make('CartPole-v1')
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    agent = DeepQAgent(enviroment=env, optimizer=optimizer)
    batch_size = 64
    num_of_episodes = 1000
    agent.q_network.summary()

    steps = []
    for i_episode in tqdm(range(num_of_episodes)):
        state = env.reset()
        state = tf.expand_dims(state, axis=0)
        reward = 0
        terminated = False
        t = 0   
        agent.epsilon = 1./((i_episode/20) + 5)
        while True:
            #env.render()
            t+=1
            action = agent.act(state)   
            next_state, reward, terminated, info = env.step(action)   
            next_state = tf.expand_dims(next_state, axis=0)
            agent.store(state, action, reward, next_state, terminated)
            state = next_state   
            if terminated:
                agent.align_target_model()
                steps.append(t)
                if (i_episode+1)%20==0:
                    print("Episode {} finished after {} mean timesteps".format(i_episode+1,np.mean(steps[-10:])))
                break

            if len(agent.experience_replay) > batch_size:
                agent.parallel_retrain(batch_size)

        if i_episode>99:
            if np.mean(steps[-50:])>=150.0:
                break
        
    plt.plot(steps)
    #plt.show()  
    agent.q_network.save('./models/qlearning.hdf5',overwrite=True,include_optimizer=False)
    plt.savefig('./models/qlearning.png')