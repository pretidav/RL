import gym
import tensorflow as tf
import numpy as np
from collections import deque
import progressbar
import matplotlib.pyplot as plt 
import random
from tqdm import tqdm

class A2CAgent():
    def __init__(self, enviroment, optimizer):
        
        # Initialize atributes
        self.enviroment = enviroment
        self._state_size = enviroment.observation_space.shape[0]
        self._action_size = enviroment.action_space.shape[0]
        self._optimizer = optimizer
        
        self.experience_replay = deque(maxlen=400000)

        # Initialize discount and exploration rate
        self.gamma = 0.99
        self.epsilon = 0.1

        self.rand_generator = np.random.RandomState() 
        # Build networks
        self.a2c_network = self._build_model()
        #self.target_network = self._build_compile_model()
        self.align_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_model(self):
        eps = 1e-5
        inp = tf.keras.layers.Input(shape=(self._state_size,))
        d1 = tf.keras.layers.Dense(50, activation='relu')(inp)
        
        mu    = tf.keras.layers.Dense(20, activation='linear')(d1)
        mu    = tf.keras.layers.Dense(1,  activation='linear')(mu)

        sigma = tf.keras.layers.Dense(1, activation='exponential')(d1)  # this ensures sigma => 0
        
        sampled_action   = tf.random.normal(shape=(1,), mean=mu, stddev=sigma, dtype=tf.dtypes.float32)
        pi = tf.clip_by_value(sampled_action,
                                self.enviroment.action_space.low[0], 
                                self.enviroment.action_space.high[0])

        v1 = tf.keras.layers.Dense(20, activation='linear')(d1)
        v  = tf.keras.layers.Dense(1, activation='linear')(v1)
        model = tf.keras.Model(inputs=inp, outputs=[v, pi])
        
        actor_loss  = tf.keras.losses.MSE
        critic_loss = -tf.math.log(pi)  #*actor_loss
        #model.compile(loss=[critic_loss,actor_loss], optimizer=self._optimizer)
        model.summary()
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
        # see also https://www.youtube.com/watch?v=D795oNqa-Vk

        minibatch = random.sample(self.experience_replay, batch_size)
        states = np.vstack([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.vstack([x[3] for x in minibatch])
        episodes_done = np.array([not x[4] for x in minibatch])
        target = self.q_network.predict(states)
        target[range(batch_size),actions] = rewards + self.gamma * np.amax(self.target_network.predict(next_states), axis=1) * episodes_done
        self.q_network.fit(states, target, epochs=1, verbose=0, batch_size=batch_size)


if __name__=="__main__":

    env = gym.make('MountainCarContinuous-v0')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    agent = A2CAgent(enviroment=env, optimizer=optimizer)

    # batch_size = 64
    # num_of_episodes = 10000
    # agent.q_network.summary()

    # cum_reward_list = []
    # won = 0
    # old_state = state = env.reset()
    # for i_episode in tqdm(range(num_of_episodes)):
    #     state = env.reset()
    #     state = tf.expand_dims(state, axis=0)
    #     cum_reward = 0
    #     reward = 0
    #     t = 0
    #     terminated = False
    #     agent.epsilon = 1./(float(i_episode/10) + 5.)
    #     if agent.epsilon<0.1:
    #         agent.epsilon=0.1
            
    #     while True:
    #         env.render()
    #         action = agent.act(state)   
    #         next_state, reward, terminated, info = env.step(action)
    #         v = next_state[1]
    #         x = next_state[0]
    #         reward = v**2 + x**2 + reward #supposing x an harmonic oscillator
    #         cum_reward+=reward
    #         next_state = tf.expand_dims(next_state, axis=0)
    #         agent.store(state, action, reward, next_state, terminated)
    #         state = next_state   
    #         t+=1

    #         if t%5==0:
    #           agent.align_target_model()

    #         if terminated:
    #           if x >= 0.5 :
    #               won+=1
    #               print('won')
    #           cum_reward_list.append(cum_reward)
    #           if (i_episode+1)%1==0:
    #               print("Episode {} finished with {}({}) mean reward, eps {}".format(i_episode+1,np.mean(cum_reward_list[-1:]),np.std(cum_reward_list[-1:]),agent.epsilon))
    #           break

    #         if len(agent.experience_replay) > batch_size:
    #             agent.parallel_retrain(batch_size)

    #     if won==100:
    #         print('--- won ---')
    #         agent.q_network.save('./models/qlearning.hdf5',overwrite=True,include_optimizer=False)
    #         break
          
    