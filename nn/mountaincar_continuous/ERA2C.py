import gym
import tensorflow as tf
import numpy as np
from collections import deque
import progressbar
import matplotlib.pyplot as plt 
import random
from tqdm import tqdm


class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    def create_model(self):
        state_input = tf.keras.layers.Input((self.state_dim,))
        dense_1 = tf.keras.layers.Dense(32, activation='relu')(state_input)
        dense_2 = tf.keras.layers.Dense(32, activation='relu')(dense_1)
        out_mu = tf.keras.layers.Dense(self.action_dim, activation='tanh')(dense_2)
        mu_output = tf.keras.layers.Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = tf.keras.layers.Dense(self.action_dim, activation='softplus')(dense_2) #why softplus instead of exp???
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model.predict(state)
        mu, std = mu[0], std[0]
        a = np.random.normal(mu, std, size=self.action_dim)
        return  np.clip(a, -self.action_bound, self.action_bound)


    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, mu, std, actions, advantages):
        log_policy_pdf = self.log_pdf(mu, std, actions)
        loss_policy = log_policy_pdf * (advantages)
        return tf.reduce_sum(-loss_policy)
        
    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(mu, std, actions, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model = self.create_model()

    def create_model(self):
        state_input = tf.keras.layers.Input((self.state_dim,))
        dense_1 = tf.keras.layers.Dense(32, activation='relu')(state_input)
        dense_2 = tf.keras.layers.Dense(32, activation='relu')(dense_1)
        dense_3 = tf.keras.layers.Dense(16, activation='relu')(dense_2)
        v       = tf.keras.layers.Dense(1, activation='linear')(dense_3)
        model   = tf.keras.models.Model(state_input, v)
        model.compile(loss='mse', optimizer=self.opt)
        return model 
        
class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-3, 1.0]
        self.gamma = 0.9
        self.experience_replay = deque(maxlen=400000)

        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound)
        self.critic = Critic(self.state_dim)

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))

    def parallel_train(self,batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)
        td_targets = np.zeros(batch_size)
        states = np.vstack([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.vstack([x[3] for x in minibatch])
        episodes_done = np.array([not x[4] for x in minibatch])
        td_targets[range(batch_size)] = rewards + self.gamma * self.critic.model.predict(next_states)[0] * episodes_done
        advantages = td_targets - self.critic.model.predict(states)
        actor_loss = self.actor.train(states, actions, advantages)
        critic_loss = self.critic.model.fit(states, td_targets, epochs=1, verbose=0, batch_size=batch_size)
            
if __name__=="__main__":

    env = gym.make('MountainCarContinuous-v0')
    agent = A2CAgent(env=env)
    print(env.observation_space.shape[0])
    print(env.action_space.shape[0])

    agent.actor.model.summary()
    agent.critic.model.summary()
    batch_size=64

    won = 0
    num_of_episodes = 1000
    for i_episode in tqdm(range(num_of_episodes)):
        
        episode_reward, done = 0, False
        state = env.reset()
        state = tf.expand_dims(state, axis=0)
        while True:
            #env.render()
            action = agent.actor.get_action(state)
            next_state, reward, done, _ = env.step(action)

            action = tf.expand_dims(action, axis=0)
            next_state = tf.expand_dims(next_state, axis=0)
            agent.store(state, action, reward, next_state, done)

            if len(agent.experience_replay) > batch_size:    
                agent.parallel_train(batch_size)

            episode_reward += reward
            state = next_state
            
            if done:
              if np.mean(state[0][0]) >= 0.45:
                won+=1
                print('won')
              if (i_episode+1)%1==0:
                print("Episode {} finished with {} mean reward".format(i_episode+1,episode_reward))
              break

        if won==100:
            print('--- won ---')
            agent.q_network.save('./models/a2c.hdf5',overwrite=True,include_optimizer=False)
            break
          
    