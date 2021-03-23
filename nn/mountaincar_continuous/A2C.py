import gym
import tensorflow as tf
import numpy as np
from collections import deque
import progressbar
import matplotlib.pyplot as plt 
import random
from tqdm import tqdm
import tensorflow_probability as tfp

class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.002)

    def create_model(self):
        
        def rescale(a):
            return a*tf.constant(self.action_bound)

        state_input = tf.keras.layers.Input((self.state_dim,))
        dense_1 = tf.keras.layers.Dense(50, activation='relu')(state_input)
        dense_2 = tf.keras.layers.Dense(32, activation='relu')(dense_1)
        out_mu = tf.keras.layers.Dense(self.action_dim, activation='tanh')(dense_2)
        mu_output = tf.keras.layers.Lambda(lambda x: rescale(x))(out_mu)
        std_output = tf.keras.layers.Dense(self.action_dim, activation='softplus')(dense_2) 
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model.predict(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=self.action_dim)

    # def log_pdf(self, mu, std, action):
    #     std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
    #     var = std ** 2
    #     log_policy_pdf = -0.5 * (action - mu) ** 2 / \
    #         var - 0.5 * tf.math.log(var * 2 * np.pi)
    #     return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, mu, std, actions, advantages):
        #log_policy_pdf = self.log_pdf(mu, std, actions)
        #log_policy_pdf = dist.log_prob(value=actions)
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        loss_policy = (-dist.log_prob(value=actions) * advantages + 0.002*dist.entropy())
        return tf.reduce_sum(loss_policy)
        
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
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.002)

    def create_model(self):
        state_input = tf.keras.layers.Input((self.state_dim,))
        dense_1 = tf.keras.layers.Dense(50, activation='relu')(state_input)
        dense_2 = tf.keras.layers.Dense(32, activation='relu')(dense_1)
        dense_3 = tf.keras.layers.Dense(16, activation='relu')(dense_2)
        v       = tf.keras.layers.Dense(1, activation='linear')(dense_3)
        return tf.keras.models.Model(state_input, v)

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    @tf.function
    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-3, 1.0]
        self.gamma = 0.99
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound)
        self.critic = Critic(self.state_dim)

    def td_target(self, reward, next_state, done):
        if done:
            return reward
        v_value = self.critic.model.predict(
            np.reshape(next_state, [1, self.state_dim]))
        return np.reshape(reward + self.gamma * v_value[0], [1, 1])

    def advantage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

            
if __name__=="__main__":

    env = gym.make('MountainCarContinuous-v0')
    agent = A2CAgent(env=env)
    print(env.observation_space.shape[0])
    print(env.action_space.shape[0])

    agent.actor.model.summary()
    agent.critic.model.summary()
    batch_size=8
    ep = 0 
    won = 0
    num_of_episodes = 1000
    all_win = []
    for i_episode in tqdm(range(num_of_episodes)):
        state_batch = []
        action_batch = []
        td_target_batch = []
        advantage_batch = []
        episode_reward, done = 0, False

        state = env.reset()
       
        while True:
            #env.render()
            action = agent.actor.get_action(state)
            action = np.clip(action, -agent.action_bound, agent.action_bound)

            next_state, reward, done, _ = env.step(action)
            if done and next_state[0] >= 0.45:
                reward += 600 
            reward += (next_state[0]-0.55)
            state = np.reshape(state, [1, agent.state_dim])
            action = np.reshape(action, [1, agent.action_dim])
            next_state = np.reshape(next_state, [1, agent.state_dim])
            reward = np.reshape(reward, [1, 1])

            td_target = agent.td_target(reward, next_state, done)
            advantage = agent.advantage(
                td_target, agent.critic.model.predict(state))

            state_batch.append(state)
            action_batch.append(action)
            td_target_batch.append(td_target)
            advantage_batch.append(advantage)

            if len(state_batch) >= batch_size or done:    
                states = agent.list_to_batch(state_batch)
                actions = agent.list_to_batch(action_batch)
                td_targets = agent.list_to_batch(td_target_batch)
                advantages = agent.list_to_batch(advantage_batch)
                actor_loss = agent.actor.train(states, actions, advantages)
                critic_loss = agent.critic.train(states, td_targets)

                state_batch = []
                action_batch = []
                td_target_batch = []
                advantage_batch = []

            episode_reward += reward[0][0]
            state = next_state[0]

            if done:
              if state[0] >= 0.45:
                won+=1
                print('won')
                all_win.append(True)
              if (i_episode+1)%1==0:
                print("Episode {} finished with {} mean reward".format(i_episode+1,episode_reward))
              break

        print('wins {}'.format(np.sum(all_win[-50:])))

        if np.sum(all_win[-50:])==50:
            print('--- won ---')
            agent.actor.model.save('./models/actor.hdf5',overwrite=True,include_optimizer=False)
            agent.critic.model.save('./models/critic.hdf5',overwrite=True,include_optimizer=False)
            break
          
