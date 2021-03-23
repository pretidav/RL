import gym
import tensorflow as tf
import numpy as np
from collections import deque
import progressbar
import matplotlib.pyplot as plt 
import random
from tqdm import tqdm
import tensorflow_probability as tfp
from threading import Thread
from multiprocessing import cpu_count
            
CUR_EPISODE = 0

class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.002)
        self.entropy_beta = 0.01
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
   
    def compute_loss(self, mu, std, actions, advantages):
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

    #@tf.function
    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss



class A3CAgent:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-3, 1.0]
        self.gamma = 0.99
        self.global_actor = Actor(self.state_dim, 
                                self.action_dim, 
                                self.action_bound, 
                                self.std_bound)
        self.global_critic = Critic(self.state_dim)
        self.num_workers = cpu_count()

    def train(self, max_episodes=1000):
        workers = []

        for i in range(self.num_workers):
            env = gym.make(self.env_name)
            workers.append(WorkerAgent(
                env, self.global_actor, self.global_critic, max_episodes))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class WorkerAgent(Thread):
    def __init__(self, env, global_actor, global_critic, max_episodes):
        Thread.__init__(self)
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]
        self.gamma = 0.99

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound)
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = self.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self):
        global CUR_EPISODE
        all_win = []
        batch_size = 8
        while self.max_episodes >= CUR_EPISODE:
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward, done = 0, False

            state = self.env.reset()

            while True:
                # self.env.render()
                action = self.actor.get_action(state)
                action = np.clip(action, -self.action_bound, self.action_bound)

                next_state, reward, done, _ = self.env.step(action)
                if done and next_state[0] >= 0.45:
                    reward += 600 
                reward += (next_state[0]-0.55)
                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)

                if len(state_batch) >= batch_size or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)

                    next_v_value = self.critic.model.predict(next_state)
                    td_targets = self.n_step_td_target(rewards, next_v_value, done)
                    advantages = td_targets - self.critic.model.predict(states)

                    actor_loss = self.global_actor.train(
                        states, actions, advantages)
                    critic_loss = self.global_critic.train(
                        states, td_targets)

                    self.actor.model.set_weights(
                        self.global_actor.model.get_weights())
                    self.critic.model.set_weights(
                        self.global_critic.model.get_weights())

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    td_target_batch = []
                    advatnage_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]

                if done:
                    if state[0] >= 0.45:
                        won+=1
                        print('won')
                        all_win.append(True)
                    if (CUR_EPISODE+1)%1==0:
                        print("Episode {} finished with {} mean reward".format(CUR_EPISODE,episode_reward))
                        CUR_EPISODE += 1
                    break

            # if np.sum(all_win[-50:])==50:
            #     print('--- won ---')
            #     agent.global_actor.model.save('./models/actorA3C.hdf5',overwrite=True,include_optimizer=False)
            #     agent.global_critic.model.save('./models/criticA3C.hdf5',overwrite=True,include_optimizer=False)
            #     break
        

    def run(self):
        self.train()


def main():
    env_name = 'MountainCarContinuous-v0'
    agent = A3CAgent(env_name)
    agent.train()


if __name__ == "__main__":
    main()
      
