import gym
import tensorflow as tf
import numpy as np
from collections import deque
import progressbar
import matplotlib.pyplot as plt 
import random
from tqdm import tqdm

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

    q_network = tf.keras.models.load_model(filepath='./models/qlearning.hdf5')
    q_network.summary()

    print("Play 5 times with optimal policy")
    for i in range(5):
        state = env.reset()
        state = tf.expand_dims(state, axis=0)
        done = False
        total_reward = 0
        while not done:
            state, reward, done, _ = env.step(np.argmax(q_network.predict(state)))
            state = tf.expand_dims(state, axis=0)
            total_reward += reward
            env.render()
        print("Iteration: %d, Total Reward: %d" % (i, total_reward))

    env.close()