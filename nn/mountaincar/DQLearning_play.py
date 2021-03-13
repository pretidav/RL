import gym
import tensorflow as tf
import numpy as np
from collections import deque
import progressbar
import matplotlib.pyplot as plt 
import random
from tqdm import tqdm

if __name__=="__main__":

    env = gym.make('MountainCar-v0')
    """
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.
    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
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