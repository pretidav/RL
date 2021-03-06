import gym
import tensorflow as tf


class model():
    def __init__(self,obs_size,action_size,neurons,layers):
        self.obs_size = obs_size
        self.action_size = action_size
        self.neurons = neurons
        self.layers = layers

    def nn(self):
        input_obs = tf.keras.Input(shape=(self.obs_size,),name='observations')
        input_act = tf.keras.Input(shape=(self.action_size,),name='actions')
        input = tf.keras.layers.concatenate([input_obs, input_act])
        l = tf.keras.layers.Dense(units=self.neurons,activation='relu',name='dense_0')(input)
        for n in range(self.layers-1):
            l = tf.keras.layers.Dense(units=self.neurons,activation='relu',name='dense_'+str(n+1))(l)
        output = tf.keras.layers.Dense(units=self.action_size,activation='softmax',name='output')(l)

        model = tf.keras.Model(inputs=[input_obs,input_act],outputs=output)
        return model


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
    """

    for i_episode in range(1):
        observation = env.reset()
        print(env.action_space.n)
        print(env.observation_space.shape[0])
        for t in range(1):
            env.render()
            print(observation)
            action = env.action_space.sample()   #<- random action
            observation, reward, done, info = env.step(action)   #<- step given random action
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
    model = model(
        obs_size    = env.observation_space.shape[0],
        action_size = env.action_space.n,
        neurons=10,
        layers=3 )
    Q = model.nn()
    Q.summary()