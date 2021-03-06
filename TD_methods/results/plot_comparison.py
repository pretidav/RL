import numpy as np
import matplotlib.pyplot as plt

data_dynaQ     = np.load('./DynaQ.npy', allow_pickle=True).item()
data_dynaQplus = np.load('./DynaQ+.npy', allow_pickle=True).item()

cum_reward_all_dyna     = data_dynaQ["cum_reward_all"]  
cum_reward_all_dynaplus = data_dynaQplus["cum_reward_all"]  
change_time = data_dynaQplus["change_t"]

fig = plt.figure()
plt.plot(np.mean(cum_reward_all_dyna, axis=0))
plt.plot(np.mean(cum_reward_all_dynaplus, axis=0))
plt.legend(['Dyna-Q','Dyna-Q+'])
plt.axvline(x=change_time, linestyle='--', color='grey', alpha=0.4)
plt.xlabel('time steps')
plt.ylabel('Cumulative Reward')
plt.savefig('./comparison-cum-rew.png')
