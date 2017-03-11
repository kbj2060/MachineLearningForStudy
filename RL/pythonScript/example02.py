import numpy as np
import gym
from gym.envs.registration import register
import random as pr
import matplotlib.pyplot as plt

rl = 0.85
dis = 0.99
num_episodes = 10

register(
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery' : False}
)

env = gym.make('FrozenLake-v3')
Q = np.zeros([env.observation_space.n, env.action_space.n])

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n) / ( i + 1 ))
        new_state, reward, done, _ = env.step(action)
        Q[state,action] = (1-rl) * Q[state, action] + rl * (reward + dis * np.max(Q[new_state, :]))

        rAll += reward
        state = new_state

    rList.append(rAll)
print("Sucess rate : " + str(sum(rList) / (num_episodes)))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)


plt.bar(range(len(rList)), rList, color="blue")
plt.show()
