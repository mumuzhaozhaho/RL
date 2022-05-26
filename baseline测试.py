import gym
import numpy as np
from stable_baselines3 import DDPG,HerReplayBuffer
import matplotlib.pyplot as plt
env = gym.make('Tank_env-v1')

model = DDPG.load("tank", env=env)

state  = env.reset()
t = np.linspace(0,1000,1001)
x = np.zeros(1001)
y = np.zeros(1001)
ep_reward = 0
while 1:
    action,_ = model.predict(state)
    next_state, rewards, dones, _ = env.step(action)
    ep_reward += rewards
    state = next_state
    x[env.time] = next_state[0]
    y[env.time] = next_state[1]
    if dones  or (env.time >999):
            break

fig1 = plt.figure(1)
plt.plot(t,x,'b-',label='x(t)')
plt.plot(t,y,'r--',label='y(t)')
plt.legend()
plt.show()