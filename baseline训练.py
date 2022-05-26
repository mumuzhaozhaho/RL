import gym
import numpy as np
import os 
import tensorboardX
from stable_baselines3 import TD3,HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise

env = gym.make('Tank_env-v1')
path = os.path.dirname(__file__)
log_dir = './tank_tensorboard/'
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(policy="MlpPolicy", env=env, buffer_size=100000,verbose=1,tensorboard_log=log_dir)
model.learn(total_timesteps=500000)
model.save("tank")