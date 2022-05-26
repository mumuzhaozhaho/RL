import sys,os
import torch
import gym
import numpy as np
import datetime
from agent import TD3

curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
	

class TD3Config:
	def __init__(self) -> None:
		self.algo = 'TD3'
		self.env = 'Tank_env-v1'
		self.start_timestep = 25e3 # Time steps initial random policy is used

		self.max_timestep = 400000 # Max time steps to run environment
		self.expl_noise = 0.1 # Std of Gaussian exploration noise
		self.batch_size = 256 # Batch size for both actor and critic
		self.gamma = 0.99 # gamma factor
		self.lr = 0.0005 # Target network update rate 
		self.policy_noise = 0.2 # Noise added to target policy during critic update
		self.noise_clip = 0.2  # Range to clip target policy noise
		self.policy_freq = 2 # Frequency of delayed policy updates
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg,env,agent):
	# Evaluate untrained policy
	state = env.reset()
	ep_reward = 0
	ep_timesteps = 0
	episode_num = 0
	rewards = []
	ma_rewards = [] # moveing average reward
	for t in range(int(cfg.max_timestep)):
		ep_timesteps += 1
		# Select action randomly or according to policy
		if t < cfg.start_timestep:
			action = env.action_space.sample()
		else:
			action = (
				agent.choose_action(np.array(state))
				+ np.random.normal(0, max_action * cfg.expl_noise, size=action_dim)
			).clip(0.0,1.0)
		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if ep_timesteps < env._max_episode_steps else 0
		# Store data in replay buffer
		agent.memory.push(state, action, next_state, reward, done_bool)
		state = next_state
		ep_reward += reward
		# Train agent after collecting sufficient data
		if t >= cfg.start_timestep:
			agent.update()
		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Episode:{episode_num+1}, Episode T:{ep_timesteps}, Reward:{ep_reward:.3f}")
			# Reset environment			
			rewards.append(ep_reward)
			# 计算滑动窗口的reward
			if ma_rewards:
				ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
			else:
				ma_rewards.append(ep_reward) 
			state = env.reset()
			ep_reward = 0
			ep_timesteps = 0
			episode_num += 1 

	return rewards, ma_rewards


if __name__ == "__main__":
	cfg  = TD3Config()
	env = gym.make(cfg.env)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	agent = TD3(state_dim,action_dim,max_action,cfg)
	rewards,ma_rewards = train(cfg,env,agent)
	print('训练结束')

	
		
