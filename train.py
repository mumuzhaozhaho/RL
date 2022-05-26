# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2022/05/23 10:53:13
@Author  :   mumuzhaozhao 
'''
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import gym
from torch.utils.tensorboard import SummaryWriter   
from agent import TD3
import os

class TD3Config:
	def __init__(self) -> None:
		self.algo = 'TD3'

		self.env = 'Tank_env-v1'
		self.start_timestep = 10e3 # Time steps initial random policy is used
		self.max_timestep = 400000 # Max time steps to run environment
		self.expl_noise = 0.1 # Std of Gaussian exploration noise
		self.batch_size = 256 # Batch size for both actor and critic
		self.gamma = 0.99 # gamma factor
		self.lr = 0.0005 # Target network update rate 
		self.policy_noise = 0.2 # Noise added to target policy during critic update
		self.train_eps = 600
		self.test_eps = 10
		self.noise_clip = 0.2  # Range to clip target policy noise
		self.policy_freq = 2 # Frequency of delayed policy updates
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境：{cfg.env}，算法：{cfg.algo}，设备：{cfg.device}')
    step  = 0 
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        t = np.linspace(0,1000,1001)
        state = env.reset()
        x = np.zeros(1001)
        y = np.zeros(1001)
        ep_reward = 0
        while True:
            step += 1
            if step < cfg.start_timestep:
                action = env.action_space.sample()
            else:
                action = (agent.choose_action(np.array(state))+ np.random.normal(0, max_action * cfg.expl_noise, size=action_dim)
			).clip(0.0,1.0)
            next_state, reward, done,_ = env.step(action)
            agent.memory.push(state, action, next_state,reward,done)
            x[env.time] = next_state[0]
            y[env.time] = next_state[1]
            ep_reward += reward
            state  = next_state
            if step >= cfg.start_timestep:
                agent.update()
            if done  or (env.time >999):
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1) % 10  == 0:
            print('回合：{}/{}，持续时间{}s,  奖励为：{:.2f}'.format(i_ep+1, cfg.train_eps, env.time,ep_reward))
        if (i_ep+1) % 20 == 0:
            fig1 = plt.figure(1)
            plt.plot(t,x,'b-',label='x(t)')
            plt.plot(t,y,'r--',label='y(t)')
            # plt.plot(t,env.operation,label='a(t)')
            plt.legend()
            plt.draw()
            plt.pause(2)
            plt.close(fig1)
    plt.plot(rewards,label = '奖励')
    plt.plot(ma_rewards,label = '滑动奖励')
    plt.legend()
    plt.savefig('Figure_1.png')
    plt.show()
    print('完成训练！')
    return rewards, ma_rewards

def test(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env}，算法：{cfg.algo}，设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        t = np.linspace(0,1000,1001)
        state = env.reset()
        x = np.zeros(1001)
        y = np.zeros(1001)
        ep_reward = 0
        while True:
            action = agent.choose_action(np.array(state))
            next_state, reward, done,_ = env.step(action)
            x[env.time] = next_state[0]
            y[env.time] = next_state[1]
            ep_reward += reward
            state  = next_state
            if done  or (env.time >999):
                break
        rewards.append(ep_reward)
        fig1 = plt.figure(1)
        plt.plot(t,x,'b-',label='x(t)')
        plt.plot(t,y,'r--',label='y(t)')
        plt.plot(t,env.operation,label='a(t)')
        plt.legend()
        plt.draw()
        plt.pause(2)
        plt.close(fig1)
        print('回合：{}/{}, 奖励：{}'.format(i_ep+1, cfg.test_eps, ep_reward))
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
       
    print('完成测试！') 
    return rewards, ma_rewards,


if __name__ == "__main__":
    path0 = os.path.dirname(__file__) + '/'
    path1 = path0 + 'result/'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family']='Simhei' #修改全局字体
    cfg = TD3Config()
    env = gym.make(cfg.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action= float(env.action_space.high[0])
    agent = TD3(state_dim,action_dim,max_action,cfg)
    '''
    训练
    '''
    agent.load(path1)
    agent.actor_target.load_state_dict(torch.load(path1 + "td3_actor"))
    agent.critic_target.load_state_dict(torch.load(path1 + "td3_critic"))
    rewards, ma_rewards = train(cfg, env, agent)
    agent.save(path1)
    print('')
    '''
    测试
    '''
    agent.load(path1)
    rewards, ma_rewards = test(cfg, env, agent)
    print('')



