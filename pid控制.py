import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from simple_pid import PID 
import gym
if __name__ == '__main__':
    #定义pid
    
    Kp = 0.78# Controller gain
    Ki = 0.0029  # Controller integral parameter
    Kd = 25  # Controller derivative parameter
    pid = PID(Kp,Ki,Kd)
    oplo = 0.0
    ophi = 1.0
    pid.output_limits = (oplo,ophi)
    pid.sample_time = 0.01


    t = np.linspace(0,1000,1001)
    env = gym.make('Tank_env-v1')
    state = env.reset()
    x = np.zeros(1001)
    y = np.zeros(1001)
    ep_reward = 0
    while 1:
        pid.setpoint = 1.0
        a = pid(state[1],dt=1)
        next_state, reward, done ,_ = env.step(a)
        x[env.time] = next_state[0]
        y[env.time] = next_state[1]
        state  = next_state
        ep_reward += reward
        if env.time >999:
            break
    print('奖励：{}'.format(ep_reward))
    fig1 = plt.figure(1)
    plt.plot(t,x,'b-',label='x(t)')
    plt.plot(t,y,'r--',label='y(t)')
    plt.legend()
    plt.show()
    print('结束')
