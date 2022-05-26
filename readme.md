[【深度强化学习】最大熵 RL：从Soft Q-Learning到SAC - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/444441890)

[最大熵逆强化学习 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/91819689)

---



## result 存放训练好的模型参数

## pid 控制为pid控制方法结果

### 仿真环境 及超参数	

**仿真环境中奖励函数进行修改，多走一步，奖励变小**

` reward = np.abs(error) - 1`

```python
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
		self.train_eps = 400
		self.test_eps = 10
		self.noise_clip = 0.2  # Range to clip target policy noise
		self.policy_freq = 2 # Frequency of delayed policy updates
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


```
