import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# 定义 DQN 网络架构
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = torch.nn.functional.leaky_relu(self.fc3(x))
        x = torch.nn.functional.leaky_relu(self.fc4(x))
        return self.fc5(x)

# 定义软更新函数
def soft_update(target_net, policy_net, tau):
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

# 训练 DQN 的主要过程
def train_dqn(env, num_episodes=500):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 初始化 DQN 和目标网络
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)  # 调整学习率
    criterion = nn.MSELoss()
    
    replay_buffer = deque(maxlen=50000)  # 增大 Replay Buffer 的容量
    gamma = 0.98  # 调整折扣因子
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 128  # 增加批量大小
    tau = 0.01  # 软更新系数
    
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state, _ = state  # 兼容新版本 Gym，解包 (observation, info)
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        total_reward = 0
        done = False
        
        while not done:
            # 选择动作（ε-贪婪策略）
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = torch.argmax(q_values).item()
            
            # 执行动作
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result
            next_state_tensor = torch.FloatTensor(np.array(next_state)).unsqueeze(0)
            total_reward += reward
            
            # 存储到 Replay Buffer
            replay_buffer.append((state, action, reward, next_state_tensor, done))
            state = next_state_tensor
            
            # 从 Replay Buffer 中抽取样本并更新网络
            if len(replay_buffer) >= batch_size:
                # 从 Replay Buffer 中随机抽取一批样本
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_batch, next_states, dones = zip(*batch)
                
                # 将这些数据转换为张量格式以进行批处理
                states = torch.cat(states)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).unsqueeze(1)
                next_states = torch.cat(next_states)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
                
                # 通过策略网络获得当前状态下的 Q 值
                q_values = policy_net(states).gather(1, actions)
                
                # 使用目标网络计算下一状态的最大 Q 值，得到目标 Q 值
                next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
                target_q_values = rewards_batch + gamma * next_q_values * (1 - dones)
                
                # 计算当前 Q 值和目标 Q 值之间的损失，并通过反向传播来更新策略网络的权重
                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 软更新目标网络
                soft_update(target_net, policy_net, tau)
        
        # 更新 epsilon，确保其不低于最小值
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        rewards_per_episode.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
    
    # 绘制每个 episode 的总回报
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.savefig('rewards_per_episode.png')  # 保存图像到文件
    # plt.show()  # 如果环境支持，可以使用 plt.show()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    train_dqn(env)
