import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x


class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 1)  # 值函數輸出是標量

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class REINFORCEBaseline:
    def __init__(self, state_size, action_size, lr=0.001):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.value_network = ValueNetwork(state_size)  # 引入值網絡作為基線
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        self.gamma = 0.99
        self.log_probs = []
        self.rewards = []
        self.states = []

    def store_outcome(self, log_prob, reward, state):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.states.append(state)

    def compute_returns(self):
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def update_policy(self):
        if not self.rewards or not self.log_probs:
            return

        returns = self.compute_returns()
        returns = torch.tensor(returns)
        states = torch.stack(self.states)
        values = self.value_network(states).squeeze()

        advantages = returns - values.detach()  # 基線是值網絡的輸出
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage)

        # 優化策略網絡
        self.policy_optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 優化值網絡
        value_loss = nn.MSELoss()(values, returns)  # 使用均方誤差更新值網絡
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

      
        self.log_probs = []
        self.rewards = []
        self.states = []


env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


agent = REINFORCEBaseline(state_size, action_size, lr=0.001)


episode_durations = []

# 訓練迴圈
for epoch in range(500):
    print(f"\nStarting epoch {epoch}")

    state = env.reset()
    episode_duration = 0

    while True:
        state = torch.FloatTensor(state).unsqueeze(0)

        action_probs = agent.policy_network(state)
        action = np.random.choice(action_size, p=action_probs.detach().numpy()[0])
        log_prob = torch.log(action_probs.squeeze(0)[action])

        result = env.step(action)
        if len(result) == 4:
            next_state, reward, done, _ = result
        else:
            next_state, reward, done, *_ = result

        agent.store_outcome(log_prob, reward, state)

        episode_duration += 1
        state = next_state

        if done:
            print(f"Epoch finished. Episode duration: {episode_duration}")
            episode_durations.append(episode_duration)
            agent.update_policy()
            break

   
    if (epoch + 1) % 10 == 0:
        average_duration = np.mean(episode_durations[-10:])
        print(f"Average duration over last 10 episodes: {average_duration}")

# 計算滑動平均
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 使用滑動平均平滑 episode_durations
smoothed_durations = moving_average(episode_durations, window_size=10)

plt.plot(smoothed_durations)
plt.title('Smoothed Episode Duration Over Time')
plt.xlabel('Training Epoch')
plt.ylabel('Episode Duration')
plt.show()

# 保存圖像到文件
plt.savefig('smoothed_episode_duration_plot.png')
print("Plot saved as 'smoothed_episode_duration_plot.png'")
