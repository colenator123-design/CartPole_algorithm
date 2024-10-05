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

class REINFORCE:
    def __init__(self, state_size, action_size, lr=0.001):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = 0.99
        self.log_probs = []
        self.rewards = []

    def store_outcome(self, log_prob, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def compute_returns(self):
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def update_policy(self):
        if not self.rewards or not self.log_probs:
            print("No rewards or log probabilities collected!")
            return

        returns = self.compute_returns()
        returns = torch.tensor(returns)
        if returns.std() != 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        if policy_loss:
            self.optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
        else:
            print("Policy loss is empty!")

        self.log_probs = []
        self.rewards = []

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = REINFORCE(state_size, action_size, lr=0.001)

episode_durations = []

for epoch in range(500):
    print(f"\nStarting epoch {epoch}")

    state = env.reset()
    episode_duration = 0

    while True:
        state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0)

        action_probs = agent.policy_network(state)
        action = np.random.choice(action_size, p=action_probs.detach().numpy()[0])
        log_prob = torch.log(action_probs.squeeze(0)[action])

        result = env.step(action)
        if len(result) == 4:
            next_state, reward, done, _ = result
        else:
            next_state, reward, done, *_ = result

        agent.store_outcome(log_prob, reward)

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

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed_durations = moving_average(episode_durations, window_size=10)

plt.plot(smoothed_durations)
plt.title('Smoothed Episode Duration Over Time')
plt.xlabel('Training Epoch')
plt.ylabel('Episode Duration')
plt.show()

plt.savefig('smoothed_episode_duration_plot.png')
print("Plot saved as 'smoothed_episode_duration_plot.png'")
