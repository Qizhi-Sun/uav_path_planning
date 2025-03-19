import torch.nn as nn
import torch
import torch.nn.functional as F
import rl_utils
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch.optim as opt


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class PPO:
    def __init__(self, lamda, gamma, state_dim, action_dim, hidden_dim, eps, actor_lr, critic_lr, epochs, device):
        self.lamda = lamda
        self.gamma = gamma
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.actor = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_opt = opt.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = opt.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(device=self.device)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)
        td_error = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lamda, td_error.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs-old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(td_target.detach(), self.critic(states)))
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()


if __name__ == "__main__":
    lamda = 0.95
    gamma = 0.98
    epochs = 10
    eps = 0.2
    device = torch.device("cuda")
    actor_lr = 1e-3
    critic_lr = 1e-3
    hidden_num = 128
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(lamda, gamma, state_dim, action_dim, hidden_num, eps, actor_lr, critic_lr, epochs, device)
    return_list = rl_utils.train_on_policy_agent(env, agent, 500)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

