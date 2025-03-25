import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from matplotlib.font_manager import weight_dict
from UAV_Env import *
import rl_utils
import matplotlib.pyplot as plt
import numpy as np
import os

save_path = 'E:\RL\graduate_design\checkpoints'

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.fc3.weight.data.normal_(0,0.1)
        self.action_bound = action_bound

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        return torch.tanh(x) * self.action_bound



class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, UAV_num):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim * UAV_num + action_dim, hidden_dim)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.fc3.weight.data.normal_(0,0.1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        return x


class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, UAV_num, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device, con):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim, UAV_num).to(device)
        self.actor_target = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_target = QValueNet(state_dim, hidden_dim, action_dim, UAV_num).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.device = device
        self.action_dim = action_dim

        self.actor_1 = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim, UAV_num).to(device)
        self.actor_target_1 = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_target_1 = QValueNet(state_dim, hidden_dim, action_dim, UAV_num).to(device)
        self.actor_target_1.load_state_dict(self.actor_1.state_dict())
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.actor_opt_1 = torch.optim.Adam(self.actor_1.parameters(), lr=actor_lr)
        self.critic_opt_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)

        self.actor_2 = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim, UAV_num).to(device)
        self.actor_target_2 = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_target_2 = QValueNet(state_dim, hidden_dim, action_dim, UAV_num).to(device)
        self.actor_target_2.load_state_dict(self.actor_2.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.actor_opt_2 = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt_2 = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.control = con
        self.counter = 0

    def take_action(self, state):
        state_1 = torch.tensor(state[0], dtype=torch.float).to(self.device)
        state_2 = torch.tensor(state[1], dtype=torch.float).to(self.device)
        state_3 = torch.tensor(state[2], dtype=torch.float).to(self.device)
        action1 = self.actor(state_1).detach().cpu().numpy()
        action2 = self.actor_1(state_2).detach().cpu().numpy()
        action3 = self.actor_2(state_3).detach().cpu().numpy()
        action1 = action1 + self.sigma * np.random.randn(self.action_dim)
        action2 = action2 + self.sigma * np.random.randn(self.action_dim)
        action3 = action3 + self.sigma * np.random.randn(self.action_dim)
        action = [action1, action2, action3]
        self.counter += 1
        return action

    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        next_actions = self.actor_target(next_states[:,0,:])
        next_q_value = self.critic_target(next_states.reshape(64, 30), next_actions).detach()
        q_targets = rewards[:,0].view(64,1) + self.gamma * next_q_value * (1 - dones)
        q_value = self.critic(states.reshape(64, 30), actions[:,0,:])
        critic_loss = torch.mean(F.mse_loss(q_targets, q_value))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        for name, param in self.critic.named_parameters():
            if param.grad is not None:
                print(f"critic Layer: {name}, Gradient Norm: {param.grad.norm()}")
            else:
                print(f"{name} has no gradient!")
        actor_loss = -torch.mean(self.critic(states.reshape(64,30), self.actor(states[:,0,:])))
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                print(f"actor Layer: {name}, Gradient Norm: {param.grad.norm()}")
            else:
                print(f"{name} has no gradient!")
        self.soft_update(self.actor, self.actor_target)  # 软更新策略网络
        self.soft_update(self.critic, self.critic_target)  # 软更新价值网络

        # next_actions_1 = self.actor_target_1(next_states[:,1,:])
        # next_q_value_1 = self.critic_target_1(next_states.reshape(64, 30), next_actions_1)
        # q_targets_1 = rewards[:,1].view(64,1) + self.gamma * next_q_value_1 * (1 - dones)
        # q_value_1 = self.critic_1(states.reshape(64, 30), actions[:,1,:])
        # critic_loss_1 = torch.mean(F.mse_loss(q_targets_1.detach(), q_value_1))
        # self.critic_opt_1.zero_grad()
        # critic_loss_1.backward()
        # self.critic_opt_1.step()
        # actor_loss_1 = -torch.mean(self.critic_1(states.reshape(64, 30), self.actor_1(states[:,1,:])))
        # self.actor_opt_1.zero_grad()
        # actor_loss_1.backward()
        # self.actor_opt_1.step()
        # self.soft_update(self.actor_1, self.actor_target_1)  # 软更新策略网络
        # self.soft_update(self.critic_1, self.critic_target_1)  # 软更新价值网络
        # next_actions_2 = self.actor_target_2(next_states[:,2,:])
        # next_q_value_2 = self.critic_target_2(next_states.reshape(64, 30), next_actions_2)
        # q_targets_2 = rewards[:,2].view(64,1) + self.gamma * next_q_value_2 * (1 - dones)
        # q_value_2 = self.critic_2(states.reshape(64, 30), actions[:,2,:])
        # critic_loss_2 = torch.mean(F.mse_loss(q_targets_2.detach(), q_value_2))
        # self.critic_opt_2.zero_grad()
        # critic_loss_2.backward()
        # self.critic_opt_2.step()
        # actor_loss_2 = -torch.mean(self.critic_2(states.reshape(64, 30), self.actor_2(states[:,2,:])))
        # self.actor_opt_2.zero_grad()
        # actor_loss_2.backward()
        # self.actor_opt_2.step()
        # self.soft_update(self.actor_2, self.actor_target_2)  # 软更新策略网络
        # self.soft_update(self.critic_2, self.critic_target_2)  # 软更新价值网络
    def save_pth(self):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(self.actor.state_dict(), os.path.join(save_path, "ddpg_uav0_actor.pth"))
        torch.save(self.actor_1.state_dict(), os.path.join(save_path, "ddpg_uav1_actor.pth"))
        torch.save(self.actor_2.state_dict(), os.path.join(save_path, "ddpg_uav2_actor.pth"))

    def load_para(self):
        self.actor.load_state_dict(torch.load(os.path.join(save_path, "ddpg_uav0_actor.pth"), weights_only=True))
        self.actor_1.load_state_dict(torch.load(os.path.join(save_path, "ddpg_uav1_actor.pth"), weights_only=True))
        self.actor_2.load_state_dict(torch.load(os.path.join(save_path, "ddpg_uav2_actor.pth"), weights_only=True))