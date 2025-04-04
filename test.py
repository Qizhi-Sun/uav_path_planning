import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
import numpy as np
import math

x_goal = 5
y_goal = 5
class GoLeftEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.state = np.array([0,0,0,0,x_goal,y_goal])
        self.action_space = gym.spaces.Box(low=np.array([-0.2, -0.2]), high=np.array([0.2, 0.2]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([0,0,-1,-1,0,0]), high=np.array([10,10,1,1,10,10]), dtype=np.float32)

    def reset(self, seed = None):
        self.state = np.array([0,0,0,0,x_goal,y_goal], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        last_speed = self.state[2:4].copy()
        print(f"before{last_speed}")
        self.state[2:4] += action[:2] # 更新速度 vx vy
        self.state[2:4] = np.clip(self.state[2:4], -1, 1)
        print(f"after{last_speed}")
        self.state[:2] += (last_speed + self.state[2:4]) / 2
        # r_edge
        if not(0<=self.state[0]<=10 and 0<=self.state[1]<=10):
            r_edge = -1
            self.state[:2] = np.clip(self.state[:2], [0,0], [10,10])
        else:
            r_edge = 0
        last_distance = math.hypot(self.state[4], self.state[5])
        x_diff = abs(self.state[0] - x_goal)
        y_diff = abs(self.state[1] - y_goal)
        distance_to_goal = math.hypot( x_diff, y_diff)
        self.state[4] = x_diff
        self.state[5] = y_diff
        r1 = last_distance - distance_to_goal + r_edge
        done = distance_to_goal <= 3
        reward = 5.0 if done else r1
        truncated = 0
        return np.array(self.state, dtype=np.float32), float(reward), bool(done), bool(truncated), {}

    def render(self, mode='console'):
        print(self.state)



env = GoLeftEnv()
state, _ = env.reset(0)
action = np.array([0.02,0.04])
s_, r, d, t, _ = env.step(action)
s_1, r1, d1, t1, _ = env.step(action)
s_2, r2, d2, t2, _ = env.step(action)
s_3, r3, d3, t3, _ = env.step(action)
