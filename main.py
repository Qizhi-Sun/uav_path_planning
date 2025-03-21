import matplotlib.pyplot as plt
from UAV_Env import *
from DDPG import *
from rl_utils import *
def main():
    actor_lr = 3e-4
    critic_lr = 3e-3
    num_episodes = 200
    hidden_dim = 64
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.01  # 高斯噪声标准差
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = 6
    action_dim = 3
    action_bound = 0.35 # 动作最大值

    # 创建环境名
    Map_name = 'Map1'
    env_t = 0
    # 初始化MAP模块
    MAP = SetConfig(Map_name)
    uav_num, map_w, map_h, map_z, buildings_location, buildings, match_pairs, uav_r, Init_state = MAP.Setting()
    # 初始化Env模块
    env = UAVEnv(uav_num, map_w, map_h, map_z, Init_state)
    # 初始化render模块
    render = Render(uav_num, env.state, buildings, map_w, map_h, map_z, uav_r, env.position_pool, match_pairs)
    # 初始化Agent
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
    # 开始
    return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(Map_name))
    plt.show()


if __name__ == "__main__":
    main()