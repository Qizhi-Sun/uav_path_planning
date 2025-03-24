import matplotlib.pyplot as plt
from UAV_Env import *
from DDPG import *
from rl_utils import *
def main():
    actor_lr = 1e-3
    critic_lr = 3e-4
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 300
    batch_size = 64
    sigma = 0.01  # 正态分布噪声幅度
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(42)
    torch.manual_seed(42)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = 10
    action_dim = 3
    UAV_num = 3
    action_bound = 0.2 # 动作最大值
    # test_episode = 50
    # test_time_step = 150
    # 创建环境名
    Map_name = 'Map1'
    # 初始化MAP模块
    MAP = SetConfig(Map_name)
    uav_num, map_w, map_h, map_z, buildings_location, buildings, match_pairs, uav_r, Init_state = MAP.Setting()
    # 初始化控制器
    con = MvController(map_w, map_h, map_z, buildings_location)
    # 初始化Env模块
    env = UAVEnv(uav_num, map_w, map_h, map_z, Init_state)
    # 初始化Agent
    agent = DDPG(state_dim, hidden_dim, action_dim, UAV_num, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device, con)
    # 初始化render
    render = Render(uav_num, env.state, buildings, map_w, map_h, map_z, uav_r, env.position_pool, match_pairs)
    # 开始
    return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, render)
    # agent.save_pth()
    plt.figure(3)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('MADDPG on {}'.format(Map_name))
    plt.show(block = True)
    # agent.load_para()
    # test
    # for i in range(test_episode):
    #     state, _ = env.reset()
    #     for j in range(test_time_step):
    #         action = agent.take_action(state)
    #         next_state, r, done, truncated, _ = env.step(action)
    #         env_t = env.tic_tok()
    #         env.recorder(env_t)
    #         render.render3D()
    #         plt.pause(0.01)
    #         state = next_state
    #

if __name__ == "__main__":
    main()