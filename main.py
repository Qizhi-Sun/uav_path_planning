import matplotlib.pyplot as plt
from UAV_Env import *
from  PPO import *
from rl_utils import *
def main():
    lamda = 0.95
    gamma = 0.98
    epochs = 10
    eps = 0.2
    device = torch.device("cuda")
    action_bound = 0.7
    actor_lr = 1e-3
    critic_lr = 1e-3
    hidden_num = 128
    torch.manual_seed(0)
    state_dim = 6
    action_dim = 3

    Map_name = 'Map1'
    env_t = 0
    # 初始化MAP模块
    MAP = SetConfig(Map_name)
    uav_num, map_w, map_h, map_z, buildings_location, buildings, match_pairs, uav_r, Init_state = MAP.Setting()
    # 初始化Env模块
    env = UAVEnv(uav_num, map_w, map_h, map_z, Init_state)
    # 初始化render模块
    render = Render(uav_num, env.state, buildings, map_w, map_h, map_z, uav_r, env.position_pool, match_pairs)
    # 初始化MVController模块
    mvcontroller = MvController(map_w, map_h, map_z, buildings_location)
    # 初始化Agent
    agent = PPO(lamda, gamma, state_dim, action_dim, hidden_num, eps, actor_lr, critic_lr, epochs, device, action_bound)
    # 开始
    return_list = rl_utils.train_on_policy_agent(env, agent, 500)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(Map_name))
    plt.show()


if __name__ == "__main__":
    main()