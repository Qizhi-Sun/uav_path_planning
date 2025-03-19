import matplotlib.pyplot as plt
from myenv import *

def main():
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
    # 开始
    actions = [[0, 0, 0, 0] for _ in range(uav_num)]
    flag = [False] * uav_num
    done = False
    while not done:
        for pair in match_pairs:
            index = pair[0]
            uav_state = env.state[index][:3]
            aim = pair[2]
            vx, vy, vz = mvcontroller.Move_to(uav_state, aim)
            if mvcontroller.Is_arrive(uav_state, aim):
                if not flag[index]:
                    flag[index] = True
                    point = render.AimsPoint[index].pop(0)
                    point.remove()
                    # render.ax.scatter(uav_state[0], uav_state[1], uav_state[2], color='red', s=50)
            if mvcontroller.Is_outside_map(uav_state, [vx, vy, vz]):
                vx, vy, vz = 0, 0, 0
            if mvcontroller.Will_enter_buildings(uav_state, [vx, vy, vz], uav_r):
                vx, vy, vz = mvcontroller.Move_up()
            actions[index] = [vx, vy, vz, 0]
        obs, reward, done, truncated,info = env.step(actions, env_t)
        env.recorder(env_t)
        render.render3D()
        plt.pause(0.01)
        env_t += 1
        if done:
            env.reset()


if __name__ == "__main__":
    main()