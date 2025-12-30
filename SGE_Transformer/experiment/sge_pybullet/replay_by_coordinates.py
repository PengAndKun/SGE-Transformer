import numpy as np
import time
import pickle
import pybullet as p
import os
import sys

# 引入依赖
from SGE_Transformer.envs.coverage_lidar_aviary import CoverageAviary

# 动作 ID 映射 (用于在屏幕上显示当前动作)
ACTION_MAP = {
    0: "Hover", 1: "Fwd", 2: "Back", 3: "Left", 4: "Right",
    5: "Up", 6: "Down", 7: "Turn L", 8: "Turn R"
}

# ==========================================
# 配置
# ==========================================
BATCHES = 10
BUDGET = 100000


def replay_ground_truth(run_id, traj_data):
    # 1. 解析数据
    states = traj_data['states']  # [x, y, z, r, p, y] 序列
    actions = traj_data['actions']  # 动作序列
    final_reward = traj_data['final_reward']

    # 恢复环境种子 (为了看障碍物是不是对得上)
    seed = BUDGET + (BATCHES * 1000) + (run_id - 1)

    print(f"\n=== Replaying Run {run_id} (Ground Truth Mode) ===")
    print(f"Seed: {seed}")
    print(f"Steps: {len(states)}")
    print("Mode: Teleportation based on saved coordinates (No Physics Calc)")

    # 2. 初始化环境
    # 注意：我们这里不需要 Controller，因为我们不计算物理，只负责“摆放”无人机
    env = CoverageAviary(gui=True, obstacles=True, user_debug_gui=True)
    env.reset(seed=seed)

    # 设置相机视角
    p.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

    # 3. 开始“放电影”
    last_pos = states[0][:3]

    # 先把无人机放到起点
    start_rpy = states[0][3:]
    start_quat = p.getQuaternionFromEuler(start_rpy)
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], last_pos, start_quat)

    # 暂停一下让你准备看
    time.sleep(1)

    for i in range(1, len(states)):
        # 获取第 i 步的状态
        current_state = states[i]
        pos = current_state[:3]
        rpy = current_state[3:]

        # 获取导致来到这个状态的动作 (索引是 i-1)
        # 因为 state[0] 是初始状态，state[1] 是执行 action[0] 后的状态
        act_id = actions[i - 1] if i - 1 < len(actions) else -1
        act_name = ACTION_MAP.get(act_id, "End")

        # === [核心] 强制设置无人机位置 ===
        # 这就是“基于坐标回放”，完全不依赖物理引擎计算
        quat = p.getQuaternionFromEuler(rpy)
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0], pos, quat)

        # 视觉效果：画轨迹线
        p.addUserDebugLine(last_pos, pos, [0, 1, 0], lineWidth=3.0, lifeTime=0)  # 绿色线
        last_pos = pos

        # 在头顶显示当前动作文字
        p.addUserDebugText(f"{act_name}", [pos[0], pos[1], pos[2] + 0.5], [0, 0, 1], lifeTime=0.1)

        # 控制播放速度 (0.05秒一帧)
        time.sleep(0.05)

        # 打印日志
        if i % 10 == 0:
            print(f"Step {i}/{len(states)} | Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | Action: {act_name}")

    print(f"\nReplay Finished.")
    print(f"Saved Reward: {final_reward:.2f}")

    input("Press Enter to close window...")
    env.close()


def main():
    # 确保读取的是修复后的文件
    filename = "../../data_pybullet/trajectories/sge_full_state_trajectories_fixed.pkl"

    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        print("请先运行修复版的提取脚本 extract_full_trajectory_fixed.py")
        return

    print(f"Loading {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    run_ids = sorted(list(data.keys()))
    print(f"Available Runs: {run_ids}")

    while True:
        user_input = input("\nEnter Run ID to view (or 'q'): ")
        if user_input.lower() == 'q': break

        try:
            run_id = int(user_input)
            if run_id in data:
                # 默认播放第一条（最优）轨迹
                replay_ground_truth(run_id, data[run_id][0])
            else:
                print("Invalid Run ID")
        except ValueError:
            print("Invalid input")


if __name__ == "__main__":
    main()