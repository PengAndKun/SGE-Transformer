import numpy as np
import pickle
import pybullet as p
import time
import os
import sys

# 引入环境和宏动作空间（确保路径正确）
from ISE_Transformer.envs.coveragel_multi_room import CoverageAviary


# 这里假设你的 MacroActionSpace27 定义在原脚本中，需要保持一致
# 如果原脚本名为 sge_main.py，可以用 from sge_main import MacroActionSpace27, execute_translation_step

# ==========================================
# 重新定义或导入必要的函数（必须与生成脚本一致）
# ==========================================
class MacroActionSpace27:
    def __init__(self, move_distance=0.5):
        self.num_actions = 27
        self.move_distance = move_distance
        self.deltas = []
        for a in range(self.num_actions):
            dx = (a // 9) - 1
            dy = ((a % 9) // 3) - 1
            dz = (a % 3) - 1
            vec = np.array([dx, dy, dz], dtype=float)
            norm = np.linalg.norm(vec)
            if norm > 0:
                unit_vec = (vec / norm) * self.move_distance
            else:
                unit_vec = np.zeros(3)
            self.deltas.append(unit_vec)

    def get_displacement(self, action_id: int):
        return self.deltas[int(action_id)]


def validate_trajectories(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    # 加载数据
    with open(file_path, 'rb') as f:
        trajectories = pickle.load(f)

    print(f"成功加载 {len(trajectories)} 条轨迹。准备开始验证...")

    # 初始化环境 (开启 GUI 以便观察)
    env = CoverageAviary(
        gui=False, obstacles=True, ctrl_freq=30, num_rays=120,
        radar_radius=8.0, viz_rays_every=10, viz_points_every=20, grid_res=0.5
    )



    # 这里的 move_distance 必须和生成数据时完全一致 (0.5)
    action_space = MacroActionSpace27(move_distance=0.5)

    # 定义高度限制
    MIN_HEIGHT = 0.1
    MAX_HEIGHT = 4
    env.reset()
    start_snapshot = env.get_snapshot()
    for t_idx, traj in enumerate(trajectories):
        print(f"\n--- 正在回放轨迹 {t_idx + 1}/{len(trajectories)} | 期望得分: {traj['final_reward']:.2f} ---")

        # 重置环境

        actions = traj['action_sequence']
        ref_states = traj['state_sequence']  # 记录的状态序列 (N+1, 6)
        env.restore_snapshot(start_snapshot)

        # 记录验证结果
        is_consistent = True
        accumulated_reward = 0.0

        # 执行动作序列
        for step, action_id in enumerate(actions):
            # 获取当前位置（执行前）
            start_pos = np.array(env.pos[0])

            # 1. 执行平移逻辑 (简化版，仅用于回放验证)
            displacement = action_space.get_displacement(action_id)
            target_pos = start_pos + displacement

            # 检查高度合法性（模拟生成时的拦截逻辑）
            if target_pos[2] < MIN_HEIGHT or target_pos[2] > MAX_HEIGHT:
                print(f"  [步骤 {step}] 警告: 动作导致高度越界 ({target_pos[2]:.2f})")
                is_consistent = False
                break

            # 瞬移到目标位置
            p.resetBasePositionAndOrientation(env.DRONE_IDS[0], target_pos, [0, 0, 0, 1], physicsClientId=env.CLIENT)

            # 计算得分
            _, reward, terminated, _, _ = env.compute_scan_at_pos(target_pos)
            accumulated_reward += reward

            # 2. 状态比对
            current_actual_state = np.concatenate([env.pos[0], env.rpy[0]])
            # ref_states[step+1] 是动作执行后的预期状态
            expected_state = ref_states[step + 1]

            # 计算位置偏差
            pos_error = np.linalg.norm(current_actual_state[:3] - expected_state[:3])

            if pos_error > 1e-4:
                print(f"  [步骤 {step}] 状态不一致! 偏差: {pos_error:.6f} action is {action_id} actions {actions[:step]}")
                print(f"   start_pos : {start_pos}")
                print(f"    实际: {current_actual_state[:3]}")
                print(f"    预期: {expected_state[:3]}")
                is_consistent = False

            if terminated:
                print(f"  [步骤 {step}] 触发碰撞终止。")
                break

            # 适当减速以便肉眼观察
            time.sleep(0.05)

        # 最终总结
        print(f"回放完成。实际总得分: {accumulated_reward:.2f} | 期望总得分: {traj['final_reward']:.2f}")
        if is_consistent and np.abs(accumulated_reward - traj['final_reward']) < 10:
            print("✅ 轨迹一致性验证通过！")
        else:
            print("❌ 轨迹验证失败：状态或得分不匹配。")

        time.sleep(1)

    env.close()


if __name__ == "__main__":
    SAVE_FILE = "elite_trajectories_v1_top_2.pkl"
    validate_trajectories(SAVE_FILE)