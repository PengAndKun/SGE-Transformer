import numpy as np
import pickle
import pybullet as p
import time
import os
import sys

# 引入环境
from gym_pybullet_drones.envs.coverage_visibility_pointcloud_aviary_optimized_add_Control import CoverageAviary


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

    print(f"成功加载 {len(trajectories)} 条轨迹。准备开始验证奖励和状态一致性...")

    # 初始化环境 (gui=False 提高速度，若需观察可改为 True)
    env = CoverageAviary(
        gui=False, obstacles=True, ctrl_freq=30, num_rays=120,
        radar_radius=8.0, viz_rays_every=10, viz_points_every=20, grid_res=0.5
    )

    action_space = MacroActionSpace27(move_distance=0.5)

    # 定义高度限制
    MIN_HEIGHT = 0.1
    MAX_HEIGHT = 4

    env.reset()
    start_snapshot = env.get_snapshot()

    for t_idx, traj in enumerate(trajectories):
        print(f"\n" + "=" * 50)
        print(f"回放轨迹 {t_idx + 1}/{len(trajectories)}")
        print(f"预期总分: {traj['final_reward']:.2f} | 步数: {len(traj['action_sequence'])}")
        print("=" * 50)

        actions = traj['action_sequence']
        ref_states = traj['state_sequence']  # 预期状态序列 (N+1, 6)
        ref_rewards = traj.get('reward_sequence', [])  # 预期单步奖励序列 (N,)

        env.restore_snapshot(start_snapshot)

        is_consistent = True
        accumulated_reward = 0.0

        for step, action_id in enumerate(actions):
            start_pos = np.array(env.pos[0])

            # 1. 物理逻辑：位移计算
            displacement = action_space.get_displacement(action_id)
            target_pos = start_pos + displacement

            # 检查高度越界
            if target_pos[2] < MIN_HEIGHT or target_pos[2] > MAX_HEIGHT:
                print(f"  [步 {step}] ❌ 高度越界: {target_pos[2]:.2f}")
                is_consistent = False
                break

            # 模拟执行：瞬移并计算奖励
            p.resetBasePositionAndOrientation(env.DRONE_IDS[0], target_pos, [0, 0, 0, 1], physicsClientId=env.CLIENT)
            _, step_reward, terminated, _, _ = env.compute_scan_at_pos(target_pos)
            accumulated_reward += step_reward

            # 2. 状态比对 (State Check)
            expected_state = ref_states[step + 1]
            pos_error = np.linalg.norm(target_pos - expected_state[:3])

            # 3. 奖励比对 (Reward Check)
            # 如果存档中有单步奖励，进行对比
            reward_error = 0.0
            if len(ref_rewards) > 0:
                expected_step_reward = ref_rewards[step]
                reward_error = abs(step_reward - expected_step_reward)

            # 判定一致性阈值
            if pos_error > 1e-4 or reward_error > 1e-4:
                print(f"  [步 {step}] ⚠️ 不一致检测:")
                if pos_error > 1e-4:
                    print(f"    - 位置偏差: {pos_error:.6f} (实际: {target_pos}, 预期: {expected_state[:3]})")
                if reward_error > 1e-4:
                    print(
                        f"    - 奖励偏差: {reward_error:.6f} (实际: {step_reward:.4f}, 预期: {expected_step_reward:.4f})")
                is_consistent = False

            if terminated:
                print(f"  [步 {step}] 碰撞终止。")
                break

        # 轨迹总结
        reward_final_diff = abs(accumulated_reward - traj['final_reward'])
        print(f"\n--- 轨迹 {t_idx + 1} 验证总结 ---")
        print(f"实际累计得分: {accumulated_reward:.4f}")
        print(f"存档最终得分: {traj['final_reward']:.4f}")
        print(f"得分总偏差: {reward_final_diff:.6f}")

        if is_consistent and reward_final_diff < 1e-3:
            print("✅ 验证结果：完全一致 (Perfect Match)")
        elif reward_final_diff < 5.0:
            print("⚠️ 验证结果：基本一致 (小额浮动偏差)")
        else:
            print("❌ 验证结果：不一致 (Consistency Failed)")

    env.close()


if __name__ == "__main__":
    # 确保文件名与你保存的一致
    SAVE_FILE = "elite_trajectories_v1_top.pkl"
    validate_trajectories(SAVE_FILE)