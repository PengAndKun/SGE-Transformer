import numpy as np
import time
import random
import pybullet as p
import heapq
import os
import pickle
import sys
from dataclasses import dataclass
from typing import List, Dict

# 引入依赖
from SGE_Transformer.envs.coverage_lidar_aviary import CoverageAviary
from stable_controller import ROS2VelocityController


# 屏蔽 PyBullet 输出
class suppress_output:
    def __init__(self):
        self._stdout, self._stderr = None, None

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        devnull = open(os.devnull, 'w')
        sys.stdout, sys.stderr = devnull, devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout, sys.stderr = self._stdout, self._stderr


# ==========================================
# 基础类定义 (保持不变)
# ==========================================
class DroneActionSpace:
    def __init__(self):
        self.V_XY = 0.5;
        self.V_Z = 0.3;
        self.W_Z = 0.5
        self.actions = {
            0: np.array([0.0, 0.0, 0.0, 0.0]), 1: np.array([self.V_XY, 0.0, 0.0, 0.0]),
            2: np.array([-self.V_XY, 0.0, 0.0, 0.0]), 3: np.array([0.0, self.V_XY, 0.0, 0.0]),
            4: np.array([0.0, -self.V_XY, 0.0, 0.0]), 5: np.array([0.0, 0.0, self.V_Z, 0.0]),
            6: np.array([0.0, 0.0, -self.V_Z, 0.0]), 7: np.array([0.0, 0.0, 0.0, self.W_Z]),
            8: np.array([0.0, 0.0, 0.0, -self.W_Z])
        }
        self.num_actions = len(self.actions)
        self.opposite_actions = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}

    def get_velocity(self, action_id):
        return self.actions.get(action_id, np.zeros(4))

    def sample(self, prev_action=None):
        if prev_action is None: return np.random.randint(0, self.num_actions)
        probs = np.ones(self.num_actions);
        probs[prev_action] *= 10.0
        if prev_action in self.opposite_actions: probs[self.opposite_actions[prev_action]] *= 0.01
        probs[0] *= 0.5;
        probs /= np.sum(probs)
        return np.random.choice(self.num_actions, p=probs)


@dataclass
class SGEConfig:
    TOTAL_STEP_BUDGET: int
    NUM_BATCHES: int
    EXPLORE_HORIZON: int = 8
    STEPS_PER_ACTION: int = 15
    CELL_SIZE: float = 1.0
    TIME_RESOLUTION: int = 10
    NUM_ELITES: int = 15


class CellNode:
    def __init__(self, key, snapshot, cumulative_reward, global_step, parent_key=None, action_from_parent=None):
        self.key = key
        self.snapshot = snapshot
        self.cumulative_reward = cumulative_reward
        self.global_step = global_step
        self.times_selected = 0
        self.parent_key = parent_key
        self.action_from_parent = action_from_parent


def get_cell_key(pos, global_step, config):
    ix = int(pos[0] / config.CELL_SIZE);
    iy = int(pos[1] / config.CELL_SIZE)
    iz = int(pos[2] / config.CELL_SIZE);
    it = int(global_step / config.TIME_RESOLUTION)
    return (ix, iy, iz, it)


def select_cell_advanced(node_pool: Dict[tuple, CellNode]):
    if not node_pool: return None
    candidates = list(node_pool.values())
    visits = np.array([n.times_selected for n in candidates])
    rewards = np.array([n.cumulative_reward for n in candidates])
    visits = np.maximum(visits, 1)
    if rewards.max() > rewards.min():
        norm_rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())
    else:
        norm_rewards = np.zeros_like(rewards)
    scores = 1.0 * norm_rewards + 2.0 * (1.0 / np.sqrt(visits))
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    sum_exp = np.sum(exp_scores)
    probs = exp_scores / sum_exp if sum_exp != 0 else np.ones(len(candidates)) / len(candidates)
    idx = np.random.choice(len(candidates), p=probs)
    return candidates[idx]


def reconstruct_trajectory(node: CellNode, archive: Dict[tuple, CellNode]):
    action_sequence = []
    current_node = node
    while current_node.parent_key is not None:
        action_sequence.append(current_node.action_from_parent)
        if current_node.parent_key in archive:
            current_node = archive[current_node.parent_key]
        else:
            print(f"[Error] Parent key {current_node.parent_key} not found!")
            break
    return action_sequence[::-1]


# ==========================================
# 实验核心逻辑 (带阈值筛选)
# ==========================================
def run_extraction_session(num_batches, total_budget, seed, run_id, score_threshold=99.0):
    np.random.seed(seed)
    random.seed(seed)

    config = SGEConfig(TOTAL_STEP_BUDGET=total_budget, NUM_BATCHES=num_batches)
    STEPS_LIMIT_PER_BATCH = total_budget // num_batches

    with suppress_output():
        env = CoverageAviary(gui=False, obstacles=True, user_debug_gui=False, ctrl_freq=30)

    action_space = DroneActionSpace()
    controller = ROS2VelocityController()
    global_archive = {}

    obs, info = env.reset(seed=seed)
    action_space.sample()

    start_snapshot = env.get_snapshot()
    start_key = get_cell_key(start_snapshot['pos'], 0, config)
    root_node = CellNode(start_key, start_snapshot, 0.0, global_step=0)
    global_archive[start_key] = root_node
    current_batch_seeds = [root_node]

    # --- SGE Main Loop ---
    for batch_idx in range(config.NUM_BATCHES):
        batch_pool = {seed.key: seed for seed in current_batch_seeds}
        current_batch_steps_counter = 0

        while current_batch_steps_counter < STEPS_LIMIT_PER_BATCH:
            current_node = select_cell_advanced(batch_pool)
            if current_node is None: break
            current_node.times_selected += 1

            env.restore_snapshot(current_node.snapshot)
            current_cum_reward = current_node.cumulative_reward
            current_global_step = current_node.global_step
            controller.reset(env.pos[0], env.rpy[0][2])

            temp_traj_node = current_node
            last_action_id = getattr(current_node, 'action_from_parent', None)

            for _ in range(config.EXPLORE_HORIZON):
                if current_batch_steps_counter >= STEPS_LIMIT_PER_BATCH: break

                action_id = action_space.sample(prev_action=last_action_id)
                last_action_id = action_id
                target_vel = action_space.get_velocity(action_id)
                step_reward_sum = 0

                for _ in range(config.STEPS_PER_ACTION):
                    if current_batch_steps_counter >= STEPS_LIMIT_PER_BATCH: break
                    rpm, _ = controller.compute_action(env.CTRL_FREQ, env._computeObs(), target_vel)
                    _, reward, terminated, _, _ = env.step(rpm)
                    step_reward_sum += reward
                    current_global_step += 1
                    current_batch_steps_counter += 1
                    if terminated: break

                current_cum_reward += step_reward_sum
                if terminated: break

                pos = env.pos[0]
                new_key = get_cell_key(pos, current_global_step, config)
                should_add = False
                if new_key not in global_archive:
                    should_add = True
                elif current_cum_reward > global_archive[new_key].cumulative_reward:
                    should_add = True

                if should_add:
                    new_snapshot = env.get_snapshot()
                    new_node = CellNode(
                        new_key, new_snapshot, current_cum_reward,
                        global_step=current_global_step,
                        parent_key=temp_traj_node.key,
                        action_from_parent=action_id
                    )
                    global_archive[new_key] = new_node
                    batch_pool[new_key] = new_node
                    temp_traj_node = new_node

        # Select Elites
        all_nodes = list(global_archive.values())
        top_elites = heapq.nlargest(config.NUM_ELITES, all_nodes, key=lambda n: n.cumulative_reward)
        current_batch_seeds = top_elites

    env.close()

    # === [关键修改] 筛选 > 99 分的轨迹 ===
    all_final_nodes = list(global_archive.values())
    # 我们先取出前 50 个最好的，再从中筛 (避免只取前15个可能漏掉99分的)
    # 或者直接遍历所有节点找 > 99 的？
    # 由于节点可能很多，通常我们只关心最好的那些。
    # 这里逻辑：取全历史 Top 50，然后保留 > score_threshold 的

    candidates = heapq.nlargest(50, all_final_nodes, key=lambda n: n.cumulative_reward)

    valid_trajectories = []

    for elite_node in candidates:
        if elite_node.cumulative_reward > score_threshold:
            # 回溯
            actions = reconstruct_trajectory(elite_node, global_archive)

            traj_info = {
                "final_reward": elite_node.cumulative_reward,
                "total_steps": len(actions) * config.STEPS_PER_ACTION,
                "action_sequence": np.array(actions, dtype=np.int32),
                "final_pos": elite_node.snapshot['pos']
            }
            valid_trajectories.append(traj_info)

    # 去重逻辑 (可选)：有时候不同的 Node 可能会回溯出极其相似的路径
    # 这里简单直接返回所有符合条件的
    return valid_trajectories


# ==========================================
# 主程序
# ==========================================
def run_and_save_high_quality():
    # 配置
    BATCHES = 10
    BUDGET = 100000
    REPEATS = 30  # [修改] 跑 30 次
    SCORE_THRESHOLD = 99.0  # [修改] 分数阈值
    SAVE_FILE = "sge_super_elite_trajectories_99plus.pkl"

    all_valid_data = {}
    total_valid_count = 0

    print("========================================")
    print(f"STARTING HIGH-QUALITY TRAJECTORY EXTRACTION")
    print(f"Repeats: {REPEATS}, Threshold: > {SCORE_THRESHOLD}")
    print("========================================")

    for i in range(REPEATS):
        run_id = i + 1
        # 种子逻辑
        seed = BUDGET + (BATCHES * 1000) + i

        print(f"Processing Run {run_id}/{REPEATS} (Seed: {seed})... ", end="", flush=True)
        start_t = time.time()

        # 运行实验
        trajs = run_extraction_session(BATCHES, BUDGET, seed, run_id, score_threshold=SCORE_THRESHOLD)

        duration = time.time() - start_t

        if len(trajs) > 0:
            # 存入字典
            all_valid_data[run_id] = trajs
            total_valid_count += len(trajs)
            best_r = trajs[0]['final_reward']
            print(f"Done ({duration:.1f}s). Found {len(trajs)} trajs > {SCORE_THRESHOLD}. Best: {best_r:.2f}")
        else:
            print(f"Done ({duration:.1f}s). No trajectories > {SCORE_THRESHOLD}.")

    # === 保存数据 ===
    if total_valid_count > 0:
        with open(SAVE_FILE, 'wb') as f:
            pickle.dump(all_valid_data, f)

        print("\n========================================")
        print(f"SUCCESS: Saved {total_valid_count} trajectories to {SAVE_FILE}")
        print("========================================")
    else:
        print("\n[Warning] No trajectories met the criteria. Nothing saved.")


if __name__ == "__main__":
    run_and_save_high_quality()