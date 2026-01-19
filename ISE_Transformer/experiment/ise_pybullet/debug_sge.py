import numpy as np
import time
import random
import pybullet as p
import heapq
import os
import csv
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict
import sys

# 引入依赖
from ISE_Transformer.envs.coverage_lidar_aviary import CoverageAviary
from stable_controller import ROS2VelocityController


# 尝试屏蔽 PyBullet 的输出干扰
class suppress_output:
    def __init__(self):
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


# 1. 离散动作库
class DroneActionSpace:
    def __init__(self):
        self.V_XY = 0.5
        self.V_Z = 0.3
        self.W_Z = 0.5
        self.actions = {
            0: np.array([0.0, 0.0, 0.0, 0.0]),
            1: np.array([self.V_XY, 0.0, 0.0, 0.0]),
            2: np.array([-self.V_XY, 0.0, 0.0, 0.0]),
            3: np.array([0.0, self.V_XY, 0.0, 0.0]),
            4: np.array([0.0, -self.V_XY, 0.0, 0.0]),
            5: np.array([0.0, 0.0, self.V_Z, 0.0]),
            6: np.array([0.0, 0.0, -self.V_Z, 0.0]),
            7: np.array([0.0, 0.0, 0.0, self.W_Z]),
            8: np.array([0.0, 0.0, 0.0, -self.W_Z])
        }
        self.num_actions = len(self.actions)
        self.opposite_actions = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}

    def get_velocity(self, action_id):
        return self.actions.get(action_id, np.zeros(4))

    def sample(self, prev_action=None):
        if prev_action is None: return np.random.randint(0, self.num_actions)
        probs = np.ones(self.num_actions)
        probs[prev_action] *= 10.0
        if prev_action in self.opposite_actions:
            probs[self.opposite_actions[prev_action]] *= 0.01
        probs[0] *= 0.5
        probs /= np.sum(probs)
        return np.random.choice(self.num_actions, p=probs)


# 10_e. 配置参数 (动态化)
@dataclass
class SGEConfig:
    TOTAL_STEP_BUDGET: int  # 总步数预算
    NUM_BATCHES: int  # 分多少批次执行

    EXPLORE_HORIZON: int = 8
    STEPS_PER_ACTION: int = 15
    CELL_SIZE: float = 1.0
    TIME_RESOLUTION: int = 10
    NUM_ELITES: int = 15


# 3. 节点定义
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
    ix = int(pos[0] / config.CELL_SIZE)
    iy = int(pos[1] / config.CELL_SIZE)
    iz = int(pos[2] / config.CELL_SIZE)
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
    # 防止溢出
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)

    sum_exp = np.sum(exp_scores)
    if sum_exp == 0:
        probs = np.ones(len(candidates)) / len(candidates)
    else:
        probs = exp_scores / sum_exp

    # 使用索引选择，避免 numpy 版本兼容问题
    idx = np.random.choice(len(candidates), p=probs)
    return candidates[idx]


# ==========================================
# 核心运行逻辑
# ==========================================
def run_single_experiment(num_batches, total_budget, seed, run_id):
    """
    运行单次实验
    """
    # 设置随机种子
    np.random.seed(seed)
    random.seed(seed)

    config = SGEConfig(TOTAL_STEP_BUDGET=total_budget, NUM_BATCHES=num_batches)

    # 计算每个 Batch 的步数限制
    STEPS_LIMIT_PER_BATCH = total_budget // num_batches

    # 为了速度，gui=False
    with suppress_output():
        env = CoverageAviary(gui=False, obstacles=True, user_debug_gui=False, ctrl_freq=30)

    action_space = DroneActionSpace()
    controller = ROS2VelocityController()

    global_archive = {}

    # 初始状态
    obs, info = env.reset()

    action_space.sample()

    start_snapshot = env.get_snapshot()
    start_key = get_cell_key(start_snapshot['pos'], 0, config)

    root_node = CellNode(start_key, start_snapshot, 0.0, global_step=0)
    global_archive[start_key] = root_node
    current_batch_seeds = [root_node]

    print(f"  -> Run {run_id}: Batches={num_batches}, Budget={total_budget}")

    for batch_idx in range(config.NUM_BATCHES):
        # 构建当前 Pool
        batch_pool = {}
        for seed in current_batch_seeds:
            batch_pool[seed.key] = seed

        current_batch_steps_counter = 0

        # 在这个 while 循环中，直到消耗完本批次的步数预算
        while current_batch_steps_counter < STEPS_LIMIT_PER_BATCH:

            # A. Select
            current_node = select_cell_advanced(batch_pool)
            if current_node is None: break  # 异常保护
            current_node.times_selected += 1

            # B. Restore
            env.restore_snapshot(current_node.snapshot)
            current_cum_reward = current_node.cumulative_reward
            current_global_step = current_node.global_step
            controller.reset(env.pos[0], env.rpy[0][2])

            # C. Explore
            temp_traj_node = current_node
            last_action_id = getattr(current_node, 'action_from_parent', None)

            for _ in range(config.EXPLORE_HORIZON):
                # 如果本批次步数耗尽，停止探索
                if current_batch_steps_counter >= STEPS_LIMIT_PER_BATCH:
                    break

                action_id = action_space.sample(prev_action=last_action_id)
                last_action_id = action_id
                target_vel = action_space.get_velocity(action_id)

                step_reward_sum = 0

                # 执行动作
                for _ in range(config.STEPS_PER_ACTION):
                    if current_batch_steps_counter >= STEPS_LIMIT_PER_BATCH:
                        break  # 硬截断，保证预算精确

                    rpm, _ = controller.compute_action(env.CTRL_FREQ, env._computeObs(), target_vel)
                    _, reward, terminated, _, _ = env.step(rpm)

                    step_reward_sum += reward
                    current_global_step += 1
                    current_batch_steps_counter += 1

                    if terminated: break

                current_cum_reward += step_reward_sum
                if terminated: break  # 撞墙结束该轨迹

                # D. Archive
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

        # --- Batch 结束，选择精英 ---
        all_nodes = list(global_archive.values())
        top_elites = heapq.nlargest(config.NUM_ELITES, all_nodes, key=lambda n: n.cumulative_reward)
        current_batch_seeds = top_elites

    env.close()

    if not global_archive:
        return 0.0
    print("\n=== All Batches Finished ===")

    print("    Loading Seeds for this end batch:")
    for seed in current_batch_seeds:
        print(f"      - Reward: {seed.cumulative_reward:.2f} | Step: {seed.global_step}")


    best_overall = max(global_archive.values(), key=lambda n: n.cumulative_reward)
    return best_overall.cumulative_reward


# ==========================================
# 批量实验管理器
# ==========================================
def run_benchmark():
    # 实验设置
    # BATCH_SETTINGS = [1, 10_e, 4, 5, 8, 10]
    # BUDGET_SETTINGS = [1000, 2000, 5000, 10000, 25000, 50000, 100000]
    BATCH_SETTINGS = [ 8]
    BUDGET_SETTINGS = [100000]
    REPEATS = 10

    # 结果保存目录
    os.makedirs("results", exist_ok=True)

    print(f"=== SGE Benchmark Started ===")
    print(f"Batches: {BATCH_SETTINGS}")
    print(f"Budgets: {BUDGET_SETTINGS}")
    print(f"Repeats per setting: {REPEATS}")
    print(f"Total experiments: {len(BATCH_SETTINGS) * len(BUDGET_SETTINGS) * REPEATS}")
    print("-" * 60)

    # 循环遍历所有配置
    for batches in BATCH_SETTINGS:
        for budget in BUDGET_SETTINGS:

            results = []
            print(f"\nProcessing: Batches={batches}, Budget={budget}")

            start_time = time.time()

            for i in range(REPEATS):
                # 动态生成种子，保证可复现性
                # 种子公式：基础偏移 + 预算 + 批次*100 + 第几次重复
                # 例如：10000 + 100 + 1 = 10101
                seed = budget + (batches * 1000) + i

                score = run_single_experiment(batches, budget, seed, i + 1)
                print(f"  Run {i + 1}: {score:.2f} seed {seed}")
                results.append(score)
                time.sleep(0.1)

                # 简易进度条
                print(f".", end="", flush=True)

            end_time = time.time()
            avg_score = np.mean(results)
            std_score = np.std(results)
            max_score = np.max(results)

            print(
                f"\n  Done in {end_time - start_time:.1f}s. Avg: {avg_score:.2f} ± {std_score:.2f}, Max: {max_score:.2f}")

            # 保存该配置的所有结果
            filename = f"results/SGE_Batch{batches}_Budget{budget}.csv"
            df = pd.DataFrame({
                "run_id": range(1, REPEATS + 1),
                "reward": results,
                "batches": batches,
                "budget": budget
            })
            df.to_csv(filename, index=False)
            print(f"  Saved to {filename}")

    print("\n=== All Benchmarks Completed ===")


if __name__ == "__main__":
    run_benchmark()
