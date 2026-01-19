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
import gc
import multiprocessing as mp
from functools import partial
from collections import defaultdict
# 引入依赖 (保持不变)
try:
    from ISE_Transformer.envs.coveragel_multi_room import CoverageAviary
except ImportError:
    print("Warning: Environment import failed. Make sure ISE_Transformer is in python path.")

@dataclass
class ACOConfig:
    alpha: float = 1.0  # 信息素重要程度
    beta: float = 2.0   # 启发式因子(即时奖励)重要程度
    rho: float = 0.1    # 信息素挥发率
    Q: float = 100.0    # 信息素强度基数
    ant_count: int = 5  # 每一代同时出发的蚂蚁数量

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
class MacroActionSpace27:
    def __init__(self, move_distance=1.5):
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



# 简化版 Node，仅用于状态恢复和记录
class CellNode:
    __slots__ = ['node_id', 'parent_id', 'logical_snapshot', 'cumulative_reward',
                 'global_step', 'action_from_parent', 'last_state', 'last_reward']

    def __init__(self, node_id, parent_id, logical_snapshot, cumulative_reward, global_step,
                 action_from_parent=None, last_state=None, last_reward=0.0):
        self.node_id = node_id
        self.parent_id = parent_id
        self.logical_snapshot = logical_snapshot
        self.cumulative_reward = cumulative_reward
        self.global_step = global_step
        self.action_from_parent = action_from_parent
        self.last_state = last_state
        self.last_reward = last_reward


# ==========================================
# 工具函数 (复用)
# ==========================================
def restore_env_logically(env, node):
    env.coverage_grid = node.logical_snapshot['grid'].copy()
    pos = node.logical_snapshot['pos']
    rpy = node.logical_snapshot['rpy']
    quat = p.getQuaternionFromEuler(rpy)
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], pos, quat, physicsClientId=env.CLIENT)
    env._updateAndStoreKinematicInformation()


def get_logical_snapshot(env):
    return {
        'grid': env.coverage_grid.copy(),
        'pos': env.pos[0].copy(),
        'rpy': env.rpy[0].copy()
    }


def execute_translation_step(env, action_id, action_space, resolution=0.5):
    # 与原代码保持一致
    MIN_HEIGHT = 0.5
    MAX_HEIGHT = 3.5

    terminated = False
    total_reward = 0.0

    displacement = action_space.get_displacement(action_id)
    start_pos = np.array(env.pos[0])
    target_pos = start_pos + displacement

    if target_pos[2] < MIN_HEIGHT or target_pos[2] > MAX_HEIGHT:
        return 0.0, True

    actual_vec = target_pos - start_pos
    dist = np.linalg.norm(actual_vec)

    if dist > 1e-6:
        num_samples = max(int(dist / resolution), 2)
        for i in range(1, num_samples + 1):
            inter_pos = start_pos + (i / num_samples) * actual_vec
            prev_inter_pos = start_pos + ((i - 1) / num_samples) * actual_vec

            ray_test = p.rayTest(prev_inter_pos, inter_pos, physicsClientId=env.CLIENT)
            if ray_test[0][0] != -1 and ray_test[0][0] != env.DRONE_IDS[0]:
                terminated = True
                break

            p.resetBasePositionAndOrientation(env.DRONE_IDS[0], inter_pos, [0, 0, 0, 1], physicsClientId=env.CLIENT)
            p.performCollisionDetection(physicsClientId=env.CLIENT)
            contacts = p.getContactPoints(bodyA=env.DRONE_IDS[0], physicsClientId=env.CLIENT)
            if len(contacts) > 0:
                terminated = True
                break

            obs, reward, terminated, _, info = env.compute_scan_at_pos(inter_pos)
            if terminated: break
            total_reward += reward
    else:
        # 原地
        obs, reward, terminated, _, info = env.compute_scan_at_pos(target_pos)
        total_reward = reward

    return total_reward, terminated



# ==========================================
# ACO 蚁群逻辑
# ==========================================
def run_aco_session(total_budget, seed, run_id):
    np.random.seed(seed)
    random.seed(seed)
    config = ACOConfig()
    action_space = MacroActionSpace27(move_distance=0.5)

    # 1. 初始化环境
    with suppress_output():
        env = CoverageAviary(gui=False, obstacles=True, ctrl_freq=30, num_rays=120, grid_res=0.5)
    env.reset(seed=seed)

    # 2. 初始化全局信息素 (使用字典模拟稀疏 3D 网格)
    # Key: (ix, iy, iz), Value: 浓度 (初始设为 1.0)
    pheromone_map = defaultdict(lambda: 1.0)

    consumed_budget = 0
    all_trajectories = []

    # 辅助函数：获取空间索引
    def get_pos_idx(pos):
        return (int(pos[0] / 0.5), int(pos[1] / 0.5), int(pos[2] / 0.5))

    # 主循环：直到预算耗尽
    while consumed_budget < total_budget:
        batch_ants_data = []

        # --- A. 蚂蚁迭代阶段 ---
        for ant_id in range(config.ant_count):
            if consumed_budget >= total_budget: break

            env.reset(seed=seed)
            curr_node = CellNode(0, None, get_logical_snapshot(env), 0.0, 0,
                                 last_state=np.concatenate([env.pos[0].copy(), env.rpy[0].copy()]))

            ant_path = [curr_node]
            ant_total_reward = 0

            # 限制单只蚂蚁的最大步数，防止死循环
            max_steps = 200
            for step in range(max_steps):
                if consumed_budget >= total_budget: break

                # 尝试所有 27 个动作并计算概率
                probs = []
                candidates = []

                for a_id in range(action_space.num_actions):
                    # 虚拟测试 (消耗预算)
                    restore_env_logically(env, curr_node)
                    reward_gain, terminated = execute_translation_step(env, a_id, action_space)
                    consumed_budget += 1

                    if not terminated:
                        # 获取目标位置的信息素
                        target_idx = get_pos_idx(env.pos[0])
                        tau = pheromone_map[target_idx]
                        eta = reward_gain + 0.01  # 启发式因子，加微量防止为0

                        score = (tau ** config.alpha) * (eta ** config.beta)
                        probs.append(score)
                        candidates.append((a_id, reward_gain, get_logical_snapshot(env),
                                           np.concatenate([env.pos[0].copy(), env.rpy[0].copy()])))

                if not candidates: break  # 无路可走

                # 轮盘赌选择
                probs = np.array(probs) / np.sum(probs)
                chosen_idx = np.random.choice(len(candidates), p=probs)
                chosen_move = candidates[chosen_idx]

                # 更新状态
                new_node = CellNode(step + 1, curr_node.node_id, chosen_move[2],
                                    curr_node.cumulative_reward + chosen_move[1],
                                    step + 1, chosen_move[0], chosen_move[3], chosen_move[1])

                curr_node = new_node
                ant_path.append(new_node)
                ant_total_reward += chosen_move[1]

            batch_ants_data.append((ant_path, ant_total_reward))

        # --- B. 信息素更新与挥发阶段 ---
        # 1. 挥发
        for key in list(pheromone_map.keys()):
            pheromone_map[key] *= (1.0 - config.rho)
            if pheromone_map[key] < 0.1: del pheromone_map[key]  # 极低浓度清理

        # 2. 增强 (奖励越高的路径信息素越多)
        for path, score in batch_ants_data:
            if score <= 0: continue
            delta = (config.Q * score) / 100.0  # 增量公式
            for node in path:
                idx = get_pos_idx(node.last_state[:3])
                pheromone_map[idx] += delta

            # 记录历史轨迹用于最后汇总
            all_trajectories.append(path)

    # 3. 清理与返回 (选择所有蚂蚁中表现最好的一条)
    env.close()
    if not all_trajectories: return []

    # 找到最终奖励最高的蚂蚁轨迹
    best_path = max(all_trajectories, key=lambda p: p[-1].cumulative_reward)

    # 转换格式
    traj_info = {
        "final_reward": best_path[-1].cumulative_reward,
        "total_steps": len(best_path) - 1,
        "action_sequence": np.array([n.action_from_parent for n in best_path if n.action_from_parent is not None]),
        "state_sequence": np.array([n.last_state for n in best_path]),
        "reward_sequence": np.array([n.last_reward for n in best_path if n.parent_id is not None]),
        "final_pos": best_path[-1].last_state[:3]
    }

    print(
        f'  [Run {run_id}] ACO finished. Best Reward: {traj_info["final_reward"]:.2f}, Budget Used: {consumed_budget}')
    return [traj_info]

def run_extraction_wrapper_aco(run_info, budget):
    """
    多进程包装器
    """
    run_id, seed = run_info
    try:
        trajs = run_aco_session(budget, seed, run_id)
        # print(f"  [Process {os.getpid()}] Greedy Run {run_id} finished. Len: {trajs[0]['total_steps']}, Score: {trajs[0]['final_reward']:.2f}")
        return trajs
    except Exception as e:
        print(f"  [Process {os.getpid()}] Greedy Run {run_id} failed: {e}")
        return []



# ==========================================
# 主程序
# ==========================================
def run_parallel_extraction_greedy(budget):
    # 配置
    BUDGET = budget  # 这里的 budget 是最大步数，不是采样次数
    NUM_PROCESSES = 10
    REPEATS = 20  # 跑 50 条 Greedy 轨迹

    # 这里的 BATCHES=0 仅用于文件名标识，表示 Greedy
    SAVE_FILE = f"../../data_pybullet/3d_trajectories/aco/aco_trajectories_budget_{budget}.pkl"

    print("========================================")
    print(f"STARTING ANT TRAJECTORY EXTRACTION")
    print(f"Repeats: {REPEATS}, Max Steps: {BUDGET}")
    print("========================================")

    # 1. 准备任务参数
    tasks = [(i + 1, 999000 + i) for i in range(REPEATS)]  # 使用不同的 seed 范围

    # 2. 偏函数
    worker_func = partial(run_extraction_wrapper_ant, budget=budget) # 需要定义该包装器

    start_time = time.time()

    # 3. 启动进程池
    with mp.Pool(processes=NUM_PROCESSES, maxtasksperchild=1) as pool:
        results = pool.map(worker_func, tasks)

    # 4. 汇总
    all_valid_data = []
    for trajs in results:
        all_valid_data.extend(trajs)

    duration = time.time() - start_time
    print(f"\n[Summary] Greedy extraction finished in {duration:.1f}s")
    print(f"Total trajectories collected: {len(all_valid_data)}")

    # 5. 保存
    if len(all_valid_data) > 0:
        # 按分数排序
        kept_trajs = sorted(all_valid_data, key=lambda t: t["final_reward"], reverse=True)

        # 确保目录存在
        os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)

        with open(SAVE_FILE, 'wb') as f:
            pickle.dump(kept_trajs, f)
        print(f"SUCCESS: Saved {len(kept_trajs)}  greedy trajectories to {SAVE_FILE}")
    else:
        print("Warning: No trajectories collected.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    # 这里的 Budget 指的是单条轨迹允许的最大步数
    BUDGETS = [20000,50000]

    for budget in BUDGETS:
        run_parallel_extraction_greedy(budget)