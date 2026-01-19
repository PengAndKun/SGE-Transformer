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

try:
    from ISE_Transformer.envs.coveragel_multi_room import CoverageAviary
except ImportError:
    print("Warning: Environment import failed. Make sure ISE_Transformer is in python path.")


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
# RRT 核心逻辑
# ==========================================
def run_rrt_session(global_budget, seed, run_id):
    """
    单次 RRT 运行：
    1. 随机采样空间点
    2. 找到树中最近节点
    3. 向采样点生长（步进）
    4. 记录覆盖奖励
     统一 Budget 维度的 RRT：
     每调用一次 execute_translation_step，扣除 1 个 Budget。
    """
    np.random.seed(seed)
    random.seed(seed)
    action_space = MacroActionSpace27(move_distance=0.5)

    with suppress_output():
        env = CoverageAviary(gui=False, obstacles=True, ctrl_freq=30, num_rays=120, grid_res=0.5)
    env.reset(seed=seed)

    # 1. 初始化树
    root_node = CellNode(
        node_id=0, parent_id=None,
        logical_snapshot=get_logical_snapshot(env),
        cumulative_reward=0.0, global_step=0,
        last_state=np.concatenate([env.pos[0].copy(), env.rpy[0].copy()])
    )
    tree_nodes = [root_node]
    node_counter = 1

    # === 关键：统一维度计数器 ===
    consumed_budget = 0

    # 环境边界
    bounds = env.grid_bounds

    while consumed_budget < global_budget:
        # A. 采样
        if random.random() < 0.8:
            p_rand = np.array([random.uniform(bounds[0][0], bounds[0][1]),
                               random.uniform(bounds[1][0], bounds[1][1]),
                               random.uniform(0.5, 3.5)])
        else:
            # 向高奖励节点附近采样 (Heuristic)
            best_node = max(tree_nodes, key=lambda n: n.cumulative_reward)
            p_rand = best_node.last_state[:3] + np.random.normal(0, 1.5, 3)

        # B. 最近邻查找
        dists = [np.linalg.norm(n.last_state[:3] - p_rand) for n in tree_nodes]
        nearest_node = tree_nodes[np.argmin(dists)]

        # C. 步进 (此处测试 27 个动作)
        best_act, min_d, best_r, best_snap, best_state = -1, 1e9, 0, None, None

        # 随机化动作顺序，确保在 budget 耗尽前公平对待所有方向
        actions = list(range(action_space.num_actions))
        random.shuffle(actions)

        for action_id in actions:
            if consumed_budget >= global_budget:
                break  # 预算耗尽，立即停止测试

            # 执行一次物理测试，消耗 1 个 budget
            restore_env_logically(env, nearest_node)
            step_reward, terminated = execute_translation_step(env, action_id, action_space)
            consumed_budget += 1

            if not terminated:
                d = np.linalg.norm(env.pos[0] - p_rand)
                if d < min_d:
                    min_d, best_act, best_r = d, action_id, step_reward
                    best_snap = get_logical_snapshot(env)
                    best_state = np.concatenate([env.pos[0].copy(), env.rpy[0].copy()])

        # D. 如果找到了有效的一步，加入树
        if best_act != -1:
            new_node = CellNode(
                node_id=node_counter, parent_id=nearest_node.node_id,
                logical_snapshot=best_snap,
                cumulative_reward=nearest_node.cumulative_reward + best_r,
                global_step=nearest_node.global_step + 1,
                action_from_parent=best_act, last_state=best_state, last_reward=best_r
            )
            tree_nodes.append(new_node)
            node_counter += 1

    # --- 最终路径提取 (寻找累积奖励最高的叶子节点) ---
    best_leaf = max(tree_nodes, key=lambda n: n.cumulative_reward)

    # 回溯回根节点
    actions_list, states_list, rewards_list = [], [], []
    curr = best_leaf
    archive_dict = {n.node_id: n for n in tree_nodes}

    while curr is not None:
        states_list.append(curr.last_state)
        if curr.parent_id is not None:
            actions_list.append(curr.action_from_parent)
            rewards_list.append(curr.last_reward)
            curr = archive_dict[curr.parent_id]
        else:
            curr = None

    # 反转列表
    traj_info = {
        "final_reward": best_leaf.cumulative_reward,
        "total_steps": len(actions_list),
        "action_sequence": np.array(actions_list[::-1], dtype=np.int32),
        "state_sequence": np.array(states_list[::-1], dtype=np.float32),
        "reward_sequence": np.array(rewards_list[::-1], dtype=np.float32),
        "final_pos": best_leaf.last_state[:3]
    }

    env.close()
    del env
    gc.collect()

    print(f'  [Run {run_id}] RRT finished. Nodes: {len(tree_nodes)}, Best Reward: {traj_info["final_reward"]:.2f}')
    return [traj_info]


# ==========================================
# 多进程包装与主程序
# ==========================================
def run_extraction_wrapper_rrt(run_info, budget):
    run_id, seed = run_info
    try:
        return run_rrt_session(budget, seed, run_id)
    except Exception as e:
        print(f"  [Run {run_id}] Error: {e}")
        return []


def run_parallel_extraction_rrt(budget):
    BUDGET = budget  # 这里指树的生长次数 (Max Nodes to add)
    NUM_PROCESSES = 10
    REPEATS = 20
    SAVE_FILE = f"../../data_pybullet/3d_trajectories/rrt/rrt_trajectories_nodes_{BUDGET}.pkl"

    print("========================================")
    print(f"STARTING RRT TRAJECTORY EXTRACTION")
    print(f"Repeats: {REPEATS}, Max Growing Iterations: {BUDGET}")
    print("========================================")

    tasks = [(i + 1, 888000 + i) for i in range(REPEATS)]
    worker_func = partial(run_extraction_wrapper_rrt, budget=BUDGET)

    with mp.Pool(processes=NUM_PROCESSES, maxtasksperchild=1) as pool:
        results = pool.map(worker_func, tasks)

    all_valid_data = [t for sub in results for t in sub]

    if all_valid_data:
        os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
        with open(SAVE_FILE, 'wb') as f:
            pickle.dump(all_valid_data, f)
        print(f"SUCCESS: Saved {len(all_valid_data)} RRT trajectories.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # RRT 的预算通常设置为 500-2000 次生长迭代比较合适
    budgets=[1000,2000,5000,10000,20000,50000]
    for budget in budgets:
        run_parallel_extraction_rrt(budget=budget)
