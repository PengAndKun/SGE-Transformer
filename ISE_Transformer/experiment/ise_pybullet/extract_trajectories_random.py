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
def run_random_session(global_budget, seed, run_id):
    """
    统一 Budget 维度的纯随机策略：
    每随机测试一个动作，扣除 1 个 Budget。
    """
    np.random.seed(seed)
    random.seed(seed)
    action_space = MacroActionSpace27(move_distance=0.5)

    with suppress_output():
        env = CoverageAviary(gui=False, obstacles=True, ctrl_freq=30, num_rays=120, grid_res=0.5)
    env.reset(seed=seed)

    # 初始化根节点
    current_node = CellNode(
        node_id=0, parent_id=None,
        logical_snapshot=get_logical_snapshot(env),
        cumulative_reward=0.0, global_step=0,
        last_state=np.concatenate([env.pos[0].copy(), env.rpy[0].copy()])
    )

    trajectory_nodes = [current_node]
    consumed_budget = 0
    node_counter = 1

    # 只要 Budget 没用完，就一直跑
    while consumed_budget < global_budget:

        # --- 随机决策 ---
        # 随机从 27 个动作中选一个
        action_id = random.randint(0, action_space.num_actions - 1)

        # 恢复到当前位置的物理状态
        restore_env_logically(env, current_node)

        # 执行动作，消耗 1 个 Budget
        step_reward, terminated = execute_translation_step(env, action_id, action_space)
        consumed_budget += 1

        # --- 结果处理 ---
        if not terminated:
            # 动作有效：生成新节点，位置发生迁移
            new_cum_reward = current_node.cumulative_reward + step_reward
            new_node = CellNode(
                node_id=node_counter,
                parent_id=current_node.node_id,
                logical_snapshot=get_logical_snapshot(env),
                cumulative_reward=new_cum_reward,
                global_step=current_node.global_step + 1,
                action_from_parent=action_id,
                last_state=np.concatenate([env.pos[0].copy(), env.rpy[0].copy()]),
                last_reward=step_reward
            )

            current_node = new_node  # 无人机“走到”了新位置
            trajectory_nodes.append(new_node)
            node_counter += 1
        else:
            # 动作无效（碰撞）：位置不动，原地尝试下一个随机动作
            pass

        # 定期打印，防止进程看起来像死掉了
        if consumed_budget % 10000 == 0:
            print(f"  [Run {run_id}] Budget: {consumed_budget}/{global_budget} | Nodes: {node_counter}")

    # --- 格式化轨迹输出 ---
    actions_list, states_list, rewards_list = [], [], []
    states_list.append(trajectory_nodes[0].last_state)
    for i in range(1, len(trajectory_nodes)):
        node = trajectory_nodes[i]
        actions_list.append(node.action_from_parent)
        rewards_list.append(node.last_reward)
        states_list.append(node.last_state)

    traj_info = {
        "final_reward": trajectory_nodes[-1].cumulative_reward,
        "total_steps": len(actions_list),
        "action_sequence": np.array(actions_list, dtype=np.int32),
        "state_sequence": np.array(states_list, dtype=np.float32),
        "reward_sequence": np.array(rewards_list, dtype=np.float32),
        "final_pos": trajectory_nodes[-1].last_state[:3]
    }

    env.close()
    del env
    gc.collect()

    print(f"  [Run {run_id}] Random finished. Final Reward: {traj_info['final_reward']:.2f}")
    return [traj_info]
# ==========================================
# 多进程包装与主程序
# ==========================================
def run_extraction_wrapper_random(run_info, budget):
    run_id, seed = run_info
    try:
        return run_random_session(budget, seed, run_id)
    except Exception as e:
        print(f"  [Run {run_id}] Error: {e}")
        return []


def run_parallel_extraction_random(budget):
    BUDGET = budget  # 这里指树的生长次数 (Max Nodes to add)
    NUM_PROCESSES = 10
    REPEATS = 20
    SAVE_FILE = f"../../data_pybullet/3d_trajectories/random/random_trajectories_budget_{BUDGET}.pkl"

    print("========================================")
    print(f"STARTING RANDOM TRAJECTORY EXTRACTION")
    print(f"Repeats: {REPEATS}, Max Growing Iterations: {BUDGET}")
    print("========================================")

    tasks = [(i + 1, 888000 + i) for i in range(REPEATS)]
    worker_func = partial(run_extraction_wrapper_random, budget=BUDGET)

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
        run_parallel_extraction_random(budget=budget)
