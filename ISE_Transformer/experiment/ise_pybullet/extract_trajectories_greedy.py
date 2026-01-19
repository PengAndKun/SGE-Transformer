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

# 引入依赖 (保持不变)
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
# Greedy 核心逻辑
# ==========================================
def run_greedy_session(total_budget, seed, run_id):
    """
    单次 Greedy 运行：在每一步尝试所有可能的动作，选择收益最高的一个执行。
    """
    np.random.seed(seed)
    random.seed(seed)

    # 动作空间
    action_space = MacroActionSpace27(move_distance=0.5)

    # 初始化环境
    with suppress_output():
        env = CoverageAviary(gui=False, obstacles=True, ctrl_freq=30, num_rays=120,
                             radar_radius=8.0, viz_rays_every=10, viz_points_every=20, grid_res=0.5)

    env.reset(seed=seed)

    # 根节点 (起点)
    root_snapshot = get_logical_snapshot(env)
    current_node = CellNode(
        node_id=0,
        parent_id=None,
        logical_snapshot=root_snapshot,
        cumulative_reward=0.0,
        global_step=0,
        last_state=np.concatenate([env.pos[0].copy(), env.rpy[0].copy()])
    )

    # 用于存储最终轨迹链
    # 因为 Greedy 是线性的，我们直接存在 list 里即可，不需要 global_archive 查找
    trajectory_nodes = [current_node]

    # 开始循环
    steps_count = 0

    sum_steps_count = 0

    # 贪心策略可能会很快陷入死胡同或者 loop，设置一个 budget
    while sum_steps_count < total_budget:

        best_action = -1
        best_reward_gain = -1e9
        best_snapshot = None
        best_state_vec = None
        found_valid_move = False

        # --- Greedy Step: 试探所有 27 个动作 ---
        # 随机打乱尝试顺序，避免在奖励相同时总是偏向某个方向
        actions_to_try = list(range(action_space.num_actions))
        random.shuffle(actions_to_try)

        for action_id in actions_to_try:
            # 1. 每次试探前，必须先将环境恢复到当前节点状态
            restore_env_logically(env, current_node)
            sum_steps_count+=1

            # 2. 执行动作
            step_reward, terminated = execute_translation_step(env, action_id, action_space)

            # 3. 如果未撞墙且奖励更优
            if not terminated:
                # 贪心准则：选择这一步获得 reward 最大的动作
                # 注意：这里比较的是 step_reward (增量)，等价于比较 cumulative_reward
                if step_reward > best_reward_gain:
                    best_reward_gain = step_reward
                    best_action = action_id

                    # 记录执行完该动作后的状态，以便稍后确认
                    best_snapshot = get_logical_snapshot(env)
                    best_state_vec = np.concatenate([env.pos[0].copy(), env.rpy[0].copy()])
                    found_valid_move = True

        # --- 决策 ---
        if found_valid_move:
            # 确认选择最佳动作
            new_cum_reward = current_node.cumulative_reward + best_reward_gain

            new_node = CellNode(
                node_id=steps_count + 1,
                parent_id=current_node.node_id,
                logical_snapshot=best_snapshot,  # 使用刚才试探时保存的 snapshot，避免重复计算
                cumulative_reward=new_cum_reward,
                global_step=steps_count + 1,
                action_from_parent=best_action,
                last_state=best_state_vec,
                last_reward=best_reward_gain
            )

            # 更新指针
            current_node = new_node
            trajectory_nodes.append(new_node)
            steps_count += 1

            # 可选：打印进度
            # if steps_count % 50 == 0:
            #     print(f"Run {run_id} | Step {steps_count} | Reward: {new_cum_reward:.2f}")

        else:
            # 如果所有动作都会导致撞墙或终止，则提前结束
            # print(f"Run {run_id} terminated early at step {steps_count} (No valid moves).")
            break

    # 循环结束，清理环境
    env.close()
    del env
    gc.collect()

    # --- 轨迹格式化 (与原 SGE 输出格式保持一致) ---
    # trajectory_nodes 列表本身就是有序的 [Start, Step1, Step2, ...]

    actions_list = []
    states_list = []
    rewards_list = []

    # 遍历节点收集数据
    # Node 0 是初始状态，没有 action_from_parent
    states_list.append(trajectory_nodes[0].last_state)  # s_0

    for i in range(1, len(trajectory_nodes)):
        node = trajectory_nodes[i]
        actions_list.append(node.action_from_parent)  # a_{t-1}
        rewards_list.append(node.last_reward)  # r_{t-1}
        states_list.append(node.last_state)  # s_t

    traj_info = {
        "final_reward": trajectory_nodes[-1].cumulative_reward,
        "total_steps": len(actions_list),
        "action_sequence": np.array(actions_list, dtype=np.int32),
        "state_sequence": np.array(states_list, dtype=np.float32),
        "reward_sequence": np.array(rewards_list, dtype=np.float32),
        "final_pos": trajectory_nodes[-1].last_state[:3]
    }
    print(f'  [Run {run_id}] Greedy finished. Steps: {traj_info["total_steps"]}, Final Reward: {traj_info["final_reward"]:.2f}')
    # Greedy 只生成一条轨迹
    return [traj_info]


def run_extraction_wrapper_greedy(run_info, budget):
    """
    多进程包装器
    """
    run_id, seed = run_info
    try:
        trajs = run_greedy_session(budget, seed, run_id)
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
    SAVE_FILE = f"../../data_pybullet/3d_trajectories/greedy/greedy_trajectories_budget_{BUDGET}.pkl"

    print("========================================")
    print(f"STARTING GREEDY TRAJECTORY EXTRACTION")
    print(f"Repeats: {REPEATS}, Max Steps: {BUDGET}")
    print("========================================")

    # 1. 准备任务参数
    tasks = [(i + 1, 999000 + i) for i in range(REPEATS)]  # 使用不同的 seed 范围

    # 2. 偏函数
    worker_func = partial(run_extraction_wrapper_greedy, budget=BUDGET)

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
    BUDGETS = [50000]

    for budget in BUDGETS:
        run_parallel_extraction_greedy(budget)