import numpy as np
import time
import random
import pybullet as p
import heapq
import os
import pickle
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple
import gc
import multiprocessing as mp
from functools import partial

# 引入依赖
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
# 基础类定义
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


# 增强版 Node，增加了 visit_count 用于 Go-Explore 的选择策略
class CellNode:
    __slots__ = ['node_id', 'parent_id', 'logical_snapshot', 'cumulative_reward',
                 'global_step', 'action_from_parent', 'last_state', 'last_reward',
                 'visit_count', 'cell_key']

    def __init__(self, node_id, parent_id, logical_snapshot, cumulative_reward, global_step,
                 action_from_parent=None, last_state=None, last_reward=0.0, cell_key=None):
        self.node_id = node_id
        self.parent_id = parent_id
        self.logical_snapshot = logical_snapshot
        self.cumulative_reward = cumulative_reward
        self.global_step = global_step
        self.action_from_parent = action_from_parent
        self.last_state = last_state
        self.last_reward = last_reward
        self.visit_count = 0  # 该节点被选中作为出发点的次数
        self.cell_key = cell_key


# ==========================================
# 工具函数
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


def get_cell_key(pos, resolution=1.0):
    """
    Go-Explore 核心：将连续位置映射到离散的 Archive Key
    resolution 决定了 Archive 的粒度
    """
    return tuple(np.round(np.array(pos) / resolution).astype(int))


def execute_translation_step(env, action_id, action_space, resolution=0.5):
    # (保持原有的物理执行逻辑不变)
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
        obs, reward, terminated, _, info = env.compute_scan_at_pos(target_pos)
        total_reward = reward

    return total_reward, terminated


# ==========================================
# Go-Explore 核心逻辑
# ==========================================
def select_node_from_archive(archive_values: List[CellNode]):
    """
    选择策略：
    Go-Explore 通常倾向于选择被访问次数较少（Promising）的节点。
    权重公式: weight = 1 / sqrt(visit_count + 1)
    """
    weights = [1.0 / np.sqrt(node.visit_count + 1) for node in archive_values]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    # 随机选择索引
    idx = np.random.choice(len(archive_values), p=probs)
    return archive_values[idx]


def run_go_explore_session(total_budget, seed, run_id):
    """
    Go-Explore 运行流程：
    1. 初始化 Archive
    2. Loop:
       a. Select: 从 Archive 中选择一个出发点
       b. Go: 恢复环境到该点
       c. Explore: 随机/贪心走出一段轨迹 (Horizon)
       d. Update: 如果发现新的 Cell 或更高的 Reward，更新 Archive
    """
    np.random.seed(seed)
    random.seed(seed)

    # 参数设置
    EXPLORE_HORIZON = 20  # 每次探索走多少步
    CELL_RESOLUTION = 1.0  # Archive 离散化粒度

    action_space = MacroActionSpace27(move_distance=0.5)

    with suppress_output():
        env = CoverageAviary(gui=False, obstacles=True, ctrl_freq=30, num_rays=120,
                             radar_radius=8.0, viz_rays_every=10, viz_points_every=20, grid_res=0.5)
    env.reset(seed=seed)

    # 根节点
    root_pos = env.pos[0].copy()
    root_key = get_cell_key(root_pos, CELL_RESOLUTION)
    root_snapshot = get_logical_snapshot(env)

    root_node = CellNode(
        node_id=0,
        parent_id=None,
        logical_snapshot=root_snapshot,
        cumulative_reward=0.0,
        global_step=0,
        last_state=np.concatenate([root_pos, env.rpy[0].copy()]),
        cell_key=root_key
    )

    # Archive: Dict[CellKey, CellNode]
    # 存储每个 Grid 对应的最优节点
    archive = {root_key: root_node}

    # 辅助存储所有节点以便回溯（因为 Archive 会被覆盖，我们需要保留生成树结构）
    # 在内存有限时，这里可以优化为只存 Archive 中的节点，但为了构建完整轨迹，我们存一个 lookup table
    # 注意：为了节省内存，可以只存父子关系，不存 snapshot
    all_nodes_lookup = {0: root_node}
    node_counter = 0

    global_sim_steps = 0
    max_reward_found = 0.0
    best_node_global = root_node

    while global_sim_steps < total_budget:
        # --- 1. Select ---
        current_node = select_node_from_archive(list(archive.values()))
        current_node.visit_count += 1

        # --- 2. Go (Restore) ---
        restore_env_logically(env, current_node)

        # 临时变量，用于这一轮 Explore
        traj_curr_node = current_node

        # --- 3. Explore (Horizon Loop) ---
        for _ in range(EXPLORE_HORIZON):
            if global_sim_steps >= total_budget:
                break

            # 策略：Robust Random (随机选择，但如果不撞墙最好)
            # 也可以替换为 Local Greedy
            actions_to_try = list(range(action_space.num_actions))
            random.shuffle(actions_to_try)

            step_reward = 0
            terminated = True
            chosen_action = -1

            # 尝试找到一个不撞墙的随机动作 (简单的碰撞避免)
            # 为了效率，我们只试探最多 5 次，找不到就强行停止或随机撞
            for attempt in range(5):
                a_id = actions_to_try[attempt]
                # 这是一个 "虚拟" 试探，这里简化为直接执行
                # 如果要严格 Go-Explore，应该 peek 一下，但 PyBullet peek 成本高
                # 这里我们假设：如果执行失败(撞墙)，本轮 Explore 提前结束

                # 这里的 execute 是真实的 step，如果撞墙 terminated=True
                # 如果我们要试错，需要先 save state。为了性能，我们直接走，撞了就换下一个 Select
                break  # 直接跳出，去执行真实动作逻辑

            # 真实执行 (这里简化为随机选一个动作执行，如果撞墙就 break Horizon)
            chosen_action = np.random.randint(action_space.num_actions)
            step_reward, terminated = execute_translation_step(env, chosen_action, action_space)
            global_sim_steps += 1

            if terminated:
                break  # 撞墙，结束这一轮探索

            # --- 4. Archive Update ---
            new_cum_reward = traj_curr_node.cumulative_reward + step_reward
            new_snapshot = get_logical_snapshot(env)
            new_pos = env.pos[0].copy()
            new_key = get_cell_key(new_pos, CELL_RESOLUTION)
            new_state_vec = np.concatenate([new_pos, env.rpy[0].copy()])

            node_counter += 1
            new_node = CellNode(
                node_id=node_counter,
                parent_id=traj_curr_node.node_id,
                logical_snapshot=new_snapshot,
                cumulative_reward=new_cum_reward,
                global_step=traj_curr_node.global_step + 1,
                action_from_parent=chosen_action,
                last_state=new_state_vec,
                last_reward=step_reward,
                cell_key=new_key
            )

            # 存入 lookup (用于最后回溯)
            all_nodes_lookup[node_counter] = new_node

            # 更新 Archive 规则：
            # 1. 如果是一个全新的区域 (new key)
            # 2. 如果是旧区域，但分数更高 (Higher Reward)
            if new_key not in archive or new_cum_reward > archive[new_key].cumulative_reward:
                archive[new_key] = new_node

                # 记录全局最优
                if new_cum_reward > max_reward_found:
                    max_reward_found = new_cum_reward
                    best_node_global = new_node

            # 指针步进
            traj_curr_node = new_node

    # 循环结束
    env.close()
    del env
    gc.collect()

    # ==========================================
    # 轨迹回溯 (Backtracking)
    # ==========================================
    # 从全局分最高的节点回溯到根节点，形成一条完整的轨迹
    trajectory_nodes = []
    curr = best_node_global

    while curr is not None:
        trajectory_nodes.append(curr)
        if curr.parent_id is None:
            break
        curr = all_nodes_lookup.get(curr.parent_id)

    # 因为是从尾到头，需要反转
    trajectory_nodes.reverse()

    # 格式化输出
    actions_list = []
    states_list = []
    rewards_list = []

    # Node 0 是初始状态
    states_list.append(trajectory_nodes[0].last_state)

    for i in range(1, len(trajectory_nodes)):
        node = trajectory_nodes[i]
        actions_list.append(node.action_from_parent)
        rewards_list.append(node.last_reward)
        states_list.append(node.last_state)

    traj_info = {
        "final_reward": best_node_global.cumulative_reward,
        "total_steps": len(actions_list),
        "action_sequence": np.array(actions_list, dtype=np.int32),
        "state_sequence": np.array(states_list, dtype=np.float32),
        "reward_sequence": np.array(rewards_list, dtype=np.float32),
        "final_pos": best_node_global.last_state[:3],
        "archive_size": len(archive)  # 额外统计信息
    }

    print(f'  [Run {run_id}] Go-Explore finished. Steps: {traj_info["total_steps"]}, '
          f'Score: {traj_info["final_reward"]:.2f}, Archive Size: {len(archive)}')

    return [traj_info]


def run_extraction_wrapper_goexplore(run_info, budget):
    run_id, seed = run_info
    try:
        trajs = run_go_explore_session(budget, seed, run_id)
        return trajs
    except Exception as e:
        print(f"  [Process {os.getpid()}] Run {run_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# ==========================================
# 主程序
# ==========================================
def run_parallel_extraction_goexplore(budget):
    # 配置
    BUDGET = budget
    NUM_PROCESSES = 10
    REPEATS = 20

    SAVE_FILE = f"../../data_pybullet/3d_trajectories/go_explore/go_explore_trajectories_budget_{BUDGET}.pkl"

    print("========================================")
    print(f"STARTING GO-EXPLORE TRAJECTORY EXTRACTION")
    print(f"Repeats: {REPEATS}, Simulation Steps Budget: {BUDGET}")
    print("========================================")

    tasks = [(i + 1, 888000 + i) for i in range(REPEATS)]  # 使用不同的 seed 范围区分 Greedy
    worker_func = partial(run_extraction_wrapper_goexplore, budget=BUDGET)

    start_time = time.time()

    with mp.Pool(processes=NUM_PROCESSES, maxtasksperchild=1) as pool:
        results = pool.map(worker_func, tasks)

    all_valid_data = []
    for trajs in results:
        all_valid_data.extend(trajs)

    duration = time.time() - start_time
    print(f"\n[Summary] Go-Explore extraction finished in {duration:.1f}s")
    print(f"Total trajectories collected: {len(all_valid_data)}")

    if len(all_valid_data) > 0:
        kept_trajs = sorted(all_valid_data, key=lambda t: t["final_reward"], reverse=True)
        os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
        with open(SAVE_FILE, 'wb') as f:
            pickle.dump(kept_trajs, f)
        print(f"SUCCESS: Saved {len(kept_trajs)} trajectories to {SAVE_FILE}")
    else:
        print("Warning: No trajectories collected.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    BUDGETS = [1000,2000,5000,10000,20000,50000]  # 总仿真步数预算

    for budget in BUDGETS:
        run_parallel_extraction_goexplore(budget)