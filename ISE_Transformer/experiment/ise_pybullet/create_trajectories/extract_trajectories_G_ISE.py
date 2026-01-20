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

# 引入环境
try:
    from ISE_Transformer.envs.coveragel_multi_room import CoverageAviary
except ImportError:
    print("Warning: Environment import failed.")


# 屏蔽输出
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
# 基础类 (保持不变)
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
            unit_vec = (vec / norm) * self.move_distance if norm > 0 else np.zeros(3)
            self.deltas.append(unit_vec)

    def get_displacement(self, action_id: int):
        return self.deltas[int(action_id)]


@dataclass
class SGEConfig:
    TOTAL_STEP_BUDGET: int
    NUM_BATCHES: int
    # 【新参数】分支因子：每次扩展保留最好的 K 个动作
    BRANCHING_FACTOR: int = 3
    CELL_SIZE: float = 0.5
    NUM_ELITES: int = 50


class CellNode:
    __slots__ = ['node_id', 'parent_id', 'logical_snapshot', 'cumulative_reward',
                 'global_step', 'times_selected', 'action_from_parent',
                 'last_state', 'last_reward']

    def __init__(self, node_id, parent_id, logical_snapshot, cumulative_reward, global_step,
                 action_from_parent=None, last_state=None, last_reward=0.0):
        self.node_id = node_id
        self.parent_id = parent_id
        self.logical_snapshot = logical_snapshot
        self.cumulative_reward = cumulative_reward
        self.global_step = global_step
        self.times_selected = 0
        self.action_from_parent = action_from_parent
        self.last_state = last_state
        self.last_reward = last_reward


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


def get_cell_key(pos, global_step, config):
    ix = int(pos[0] / config.CELL_SIZE)
    iy = int(pos[1] / config.CELL_SIZE)
    iz = int(pos[2] / config.CELL_SIZE)
    it = int(global_step)
    return (ix, iy, iz, it)


def execute_translation_step(env, action_id, action_space, resolution=0.5):
    # (此处保持原有的物理检测逻辑，为了节省篇幅省略，请直接复用之前的代码)
    # 必须包含：高度检查、射线检测、碰撞检测、compute_scan_at_pos
    # ...
    # 假设你已经复制了之前的 execute_translation_step 代码

    # --- 简化版占位 (请替换回你的完整逻辑) ---
    displacement = action_space.get_displacement(action_id)
    target_pos = env.pos[0] + displacement
    if target_pos[2] < 0.5 or target_pos[2] > 3.5: return 0.0, True

    # 简单碰撞模拟
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], target_pos, [0, 0, 0, 1], physicsClientId=env.CLIENT)
    p.performCollisionDetection(physicsClientId=env.CLIENT)
    if len(p.getContactPoints(bodyA=env.DRONE_IDS[0], physicsClientId=env.CLIENT)) > 0: return 0.0, True

    obs, reward, terminated, _, info = env.compute_scan_at_pos(target_pos)
    return reward, terminated
    # ------------------------------------


# ==========================================
# 【核心修改】贪心分支扩展逻辑
# ==========================================
def expand_node_greedily(env, parent_node, action_space, branching_factor):
    """
    在当前 parent_node 的基础上，试探所有动作，
    返回 Top-K (branching_factor) 个最好的可行结果。
    """
    candidates = []
    computing_count = 0

    # 随机打乱顺序，保证如果分数一样，取向具有随机性
    all_actions = list(range(action_space.num_actions))
    random.shuffle(all_actions)

    for action_id in all_actions:
        # 1. 每次试探前恢复状态
        restore_env_logically(env, parent_node)
        computing_count += 1

        # 2. 执行试探
        step_reward, terminated = execute_translation_step(env, action_id, action_space)

        if not terminated:
            # 记录结果，稍后排序
            # 我们需要保存这一步结束后的 snapshot，因为如果它被选中，我们需要根据这个生成 Node
            candidates.append({
                'action_id': action_id,
                'reward': step_reward,
                'snapshot': get_logical_snapshot(env),
                'state_vec': np.concatenate([env.pos[0].copy(), env.rpy[0].copy()])
            })

    # 3. 排序并取 Top-K
    # 按 reward 从大到小排
    candidates.sort(key=lambda x: x['reward'], reverse=True)

    # 返回前 K 个，如果不足 K 个则全返回
    return candidates[:branching_factor],computing_count


# ==========================================
# 混合 SGE 主逻辑
# ==========================================
def select_node_weighted(node_pool):
    """从 Pool 中根据分数加权选择一个父节点"""
    if not node_pool: return None
    rewards = np.array([n.cumulative_reward for n in node_pool])
    # 简单的 Softmax 变体或直接归一化
    if rewards.max() == rewards.min():
        probs = np.ones_like(rewards) / len(rewards)
    else:
        norm = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-6)
        scores = np.exp(norm * 3.0)  # Temperature = 3.0, 越强者越容易被选中去扩展
        probs = scores / np.sum(scores)

    return np.random.choice(node_pool, p=probs)


def run_hybrid_session(num_batches, total_budget, seed, run_id):
    np.random.seed(seed)
    random.seed(seed)

    # 配置：branching_factor=3 表示每一步我们保留最好的3个动作，变成3个新节点
    config = SGEConfig(TOTAL_STEP_BUDGET=total_budget, NUM_BATCHES=num_batches, BRANCHING_FACTOR=3)

    action_space = MacroActionSpace27(move_distance=0.5)

    global_archive = {}
    grid_to_node_id = {}
    node_counter = 0

    # 初始化环境
    with suppress_output():
        env = CoverageAviary(gui=False, obstacles=True, ctrl_freq=30, num_rays=120,
                             radar_radius=8.0, viz_rays_every=10, viz_points_every=20, grid_res=0.5)
    env.reset(seed=seed)

    # 根节点
    root_node = CellNode(
        node_id=node_counter,
        parent_id=None,
        logical_snapshot=get_logical_snapshot(env),
        cumulative_reward=0.0,
        global_step=0,
        last_state=np.concatenate([env.pos[0].copy(), env.rpy[0].copy()])
    )
    global_archive[node_counter] = root_node
    node_counter += 1

    # 初始 Batch 种子
    current_batch_seeds = [root_node]
    steps_consumed = 0

    STEPS_PER_BATCH = total_budget // num_batches

    for batch_idx in range(config.NUM_BATCHES):
        # print(f"Batch {batch_idx} | Seeds: {len(current_batch_seeds)} | Archive: {len(global_archive)}")

        # 每一轮开始，重置环境以释放内存
        env.close()
        with suppress_output():
            env = CoverageAviary(gui=False, obstacles=True, ctrl_freq=30, num_rays=120,
                                 radar_radius=8.0, viz_rays_every=10, viz_points_every=20, grid_res=0.5)
        env.reset(seed=seed)

        batch_pool = list(current_batch_seeds)
        batch_steps = 0

        while batch_steps < STEPS_PER_BATCH:
            # 1. 选择一个父节点 (偏向高分节点)
            parent_node = select_node_weighted(batch_pool)
            if not parent_node: break

            # 2. 【核心】贪心扩展 Top-K
            # 这里不需要 loop horizon，因为每次扩展 Top-K 本身就是一种广度优先推进

            # 这里的代价是：每次扩展都要跑 27 次物理仿真。
            # 为了计算 Budget，我们认为这次操作消耗了 config.BRANCHING_FACTOR 个步数
            # (或者你也可以按实际物理仿真次数算，这里按生成的有效节点算比较简单)

            top_k_results,computing_count = expand_node_greedily(env, parent_node, action_space, config.BRANCHING_FACTOR)
            batch_steps =batch_steps + computing_count-1
            if len(top_k_results) == 0:
                # 这是一个死节点，以后别选它了 (简单处理：从 pool 移除)
                if parent_node in batch_pool:
                    batch_pool.remove(parent_node)
                continue

            for res in top_k_results:
                new_cum_reward = parent_node.cumulative_reward + res['reward']
                new_pos = res['state_vec'][:3]
                new_grid_key = get_cell_key(new_pos, parent_node.global_step + 1, config)

                # 3. 空间竞争逻辑 (同 SGE)
                is_better = False
                if new_grid_key not in grid_to_node_id:
                    is_better = True
                else:
                    existing_node = global_archive[grid_to_node_id[new_grid_key]]
                    if new_cum_reward > existing_node.cumulative_reward:
                        is_better = True

                if is_better:
                    new_node = CellNode(
                        node_id=node_counter,
                        parent_id=parent_node.node_id,
                        logical_snapshot=res['snapshot'],
                        cumulative_reward=new_cum_reward,
                        global_step=parent_node.global_step + 1,
                        action_from_parent=res['action_id'],
                        last_state=res['state_vec'],
                        last_reward=res['reward']
                    )

                    global_archive[node_counter] = new_node
                    grid_to_node_id[new_grid_key] = node_counter
                    batch_pool.append(new_node)
                    node_counter += 1

                    batch_steps += 1  # 计入消耗

        steps_consumed += batch_steps

        # --- Batch 结束：精英保留 ---
        # 选出全 Archive 中最好的节点，作为下一轮的种子
        elites = heapq.nlargest(config.NUM_ELITES, global_archive.values(),
                                key=lambda n: n.cumulative_reward)

        # 这里的剪枝逻辑可以简化：下一轮只允许从精英继续生长
        current_batch_seeds = elites

        # (内存清理逻辑同 SGE，此处略，建议保留原始代码中的 keep_ids 逻辑)

    env.close()

    # --- 提取轨迹 ---
    # 逻辑同 SGE，从 global_archive 找最高分的叶子回溯
    final_candidates = heapq.nlargest(50, global_archive.values(), key=lambda n: n.cumulative_reward)
    valid_trajectories = []

    for elite_node in final_candidates:
        actions_list = []
        states_list = []
        rewards_list = []
        curr = elite_node
        while curr is not None:
            states_list.insert(0, curr.last_state)
            if curr.parent_id is not None:
                actions_list.insert(0, curr.action_from_parent)
                rewards_list.insert(0, curr.last_reward)
            curr = global_archive.get(curr.parent_id)

        valid_trajectories.append({
            "final_reward": elite_node.cumulative_reward,
            "total_steps": len(actions_list),
            "action_sequence": np.array(actions_list),
            "state_sequence": np.array(states_list),
            "reward_sequence": np.array(rewards_list)
        })

    print(f"Final reward: {valid_trajectories[0]['final_reward']}, Steps: {valid_trajectories[0]['total_steps']}")

    return valid_trajectories


# ==========================================
# 并行入口 (复用之前结构)
# ==========================================
def run_worker_hybrid(info, batches, budget):
    run_id, seed = info
    try:
        return run_hybrid_session(batches, budget, seed, run_id)
    except Exception as e:
        print(f"Error: {e}")
        return []


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # 示例运行
    # 注意：因为这里计算量变大了 (每一步算27次)，建议调小 Budget 或 Batches
    BATCHES = [1,2,4,8]
    BUDGETS = [1000,2000,5000,10000,20000,50000]
    NUM_PROCESSES = 10
    REPEATS = 20  # 跑 50 条 Greedy 轨迹

    for j in BATCHES:
        for budget in BUDGETS:
            SAVE_FILE = f"../../data_pybullet/3d_trajectories/hybrid/hybrid_trajectories_batches_{j}_budget_{budget}.pkl"

            print("========================================")
            print(f"STARTING Hybrid TRAJECTORY EXTRACTION")
            print(f"Repeats: {REPEATS}, Max Steps: {budget} , Batches: {j}")
            print("========================================")

            start_time = time.time()

            tasks = [(i + 1, 999000 + i) for i in range(REPEATS)]  # 使用不同的 seed 范围

            worker_func = partial(run_worker_hybrid, batches=j, budget=budget)

            with mp.Pool(processes=NUM_PROCESSES, maxtasksperchild=1) as pool:
                results = pool.map(worker_func, tasks)

                # 4. 汇总
            all_valid_data = []
            for trajs in results:
                all_valid_data.extend(trajs)

            duration = time.time() - start_time
            print(f"\n[Summary] hydrid extraction finished in {duration:.1f}s")
            print(f"Total trajectories collected: {len(all_valid_data)}")

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



