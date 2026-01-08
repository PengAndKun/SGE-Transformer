
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

#优化环境创建

# 引入依赖
from SGE_Transformer.envs.coverage_visibility_pointcloud_aviary_optimized_add_Control import CoverageAviary
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
    """27 macro-actions: (dx,dy,dz) in {-1,0,1}^3 mapped to ids 0..26.

    Sampling policy (per user request):
    - No preference for repeating the previous action (removes '连续选择' bias)
    - Supports excluding actions already used in the current rollout segment
    """

    def __init__(self,move_distance=1.5):
        self.num_actions = 27
        self.move_distance = move_distance  # 每次平移的固定物理距离
        # Precompute deltas for each action id (lexicographic)
        self.deltas = []
        for a in range(self.num_actions):
            dx = (a // 9) - 1
            dy = ((a % 9) // 3) - 1
            dz = (a % 3) - 1

            vec = np.array([dx, dy, dz], dtype=float)
            norm = np.linalg.norm(vec)
            # 处理 (0,0,0) 动作，即原地停顿或微小步进
            if norm > 0:
                unit_vec = (vec / norm) * self.move_distance
            else:
                unit_vec = np.zeros(3)
            self.deltas.append(unit_vec)

    def get_displacement(self, action_id: int):
        return self.deltas[int(action_id)]

    def get_delta(self, action_id: int):
        return self.deltas[int(action_id)]

    def sample(self, prev_action=None, exclude_actions=None):
        """Uniformly sample an action id.

        prev_action: if provided, will avoid sampling the same action consecutively.
        exclude_actions: optional set/list of action ids to avoid (e.g., already used within horizon).
        """
        exclude = set(exclude_actions) if exclude_actions is not None else set()
        if prev_action is not None:
            exclude.add(int(prev_action))

        candidates = [a for a in range(self.num_actions) if a not in exclude]

        # Fallback: if everything excluded, allow all but still avoid immediate repeat if possible
        if len(candidates) == 0:
            candidates = list(range(self.num_actions))
            if prev_action is not None and len(candidates) > 1:
                candidates.remove(int(prev_action))

        return int(np.random.choice(candidates))

    def is_displacement_exceed(self, pos_a, pos_b, strict=True):
        """
        判断两个位置之间的位移是否超过 move_distance

        Parameters
        ----------
        pos_a : array-like, shape (3,)
            起始位置 (x, y, z)
        pos_b : array-like, shape (3,)
            目标位置 (x, y, z)
        strict : bool
            True  -> 判断 |Δ| > move_distance
            False -> 判断 |Δ| >= move_distance

        Returns
        -------
        bool
            True  : 位移超过（或达到）move_distance
            False : 位移在允许范围内
        """
        pos_a = np.asarray(pos_a, dtype=float)
        pos_b = np.asarray(pos_b, dtype=float)

        displacement = np.linalg.norm(pos_b - pos_a)

        if strict:
            return displacement > self.move_distance
        else:
            return displacement >= self.move_distance

@dataclass
class SGEConfig:
    TOTAL_STEP_BUDGET: int
    NUM_BATCHES: int
    EXPLORE_HORIZON: int = 1
    CELL_SIZE: float = 0.2
    NUM_ELITES: int = 20


class CellNode:
    __slots__ = ['node_id', 'parent_id', 'logical_snapshot', 'cumulative_reward',
                 'global_step', 'times_selected', 'action_from_parent',
                 'last_state', 'last_reward']

    def __init__(self, node_id, parent_id, logical_snapshot, cumulative_reward, global_step,
                 action_from_parent=None, last_state=None, last_reward=0.0):
        self.node_id = node_id
        self.parent_id = parent_id
        # 存储逻辑快照：包含位置和覆盖掩码副本，不依赖物理引擎 ID
        self.logical_snapshot = logical_snapshot
        self.cumulative_reward = cumulative_reward
        self.global_step = global_step
        self.times_selected = 0
        self.action_from_parent = action_from_parent
        self.last_state = last_state
        self.last_reward = last_reward


# ==========================================
# 逻辑状态恢复函数
# ==========================================
def restore_env_logically(env, node):
    """
    手动将环境恢复到节点记录的状态
    """
    # 1. 恢复无人机物理位置
    env.coverage_grid = node.logical_snapshot['grid'].copy()
    # 2. 物理位置同步
    pos = node.logical_snapshot['pos']
    rpy = node.logical_snapshot['rpy']
    quat = p.getQuaternionFromEuler(rpy)
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], pos, quat, physicsClientId=env.CLIENT)

    # 强制刷新类内部的运动学缓存
    env._updateAndStoreKinematicInformation()
    # 可视化同步：如果你在 GUI 模式下，需要重绘蓝色点阵
    if env.GUI:
        env._redraw_full_coverage()


def get_logical_snapshot(env):
    """
    提取不依赖于物理引擎 ID 的逻辑状态
    """
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


def select_from_batch_pool(node_pool: List[CellNode]):
    """
    只从当前批次中持有有效快照的节点池中选择
    """
    if not node_pool:
        return None

    # 1. 提取指标
    # 注意：这里的 candidates 仅限于本 Batch 允许探索的节点
    visits = np.array([n.times_selected for n in node_pool], dtype=np.float32)
    rewards = np.array([n.cumulative_reward for n in node_pool], dtype=np.float32)

    # 2. 归一化奖励 (Exploitation)
    if rewards.max() > rewards.min():
        norm_rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())
    else:
        norm_rewards = np.zeros_like(rewards)

    # 3. 计算新颖度分数 (Exploration)
    # 对于本轮新产生的节点，visits 通常很小，给予更高的权重
    exploration_score = 1.0 / (np.log1p(visits) + 1.0)

    # 4. 综合得分并采样
    # 增加温度系数 T=0.5 使选择更集中在优良节点上
    scores = (1.0 * norm_rewards + 2.0 * exploration_score) / 0.5
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)

    return np.random.choice(node_pool, p=probs)

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


def execute_translation_step(env, action_id, action_space, resolution=0.5):
    """
    执行平移：从 A 点到 B 点，并在路径上按 resolution 采样计算覆盖
    """

    # 定义高度限制（根据你的场景调整，例如 0.5m 到 5.0m）
    MIN_HEIGHT = 0.5
    MAX_HEIGHT = 3.5

    terminated = False
    total_reward = 0.0

    displacement = action_space.get_displacement(action_id)
    start_pos = np.array(env.pos[0])  # 获取当前无人机位置
    target_pos = start_pos + displacement
    # print(f"[平移] 尝试从 {start_pos} 移动到 {target_pos}")

    if target_pos[2] < MIN_HEIGHT or target_pos[2] > MAX_HEIGHT:
        terminated = True
        # print(f"Terminated: Altitude out of bounds ({target_pos[2]:.2f})")
        return 0.0, terminated

        # 实际移动向量
    actual_vec = target_pos - start_pos
    dist = np.linalg.norm(actual_vec)

    if dist > 1e-6:
        # 增加采样密度确保不会“跳过”薄墙
        num_samples = max(int(dist / resolution), 2)

        for i in range(1, num_samples + 1):
            inter_pos = start_pos + (i / num_samples) * actual_vec

            # --- 3. 穿墙检测 (Raycasting) ---
            # 从上一个采样点到当前采样点发射一条射线，看是否撞击障碍物
            prev_inter_pos = start_pos + ((i - 1) / num_samples) * actual_vec
            ray_test = p.rayTest(prev_inter_pos, inter_pos, physicsClientId=env.CLIENT)

            # 如果射线检测到物体且不是无人机本身
            if ray_test[0][0] != -1 and ray_test[0][0] != env.DRONE_IDS[0]:
                terminated = True
                break

            # --- 4. 物理层更新与接触检测 ---
            p.resetBasePositionAndOrientation(env.DRONE_IDS[0], inter_pos, [0, 0, 0, 1], physicsClientId=env.CLIENT)
            # 强制物理引擎同步碰撞状态
            p.performCollisionDetection(physicsClientId=env.CLIENT)

            # 检查是否有物体正在重叠（接触检测）
            contacts = p.getContactPoints(bodyA=env.DRONE_IDS[0], physicsClientId=env.CLIENT)
            if len(contacts) > 0:
                terminated = True
                # print(f"Terminated: Collision detected at {inter_pos}")
                break

            # 5. 计算覆盖奖励 (仅在未碰撞时)
            obs, reward, terminated, _, info = env.compute_scan_at_pos(inter_pos)
            # print(f"[覆盖] 尝试覆盖 {inter_pos} | 奖励: {reward:.2f}")
            if terminated:
                break
            total_reward += reward

    else:
        # 原地动作处理
        obs, reward, terminated, _, info = env.compute_scan_at_pos(target_pos)
        total_reward = reward

    return total_reward, terminated

# ==========================================
# 实验核心逻辑 (带阈值筛选)
# ==========================================
def run_extraction_session(num_batches, total_budget, seed, run_id, score_threshold=99.0):
    np.random.seed(seed)
    random.seed(seed)

    config = SGEConfig(TOTAL_STEP_BUDGET=total_budget, NUM_BATCHES=num_batches)
    STEPS_LIMIT_PER_BATCH = total_budget // num_batches


    action_space = MacroActionSpace27(move_distance=0.5)

    global_archive = {}
    grid_to_node_id = {}   # {(x,y,z,t): node_id} -> 用于处理网格内的路径竞争
    node_counter = 0

    # 初始节点准备
    with suppress_output():
        temp_env = CoverageAviary(gui=False, obstacles=True,ctrl_freq=30,num_rays=120,
        radar_radius=8.0, viz_rays_every=10,viz_points_every=20,grid_res=0.5,)
    temp_env.reset(seed=seed)
    root_state = np.concatenate([temp_env.pos[0].copy(), temp_env.rpy[0].copy()])
    root_node = CellNode(
        node_id=node_counter,
        parent_id=None,
        logical_snapshot=get_logical_snapshot(temp_env),
        cumulative_reward=0.0,
        global_step=0,
        last_state=np.concatenate([temp_env.pos[0], temp_env.rpy[0]])
    )
    global_archive[node_counter] = root_node
    grid_to_node_id[get_cell_key(temp_env.pos[0], 0, config)] = node_counter
    node_counter += 1
    current_batch_seeds = [root_node]
    temp_env.close()

    # --- SGE Main Loop ---
    for batch_idx in range(config.NUM_BATCHES):
        # batch_pool 存储本轮可供选择的节点
        print(f"[Batch {batch_idx}] len pool: {len(current_batch_seeds)} all len {len(global_archive)} | Best score: {max(s.cumulative_reward for s in current_batch_seeds):.2f}")

        with suppress_output():
            env = CoverageAviary(gui=False, obstacles=True, ctrl_freq=30, num_rays=120,
                                 radar_radius=8.0, viz_rays_every=10, viz_points_every=20, grid_res=0.5, )
        env.reset(seed=seed)

        batch_pool = list(current_batch_seeds)
        steps_count = 0

        while steps_count < STEPS_LIMIT_PER_BATCH:
            # 这里的选择逻辑需要传入 node_id 列表
            curr_node = select_from_batch_pool(batch_pool)
            if not curr_node: break

            curr_node.times_selected += 1

            # --- 关键：使用逻辑恢复而非物理恢复 ---
            restore_env_logically(env, curr_node)

            parent_node = curr_node
            for _ in range(config.EXPLORE_HORIZON):
                if steps_count >= STEPS_LIMIT_PER_BATCH: break

                action_id = action_space.sample()
                step_reward, terminated = execute_translation_step(env, action_id, action_space)
                steps_count += 1
                if terminated: break

                new_pos = env.pos[0].copy()
                new_grid_key = get_cell_key(new_pos, parent_node.global_step + 1, config)
                new_cum_reward = parent_node.cumulative_reward + step_reward

                # 判断是否是该网格中更好的路径
                is_better = False
                if new_grid_key not in grid_to_node_id:
                    is_better = True
                else:
                    existing_node = global_archive[grid_to_node_id[new_grid_key]]
                    new_node_step=parent_node.global_step + 1
                    if (new_cum_reward > existing_node.cumulative_reward) or \
                            (np.isclose(new_cum_reward, existing_node.cumulative_reward) and
                             (new_node_step < existing_node.global_step)):
                        is_better = True

                if is_better:
                    # 创建新节点，分配唯一 ID
                    # print(f"{batch_idx} [新节点] 创建新节点 {node_counter} | 父节点: {parent_node.node_id} | 步骤: {parent_node.global_step} | 奖励: {new_cum_reward:.2f}")
                    new_node = CellNode(
                        node_id=node_counter,
                        parent_id=parent_node.node_id,  # 关键：指向父节点 ID
                        logical_snapshot=get_logical_snapshot(env),
                        cumulative_reward=new_cum_reward,
                        global_step=parent_node.global_step + 1,
                        action_from_parent=action_id,
                        last_state=np.concatenate([new_pos, env.rpy[0].copy()]),
                        last_reward=step_reward
                    )
                    global_archive[node_counter] = new_node
                    grid_to_node_id[new_grid_key] = node_counter  # 更新索引，但不删除旧 node
                    batch_pool.append(new_node)

                    parent_node = new_node
                    node_counter += 1

        # --- 内存管理：剪枝 ---
        # A. 选出本批次的精英（用于下一批次的种子）
        elites = heapq.nlargest(config.NUM_ELITES, global_archive.values(),
                                key=lambda n: (n.cumulative_reward, -n.global_step))

        # B. 追溯所有必须保留的节点 ID (精英 + 精英的祖先)
        keep_ids = set()
        for e_node in elites:
            curr_id = e_node.node_id
            while curr_id is not None:
                if curr_id in keep_ids: break
                keep_ids.add(curr_id)
                curr_id = global_archive[curr_id].parent_id

        # C. 【关键】物理重建 Archive，丢弃所有冗余节点对象
        new_global_archive = {nid: global_archive[nid] for nid in keep_ids}

        # D. 【关键】物理重构 Grid 索引，确保它只指向存在的节点
        new_grid_to_node_id = {k: v for k, v in grid_to_node_id.items() if v in keep_ids}

        # E. 释放旧引用，允许 GC 回收
        global_archive.clear()
        grid_to_node_id.clear()
        global_archive = new_global_archive
        grid_to_node_id = new_grid_to_node_id


        # 4. 继承种子
        current_batch_seeds = elites
        env.close()
        del env
        gc.collect()


    # === [优化版] 轨迹回溯与提取 ===

    # 1. 确定候选节点：从全局存档中找出得分最高的前 50 个独立节点
    # 注意：即使 Cell 坐标相同，只要 node_id 不同，它们就是不同的物理路径
    all_final_nodes = list(global_archive.values())
    candidates = heapq.nlargest(50, all_final_nodes, key=lambda n: (n.cumulative_reward, -n.global_step))

    valid_trajectories = []

    for elite_node in candidates:
        if elite_node.cumulative_reward > score_threshold:

            # --- 核心回溯逻辑 ---
            # 初始化临时容器
            actions_list = []
            states_list = []
            rewards_list = []

            curr_node = elite_node

            # 从叶子节点一路向上爬到根节点 (parent_id 为 None)
            while curr_node is not None:
                # 插入当前步的状态 (插入到最前面)
                states_list.insert(0, curr_node.last_state)

                # 如果不是根节点，则存在“通往该节点的动作”和“该步获得的奖励”
                if curr_node.parent_id is not None:
                    actions_list.insert(0, curr_node.action_from_parent)
                    rewards_list.insert(0, curr_node.last_reward)

                # 移动到父节点
                curr_node = global_archive.get(curr_node.parent_id)

            # --- 封装轨迹信息 ---
            # 此时：
            # states_list 长度为 N+1 (包含初始位置)
            # actions_list 长度为 N
            # rewards_list 长度为 N
            traj_info = {
                "final_reward": elite_node.cumulative_reward,
                "total_steps": len(actions_list),
                "action_sequence": np.array(actions_list, dtype=np.int32),
                "state_sequence": np.array(states_list, dtype=np.float32),
                "reward_sequence": np.array(rewards_list, dtype=np.float32),
                "final_pos": elite_node.last_state[:3]  # 优化：不再依赖 snapshot 里的 pos
            }
            valid_trajectories.append(traj_info)

    # 去重逻辑 (可选)：有时候不同的 Node 可能会回溯出极其相似的路径
    # 这里简单直接返回所有符合条件的

    # 显式清空字典
    global_archive.clear()
    grid_to_node_id.clear()

    # 删除引用
    del global_archive
    del grid_to_node_id
    del all_final_nodes
    del candidates
    gc.collect()
    return valid_trajectories

def run_extraction_wrapper(run_info, batches, budget, score_threshold):
    """
    单次实验的包装器，供多进程 Pool 调用
    """
    run_id, seed = run_info
    try:
        # 运行你之前的核心函数
        trajs = run_extraction_session(batches, budget, seed, run_id, score_threshold)
        print(f"  [Process {os.getpid()}] Run {run_id} finished. Found {len(trajs)} trajs.")
        return trajs
    except Exception as e:
        print(f"  [Process {os.getpid()}] Run {run_id} failed: {e}")
        return []

# ==========================================
# 主程序
# ==========================================
def run_parallel_extraction(batches,budget):
    # 配置
    BATCHES = batches
    BUDGET = budget
    NUM_PROCESSES = 10      # [功能增加] 指定同时运行的进程数

    REPEATS = 20  # [修改] 跑 30 次
    SCORE_THRESHOLD = 0.0  # [修改] 分数阈值
    PERCENTILE_KEEP = 0  # 可调：90 表示保留 ≥90分位（前10%）
    TOP_K_TRAJS = 500
    SAVE_FILE = f"../../data_pybullet/3d_trajectories/ise/elite_trajectories_v1_multiprocessing_{BATCHES}_{BUDGET}.pkl"

    all_valid_data = []
    total_valid_count = 0

    print("========================================")
    print(f"STARTING HIGH-QUALITY TRAJECTORY EXTRACTION")
    print(f"Repeats: {REPEATS}, Threshold: > {SCORE_THRESHOLD}")
    print("========================================")

    # 1. 准备任务参数 (Run ID 和 对应的 Seed)
    # 确保每个任务的种子都是唯一的
    tasks = [(i + 1, BUDGET + (BATCHES * 1000) + i) for i in range(REPEATS)]

    # 2. 创建偏函数以固定其他配置参数
    worker_func = partial(
        run_extraction_wrapper,
        batches=BATCHES,
        budget=BUDGET,
        score_threshold=SCORE_THRESHOLD
    )

    start_time = time.time()

    # 3. 启动进程池
    # maxtasksperchild=1 极其重要：每个任务执行完后销毁进程并重建，彻底解决内存累积
    with mp.Pool(processes=NUM_PROCESSES, maxtasksperchild=1) as pool:
        # 使用 imap_unordered 可以一边运行一边获取结果，也可以用 map 直接获取全部
        results = pool.map(worker_func, tasks)

    # 4. 汇总所有进程的结果
    all_valid_data = []
    for trajs in results:
        all_valid_data.extend(trajs)

    duration = time.time() - start_time
    print(f"\n[Summary] Parallel extraction finished in {duration:.1f}s")
    print(f"Total trajectories collected: {len(all_valid_data)}")

    # 5. 后续筛选逻辑 (保持不变)
    if len(all_valid_data) > 0:
        scores = [t["final_reward"] for t in all_valid_data]
        threshold = float(np.percentile(scores, PERCENTILE_KEEP))
        kept_trajs = sorted([t for t in all_valid_data if t["final_reward"] >= threshold],
                            key=lambda t: (t["final_reward"], -t["total_steps"]), reverse=True)

        with open(SAVE_FILE, 'wb') as f:
            pickle.dump(kept_trajs, f)
        print(f"SUCCESS: Saved {len(kept_trajs)} elite trajectories to {SAVE_FILE}")
    else:
        print("Warning: No trajectories collected.")

    


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    BATCHES = [1,2,4,5,8,10]
    BUDGET = [1000,2000,5000,10000,20000,50000,100000]
    for batches in BATCHES:
        for budget in BUDGET:
            run_parallel_extraction(batches,budget)