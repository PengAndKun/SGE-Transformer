
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
    STEPS_PER_ACTION: int = 1  # (unused with macro_step)nt = 15
    CELL_SIZE: float = 0.2
    NUM_ELITES: int = 15


class CellNode:
    def __init__(self, key, snapshot, cumulative_reward, global_step, parent_key=None, action_from_parent=None
                , action_bytes=b"",state_history=None, reward_history=None):
        self.key = key
        self.snapshot = snapshot
        self.cumulative_reward = cumulative_reward
        self.global_step = global_step
        self.times_selected = 0
        self.parent_key = parent_key
        self.action_from_parent = action_from_parent

        self.action_bytes = action_bytes

        self.state_history = state_history if state_history is not None else []

        self.reward_history = reward_history if reward_history is not None else []



def get_cell_key(pos, global_step, config):
    ix = int(pos[0] / config.CELL_SIZE)
    iy = int(pos[1] / config.CELL_SIZE)
    iz = int(pos[2] / config.CELL_SIZE)
    it = int(global_step)
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

    with suppress_output():
        env = CoverageAviary(gui=False, obstacles=True, ctrl_freq=30,num_rays=120,
        radar_radius=8.0, viz_rays_every=10,viz_points_every=20,grid_res=0.5,)

    action_space = MacroActionSpace27(move_distance=0.5)
    global_archive = {}

    obs, info = env.reset(seed=seed)
    action_space.sample()

    start_snapshot = env.get_snapshot()
    start_key = get_cell_key(start_snapshot['pos'], 0, config)
    start_state = np.concatenate([start_snapshot['pos'], [0, 0, 0]])  # 假设初始角度全0
    root_node = CellNode(start_key, start_snapshot, 0.0, global_step=0,action_bytes=b"",state_history=[start_state],reward_history=[])
    global_archive[start_key] = root_node
    current_batch_seeds = [root_node]

    # --- SGE Main Loop ---
    for batch_idx in range(config.NUM_BATCHES):
        # print(f"[Batch {batch_idx}]")
        batch_pool = {seed.key: seed for seed in current_batch_seeds}
        current_batch_steps_counter = 0

        while current_batch_steps_counter < STEPS_LIMIT_PER_BATCH:

            current_node = select_cell_advanced(batch_pool)
            # print(f"[Batch {batch_idx}] Selected node {current_node.key} | Steps {current_batch_steps_counter} / {STEPS_LIMIT_PER_BATCH}")
            if current_node is None: break
            current_node.times_selected += 1

            # print(f"[Batch- {batch_idx}] Restoring snapshot {current_node.key} pos {current_node.snapshot['pos']}")
            # print(f"[Batch- {batch_idx}] pre pos: {env.pos[0]}")

            env.restore_snapshot(current_node.snapshot)
            # print(f"[Batch- {batch_idx}] Current pos: {env.pos[0]}")
            current_cum_reward = current_node.cumulative_reward
            # print(f"[Batch {batch_idx}] Selected node {current_node.key} current_cum_reward: {current_cum_reward} | Steps {current_batch_steps_counter} / {STEPS_LIMIT_PER_BATCH}")
            current_global_step = current_node.global_step
            temp_traj_node = current_node
            # print(f"[Batch- {batch_idx}] pre pos: {env.pos[0]} | pos {current_node.snapshot['pos']}  |")


            for _ in range(config.EXPLORE_HORIZON):
                if current_batch_steps_counter >= STEPS_LIMIT_PER_BATCH: break
                # if current_batch_steps_counter % 100 == 0:
                #     print(f"[Batch {batch_idx}] Steps {current_batch_steps_counter} / {STEPS_LIMIT_PER_BATCH}")

                action_id = action_space.sample()

                # Execute one macro-action (single logical step)
                pre_pos = env.pos[0].copy()

                pos_error = np.linalg.norm(temp_traj_node.state_history[-1][:3] - pre_pos[:3])
                if pos_error > 1e-4:
                    print(f"[Error] pre_pos {pre_pos} pos_error {pos_error} state_history {temp_traj_node.state_history}")
                    print(f"[Error] pos_error {pos_error}  ")
                step_reward, terminated = execute_translation_step(env, action_id, action_space)                # print(f"[Batch {batch_idx}] Step {current_batch_steps_counter} / {STEPS_LIMIT_PER_BATCH} | Action {action_id} | Reward {reward}")
                current_pos = env.pos[0].copy()
                if np.linalg.norm(pre_pos - current_pos) < 1e-6:
                    terminated = True
                if np.linalg.norm(pre_pos - current_pos) >0.55:
                    print(f"[Batch {batch_idx}] Warning: Action {action_id} moved the drone too far.")
                    terminated = True
                current_global_step += 1
                current_batch_steps_counter += 1
                if terminated :
                    break
                current_cum_reward += step_reward

                    # Even if truncated (didn't fully converge), we still keep the transition;
                    # archive logic below will handle it. Stop the horizon early on termination.

                new_key = get_cell_key(current_pos, current_global_step, config)
                should_add = False
                if new_key not in global_archive:
                    should_add = True
                elif current_cum_reward > global_archive[new_key].cumulative_reward:
                    should_add = True

                if should_add:



                    new_snapshot = env.get_snapshot()
                    new_bytes = temp_traj_node.action_bytes + bytes([int(action_id)])
                    # if batch_idx<2:
                    #     print(
                    #         f"[Batch {batch_idx}] Adding node {new_key} with reward {current_cum_reward} envstep {current_global_step}")
                    #     print(
                    #         f"[Batch {batch_idx}] Node {new_key} action_id {action_id} pre_pos {pre_pos} pos {current_pos}")
                    #     print(
                    #         f"[Batch {batch_idx}] Node {new_key} action_bytes {np.array(list(new_bytes), dtype=np.int32)}")
                    #     print(f"temp_traj_node.state_history {temp_traj_node.state_history[-1]}")



                    # 获取当前无人机完整的 6DOF 状态 (Pos + RPY)
                    # 注意：gym-pybullet-drones 的 env.pos 和 env.rpy 通常是 list of arrays
                    current_full_state = np.concatenate([env.pos[0].copy(), env.rpy[0].copy()])
                    # 继承父节点的状态列表并添加当前状态
                    new_state_history = temp_traj_node.state_history + [current_full_state]
                    new_reward_history = temp_traj_node.reward_history + [step_reward]


                    new_node = CellNode(
                        new_key, new_snapshot, current_cum_reward,
                        global_step=current_global_step,
                        parent_key=temp_traj_node.key,
                        action_from_parent=action_id,
                        action_bytes=new_bytes,
                        state_history=new_state_history,  # 关键：保存状态序列
                        reward_history = new_reward_history  # 存入新节点
                    )

                    # print(f'current_global_step :{current_global_step} pos {new_node.snapshot["pos"]}')
                    # print(f"[Batch {batch_idx}] Adding node {new_key} with reward {current_cum_reward} envstep {current_global_step}")
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
            actions = list(elite_node.action_bytes)  # list[int]

            traj_info = {
                "final_reward": elite_node.cumulative_reward,
                "total_steps": len(actions) * config.STEPS_PER_ACTION,
                "action_sequence": np.array(actions, dtype=np.int32),
                # 新增：保存状态序列序列 (N+1, 6)
                "state_sequence": np.array(elite_node.state_history, dtype=np.float32),
                "reward_sequence": np.array(elite_node.reward_history, dtype=np.float32),
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
    BUDGET = 20000
    REPEATS = 10  # [修改] 跑 30 次
    SCORE_THRESHOLD = 400.0  # [修改] 分数阈值
    PERCENTILE_KEEP = 80  # 可调：90 表示保留 ≥90分位（前10%）
    TOP_K_TRAJS = 500
    SAVE_FILE = "elite_trajectories_v1_top_2.pkl"

    all_valid_data = []
    total_valid_count = 0

    print("========================================")
    print(f"STARTING HIGH-QUALITY TRAJECTORY EXTRACTION")
    print(f"Repeats: {REPEATS}, Threshold: > {SCORE_THRESHOLD}")
    print("========================================")

    for i in range(REPEATS):
        run_id = i + 1
        # 种子逻辑
        seed = BUDGET + (BATCHES * 1000) + i

        print(f"Processing Run {run_id}/{REPEATS} (Seed: {seed})... \n", end="", flush=True)
        start_t = time.time()

        # 运行实验
        trajs = run_extraction_session(BATCHES, BUDGET, seed, run_id, score_threshold=SCORE_THRESHOLD)

        duration = time.time() - start_t

        if len(trajs) > 0:
            # 存入字典
            for traj in trajs:
                all_valid_data.append(traj)
            total_valid_count += len(trajs)
            best_r = trajs[0]['final_reward']
            print(f"Done ({duration:.1f}s). Found {len(trajs)} trajs > {SCORE_THRESHOLD}. Best: {best_r:.2f}")

        else:
            print(f"Done ({duration:.1f}s). No trajectories > {SCORE_THRESHOLD}.")

    # trajs_sorted = sorted(
    #     all_valid_data,
    #     key=lambda t: t["final_reward"],
    #     reverse=True
    # )
    # all_valid_data_top_trajs = trajs_sorted[:num_after]
    # print(f"[Summary] Found {num_before} trajectories. Saving top {num_after} to {SAVE_FILE}")

    scores = [t["final_reward"] for t in all_valid_data]

    threshold = float(np.percentile(scores, PERCENTILE_KEEP))
    kept_trajs = [t for t in all_valid_data if t["final_reward"] >= threshold]

    kept_trajs = sorted(kept_trajs, key=lambda t: t["final_reward"], reverse=True)
    print(f"\nPercentile keep: >= {PERCENTILE_KEEP}th")
    print(f"Threshold score: {threshold:.4f}")
    print(
        f"Kept trajectories: {len(kept_trajs)}/{len(all_valid_data)} ({len(kept_trajs) / len(all_valid_data) * 100:.2f}%)")
    # === 保存数据 ===
    if len(kept_trajs) > 0:
        with open(SAVE_FILE, 'wb') as f:
            pickle.dump(kept_trajs, f)

        print("\n========================================")
        print(f"SUCCESS: Saved {len(kept_trajs)} trajectories to {SAVE_FILE}")
        print("========================================")
    else:
        print("\n[Warning] No trajectories met the criteria. Nothing saved.")


if __name__ == "__main__":
    run_and_save_high_quality()