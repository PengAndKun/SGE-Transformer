import numpy as np
import time
import random
import pybullet as p
import math
import sys
import os
import pandas as pd
from dataclasses import dataclass
from typing import List

# 引入环境
from SGE_Transformer.envs.coverage_lidar_aviary import CoverageAviary
from stable_controller import ROS2VelocityController


# ==========================================
# 0. 辅助工具 (屏蔽输出)
# ==========================================
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


# ==========================================
# 1. 基础配置与类定义
# ==========================================
@dataclass
class BenchmarkConfig:
    TOTAL_STEP_BUDGET: int
    STEPS_PER_ACTION: int = 15
    EXPLORE_HORIZON: int = 8
    CELL_SIZE: float = 1.0
    TIME_RESOLUTION: int = 10
    NUM_ELITES: int = 15


class DroneActionSpace:
    def __init__(self):
        self.V_XY = 0.5;
        self.V_Z = 0.3;
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

    def get_velocity(self, action_id): return self.actions.get(action_id, np.zeros(4))


class CellNode:
    def __init__(self, key, snapshot, cumulative_reward,
                 steps_from_root, actions_from_root,
                 parent_key=None, action_from_parent=None):
        self.key = key
        self.snapshot = snapshot
        self.cumulative_reward = cumulative_reward
        self.steps_from_root = steps_from_root
        self.actions_from_root = actions_from_root
        self.parent_key = parent_key
        self.action_from_parent = action_from_parent
        self.pos = None


def get_cell_key(pos, steps, config):
    ix = int(pos[0] / config.CELL_SIZE)
    iy = int(pos[1] / config.CELL_SIZE)
    iz = int(pos[2] / config.CELL_SIZE)
    it = int(steps / config.TIME_RESOLUTION)
    return (ix, iy, iz, it)


# ==========================================
# 2. RRT 单次实验逻辑
# ==========================================
def run_rrt_session(budget, seed, run_id):
    """
    运行单次 RRT 实验
    返回: best_reward
    """
    # 设置随机种子
    np.random.seed(seed)
    random.seed(seed)

    config = BenchmarkConfig(TOTAL_STEP_BUDGET=budget)

    # 关闭 GUI 以加速训练
    with suppress_output():
        env = CoverageAviary(gui=False, obstacles=True, user_debug_gui=False, ctrl_freq=30)

    action_space = DroneActionSpace()
    controller = ROS2VelocityController()

    # 显式设置种子并初始化
    obs, info = env.reset(seed=seed)
    controller.reset(env.pos[0], env.rpy[0][2])

    # RRT 树列表
    nodes_list = []

    start_snapshot = env.get_snapshot()
    start_pos = env.pos[0]
    start_key = get_cell_key(start_pos, 0, config)

    root_node = CellNode(start_key, start_snapshot, 0.0, 0, 0)
    root_node.pos = start_pos
    nodes_list.append(root_node)

    steps_used = 0
    best_reward = 0.0

    # 采样范围 (根据地图大小调整)
    x_range = [-5, 5]
    y_range = [-5, 5]
    z_range = [0.2, 2.0]

    while steps_used < budget:

        # --- 1. Sample (采样) ---
        rand_pt = np.array([
            np.random.uniform(x_range[0], x_range[1]),
            np.random.uniform(y_range[0], y_range[1]),
            np.random.uniform(z_range[0], z_range[1])
        ])

        # --- 2. Nearest (找最近节点) ---
        # 简单线性查找 (对于 < 10000 节点是可以接受的)
        nearest_node = min(nodes_list, key=lambda n: np.linalg.norm(n.pos - rand_pt))

        # --- 3. Steer (选择动作) ---
        env.restore_snapshot(nearest_node.snapshot)
        controller.reset(env.pos[0], env.rpy[0][2])

        current_pos = nearest_node.pos
        desired_vec = rand_pt - current_pos
        desired_norm = np.linalg.norm(desired_vec)

        best_action_id = 0

        if desired_norm > 0.1:
            desired_vec = desired_vec / desired_norm
            max_dot = -float('inf')

            for act_id in range(action_space.num_actions):
                vel = action_space.get_velocity(act_id)
                vel_vec = vel[:3]
                vel_norm = np.linalg.norm(vel_vec)

                if vel_norm == 0: continue

                dot_prod = np.dot(vel_vec / vel_norm, desired_vec)
                if dot_prod > max_dot:
                    max_dot = dot_prod
                    best_action_id = act_id

        # --- 4. Extend (执行动作) ---
        target_vel = action_space.get_velocity(best_action_id)

        step_reward_sum = 0
        terminated = False

        for _ in range(15):
            if steps_used >= budget: break

            rpm, _ = controller.compute_action(env.CTRL_FREQ, env._computeObs(), target_vel)
            _, reward, _, _, _ = env.step(rpm)

            step_reward_sum += max(0, reward)
            steps_used += 1

            if env.check_collision():
                terminated = True
                break

        # --- 5. Add to Tree ---
        if not terminated:
            new_snapshot = env.get_snapshot()
            new_pos = env.pos[0]
            new_cum_reward = nearest_node.cumulative_reward + step_reward_sum

            new_node = CellNode(
                key=get_cell_key(new_pos, steps_used, config),
                snapshot=new_snapshot,
                cumulative_reward=new_cum_reward,
                steps_from_root=nearest_node.steps_from_root + 15,
                actions_from_root=nearest_node.actions_from_root + 1,
                parent_key=nearest_node.key,
                action_from_parent=best_action_id
            )
            new_node.pos = new_pos
            nodes_list.append(new_node)

            if new_cum_reward > best_reward:
                best_reward = new_cum_reward

    env.close()
    return best_reward


# ==========================================
# 3. 批量基准测试主程序
# ==========================================
def run_rrt_benchmark():
    BUDGETS = [1000, 2000, 5000, 10000, 25000, 50000, 100000]
    REPEATS = 10
    BASE_SEED = 2024  # 基础种子

    OUTPUT_FILE = "rrt_benchmark_results.csv"

    records = []

    print("========================================")
    print("STARTING RRT BENCHMARK")
    print(f"Budgets: {BUDGETS}")
    print(f"Repeats: {REPEATS}")
    print("========================================")

    for budget in BUDGETS:
        print(f"\nProcessing Budget: {budget}")
        start_time = time.time()

        for i in range(REPEATS):
            # 种子生成逻辑：保证每次 Run 不一样，且与其他算法对齐
            # 这里的逻辑建议与 SGE 保持一致，例如 seed = budget + i*1000
            # 简单起见，这里使用：
            current_seed = BASE_SEED + budget + i * 7

            reward = run_rrt_session(budget, current_seed, i)

            records.append({
                "Algorithm": "RRT",
                "Budget": budget,
                "Run_ID": i,
                "Episode_Score": reward  # 这里的 Episode Score 指的是树中最好的节点分数
            })

            print(f".", end="", flush=True)

        elapsed = time.time() - start_time
        print(f" Done ({elapsed:.2f}s)")

        # 实时保存，防止跑一半崩了
        df_temp = pd.DataFrame(records)
        df_temp.to_csv(OUTPUT_FILE, index=False)

    print("\n========================================")
    print(f"RRT Benchmark Finished. Data saved to {OUTPUT_FILE}")
    print("========================================")


if __name__ == "__main__":
    run_rrt_benchmark()