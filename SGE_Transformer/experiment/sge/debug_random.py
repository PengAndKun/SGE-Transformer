import numpy as np
import time
import random
import pybullet as p
import pandas as pd
import os

# 引入环境
from SGE_Transformer.envs.coverage_lidar_aviary import CoverageAviary
from stable_controller import ROS2VelocityController


# ==========================================
# 1. 动作空间
# ==========================================
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
        if prev_action in self.opposite_actions: probs[self.opposite_actions[prev_action]] *= 0.01
        probs[0] *= 0.5
        probs /= np.sum(probs)
        return np.random.choice(self.num_actions, p=probs)


# ==========================================
# 2. 单次实验运行器 (收集所有回合分数)
# ==========================================
def run_random_session(mode, run_id, total_budget=2000, seed=2024):
    """
    返回: List[float] -> 包含本次实验中每一次坠机前的得分
    """
    # print(f"--- Running {mode.upper()} Run {run_id} ---")

    current_seed = seed + run_id
    np.random.seed(current_seed)
    random.seed(current_seed)

    # 设为 False 加速，设为 True 观察
    env = CoverageAviary(gui=False, obstacles=True, user_debug_gui=False, ctrl_freq=30)
    action_space = DroneActionSpace()
    controller = ROS2VelocityController()

    obs, info = env.reset()
    controller.reset(env.pos[0], env.rpy[0][2])

    # [修改] 这里不再存 best，而是存一个列表
    all_episode_rewards = []

    current_episode_reward = 0.0
    steps_used = 0
    last_action = None

    while steps_used < total_budget:
        # 策略选择
        if mode == 'inertia':
            action_id = action_space.sample(prev_action=last_action)
        else:
            action_id = action_space.sample(prev_action=None)

        target_vel = action_space.get_velocity(action_id)
        last_action = action_id

        for _ in range(6):
            if steps_used >= total_budget: break

            rpm, _ = controller.compute_action(env.CTRL_FREQ, env._computeObs(), target_vel)
            obs, reward, terminated, _, _ = env.step(rpm)

            current_episode_reward += reward
            steps_used += 1

            # --- 碰撞检测 ---
            contact_points = p.getContactPoints(bodyA=env.DRONE_IDS[0], physicsClientId=env.CLIENT)
            has_collision = len(contact_points) > 0
            r, p_angle, y = env.rpy[0]
            is_flipped = abs(r) > 1.5 or abs(p_angle) > 1.5
            z_height = env.pos[0][2]
            out_of_bounds = z_height < 0.05 or z_height > 2.5

            if has_collision or is_flipped or out_of_bounds:
                terminated = True

                # [关键] 记录这一条命的分数
                all_episode_rewards.append(current_episode_reward)
                # print(f"    [Crash] Episode Score: {current_episode_reward:.2f}")

                env.reset()
                controller.reset(env.pos[0], env.rpy[0][2])
                current_episode_reward = 0.0
                last_action = None
                break

    # 预算耗尽时，如果还有未完成的回合，也把它的分数加上去（或者可以选择丢弃）
    if current_episode_reward > 0:
        all_episode_rewards.append(current_episode_reward)

    env.close()
    return all_episode_rewards


# ==========================================
# 3. 主程序：全量数据收集与统计
# ==========================================
def run_comparative_experiment_full_stats():
    NUM_RUNS = 10
    BUDGET = 100000
    BASE_SEED = 2024
    OUTPUT_FILE = "random_100000_all_episodes_results.csv"

    # 用于存储每一条命的数据，方便存 pandas
    # 格式: {'Algorithm': 'Pure', 'Run_ID': 0, 'Episode_Score': 12.5}
    full_data_log = []

    print("========================================")
    print("STARTING FULL STATISTICS EXPERIMENT")
    print("========================================")

    # --- 1. Pure Random ---
    print("Running Pure Random...")
    for i in range(NUM_RUNS):
        # 拿到这一次 Run 里所有回合的分数列表
        scores = run_random_session(mode='pure', run_id=i, total_budget=BUDGET, seed=BASE_SEED)

        # 将列表里的每个分数都拆出来存
        for score in scores:
            full_data_log.append({
                "Algorithm": "Pure Random",
                "Run_ID": i,
                "Episode_Score": score
            })
        print(f"  Run {i}: Generated {len(scores)} episodes.")

    # --- 2. Inertia Random ---
    print("\nRunning Inertia Random...")
    for i in range(NUM_RUNS):
        scores = run_random_session(mode='inertia', run_id=i, total_budget=BUDGET, seed=BASE_SEED)

        for score in scores:
            full_data_log.append({
                "Algorithm": "Inertia Random",
                "Run_ID": i,
                "Episode_Score": score
            })
        print(f"  Run {i}: Generated {len(scores)} episodes.")

    # --- 3. 统计分析 ---
    df = pd.DataFrame(full_data_log)

    # 保存原始数据
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAll episode data saved to {OUTPUT_FILE}")

    # 计算统计量：针对 Episode_Score 列进行聚合
    # 这会计算所有 Run、所有 Episode 的总均值和总方差
    stats = df.groupby("Algorithm")["Episode_Score"].agg(['count', 'mean', 'var', 'std'])

    print("\n========================================")
    print("FINAL GLOBAL STATISTICS (All Episodes)")
    print("========================================")
    print(stats)
    print("========================================")
    print("count: 总回合数 (坠机次数)")
    print("mean : 平均每次存活获得的覆盖分")
    print("var  : 分数方差")
    print("std  : 分数标准差")


if __name__ == "__main__":
    run_comparative_experiment_full_stats()