import numpy as np
import time
import random
import pybullet as p
import math
import copy
import pandas as pd  # [新增] 用于保存数据
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
            0: np.array([0.0, 0.0, 0.0, 0.0]),  # Hover
            1: np.array([self.V_XY, 0.0, 0.0, 0.0]),  # Fwd
            2: np.array([-self.V_XY, 0.0, 0.0, 0.0]),  # Back
            3: np.array([0.0, self.V_XY, 0.0, 0.0]),  # Left
            4: np.array([0.0, -self.V_XY, 0.0, 0.0]),  # Right
            5: np.array([0.0, 0.0, self.V_Z, 0.0]),  # Up
            6: np.array([0.0, 0.0, -self.V_Z, 0.0]),  # Down
            7: np.array([0.0, 0.0, 0.0, self.W_Z]),  # Turn L
            8: np.array([0.0, 0.0, 0.0, -self.W_Z])  # Turn R
        }
        self.num_actions = len(self.actions)

    def get_velocity(self, action_id): return self.actions.get(action_id, np.zeros(4))


# ==========================================
# 2. 单次 Greedy 实验 (包含你的核心 V8 逻辑)
# ==========================================
def run_greedy_session(run_id, total_budget=20000, seed=2024, gui=False):
    """
    运行一次完整的 Greedy 实验，返回该实验中产生的所有 Episode 分数列表。
    """
    # 设置随机种子 (Base Seed + Run ID)
    current_seed = seed + run_id
    np.random.seed(current_seed)
    random.seed(current_seed)

    print(f"--- Running Greedy Agent V8 | Run {run_id} ---")

    # 初始化环境 (注意：批量跑数据时建议 gui=False，想看画面设为 True)
    env = CoverageAviary(gui=gui, obstacles=True, user_debug_gui=False, ctrl_freq=30)
    action_space = DroneActionSpace()
    controller = ROS2VelocityController()

    obs, info = env.reset()
    controller.reset(env.pos[0], env.rpy[0][2])

    # [修改] 用于记录本轮预算内所有 Episode 的分数
    all_episode_rewards = []

    current_episode_reward = 0.0
    steps_used = 0

    # [核心] 明确指定环境中的覆盖变量名
    coverage_var_name = 'coverage_grid'
    if not hasattr(env, coverage_var_name):
        print(f"[Critical Error] Env does not have attribute '{coverage_var_name}'!")
        return []

    all_possible_actions = range(action_space.num_actions)

    while steps_used < total_budget:
        # ==========================================
        # Phase 1: Planning (全量前视模拟)
        # ==========================================

        # 1. 保存物理状态
        current_snapshot = p.saveState(physicsClientId=env.CLIENT)

        # 2. 保存 Python 覆盖状态 (深拷贝)
        original_grid = getattr(env, coverage_var_name)
        saved_grid_state = original_grid.copy()

        best_action = 0
        max_decision_score = -99999.0
        safe_actions = []

        for act_id in all_possible_actions:
            # A. 物理回滚
            p.restoreState(current_snapshot, physicsClientId=env.CLIENT)
            env._updateAndStoreKinematicInformation()

            # B. Python 覆盖状态回滚
            setattr(env, coverage_var_name, saved_grid_state.copy())

            target_vel = action_space.get_velocity(act_id)
            sim_decision_score = 0
            crashed_in_sim = False

            # C. 模拟执行
            for _ in range(15):
                rpm, _ = controller.compute_action(env.CTRL_FREQ, env._computeObs(), target_vel)
                _, r, _, _, _ = env.step(rpm)

                steps_used += 1

                # 模拟中使用内置的 collision check
                if env.check_collision():
                    sim_decision_score = -100.0
                    crashed_in_sim = True
                    break

                sim_decision_score += max(0, r)

            if not crashed_in_sim:
                sim_decision_score += 0.01
                safe_actions.append(act_id)

            if sim_decision_score > max_decision_score:
                max_decision_score = sim_decision_score
                best_action = act_id

        # 3. 规划结束，回到现实
        # A. 物理回滚
        p.restoreState(current_snapshot, physicsClientId=env.CLIENT)
        p.removeState(current_snapshot, physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()

        # B. Python 覆盖状态回滚 (回到规划前的真实状态)
        setattr(env, coverage_var_name, saved_grid_state)

        # ==========================================
        # 防卡死
        # ==========================================
        if max_decision_score < 0.05 and len(safe_actions) > 0:
            best_action = random.choice(safe_actions)

        # ==========================================
        # Phase 2: Execution (真实执行)
        # ==========================================
        target_vel = action_space.get_velocity(best_action)
        # print(f"Step {steps_used}: Executing Action {best_action} (Exp Score: {max_decision_score:.2f})")

        for _ in range(15):
            if steps_used >= total_budget: break

            rpm, _ = controller.compute_action(env.CTRL_FREQ, env._computeObs(), target_vel)
            _, reward, terminated, _, _ = env.step(rpm)

            current_episode_reward += max(0, reward)
            steps_used += 1

            # 使用环境内置的 check_collision
            if env.check_collision():
                terminated = True
                print(f"    [Crash] Run {run_id} | Score: {current_episode_reward:.4f}")

                # [关键] 记录这一条命的分数
                all_episode_rewards.append(current_episode_reward)

                env.reset()
                controller.reset(env.pos[0], env.rpy[0][2])
                current_episode_reward = 0.0

                # 暂停一下，如果开启 GUI 方便观察
                if gui: time.sleep(0.5)
                break

        # 实时打印当前分数（可选）
        # print(f'current_episode_reward: {current_episode_reward}')

    # 预算结束，如果有未完成的回合分数且大于0，也记录下来
    if current_episode_reward > 0:
        all_episode_rewards.append(current_episode_reward)

    env.close()
    return all_episode_rewards


# ==========================================
# 3. 主程序：批量实验与数据分析
# ==========================================
def run_greedy_stats_experiment():
    NUM_RUNS = 10  # 实验重复次数 (因为 Greedy 比较慢，先设 5 次，你可以改为 10)
    BUDGET = 5000  # 你的总预算
    BASE_SEED = 2024
    OUTPUT_FILE = "greedy_5000_all_episodes.csv"

    full_data_log = []

    print("========================================")
    print("STARTING GREEDY V8 STATISTICS EXPERIMENT")
    print(f"Runs: {NUM_RUNS}, Budget: {BUDGET}")
    print("========================================")

    for i in range(NUM_RUNS):
        # 运行单次实验 (开启 gui=True 可以看画面，设为 False 跑得快)
        scores = run_greedy_session(run_id=i, total_budget=BUDGET, seed=BASE_SEED, gui=False)

        # 收集数据
        for score in scores:
            full_data_log.append({
                "Algorithm": "Greedy",
                "Run_ID": i,
                "Episode_Score": score
            })
        print(f"-> Run {i} Finished. Generated {len(scores)} episodes.\n")

    # --- 数据保存与统计 ---
    if not full_data_log:
        print("No episodes recorded!")
        return

    df = pd.DataFrame(full_data_log)

    # 保存 CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAll episode data saved to {OUTPUT_FILE}")

    # 计算统计量
    stats = df.groupby("Algorithm")["Episode_Score"].agg(['count', 'mean', 'var', 'std', 'max', 'min'])

    print("\n========================================")
    print("FINAL GREEDY STATISTICS (All Episodes)")
    print("========================================")
    print(stats)
    print("========================================")
    print("count: 总回合数 (坠机次数)")
    print("mean : 平均每次存活获得的覆盖分")
    print("var  : 分数方差")
    print("std  : 分数标准差")
    print("max  : 历史最高单次得分")


if __name__ == "__main__":
    run_greedy_stats_experiment()