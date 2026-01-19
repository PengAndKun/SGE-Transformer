import os
import pickle
import numpy as np
import pandas as pd

# ==========================================
# 配置参数
# ==========================================
BUDGETS = [1000, 2000, 5000, 10000, 20000, 50000]
BATCHES = [1, 2, 4, 5, 8]

# 定义算法及其对应的文件路径模板
# {} 将被 budget 或 (batch, budget) 填充
PATH_TEMPLATES = {
    "Random": "../../data_pybullet/3d_trajectories/random/random_trajectories_budget_{}.pkl",
    "Greedy": "../../data_pybullet/3d_trajectories/greedy/greedy_trajectories_budget_{}.pkl",
    "RRT": "../../data_pybullet/3d_trajectories/rrt/rrt_trajectories_nodes_{}.pkl",
    "Random-ISE": "../../data_pybullet/3d_trajectories/hybrid_random/hybrid_random_trajectories_batches_{}_budget_{}.pkl",
    "G-ISE (Ours)": "../../data_pybullet/3d_trajectories/hybrid/hybrid_trajectories_batches_{}_budget_{}.pkl",
    "ACO": "../../data_pybullet/3d_trajectories/aco/aco_trajectories_budget_{}.pkl",
    "PSO": "../../data_pybullet/3d_trajectories/pso/pso_trajectories_budget_{}.pkl",
    "Go-Explore": "../../data_pybullet/3d_trajectories/go_explore/go_explore_trajectories_budget_{}.pkl",
}


def analyze_data():
    results = []

    for method, template in PATH_TEMPLATES.items():
        for budget in BUDGETS:
            # 判断该算法是否依赖 BATCHES 参数
            current_batches = BATCHES if method in ["Random-ISE", "G-ISE (Ours)"] else [None]

            for j in current_batches:
                # 构建路径
                if j is None:
                    file_path = template.format(budget)
                else:
                    file_path = template.format(j, budget)

                # 检查文件是否存在
                if not os.path.exists(file_path):
                    # print(f"Skipping: {file_path} (Not found)")
                    continue

                try:
                    with open(file_path, 'rb') as f:
                        trajs = pickle.load(f)

                    if not trajs:
                        continue

                    # 1. 提取 final_reward
                    rewards = np.array([t["final_reward"] for t in trajs])

                    # 2. 计算步数效率 (final_reward / total_steps)
                    # 避免 total_steps 为 0 的异常
                    efficiencies = np.array([
                        t["final_reward"] / t["total_steps"] if t["total_steps"] > 0 else 0
                        for t in trajs
                    ])

                    # 记录统计结果
                    results.append({
                        "Method": method,
                        "Budget": budget,
                        "Batches": j if j is not None else "-",
                        "Mean_Reward": np.mean(rewards),
                        "Std_Reward": np.std(rewards),
                        "Mean_Efficiency": np.mean(efficiencies),
                        "Std_Efficiency": np.std(efficiencies),
                        "Traj_Count": len(trajs)
                    })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # 使用 Pandas 整理表格
    df = pd.DataFrame(results)

    # 格式化输出
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 4)

    # 打印最终统计表
    print("\n" + "=" * 100)
    print("算法性能对比汇总表 (Reward & Efficiency)")
    print("=" * 100)
    print(df[["Method", "Budget", "Batches", "Mean_Reward", "Std_Reward", "Mean_Efficiency", "Std_Efficiency"]])

    # 导出到 CSV 方便你在 Excel 中画图
    df.to_csv("trajectory_analysis_results.csv", index=False)
    print("\n✅ 分析完成！结果已保存至: trajectory_analysis_results.csv")


if __name__ == "__main__":
    analyze_data()