import pandas as pd
import numpy as np
import os


def process_benchmark_data():
    # === 1. 参数设置 ===
    n_values = [1000, 2000, 5000, 10000, 25000, 50000, 100000]  # Budget
    m_values = [1, 2, 4, 5, 8, 10]  # Batches

    # 用于存储最终汇总数据的列表
    summary_data = []

    print("=== 开始处理数据 ===")

    # ==========================================
    # A. 处理 Greedy 数据
    # 文件名: greedy_{n}_all_episodes.csv
    # ==========================================
    print("\n--- Processing Greedy ---")
    for n in n_values:
        filename = f"greedy_{n}_all_episodes.csv"

        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)

                # 关键步骤：Greedy 在一次 Run 中有多次坠机
                # 我们取每次 Run_ID 下的【最高分】作为该次 Run 的最终表现
                run_performance = df.groupby('Run_ID')['Episode_Score'].max()

                mean_val = run_performance.mean()
                var_val = run_performance.var()
                std_val = run_performance.std()

                summary_data.append({
                    'Algorithm': 'Greedy',
                    'Budget': n,
                    'Batch': '-',  # Greedy 没有 Batch 概念
                    'Mean': mean_val,
                    'Variance': var_val,
                    'Std_Dev': std_val,
                    'Sample_Size': len(run_performance)
                })
                print(f"Loaded {filename}: Mean={mean_val:.2f}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"Warning: {filename} not found.")

    # ==========================================
    # B. 处理 Random 数据 (Pure & Inertia)
    # 文件名: random_{n}_all_episodes_results.csv
    # ==========================================
    print("\n--- Processing Random ---")
    for n in n_values:
        filename = f"random_{n}_all_episodes_results.csv"

        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)

                # Random 文件里混合了 'Pure Random' 和 'Inertia Random'
                # 我们需要分别筛选
                for algo_name in df['Algorithm'].unique():
                    sub_df = df[df['Algorithm'] == algo_name]

                    # 同样，取每个 Run_ID 的最大值
                    run_performance = sub_df.groupby('Run_ID')['Episode_Score'].max()

                    mean_val = run_performance.mean()
                    var_val = run_performance.var()
                    std_val = run_performance.std()

                    summary_data.append({
                        'Algorithm': algo_name,
                        'Budget': n,
                        'Batch': '-',
                        'Mean': mean_val,
                        'Variance': var_val,
                        'Std_Dev': std_val,
                        'Sample_Size': len(run_performance)
                    })
                print(f"Loaded {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"Warning: {filename} not found.")

    # ==========================================
    # C. 处理 SGE 数据
    # 文件名: results/SGE_Batch{m}_Budget{n}.csv (假设在 results 文件夹)
    # 或者根目录: SGE_Batch{m}_Budget{n}.csv
    # ==========================================
    print("\n--- Processing SGE ---")
    for n in n_values:
        for m in m_values:
            # 优先检查 results 文件夹，如果没有则检查根目录
            paths_to_check = [
                f"results/SGE_Batch{m}_Budget{n}.csv",
                f"SGE_Batch{m}_Budget{n}.csv"
            ]

            found = False
            for filename in paths_to_check:
                if os.path.exists(filename):
                    try:
                        df = pd.read_csv(filename)

                        # SGE 的 CSV 结构已经是：每个 Run_ID 一行最优解 (reward)
                        # 所以直接计算 reward 列的统计量即可
                        rewards = df['reward']

                        mean_val = rewards.mean()
                        var_val = rewards.var()
                        std_val = rewards.std()

                        summary_data.append({
                            'Algorithm': f"SGE_Batch{m}",
                            'Budget': n,
                            'Batch': m,
                            'Mean': mean_val,
                            'Variance': var_val,
                            'Std_Dev': std_val,
                            'Sample_Size': len(rewards)
                        })
                        # print(f"Loaded {filename}: Mean={mean_val:.2f}")
                        found = True
                        break  # 找到了就跳出路径循环
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

            if not found:
                # 只有当 Budget 能被 Batch 整除时文件才应该存在，否则可能是逻辑上跳过的
                # 这里只打印简单的 Warning
                pass
                # print(f"Warning: SGE file for Batch {m} Budget {n} not found.")

    # ==========================================
    # 4. 生成最终表格
    # ==========================================
    if not summary_data:
        print("\nNo data found! Please check file paths.")
        return

    final_df = pd.DataFrame(summary_data)

    # 调整列顺序
    cols = ['Algorithm', 'Budget', 'Batch', 'Mean', 'Variance', 'Std_Dev', 'Sample_Size']
    final_df = final_df[cols]

    # 按 Budget 和 Mean 排序，方便查看
    final_df = final_df.sort_values(by=['Budget', 'Mean'], ascending=[True, False])

    # 打印预览
    print("\n" + "=" * 80)
    print("FINAL AGGREGATED STATISTICS")
    print("=" * 80)
    print(final_df.to_string(index=False))
    print("=" * 80)

    # 保存结果
    output_file = "Final_Benchmark_Statistics.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\nResult saved to: {output_file}")


if __name__ == "__main__":
    process_benchmark_data()