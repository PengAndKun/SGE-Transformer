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
                # 取每次 Run_ID 下的【最高分】
                run_performance = df.groupby('Run_ID')['Episode_Score'].max()

                summary_data.append({
                    'Algorithm': 'Greedy',
                    'Budget': n,
                    'Batch': '-',
                    'Mean': run_performance.mean(),
                    'Variance': run_performance.var(),
                    'Std_Dev': run_performance.std(),
                    'Sample_Size': len(run_performance)
                })
                print(f"Loaded {filename}: Mean={run_performance.mean():.2f}")
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
                for algo_name in df['Algorithm'].unique():
                    sub_df = df[df['Algorithm'] == algo_name]
                    run_performance = sub_df.groupby('Run_ID')['Episode_Score'].max()

                    summary_data.append({
                        'Algorithm': algo_name,
                        'Budget': n,
                        'Batch': '-',
                        'Mean': run_performance.mean(),
                        'Variance': run_performance.var(),
                        'Std_Dev': run_performance.std(),
                        'Sample_Size': len(run_performance)
                    })
                print(f"Loaded {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"Warning: {filename} not found.")

    # ==========================================
    # C. 处理 RRT 数据 [新增]
    # 文件名: rrt_benchmark_results.csv (通常包含所有 Budget)
    # ==========================================
    print("\n--- Processing RRT ---")
    rrt_filename = "rrt_benchmark_results.csv"
    if os.path.exists(rrt_filename):
        try:
            rrt_df = pd.read_csv(rrt_filename)

            # RRT 文件包含所有 Budget，需要按 Budget 循环筛选
            for n in n_values:
                # 筛选当前 Budget 的数据
                sub_df = rrt_df[rrt_df['Budget'] == n]

                if not sub_df.empty:
                    # RRT 脚本通常每个 Run_ID 只保存一行（该 Run 的最佳结果）
                    # 但为了保险，还是做一次 groupby max
                    run_performance = sub_df.groupby('Run_ID')['Episode_Score'].max()

                    summary_data.append({
                        'Algorithm': 'RRT',
                        'Budget': n,
                        'Batch': '-',
                        'Mean': run_performance.mean(),
                        'Variance': run_performance.var(),
                        'Std_Dev': run_performance.std(),
                        'Sample_Size': len(run_performance)
                    })
                    print(f"Processed RRT for Budget {n}: Mean={run_performance.mean():.2f}")
                else:
                    pass
                    # print(f"Warning: No RRT data found for Budget {n}")
        except Exception as e:
            print(f"Error reading {rrt_filename}: {e}")
    else:
        print(f"Warning: {rrt_filename} not found.")

    # ==========================================
    # D. 处理 SGE 数据
    # ==========================================
    print("\n--- Processing SGE ---")
    for n in n_values:
        for m in m_values:
            paths_to_check = [
                f"results/SGE_Batch{m}_Budget{n}.csv",
                f"SGE_Batch{m}_Budget{n}.csv"
            ]

            found = False
            for filename in paths_to_check:
                if os.path.exists(filename):
                    try:
                        df = pd.read_csv(filename)
                        rewards = df['reward']

                        summary_data.append({
                            'Algorithm': f"SGE_Batch{m}",
                            'Budget': n,
                            'Batch': m,
                            'Mean': rewards.mean(),
                            'Variance': rewards.var(),
                            'Std_Dev': rewards.std(),
                            'Sample_Size': len(rewards)
                        })
                        found = True
                        break
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
            if not found:
                pass

    # ==========================================
    # E. 处理 SGE-Random (Ablation) 数据 [新增]
    # 文件名: SGE_Random_Batch{m}_Budget{n}.csv
    # ==========================================
    print("\n--- Processing SGE-Random (Ablation) ---")
    for n in n_values:
        for m in m_values:
            # SGE-Random 通常保存在 results 文件夹下
            paths_to_check = [
                f"SGE_Random_Batch{m}_Budget{n}.csv",
                f"SGE_Random_Batch{m}_Budget{n}.csv"
            ]

            found = False
            for filename in paths_to_check:
                if os.path.exists(filename):
                    try:
                        df = pd.read_csv(filename)
                        rewards = df['reward']

                        summary_data.append({
                            # 命名为 SGE-Random 方便区分
                            'Algorithm': f"SGE-Random_Batch{m}",
                            'Budget': n,
                            'Batch': m,
                            'Mean': rewards.mean(),
                            'Variance': rewards.var(),
                            'Std_Dev': rewards.std(),
                            'Sample_Size': len(rewards)
                        })
                        # print(f"Loaded {filename}")
                        found = True
                        break
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

            if not found:
                pass

    # ==========================================
    # 5. 生成最终表格
    # ==========================================
    if not summary_data:
        print("\nNo data found! Please check file paths.")
        return

    final_df = pd.DataFrame(summary_data)

    # 调整列顺序
    cols = ['Algorithm', 'Budget', 'Batch', 'Mean', 'Variance', 'Std_Dev', 'Sample_Size']
    final_df = final_df[cols]

    # 按 Budget (升序) -> Mean (降序) 排序
    final_df = final_df.sort_values(by=['Budget', 'Mean'], ascending=[True, False])

    # 打印预览
    print("\n" + "=" * 100)
    print("FINAL AGGREGATED STATISTICS")
    print("=" * 100)
    # 设置 pandas 显示选项以避免列被截断
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    print(final_df.to_string(index=False))
    print("=" * 100)

    # 保存结果
    output_file = "Final_Benchmark_Statistics.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\nResult saved to: {output_file}")


if __name__ == "__main__":
    process_benchmark_data()