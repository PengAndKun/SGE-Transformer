import numpy as np
import torch
import torch.nn.functional as F
import pybullet as p
import time
import os
import sys

# 尝试添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入必要的模块
from SGE_Transformer.envs.coveragel_multi_room import CoverageAviary
from SGE_Transformer.experiment.sge_pybullet.extract_trajectories_macro import MacroActionSpace27
# 导入我们刚刚创建的 GPTNeo 类
from SGE_Transformer.experiment.sge_pybullet.sge_gptneo_bc import HuggingFaceGPTNeoBCNetwork

def test_gptneo_bc(model_path, num_test_episodes=10, stochastic=True):
    # --- 1. 参数配置 (必须与训练时完全一致) ---
    STATE_DIM = 6
    NUM_ACTIONS = 27
    H = 20  # 观察的历史长度 (必须与训练时一致)
    MOVE_DISTANCE = 0.5

    # 归一化参数
    POS_SCALE = 25.0
    RPY_SCALE = 3.14
    
    # 模型架构参数 (必须与训练时一致)
    D_MODEL = 128
    NUM_LAYERS = 4
    N_HEAD = 4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. 加载模型 ---
    print(f"Loading GPTNeo model from: {model_path}")
    network = HuggingFaceGPTNeoBCNetwork(
        state_dim=STATE_DIM,
        num_actions=NUM_ACTIONS,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        nhead=N_HEAD,
        max_seq_length=H
    ).to(DEVICE)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        network.load_state_dict(checkpoint['network_state_dict'])
        network.eval()
        print(f"✅ Successfully loaded model")
    else:
        print(f"❌ Model file not found: {model_path}")
        return

    # --- 3. 初始化环境 ---
    env = CoverageAviary(
        gui=False, 
        obstacles=True,
        num_rays=120,
        radar_radius=8.0,
        grid_res=0.5
    )
    action_space = MacroActionSpace27(move_distance=MOVE_DISTANCE)

    obs, info = env.reset()
    start_snapshot = env.get_snapshot()

    episode_rewards = []
    episode_coverages = []

    for episode in range(num_test_episodes):
        env.restore_snapshot(start_snapshot)
        
        # 初始化状态 [x, y, z, r, p, y]
        init_state = np.concatenate([env.pos[0], env.rpy[0]])
        # 用初始状态填充历史窗口
        state_history = [init_state] * H

        total_reward = 0.0
        done = False
        step_count = 0
        max_steps = 150
        collision = False

        print(f"\n--- Episode {episode + 1} Start ---")

        while not done and step_count < max_steps:
            # --- 4. 准备输入 ---
            # 取最近H个状态
            current_seq = np.array(state_history[-H:], dtype=np.float32)
            
            # 归一化
            input_states = current_seq.copy()
            input_states[:, :3] /= POS_SCALE
            input_states[:, 3:] /= RPY_SCALE

            # 转 Tensor
            input_tensor = torch.FloatTensor(input_states).unsqueeze(0).to(DEVICE)

            # --- 5. 模型预测 ---
            with torch.no_grad():
                # 使用确定性策略 (argmax) 或 随机采样 进行评估
                action_id, probs = network.get_action(input_tensor, deterministic=not stochastic)
                action_id = action_id.item()

            # --- 6. 执行动作 ---
            start_pos = env.pos[0].copy()
            displacement = action_space.get_displacement(action_id)
            target_pos = start_pos + displacement

            if 0.5 <= target_pos[2] <= 3.5:
                p.resetBasePositionAndOrientation(env.DRONE_IDS[0], target_pos, [0, 0, 0, 1], physicsClientId=env.CLIENT)
                env._updateAndStoreKinematicInformation()
                
                _, reward, terminated, truncated, _ = env.compute_scan_at_pos(target_pos)
                total_reward += reward
                
                if terminated:
                    print(f"💥 Collision at step {step_count}")
                    done = True
                    collision = True
            else:
                # print(f"⚠️ Out of bounds action: {action_id}")
                reward = 0
                done = True # 也可以选择 continue
                collision = True # 视越界为失败

            # --- 7. 更新历史 ---
            current_state = np.concatenate([env.pos[0], env.rpy[0]])
            state_history.append(current_state)

            step_count += 1
            if step_count % 10 == 0:
                print(f"Step {step_count} | Reward: {total_reward:.2f}")

            time.sleep(0.01)

        final_coverage = env._computeInfo()['coverage_ratio']
        episode_rewards.append(total_reward)
        episode_coverages.append(final_coverage)

        print(f"🏁 Episode {episode + 1} End | Total Reward: {total_reward:.2f} | Coverage: {final_coverage:.2%}")

    env.close()
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_coverage = np.mean(episode_coverages)
    std_coverage = np.std(episode_coverages)
    
    print("\n" + "="*40)
    print(f"📊 Test Summary ({num_test_episodes} Episodes)")
    print(f"Average Reward:   {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Coverage: {avg_coverage:.2%} ± {std_coverage:.2%}")
    print("="*40 + "\n")
    
    return avg_reward, std_reward, avg_coverage, std_coverage


if __name__ == "__main__":
    import csv
    
    EPOCHS = 20000
    INTERVAL = 500
    RESULTS_FILE = "gptneo_bc_benchmark_results.csv"

    all_results = []

    for epoch in range(0, EPOCHS + 1, INTERVAL):
        MODEL_NAME = f"sge_gptneo_bc_model_{epoch}.pth"
        MODEL_PATH = None
        
        # Search path
        candidates = [
            MODEL_NAME,
            os.path.join(os.path.dirname(__file__), MODEL_NAME),
            "SGE_Transformer/experiment/sge_pybullet/" + MODEL_NAME
        ]
        for c in candidates:
            if os.path.exists(c):
                MODEL_PATH = c
                break

        if MODEL_PATH:
            print(f">>> Testing Checkpoint: {epoch} <<<")
            avg_rew, std_rew, avg_cov, std_cov = test_gptneo_bc(MODEL_PATH, num_test_episodes=20, stochastic=True)
            all_results.append([epoch, avg_rew, std_rew, avg_cov, std_cov])
        else:
            print(f"Skipping epoch {epoch}: Model file not found.")

    # Save results to file
    with open(RESULTS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average Reward', 'Reward Std', 'Average Coverage', 'Coverage Std'])
        writer.writerows(all_results)
    print(f"Detailed results saved to {RESULTS_FILE}")
