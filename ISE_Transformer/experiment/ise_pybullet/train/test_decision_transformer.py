import numpy as np
import torch
import torch.nn.functional as F
import pybullet as p
import time
import os
import sys

# 尝试添加项目根目录到 sys.path 以便导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入环境和模块
from SGE_Transformer.envs.coveragel_multi_room import CoverageAviary
from SGE_Transformer.experiment.sge_pybullet.sge_decision_transformer import DecisionTransformer
from SGE_Transformer.experiment.sge_pybullet.extract_trajectories_macro import MacroActionSpace27

def test_decision_transformer(model_path, num_test_episodes=10, target_return=1000.0, stochastic=True):
    # --- 1. 参数配置 (需与训练一致) ---
    STATE_DIM = 6
    ACT_DIM = 27
    CONTEXT_LEN = 50
    
    # 归一化参数
    POS_SCALE = 25.0
    RPY_SCALE = 3.14
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 2. 加载模型 ---
    model = DecisionTransformer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        max_length=CONTEXT_LEN,
        n_layer=4,        # 确保这些参数与训练时的配置一致
        n_head=4,
        hidden_size=128
    ).to(DEVICE)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        # 兼容处理：检查是否包含 state_dict key 或者是直接保存的 state_dict
        if 'state_dict' in checkpoint: # 如果是 pytorch lightning 风格
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            print("❌ 模型文件格式无法识别")
            return
        print(f"✅ 成功加载模型: {model_path}")
    else:
        print(f"❌ 模型文件未找到: {model_path}")
        return

    model.eval()

    # --- 3. 初始化环境 ---
    env = CoverageAviary(
        gui=False, 
        obstacles=True,
        num_rays=120,
        radar_radius=8.0,
        grid_res=0.5
    )
    action_space = MacroActionSpace27(move_distance=0.5)

    obs, info = env.reset()
    start_snapshot = env.get_snapshot()

    episode_rewards = []
    episode_coverages = []

    for episode in range(num_test_episodes):
        env.restore_snapshot(start_snapshot)
        
        # 初始状态 [x, y, z, r, p, y]
        init_state = np.concatenate([env.pos[0], env.rpy[0]])
        
        # 序列历史容器
        states = [init_state]
        actions = [] # 历史动作，初始为空
        rewards = [] # 历史奖励
        rtgs = [target_return] # Returns-to-go
        timesteps = [0]
        
        total_reward = 0.0
        done = False
        step_count = 0
        max_steps = 150
        collision = False

        print(f"\n--- Episode {episode + 1} Start (Target Return: {target_return}) ---")

        while not done and step_count < max_steps:
            # --- 4. 准备模型输入 ---
            # 获取最近的 CONTEXT_LEN 长度的数据
            cur_len = len(states)
            start_idx = max(0, cur_len - CONTEXT_LEN)
            
            # SGE_Transformer/experiment/sge_pybullet/gise_decision_transformer.py 中 get_batch 的逻辑：
            # s[i] 对应 action[i]
            # 为了预测 t 时刻的动作 a_t，我们需要输入 s_{0:t}, a_{0:t-1}, R_{0:t}
            # 在 transformer 输入序列中，最后一项应该是 s_t 和 R_t，以及一个 a_t 的占位符
            
            s_seq = np.array(states[start_idx:])
            # 补当前动作占位 (0)
            a_seq = np.array(actions[start_idx:] + [0]) 
            r_seq = np.array(rtgs[start_idx:])
            t_seq = np.array(timesteps[start_idx:])
            
            # 归一化 State
            s_seq = s_seq.copy()
            s_seq[:, :3] /= POS_SCALE
            s_seq[:, 3:] /= RPY_SCALE
            
            # 转 Tensor并增加 Batch 维度 (1, Seq_Len, Dim)
            s_tensor = torch.from_numpy(s_seq).float().unsqueeze(0).to(DEVICE)
            a_tensor = torch.from_numpy(a_seq).long().unsqueeze(0).to(DEVICE)
            r_tensor = torch.from_numpy(r_seq).float().unsqueeze(0).unsqueeze(2).to(DEVICE)
            t_tensor = torch.from_numpy(t_seq).long().unsqueeze(0).to(DEVICE)
            
            # --- 5. 模型推理 ---
            with torch.no_grad():
                # 使用 forward 计算
                action_preds = model.forward(s_tensor, a_tensor, r_tensor, t_tensor)
                
                # 取序列最后一个时间步的输出
                last_logits = action_preds[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                
                if stochastic:
                     # 随机采样
                    action_id = torch.multinomial(probs, num_samples=1).item()
                else:
                     # 贪婪采样
                    action_id = torch.argmax(probs).item()
            
            # --- 6. 执行动作 ---
            # 记录执行的动作
            actions.append(action_id)
            
            # 环境物理执行
            start_pos = env.pos[0].copy()
            displacement = action_space.get_displacement(action_id)
            target_pos = start_pos + displacement

            step_reward = 0.0
            # 高度限制检查
            if 0.5 <= target_pos[2] <= 3.5:
                p.resetBasePositionAndOrientation(env.DRONE_IDS[0], target_pos, [0, 0, 0, 1], physicsClientId=env.CLIENT)
                env._updateAndStoreKinematicInformation()
                
                # 触发扫描并计算奖励
                _, step_reward, terminated, truncated, _ = env.compute_scan_at_pos(target_pos)
                
                total_reward += step_reward
                if terminated:
                    print(f"💥 碰撞发生! 步数: {step_count}")
                    done = True
                    collision = True
            else:
                # 越界惩罚 / 结束
                # print(f"⚠️ 动作越界: {action_id} -> Target Z: {target_pos[2]:.2f}")
                step_reward = 0
                done = True 
                collision = True
            
            # --- 7. 更新历史信息 ---
            next_state = np.concatenate([env.pos[0], env.rpy[0]])
            states.append(next_state)
            rewards.append(step_reward)
            
            # 更新 Returns-to-go
            # R_{t+1} = R_t - r_t
            current_rtg = rtgs[-1] - step_reward
            rtgs.append(current_rtg)
            
            timesteps.append(step_count + 1)
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"Step {step_count} | Action: {action_id} | Reward: {step_reward:.2f} | RTG: {current_rtg:.2f} | Cov: {env._computeInfo()['coverage_ratio']:.2%}")
                
            time.sleep(0.01)

        final_coverage = env._computeInfo()['coverage_ratio']
        episode_rewards.append(total_reward)
        episode_coverages.append(final_coverage)

        print(f"🏁 Episode {episode + 1} End | Total Reward: {total_reward:.2f} | Final Coverage: {final_coverage:.2%}")

    env.close()

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_coverage = np.mean(episode_coverages)
    std_coverage = np.std(episode_coverages)
    
    print("\n" + "="*40)
    print(f"📊 Test Summary ({num_test_episodes} Episodes)")
    print(f"Average Reward:   {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Coverage: {avg_coverage:.2%} ± {std_coverage:.2%}")
    
    return avg_reward, std_reward, avg_coverage, std_coverage

if __name__ == "__main__":
    import csv
    
    EPOCHS = 20000
    INTERVAL = 500
    RESULTS_FILE = "sge_dt_benchmark_results.csv"
    
    all_results = []

    for epoch in range(0, EPOCHS + 1, INTERVAL):
        MODEL_NAME = f"sge_dt_model_{epoch}.pth"
        MODEL_PATH = None
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
            print(f">>> Testing Checkpoint: {epoch} epochs <<<")
            avg_r, std_r, avg_c, std_c = test_decision_transformer(MODEL_PATH, num_test_episodes=20, target_return=1000.0, stochastic=True)
            all_results.append([epoch, avg_r, std_r, avg_c, std_c])
        else:
             print(f"Skipping epoch {epoch}: Model not found.")
    # Save results to file
    with open(RESULTS_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Average Reward', 'Reward Std', 'Average Coverage', 'Coverage Std'])
        writer.writerows(all_results)
    print(f"Detailed results saved to {RESULTS_FILE}")
    print("="*40 + "\n")


