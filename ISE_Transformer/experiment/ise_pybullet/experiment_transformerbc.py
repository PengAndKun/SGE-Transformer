import numpy as np
import torch
import torch.nn.functional as F
import pybullet as p
import time
import os

# 导入你的环境、模型类和宏动作空间
from ISE_Transformer.envs.coverage_visibility_pointcloud_aviary_optimized_add_Control import CoverageAviary


# 假设你的定义在之前的文件中，这里需要确保能引用到
from ISE_Transformer.experiment.ise_pybullet.transformerbc import HuggingFaceTransformerBCNetwork
from ISE_Transformer.experiment.ise_pybullet.extract_trajectories_macro import MacroActionSpace27

def agent_performance(model_path, num_test_episodes=5):
    # --- 1. 参数配置 (必须与训练时完全一致) ---
    STATE_DIM = 6
    NUM_ACTIONS = 27
    H = 8  # 观察的历史长度
    MOVE_DISTANCE = 0.5

    # 归一化参数 (建议与训练代码中使用的缩放比例一致)
    POS_SCALE = 25.0
    RPY_SCALE = 3.14

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. 加载模型 ---
    network = HuggingFaceTransformerBCNetwork(
        state_dim=STATE_DIM,
        num_actions=NUM_ACTIONS,
        max_seq_length=H
    ).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()
    print(f"✅ 成功加载模型: {model_path}")

    # --- 3. 初始化环境 ---
    env = CoverageAviary(
        gui=True,  # 开启界面观察
        obstacles=True,
        num_rays=120,
        radar_radius=8.0,
        grid_res=0.5
    )
    action_space = MacroActionSpace27(move_distance=MOVE_DISTANCE)

    obs, info = env.reset()
    start_snapshot = env.get_snapshot()

    for episode in range(num_test_episodes):

        env.restore_snapshot(start_snapshot)

        # 初始化状态历史队列 (用于 Transformer 输入)
        # 初始状态：[x, y, z, r, p, y]
        init_state = np.concatenate([env.pos[0], env.rpy[0]])
        state_history = [init_state] * H  # 初始用起始状态填满窗口

        total_reward = 0.0
        done = False
        step_count = 0
        max_steps = 150  # 设置一个最大测试步数

        print(f"\n--- Episode {episode + 1} Start ---")

        while not done and step_count < max_steps:
            # --- 4. 准备 Transformer 输入 ---
            # 取最近的 H 个状态并进行归一化
            input_states = np.array(state_history[-H:], dtype=np.float32)
            input_states[:, :3] /= POS_SCALE
            input_states[:, 3:] /= RPY_SCALE

            # 转为 Tensor (Batch_size=1, Seq_len=H, State_dim=6)
            input_tensor = torch.FloatTensor(input_states).unsqueeze(0).to(DEVICE)

            # --- 5. 模型预测动作 ---
            with torch.no_grad():
                logits = network(input_tensor)
                probs = F.softmax(logits, dim=-1)
                # 评估时通常使用 deterministic (argmax)
                action_id = torch.argmax(probs, dim=-1).item()

            # --- 6. 在环境中执行动作 (平移逻辑) ---
            # 这里调用你之前的 execute_translation_step 逻辑
            # 为了简化，直接在测试脚本里快速实现单步平移
            start_pos = env.pos[0].copy()
            displacement = action_space.get_displacement(action_id)
            target_pos = start_pos + displacement

            # 简单检查限高 (与训练逻辑对齐)
            if 0.5 <= target_pos[2] <= 3.5:
                # 瞬移并扫描
                p.resetBasePositionAndOrientation(env.DRONE_IDS[0], target_pos, [0, 0, 0, 1],
                                                  physicsClientId=env.CLIENT)
                env._updateAndStoreKinematicInformation()
                _, reward, terminated, truncated, _ = env.compute_scan_at_pos(target_pos)

                total_reward += reward
                if terminated:
                    print(f"💥 碰撞发生! 步数: {step_count}")
                    done = True
            else:
                print(f"⚠️ 动作越界(限高拦截)! Action: {action_id}")
                reward = 0
                done = True  # 或者跳过该动作

            # --- 7. 更新状态历史 ---
            current_state = np.concatenate([env.pos[0], env.rpy[0]])
            state_history.append(current_state)

            step_count += 1
            if step_count % 10 == 0:
                print(f"Step {step_count} | 当前累积覆盖奖励: {total_reward:.2f}")

            time.sleep(0.02)  # 稍微减速方便观察

        print(
            f"🏁 Episode {episode + 1} 结束 | 总得分: {total_reward:.2f} | 覆盖率: {env._computeInfo()['coverage_ratio']:.2%}")

    env.close()


if __name__ == "__main__":
    MODEL_FILE = "sge_bc_model.pth"
    if os.path.exists(MODEL_FILE):
        agent_performance(MODEL_FILE)
    else:
        print("未找到模型文件，请先运行训练脚本。")