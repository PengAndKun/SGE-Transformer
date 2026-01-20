import numpy as np
import torch
import torch.nn.functional as F
import pybullet as p
import time
import os
import sys

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

from SGE_Transformer.envs.coveragel_multi_room import CoverageAviary
from SGE_Transformer.experiment.sge_pybullet.extract_trajectories_macro import MacroActionSpace27
from SGE_Transformer.experiment.sge_pybullet.sge_iql import IQLNetwork

def test_iql(model_path, num_test_episodes=10, stochastic=True):
    STATE_DIM = 6
    NUM_ACTIONS = 27
    HIDDEN_DIM = 256
    MOVE_DISTANCE = 0.5
    
    POS_SCALE = 25.0
    RPY_SCALE = 3.14
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading IQL model from: {model_path}")
    network = IQLNetwork(STATE_DIM, NUM_ACTIONS, HIDDEN_DIM).to(DEVICE)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        # Handle the dict structure we created
        state_dict = checkpoint['network_state_dict']
        network.v_net.load_state_dict(state_dict['v_net'])
        network.q_net.load_state_dict(state_dict['q_net'])
        network.pi_net.load_state_dict(state_dict['pi_net'])
        
        network.eval()
        print(f"✅ Successfully loaded model")
    else:
        print(f"❌ Model file not found: {model_path}")
        return

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
        
        total_reward = 0.0
        done = False
        step_count = 0
        max_steps = 150
        collision = False

        print(f"\n--- Episode {episode + 1} Start ---")

        while not done and step_count < max_steps:
            current_state = np.concatenate([env.pos[0], env.rpy[0]]).astype(np.float32)
            input_state = current_state.copy()
            input_state[:3] /= POS_SCALE
            input_state[3:] /= RPY_SCALE
            
            input_tensor = torch.FloatTensor(input_state).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                # IQL uses Policy Network for action selection
                action_id, probs = network.get_action(input_tensor, deterministic=not stochastic)
                action_id = action_id.item()

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
                reward = 0
                done = True
                collision = True

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
    RESULTS_FILE = "iql_benchmark_results.csv"

    all_results = []

    for epoch in range(0, EPOCHS + 1, INTERVAL):
        MODEL_NAME = f"sge_iql_model_{epoch}.pth"
        PATHS_TO_CHECK = [
            MODEL_NAME,
            # os.path.join(os.path.dirname(__file__), MODEL_NAME),
            "SGE_Transformer/experiment/sge_pybullet/" + MODEL_NAME
        ]
        
        MODEL_PATH = None
        # Simple check in cur dir or relative
        if os.path.exists(MODEL_NAME): MODEL_PATH = MODEL_NAME
        elif os.path.exists(os.path.join("SGE_Transformer/experiment/sge_pybullet/", MODEL_NAME)):
             MODEL_PATH = os.path.join("SGE_Transformer/experiment/sge_pybullet/", MODEL_NAME)

        if MODEL_PATH:
            print(f">>> Testing Checkpoint: {epoch} <<<")
            avg_rew, std_rew, avg_cov, std_cov = test_iql(MODEL_PATH, num_test_episodes=20, stochastic=True)
            all_results.append([epoch, avg_rew, std_rew, avg_cov, std_cov])
        else:
            print(f"Skipping epoch {epoch}: Model file not found.")

    # Save results to file
    with open(RESULTS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average Reward', 'Reward Std', 'Average Coverage', 'Coverage Std'])
        writer.writerows(all_results)
    print(f"Detailed results saved to {RESULTS_FILE}")
