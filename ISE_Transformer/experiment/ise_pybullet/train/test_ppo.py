
import numpy as np
import torch
import torch.nn as nn
import pybullet as p
import time
import os
import sys
from torch.distributions import Categorical

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Try import
try:
    from SGE_Transformer.envs.coveragel_multi_room import CoverageAviary
    from SGE_Transformer.experiment.sge_pybullet.extract_trajectories_macro import MacroActionSpace27
except ImportError:
    sys.path.append(os.getcwd())
    from SGE_Transformer.envs.coveragel_multi_room import CoverageAviary
    from SGE_Transformer.experiment.sge_pybullet.extract_trajectories_macro import MacroActionSpace27

# Re-define Actor for loading (Must match gise_ppo.py structure, but forward supports logits for stability)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), # Modified: +1 for previous reward
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        # Return logits directly for numerical stability in Categorical
        return self.net(state)
        
    def act(self, state):
        # Use logits for Categorical distribution to avoid softmax precision issues
        logits = self.forward(state)
        
        # Custom Exploration: 0.9 Greedy, 0.1 Random Other
        if np.random.rand() < 0.9:
             return torch.argmax(logits, dim=-1).item()
        else:
             # Sample from others
             num_actions = logits.shape[-1]
             best_action = torch.argmax(logits, dim=-1).item()
             other_actions = [i for i in range(num_actions) if i != best_action]
             return np.random.choice(other_actions)

    def get_action_deterministic(self, state):
        logits = self.forward(state)
        return torch.argmax(logits, dim=-1).item()

def test_ppo(model_path, num_test_episodes=20, stochastic=True):
    STATE_DIM = 6
    NUM_ACTIONS = 27
    HIDDEN_DIM = 256
    
    POS_SCALE = 25.0
    RPY_SCALE = 3.14
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading PPO model from: {model_path}")
    policy = Actor(STATE_DIM, NUM_ACTIONS, HIDDEN_DIM).to(DEVICE)

    if os.path.exists(model_path):
        # Allow loading both direct state_dict and wrapped
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if 'network_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['network_state_dict'])
            else:
                policy.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None, None

        policy.eval()
        print(f"✅ Successfully loaded model")
    else:
        print(f"❌ Model file not found: {model_path}")
        return None, None, None, None

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
        
        total_reward = 0.0
        done = False
        step_count = 0
        max_steps = 150
        prev_reward = 0.0
        
        # print(f"\n--- Episode {episode + 1} Start ---")

        while not done and step_count < max_steps:
            current_state = np.concatenate([env.pos[0], env.rpy[0]]).astype(np.float32)
            input_state = current_state.copy()
            input_state[:3] /= POS_SCALE
            input_state[3:] /= RPY_SCALE
            
            input_state = np.append(input_state, prev_reward).astype(np.float32)

            input_tensor = torch.FloatTensor(input_state).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                if stochastic:
                    action_id = policy.act(input_tensor)
                else:
                    action_id = policy.get_action_deterministic(input_tensor)

            start_pos = env.pos[0].copy()
            displacement = action_space.get_displacement(action_id)
            target_pos = start_pos + displacement

            if 0.5 <= target_pos[2] <= 3.5:
                p.resetBasePositionAndOrientation(env.DRONE_IDS[0], target_pos, [0, 0, 0, 1], physicsClientId=env.CLIENT)
                env._updateAndStoreKinematicInformation()
                
                _, reward, terminated, truncated, _ = env.compute_scan_at_pos(target_pos)
                total_reward += reward
                
                if terminated:
                    done = True
            else:
                reward = 0
                done = True
            
            prev_reward = reward
            step_count += 1
            # time.sleep(0.01)

        final_coverage = env._computeInfo()['coverage_ratio']
        episode_rewards.append(total_reward)
        episode_coverages.append(final_coverage)

        # print(f"🏁 Episode {episode + 1} End | Total Reward: {total_reward:.2f} | Coverage: {final_coverage:.2%}")

    env.close()
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_coverage = np.mean(episode_coverages)
    std_coverage = np.std(episode_coverages)
    
    return avg_reward, std_reward, avg_coverage, std_coverage

if __name__ == "__main__":
    import csv

    EPOCHS = 20000
    INTERVAL = 500
    RESULTS_FILE = "ppo_benchmark_results.csv"

    all_results = []
    
    print(f"Starting Benchmark... Saving to {RESULTS_FILE}")

    for epoch in range(0, EPOCHS + 1, INTERVAL):
        MODEL_NAME = f"sge_ppo_model_{epoch}.pth"
        
        # Check paths
        MODEL_PATH = None
        if os.path.exists(MODEL_NAME): 
            MODEL_PATH = MODEL_NAME
        else:
             candidates = [
                 os.path.join(os.path.dirname(__file__), MODEL_NAME),
                 "SGE_Transformer/experiment/sge_pybullet/" + MODEL_NAME,
                 os.path.join(current_dir, MODEL_NAME)
             ]
             for c in candidates:
                 if os.path.exists(c):
                     MODEL_PATH = c
                     break
        
        if MODEL_PATH:
            print(f">>> Testing Checkpoint: {epoch} <<<")
            avg_rew, std_rew, avg_cov, std_cov = test_ppo(MODEL_PATH, num_test_episodes=20, stochastic=True)
            if avg_rew is not None:
                all_results.append([epoch, avg_rew, std_rew, avg_cov, std_cov])
                print(f"Result: R={avg_rew:.2f}, C={avg_cov:.2%}")
        else:
            print(f"Skipping epoch {epoch}: Model file not found.")

    # Save results to file
    with open(RESULTS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average Reward', 'Reward Std', 'Average Coverage', 'Coverage Std'])
        writer.writerows(all_results)
    print(f"Detailed results saved to {RESULTS_FILE}")
