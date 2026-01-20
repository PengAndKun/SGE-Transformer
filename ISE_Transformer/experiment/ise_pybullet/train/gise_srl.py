import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import pybullet as p
from tqdm import tqdm

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
    # Fallback if running from root
    sys.path.append(os.getcwd())
    from SGE_Transformer.envs.coveragel_multi_room import CoverageAviary
    from SGE_Transformer.experiment.sge_pybullet.extract_trajectories_macro import MacroActionSpace27

class SRLActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SRLActor, self).__init__()
        # subrl/main.py Logic: Input is State + Time Index
        # SGE State (6) + Time (1) = 7
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        logits = self.net(state)
        # subrl/main.py uses Softmax output for Probabilities
        return torch.softmax(logits, dim=-1)

def train():
    # --- Config (Consistent with SGE) ---
    STATE_DIM = 6
    NUM_ACTIONS = 27
    LR = 1e-4 # Consistent with main.py/SGE
    
    # subrl params adjusted for SGE
    MAX_EPISODES = 20000 
    MAX_STEPS = 150 # Horizon (H)
    ENT_COEF = 0.1  # Entropy coefficient from subrl/params typically
    
    POS_SCALE = 25.0
    RPY_SCALE = 3.14
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Env Setup ---
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

    # --- Agent Setup ---
    # Input dim is 6, but we will append time manually in the loop
    agent = SRLActor(STATE_DIM, NUM_ACTIONS).to(DEVICE)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    # Save untrained model
    save_path = os.path.join(current_dir, "sge_srl_model_0.pth")
    torch.save(agent.state_dict(), save_path)
    
    pbar = tqdm(range(1, MAX_EPISODES + 1), desc="SRL Training", unit="episode")
    
    for i_episode in pbar:
        env.restore_snapshot(start_snapshot)
        
        # Buffer for Update
        mat_action = [] # Tensor of actions
        list_batch_state = [] # List of state tensors
        marginal_return = [] # List of returns (cumulative coverage)
        
        # State Prep
        current_state_raw = np.concatenate([env.pos[0], env.rpy[0]]).astype(np.float32)
        current_state_norm = current_state_raw.copy()
        current_state_norm[:3] /= POS_SCALE
        current_state_norm[3:] /= RPY_SCALE
        
        total_reward = 0.0 # Standard RL reward (coverage gain)
        cumulative_coverage = 0.0 # "Return" for subrl
        
        for h_iter in range(MAX_STEPS):
            # 1. State + Time Encoding (Logic from main.py: M/SRL type)
            # Normalize time to be roughly similar scale or just index? 
            # main.py uses raw integer h_iter. SGE NN usually expects norms.
            # Let's use h_iter / MAX_STEPS to keep it conditioned well, 
            # OR stick to raw if main.py did so. main.py used raw: append h_iter*ones...
            # We will use raw h_iter for strict adherence, but float.
            time_feature = float(h_iter) 
            
            input_state = np.append(current_state_norm, time_feature).astype(np.float32)
            input_tensor = torch.FloatTensor(input_state).unsqueeze(0).to(DEVICE)
            
            # 2. Action Selection
            action_probs = agent(input_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            # Store for update
            mat_action.append(action)
            list_batch_state.append(input_tensor)
            
            # 3. Environment Step
            start_pos = env.pos[0].copy()
            displacement = action_space.get_displacement(action.item())
            target_pos = start_pos + displacement
            
            reward = 0
            done = False
            
            if 0.5 <= target_pos[2] <= 3.5:
                p.resetBasePositionAndOrientation(env.DRONE_IDS[0], target_pos, [0, 0, 0, 1], physicsClientId=env.CLIENT)
                env._updateAndStoreKinematicInformation()
                _, r, terminated, truncated, _ = env.compute_scan_at_pos(target_pos)
                reward = r
                if terminated:
                    done = True
            else:
                reward = 0 
                done = True 
            
            # 4. Calculate Return (Logic from main.py)
            # weighted_traj_return in SGE is the Total Coverage Ratio so far
            # In SGE, `env._computeInfo()['coverage_ratio']` gives total coverage
            current_total_coverage = env._computeInfo()['coverage_ratio']
            
            # main.py: if SRL: append(current_traj_return)
            marginal_return.append(current_total_coverage)
            
            # Standard RL tracking
            total_reward += reward
            
            # Next state prep
            current_state_raw = np.concatenate([env.pos[0], env.rpy[0]]).astype(np.float32)
            current_state_norm = current_state_raw.copy()
            current_state_norm[:3] /= POS_SCALE
            current_state_norm[3:] /= RPY_SCALE
            
            if done:
                break
        
        # --- Update Step (Logic from main.py) ---
        if len(list_batch_state) > 0:
            states_visited = torch.vstack(list_batch_state).to(DEVICE)
            actions_taken = torch.tensor([a.item() for a in mat_action]).to(DEVICE)
            
            # Re-evaluate distribution 
            # (In main.py it acts on states_visited directly)
            policy_dist = Categorical(agent(states_visited))
            log_prob = policy_dist.log_prob(actions_taken)
            
            # Normalize return (MAX_Ret in main.py)
            # In SGE coverage is 0.0 to 1.0, so it's already normalized!
            # We can use it directly.
            batch_return_tensor = torch.tensor(marginal_return, dtype=torch.float32).to(DEVICE)
            
            # Loss: - (mean(log_prob * return) + entropy_bonus)
            # main.py: ent_coef / (t_eps + 1) -> Decay entropy
            entropy_loss = ENT_COEF * policy_dist.entropy().mean() / (i_episode * 0.01 + 1) # Slow decay
            
            J_obj = -1 * (torch.mean(log_prob * batch_return_tensor) + entropy_loss)
            
            optimizer.zero_grad()
            J_obj.backward()
            optimizer.step()
        
        # Logging
        if i_episode % 1 == 0:
            pbar.set_postfix({'Cov': f'{marginal_return[-1]:.4f}'})

        if i_episode % 500 == 0:
            save_path = os.path.join(current_dir, f"sge_srl_model_{i_episode}.pth")
            torch.save(agent.state_dict(), save_path)

    env.close()

if __name__ == '__main__':
    train()
