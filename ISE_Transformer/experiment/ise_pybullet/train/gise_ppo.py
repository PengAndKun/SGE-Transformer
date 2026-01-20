
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pybullet as p
from tqdm import tqdm
import torch.nn.functional as F

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
        logits = self.net(state)
        # Apply Softmax to get probabilities (similar to reference implementation)
        return torch.softmax(logits, dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), # Modified: +1 for previous reward
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.net(state)

class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, gae_lambda, K_epochs, eps_clip, device):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.mse_loss = nn.MSELoss()
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), torch.log(probs.squeeze(0)[action.item()] + 1e-10).item()

    def update(self, memory):
        states = torch.tensor(np.array(memory.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(memory.actions), dtype=torch.float32).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(memory.rewards), dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(memory.next_states), dtype=torch.float32).to(self.device)
        is_terminals = torch.tensor(np.array(memory.is_terminals), dtype=torch.float32).view(-1, 1).to(self.device)
        
        # We need V(s) and V(s')
        v_s = self.critic(states)
        v_next = self.critic(next_states)
        
        td_target = rewards + self.gamma * v_next * (1 - is_terminals)
        td_delta = td_target - v_s
        
        # Compute Advantage (GAE) manually as reference does with rl_utils
        advantages = []
        advantage = 0.0
        td_delta_cpu = td_delta.cpu().detach().numpy()
        for delta in td_delta_cpu[::-1]:
            advantage = self.gamma * self.gae_lambda * advantage + delta
            advantages.insert(0, advantage)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32).view(-1, 1).to(self.device)
        
        # Recalculate old_log_probs for the batch once (detach)
        # old_probs = self.actor(states)
        old_log_probs = torch.log(self.actor(states).gather(1, actions.long())).detach()
        
        for _ in range(self.K_epochs):
            # Calculate current log probs and values
            probs = self.actor(states)
            log_probs = torch.log(probs.gather(1, actions.long()))
            
            # Ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Surrogate Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.rewards[:]
        del self.is_terminals[:]

def train():
    # Parameters matches sge_mlp_bc where possible
    STATE_DIM = 6
    NUM_ACTIONS = 27
    LR = 1e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    K_EPOCHS = 4
    EPS_CLIP = 0.2
    MAX_EPISODES = 20000 
    MAX_STEPS = 150
    # Update per episode as requested
    
    POS_SCALE = 25.0
    RPY_SCALE = 3.14
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    ppo_agent = PPO(STATE_DIM, NUM_ACTIONS, LR, GAMMA, GAE_LAMBDA, K_EPOCHS, EPS_CLIP, DEVICE)
    memory = Memory()

    # Save untrained model (Episode 0)
    save_path = os.path.join(current_dir, "sge_ppo_model_0.pth")
    torch.save(ppo_agent.actor.state_dict(), save_path)
    
    pbar = tqdm(range(1, MAX_EPISODES + 1), desc="PPO Training", unit="episode")
    
    for i_episode in pbar:
        env.restore_snapshot(start_snapshot)
        state_vec = np.concatenate([env.pos[0], env.rpy[0]]).astype(np.float32)
        # Normalize
        state_vec[:3] /= POS_SCALE
        state_vec[3:] /= RPY_SCALE
        
        # Initial previous reward is 0
        current_state = np.append(state_vec, 0.0).astype(np.float32)
        
        ep_reward = 0
        
        for t in range(MAX_STEPS):
            
            action, _ = ppo_agent.select_action(current_state)
            
            # Execute action logic:
            start_pos = env.pos[0].copy()
            displacement = action_space.get_displacement(action)
            target_pos = start_pos + displacement
            
            reward = 0
            done = False
            
            # Bounds check & Teleport & Scan
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
            
            # Next state
            next_state_vec = np.concatenate([env.pos[0], env.rpy[0]]).astype(np.float32)
            next_state_vec[:3] /= POS_SCALE
            next_state_vec[3:] /= RPY_SCALE
            
            # Append current reward as previous reward for the next state
            next_state = np.append(next_state_vec, reward).astype(np.float32)
            
            memory.states.append(current_state)
            memory.actions.append(action)
            memory.next_states.append(next_state)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            current_state = next_state
            ep_reward += reward
            
            if done:
                break
        
        # Update PPO after EACH episode (single trajectory batch)
        ppo_agent.update(memory)
        memory.clear()
        
        if i_episode % 1 == 0:
            pbar.set_postfix({'Reward': f'{ep_reward:.2f}'})

        if i_episode % 500 == 0:
            # Save to current dir, consistent with gise_mlp_bc.py
            save_path = os.path.join(current_dir, f"sge_ppo_model_{i_episode}.pth")
            torch.save(ppo_agent.actor.state_dict(), save_path)
            # pbar.write(f"Saved model to {save_path}")
            
    env.close()

if __name__ == '__main__':
    train()
