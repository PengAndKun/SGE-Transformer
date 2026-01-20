
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    sys.path.append(os.getcwd())
    from SGE_Transformer.envs.coveragel_multi_room import CoverageAviary
    from SGE_Transformer.experiment.sge_pybullet.extract_trajectories_macro import MacroActionSpace27

class RecurrentActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(RecurrentActor, self).__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x, hidden):
        # x: (batch, seq, dim)
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(x, hidden)
        return out, hidden

    def get_logits(self, x, hidden):
        out, hidden = self.forward(x, hidden)
        logits = self.fc(out)
        return logits, hidden

class RecurrentCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(RecurrentCritic, self).__init__()
        # Critic also needs LSTM to process sequence history for state value
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, hidden=None):
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(x, hidden)
        value = self.fc(out)
        return value, hidden

class LSTMPPOTrainer:
    def __init__(self, state_dim, action_dim, lr, gamma, gae_lambda, K_epochs, eps_clip, device):
        self.actor = RecurrentActor(state_dim, action_dim).to(device)
        self.critic = RecurrentCritic(state_dim).to(device)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Policy old for sampling
        self.actor_old = RecurrentActor(state_dim, action_dim).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.mse_loss = nn.MSELoss()
        
    def get_initial_hidden(self):
        return (torch.zeros(1, 1, 256).to(self.device),
                torch.zeros(1, 1, 256).to(self.device))

    def select_action(self, state, hidden):
        # state: (1, 1, dim)
        with torch.no_grad():
            logits, hidden = self.actor_old.get_logits(state, hidden)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            # return action, hidden
            return action.item(), hidden

    def update(self, rollouts):
        # rollouts = list of (states, actions, next_states, rewards, is_terminals)
        
        for _ in range(self.K_epochs):
            loss_sum = 0
            count = 0 
            
            for states_t, actions_t, next_states_t, rewards_t, is_terminals_t in rollouts:
                # Add batch dim of 1
                states = states_t.unsqueeze(0).to(self.device)   # (1, Seq, Dim)
                actions = actions_t.unsqueeze(0).to(self.device) # (1, Seq)
                next_states = next_states_t.unsqueeze(0).to(self.device)
                rewards = rewards_t.unsqueeze(0).to(self.device).unsqueeze(-1) # (1, Seq, 1)
                is_terminals = is_terminals_t.unsqueeze(0).to(self.device).unsqueeze(-1)

                # Initialize hidden states for LSTM (zero for start of episode)
                # Note: This simplifies LSTM training by assuming full trajectory backprop from zero hidden state
                # accurate for "update per episode" strategy
                h0_actor = (torch.zeros(1, 1, 256).to(self.device), torch.zeros(1, 1, 256).to(self.device))
                h0_critic = (torch.zeros(1, 1, 256).to(self.device), torch.zeros(1, 1, 256).to(self.device))
                
                # --- Get Values ---
                v_s, _ = self.critic(states, h0_critic)
                v_next, _ = self.critic(next_states, h0_critic) # We reuse h0 because next_states IS states shifted by 1 roughly, but for proper correct LSTM next_state value, better to run sequence
                # Actually v_next for t should correspond to v_s at t+1. 
                # Let's use the standard "target = r + gamma * v(s')" formulation point-wise
                
                td_target = rewards + self.gamma * v_next * (1 - is_terminals)
                td_delta = td_target - v_s

                # --- GAE ---
                advantages = []
                advantage = 0.0
                td_delta_cpu = td_delta.squeeze(0).squeeze(-1).cpu().detach().numpy() # (Seq,)
                
                for delta in td_delta_cpu[::-1]:
                    advantage = self.gamma * self.gae_lambda * advantage + delta
                    advantages.insert(0, advantage)
                
                advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(self.device).view(1, -1, 1) # (1, Seq, 1)
                
                # --- Forward Pass Current Policy ---
                current_logits, _ = self.actor.get_logits(states, h0_actor)
                current_probs = torch.softmax(current_logits, dim=-1)
                current_log_probs = torch.log(current_probs.gather(2, actions.long().unsqueeze(-1)) + 1e-10)
                
                # --- Forward Pass Old Policy (Recalculate or Store?) ---
                # Recalulating to save memory logic from PPO
                with torch.no_grad():
                    old_logits, _ = self.actor_old.get_logits(states, h0_actor)
                    old_probs = torch.softmax(old_logits, dim=-1)
                    old_log_probs = torch.log(old_probs.gather(2, actions.long().unsqueeze(-1)) + 1e-10)

                # --- PPO Loss ---
                ratios = torch.exp(current_log_probs - old_log_probs)
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
                loss = -torch.min(surr1, surr2) + 0.5*self.mse_loss(v_s, td_target.detach())
                
                loss_sum += loss.mean()
                count += 1
            
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            (loss_sum / count).backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()
            
        self.actor_old.load_state_dict(self.actor.state_dict())

def train():
    STATE_DIM = 6
    NUM_ACTIONS = 27
    LR = 1e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    K_EPOCHS = 4
    EPS_CLIP = 0.2
    MAX_EPISODES = 20000 
    MAX_STEPS = 150
    # Update per episode
    
    POS_SCALE = 25.0
    RPY_SCALE = 3.14
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    agent = LSTMPPOTrainer(STATE_DIM, NUM_ACTIONS, LR, GAMMA, GAE_LAMBDA, K_EPOCHS, EPS_CLIP, DEVICE)
    
    rollout_buffer = []

    # Save untrained model (Episode 0)
    save_path = os.path.join(current_dir, "sge_lstm_ppo_model_0.pth")
    torch.save(agent.actor.state_dict(), save_path)

    # print(f"Starting LSTM-PPO Training for {MAX_EPISODES} episodes...")
    pbar = tqdm(range(1, MAX_EPISODES + 1), desc="LSTM-PPO Training", unit="episode")
    
    for i_episode in pbar:
        env.restore_snapshot(start_snapshot)
        current_state = np.concatenate([env.pos[0], env.rpy[0]]).astype(np.float32)
        current_state[:3] /= POS_SCALE
        current_state[3:] /= RPY_SCALE
        
        ep_reward = 0
        hidden = agent.get_initial_hidden()
        
        ep_states = []
        ep_actions = []
        ep_next_states = []
        ep_rewards = []
        ep_terminals = []
        
        for t in range(MAX_STEPS):
            state_tensor = torch.FloatTensor(current_state).to(DEVICE).view(1, 1, -1)
            
            action, hidden = agent.select_action(state_tensor, hidden)
            
            start_pos = env.pos[0].copy()
            displacement = action_space.get_displacement(action)
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
                
            next_state_raw = np.concatenate([env.pos[0], env.rpy[0]]).astype(np.float32)
            next_state = next_state_raw.copy()
            next_state[:3] /= POS_SCALE
            next_state[3:] /= RPY_SCALE
            
            ep_states.append(current_state)
            ep_actions.append(action)
            ep_next_states.append(next_state)
            ep_rewards.append(reward)
            ep_terminals.append(done)
            
            current_state = next_state
            ep_reward += reward
            
            if done:
                break
        
        # Calculate discounted rewards form episode
        # discounted_rewards = []
        # R = 0
        # for r in reversed(ep_rewards):
        #     R = r + GAMMA * R
        #     discounted_rewards.insert(0, R)
        
        # rew_tensor = torch.tensor(discounted_rewards, dtype=torch.float32)
        # if rew_tensor.std() > 1e-7:
        #     rew_tensor = (rew_tensor - rew_tensor.mean()) / (rew_tensor.std() + 1e-7)
            
        rollout_buffer.append((
            torch.tensor(np.array(ep_states), dtype=torch.float32),
            torch.tensor(np.array(ep_actions), dtype=torch.float32),
            torch.tensor(np.array(ep_next_states), dtype=torch.float32),
            torch.tensor(np.array(ep_rewards), dtype=torch.float32),
            torch.tensor(np.array(ep_terminals), dtype=torch.float32)
        ))
        
        # Update immediately
        agent.update(rollout_buffer)
        rollout_buffer = []

        if i_episode % 10 == 0:
            pbar.set_postfix({'Reward': f'{ep_reward:.2f}'})

        if i_episode % 500 == 0:
            save_path = os.path.join(current_dir, f"sge_lstm_ppo_model_{i_episode}.pth")
            torch.save(agent.actor.state_dict(), save_path)
            # pbar.write(f"Saved model to {save_path}")

    env.close()

if __name__ == '__main__':
    train()
