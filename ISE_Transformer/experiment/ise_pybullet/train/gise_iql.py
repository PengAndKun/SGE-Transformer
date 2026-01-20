import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from typing import List
import time
from tqdm import tqdm
import os
import copy

class IQLNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        # V-Network
        self.v_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q-Network (Discrete actions)
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        # Policy Network
        self.pi_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def get_action(self, state, deterministic=True):
        logits = self.pi_net(state)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action_idx = torch.argmax(probs, dim=-1)
        else:
            action_idx = torch.multinomial(probs, 1).squeeze(-1)
        return action_idx, probs

class IQLTrainer:
    def __init__(self, network: IQLNetwork, lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.7, beta: float = 3.0, device: str = None):
        self.device = device if device is not None else torch.device('cpu')
        self.network = network.to(self.device)
        self.target_q_net = copy.deepcopy(self.network.q_net).to(self.device) # Not strictly needed for IQL but good practice?
        # Actually IQL uses V(s') for target, so we assume V is stable enough or use target V?
        # Standard IQL uses target Q sometimes, but often just single Q.
        # Let's simple use single Q and V.
        
        self.v_optimizer = optim.Adam(self.network.v_net.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.network.q_net.parameters(), lr=lr)
        self.pi_optimizer = optim.Adam(self.network.pi_net.parameters(), lr=lr)
        
        self.gamma = gamma
        self.tau = tau
        self.beta = beta

    def prepare_dataset(self, trajectories: List[dict]):
        # Same as CQL
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []

        for traj in trajectories:
            states = traj['state_sequence']
            actions = traj['action_sequence']
            rewards = traj.get('reward_sequence', [0]*len(actions))
            if len(rewards) == 0: rewards = [0.0]*len(actions)

            N = len(actions)
            for i in range(N):
                all_states.append(states[i])
                all_actions.append(actions[i])
                all_rewards.append(rewards[i] / 100.0) # Reward Scaling
                all_next_states.append(states[i+1])
                is_done = (i == N - 1)
                all_dones.append(1.0 if is_done else 0.0)

        return (
            np.array(all_states, dtype=np.float32), 
            np.array(all_actions, dtype=np.int64),
            np.array(all_rewards, dtype=np.float32),
            np.array(all_next_states, dtype=np.float32),
            np.array(all_dones, dtype=np.float32)
        )

    def expectile_loss(self, diff, tau):
        weight = torch.where(diff > 0, tau, 1 - tau)
        return weight * (diff**2)

    def learn(self, states, actions, rewards, next_states, dones, batch_size, num_epochs):
        dataset_size = len(states)
        if dataset_size == 0: return

        t_states = torch.FloatTensor(states).to(self.device)
        t_actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        t_rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        t_next_states = torch.FloatTensor(next_states).to(self.device)
        t_dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        self.network.train()
        
        epoch_pbar = tqdm(range(num_epochs), desc="IQL Training", unit="epoch")
        
        for epoch in epoch_pbar:
            # Consistent with sge_gptneo_bc: 1 Epoch = 1 Batch Update
            batch_idx = torch.randperm(dataset_size)[:batch_size]

            b_s = t_states[batch_idx]
            b_a = t_actions[batch_idx]
            b_r = t_rewards[batch_idx]
            b_ns = t_next_states[batch_idx]
            b_d = t_dones[batch_idx]

            # --- 1. Train Value Net (Expectile Regression) ---
            with torch.no_grad():
                # Target is Q(s,a)
                # We use current Q network
                current_q = self.network.q_net(b_s)
                target_q_for_v = current_q.gather(1, b_a)

            current_v = self.network.v_net(b_s)
            diff = target_q_for_v - current_v
            v_loss = self.expectile_loss(diff, self.tau).mean()

            self.v_optimizer.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.v_net.parameters(), 1.0)
            self.v_optimizer.step()

            # --- 2. Train Q Net (MSE) ---
            with torch.no_grad():
                # Target Q = r + gamma * V(s')
                next_v = self.network.v_net(b_ns)
                target_q = b_r + self.gamma * (1 - b_d) * next_v

            current_q_vals = self.network.q_net(b_s)
            q_pred = current_q_vals.gather(1, b_a)
            q_loss = F.mse_loss(q_pred, target_q)

            self.q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.q_net.parameters(), 1.0)
            self.q_optimizer.step()

            # --- 3. Train Policy (Advantage Weighted Regression) ---
            # A = Q(s,a) - V(s)
            # Weights = exp(beta * A)
            with torch.no_grad():
                curr_q = self.network.q_net(b_s).gather(1, b_a)
                curr_v = self.network.v_net(b_s)
                adv = curr_q - curr_v
                weights = torch.exp(self.beta * adv)
                # Clip weights for stability
                weights = torch.clamp(weights, max=100.0)

            pi_logits = self.network.pi_net(b_s)
            # Log prob of actions taken in dataset
            log_probs = F.log_softmax(pi_logits, dim=-1)
            log_prob_a = log_probs.gather(1, b_a)
            
            pi_loss = -(weights * log_prob_a).mean()

            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.pi_net.parameters(), 1.0)
            self.pi_optimizer.step()
            
            if (epoch+1) % 10 == 0:
                epoch_pbar.set_postfix({
                    'v_loss': f'{v_loss.item():.3f}',
                    'q_loss': f'{q_loss.item():.3f}',
                    'pi_loss': f'{pi_loss.item():.3f}'
                })
            
            if (epoch ) % 500 == 0:
                self.save_model(f"sge_iql_model_{epoch}.pth")

    def train_from_pkl(self, pkl_path: str, batch_size: int, epochs: int):
        if not os.path.exists(pkl_path):
            print(f"Error: {pkl_path} not found.")
            return

        with open(pkl_path, 'rb') as f:
            trajectories = pickle.load(f)

        print(f"Loading {len(trajectories)} trajectories for IQL training...")
        s, a, r, ns, d = self.prepare_dataset(trajectories)
        
        # Norm
        s = s.copy(); ns = ns.copy()
        s[..., :3] /= 25.0; s[..., 3:] /= 3.14
        ns[..., :3] /= 25.0; ns[..., 3:] /= 3.14

        print(f"Dataset prepared: {s.shape} samples")
        self.learn(s, a, r, ns, d, batch_size, epochs)
        
    def save_model(self, filename: str):
        # Save all components
        torch.save({
            'network_state_dict': {
                'v_net': self.network.v_net.state_dict(),
                'q_net': self.network.q_net.state_dict(),
                'pi_net': self.network.pi_net.state_dict()
            }
        }, filename)
        print(f"✓ Model saved: {filename}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    POSSIBLE_PKL_NAME = "hybrid_trajectories_batches_8_budget_50000.pkl"
    PKL_FILE = os.path.join(current_dir, POSSIBLE_PKL_NAME)
    
    if not os.path.exists(PKL_FILE):
        PKL_FILE = POSSIBLE_PKL_NAME

    STATE_DIM = 6
    NUM_ACTIONS = 27
    HIDDEN_DIM = 256
    BATCH_SIZE = 128    # Consistent with sge_gptneo_bc
    EPOCHS = 20000      # Consistent with sge_gptneo_bc
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing IQL Network...")
    network = IQLNetwork(STATE_DIM, NUM_ACTIONS, HIDDEN_DIM)
    trainer = IQLTrainer(network, lr=1e-4, device=DEVICE) # Consistent learning rate (was 3e-4)

    if os.path.exists(PKL_FILE):
        trainer.train_from_pkl(PKL_FILE, BATCH_SIZE, EPOCHS)
        trainer.save_model(f"sge_iql_model_{EPOCHS}.pth")
    else:
        print(f"Error: {PKL_FILE} not found.")
