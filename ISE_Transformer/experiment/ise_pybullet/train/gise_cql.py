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

class CQLNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state):
        return self.net(state)

    def get_action(self, state, deterministic=True):
        # Q-learning policy: argmax Q
        # For evaluation, we usually pick argmax. 
        # For stochastic (if requested), we can use softmax on Q-values (Soft-Q).
        q_values = self.forward(state)
        
        if deterministic:
            action_idx = torch.argmax(q_values, dim=-1)
            probs = F.one_hot(action_idx, num_classes=q_values.shape[-1]).float()
        else:
            # Softmax Q-values (Boltzmann exploration-like)
            probs = F.softmax(q_values, dim=-1)
            action_idx = torch.multinomial(probs, 1).squeeze(-1)
            
        return action_idx, probs

class CQLTrainer:
    def __init__(self, network: CQLNetwork, lr: float = 1e-4, gamma: float = 0.99, cql_weight: float = 1.0, device: str = None):
        self.device = device if device is not None else torch.device('cpu')
        
        self.q_net = network.to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.gamma = gamma
        self.cql_weight = cql_weight

    def prepare_dataset(self, trajectories: List[dict]):
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []

        for traj in trajectories:
            states = traj['state_sequence']   # (N+1, 6)
            actions = traj['action_sequence'] # (N,)
            rewards = traj.get('reward_sequence', [0]*len(actions)) 
            
            # Simple fallback if rewards missing
            if len(rewards) == 0:
                 rewards = [0.0] * len(actions)

            N = len(actions)
            for i in range(N):
                all_states.append(states[i])
                all_actions.append(actions[i])
                all_rewards.append(rewards[i] / 100.0) # Reward Scaling: Divide by 100 to prevent Q-value explosion
                all_next_states.append(states[i+1])
                
                # Assume last step is done
                is_done = (i == N - 1)
                all_dones.append(1.0 if is_done else 0.0)

        return (
            np.array(all_states, dtype=np.float32), 
            np.array(all_actions, dtype=np.int64),
            np.array(all_rewards, dtype=np.float32),
            np.array(all_next_states, dtype=np.float32),
            np.array(all_dones, dtype=np.float32)
        )
        
    def learn(self, states, actions, rewards, next_states, dones, batch_size, num_epochs):
        dataset_size = len(states)
        if dataset_size == 0: return

        # To tensor
        t_states = torch.FloatTensor(states).to(self.device)
        t_actions = torch.LongTensor(actions).to(self.device).unsqueeze(1) # (B, 1)
        t_rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) # (B, 1)
        t_next_states = torch.FloatTensor(next_states).to(self.device)
        t_dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1) # (B, 1)

        self.q_net.train()
        
        epoch_pbar = tqdm(range(num_epochs), desc="CQL Training", unit="epoch")
        
        for epoch in epoch_pbar:
            # Consistent with sge_gptneo_bc: 1 Epoch = 1 Batch Update
            batch_idx = torch.randperm(dataset_size)[:batch_size]
            
            b_s = t_states[batch_idx]
            b_a = t_actions[batch_idx]
            b_r = t_rewards[batch_idx]
            b_ns = t_next_states[batch_idx]
            b_d = t_dones[batch_idx]

            # 1. Compute Target Q
            with torch.no_grad():
                # Max Q over next actions
                next_q_values = self.target_q_net(b_ns)
                max_next_q, _ = next_q_values.max(dim=1, keepdim=True)
                target_q = b_r + self.gamma * (1 - b_d) * max_next_q

            # 2. Compute Current Q
            q_values = self.q_net(b_s)
            q_pred = q_values.gather(1, b_a) # Q(s, a)

            # 3. Bellman Loss
            bellman_loss = F.mse_loss(q_pred, target_q)
            
            # 4. CQL Loss (Discrete)
            # min Q(s,a) (data) + logsumexp Q(s, a') (all actions)
            # L_cql = alpha * (logsumexp(Q(s)) - Q(s, a_data))
            cql_loss = torch.logsumexp(q_values, dim=1, keepdim=True) - q_pred
            cql_loss = cql_loss.mean()

            total_loss = bellman_loss + self.cql_weight * cql_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0) # Gradient Clipping
            self.optimizer.step()
            
            # Update Target Net sparingly
            if (epoch + 1) % 100 == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            if (epoch + 1) % 10 == 0:
                epoch_pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

            if (epoch ) % 500 == 0:
                self.save_model(f"sge_cql_model_{epoch}.pth")

    def train_from_pkl(self, pkl_path: str, batch_size: int, epochs: int):
        if not os.path.exists(pkl_path):
            print(f"Error: {pkl_path} not found.")
            return

        with open(pkl_path, 'rb') as f:
            trajectories = pickle.load(f)

        print(f"Loading {len(trajectories)} trajectories for CQL training...")
        s, a, r, ns, d = self.prepare_dataset(trajectories)
        
        # Norm
        s = s.copy(); ns = ns.copy()
        s[..., :3] /= 25.0; s[..., 3:] /= 3.14
        ns[..., :3] /= 25.0; ns[..., 3:] /= 3.14

        print(f"Dataset prepared: {s.shape} samples")
        self.learn(s, a, r, ns, d, batch_size, epochs)
        
    def save_model(self, filename: str):
        torch.save({'network_state_dict': self.q_net.state_dict()}, filename)
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

    print("Initializing CQL Network...")
    network = CQLNetwork(STATE_DIM, NUM_ACTIONS, HIDDEN_DIM)
    trainer = CQLTrainer(network, lr=1e-4, device=DEVICE) # Consistent learning rate

    if os.path.exists(PKL_FILE):
        trainer.train_from_pkl(PKL_FILE, BATCH_SIZE, EPOCHS)
        trainer.save_model(f"sge_cql_model_{EPOCHS}.pth")
    else:
        print(f"Error: {PKL_FILE} not found.")
