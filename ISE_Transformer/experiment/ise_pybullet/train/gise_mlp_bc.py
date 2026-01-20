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
import sys

class MLPBCNetwork(nn.Module):
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
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action_idx = torch.argmax(probs, dim=-1)
        else:
            action_idx = torch.multinomial(probs, 1).squeeze(-1)
        return action_idx, probs

class MLPBCTrainer:
    def __init__(self, network: MLPBCNetwork, lr: float = 1e-4, device: str = None):
        self.device = device if device is not None else torch.device('cpu')
        self.network = network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def prepare_dataset(self, trajectories: List[dict]):
        all_states = []
        all_actions = []

        for traj in trajectories:
            states = traj['state_sequence']   # (N+1, 6)
            actions = traj['action_sequence'] # (N,)

            # Flatten transitions
            for i in range(len(actions)):
                all_states.append(states[i])
                all_actions.append(actions[i])

        return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.int64)

    def learn(self, states, actions, batch_size, num_epochs):
        if len(states) == 0:
            return

        all_states = torch.FloatTensor(states).to(self.device)
        all_actions = torch.LongTensor(actions).to(self.device)
        
        dataset_size = len(all_states)
        indices = torch.arange(dataset_size)

        self.network.train()
        
        epoch_pbar = tqdm(range(num_epochs), desc="MLP BC Training", unit="epoch")
        
        for epoch in epoch_pbar:
            # Consistent with sge_gptneo_bc: 1 Epoch = 1 Batch Update
            # Random sampling one batch per epoch
            batch_idx = torch.randperm(dataset_size)[:batch_size]
            
            batch_states = all_states[batch_idx]
            batch_actions = all_actions[batch_idx]

            self.optimizer.zero_grad()
            logits = self.network(batch_states)
            loss = F.cross_entropy(logits, batch_actions)
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                epoch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if (epoch ) % 500 == 0:
                self.save_model(f"sge_mlp_bc_model_{epoch}.pth")

    def train_from_pkl(self, pkl_path: str, batch_size: int, epochs: int):
        if not os.path.exists(pkl_path):
            print(f"Error: {pkl_path} not found.")
            return

        with open(pkl_path, 'rb') as f:
            trajectories = pickle.load(f)

        print(f"Loading {len(trajectories)} trajectories for BC training...")
        s_train, a_train = self.prepare_dataset(trajectories)
        
        # State Norm
        s_train = s_train.copy()
        s_train[..., :3] /= 25.0
        s_train[..., 3:] /= 3.14

        print(f"Dataset prepared: {s_train.shape} samples")
        self.learn(s_train, a_train, batch_size, epochs)
        
    def save_model(self, filename: str):
        torch.save({'network_state_dict': self.network.state_dict()}, filename)
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

    print("Initializing MLP BC Network...")
    network = MLPBCNetwork(STATE_DIM, NUM_ACTIONS, HIDDEN_DIM)
    trainer = MLPBCTrainer(network, lr=1e-4, device=DEVICE)  # Consistent learning rate

    if os.path.exists(PKL_FILE):
        trainer.train_from_pkl(PKL_FILE, BATCH_SIZE, EPOCHS)
        trainer.save_model(f"sge_mlp_bc_model_{EPOCHS}.pth")
    else:
        print(f"Error: {PKL_FILE} not found.")
