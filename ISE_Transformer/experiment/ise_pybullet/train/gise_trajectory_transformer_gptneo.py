import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
import sys
import time
from tqdm import tqdm
from typing import List, Tuple

# Try to import transformers
try:
    from transformers import GPTNeoModel, GPTNeoConfig
except ImportError:
    print("Error: transformers library not found. Please install it using 'pip install transformers'")
    sys.exit(1)

class TrajectoryTransformerGPTNeo(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_length: int = 20,  # Context length (K)
        max_ep_len: int = 500, # Max episode length for timestep embedding
        hidden_size: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        # GPTNeo specific
        attention_types: List =  [[["local", "global"], 2]],
        window_size: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        # --- Embedding Layers ---
        # State Embedding
        self.embed_state = nn.Linear(state_dim, hidden_size)
        # Action Embedding
        self.embed_action = nn.Embedding(act_dim, hidden_size) 
        # Reward Embedding (Immediate Reward)
        self.embed_reward = nn.Linear(1, hidden_size)
        # Timestep Embedding
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        # Layer Norm
        self.embed_ln = nn.LayerNorm(hidden_size)

        # --- GPT-Neo Backbone ---
        config = GPTNeoConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            num_layers=n_layer,
            num_heads=n_head,
            max_position_embeddings=max_length * 3 + 10, 
            attention_types=attention_types if attention_types else [[["global"], n_layer]],
            use_cache=False,
            window_size=window_size
        )
        self.transformer = GPTNeoModel(config)

        # --- Prediction Heads ---
        # 1. Action Head
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), # Using ReLU as in previous Neo implementation
            nn.Linear(hidden_size, act_dim)
        )
        # 2. Reward Head
        self.predict_reward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        # 3. State Head
        self.predict_state = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )

    def forward(self, states, actions, rewards, timesteps):
        """
        Input Sequence Order: (s_1, a_1, r_1, s_2, ...)
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Embeddings
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        reward_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        # Add timestep embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        reward_embeddings = reward_embeddings + time_embeddings

        # Stack inputs
        stacked_inputs = torch.zeros((batch_size, 3 * seq_len, self.hidden_size), device=states.device)
        stacked_inputs[:, 0::3, :] = state_embeddings
        stacked_inputs[:, 1::3, :] = action_embeddings
        stacked_inputs[:, 2::3, :] = reward_embeddings
        
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Transformer Forward
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        hidden_states = transformer_outputs.last_hidden_state

        # --- Extract Features and Predict ---
        
        # 1. Action Prediction (using s_t)
        action_feats = hidden_states[:, 0::3, :] 
        action_preds = self.predict_action(action_feats)

        # 2. Reward Prediction (using s_t, a_t -> aligned at a_t pos)
        reward_feats = hidden_states[:, 1::3, :]
        reward_preds = self.predict_reward(reward_feats)

        # 3. State Prediction (using s_t, a_t, r_t -> aligned at r_t pos)
        state_feats = hidden_states[:, 2::3, :]
        state_preds = self.predict_state(state_feats)

        return action_preds, reward_preds, state_preds


class TTGPTNeoTrainer:
    def __init__(self, model, lr=1e-4, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Loss Functions
        self.loss_cls = nn.CrossEntropyLoss(reduction='none') 
        self.loss_reg = nn.MSELoss(reduction='none')

    def prepare_data(self, trajectories, context_len):
        self.trajectories = trajectories
        self.context_len = context_len
        
        for traj in self.trajectories:
            if 'reward_sequence' not in traj:
                rewards = np.zeros(len(traj['action_sequence']))
                rewards[-1] = traj['final_reward']
                traj['rewards'] = rewards
            else:
                traj['rewards'] = np.array(traj['reward_sequence'])

    def get_batch(self, batch_size, normalize_params):
        states, actions, rewards, next_states, timesteps, masks = [], [], [], [], [], []
        
        for _ in range(batch_size):
            traj_idx = np.random.randint(len(self.trajectories))
            traj = self.trajectories[traj_idx]
            
            traj_len = len(traj['action_sequence'])
            si = np.random.randint(traj_len)
            s_start = max(0, si - self.context_len + 1)
            
            s_seg = traj['state_sequence'][s_start : si + 1]
            a_seg = traj['action_sequence'][s_start : si + 1]
            r_seg = traj['rewards'][s_start : si + 1].reshape(-1, 1)
            t_seg = np.arange(s_start, si + 1)
            
            # Next States
            full_s = traj['state_sequence']
            ns_indices = np.arange(s_start + 1, si + 2)
            ns_indices = np.clip(ns_indices, 0, len(full_s) - 1)
            ns_seg = full_s[ns_indices]
            
            # Normalize
            s_seg = s_seg.copy()
            ns_seg = ns_seg.copy()
            s_seg[:, :3] /= normalize_params['pos_scale']
            s_seg[:, 3:] /= normalize_params['rpy_scale']
            ns_seg[:, :3] /= normalize_params['pos_scale']
            ns_seg[:, 3:] /= normalize_params['rpy_scale']
            
            # Padding
            pad_len = self.context_len - len(s_seg)
            
            s_padded = np.zeros((self.context_len, 6))
            s_padded[pad_len:] = s_seg
            ns_padded = np.zeros((self.context_len, 6))
            ns_padded[pad_len:] = ns_seg
            a_padded = np.zeros((self.context_len,), dtype=np.int64)
            a_padded[pad_len:] = a_seg
            r_padded = np.zeros((self.context_len, 1))
            r_padded[pad_len:] = r_seg
            t_padded = np.zeros((self.context_len,), dtype=np.int64)
            t_padded[pad_len:] = t_seg
            mask = np.zeros((self.context_len,), dtype=np.float32)
            mask[pad_len:] = 1.0

            states.append(s_padded)
            actions.append(a_padded)
            rewards.append(r_padded)
            next_states.append(ns_padded)
            timesteps.append(t_padded)
            masks.append(mask)

        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(actions), dtype=torch.long, device=self.device),
            torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(timesteps), dtype=torch.long, device=self.device),
            torch.tensor(np.array(masks), dtype=torch.float32, device=self.device)
        )

    def train(self, steps, batch_size, normalize_params):
        self.model.train()
        pbar = tqdm(range(steps), desc="TT-GPTNeo Training")
        
        for step in pbar:
            states, actions, rewards, next_states, timesteps, masks = self.get_batch(batch_size, normalize_params)
            
            a_preds, r_preds, s_preds = self.model(states, actions, rewards, timesteps)
            
            mask = masks.reshape(-1)
            denom = mask.sum()
            
            loss_a = self.loss_cls(a_preds.reshape(-1, self.model.act_dim), actions.reshape(-1))
            loss_a = (loss_a * mask).sum() / denom
            
            loss_r = self.loss_reg(r_preds.reshape(-1, 1), rewards.reshape(-1, 1))
            loss_r = (loss_r.squeeze() * mask).sum() / denom
            
            loss_s = self.loss_reg(s_preds.reshape(-1, self.model.state_dim), next_states.reshape(-1, self.model.state_dim))
            loss_s = (loss_s.mean(dim=1) * mask).sum() / denom

            loss = loss_a + loss_r + loss_s

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (step + 1) % 10 == 0:
                pbar.set_postfix({
                    'L': f'{loss.item():.2f}', 
                    'A': f'{loss_a.item():.2f}',
                    'R': f'{loss_r.item():.2f}',
                    'S': f'{loss_s.item():.2f}'
                })
            
            if (step) % 500 == 0:
                 torch.save(self.model.state_dict(), f"sge_tt_gptneo_model_{step}.pth")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    PKL_FILE_NAME = "hybrid_trajectories_batches_8_budget_50000.pkl"
    PKL_PATHS = [
        os.path.join(current_dir, PKL_FILE_NAME),
        os.path.join(project_root, "SGE_Transformer", "experiment", "sge_pybullet", PKL_FILE_NAME),
        PKL_FILE_NAME
    ]
    
    PKL_FILE = None
    for p in PKL_PATHS:
        if os.path.exists(p):
            PKL_FILE = p
            break
            
    if not PKL_FILE:
        print(f"Error: .pkl file not found.")
        sys.exit(1)

    with open(PKL_FILE, 'rb') as f:
        trajs = pickle.load(f)

    STATE_DIM = 6
    ACT_DIM = 27
    CONTEXT_LEN = 20
    HIDDEN_SIZE = 128
    
    tt_model = TrajectoryTransformerGPTNeo(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        max_length=CONTEXT_LEN,
        n_layer=4, 
        n_head=4,
        hidden_size=HIDDEN_SIZE,
        attention_types=[[["local", "global"], 2]],
        max_ep_len=500
    )
    
    trainer = TTGPTNeoTrainer(tt_model, device="cuda" if torch.cuda.is_available() else "cpu")
    
    trainer.prepare_data(trajs, CONTEXT_LEN)
    norm_params = {'pos_scale': 25.0, 'rpy_scale': 3.14}
    
    STEPS = 20000
    trainer.train(steps=STEPS, batch_size=64, normalize_params=norm_params)
    
    SAVE_PATH = f"sge_tt_gptneo_model_{STEPS}.pth"
    torch.save(tt_model.state_dict(), SAVE_PATH)
    print(f"Model saved to: {SAVE_PATH}")
