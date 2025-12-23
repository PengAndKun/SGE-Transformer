import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from typing import List, Tuple, Dict, Optional
import math
import time
from tqdm import tqdm
import random
import csv
from mamba_ssm import Mamba

class Mamba2BCNetwork(nn.Module):
    """Behavior cloning network based on mamba-ssm"""

    def __init__(self,
                 state_dim: int,
                 num_actions: int = 9,
                 d_model: int = 128,
                 num_decoder_layers: int = 3,
                 max_seq_length: int = 200):  # max_seq_length 在此模型中不直接使用，但为保持接口一致性而保留  Max_Sq_1ength is not directly used in this model, but is reserved to maintain interface consistency
        super().__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.max_seq_length = max_seq_length

        # Input embedding layer
        self.input_embedding = nn.Linear(state_dim, d_model)

        # #The Mamba model of Mamba-ssm is a ModulaList that requires manual stacking
        self.layers = nn.ModuleList([Mamba(d_model=d_model) for _ in range(num_decoder_layers)])

        # Output Head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_actions)
        )

    def forward(self, state_sequences):
        """
        forward propagation

        Args:
            state_sequences: (batch_size, seq_len, state_dim)
        """
        # input embedding
        x = self.input_embedding(state_sequences)  # (batch, seq_len, d_model)

        # Pass through each Mamba layer in sequence
        for layer in self.layers:
            x = layer(x)

        # Output prediction (using only the last time step)
        logits = self.output_head(x[:, -1, :])  # (batch, num_actions)

        return logits

    def get_action(self, state_sequences, deterministic=True):
        """Obtain the index of discrete actions"""
        logits = self.forward(state_sequences)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action_idx = torch.argmax(probs, dim=-1)
        else:
            action_idx = torch.multinomial(probs, 1).squeeze(-1)

        return action_idx, probs


class Mamba2BCTrainer:
    def __init__(self,
                 network: Mamba2BCNetwork,
                 lr: float = 1e-4,
                 device: str = None,
                 ):

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.empty_cache()
                print(f" Training with GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print(" Training with CPU")
        else:
            self.device = torch.device(device)

        # Initialize network
        self.network = network.to(self.device)
        self.state_dim = self.network.state_dim
        self.num_actions = self.network.num_actions
        self.max_seq_length = self.network.max_seq_length

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.training_losses = []

    def learn(self, state_sequences, action_sequences, batch_size, num_epochs: int):
        if len(state_sequences)==0 or len(action_sequences)==0:
             return

        all_state_sequences =torch.FloatTensor(np.array(state_sequences)).to(self.device)
        all_action_sequences = torch.LongTensor(np.array(action_sequences)).to(self.device)


        start_time = time.time()
        epoch_pbar = tqdm(range(num_epochs), desc="training", unit="epoch")

        for epoch in epoch_pbar:
            self.network.train()

            sample_indices = np.random.randint(low=0, high=len(all_state_sequences), size=batch_size)

            batch_sequences = all_state_sequences[sample_indices]
            batch_targets = all_action_sequences[sample_indices]


            self.optimizer.zero_grad()
            logits = self.network(batch_sequences)
            loss = F.cross_entropy(logits, batch_targets)
            loss.backward()
            self.optimizer.step()

        end_time = time.time()
        print(f"completed in {end_time - start_time:.2f} seconds")

    def save_model(self, filename: str):
        torch.save({'network_state_dict': self.network.state_dict()}, filename)
        print(f"✓ The model has been saved: {filename}")

    def load_model(self, filename: str):
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.network.load_state_dict(
                checkpoint['network_state_dict']);
            print(f"✓ The model has been loaded: {filename}")
        except FileNotFoundError:
            print(f"✗ The file does not exist: {filename}")

    def predict(self, state_history: List[np.ndarray],H: int, deterministic: bool = False):
        self.network.eval()
        with torch.no_grad():

            states_array = np.array(state_history, dtype=np.int64)

            padded_sequence = np.zeros((states_array.shape[0],H-1, self.state_dim), dtype=np.float32)

            for i, seq in enumerate(states_array):
                for j,seq_j in enumerate(seq):
                    padded_sequence[i,-len(seq)+j,seq_j] = 1.0

            sequence_tensor = torch.FloatTensor(padded_sequence).to(self.device)
            action_idx, probs = self.network.get_action(sequence_tensor, deterministic)
            return action_idx.cpu(), probs.cpu().numpy()