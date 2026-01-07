import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from typing import List, Tuple
import time
from tqdm import tqdm
import random
import os
from transformers import GPT2Model, GPT2Config


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
# import sys
# from io import StringIO
# original_stdout = sys.stdout
# sys.stdout = StringIO()

class HuggingFaceTransformerBCNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int = 9, d_model: int = 128, num_decoder_layers: int = 3, nhead: int = 8, max_seq_length: int = 200):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.max_seq_length = max_seq_length
        self.input_embedding = nn.Linear(state_dim, d_model)
        config = GPT2Config(vocab_size=1, n_positions=max_seq_length, n_embd=d_model, n_layer=num_decoder_layers, n_head=nhead, use_cache=False)
        self.transformer = GPT2Model(config)
        self.output_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model // 2, num_actions))

    def forward(self, state_sequences):
        batch_size, seq_len,_ = state_sequences.shape
        device = state_sequences.device
        inputs_embeds = self.input_embedding(state_sequences)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden_states = transformer_outputs.last_hidden_state
        last_step_output = hidden_states[:, -1, :]
        logits = self.output_head(last_step_output)
        return logits

    def get_action(self, state_sequences, deterministic=True):
        logits = self.forward(state_sequences)
        probs = F.softmax(logits, dim=-1)
        if deterministic: action_idx = torch.argmax(probs, dim=-1)
        else: action_idx = torch.multinomial(probs, 1).squeeze(-1)
        return action_idx, probs

class BCTrainer:
    def __init__(self,
                 network: HuggingFaceTransformerBCNetwork,
                 lr: float = 1e-4,
                 device: str = None):
        self.device = device if device is not None else torch.device('cpu')
        self.network = network.to(self.device)
        self.state_dim = self.network.state_dim
        self.num_actions = self.network.num_actions
        self.max_seq_length = self.network.max_seq_length

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

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
                    checkpoint['network_state_dict']); print(f"✓ The model has been loaded: {filename}")
            except FileNotFoundError:
                print(f"✗ The file does not exist: {filename}")

    def predict(self, state_history: List[np.ndarray],H: int, deterministic: bool = False):
        self.network.eval()
        with torch.no_grad():
            # seq_len = len(state_history)
            # expert_s_transformed=[]
            # for seq in zip(state_history):
            #     valid_vals = seq[seq != -1]
            #     new_arr = np.zeros_like(seq)
            #     new_arr[-len(valid_vals):] = valid_vals
            #     expert_s_transformed.append(new_arr)

            states_array = np.array(state_history, dtype=np.int64)

            padded_sequence = np.zeros((states_array.shape[0],H-1, self.state_dim), dtype=np.float32)

            for i, seq in enumerate(states_array):
                for j,seq_j in enumerate(seq):
                    padded_sequence[i,-len(seq)+j,seq_j] = 1.0
            # rows = np.arange(states_array.shape[0])[:, None]
            # cols = np.arange(states_array.shape[1])
            # padded_sequence[rows, cols, states_array] = 1.0

            # a= padded_sequence[0]
            # padded_sequence = np.zeros((self.max_seq_length, self.state_dim))
            # if seq_len <= self.max_seq_length:
            #     padded_sequence[-seq_len:] = np.array(state_history)
            # else: padded_sequence = np.array(state_history[-self.max_seq_length:])

            sequence_tensor = torch.FloatTensor(padded_sequence).to(self.device)
            action_idx, probs = self.network.get_action(sequence_tensor, deterministic)
            # print("action_idx",action_idx," ",action_idx.shape," probs", probs, "   ",probs.shape)
            return action_idx.cpu(), probs.cpu().numpy()


class SGEBCTrainer(BCTrainer):
    """
    专为 SGE 轨迹设计的训练器
    """

    def prepare_dataset(self, trajectories: List[dict], H: int):
        """
        将 SGE 轨迹切分为固定长度 H 的子序列
        """
        all_states = []
        all_actions = []

        for traj in trajectories:
            states = traj['state_sequence']  # (N+1, 6)
            actions = traj['action_sequence']  # (N,)

            # states 包含起始点，所以 states[i] 对应 action[i]
            # 我们需要长度为 H 的历史状态来预测 action
            for i in range(len(actions)):
                # 提取历史状态 [i-H+1, ..., i]
                # 如果历史不足，向左填充
                start_idx = max(0, i - H + 1)
                sub_seq = states[start_idx: i + 1]

                # 填充到固定长度 H
                if len(sub_seq) < H:
                    pad = np.tile(sub_seq[0], (H - len(sub_seq), 1))
                    sub_seq = np.vstack([pad, sub_seq])

                all_states.append(sub_seq)
                all_actions.append(actions[i])

        return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.int64)

    def train_from_pkl(self, pkl_path: str, H: int, batch_size: int, epochs: int):
        with open(pkl_path, 'rb') as f:
            trajectories = pickle.load(f)

        print(f"Loading {len(trajectories)} trajectories for BC training...")

        # 1. 准备数据
        s_train, a_train = self.prepare_dataset(trajectories, H)

        # 2. 状态归一化 (简单线性缩放，假设环境在 -25 到 25 之间)
        # 建议根据你的 grid_bounds 动态调整
        s_train[..., :3] /= 25.0  # 缩放 Pos
        s_train[..., 3:] /= 3.14  # 缩放 RPY

        print(f"Dataset prepared: {s_train.shape} samples")

        # 3. 开始学习
        self.learn(s_train, a_train, batch_size, epochs)


if __name__ == "__main__":
    # 配置参数
    PKL_FILE = "elite_trajectories_v1_top.pkl"
    STATE_DIM = 6      # [x, y, z, roll, pitch, yaw]
    NUM_ACTIONS = 27   # 27个宏动作
    H = 8             # Transformer 观察的历史长度 (与 EXPLORE_HORIZON 保持一致或略大)
    BATCH_SIZE = 128
    EPOCHS = 30000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 初始化模型
    # 使用 HuggingFace Transformer BC 架构
    network = HuggingFaceTransformerBCNetwork(
        state_dim=STATE_DIM,
        num_actions=NUM_ACTIONS,
        d_model=128,
        num_decoder_layers=3,
        nhead=4,
        max_seq_length=H
    )

    trainer = SGEBCTrainer(network=network, lr=1e-4, device=DEVICE)

    # 2. 训练
    if os.path.exists(PKL_FILE):
        trainer.train_from_pkl(PKL_FILE, H, BATCH_SIZE, EPOCHS)
        trainer.save_model("sge_bc_model.pth")
    else:
        print(f"Error: {PKL_FILE} not found. Please run the SGE script first.")