import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import sys

# ==========================================
# 1. 配置
# ==========================================
CONFIG = {
    "data_path": "../../../ISE_Transformer/data_pybullet/trajectories/sge_full_state_trajectories_with_ceiling_2.pkl",
    "save_model_name": "./train_model/mask/100/traj_predictor.pth",
    "save_stats_name": "./train_model/mask/100/traj_stats.npz",

    "input_dim": 3,  # [x, y, z]
    "output_dim": 3,

    # 这里的 context_len 就是最大轨迹长度
    # 所有的轨迹都会被 pad 到这个长度，或者被截断
    "max_len": 1000,

    "hidden_size": 128,
    "n_layer": 3,
    "n_head": 4,
    "batch_size": 32,  # 因为输入变长了，Batch Size 可以适当调小
    "lr": 1e-4,
    "epochs": 500,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# ==========================================
# 2. 全序列数据集 (Full Trajectory)
# ==========================================
class FullTrajectoryDataset(Dataset):
    def __init__(self, cfg):
        self.max_len = cfg['max_len']
        print(f"Loading {cfg['data_path']}...")

        with open(cfg['data_path'], 'rb') as f:
            raw_data = pickle.load(f)

        self.trajectories = []
        all_positions = []

        for run_id, traj_list in raw_data.items():
            for traj in traj_list:
                # 提取 (T, 3) 位置
                positions = traj['states'][:, 0:3]
                all_positions.append(positions)

                length = len(positions)

                # 存入列表
                self.trajectories.append({
                    "pos": positions,
                    "length": length
                })

        # 计算全局归一化参数
        all_pos_cat = np.concatenate(all_positions, axis=0)
        self.mean = np.mean(all_pos_cat, axis=0)
        self.std = np.std(all_pos_cat, axis=0) + 1e-6

        print(f"Loaded {len(self.trajectories)} full trajectories.")
        print(f"Pos Mean: {self.mean}, Std: {self.std}")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        pos = traj['pos']  # (L, 3)
        length = traj['length']

        # 1. 归一化
        pos = (pos - self.mean) / self.std

        # 2. 截断与 Padding
        # 我们需要输入和目标错开一位
        # Input:  P_0, ..., P_{T-1}
        # Target: P_1, ..., P_T

        # 实际有效长度 (最多取 max_len)
        valid_len = min(length, self.max_len)

        # 构造容器
        padded_pos = np.zeros((self.max_len, 3), dtype=np.float32)
        padded_pos[:valid_len] = pos[:valid_len]

        # Mask: 1 表示有效数据，0 表示 Padding
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:valid_len] = 1.0

        return {
            "pos": torch.from_numpy(padded_pos),  # (MaxLen, 3)
            "mask": torch.from_numpy(mask),  # (MaxLen, )
            "length": valid_len
        }


# ==========================================
# 3. GPT 风格 Transformer (Causal)
# ==========================================
class TrajectoryGPT(nn.Module):
    def __init__(self, input_dim, hidden_size, max_len):
        super().__init__()
        self.hidden_size = hidden_size

        # Embedding
        self.pos_emb = nn.Linear(input_dim, hidden_size)
        self.time_emb = nn.Embedding(max_len, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.embed_dropout = nn.Dropout(0.1)

        # GPT Encoder (Decoder-only architecture in PyTorch is confusingly called TransformerEncoder if we manually mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=4 * hidden_size,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Prediction Head
        self.predict_head = nn.Linear(hidden_size, input_dim)

    def forward(self, x, padding_mask=None):
        # x: (B, L, 3)
        B, L, _ = x.shape

        # 1. Embeddings
        timesteps = torch.arange(L, device=x.device).unsqueeze(0).repeat(B, 1)
        emb = self.pos_emb(x) + self.time_emb(timesteps)
        emb = self.embed_ln(emb)
        emb = self.embed_dropout(emb)

        # 2. Causal Mask (核心！)
        # 生成一个上三角矩阵，值为 -inf，对角线及以下为 0
        # 这样位置 t 只能看到 0...t
        causal_mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1).to(x.device)

        # 3. Transformer Forward
        # src_key_padding_mask: (B, L), True 表示被 Mask (Padding部分)
        # 注意: padding_mask 传入时 1是有效, 0是padding。PyTorch需要 True是padding
        key_padding_mask = (padding_mask == 0) if padding_mask is not None else None

        out = self.transformer(emb, mask=causal_mask, src_key_padding_mask=key_padding_mask)

        # 4. Predict
        return self.predict_head(out)


# ==========================================
# 4. 训练循环 (Sequence-to-Sequence)
# ==========================================
def train():
    device = CONFIG['device']
    print(f"=== Training GPT Trajectory Model on {device} ===")

    dataset = FullTrajectoryDataset(CONFIG)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # 保存归一化参数
    np.savez(CONFIG['save_stats_name'], mean=dataset.mean, std=dataset.std)

    model = TrajectoryGPT(
        input_dim=CONFIG['input_dim'],
        hidden_size=CONFIG['hidden_size'],
        max_len=CONFIG['max_len']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss(reduction='none')  # 使用 None 以便手动 Mask

    model.train()

    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        for batch in dataloader:
            # 数据: (B, MaxLen, 3)
            full_seq = batch['pos'].to(device)
            mask = batch['mask'].to(device)

            # --- 构建 Input 和 Target ---
            # 任务：根据 P_0...P_t 预测 P_{t+1}
            # Input:  序列的前面部分 [0 : L-1]
            # Target: 序列的后面部分 [1 : L]

            inp = full_seq[:, :-1, :]  # (B, L-1, 3)
            tgt = full_seq[:, 1:, :]  # (B, L-1, 3)

            # 对应的 Mask 也得切掉一位
            # 因为如果是 Padding，预测它也没意义
            curr_mask = mask[:, :-1]  # (B, L-1)

            # Forward
            # 注意：传入 padding_mask 告诉 Attention 忽略补零的部分
            preds = model(inp, padding_mask=curr_mask)

            # Calculate Loss
            loss = criterion(preds, tgt)  # (B, L-1, 3)

            # 只计算有效数据的 Loss
            # loss shape: (B, L-1, 3) -> mean over last dim -> (B, L-1)
            loss = loss.mean(dim=-1)
            loss = (loss * curr_mask).sum() / curr_mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{CONFIG['epochs']} | Loss: {total_loss / len(dataloader):.6f}")

    torch.save(model.state_dict(), CONFIG['save_model_name'])
    print("Training Finished.")


if __name__ == "__main__":
    train()