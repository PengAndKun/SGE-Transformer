import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import random
import sys

# ==========================================
# 1. 配置参数
# ==========================================
CONFIG = {
    "data_path": "../../../ISE_Transformer/data_pybullet/trajectories/sge_full_state_trajectories_with_ceiling_2.pkl",
    "save_model_name": "./train_model/traj_predictor.pth",
    "save_stats_name": "./train_model/traj_stats.npz",

    "input_dim": 3,  # 输入只有 [x, y, z]
    "output_dim": 3,  # 输出只有 [x, y, z]
    "context_len": 10,  # 往回看 10 个点

    "hidden_size": 128,
    "batch_size": 64,
    "lr": 1e-4,
    "epochs": 200,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# ==========================================
# 10_e. 纯位置数据集
# ==========================================
class PositionDataset(Dataset):
    def __init__(self, cfg):
        self.context_len = cfg['context_len']

        print(f"Loading {cfg['data_path']}...")
        if not os.path.exists(cfg['data_path']):
            print(f"[Error] File not found: {cfg['data_path']}")
            sys.exit(1)

        with open(cfg['data_path'], 'rb') as f:
            raw_data = pickle.load(f)

        self.samples = []
        all_positions = []

        for run_id, traj_list in raw_data.items():
            for traj in traj_list:
                # 提取状态的前3列 (x, y, z)
                # 假设 states 形状是 (T, 12) 或 (T, 6)，前3列总是位置
                states = traj['states']
                positions = states[:, 0:3]

                # 简单的滑动窗口：用前 K 个点预测第 K+1 个点
                # Input: p[i : i+K]
                # Label: p[i+K]
                num_points = len(positions)
                if num_points <= self.context_len:
                    continue

                for i in range(num_points - self.context_len):
                    window = positions[i: i + self.context_len]
                    target = positions[i + self.context_len]
                    self.samples.append((window, target))

                all_positions.append(positions)

        # 计算归一化参数
        all_pos_cat = np.concatenate(all_positions, axis=0)
        self.mean = np.mean(all_pos_cat, axis=0)
        self.std = np.std(all_pos_cat, axis=0) + 1e-6

        print(f"Loaded {len(self.samples)} samples.")
        print(f"Pos Mean: {self.mean}")
        print(f"Pos Std:  {self.std}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, target = self.samples[idx]

        # 归一化
        window = (window - self.mean) / self.std
        target = (target - self.mean) / self.std

        return {
            "input": torch.from_numpy(window).float(),
            "target": torch.from_numpy(target).float()
        }


# ==========================================
# 3. Transformer 模型
# ==========================================
class PositionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_size, context_len):
        super().__init__()
        # 位置嵌入
        self.pos_emb = nn.Linear(input_dim, hidden_size)
        # 时间嵌入 (0, 1, ..., K-1)
        self.time_emb = nn.Embedding(context_len, hidden_size)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=4 * hidden_size,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 预测头
        self.predict_head = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        # x: (B, K, 3)
        B, K, _ = x.shape

        # 生成时间步索引 [0, 1, ..., K-1]
        timesteps = torch.arange(K, device=x.device).unsqueeze(0).repeat(B, 1)

        # Embedding
        emb = self.pos_emb(x) + self.time_emb(timesteps)

        # Transformer Forward
        feat = self.transformer(emb)

        # 只取最后一个时间步的特征来预测下一步
        last_feat = feat[:, -1, :]

        return self.predict_head(last_feat)


# ==========================================
# 4. 训练主循环
# ==========================================
def train():
    device = CONFIG['device']
    print(f"Start Training on {device}...")

    # 准备数据
    dataset = PositionDataset(CONFIG)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # 保存归一化参数 (推理必用)
    np.savez(CONFIG['save_stats_name'], mean=dataset.mean, std=dataset.std)

    # 初始化模型
    model = PositionTransformer(
        input_dim=CONFIG['input_dim'],
        hidden_size=CONFIG['hidden_size'],
        context_len=CONFIG['context_len']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()

    model.train()

    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        for batch in dataloader:
            inp = batch['input'].to(device)
            tgt = batch['target'].to(device)

            pred = model(inp)
            loss = criterion(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{CONFIG['epochs']} | MSE Loss: {total_loss / len(dataloader):.6f}")

    torch.save(model.state_dict(), CONFIG['save_model_name'])
    print("Training Finished.")


if __name__ == "__main__":
    train()