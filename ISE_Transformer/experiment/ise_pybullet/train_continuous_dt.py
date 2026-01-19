import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import random

# ==========================================
# 配置参数
# ==========================================
CONFIG = {
    # 请确保这里指向你正确的数据文件
    "data_path": "../../../ISE_Transformer/data_pybullet/trajectories/sge_full_state_trajectories_with_ceiling.pkl",
    "save_model_name": "dt_pos_only_policy.pth",

    # 物理参数
    "ctrl_freq": 30.0,  # 控制频率
    "frames_per_step": 5,  # 每5帧取样

    # 维度定义 (保持 12 不变，我们在 Dataset 里把数据凑齐 12 维)
    "state_dim": 12,  # [pos(3), rpy(3), vel(3), ang_vel(3)]
    "act_dim": 3,  # [vx, vy, vz]

    # 模型超参数
    "hidden_size": 128,
    "max_length": 20,
    "max_ep_len": 1000,
    "batch_size": 64,
    "lr": 1e-4,
    "epochs": 10000,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# ==========================================
# 1. 自动填充维度的 Dataset (核心修复)
# ==========================================
class PosOnlyDroneDataset(Dataset):
    def __init__(self, data_path, context_len, dt_seconds):
        self.context_len = context_len
        self.dt = dt_seconds

        print(f"Loading data from {data_path}...")
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)

        self.trajectories = []
        all_states = []
        all_actions = []

        for run_id, traj_list in raw_data.items():
            for traj in traj_list:
                # 原始状态: 可能是 (T, 6) 也可能是 (T, 12)
                raw_states = traj['states']

                # 1. 提取位置 pos (前3列)
                positions = raw_states[:, 0:3]

                # 10_e. 计算速度 (动作标签)
                # v[t] = (p[t+1] - p[t]) / dt
                next_pos = positions[1:]
                curr_pos = positions[:-1]
                calculated_vel = (next_pos - curr_pos) / self.dt  # (T-1, 3)

                # 动作就是速度
                actions = calculated_vel

                # 3. 构建 12维 State (输入)
                # 截取前 T-1 个状态
                current_states_base = raw_states[:-1]  # (T-1, D)

                # 如果原始数据只有 6维 (pos, rpy)，我们需要拼凑出 12维
                if current_states_base.shape[1] == 6:
                    # 构造: [Pos(3), RPY(3), Vel(3), AngVel(3)]
                    # Vel 用我们刚算的 calculated_vel
                    # AngVel 用 0 填充
                    ang_vel_zeros = np.zeros_like(calculated_vel)

                    states_12d = np.concatenate([
                        current_states_base,  # Pos + RPY
                        calculated_vel,  # Calculated Vel (填补进去!)
                        ang_vel_zeros  # Ang Vel (0)
                    ], axis=1)  # (T-1, 12)

                elif current_states_base.shape[1] >= 12:
                    # 如果本来就是 12维或更多，直接切片
                    states_12d = current_states_base[:, :12]
                else:
                    raise ValueError(f"Unexpected state dimension: {current_states_base.shape[1]}")

                # 4. RTG 计算
                final_reward = traj['final_reward']
                length = len(states_12d)
                timesteps = np.arange(length)
                rtg = final_reward * (1 - timesteps / length)
                rtg = rtg[..., None]

                all_states.append(states_12d)
                all_actions.append(actions)

                self.trajectories.append({
                    "states": states_12d,
                    "actions": actions,
                    "rtg": rtg,
                    "timesteps": timesteps,
                    "length": length
                })

        # 归一化统计
        all_states_concat = np.concatenate(all_states, axis=0)
        all_actions_concat = np.concatenate(all_actions, axis=0)

        self.state_mean = np.mean(all_states_concat, axis=0)
        self.state_std = np.std(all_states_concat, axis=0) + 1e-6
        self.act_mean = np.mean(all_actions_concat, axis=0)
        self.act_std = np.std(all_actions_concat, axis=0) + 1e-6

        print(f"Loaded {len(self.trajectories)} trajectories.")
        print(f"State Dim: {all_states_concat.shape[1]} (Should be 12)")

    def get_stats(self):
        return self.state_mean, self.state_std, self.act_mean, self.act_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        l = traj['length']
        si = random.randint(0, l - 1)

        start_idx = max(0, si - self.context_len + 1)
        end_idx = si + 1

        s = traj['states'][start_idx:end_idx]
        a = traj['actions'][start_idx:end_idx]
        r = traj['rtg'][start_idx:end_idx]
        t = traj['timesteps'][start_idx:end_idx]

        # 归一化
        s = (s - self.state_mean) / self.state_std
        a = (a - self.act_mean) / self.act_std

        # Padding
        pad_len = self.context_len - len(s)

        s = np.pad(s, ((pad_len, 0), (0, 0)), 'constant')
        a = np.pad(a, ((pad_len, 0), (0, 0)), 'constant')
        r = np.pad(r, ((pad_len, 0), (0, 0)), 'constant')
        t = np.pad(t, (pad_len, 0), 'constant')
        mask = np.zeros(self.context_len)
        mask[pad_len:] = 1

        return {
            "states": torch.from_numpy(s).float(),
            "actions": torch.from_numpy(a).float(),
            "rtg": torch.from_numpy(r).float(),
            "timesteps": torch.from_numpy(t).long(),
            "mask": torch.from_numpy(mask).float()
        }


# ==========================================
# 10_e. 模型定义 (连续版 DT)
# ==========================================
class ContinuousDecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size, max_length, max_ep_len, **kwargs):
        super().__init__()
        self.state_emb = nn.Linear(state_dim, hidden_size)
        self.act_emb = nn.Linear(act_dim, hidden_size)
        self.ret_emb = nn.Linear(1, hidden_size)
        self.pos_emb = nn.Embedding(max_ep_len, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.embed_dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=4 * hidden_size,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.predict_action = nn.Linear(hidden_size, act_dim)

    def forward(self, states, actions, rtg, timesteps, mask=None):
        B, K, _ = states.shape
        # 这里之前报错是因为 states 是 6维，但 Linear 期望 12维
        # 现在通过 Dataset 修复，这里 states 已经是 12维了
        x = self.state_emb(states) + self.ret_emb(rtg) + self.pos_emb(timesteps)

        x = self.embed_ln(x)
        x = self.embed_dropout(x)

        causal_mask = torch.triu(torch.ones(K, K), diagonal=1).bool().to(x.device)
        key_padding_mask = (mask == 0)

        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.predict_action(out)


# ==========================================
# 3. 训练函数
# ==========================================
def train():
    dt = CONFIG['frames_per_step'] / CONFIG['ctrl_freq']
    print(f"Calculated dt per step: {dt:.4f} seconds")

    dataset = PosOnlyDroneDataset(CONFIG['data_path'], CONFIG['max_length'], dt_seconds=dt)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # 保存归一化参数
    s_mean, s_std, a_mean, a_std = dataset.get_stats()
    np.savez("dt_pos_stats.npz",
             state_mean=s_mean, state_std=s_std,
             act_mean=a_mean, act_std=a_std,
             dt=dt)

    device = CONFIG['device']
    model = ContinuousDecisionTransformer(
        state_dim=CONFIG['state_dim'],
        act_dim=CONFIG['act_dim'],
        hidden_size=CONFIG['hidden_size'],
        max_length=CONFIG['max_length'],
        max_ep_len=CONFIG['max_ep_len']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss(reduction='none')

    model.train()
    print(f"Start Training on {device}...")

    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        for batch in dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtg = batch['rtg'].to(device)
            timesteps = batch['timesteps'].to(device)
            mask = batch['mask'].to(device)

            preds = model(states, actions, rtg, timesteps, mask)
            loss = criterion(preds, actions)

            mask_exp = mask.unsqueeze(-1).expand_as(loss)
            loss = (loss * mask_exp).sum() / mask_exp.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{CONFIG['epochs']} | Loss: {total_loss / len(dataloader):.6f}")

    torch.save(model.state_dict(), CONFIG['save_model_name'])
    print("Training Finished.")


if __name__ == "__main__":
    train()