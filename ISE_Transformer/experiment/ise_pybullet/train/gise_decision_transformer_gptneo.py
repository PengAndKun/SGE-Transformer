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

# 尝试导入 transformers
try:
    from transformers import GPTNeoModel, GPTNeoConfig
except ImportError:
    print("Error: transformers library not found. Please install it using 'pip install transformers'")
    sys.exit(1)

class DecisionTransformerGPTNeo(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_length: int = 20,  # 上下文窗口长度 (K)
        max_ep_len: int = 500, # 最大回合步数 (用于时间步编码)
        hidden_size: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        # GPTNeo 特有参数
        attention_types: List = [[["local", "global"], 2]], # 默认局部+全局注意力
        window_size: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        # 1. 嵌入层 (Embeddings)
        # 状态嵌入
        self.embed_state = nn.Linear(state_dim, hidden_size)
        # 动作嵌入
        self.embed_action = nn.Embedding(act_dim, hidden_size) 
        # 回报嵌入
        self.embed_return = nn.Linear(1, hidden_size)
        # 时间步嵌入
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        # 层归一化
        self.embed_ln = nn.LayerNorm(hidden_size)

        # 2. GPT-Neo Backbone
        # 注意: max_position_embeddings 需要足够大以容纳 3 * max_length
        # 因为 DT 的输入序列长度是 context_len 的 3 倍 (s, a, r)

        config = GPTNeoConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            num_layers=n_layer,
            num_heads=n_head,
            max_position_embeddings=max_length * 3 + 10, 
            attention_types= attention_types,
            use_cache=False,
            window_size=window_size
        )
        self.transformer = GPTNeoModel(config)

        # 3. 预测头
        # 我们只预测动作，输入是 (Return, State)，输出 Action Logits
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        """
        states: (batch, seq_len, state_dim)
        actions: (batch, seq_len)
        returns_to_go: (batch, seq_len, 1)
        timesteps: (batch, seq_len)
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # 嵌入
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # 将时间步编码加到所有输入上
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # 堆叠输入: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # 最终 shape: (batch, 3 * seq_len, hidden_size)
        stacked_inputs = torch.zeros((batch_size, 3 * seq_len, self.hidden_size), device=states.device)
        stacked_inputs[:, 0::3, :] = returns_embeddings
        stacked_inputs[:, 1::3, :] = state_embeddings
        stacked_inputs[:, 2::3, :] = action_embeddings
        
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Transformer Forward
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        hidden_states = transformer_outputs.last_hidden_state

        # 提取用于预测动作的特征
        # 我们希望根据 (R_t, s_t) 预测 a_t
        # 在堆叠序列中，s_t 的索引是 1, 4, 7... (即 1::3)
        action_feats = hidden_states[:, 1::3, :] 

        action_preds = self.predict_action(action_feats)  # (batch, seq_len, act_dim)

        return action_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps):
        # 这里的 get_action 通常用于推理时的单步调用
        # 需要整理成 sequence 输入
        
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # 截断到最大上下文长度
        if states.shape[1] > self.max_length:
            states = states[:, -self.max_length:, :]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:, :]
            timesteps = timesteps[:, -self.max_length:]

        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        
        # 取最后一个时间步的预测
        last_action_logits = action_preds[0, -1, :]
        probs = F.softmax(last_action_logits, dim=-1)
        action_idx = torch.argmax(probs).item()
        
        return action_idx


class DTGPTNeoTrainer:
    def __init__(self, model, lr=1e-4, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # 如果是离散动作分类，使用 CrossEntropy
        self.loss_fn = nn.CrossEntropyLoss()

    def prepare_data(self, trajectories, context_len):
        """
        将轨迹数据转换为 DT 需要的 (s, a, rtg, t) 序列片段
        假设 traj 包含: 'state_sequence', 'action_sequence', 'reward_sequence'
        """
        self.trajectories = trajectories
        self.context_len = context_len
        
        # 预计算每条轨迹的 RTG
        for traj in self.trajectories:
            if 'reward_sequence' not in traj:
                rewards = np.zeros(len(traj['action_sequence']))
                rewards[-1] = traj['final_reward']
                traj['rewards'] = rewards
            else:
                traj['rewards'] = np.array(traj['reward_sequence'])
            
            # 计算 Returns-to-Go
            rtg = np.zeros_like(traj['rewards'])
            curr_ret = 0
            for i in reversed(range(len(traj['rewards']))):
                curr_ret += traj['rewards'][i]
                rtg[i] = curr_ret
            traj['rtg'] = rtg

    def get_batch(self, batch_size, normalize_params):
        states, actions, rtgs, timesteps, masks = [], [], [], [], []
        
        for _ in range(batch_size):
            traj_idx = np.random.randint(len(self.trajectories))
            traj = self.trajectories[traj_idx]
            
            traj_len = len(traj['action_sequence'])
            si = np.random.randint(traj_len)
            
            s_start = max(0, si - self.context_len + 1)
            
            s_seg = traj['state_sequence'][s_start : si + 1]
            a_seg = traj['action_sequence'][s_start : si + 1]
            r_seg = traj['rtg'][s_start : si + 1]
            t_seg = np.arange(s_start, si + 1)
            
            # 归一化 State
            s_seg = s_seg.copy()
            s_seg[:, :3] /= normalize_params['pos_scale']
            s_seg[:, 3:] /= normalize_params['rpy_scale']
            
            # 填充 (Padding) 
            pad_len = self.context_len - len(s_seg)
            
            s_padded = np.zeros((self.context_len, 6))
            s_padded[pad_len:] = s_seg
            
            a_padded = np.zeros((self.context_len,), dtype=np.int64)
            a_padded[pad_len:] = a_seg
            
            r_padded = np.zeros((self.context_len, 1))
            r_padded[pad_len:, 0] = r_seg
            
            t_padded = np.zeros((self.context_len,), dtype=np.int64)
            t_padded[pad_len:] = t_seg
            
            mask = np.zeros((self.context_len,), dtype=np.float32)
            mask[pad_len:] = 1.0

            states.append(s_padded)
            actions.append(a_padded)
            rtgs.append(r_padded)
            timesteps.append(t_padded)
            masks.append(mask)

        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(actions), dtype=torch.long, device=self.device),
            torch.tensor(np.array(rtgs), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(timesteps), dtype=torch.long, device=self.device),
            torch.tensor(np.array(masks), dtype=torch.float32, device=self.device)
        )

    def train(self, steps, batch_size, normalize_params):
        self.model.train()
        
        # 使用 tqdm 进度条
        pbar = tqdm(range(steps), desc="DT-GPTNeo Training")
        
        for step in pbar:
            states, actions, rtgs, timesteps, masks = self.get_batch(batch_size, normalize_params)
            
            action_preds = self.model(states, actions, rtgs, timesteps)
            
            action_preds = action_preds.reshape(-1, self.model.act_dim)
            action_targets = actions.reshape(-1)
            mask = masks.reshape(-1)
            
            loss = self.loss_fn(action_preds, action_targets)
            loss = (loss * mask).mean() / mask.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if step % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if (step) % 500 == 0:
                 torch.save(self.model.state_dict(), f"sge_dt_gptneo_model_{step}.pth")


if __name__ == "__main__":
    # 路径配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    PKL_FILE_NAME = "hybrid_trajectories_batches_8_budget_50000.pkl"
    # 尝试在不同位置寻找 PKL 文件
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
        print(f"Error: {PKL_FILE_NAME} not found.")
        sys.exit(1)

    print(f"Loading trajectories from {PKL_FILE}...")
    with open(PKL_FILE, 'rb') as f:
        trajs = pickle.load(f)

    # DT 参数
    STATE_DIM = 6
    ACT_DIM = 27
    CONTEXT_LEN = 20
    HIDDEN_SIZE = 128
    
    # 实例化模型
    dt_model = DecisionTransformerGPTNeo(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        max_length=CONTEXT_LEN,
        n_layer=4, 
        n_head=4,
        hidden_size=HIDDEN_SIZE,
        attention_types=[[["local", "global"], 2]],
        max_ep_len=500
    )
    
    trainer = DTGPTNeoTrainer(dt_model, device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("Preparing data...")
    trainer.prepare_data(trajs, CONTEXT_LEN)
    
    norm_params = {'pos_scale': 25.0, 'rpy_scale': 3.14}
    
    EPOCHS = 20000
    print(f"Starting training for {EPOCHS} steps...")
    trainer.train(steps=EPOCHS, batch_size=64, normalize_params=norm_params)
    
    SAVE_PATH = f"sge_dt_gptneo_model_{EPOCHS}.pth"
    torch.save(dt_model.state_dict(), SAVE_PATH)
    print(f"Model saved to: {SAVE_PATH}")
