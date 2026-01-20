import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
import random
from typing import List, Tuple
from tqdm import tqdm
from transformers import GPT2Model, GPT2Config

class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_length: int = 50,  # 上下文窗口长度 (K)
        max_ep_len: int = 500, # 最大回合步数 (用于时间步编码)
        hidden_size: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        n_inner: int = 4 * 128,
        activation_function: str = 'relu',
        n_positions: int = 1024,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        # 1. 嵌入层 (Embeddings)
        # 状态嵌入
        self.embed_state = nn.Linear(state_dim, hidden_size)
        # 动作嵌入 (输入是动作的索引，如果输入是 one-hot 或连续值则需改为 Linear)
        self.embed_action = nn.Embedding(act_dim, hidden_size) 
        # 回报嵌入 (Returns-to-Go 是一个标量)
        self.embed_return = nn.Linear(1, hidden_size)
        # 时间步嵌入
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        # 层归一化
        self.embed_ln = nn.LayerNorm(hidden_size)

        # 2. GPT2 Backbone
        config = GPT2Config(
            vocab_size=1,  # 不使用词表
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            n_positions=n_positions,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        # 3. 预测头
        # 我们只预测动作，输入是 (Return, State)，输出 Action Logits
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 可选的中间层
            nn.ReLU(),                           
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        """
        states: (batch, seq_len, state_dim)
        actions: (batch, seq_len) - 这里的 actions 是历史动作，注意 DT 在预测 t 时，输入的是 t-1 的动作
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
        # 用 s_t 对应的输出来预测动作是标准做法
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


class DTTrainer:
    def __init__(self, model, lr=1e-4, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # 如果是离散动作分类，使用 CrossEntropy
        self.loss_fn = nn.CrossEntropyLoss()

    def prepare_data(self, trajectories, context_len):
        """
        将轨迹数据转换为 DT 需要的 (s, a, rtg, t) 序列片段
        假设 traj 包含: 'state_sequence', 'action_sequence', 'reward_sequence' (如果没有 reward, 需要根据 final_reward 推算)
        """
        self.trajectories = trajectories
        self.context_len = context_len
        
        # 预计算每条轨迹的 RTG
        for traj in self.trajectories:
            # 假设 traj['reward_sequence'] 存在。如果不存在，需要在这里进行计算
            # 兼容性处理：如果没有 reward_sequence，但有 final_reward
            if 'reward_sequence' not in traj:
                rewards = np.zeros(len(traj['action_sequence']))
                # 假设稀疏奖励：只有最后一步有分 (这在 DT 中效果一般，最好有稠密奖励)
                # 或者假设均匀分布
                rewards[-1] = traj['final_reward']
                traj['rewards'] = rewards
            else:
                traj['rewards'] = np.array(traj['reward_sequence'])
            
            # 计算 Returns-to-Go
            # RTG[t] = sum(reward[t:])
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
            # 随机选择一个结束点 si
            si = np.random.randint(traj_len)
            
            # 获取上下文长度的数据
            s_start = max(0, si - self.context_len + 1)
            
            # 提取片段
            # state 需要注意: len(s)通常比 len(a) 多 1 (包含初始状态)，我们要取对应 action 之前的 state
            # s[0] -> a[0], s[1] -> a[1] ...
            s_seg = traj['state_sequence'][s_start : si + 1]
            a_seg = traj['action_sequence'][s_start : si + 1]
            r_seg = traj['rtg'][s_start : si + 1]
            t_seg = np.arange(s_start, si + 1)
            
            # 归一化 State (非常重要)
            s_seg = s_seg.copy() # 避免修改原数据
            s_seg[:, :3] /= normalize_params['pos_scale']
            s_seg[:, 3:] /= normalize_params['rpy_scale']
            
            # 填充 (Padding) 到 context_len
            # 在左侧填充 0
            pad_len = self.context_len - len(s_seg)
            
            # States (Pad with zeros)
            s_padded = np.zeros((self.context_len, 6))
            s_padded[pad_len:] = s_seg
            
            # Actions (Pad with zeros or specific token, DT usually expects act dim embedding)
            # 这里 embedding 输入是 int，我们可以用 0 做 padding, 但要注意 mask
            a_padded = np.zeros((self.context_len,), dtype=np.int64)
            a_padded[pad_len:] = a_seg
            
            # RTG
            r_padded = np.zeros((self.context_len, 1))
            r_padded[pad_len:, 0] = r_seg
            
            # Timesteps
            t_padded = np.zeros((self.context_len,), dtype=np.int64)
            t_padded[pad_len:] = t_seg
            
            # Mask (1 for real data, 0 for padding)
            mask = np.zeros((self.context_len,), dtype=np.float32)
            mask[pad_len:] = 1.0

            states.append(s_padded)
            actions.append(a_padded)
            rtgs.append(r_padded)
            timesteps.append(t_padded)
            masks.append(mask)

        # Convert to Tensor
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(actions), dtype=torch.long, device=self.device),
            torch.tensor(np.array(rtgs), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(timesteps), dtype=torch.long, device=self.device),
            torch.tensor(np.array(masks), dtype=torch.float32, device=self.device)
        )

    def train(self, steps, batch_size, normalize_params):
        self.model.train()
        pbar = tqdm(range(steps), desc="DT Training")
        
        for step in pbar:
            states, actions, rtgs, timesteps, masks = self.get_batch(batch_size, normalize_params)
            
            # Forward
            # actions 输入: 为了预测 t 时刻动作，DT 的 action 输入通常需要向右 shift，
            # 即第 1 个 action token 通常是 dummy 或者 zero，第 k 个 token 是 a_{k-1}
            # 但如果你看我的 forward 实现: stacked_inputs[:, 2::3, :] = action_embeddings
            # 如果我们希望 s_t 预测 a_t，那么输入里是否已经包含了 a_t?
            # 官方实现中，训练时输入完整的 sequence，但在计算 loss 时，只计算预测正确的 a_t
            
            action_preds = self.model(states, actions, rtgs, timesteps)
            
            # Calculate Loss
            # action_preds: (B, K, Act_Dim)
            # actions: (B, K)
            
            # Flatten for CrossEntropy
            action_preds = action_preds.reshape(-1, self.model.act_dim)
            action_targets = actions.reshape(-1)
            mask = masks.reshape(-1)
            
            loss = self.loss_fn(action_preds, action_targets)
            loss = (loss * mask).mean() / mask.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (step + 1) % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if (step) % 500 == 0:
                 torch.save(self.model.state_dict(), f"sge_dt_model_{step}.pth")
                 # tqdm.write avoid interrupting the progress bar
                 # tqdm.write(f"Model saved: sge_dt_model_{step+1}.pth")


if __name__ == "__main__":
    # 配置
    PKL_FILE = "hybrid_trajectories_batches_8_budget_50000.pkl"
    # 尝试加载
    if not os.path.exists(PKL_FILE):
        print(f"Error: {PKL_FILE} not found.")
        exit()

    with open(PKL_FILE, 'rb') as f:
        trajs = pickle.load(f)

    # 简单检查数据里有没有 reward_sequence
    if 'reward_sequence' not in trajs[0]:
        print("警告: 轨迹数据中缺少 'reward_sequence'。将使用 final_reward 对最后一步进行赋值(稀疏奖励)。")
        # 如果你的验证代码能跑，你可以修改这里去重新计算 per-step reward

    # DT 参数
    STATE_DIM = 6
    ACT_DIM = 27
    CONTEXT_LEN = 20
    HIDDEN_SIZE = 128
    
    dt_model = DecisionTransformer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        max_length=CONTEXT_LEN,
        n_layer=4, 
        n_head=4,
        hidden_size=HIDDEN_SIZE
    )
    
    trainer = DTTrainer(dt_model, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备数据
    trainer.prepare_data(trajs, CONTEXT_LEN)
    
    # 归一化参数 dict
    norm_params = {'pos_scale': 25.0, 'rpy_scale': 3.14}
    
    # 开始训练
    EPOCHS = 20000
    trainer.train(steps=EPOCHS, batch_size=64, normalize_params=norm_params)
    
    # 保存
    torch.save(dt_model.state_dict(), f"sge_dt_model_{EPOCHS}.pth")
    print(f"模型已保存: sge_dt_model_{EPOCHS}.pth")