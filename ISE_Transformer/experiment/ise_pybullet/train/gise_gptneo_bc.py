import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from typing import List, Tuple
import time
from tqdm import tqdm
import os
import sys

# 尝试导入 transformers，如果没有安装用户应当自行安装
try:
    from transformers import GPTNeoModel, GPTNeoConfig
except ImportError:
    print("Error: transformers library not found. Please install it using 'pip install transformers'")
    sys.exit(1)

class HuggingFaceGPTNeoBCNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int = 27, d_model: int = 128, num_layers: int = 4, nhead: int = 4, max_seq_length: int = 20):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.max_seq_length = max_seq_length
        
        # 状态嵌入层：将状态维度映射到 Transformer 的隐藏层维度
        self.input_embedding = nn.Linear(state_dim, d_model)
        
        # GPT-Neo 配置
        # 对于小规模模型，我们使用简单的 Global Attention
        config = GPTNeoConfig(
            vocab_size=1,  # 我们不使用 token embedding，但参数是必须的
            hidden_size=d_model,
            num_layers=num_layers,
            num_heads=nhead,
            max_position_embeddings=max_seq_length,
            attention_types=[[["local", "global"], num_layers//2]], 
            use_cache=False,
            window_size=max_seq_length # 防止 local attention 报错
        )
        
        self.transformer = GPTNeoModel(config)
        
        # 动作预测头
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(d_model // 2, num_actions)
        )

    def forward(self, state_sequences):
        """
        state_sequences: (Batch, Seq_Len, State_Dim)
        """
        batch_size, seq_len, _ = state_sequences.shape
        device = state_sequences.device
        
        # 1. 计算 Embeddings
        # 映射输入状态
        inputs_embeds = self.input_embedding(state_sequences)
        
        # 2. Transformer Forward
        # GPTNeoModel 支持直接传入 inputs_embeds，它会自动加上 Position Embeddings
        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds)
        hidden_states = transformer_outputs.last_hidden_state
        
        # 3. 取最后一个时间步的输出
        last_step_output = hidden_states[:, -1, :]
        
        # 4. 预测动作 Logits
        logits = self.output_head(last_step_output)
        
        return logits

    def get_action(self, state_sequences, deterministic=True):
        logits = self.forward(state_sequences)
        probs = F.softmax(logits, dim=-1)
        if deterministic: 
            action_idx = torch.argmax(probs, dim=-1)
        else: 
            action_idx = torch.multinomial(probs, 1).squeeze(-1)
        return action_idx, probs


class SGEGPTNeoBCTrainer:
    def __init__(self,
                 network: HuggingFaceGPTNeoBCNetwork,
                 lr: float = 1e-4,
                 device: str = None):
        self.device = device if device is not None else torch.device('cpu')
        self.network = network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

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
            for i in range(len(actions)):
                # 提取历史状态 [i-H+1, ..., i]
                start_idx = max(0, i - H + 1)
                sub_seq = states[start_idx: i + 1]

                # padding
                if len(sub_seq) < H:
                    pad = np.tile(sub_seq[0], (H - len(sub_seq), 1))
                    sub_seq = np.vstack([pad, sub_seq])

                all_states.append(sub_seq)
                all_actions.append(actions[i])

        return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.int64)

    def learn(self, state_sequences, action_sequences, batch_size, num_epochs: int):
        if len(state_sequences) == 0:
            return

        all_state_sequences = torch.FloatTensor(state_sequences).to(self.device)
        all_action_sequences = torch.LongTensor(action_sequences).to(self.device)

        start_time = time.time()
        
        self.network.train()
        
        # 使用 tqdm 进度条
        epoch_pbar = tqdm(range(num_epochs), desc="GPTNeo BC Training", unit="epoch")
        
        for epoch in epoch_pbar:
            # Random sampling
            indices = torch.randperm(len(all_state_sequences))[:batch_size]
            
            batch_sequences = all_state_sequences[indices]
            batch_targets = all_action_sequences[indices]

            self.optimizer.zero_grad()
            logits = self.network(batch_sequences)
            loss = F.cross_entropy(logits, batch_targets)
            loss.backward()
            self.optimizer.step()
            
            # 更新进度条显示的 loss
            if (epoch + 1) % 10 == 0:
                epoch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if (epoch) % 500 == 0:
                self.save_model(f"sge_gptneo_bc_model_{epoch}.pth")

        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")

    def train_from_pkl(self, pkl_path: str, H: int, batch_size: int, epochs: int):
        if not os.path.exists(pkl_path):
            print(f"Error: {pkl_path} not found.")
            return

        with open(pkl_path, 'rb') as f:
            trajectories = pickle.load(f)

        print(f"Loading {len(trajectories)} trajectories for BC training...")

        # 1. 准备数据
        s_train, a_train = self.prepare_dataset(trajectories, H)

        # 2. 状态归一化 
        s_train[..., :3] /= 25.0  # Pos scale
        s_train[..., 3:] /= 3.14  # RPY scale

        print(f"Dataset prepared: {s_train.shape} samples")

        # 3. 开始学习
        self.learn(s_train, a_train, batch_size, epochs)
        
    def save_model(self, filename: str):
        torch.save({'network_state_dict': self.network.state_dict()}, filename)
        print(f"✓ Model saved: {filename}")


if __name__ == "__main__":
    # 当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 这里我们查找可能的 PKL 文件路径
    POSSIBLE_PKL_NAME = "hybrid_trajectories_batches_8_budget_50000.pkl"
    PKL_FILE = os.path.join(current_dir, POSSIBLE_PKL_NAME)
    
    # 如果找不到，试着找当前工作目录
    if not os.path.exists(PKL_FILE):
        PKL_FILE = POSSIBLE_PKL_NAME

    # 配置参数 - 与原 TransformerBC 保持一致，方便对比
    STATE_DIM = 6
    NUM_ACTIONS = 27
    H = 20             
    BATCH_SIZE = 128
    EPOCHS = 20000
    
    # 模型规模参数
    D_MODEL = 128
    NUM_LAYERS = 4
    N_HEAD = 4
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing GPT-Neo BC Network...")
    network = HuggingFaceGPTNeoBCNetwork(
        state_dim=STATE_DIM,
        num_actions=NUM_ACTIONS,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        nhead=N_HEAD,
        max_seq_length=H
    )

    trainer = SGEGPTNeoBCTrainer(network=network, lr=1e-4, device=DEVICE)

    if os.path.exists(PKL_FILE):
        trainer.train_from_pkl(PKL_FILE, H, BATCH_SIZE, EPOCHS)
        trainer.save_model(f"sge_gptneo_bc_model_{EPOCHS}.pth")
    else:
        print(f"Error: {PKL_FILE} not found. Please make sure the trajectory file exists.")
