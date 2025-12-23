import pickle
import torch.nn.functional as F
import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm
from visualization import Visu

from environment import GridWorld
from network import append_state

from transformerbc import BCTrainer, HuggingFaceTransformerBCNetwork
from GPTNeobc import GPTNEOBCTrainer,HuggingFaceGptNeoBCNetwork
from mamba2bc import Mamba2BCTrainer,Mamba2BCNetwork

def expand_trajectory_states(trajectory_states, H):
    """
    将轨迹状态按照 append_state 的方式进行拓展

    Args:
        trajectory_states: 轨迹中的状态列表
        H: 时间范围参数

    Returns:
        expanded_states: 拓展后的状态列表
    """
    expanded_states = []

    # 模拟原始代码中的 mat_state 构建过程
    mat_state = []

    for i, state in enumerate(trajectory_states):
        # mat_state.append(state)
        mat_state.append(torch.tensor([state]))

        # 对于除了最后一个状态外的所有状态，都进行 append_state 拓展
        if i < H - 1:
            # 使用 append_state 函数进行状态拓展
            batch_state = append_state(mat_state, H - 1)
            expanded_states.append(batch_state)
        else:
            expanded_states.append(expanded_states[-1])  # 最后一个状态不需要拓展，直接重复最后一个状态

    return expanded_states

def expand_batch_states(batch_trajectory_states, H):

    expanded_batch_states = []

    # 模拟原始代码中的 mat_state 构建过程
    # mat_state = []

    for i, states in enumerate(batch_trajectory_states):

        new_arr = np.zeros(H-1, dtype=np.int64)
        for j, state in enumerate(states):
            new_arr[-(j+1)]= state

        expanded_batch_states.append(new_arr)

    # expanded_batch_states_transformed =[]
    # for seq in zip(expanded_batch_states):
    #     valid_vals = seq[seq != -1]
    #     new_arr = np.zeros_like(seq)
    #     new_arr[-len(valid_vals):] = valid_vals
    #     expanded_batch_states_transformed.append(new_arr)

    return np.array(expanded_batch_states, dtype=np.int64)


def main(epoch:int):
    H=80
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('optimal_trajectory_archive2.pkl', 'rb') as f:
        optimal_trajectory_archive = pickle.load(f)
    best_nodes = []
    max_overall_value = -1
    for node in optimal_trajectory_archive.values():
        if node.value > max_overall_value:
            max_overall_value = node.value
            best_nodes = [node]
        elif node.value == max_overall_value:
            # if node.value > 140:
            best_nodes.append(node)

    optimal_paths = [node.path_states for node in best_nodes]
    n_rows, n_cols = 11, 18

    def coord_to_index(y, x, n_rows, n_cols):
        # 左下角为0，右上角为最大
        return (n_rows - 1 - y + 1) * n_cols + x - 1

    optimal_paths_index = [
        [coord_to_index(y, x, n_rows, n_cols) for (y, x) in node.path_states]
        for node in best_nodes
    ]
    # print(optimal_paths_index[0])
    action_mapping = {0: 2, 1: 4, 2: 3, 3: 1}

    optimal_actions = [
        [action_mapping[a] for a in node.path_actions]
        for node in best_nodes
    ]
    expert_trajectories = [
        {
            'states': optimal_paths_index[i],
            'actions': optimal_actions[i]
        }
        for i in range(len(optimal_paths_index))
    ]
    expert_s = []
    expert_a = []
    for traj_i in expert_trajectories:
        expert_s.extend(np.array(expand_trajectory_states(traj_i['states'], H)[:-1]).squeeze())
        expert_a.extend(np.array(traj_i['actions']))
    expert_s = np.array(expert_s)
    expert_a = np.array(expert_a)
    print(len(expert_s), len(expert_a))
    print(f"专家轨迹状态数量: {expert_s[0]}")

    expert_s_transformed = []
    for seq, action in zip(expert_s, expert_a):
        valid_vals = seq[seq != -1]
        new_arr = -np.ones_like(seq)
        new_arr[-len(valid_vals):] = valid_vals
        expert_s_transformed.append(new_arr)

    print(f'expert_s_transformed: {len(expert_s_transformed)} {expert_s_transformed[0]}')

    batch_sequences = expert_s_transformed[:]
    state_dim=198
    expert_s_array = np.array(expert_s_transformed, dtype=np.int64)
    expert_s_one_hot = np.zeros((expert_s_array.shape[0], expert_s_array.shape[1], state_dim), dtype=np.float32)
    for i in range(expert_s_array.shape[0]):
        for j in range(expert_s_array.shape[1]):
            if expert_s_array[i][j] != -1:
                expert_s_one_hot[i][j][expert_s_array[i][j]] = 1.0


    # rows = np.arange(expert_s_array.shape[0])[:, None]
    # cols = np.arange(expert_s_array.shape[1])
    # expert_s_one_hot[rows, cols, expert_s_array] = 1.0

    network = HuggingFaceTransformerBCNetwork(state_dim=state_dim, num_actions=5, d_model=128, num_decoder_layers=3, nhead=8,
                                              max_seq_length=80)
    trainer = BCTrainer(network, lr=1e-4, device=device)

    trainer.learn(expert_s_one_hot.tolist(), expert_a, 64, num_epochs=epoch)
    trainer.save_model(f'transbc_2_room_198_model/transbc_model_{str(epoch)}.pth')

def train_gpt_neo_bc(epoch:int):
    H=80
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('optimal_trajectory_archive2.pkl', 'rb') as f:
        optimal_trajectory_archive = pickle.load(f)
    best_nodes = []
    max_overall_value = -1
    for node in optimal_trajectory_archive.values():
        if node.value > max_overall_value:
            max_overall_value = node.value
            best_nodes = [node]
        elif node.value == max_overall_value:
            # if node.value > 140:
            best_nodes.append(node)

    optimal_paths = [node.path_states for node in best_nodes]
    n_rows, n_cols = 11, 18

    def coord_to_index(y, x, n_rows, n_cols):
        # 左下角为0，右上角为最大
        return (n_rows - 1 - y + 1) * n_cols + x - 1

    optimal_paths_index = [
        [coord_to_index(y, x, n_rows, n_cols) for (y, x) in node.path_states]
        for node in best_nodes
    ]
    # print(optimal_paths_index[0])
    action_mapping = {0: 2, 1: 4, 2: 3, 3: 1}

    optimal_actions = [
        [action_mapping[a] for a in node.path_actions]
        for node in best_nodes
    ]
    expert_trajectories = [
        {
            'states': optimal_paths_index[i],
            'actions': optimal_actions[i]
        }
        for i in range(len(optimal_paths_index))
    ]
    expert_s = []
    expert_a = []
    for traj_i in expert_trajectories:
        expert_s.extend(np.array(expand_trajectory_states(traj_i['states'], H)[:-1]).squeeze())
        expert_a.extend(np.array(traj_i['actions']))
    expert_s = np.array(expert_s)
    expert_a = np.array(expert_a)
    print(len(expert_s), len(expert_a))
    print(f"专家轨迹状态数量: {expert_s[0]}")

    expert_s_transformed = []
    for seq, action in zip(expert_s, expert_a):
        valid_vals = seq[seq != -1]
        new_arr = -np.ones_like(seq)
        new_arr[-len(valid_vals):] = valid_vals
        expert_s_transformed.append(new_arr)

    print(f'expert_s_transformed: {len(expert_s_transformed)} {expert_s_transformed[0]}')

    batch_sequences = expert_s_transformed[:]
    state_dim=198
    expert_s_array = np.array(expert_s_transformed, dtype=np.int64)
    expert_s_one_hot = np.zeros((expert_s_array.shape[0], expert_s_array.shape[1], state_dim), dtype=np.float32)
    for i in range(expert_s_array.shape[0]):
        for j in range(expert_s_array.shape[1]):
            if expert_s_array[i][j] != -1:
                expert_s_one_hot[i][j][expert_s_array[i][j]] = 1.0


    # rows = np.arange(expert_s_array.shape[0])[:, None]
    # cols = np.arange(expert_s_array.shape[1])
    # expert_s_one_hot[rows, cols, expert_s_array] = 1.0

    network = HuggingFaceGptNeoBCNetwork(state_dim=state_dim, num_actions=5, d_model=128, num_decoder_layers=3, nhead=8,
                                              max_seq_length=80)
    trainer = GPTNEOBCTrainer(network, lr=1e-4, device=device)

    trainer.learn(expert_s_one_hot.tolist(), expert_a, 64, num_epochs=epoch)
    trainer.save_model(f'gptneobc_2_room_198_model/transbc_model_{str(epoch)}.pth')


def all_train_gpt2_3_mamba2(epoch:int):
    H=80
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('optimal_trajectory_archive2.pkl', 'rb') as f:
        optimal_trajectory_archive = pickle.load(f)
    best_nodes = []
    max_overall_value = -1
    for node in optimal_trajectory_archive.values():
        if node.value > max_overall_value:
            max_overall_value = node.value
            best_nodes = [node]
        elif node.value == max_overall_value:
            # if node.value > 140:
            best_nodes.append(node)

    optimal_paths = [node.path_states for node in best_nodes]
    n_rows, n_cols = 11, 18

    def coord_to_index(y, x, n_rows, n_cols):
        # 左下角为0，右上角为最大
        return (n_rows - 1 - y + 1) * n_cols + x - 1

    optimal_paths_index = [
        [coord_to_index(y, x, n_rows, n_cols) for (y, x) in node.path_states]
        for node in best_nodes
    ]
    # print(optimal_paths_index[0])
    action_mapping = {0: 2, 1: 4, 2: 3, 3: 1}

    optimal_actions = [
        [action_mapping[a] for a in node.path_actions]
        for node in best_nodes
    ]
    expert_trajectories = [
        {
            'states': optimal_paths_index[i],
            'actions': optimal_actions[i]
        }
        for i in range(len(optimal_paths_index))
    ]
    expert_s = []
    expert_a = []
    for traj_i in expert_trajectories:
        expert_s.extend(np.array(expand_trajectory_states(traj_i['states'], H)[:-1]).squeeze())
        expert_a.extend(np.array(traj_i['actions']))
    expert_s = np.array(expert_s)
    expert_a = np.array(expert_a)
    print(len(expert_s), len(expert_a))
    print(f"专家轨迹状态数量: {expert_s[0]}")

    expert_s_transformed = []
    for seq, action in zip(expert_s, expert_a):
        valid_vals = seq[seq != -1]
        new_arr = -np.ones_like(seq)
        new_arr[-len(valid_vals):] = valid_vals
        expert_s_transformed.append(new_arr)

    print(f'expert_s_transformed: {len(expert_s_transformed)} {expert_s_transformed[0]}')

    batch_sequences = expert_s_transformed[:]
    state_dim=198
    expert_s_array = np.array(expert_s_transformed, dtype=np.int64)
    expert_s_one_hot = np.zeros((expert_s_array.shape[0], expert_s_array.shape[1], state_dim), dtype=np.float32)
    for i in range(expert_s_array.shape[0]):
        for j in range(expert_s_array.shape[1]):
            if expert_s_array[i][j] != -1:
                expert_s_one_hot[i][j][expert_s_array[i][j]] = 1.0


    # rows = np.arange(expert_s_array.shape[0])[:, None]
    # cols = np.arange(expert_s_array.shape[1])
    # expert_s_one_hot[rows, cols, expert_s_array] = 1.0

    # network = HuggingFaceTransformerBCNetwork(state_dim=state_dim, num_actions=5, d_model=128, num_decoder_layers=3, nhead=8,
    #                                           max_seq_length=80)
    # trainer = BCTrainer(network, lr=1e-4, device=device)
    #
    # trainer.learn(expert_s_one_hot.tolist(), expert_a, 64, num_epochs=epoch)
    # trainer.save_model(f'transbc_2_room_198_model/transbc_model_{str(epoch)}.pth')

    network = Mamba2BCNetwork(state_dim=state_dim, num_actions=5, d_model=128, num_decoder_layers=3,
                                              max_seq_length=80)
    trainer = Mamba2BCTrainer(network, lr=1e-4, device=device)

    trainer.learn(expert_s_one_hot.tolist(), expert_a, 64, num_epochs=epoch)
    trainer.save_model(f'mamba2bc_2_room_198_model/mamba2bc_model_{str(epoch)}.pth')



if __name__ == "__main__":
    train_epoch = 50000
    all_train_gpt2_3_mamba2(train_epoch)
    # main(train_epoch)
    train_gpt_neo_bc(train_epoch)
    # print("start")


