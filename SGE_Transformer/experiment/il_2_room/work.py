import torch
import numpy as np
print(torch.__version__)
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

a = [[0,1,2],[3,4,5]]
expert_s_array = np.array(a, dtype=np.int64)
expert_s_one_hot = np.zeros((expert_s_array.shape[0], expert_s_array.shape[1], 6), dtype=np.float32)
print(expert_s_one_hot)
rows = np.arange(expert_s_array.shape[0])[:, None]
cols = np.arange(expert_s_array.shape[1])
print( f'rows {rows} \n cols {cols} \n {expert_s_array.shape[1]}')
expert_s_one_hot[rows, cols, expert_s_array] = 1.0
print(expert_s_one_hot)