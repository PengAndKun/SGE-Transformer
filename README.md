# G-ISE-Transformer
G-ISE-Transformer: Submodular-driven Multi-batch Evolutionary Strategy for Coverage
## 中文说明

1. `ISE_Transformer`：模型主目录，包含模型核心代码和入口脚本。  
2. `ISE_Transformer/experiment`：实验代码目录，存放不同实验脚本、配置和结果管理。  
3. `ISE_Transformer/envs`：实验环境实现（例如基于 PyBullet 的仿真环境），提供环境接口与物理仿真实现。  
4. `ISE_Transformer/data_pybullet`：用于存放与读取 PyBullet 相关的数据（轨迹、采样数据、序列化文件等）。  
5. `ISE_Transformer/experiment/preliminary_experiment`：预备实验文件夹，可直接运行，用于快速验证和小规模测试。  
6. `ISE_Transformer/experiment/ise_pybullet/create_trajectories`：轨迹生成与比较算法实现目录，可直接运行以生成和比较轨迹。  
7. `ISE_Transformer/experiment/ise_pybullet/train`：模型训练相关代码与消融实验脚本，包含训练流程、超参设置与评估。  
8. `ISE_Transformer/experiment/ise_pybullet/final_data.py`：用于 G-ISE 算法比较实验的数据处理脚本，负责整理/转换用于对比的最终数据集。

## English description

1. `ISE_Transformer`: Main project folder containing the core model code and entry scripts.  
2. `ISE_Transformer/experiment`: Experiments directory holding experiment scripts, configurations, and result management.  
3. `ISE_Transformer/envs`: Environment implementations (e.g., PyBullet simulations), providing environment interfaces and physics simulation.  
4. `ISE_Transformer/data_pybullet`: Storage for PyBullet-related data (trajectories, sampled data, serialized files, etc.).  
5. `ISE_Transformer/experiment/preliminary_experiment`: Preliminary experiments folder, runnable directly for quick validation and small-scale tests.  
6. `ISE_Transformer/experiment/ise_pybullet/create_trajectories`: Folder with trajectory generation and comparison algorithms, runnable to produce and compare trajectories.  
7. `ISE_Transformer/experiment/ise_pybullet/train`: Model training and ablation experiment scripts, including training pipelines, hyperparameter settings, and evaluation.  
8. `ISE_Transformer/experiment/ise_pybullet/final_data.py`: Data processing script for G-ISE comparison experiments, responsible for preparing the final comparison datasets.