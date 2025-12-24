import os
import random
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import dill as pickle
import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm
from visualization import Visu

from environment import GridWorld
from network import append_state

from transformerbc import BCTrainer, HuggingFaceTransformerBCNetwork
from transformerbc_test import expand_trajectory_states,expand_batch_states
from GPTNeobc import GPTNEOBCTrainer,HuggingFaceGptNeoBCNetwork
from mamba2bc import Mamba2BCTrainer,Mamba2BCNetwork

def test_agent_TRANSBC2(agent: BCTrainer, env, n_episode,params,H, appear=False):
    env.common_params["batch_size"] = n_episode
    mat_state = []
    mat_return = []
    env.initialize(params["env"]["initial"])
    mat_state.append(env.state)
    init_state = env.state
    for h_iter in range(H - 1):
        batch_state = append_state(mat_state, h_iter + 1).long()
        # print(len(batch_state))
        # b=np.array(expand_batch_states(batch_state,H)).squeeze()
        # print(len(b))
        actions, _ = agent.predict(batch_state,H, deterministic=False)
        # print("actions", actions)
        env.step(h_iter, actions)
        mat_state.append(env.state)  # s+1

    mat_return = env.weighted_traj_return(mat_state, type=params["alg"]["type"]).float().mean()
    if appear == True:
        obj = env.weighted_traj_return(mat_state).float()
        print(" mean ", obj.mean(), " max ",
              obj.max(), " median ", obj.median(), " min ", obj.min())
    return mat_return

def test_agent_GPTNEOBC(agent: GPTNEOBCTrainer, env, n_episode,params,H, appear=False):
    env.common_params["batch_size"] = n_episode
    mat_state = []
    mat_return = []
    env.initialize(params["env"]["initial"])
    mat_state.append(env.state)
    init_state = env.state
    for h_iter in range(H - 1):
        batch_state = append_state(mat_state, h_iter + 1).long()
        # print(len(batch_state))
        # b=np.array(expand_batch_states(batch_state,H)).squeeze()
        # print(len(b))
        actions, _ = agent.predict(batch_state,H, deterministic=False)
        # print("actions", actions)
        env.step(h_iter, actions)
        mat_state.append(env.state)  # s+1

    mat_return = env.weighted_traj_return(mat_state, type=params["alg"]["type"]).float().mean()
    if appear == True:
        obj = env.weighted_traj_return(mat_state).float()
        print(" mean ", obj.mean(), " max ",
              obj.max(), " median ", obj.median(), " min ", obj.min())
    return mat_return

def test_agent_MAMBA2BC(agent: Mamba2BCTrainer, env, n_batch,params,H, appear=False):
    env.common_params["batch_size"] = n_batch
    mat_state = []
    mat_return = []
    env.initialize(params["env"]["initial"])
    mat_state.append(env.state)
    init_state = env.state
    for h_iter in range(H - 1):
        batch_state = append_state(mat_state, h_iter + 1).long()
        # print(len(batch_state))
        # b=np.array(expand_batch_states(batch_state,H)).squeeze()
        # print(len(b))
        actions, _ = agent.predict(batch_state,H, deterministic=False)
        # print("actions", actions)
        env.step(h_iter, actions)
        mat_state.append(env.state)  # s+1

    mat_return = env.weighted_traj_return(mat_state, type=params["alg"]["type"]).float().mean()
    if appear == True:
        obj = env.weighted_traj_return(mat_state).float()
        print(" mean ", obj.mean(), " max ",
              obj.max(), " median ", obj.median(), " min ", obj.min())
    return mat_return


def test_gpt2bc(state_dim, env,params,device,H,to_train:str,n_episode:int):

    print("------gpt2bc---------")
    network = HuggingFaceTransformerBCNetwork(state_dim=state_dim, num_actions=5, d_model=128, num_decoder_layers=3,
                                              nhead=8,
                                              max_seq_length=80)
    trainer = BCTrainer(network, lr=1e-4, device=device)
    path = f'transbc_2_room_198_model/transbc_model_{to_train}.pth'
    trainer.load_model(path)
    a = test_agent_TRANSBC2(trainer, env, n_episode,params,H, appear=True)


def test_gptneobc(state_dim, env, params, device, H, to_train: str,n_episode:int):

    print("------gptneobc---------")

    network = HuggingFaceGptNeoBCNetwork(state_dim=state_dim, num_actions=5, d_model=128, num_decoder_layers=3,
                                              nhead=8,
                                              max_seq_length=80)
    trainer = GPTNEOBCTrainer(network, lr=1e-4, device=device)
    path = f'gptneobc_2_room_198_model/transbc_model_{to_train}.pth'
    trainer.load_model(path)
    a = test_agent_GPTNEOBC(trainer, env, n_episode,params,H, appear=True,)

def test_mambabc(state_dim, env, params, device, H, to_train: str,n_episode:int):

    print("------gptneobc---------")

    network = Mamba2BCNetwork(state_dim=state_dim, num_actions=5, d_model=128, num_decoder_layers=3,
                                              max_seq_length=80)
    trainer = Mamba2BCTrainer(network, lr=1e-4, device=device)
    path = f'mamba2bc_2_room_198_model/mamba2bc_model_{to_train}.pth'
    trainer.load_model(path)
    a = test_agent_MAMBA2BC(trainer, env, n_episode,params,H, appear=True)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    workspace = "2r198"

    params = {
        "env": {
            "start": 1,
            "step_size": 0.1,
            "shape": {"x": 11, "y": 18},
            "horizon": 80,
            "node_weight": "constant",
            "disc_size": "small",
            "n_players": 3,
            "Cx_lengthscale": 2,
            "Cx_noise": 0.001,
            "Fx_lengthscale": 1,
            "Fx_noise": 0.001,
            "Cx_beta": 1.5,
            "Fx_beta": 1.5,
            "generate": False,
            "env_file_name": 'env_data.pkl',
            "cov_module": 'Matern',
            "stochasticity": 0.0,
            "domains": "two_room_2",
            "num": 1,  # 替代原来的args.env
            "initial": 80
        },
        "alg": {
            "gamma": 1,
            "type": "NM",
            "ent_coef": 0.0,
            "epochs": 140,
            "lr": 0.02
        },
        "common": {
            "a": 1,
            "subgrad": "greedy",
            "grad": "pytorch",
            "algo": "both",
            "init": "deterministic",
            "batch_size": 3000
        },
        "visu": {
            "wb": "disabled",
            "a": 1
        }
    }

    env_load_path = workspace + \
                    "/environments/" + params["env"]["node_weight"] + "/env_1"

    params['env']['num'] = 1
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="code-" + params["env"]["node_weight"],
    #     mode=params["visu"]["wb"],
    #     config=params
    # )

    epochs = params["alg"]["epochs"]

    H = params["env"]["horizon"]
    MAX_Ret = 2 * (H + 1)
    if params["env"]["disc_size"] == "large":
        MAX_Ret = 3 * (H + 2)

    env = GridWorld(
        env_params=params["env"], common_params=params["common"], visu_params=params["visu"],
        env_file_path=env_load_path)
    node_size = params["env"]["shape"]['x'] * params["env"]["shape"]['y']
    # TransitionMatrix = torch.zeros(node_size, node_size)

    if params["env"]["node_weight"] == "entropy" or params["env"]["node_weight"] == "steiner_covering" or params["env"][
        "node_weight"] == "GP":
        a_file = open(env_load_path + ".pkl", "rb")
        data = pickle.load(a_file)
        a_file.close()

    if params["env"]["node_weight"] == "entropy":
        env.cov = data
    if params["env"]["node_weight"] == "steiner_covering":
        env.items_loc = data
    if params["env"]["node_weight"] == "GP":
        env.weight = data

    visu = Visu(env_params=params["env"])

    env.get_horizon_transition_matrix()

    state_dim = 198

    to_t="5000"
    n_episode=100

    test_gpt2bc(state_dim, env, params, device, H, to_train=to_t,n_episode=n_episode)
    test_gptneobc(state_dim, env, params, device, H, to_train=to_t,n_episode=n_episode)
    # test_mambabc(state_dim, env, params, device, H, to_train=to_t,n_episode=n_episode)


if __name__ == "__main__":
    main()