import numpy as np
import torch
import torch.nn.functional as F
import pybullet as p
import time
import os
import sys

# Add path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Imports
from SGE_Transformer.envs.coveragel_multi_room import CoverageAviary
from SGE_Transformer.experiment.sge_pybullet.sge_trajectory_transformer_gptneo import TrajectoryTransformerGPTNeo
from SGE_Transformer.experiment.sge_pybullet.extract_trajectories_macro import MacroActionSpace27

def beam_search_plan_neo(model, state_seq, action_seq, reward_seq, timestep_seq, 
                    steps_to_plan=5, beam_width=5, num_candidates=5, device='cpu', stochastic=True):
    """
    Perform Beam Search to find the best first action.
    """
    # 1. Prepare initial tensors (Batch=1)
    s_t = torch.tensor(state_seq, dtype=torch.float32, device=device).unsqueeze(0)
    a_t = torch.tensor(action_seq, dtype=torch.long, device=device).unsqueeze(0) 
    r_t = torch.tensor(reward_seq, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(2)
    t_t = torch.tensor(timestep_seq, dtype=torch.long, device=device).unsqueeze(0)
    
    # We are currently at a step where we have a 'dummy' action at the end of a_t.
    # We want to fill this dummy action. This index is:
    plan_start_idx = a_t.shape[1] - 1
    
    # Helper for padding to match training distribution (Right-aligned valid data)
    def prepare_model_inputs(s, a, r, t, ctx_len):
        cur_len = s.shape[1]
        if cur_len >= ctx_len:
            # Truncate to last context_len
            return (s[:, -ctx_len:, :], 
                    a[:, -ctx_len:], 
                    r[:, -ctx_len:, :], 
                    t[:, -ctx_len:])
        else:
            # Left Pad with zeros
            pad_len = ctx_len - cur_len
            s_pad = torch.zeros((1, pad_len, s.shape[2]), device=device)
            a_pad = torch.zeros((1, pad_len), dtype=torch.long, device=device)
            r_pad = torch.zeros((1, pad_len, 1), device=device)
            t_pad = torch.zeros((1, pad_len), dtype=torch.long, device=device)
            
            return (torch.cat([s_pad, s], dim=1),
                    torch.cat([a_pad, a], dim=1),
                    torch.cat([r_pad, r], dim=1),
                    torch.cat([t_pad, t], dim=1))

    # Initial Beam
    # Each beam item: (score, s_seq_tensor, a_seq_tensor, r_seq_tensor, t_seq_tensor)
    beams = [(0.0, s_t, a_t, r_t, t_t)]
    
    for step in range(steps_to_plan):
        candidates = []
        
        for score, s, a, r, t in beams:
            # Context window handling
            s_in, a_in, r_in, t_in = prepare_model_inputs(s, a, r, t, model.max_length)
                
            # 1. Predict Action
            with torch.no_grad():
                a_preds, _, _ = model(s_in, a_in, r_in, t_in)
                last_logits = a_preds[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                
                if stochastic:
                    num_samples = min(num_candidates, probs.shape[-1])
                    top_indices = torch.multinomial(probs, num_samples)
                    top_probs = probs.gather(-1, top_indices)
                else:
                    top_probs, top_indices = torch.topk(probs, num_candidates)
                
            for i in range(len(top_indices)):
                action_idx = top_indices[i].item()
                
                # --- Branching ---
                # Key fix: Replace dummy instead of appending
                branch_a = a.clone()
                branch_a[0, -1] = action_idx
                
                # --- Step 1: Predict Reward (using s_t, a_t) ---
                s_in_r, a_in_r, r_in_r, t_in_r = prepare_model_inputs(s, branch_a, r, t, model.max_length)
                
                with torch.no_grad():
                     _, r_pred_seq, _ = model(s_in_r, a_in_r, r_in_r, t_in_r)
                
                pred_reward = r_pred_seq[0, -1, 0].item()
                
                # --- Step 2: Predict Next State (using s_t, a_t, r_t_pred) ---
                branch_r = r.clone()
                branch_r[0, -1, 0] = pred_reward
                
                s_in_s, a_in_s, r_in_s, t_in_s = prepare_model_inputs(s, branch_a, branch_r, t, model.max_length)
                
                with torch.no_grad():
                    _, _, s_pred_seq = model(s_in_s, a_in_s, r_in_s, t_in_s)
                    
                pred_next_state = s_pred_seq[0, -1, :].unsqueeze(0).unsqueeze(0)
                
                # 3. Prepare for NEXT planning step
                # s: Append predicted next state
                next_s = torch.cat([s, pred_next_state], dim=1)
                
                # a: Append NEW dummy action for next step
                dummy_a = torch.zeros((1, 1), dtype=torch.long, device=device)
                next_a = torch.cat([branch_a, dummy_a], dim=1)
                
                # r: Append NEW dummy reward for next step
                dummy_r = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
                next_r = torch.cat([branch_r, dummy_r], dim=1)
                
                # t: Append next timestep
                next_time_val = t[0, -1] + 1
                next_t = torch.cat([t, next_time_val.view(1, 1)], dim=1)
                
                candidates.append((score + pred_reward, next_s, next_a, next_r, next_t))
        
        # Select best K
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
    
    best_beam = beams[0]
    best_a_seq = best_beam[2]

    first_planned_action = best_a_seq[0, plan_start_idx].item()
    return first_planned_action


def test_trajectory_transformer_gptneo(model_path, num_test_episodes=10, stochastic=True):
    STATE_DIM = 6
    ACT_DIM = 27
    CONTEXT_LEN = 20
    HIDDEN_SIZE = 128
    
    POS_SCALE = 25.0
    RPY_SCALE = 3.14
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load Model (GPT-Neo)
    model = TrajectoryTransformerGPTNeo(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        max_length=CONTEXT_LEN,
        n_layer=4, 
        n_head=4,
        hidden_size=HIDDEN_SIZE,
        attention_types=[[["local", "global"], 2]]
    ).to(DEVICE)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            print("❌ Invalid checkpoint format")
            return
        print(f"✅ Model loaded: {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
        return

    model.eval()

    env = CoverageAviary(
        gui=False, 
        obstacles=True,
        num_rays=120,
        radar_radius=8.0,
        grid_res=0.5
    )
    action_space = MacroActionSpace27(move_distance=0.5)

    obs, info = env.reset()
    start_snapshot = env.get_snapshot()

    episode_rewards = []
    episode_coverages = []

    for episode in range(num_test_episodes):
        env.restore_snapshot(start_snapshot)
        
        init_state = np.concatenate([env.pos[0], env.rpy[0]])
        
        states = [init_state]
        actions = [0] 
        rewards = [0.0]
        timesteps = [0]
        
        total_reward = 0.0
        done = False
        step_count = 0
        max_steps = 150
        
        print(f"\n--- Episode {episode + 1} Start (Neo Beam Search) ---")

        while not done and step_count < max_steps:
            
            cur_s = np.array(states)
            cur_a = np.array(actions)
            cur_r = np.array(rewards)
            cur_t = np.array(timesteps)
            
            # Normalize
            cur_s_norm = cur_s.copy()
            cur_s_norm[:, :3] /= POS_SCALE
            cur_s_norm[:, 3:] /= RPY_SCALE
            
            # BEAM SEARCH
            best_action = beam_search_plan_neo(
                model, cur_s_norm, cur_a, cur_r, cur_t,
                steps_to_plan=3, 
                beam_width=3,
                num_candidates=5,
                device=DEVICE,
                stochastic=stochastic
            )
            
            action_id = best_action
            
            if len(actions) == len(states): 
                 actions[-1] = action_id
            else:
                 actions.append(action_id)
            
            start_pos = env.pos[0].copy()
            displacement = action_space.get_displacement(action_id)
            target_pos = start_pos + displacement

            step_reward = 0.0
            if 0.5 <= target_pos[2] <= 3.5:
                p.resetBasePositionAndOrientation(env.DRONE_IDS[0], target_pos, [0, 0, 0, 1], physicsClientId=env.CLIENT)
                env._updateAndStoreKinematicInformation()
                _, step_reward, terminated, truncated, _ = env.compute_scan_at_pos(target_pos)
                
                total_reward += step_reward
                if terminated:
                    print(f"💥 Collision at step {step_count}")
                    done = True
            else:
                step_reward = 0
                done = True 
            
            rewards[-1] = step_reward 
            
            next_state = np.concatenate([env.pos[0], env.rpy[0]])
            states.append(next_state)
            
            actions.append(0) 
            rewards.append(0.0) 
            timesteps.append(step_count + 1)
            
            step_count += 1
            
            # if step_count % 10 == 0:
            #     print(f"Step {step_count} | Action: {action_id} | Reward: {step_reward:.2f} | Cov: {env._computeInfo()['coverage_ratio']:.2%}")

        final_coverage = env._computeInfo()['coverage_ratio']
        episode_rewards.append(total_reward)
        episode_coverages.append(final_coverage)

        print(f"🏁 Episode {episode + 1} End | Total: {total_reward:.2f} | Coverage: {final_coverage:.2%}")

    env.close()

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_coverage = np.mean(episode_coverages)
    std_coverage = np.std(episode_coverages)
    
    print("\n" + "="*40)
    print(f"📊 TT-Neo Test Summary")
    print(f"Average Reward:   {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Coverage: {avg_coverage:.2%} ± {std_coverage:.2%}")
    
    return avg_reward, std_reward, avg_coverage, std_coverage

if __name__ == "__main__":
    import csv
    
    EPOCHS = 20000
    INTERVAL = 500
    RESULTS_FILE = "sge_tt_gptneo_benchmark_results.csv"
    
    all_results = []
    
    # Path handling helper
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for epoch in range(0, EPOCHS + 1, INTERVAL):
        MODEL_NAME = f"sge_tt_gptneo_model_{epoch}.pth"
        
        candidates = [
            MODEL_NAME,
            os.path.join(current_dir, MODEL_NAME),
            os.path.join("SGE_Transformer/experiment/sge_pybullet", MODEL_NAME)
        ]
        
        MODEL_PATH = None
        for cand_path in candidates:
            if os.path.exists(cand_path):
                MODEL_PATH = cand_path
                break
                
        if MODEL_PATH:
            print(f">>> Testing Checkpoint: {epoch} steps <<<")
            # Beam search is slow, so we use fewer episodes for benchmark (e.g. 5 or 10)
            avg_r, std_r, avg_c, std_c = test_trajectory_transformer_gptneo(
                MODEL_PATH, 
                num_test_episodes=20, 
                stochastic=True
            )
            all_results.append([epoch, avg_r, std_r, avg_c, std_c])
        else:
             print(f"Skipping epoch {epoch}: Model {MODEL_NAME} not found.")

    # Save results to file
    with open(RESULTS_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Average Reward', 'Reward Std', 'Average Coverage', 'Coverage Std'])
        writer.writerows(all_results)
        
    print(f"Detailed results saved to {RESULTS_FILE}")
    print("="*40 + "\n")
