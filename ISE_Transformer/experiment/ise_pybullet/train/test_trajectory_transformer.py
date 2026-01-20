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
from SGE_Transformer.experiment.sge_pybullet.sge_trajectory_transformer import TrajectoryTransformer
from SGE_Transformer.experiment.sge_pybullet.extract_trajectories_macro import MacroActionSpace27

def beam_search_plan(model, state_seq, action_seq, reward_seq, timestep_seq, 
                    steps_to_plan=5, beam_width=5, num_candidates=5, device='cpu', stochastic=True):
    """
    Perform Vectorized Beam Search to find the best first action.
     Optimized for speed using full batch processing.
    """
    # 1. Prepare initial tensors (Batch=1)
    s_b = torch.tensor(state_seq, dtype=torch.float32, device=device).unsqueeze(0)
    a_b = torch.tensor(action_seq, dtype=torch.long, device=device).unsqueeze(0) 
    r_b = torch.tensor(reward_seq, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(2)
    t_b = torch.tensor(timestep_seq, dtype=torch.long, device=device).unsqueeze(0)
    
    # Track accumulated scores for each beam
    scores = torch.zeros(1, device=device)
    
    # We want the action at this specific index from the best beam later
    plan_start_idx = a_b.shape[1] - 1
    ctx_len = model.max_length

    # Helper: Vectorized padding/truncation for a batch of sequences
    def prepare_batched_inputs(s, a, r, t, ctx):
        curr_l = s.shape[1]
        bs = s.shape[0]
        if curr_l >= ctx:
            return s[:, -ctx:], a[:, -ctx:], r[:, -ctx:], t[:, -ctx:]
        
        # Left Pad
        pad_l = ctx - curr_l
        s_pad = torch.zeros((bs, pad_l, s.shape[2]), device=device)
        a_pad = torch.zeros((bs, pad_l), dtype=torch.long, device=device)
        r_pad = torch.zeros((bs, pad_l, 1), device=device)
        t_pad = torch.zeros((bs, pad_l), dtype=torch.long, device=device)
        
        return (torch.cat([s_pad, s], dim=1),
                torch.cat([a_pad, a], dim=1),
                torch.cat([r_pad, r], dim=1),
                torch.cat([t_pad, t], dim=1))

    for step in range(steps_to_plan):
        # --- Batch Step 1: Predict Action Distribution ---
        # Input shape: (Batch, Seq_Len, ...)
        s_in, a_in, r_in, t_in = prepare_batched_inputs(s_b, a_b, r_b, t_b, ctx_len)
        
        with torch.no_grad():
            # Use mixed precision for extra speed on CUDA
            with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
                a_preds, _, _ = model(s_in, a_in, r_in, t_in)
            
            # Get logits at the last position (where we want to plan)
            last_logits = a_preds[:, -1, :] # (Batch, Act_Dim)
            probs = F.softmax(last_logits, dim=-1)

        # Sampling / Selection
        if stochastic:
            # Vectorized multinomial sampling
            top_indices = torch.multinomial(probs, num_candidates) # (Batch, Num_Cand)
            # top_probs = probs.gather(1, top_indices) # Not used in score currently
        else:
            _, top_indices = torch.topk(probs, num_candidates) # (Batch, Num_Cand)

        # --- Batch Step 2: Expand Beams (Branching) ---
        # Current Batch Size: B
        # New Batch Size: B * Num_Cand
        B = s_b.shape[0]
        K = num_candidates
        
        # Repeat existing history K times
        # View as (B, K, ...) then flatten to (B*K, ...)
        s_b = s_b.unsqueeze(1).repeat(1, K, 1, 1).flatten(0, 1)
        a_b = a_b.unsqueeze(1).repeat(1, K, 1).flatten(0, 1)
        r_b = r_b.unsqueeze(1).repeat(1, K, 1, 1).flatten(0, 1)
        t_b = t_b.unsqueeze(1).repeat(1, K, 1).flatten(0, 1)
        scores = scores.unsqueeze(1).repeat(1, K).flatten(0, 1)
        
        # Fill the "dummy" action slot with the chosen candidates
        current_actions = top_indices.flatten() # (B*K,)
        a_b[:, -1] = current_actions

        # --- Batch Step 3: Predict Reward & Next State ---
        # 3.1 Predict Reward using filled action
        s_in, a_in, r_in, t_in = prepare_batched_inputs(s_b, a_b, r_b, t_b, ctx_len)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
                _, r_preds, _ = model(s_in, a_in, r_in, t_in)
        
        # Get predicted reward
        pred_reward = r_preds[:, -1, 0] # (B*K,)
        
        # Update Reward History: replace dummy 0 with predicted reward
        # CRITICAL: This enables the State Head to see the correct (s, a, r) context
        r_b[:, -1, 0] = pred_reward
        
        # Update Scores
        scores += pred_reward

        # 3.2 Predict Next State using filled reward
        s_in, a_in, r_in, t_in = prepare_batched_inputs(s_b, a_b, r_b, t_b, ctx_len)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
                _, _, s_preds = model(s_in, a_in, r_in, t_in)
        
        pred_next_state = s_preds[:, -1, :].unsqueeze(1) # (B*K, 1, State_Dim)

        # --- Batch Step 4: Prepare Next Step (Append Dummies) ---
        # Append predicted state
        s_b = torch.cat([s_b, pred_next_state], dim=1)
        
        # Append new dummies
        dummy_a = torch.zeros((s_b.shape[0], 1), dtype=torch.long, device=device)
        a_b = torch.cat([a_b, dummy_a], dim=1)
        
        dummy_r = torch.zeros((s_b.shape[0], 1, 1), dtype=torch.float32, device=device)
        r_b = torch.cat([r_b, dummy_r], dim=1)
        
        next_t_val = t_b[:, -1:] + 1
        t_b = torch.cat([t_b, next_t_val], dim=1)

        # --- Batch Step 5: Pruning (Global Top-K) ---
        # If total branches > beam_width, keep only best ones
        if s_b.shape[0] > beam_width:
            _, best_indices = torch.topk(scores, beam_width)
            
            s_b = s_b[best_indices]
            a_b = a_b[best_indices]
            r_b = r_b[best_indices]
            t_b = t_b[best_indices]
            scores = scores[best_indices]

    # Return the first planned action of the highest scoring beam
    best_idx = torch.argmax(scores)
    best_a_seq = a_b[best_idx]
    
    first_planned_action = best_a_seq[plan_start_idx].item()
    return first_planned_action


def test_trajectory_transformer(model_path, num_test_episodes=10, stochastic=True):
    # Parameters
    STATE_DIM = 6
    ACT_DIM = 27
    CONTEXT_LEN = 20
    # Add Hidden Size param explicitly to match training
    HIDDEN_SIZE = 128
    
    POS_SCALE = 25.0
    RPY_SCALE = 3.14
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load Model
    model = TrajectoryTransformer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        max_length=CONTEXT_LEN,
        n_layer=4, 
        n_head=4,
        hidden_size=HIDDEN_SIZE
    ).to(DEVICE)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        # Handle formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict']) # Lightning
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint) # Standard
        else:
            print("❌ Invalid checkpoint format")
            return
        print(f"✅ Model loaded: {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
        return

    model.eval()

    # Environment
    env = CoverageAviary(
        gui=False, 
        obstacles=True,
        num_rays=120,
        radar_radius=8.0,
        grid_res=0.5
    )
    action_space = MacroActionSpace27(move_distance=0.5)

    obs, info = env.reset()
    # Fix the issue with 0 initial coverage by manually setting higher Z if needed or just ignoring first step
    start_snapshot = env.get_snapshot()

    episode_rewards = []
    episode_coverages = []

    for episode in range(num_test_episodes):
        env.restore_snapshot(start_snapshot)
        
        # Init State
        init_state = np.concatenate([env.pos[0], env.rpy[0]])
        
        # History
        # Note: We need to feed normalized states to model, but keep raw for history?
        # Let's store raw in lists, normalize before feeding.
        states = [init_state]
        actions = [0] # Dummy action at t=0? Or we start with NO action?
        # standard: s0. Predict a0.
        # model inputs: s, a, r.
        # input: s=[s0], a=[0], r=[0] -> pred a0? 
        # Yes, using dummy placeholders.
        
        rewards = [0.0]
        timesteps = [0]
        
        total_reward = 0.0
        done = False
        step_count = 0
        max_steps = 150
        
        print(f"\n--- Episode {episode + 1} Start (Beam Search Planning) ---")

        while not done and step_count < max_steps:
            
            # Prepare inputs (Lists -> Numpy -> Norm -> Tensor)
            # Use last CONTEXT_LEN
            
            # 1. Prepare raw sequences
            # We strip the "future placeholders".
            # Current knowns: s[0...t], a[0...t-1], r[0...t-1]
            # But my lists have placeholders initialized above.
            
            # Let's clean up lists logic:
            # t=0: states=[s0], actions=[], rewards=[]
            # But to make tensor, we need at least one dim.
            
            # Input construction:
            cur_s = np.array(states)
            cur_a = np.array(actions) # Length must match s? 
            cur_r = np.array(rewards)
            cur_t = np.array(timesteps)
            
            # Normalize State
            cur_s_norm = cur_s.copy()
            cur_s_norm[:, :3] /= POS_SCALE
            cur_s_norm[:, 3:] /= RPY_SCALE
            
            # BEAM SEARCH
            # We pass the collected history.
            # But actions/rewards are currently 1 length longer (dummies)? 
            # Let's handle alignment carefully.
            
            # In loop t=0:
            # We have s0.
            # We pass s=[s0], a=[0], r=[0].
            # Model predicts a0.
            
            # Execute Beam Search
            # It returns the best a0.
            
            best_action = beam_search_plan(
                model, cur_s_norm, cur_a, cur_r, cur_t,
                steps_to_plan=3, # Small horizon for speed
                beam_width=3,
                num_candidates=5,
                device=DEVICE,
                stochastic=stochastic
            )
            
            action_id = best_action
            
            # --- Execution ---
            # Update 'actions' list with the REAL action taken (replace dummy if needed)
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
            
            # Update History
            rewards[-1] = step_reward # Replace dummy reward with real reward
            
            next_state = np.concatenate([env.pos[0], env.rpy[0]])
            states.append(next_state)
            
            # Appending dummies for next step
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
    print(f"📊 TT (Beam Search) Test Summary")
    print(f"Average Reward:   {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Coverage: {avg_coverage:.2%} ± {std_coverage:.2%}")
    
    return avg_reward, std_reward, avg_coverage, std_coverage

if __name__ == "__main__":
    import csv
    
    EPOCHS = 20000
    INTERVAL = 500
    RESULTS_FILE = "sge_tt_benchmark_results.csv"
    
    all_results = []
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for epoch in range(0, EPOCHS + 1, INTERVAL):
        MODEL_NAME = f"sge_tt_model_{epoch}.pth"
        
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
            print(f">>> Testing Checkpoint: {epoch} epochs <<<")
            avg_r, std_r, avg_c, std_c = test_trajectory_transformer(MODEL_PATH, num_test_episodes=20, stochastic=True)
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
