import torch
import numpy as np
import time
from config.constants import STATE_TO_IDX


def run_live_pytorch_monte_carlo(
    q_matrix: torch.Tensor,
    current_clock: float,
    current_state_idx: int,
    live_home_goals: int,
    live_away_goals: int,
    num_simulations: int = 10000,
    match_seconds: float = 5400.0,
    verbose: bool = False,
):
    """
    Ultra-fast Vectorized Monte Carlo engine using Fixed-Width Masking and Zero Branching.
    Accepts raw VRAM tensors and live scoreboard states for sub-20ms execution.
    """
    if verbose:
        print(f"\n[*] Prepping matrix for {num_simulations} parallel simulations...")

    # Inherit device directly from the incoming tensor to avoid CPU-GPU PCIe syncs
    device = q_matrix.device
    num_states = q_matrix.shape[0]  # Guaranteed to be 12

    # Canonical index mapping from your database architecture
    kickoff_h_idx = STATE_TO_IDX["Z:2_P:H"]
    kickoff_a_idx = STATE_TO_IDX["Z:2_P:A"]
    goal_h_idx = STATE_TO_IDX["Goal_H"]
    goal_a_idx = STATE_TO_IDX["Goal_A"]

    # =========================================================================
    # 1. 100% VECTORIZED MATRIX SETUP (Zero Python loops or Pandas calls)
    # =========================================================================
    # Zero out the diagonal to isolate true positive departure transition rates
    q_off_diag = q_matrix.clone()
    q_off_diag.fill_diagonal_(0.0)

    # Sum only off-diagonal elements to get the true positive exit rate out of each state
    exit_rates = q_off_diag.sum(dim=1)

    # Prevent division-by-zero on empty states by substituting a tiny epsilon
    safe_rates = torch.where(
        exit_rates == 0, torch.tensor(0.0001, device=device), exit_rates
    )

    # Broadcast division using the clean off-diagonal matrix to get valid probabilities
    transition_probs = q_off_diag / safe_rates.unsqueeze(1)

    # -------------------------------------------------------------------------
    # CRITICAL MATH SAFETY INTERACTION LAYER
    # -------------------------------------------------------------------------
    # 1. Force any accidentally negative probabilities to 0.0
    transition_probs = torch.clamp(transition_probs, min=0.0)

    # 2. Handle zero-sum rows to prevent NaNs/Infs during sampling
    row_sums = transition_probs.sum(dim=1, keepdim=True)
    zero_rows = row_sums == 0.0

    transition_probs = torch.where(
        zero_rows,
        torch.full_like(transition_probs, 1.0 / num_states),
        transition_probs,
    )
    row_sums = torch.where(zero_rows, torch.tensor(1.0, device=device), row_sums)

    # 3. Re-normalize to ensure every single row sums exactly to 1.0
    transition_probs = transition_probs / row_sums
    # -------------------------------------------------------------------------

    # Enforce absorbing state rules for Goal_H (10) and Goal_A (11)
    exit_rates[goal_h_idx:] = 1.0
    transition_probs[goal_h_idx:, :] = 0.0
    transition_probs[goal_h_idx, goal_h_idx] = 1.0
    transition_probs[goal_a_idx, goal_a_idx] = 1.0

    # =========================================================================
    # 2. DYNAMIC TIME CEILING (Handles Stoppage Time & Extra Time)
    # =========================================================================
    # If clock is past 88 mins, ensure we simulate at least +4 mins of stoppage time
    effective_max_time = max(float(match_seconds), float(current_clock) + 240.0)

    # =========================================================================
    # 3. FIXED-WIDTH STATE ALLOCATION
    # =========================================================================
    current_states = torch.full(
        (num_simulations,), int(current_state_idx), device=device, dtype=torch.long
    )
    times = torch.full(
        (num_simulations,), float(current_clock), device=device, dtype=torch.float32
    )
    home_goals = torch.full(
        (num_simulations,), int(live_home_goals), device=device, dtype=torch.long
    )
    away_goals = torch.full(
        (num_simulations,), int(live_away_goals), device=device, dtype=torch.long
    )

    if verbose:
        print("[*] Firing PyTorch Engine...")

    start_time = time.time()
    active_mask = times < effective_max_time

    # The outer loop only checks CPU-GPU motherboard sync ONCE every 100 steps
    while active_mask.any():

        for _ in range(100):
            # 1. Unconditional Goal Detection
            is_goal_h = (current_states == goal_h_idx) & active_mask
            is_goal_a = (current_states == goal_a_idx) & active_mask

            home_goals += is_goal_h.long()
            away_goals += is_goal_a.long()

            # Add 30 seconds for celebrations only where goals occurred
            times += torch.where(is_goal_h | is_goal_a, 30.0, 0.0)

            # Instantly reset goal states back to the opponent's kickoff
            current_states = torch.where(is_goal_h, kickoff_a_idx, current_states)
            current_states = torch.where(is_goal_a, kickoff_h_idx, current_states)

            # 2. Vectorized Time Jumps across the entire batch
            rates = exit_rates[current_states]
            u = torch.rand(num_simulations, device=device)
            dt = -torch.log(u) / rates

            times += torch.where(active_mask, dt, 0.0)
            active_mask = times < effective_max_time

            # 3. Vectorized State Transitions
            probs = transition_probs[current_states]
            next_states = torch.multinomial(probs, 1).squeeze(1)

            current_states = torch.where(active_mask, next_states, current_states)

    elapsed = time.time() - start_time
    h_goals = home_goals.cpu().numpy()
    a_goals = away_goals.cpu().numpy()

    home_wins = np.sum(h_goals > a_goals)
    draws = np.sum(h_goals == a_goals)
    away_wins = np.sum(a_goals > h_goals)

    prob_h = float(home_wins / num_simulations)
    prob_d = float(draws / num_simulations)
    prob_a = float(away_wins / num_simulations)

    sim_xg_home = float(np.mean(h_goals))
    sim_xg_away = float(np.mean(a_goals))

    if verbose:
        print("\n==================================")
        print("      FINAL MATCH FORECAST        ")
        print("==================================")
        print(f"Hardware Time: {elapsed:.3f} seconds")
        print(f"Home Win:  {(prob_h * 100):.2f}%")
        print(f"Draw:      {(prob_d * 100):.2f}%")
        print(f"Away Win:  {(prob_a * 100):.2f}%")
        print("----------------------------------")
        print(f"Expected Score: Home {sim_xg_home:.2f} - {sim_xg_away:.2f} Away")
        print("==================================\n")

    return prob_h, prob_d, prob_a
