import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from config.constants import CLUB_ID_MAP
from helpers import get_club_elos
from src.ctmc.ctmc_builder import (
    build_global_matrices,
    standardise_possessions,
    align_team_perspective,
)
from src.strategies.matrix_pipeline import MatrixPipeline
from src.strategies.elo_strategy import EloModifier
from src.engine.live_simulator import run_live_pytorch_monte_carlo
from src.market.metrics import calculate_market_rmse
from tqdm import tqdm

torch.set_num_threads(os.cpu_count())

# Auto-detect GPU hardware acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_gamma_candidate(
    gamma: float,
    alpha_candidates: List[float],
    beta_candidates: List[float],
    val_fixtures: List[dict],
    elos: Dict[str, float],
    master_df: pd.DataFrame,
    team_dfs: Dict[str, pd.DataFrame],
    num_simulations: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Re-builds global & team matrices for a specific gamma and tunes (alpha, beta).
    Returns (best_rmse_for_gamma, best_alpha, best_beta).
    """
    # 1. Build Global Matrices for candidate gamma
    global_N, global_T, _ = build_global_matrices(master_df, gamma=gamma)

    # 2. Build Team Matrices for candidate gamma
    team_matrices = {}
    for club in CLUB_ID_MAP.keys():
        if club in team_dfs and not team_dfs[club].empty:
            df_aligned = align_team_perspective(
                team_dfs[club], CLUB_ID_MAP[club], sim_role="H"
            )
            N_h, T_h, _ = build_global_matrices(df_aligned, gamma=gamma)

            df_aligned_a = align_team_perspective(
                team_dfs[club], CLUB_ID_MAP[club], sim_role="A"
            )
            N_a, T_a, _ = build_global_matrices(df_aligned_a, gamma=gamma)

            team_matrices[club] = {
                "home_N": N_h,
                "home_T": T_h,
                "away_N": N_a,
                "away_T": T_a,
            }

    # 3. Tune (alpha, beta) on these matrices
    pipeline = MatrixPipeline(strategies=[EloModifier()])
    best_gamma_rmse = float("inf")
    best_alpha, best_beta = None, None

    total_pairs = len(alpha_candidates) * len(beta_candidates)

    with tqdm(
        total=total_pairs,
        desc=f"  Tuning (alpha, beta) for Gamma={gamma:.2f}",
        leave=False,
    ) as pbar:

        for alpha in alpha_candidates:
            for beta in beta_candidates:
                torch.manual_seed(seed)
                np.random.seed(seed)
                rmse_scores = []

                for fixture in val_fixtures:
                    home, away = fixture["home_team"], fixture["away_team"]
                    odds = fixture["bookie_odds"]
                    clock_sec = fixture.get("clock_seconds", 0.0)

                    if home not in team_matrices or away not in team_matrices:
                        continue

                    home_N = team_matrices[home]["home_N"]
                    home_T = team_matrices[home]["home_T"]
                    away_N = team_matrices[away]["away_N"]
                    away_T = team_matrices[away]["away_T"]

                    ctx = {
                        "home_team": home,
                        "away_team": away,
                        "elo_home": elos.get(home, 1500.0),
                        "elo_away": elos.get(away, 1500.0),
                        "alpha": alpha,
                        "beta": beta,
                    }

                    Q_pre_np = pipeline.build_grid_fast(
                        global_N, global_T, home_N, home_T, away_N, away_T, ctx
                    )
                    Q_pre_tensor = torch.tensor(
                        Q_pre_np, dtype=torch.float32, device=device
                    )

                    prob_h, prob_d, prob_a = run_live_pytorch_monte_carlo(
                        q_matrix=Q_pre_tensor,
                        current_clock=clock_sec,
                        current_state_idx=2,
                        live_home_goals=0,
                        live_away_goals=0,
                        num_simulations=num_simulations,
                    )

                    rmse = calculate_market_rmse([prob_h, prob_d, prob_a], odds)
                    rmse_scores.append(rmse)

                avg_rmse = float(np.mean(rmse_scores)) if rmse_scores else 999.0

                if avg_rmse < best_gamma_rmse:
                    best_gamma_rmse = avg_rmse
                    best_alpha = alpha
                    best_beta = beta
                pbar.update(1)

    return best_gamma_rmse, best_alpha, best_beta


def run_gamma_tuning(num_simulations: int = 2000):
    print("==================================================================")
    print("      STAGE 2: TUNING HISTORICAL SEASON DECAY RATE (GAMMA)        ")
    print("==================================================================")

    # 1. Fetch Elos and load raw match DataFrames ONCE
    print("[*] Fetching up-to-date ClubElo ratings...")
    elos = get_club_elos(cache_dir="./cache/")

    print("[*] Pre-loading historical raw match DataFrames...")
    master_df = pd.read_parquet("./cache/master_df.parquet")

    team_dfs = {}
    for club in CLUB_ID_MAP.keys():
        cleaned_name = club.replace(" ", "_").lower()
        raw_df = pd.read_parquet(f"./cache/{cleaned_name}_events_df.parquet")
        if not raw_df.empty:
            team_dfs[club] = standardise_possessions(raw_df)

    # 2. Validation Fixtures
    val_fixtures = [
        {
            "home_team": "Bournemouth",
            "away_team": "Leeds",
            "bookie_odds": [1.98, 4.00, 4.40],
        },
        {
            "home_team": "Sunderland",
            "away_team": "Nottingham Forest",
            "bookie_odds": [2.78, 3.24, 2.92],
        },
        {
            "home_team": "Liverpool",
            "away_team": "Crystal Palace",
            "bookie_odds": [1.65, 4.68, 5.80],
        },
        {
            "home_team": "Arsenal",
            "away_team": "Newcastle",
            "bookie_odds": [1.44, 5.25, 8.25],
        },
        {
            "home_team": "West Ham",
            "away_team": "Everton",
            "bookie_odds": [2.56, 3.61, 3.03],
        },
        {
            "home_team": "Wolves",
            "away_team": "Tottenham",
            "bookie_odds": [5.10, 4.70, 1.71],
        },
        {
            "home_team": "Man Utd",
            "away_team": "Brentford",
            "bookie_odds": [2.05, 3.94, 3.93],
        },
        {
            "home_team": "Aston Villa",
            "away_team": "Tottenham",
            "bookie_odds": [2.35, 3.81, 3.17],
        },
    ]

    # 3. Candidate Grids
    gamma_candidates = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
    alpha_candidates = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
    beta_candidates = [0.00001, 0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030]

    overall_best_rmse = float("inf")
    overall_best_gamma = None
    overall_best_alpha = None
    overall_best_beta = None

    print(f"[*] Testing {len(gamma_candidates)} Gamma values...")
    print("------------------------------------------------------------------")

    for g_idx, gamma in enumerate(gamma_candidates, 1):
        g_rmse, g_alpha, g_beta = evaluate_gamma_candidate(
            gamma=gamma,
            alpha_candidates=alpha_candidates,
            beta_candidates=beta_candidates,
            val_fixtures=val_fixtures,
            elos=elos,
            master_df=master_df,
            team_dfs=team_dfs,
            num_simulations=num_simulations,
        )

        print(
            f"[{g_idx}/{len(gamma_candidates)}] Gamma: {gamma:<5.2f} --> Best RMSE: {g_rmse:.4f} (Alpha: {g_alpha}, Beta: {g_beta})"
        )

        if g_rmse < overall_best_rmse:
            overall_best_rmse = g_rmse
            overall_best_gamma = gamma
            overall_best_alpha = g_alpha
            overall_best_beta = g_beta

    print("\n==================================================================")
    print("      GAMMA TUNING COMPLETE: OVERALL OPTIMAL TRIPLET FOUND        ")
    print("==================================================================")
    print(f"BEST GAMMA: {overall_best_gamma}")
    print(f"BEST ALPHA: {overall_best_alpha}")
    print(f"BEST BETA:  {overall_best_beta}")
    print(f"MINIMUM MARKET RMSE: {overall_best_rmse:.4f}")
    print("==================================================================\n")


if __name__ == "__main__":
    run_gamma_tuning(num_simulations=800)
