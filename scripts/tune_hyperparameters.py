import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import pandas as pd
from typing import List, Dict
from helpers import get_club_elos
from src.strategies.matrix_pipeline import MatrixPipeline
from src.strategies.elo_strategy import EloModifier
from src.engine.live_simulator import run_live_pytorch_monte_carlo
from src.market.metrics import calculate_market_rmse


def evaluate_hyperparameter_pair(
    alpha: float,
    beta: float,
    val_fixtures: List[dict],
    elos: Dict[str, float],
    global_N: np.ndarray,
    global_T: np.ndarray,
    cache_dir: str = "./cache/",
    num_simulations: int = 2000,
) -> float:
    """
    Evaluates a specific (alpha, beta) pair using pre-loaded Elos and Global priors.
    """
    rmse_scores = []
    pipeline = MatrixPipeline(strategies=[EloModifier()])

    for fixture in val_fixtures:
        home, away = fixture["home_team"], fixture["away_team"]
        odds = fixture["bookie_odds"]
        clock_sec = fixture.get("clock_seconds", 0.0)

        h_clean = home.replace(" ", "_").lower()
        a_clean = away.replace(" ", "_").lower()

        h_n_path = os.path.join(cache_dir, f"{h_clean}_N_matrix.csv")
        a_n_path = os.path.join(cache_dir, f"{a_clean}_N_matrix.csv")

        if not os.path.exists(h_n_path) or not os.path.exists(a_n_path):
            continue

        home_N = pd.read_csv(h_n_path).to_numpy(dtype=np.float64)
        home_T = pd.read_csv(
            os.path.join(cache_dir, f"{h_clean}_T_vector.csv")
        ).to_numpy(dtype=np.float64)
        away_N = pd.read_csv(a_n_path).to_numpy(dtype=np.float64)
        away_T = pd.read_csv(
            os.path.join(cache_dir, f"{a_clean}_T_vector.csv")
        ).to_numpy(dtype=np.float64)

        ctx = {
            "home_team": home,
            "away_team": away,
            "elo_home": elos.get(home, 1500.0),
            "elo_away": elos.get(away, 1500.0),
            "alpha": alpha,
            "beta": beta,
        }

        # Build Q_pre using candidate (alpha, beta)
        Q_pre_np = pipeline.build_grid_fast(
            global_N, global_T, home_N, home_T, away_N, away_T, ctx
        )
        Q_pre_tensor = torch.tensor(Q_pre_np, dtype=torch.float32)

        # Monte Carlo Forecast
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

    return float(np.mean(rmse_scores)) if rmse_scores else 999.0


def run_grid_search(cache_dir: str = "./cache/"):
    print("==================================================================")
    print("      HYPERPARAMETER TUNING: OPTIMIZING ALPHA AND BETA            ")
    print("==================================================================")

    # 1. Fetch Elos ONCE at the top before all loops (<1ms execution)
    print("[*] Fetching up-to-date ClubElo ratings once...")
    elos = get_club_elos(cache_dir=cache_dir)

    # 2. Load Global Priors ONCE before all loops (<1ms execution)
    global_N = pd.read_csv(
        os.path.join(cache_dir, "global_priors", "global_N_matrix.csv")
    ).to_numpy(dtype=np.float64)
    global_T = pd.read_csv(
        os.path.join(cache_dir, "global_priors", "global_T_vector.csv")
    ).to_numpy(dtype=np.float64)

    # 3. Sample Validation Fixtures with closing odds
    val_fixtures = [
        {
            "home_team": "Bournemouth",
            "away_team": "Leeds United",
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
            "home_team": "Manchester United",
            "away_team": "Brentford",
            "bookie_odds": [2.05, 3.94, 3.93],
        },
        {
            "home_team": "Aston Villa",
            "away_team": "Tottenham",
            "bookie_odds": [2.35, 3.81, 3.17],
        },
    ]

    # 4. Correct Fractional Alpha Candidates & Beta Candidates
    alpha_candidates = [0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    beta_candidates = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010, 0.0011, 0.0012]

    best_rmse = float("inf")
    best_alpha = None
    best_beta = None

    total_combinations = len(alpha_candidates) * len(beta_candidates)
    evaluated = 0

    print(
        f"[*] Grid search across {total_combinations} candidate (alpha, beta) pairs..."
    )
    print("------------------------------------------------------------------")

    for alpha in alpha_candidates:
        for beta in beta_candidates:
            evaluated += 1
            avg_rmse = evaluate_hyperparameter_pair(
                alpha=alpha,
                beta=beta,
                val_fixtures=val_fixtures,
                elos=elos,
                global_N=global_N,
                global_T=global_T,
                cache_dir=cache_dir,
                num_simulations=2000,
            )

            print(
                f"({evaluated:2d}/{total_combinations}) Alpha: {alpha:<6.3f} | Beta: {beta:<7.4f} --> Mean Market RMSE: {avg_rmse:.4f}"
            )

            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_alpha = alpha
                best_beta = beta

    print("\n==================================================================")
    print("      OPTIMIZATION COMPLETE: BEST HYPERPARAMETERS FOUND           ")
    print("==================================================================")
    print(f"BEST ALPHA: {best_alpha}")
    print(f"BEST BETA:  {best_beta}")
    print(f"MINIMUM MARKET RMSE: {best_rmse:.4f}")
    print("==================================================================\n")


if __name__ == "__main__":
    run_grid_search()
