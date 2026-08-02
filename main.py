import time
import torch
import os
from config.constants import CLUB_ID_MAP
from src.ctmc.ctmc_builder import (
    create_master_df,
    build_global_matrices,
    create_full_team_df,
    standardise_possessions,
    align_team_perspective,
    calculate_specific_q,
)
from src.engine.vectoriser import TacticalVectoriser
from src.engine.bayesian_decay import BayesianDecayEngine
from src.engine.knn_indexer import TacticalKNNIndexer
from src.engine.live_scraper import LiveEventScraper
from src.engine.live_simulator import run_live_pytorch_monte_carlo
from src.market.alpha_engine import AlphaEngine
from typing import Optional, List, Dict

# Global cache for database slices to avoid re-loading from disk on every live tick
_GLOBAL_KNN_INDEXER: Optional[TacticalKNNIndexer] = None


def load_knn_indexer(
    db_dir: str = "./compiled_db/", k_neighbours: int = 50
) -> TacticalKNNIndexer:
    """Pre-loads all 90 minute-bucket slices into memory once."""

    global _GLOBAL_KNN_INDEXER
    if _GLOBAL_KNN_INDEXER is not None:
        return _GLOBAL_KNN_INDEXER

    print(f"[*] Pre-loading 90 database slices from '{db_dir}'...")

    indexer = TacticalKNNIndexer(k_neighbours=k_neighbours)
    for minute in range(1, 91):
        slice_path = os.path.join(db_dir, f"slice_min_{minute}.pt")

        if os.path.exists(slice_path):
            slice_data = torch.load(slice_path, weights_only=False)

            indexer.register_historical_slice(
                minute_timestamp=minute,
                vectors_matrix=slice_data["vectors_normalised"],
                n_future_matrix=slice_data["n_future"],
                T_future_matrix=slice_data["T_future"],
            )

    _GLOBAL_KNN_INDEXER = indexer
    return indexer


def predict_live_match(
    home_team: str,
    away_team: str,
    clock_seconds: float,
    home_goals: int = 0,
    away_goals: int = 0,
    bookie_odds: List[float] = None,
    db_dir: str = "./compiled_db/",
    alpha: float = 0.05,
    num_simulations: int = 10000,
):
    """
    Live match predictor function.
    """
    if home_team not in CLUB_ID_MAP or away_team not in CLUB_ID_MAP:
        raise ValueError(
            f"[-] Team not recognized. Must be one of: {list(CLUB_ID_MAP.keys())}"
        )
    home_id, away_id = CLUB_ID_MAP[home_team], CLUB_ID_MAP[away_team]

    # load in K-NN database (cached in VRAM or RAM)
    knn_indexer = load_knn_indexer(db_dir=db_dir)

    master_df = create_master_df(folder_path="./data")
    global_N, global_T
