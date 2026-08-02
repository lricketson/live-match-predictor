import os
import time
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from config.constants import CLUB_ID_MAP
from src.ctmc.ctmc_builder import calculate_specific_q
from src.engine.live_scraper import LiveEventScraper
from src.engine.vectoriser import TacticalVectoriser
from src.engine.knn_indexer import TacticalKNNIndexer
from src.engine.bayesian_decay import BayesianDecayEngine
from src.engine.live_simulator import run_live_pytorch_monte_carlo
from src.market.alpha_engine import AlphaEngine


class LiveMatchPredictorSession:
    """
    Persistent stateful session for an active live match.
    Pre-loads priors and 90 KNN VRAM/RAM slices once at initialization;
    ingests raw Opta events via LiveEventScraper and executes sub-20ms simulation ticks.
    """

    def __init__(
        self,
        home_team: str,
        away_team: str,
        cache_dir: str = "./cache/",
        db_dir: str = "./compiled_db/",
        alpha: float = 50.0,
        min_ev_threshold: float = 0.02,
        kelly_fraction: float = 0.25,
        k_neighbours: int = 50,
    ):
        if home_team not in CLUB_ID_MAP:
            raise ValueError(f"[-] Home team '{home_team}' not found in CLUB_ID_MAP.")
        if away_team not in CLUB_ID_MAP:
            raise ValueError(f"[-] Away team '{away_team}' not found in CLUB_ID_MAP.")
        self.home_team = home_team
        self.away_team = away_team
        self.home_id = CLUB_ID_MAP[home_team]
        self.away_id = CLUB_ID_MAP[away_team]
        self.db_dir = db_dir
        # 1. Instantiate persistent LiveEventScraper for real-time RAM state ledgers
        self.scraper = LiveEventScraper(
            home_team_id=self.home_id, away_team_id=self.away_id
        )
        # 2. Instant Load of Pre-Cached CSV Priors (<1ms)
        global_N = self._load_csv_matrix(
            os.path.join(cache_dir, "global_priors", "global_N_matrix.csv")
        )
        global_T = self._load_csv_matrix(
            os.path.join(cache_dir, "global_priors", "global_T_vector.csv")
        )
        h_clean = home_team.replace(" ", "_").lower()
        a_clean = away_team.replace(" ", "_").lower()
        home_N = self._load_csv_matrix(
            os.path.join(cache_dir, f"{h_clean}_N_matrix.csv")
        )
        home_T = self._load_csv_matrix(
            os.path.join(cache_dir, f"{h_clean}_T_vector.csv")
        )
        away_N = self._load_csv_matrix(
            os.path.join(cache_dir, f"{a_clean}_N_matrix.csv")
        )
        away_T = self._load_csv_matrix(
            os.path.join(cache_dir, f"{a_clean}_T_vector.csv")
        )
        # 3. Compute Q_pre ONCE (<1ms)
        _, _, Q_pre_np = calculate_specific_q(
            global_N, global_T, home_N, home_T, away_N, away_T, alpha
        )
        self.Q_pre = torch.tensor(Q_pre_np, dtype=torch.float32)
        # 4. Instantiate Persistent Engines
        self.decay_engine = BayesianDecayEngine(historical_baseline=self.Q_pre)
        self.vectoriser = TacticalVectoriser()
        self.alpha_engine = AlphaEngine(
            min_ev_threshold=min_ev_threshold, kelly_fraction=kelly_fraction
        )
        # 5. Pre-load 90 KNN Database Slices into Memory/VRAM ONCE
        self.knn_indexer = TacticalKNNIndexer(k_neighbours=k_neighbours)
        self._pre_load_knn_slices()

    @staticmethod
    def _load_csv_matrix(filepath: str) -> np.ndarray:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"[-] Critical Error: Cached matrix file '{filepath}' not found."
            )
        return pd.read_csv(filepath).to_numpy(dtype=np.float64)

    def _pre_load_knn_slices(self):
        loaded_count = 0
        for minute in range(1, 91):
            path = os.path.join(self.db_dir, f"slice_min_{minute}.pt")
            if os.path.exists(path):
                slice_data = torch.load(path, weights_only=False)
                self.knn_indexer.register_historical_slice(
                    minute_timestamp=minute,
                    vectors_matrix=slice_data["vectors_normalised"],
                    n_future_matrix=slice_data["n_future"],
                    T_future_matrix=slice_data["T_future"],
                )
                loaded_count += 1
        if loaded_count == 0:
            raise FileNotFoundError(
                f"[-] Critical Error: No .pt slices found in '{self.db_dir}'."
            )

    def process_opta_event(
        self,
        event_packet: Dict[str, Any],
        bookie_odds: List[float],
        num_simulations: int = 10000,
    ) -> Optional[Dict[str, Any]]:
        """
        Ingests a single raw Opta streaming packet, updates LiveEventScraper, and if it's a valid touch event, executes
        a full prediction and alpha detection tick.
        """

        # 1. update LiveEventScraper's real-time RAM ledgers (n_live, T_live, clock, scoreboard)
        is_touch = self.scraper.process_event(event_packet)
        if not is_touch:
            return None  # skip non-touch events like substitutions

        # export clean engine payload snapshot from LiveEventScraper
        payload = self.scraper.export_engine_payload()
        payload["lambda_live"] = self.scraper.get_live_transition_rates()

        return self._execute_prediction_tick(payload, bookie_odds, num_simulations)

    def process_stream_chunk(
        self,
        event_packets: List[Dict[str, Any]],
        bookie_odds: List[float],
        num_simulations: int = 10000,
    ) -> Optional[Dict[str, Any]]:
        """
        Ingests a batch/chunk of incoming raw Opta event packets via LiveEventScraper.
        """
        state_updated = self.scraper.ingest_stream_chunk(event_packets)
        if not state_updated:
            return None

        payload = self.scraper.export_engine_payload()
        payload["lambda_live"] = self.scraper.get_live_transition_rates()

        return self._execute_prediction_tick(payload, bookie_odds, num_simulations)

    def process_payload(
        self,
        payload: Dict[str, Any],
        bookie_odds: List[float],
        num_simulations: int = 10000,
    ) -> Dict[str, Any]:
        """
        Direct execution for pre-constructed payload dictionaries.
        """
        if "lambda_live" not in payload:
            n_live = payload["n_live"]
            T_live = payload["T_live"]
            payload["lambda_live"] = n_live / (T_live.unsqueeze(1) + 1e-6)

        return self._execute_prediction_tick(payload, bookie_odds, num_simulations)
