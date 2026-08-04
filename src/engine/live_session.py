import os
import time
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from config.constants import CLUB_ID_MAP, DEFAULT_ELO, BEST_ALPHA, BEST_BETA
from src.engine.live_scraper import LiveEventScraper
from src.engine.vectoriser import TacticalVectoriser
from src.engine.knn_indexer import TacticalKNNIndexer
from src.engine.bayesian_decay import BayesianDecayEngine
from src.engine.live_simulator import run_live_pytorch_monte_carlo
from src.market.alpha_engine import AlphaEngine
from helpers import get_club_elos
from src.strategies.elo_strategy import EloModifier
from src.strategies.matrix_pipeline import MatrixPipeline


class LiveMatchPredictorSession:
    """
    Persistent stateful orchestration engine for an active live match.
    Pre-loads priors and 90 KNN VRAM/RAM slices once at initialization;
    ingests raw Opta events via LiveEventScraper and executes sub-20ms simulation ticks.
    """

    def __init__(
        self,
        home_team: str,
        away_team: str,
        cache_dir: str = "./cache/",
        db_dir: str = "./compiled_db/",
        alpha: float = BEST_ALPHA,
        beta: float = BEST_BETA,
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

        elos = get_club_elos(cache_dir=cache_dir)

        # 3. Fetch up-to-date club Elo ratings
        ctx = {
            "home_team": home_team,
            "away_team": away_team,
            "elo_home": float(elos.get(home_team, DEFAULT_ELO)),
            "elo_away": float(elos.get(away_team, DEFAULT_ELO)),
            "alpha": alpha,
            "beta": beta,
        }

        # 4. Compute Q_pre with MatrixPipeline (Bayesian conjugate updating + Elo scaling)
        pipeline = MatrixPipeline(strategies=[EloModifier()])
        Q_pre_np = pipeline.build_grid_fast(
            global_N, global_T, home_N, home_T, away_N, away_T, ctx
        )
        # turn it into a PyTorch tensor
        self.Q_pre = torch.tensor(Q_pre_np, dtype=torch.float32)

        # 5. Instantiate Persistent Prediction and Market Engines
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

    # ------------------------------------------------------------------------
    #                         PUBLIC ENTRY POINTS
    # ------------------------------------------------------------------------

    def process_opta_event(
        self,
        event_packet: Dict[str, Any],
        bookie_odds: List[float],
        num_simulations: int = 10000,
    ) -> Optional[Dict[str, Any]]:
        """
        Ingests a single raw Opta streaming packet, updates LiveEventScraper, and if it's a valid touch event, executes
        a full prediction and alpha detection tick.
        This method is used for a single Opta ball event, so it's probably the one I'll use in production.
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
        Since, in live matches, ball events will be coming in one by one, this is less likely to be used in
        production. But running Monte Carlo simulations every 3-4 events instead of every single event will
        save GPU computation.
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
        This method assumes the data is already coming in nicely parsed into a payload dictionary, so it
        completely bypasses LiveEventScraper. This method can be used for backtesting, but not production.
        """
        if "lambda_live" not in payload:
            n_live = payload["n_live"]
            T_live = payload["T_live"]
            payload["lambda_live"] = n_live / (T_live.unsqueeze(1) + 1e-6)

        return self._execute_prediction_tick(payload, bookie_odds, num_simulations)

    def _execute_prediction_tick(
        self, payload: Dict[str, Any], bookie_odds: List[float], num_simulations: int
    ) -> Dict[str, Any]:
        """
        Executes the 6-stage predictive pipeline in under 20ms.
        1. LiveEventScraper parses and packages live data into clean payload, and keeps an up-to-date
            Q_live matrix by holding n_live and T_live in pinned RAM

        2. TacticalVectoriser calculates v0, ... v4 and forms the 5D until-minute-t match vector,
            then normalises it with historical mu and sigma values

        3. TacticalKNNIndexer calculates k nearest neighbour match feature vectors with torch.cdist()
            and aggregates minute-t-to-90 counts and holding times from the neighbours to create Q_KNN

        4. BayesianDecayEngine calculates Q_active from a dynamically weighted sum of Q_pre, Q_KNN,
            and Q_live

        5. Q_active and the current scoreboard/timestamp/ball state are fed into run_live_pytorch_monte_carlo,
           which calculates the current match outcome probabilities

        6. The AlphaEngine devigs current bookmaker odds with the power law method, calculates market RMSE,
           and flags +EV bet signals
        """
        t0 = time.time()
        clock_sec = float(payload["clock_seconds"])
        minute_bucket = max(1, min(90, int(round(clock_sec / 60.0))))

        # step 1: update minute-bucket Z-score normalisation parameters
        slice_path = os.path.join(self.db_dir, f"slice_min_{minute_bucket}.pt")
        slice_data = torch.load(slice_path, weights_only=False)
        self.vectoriser.set_normalisation_params(slice_data["mu"], slice_data["sigma"])

        # step 2: vectorise 5D live state and call it z_live
        z_live = self.vectoriser.vectorise(payload)

        # step 3: GPU K-NN trajectory query -> Q_KNN
        lambda_knn, _, _ = self.knn_indexer.get_pseudo_prior(
            live_vector=z_live, clock_seconds=clock_sec
        )

        # step 4: tri-modal blending -> Q_active
        Q_active = self.decay_engine.blend(
            lambda_live=payload["lambda_live"],
            T_live=payload["T_live"],
            lambda_knn=lambda_knn,
            clock_seconds=clock_sec,
        )

        # step 5: PyTorch GPU Monte Carlo simulation -> outcome probabilities
        scoreboard = payload["scoreboard"]
        home_g = int(scoreboard[0].item())
        away_g = int(scoreboard[1].item())
        active_state_idx = payload.get(
            "active_ball_state_idx", payload.get("current_state_idx", 2)
        )

        prob_h, prob_d, prob_a = run_live_pytorch_monte_carlo(
            q_matrix=Q_active,
            current_clock=clock_sec,
            current_state_idx=active_state_idx,
            live_home_goals=home_g,
            live_away_goals=away_g,
            num_simulations=num_simulations,
        )

        # step 6: devigging and +EV arbitrage discovery
        market_res = self.alpha_engine.evaluate(
            clock_seconds=clock_sec,
            model_probs=[prob_h, prob_d, prob_a],
            bookie_odds=bookie_odds,
        )

        latency_ms = (time.time() - t0) * 1000.0

        return {
            "clock_seconds": clock_sec,
            "minute": minute_bucket,
            "score": f"{home_g}-{away_g}",
            "home_goals": home_g,
            "away_goals": away_g,
            "probs": {"home": prob_h, "draw": prob_d, "away": prob_a},
            "market_rmse": market_res["market_rmse"],
            "signals": market_res["signals"],
            "devigged_market_probs": market_res["devigged_market_probs"],
            "latency_ms": latency_ms,
        }

    def reset(self):
        """Resets the scraper for a new match."""
        self.scraper.reset()
