import json
import torch
import pandas as pd
from tqdm import tqdm
from src.engine.live_session import LiveMatchPredictorSession
from config.constants import (
    BEST_ALPHA,
    BEST_BETA,
    DEFAULT_K,
    MIN_EV_THRESHOLD,
    DEFAULT_KELLY_FRACTION,
)


def simulate_historical_match(
    match_json_path: str,
    home_team: str,
    away_team: str,
    closing_odds: list,
    num_simulations: int = 3000,
    eval_interval_seconds: int = 60,  # Evaluate simulation once per minute
) -> pd.DataFrame:
    """
    Chronologically feeds Opta JSON events into LiveMatchPredictorSession.
    Updates CTMC state continuously, but triggers heavy Monte Carlo evaluations
    at a controlled interval to eliminate duplicate signals and run in seconds.
    """
    cuda_available = torch.cuda.is_available()
    device_name = (
        torch.cuda.get_device_name(0) if cuda_available else "CPU (CUDA not detected)"
    )
    print(f"\n=======================================================")
    print(f"[*] Engine Hardware: {device_name}")
    print(f"[*] Simulating: {home_team} vs {away_team}")
    print(
        f"[*] Evaluation Interval: Every {eval_interval_seconds}s | Sims: {num_simulations}"
    )
    print(f"=======================================================\n")

    session = LiveMatchPredictorSession(
        home_team=home_team,
        away_team=away_team,
        alpha=BEST_ALPHA,
        beta=BEST_BETA,
        k_neighbours=DEFAULT_K,
        min_ev_threshold=MIN_EV_THRESHOLD,
        kelly_fraction=DEFAULT_KELLY_FRACTION,
    )

    with open(match_json_path, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    events_list = match_data.get("events", [])
    ev_ledger = []

    last_eval_time = -999.0
    last_score = "0-0"

    pbar = tqdm(events_list, desc="Simulating Match", unit="event")

    for event in pbar:
        if not event.get("isTouch", False):
            continue

        # Ingest touch event into scraper to keep live state & transition counts accurate
        session.scraper.process_event(event)
        current_clock = session.scraper.current_clock
        current_score = f"{session.scraper.scoreboard[0].item()}-{session.scraper.scoreboard[1].item()}"

        # Trigger Monte Carlo simulation if interval elapsed OR if a goal was scored
        is_goal_event = current_score != last_score
        time_elapsed = (current_clock - last_eval_time) >= eval_interval_seconds

        if time_elapsed or is_goal_event:
            last_eval_time = current_clock
            last_score = current_score

            payload = session.scraper.export_engine_payload()
            payload["lambda_live"] = session.scraper.get_live_transition_rates()

            tick_result = session._execute_prediction_tick(
                payload=payload,
                bookie_odds=closing_odds,
                num_simulations=num_simulations,
            )

            pbar.set_postfix(
                {
                    "Min": f"{tick_result['minute']}'",
                    "Score": tick_result["score"],
                    "+EV Signals": len(ev_ledger),
                }
            )

            if tick_result.get("signals"):
                for signal in tick_result["signals"]:
                    ev_ledger.append(
                        {
                            "minute": tick_result["minute"],
                            "clock_seconds": signal.timestamp_sec,
                            "outcome": signal.outcome,
                            "bookie_odds": signal.bookie_odds,
                            "model_prob": signal.model_prob,
                            "market_prob": signal.market_prob,
                            "edge_percent": signal.ev_percent,
                            "kelly_stake_percent": signal.kelly_stake_percent,
                        }
                    )

    return pd.DataFrame(ev_ledger)
