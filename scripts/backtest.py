import json
import pandas as pd
from src.engine.live_session import LiveMatchPredictorSession
from config.constants import (
    BEST_ALPHA,
    BEST_BETA,
    BEST_GAMMA,
    DEFAULT_K,
    MIN_EV_THRESHOLD,
    DEFAULT_KELLY_FRACTION,
)


def simulate_historical_match(
    match_json_path: str, home_team: str, away_team: str, closing_odds: list
) -> pd.DataFrame:
    """
    Chronologically feeds Opta JSON events into the LiveMatchPredictorSession to extract +EV signals generated
    during the match.
    """

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

    print(f"[*] Simulating {home_team} vs {away_team}...")

    for event in events_list:
        tick_result = session.process_opta_event(
            event_packet=event, bookie_odds=closing_odds, num_simulations=10000
        )

        if tick_result and tick_result.get("signals"):
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
