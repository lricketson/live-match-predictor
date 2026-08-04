from dataclasses import dataclass
from typing import Dict, List
from src.market.devig import devig_power_law
from src.market.metrics import calculate_market_rmse


@dataclass
class AlphaSignal:
    timestamp_sec: float
    outcome: str  # "HOME", "DRAW", or "AWAY"
    bookie_odds: float
    model_prob: float
    market_prob: float
    ev_percent: float
    kelly_stake_percent: float


class AlphaEngine:
    """
    Ingests live CTMC Monte Carlo probabilities and streaming bookmaker odds,
    computes market alignment metrics like RMSE, and flags +EV bets (arbitrage opportunities).
    """

    def __init__(self, min_ev_threshold: float = 0.02, kelly_fraction: float = 0.25):
        """
        min_ev_threshold: Minimum EV required to make a bet (e.g. 0.02 means at least +2.0% edge)
        kelly_fraction: Fractional Kelly sizing (e.g. 0.25 = Quarter Kelly)
        """
        self.min_ev = min_ev_threshold
        self.kelly_fraction = kelly_fraction
        self.outcomes = ["HOME", "DRAW", "AWAY"]

    def evaluate(
        self, clock_seconds: float, model_probs: List[float], bookie_odds: List[float]
    ) -> Dict[str, any]:
        """
        Evaluates the current state of the match for market odds alignment and +EV signals.
        model_probs: [p_home, p_draw, p_away] as calculated by the model
        bookie_odds: [o_home, o_draw, o_away] as listed on bookmakers' websites
        """
        devigged_probs = devig_power_law(bookie_odds)
        rmse = calculate_market_rmse(model_probs, bookie_odds)

        # identify +EV opportunities
        signals: List[AlphaSignal] = []

        for idx, outcome in enumerate(self.outcomes):
            p_model = model_probs[idx]
            p_market = devigged_probs[idx]
            odds = bookie_odds[idx]

            ev = (p_model * odds) - 1.0

            if ev > self.min_ev:
                full_kelly = ev / (odds - 1.0)
                recommended_stake = max(0.0, self.kelly_fraction * full_kelly)

                signals.append(
                    AlphaSignal(
                        clock_seconds,
                        outcome,
                        odds,
                        round(p_model, 4),
                        round(p_market, 4),
                        round(ev * 100.0, 2),
                        round(recommended_stake * 100.0, 2),
                    )
                )
        return {
            "market_rmse": round(rmse, 4),
            "signals": signals,
            "devigged_market_probs": [round(p, 4) for p in devigged_probs],
        }
