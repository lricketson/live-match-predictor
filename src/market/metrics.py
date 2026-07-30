import numpy as np


def probability_to_odds(probability: float, vig_margin: float = 0.0) -> float:
    """
    Converts a probability to its corresponding odds metric, optionally baking in
    a bookmaker's vig.
    """
    implied_probability = probability * (1 + vig_margin)
    odds = 1 / implied_probability
    return round(odds, 3)


def match_outcome_probs_to_odds(prob_home: float, prob_draw: float, prob_away: float):
    odds_home = probability_to_odds(prob_home)
    odds_draw = probability_to_odds(prob_draw)
    odds_away = probability_to_odds(prob_away)
    return odds_home, odds_draw, odds_away


def calculate_market_rmse(model_probs: list, bookie_odds: list):
    """
    Calculates the RMSE betweeen model probabilities and de-vigged bookie odds.
    model_probs: [prob_h, prob_d, prob_a] (e.g. [0.4, 0.25, 0.35])
    bookie_odds: [odds_h, odds_d, odds_a] (e.g. [2.50, 3.20, 2.70])
    """
    # de-vig the bookie odds to find true market probability
    implied_probs = [1 / o for o in bookie_odds]
    market_margin = sum(implied_probs)
    # this assumes the vig was distributed uniformly, when in reality it often isn't.
    # the vig's distribution is often skewed towards underdogs since bettors love to bet on underdogs
    # hence this is a temporary simplification but it is an opportunity for improvement
    true_market_probs = [ip / market_margin for ip in implied_probs]

    errors = [
        (model_probs[0] - true_market_probs[0]) ** 2,
        (model_probs[1] - true_market_probs[1]) ** 2,
        (model_probs[2] - true_market_probs[2]) ** 2,
    ]

    rmse = np.sqrt(sum(errors) / 3)
    return rmse
