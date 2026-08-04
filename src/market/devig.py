import numpy as np
from scipy.optimize import root_scalar
from typing import List


def devig_multiplicative(odds: List[float]) -> List[float]:
    """
    Basic proportional devigging. It's an inferior method so used only as a fallback.
    """
    implied = [1 / odd for odd in odds]
    margin = sum(implied)
    return [p / margin for p in implied]


def devig_power_law(odds: List[float]) -> List[float]:
    """
    Power law devigging. Solves for k in sum(1/(O_i)^k) = 1.0.
    Is realistically skewed towards assigning vig to longshots than favourites.
    """
    implied = np.array([1 / odd for odd in odds], dtype=np.float64)

    def objective(k):
        return np.sum(np.power(implied, k)) - 1.0

    try:
        # solve for k in interval [1.0, 3.0]
        res = root_scalar(objective, bracket=[1.0, 3.0], method="brentq")
        k_opt = res.root
        true_probs = np.power(implied, k_opt)
        return true_probs.tolist()
    except Exception:
        # fallback to multiplicative if root finder fails
        return devig_multiplicative(odds)


def devig_shin(odds: List[float]) -> List[float]:
    """
    Shin's method for 3-outcome football market (Home, Draw, Away).
    Models the insider trading proportion parameter z to extract true probabilities.
    """
    implied = np.array([1 / odd for odd in odds], dtype=np.float64)
    S = np.sum(implied)

    if len(odds) != 3:
        return devig_power_law(odds)

    def objective(z):
        p_i = (np.sqrt(z**2 + 4 * (1 - z) * (implied**2) / S) - z) / (2 * (1 - z))
        return np.sum(p_i)

    try:
        res = root_scalar(objective, bracket=[1e-5, 0.4], method="brentq")
        z_opt = res.root
        true_probs = (
            np.sqrt(z_opt**2 + 4 * (1 - z_opt) * (implied**2) / S) - z_opt
        ) / (2 * (1 - z_opt))
        return (true_probs / np.sum(true_probs)).tolist()
    except Exception:
        return devig_power_law(odds)
