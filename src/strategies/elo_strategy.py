import numpy as np
from base_strategy import FeatureStrategy


def _build_elo_direction_matrix() -> np.ndarray:
    """
    Pre-computes a static 12x12 matrix of direction multipliers relative to (elo_home - elo_away):
        +1: Transition is positive for Home and adverse for Away when elo_home > elo_away
        -1: Transition is adverse for Home and positive for Away when elo_home > elo_away
            0: Transition is neutral or self-loop
    """
    M = np.zeros((12, 12), dtype=np.float64)

    # 1. Home starting states (rows 0...4)
    for i in range(5):
        for j in range(12):
            if i == j:
                continue  # self loop
            if j < 5:  # Same possession (Home progression / regression)
                M[i, j] = 1.0 if j > i else -1.0
            elif 5 <= j < 10:  # Turnover to Away
                M[i, j] = -1.0
            elif j == 10:  # Goal Home (Scoring)
                M[i, j] = 1.0
            elif j == 11:  # Goal Away
                M[i, j] = -1.0

    # 2. Away starting states (rows 5...9)
    for i in range(5, 10):
        start_zone = i - 5
        for j in range(12):
            if i == j:
                continue  # self loop
            if 5 <= j < 10:  # Same possession (Away progression / regression)
                finish_zone = j - 5
                # Note: From elo_home - elo_away perspective, Away progression is -1 multiplier
                M[i, j] = -1.0 if finish_zone > start_zone else 1.0
            elif j < 5:  # Turnover to Home
                M[i, j] = 1.0
            elif j == 11:  # Goal Away (Scoring for Away)
                M[i, j] = -1.0
            elif j == 10:  # Goal Home
                M[i, j] = 1.0
    return M


# pre-compile the static 12x12 elo direction matrix
ELO_DIRECTION_MATRIX = _build_elo_direction_matrix()


class EloModifier(FeatureStrategy):
    """
    Applies log-linear scaling to off-ball and on-ball transition intensities based on
    pre-match Elo differentials, while preserving CTMC zero-sum row validity.
    """

    def apply(self, Q: np.ndarray, ctx: dict) -> np.ndarray:

        elo_home = ctx["elo_home"]
        elo_away = ctx["elo_away"]
        beta = ctx.get("beta", 0.001)  # Default scaling hyperparameter

        elo_diff = elo_home - elo_away

        scaling_factors = np.exp(beta * elo_diff * ELO_DIRECTION_MATRIX)
        Q_updated = Q * scaling_factors

        # repair diagonal
        for i in range(10):
            Q_updated[i, i] = 0.0
            Q_updated[i, i] = -np.sum(Q_updated[i, :])

        return Q_updated
