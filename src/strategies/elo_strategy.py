import numpy as np
from base_strategy import FeatureStrategy
import pandas as pd


class EloModifier(FeatureStrategy):
    """
    Applies log-linear scaling to off-ball and on-ball transition intensities based on
    pre-match Elo differentials, while preserving CTMC zero-sum row validity.
    """

    def apply(self, matrix: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        df = matrix.copy()

        elo_home = ctx["elo_home"]
        elo_away = ctx["elo_away"]
        beta = ctx.get("beta", 0.001)  # Default scaling hyperparameter

        # 1. Parse spatial zones and possession flags
        # Assumes state string format like 'Z:2_P:H'
        df["start_zone"] = (
            df["starting_state"].str.extract(r"Z:(\d)").fillna(-1).astype(int)
        )
        df["finish_zone"] = (
            df["finishing_state"].str.extract(r"Z:(\d)").fillna(-1).astype(int)
        )

        df["start_poss"] = df["starting_state"].str[-1]
        df["finish_poss"] = df["finishing_state"].str[-1]

        is_goal = df["finishing_state"].str.startswith("Goal")
        is_self_loop = df["starting_state"] == df["finishing_state"]

        # 2. Determine directional Elo differential from the ball-holder's perspective
        active_diff = np.where(
            df["start_poss"] == "H", elo_home - elo_away, elo_away - elo_home
        )

        # 3. Define Transition Classifications
        is_progression = (
            (~is_goal)
            & (df["finish_zone"] > df["start_zone"])
            & (df["start_poss"] == df["finish_poss"])
        )
        is_scoring = is_goal & (df["start_poss"] == df["finish_poss"])

        is_turnover = (~is_goal) & (df["start_poss"] != df["finish_poss"])
        is_regression = (
            (~is_goal)
            & (df["finish_zone"] < df["start_zone"])
            & (df["start_poss"] == df["finish_poss"])
        )

        # 4. Use np.select for a 3-way split: Positive (+1), Adverse (-1), Neutral/Self-Loop (0)
        conditions = [
            (is_progression | is_scoring) & (~is_self_loop),
            (is_turnover | is_regression) & (~is_self_loop),
        ]
        choices = [active_diff, -active_diff]

        modifier = np.select(conditions, choices, default=0.0)

        # 5. Apply Log-Linear Scaling: lambda_final = lambda * exp(beta * modifier)
        df["updated_lambda_ij"] = df["updated_lambda_ij"] * np.exp(beta * modifier)

        # 6. REPAIR DIAGONALS: Ensure sum of off-diagonals equals negative diagonal
        # Zero out self-loops first so we can sum off-diagonals cleanly
        df.loc[is_self_loop, "updated_lambda_ij"] = 0.0

        row_sums = df.groupby("starting_state")["updated_lambda_ij"].transform("sum")
        df.loc[is_self_loop, "updated_lambda_ij"] = -row_sums

        # Clean up temporary parsing columns before returning
        return df.drop(
            columns=["start_zone", "finish_zone", "start_poss", "finish_poss"]
        )
