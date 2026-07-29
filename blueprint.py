from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from constants import BEST_ALPHA, BEST_BETA
from util import (
    calculate_global_q,
    calculate_specific_q,
    create_full_team_df,
    neutralise_global_prior,
)
from helpers import standardise_possessions, align_team_perspective

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class FeatureStrategy(ABC):
    """
    The contract that all matrix feature modifiers must follow.
    """

    @abstractmethod
    def apply(self, matrix: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        pass


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


class MatrixPipeline:
    """
    Orchestrates the Bayesian prior and passes it through an assembly line of modular strategies.
    """

    def __init__(self, strategies: list):
        self.strategies = strategies

    def build_grid(self, global_q: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        home_team, home_id = ctx["home_team"], ctx["home_id"]
        away_team, away_id = ctx["away_team"], ctx["away_id"]

        alpha = ctx["alpha"]

        # Symmetrize prior to strip out administrative seeding bias
        neutral_q = neutralise_global_prior(global_q)

        full_home_df = standardise_possessions(create_full_team_df(home_team))
        full_away_df = standardise_possessions(create_full_team_df(away_team))

        aligned_home_df = align_team_perspective(full_home_df, home_id, sim_role="H")
        aligned_away_df = align_team_perspective(full_away_df, away_id, sim_role="A")

        home_counts, _ = calculate_global_q(aligned_home_df)
        away_counts, _ = calculate_global_q(aligned_away_df)

        # Conjugate update uses the dynamic loop alpha
        home_q_matrix, _ = calculate_specific_q(neutral_q, alpha, home_counts)
        away_q_matrix, _ = calculate_specific_q(neutral_q, alpha, away_counts)

        home_attacking_rows = home_q_matrix[
            home_q_matrix["starting_state"].str.endswith("H")
        ]
        away_attacking_rows = away_q_matrix[
            away_q_matrix["starting_state"].str.endswith("A")
        ]

        combined_matrix = pd.concat([home_attacking_rows, away_attacking_rows])

        # Assembly line passes ctx down. EloModifier will read ctx["beta"] natively.
        for strategy in self.strategies:
            combined_matrix = strategy.apply(combined_matrix, ctx)

        final_q_grid = combined_matrix.pivot(
            index="starting_state",
            columns="finishing_state",
            values="updated_lambda_ij",
        ).fillna(0)

        return final_q_grid

    def build_grid_fast(self, neutral_q: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        """
        High-speed execution method that assumes data loading and prior
        neutralization have already occurred outside the primary loop.
        """
        alpha = ctx["alpha"]
        home_counts = ctx["home_counts"]
        away_counts = ctx["away_counts"]

        # Conjugate updating using the active loop alpha candidate
        home_q_matrix, _ = calculate_specific_q(neutral_q, alpha, home_counts)
        away_q_matrix, _ = calculate_specific_q(neutral_q, alpha, away_counts)

        home_attacking_rows = home_q_matrix[
            home_q_matrix["starting_state"].str.endswith("H")
        ]
        away_attacking_rows = away_q_matrix[
            away_q_matrix["starting_state"].str.endswith("A")
        ]

        combined_matrix = pd.concat([home_attacking_rows, away_attacking_rows])

        # Apply modifiers (EloModifier intercepts ctx["beta"])
        for strategy in self.strategies:
            combined_matrix = strategy.apply(combined_matrix, ctx)

        # Pivot to final intensity grid structure
        final_q_grid = combined_matrix.pivot(
            index="starting_state",
            columns="finishing_state",
            values="updated_lambda_ij",
        ).fillna(0)

        return final_q_grid
