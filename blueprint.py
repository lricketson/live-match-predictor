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


class FeatureStrategy(ABC):
    """
    The contract that all matrix feature modifiers must follow.
    """

    @abstractmethod
    def apply(self, matrix: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        pass


class EloModifier(FeatureStrategy):
    """
    Applies exponential scaling to on-ball actions based on Elo differentials.
    """

    def apply(self, matrix: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        df = matrix.copy()

        elo_home = ctx["elo_home"]
        elo_away = ctx["elo_away"]
        beta = ctx.get("beta", BEST_BETA)

        df["start_zone"] = df["starting_state"].str[2]
        df["start_poss"] = df["starting_state"].str[-1]
        df["finish_poss"] = df["finishing_state"].str[-1]

        is_goal = df["finishing_state"].str.startswith("Goal")
        df["finish_zone"] = df["finishing_state"].str[2]

        # if home team has the ball, the elo diff is elo_home - elo_away, and vice versa
        active_diff = np.where(
            df["start_poss"] == "H", elo_home - elo_away, elo_away - elo_home
        )

        is_progression = (
            (~is_goal)
            & (df["finish_zone"] > df["start_zone"])
            & (df["start_poss"] == df["finish_poss"])
        )
        is_scoring = (is_goal) & (df["start_poss"] == df["finish_poss"])

        is_positive_action = is_progression | is_scoring

        modifier = np.where(is_positive_action, active_diff, -active_diff)
        df["updated_lambda_ij"] = df["updated_lambda_ij"] * np.exp(beta * modifier)

        return df


class HostAdvantageModifier(FeatureStrategy):
    """
    Applies a clean, mathematically rigorous Home Field Advantage (HFA) boost to tournament
    host nations playing on home soil, operating on top of a neutral prior.
    """

    def apply(self, matrix: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        df = matrix.copy()

        # check if the match really is a true home game for a host nation
        host_nations = {"Canada", "Mexico", "USA"}
        is_host_at_home = ctx.get("is_host_at_home", ctx["home_team"] in host_nations)
        if not is_host_at_home:
            return df
        print(
            f"[*] Host Nation active ({ctx['home_team']}). Applying explicit HFA boost..."
        )

        # define hardcoded HFA boost. in international football it typically adds between 6% and 8% transition intensity.
        hfa_boost = ctx.get("hfa_boost_factor", 0.07)

        # identify positive transitions for the home side
        is_home_action = df["starting_state"].str.endswith("H")
        is_progression = df["finishing_state"].str[2] > df["starting_state"].str[2]
        is_scoring = df["finishing_state"].str.contains("Goal_H")

        is_positive_home = is_home_action & (is_progression | is_scoring)

        df.loc[is_positive_home, "updated_lambda_ij"] *= 1.0 + hfa_boost
        return df


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
