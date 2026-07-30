import pandas as pd
from ctmc.ctmc_builder import (
    standardise_possessions,
    align_team_perspective,
    create_full_team_df,
    build_global_matrices,
    calculate_specific_q,
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

        full_home_df = standardise_possessions(create_full_team_df(home_team))
        full_away_df = standardise_possessions(create_full_team_df(away_team))

        aligned_home_df = align_team_perspective(full_home_df, home_id, sim_role="H")
        aligned_away_df = align_team_perspective(full_away_df, away_id, sim_role="A")

        home_N, home_T, _ = build_global_matrices(aligned_home_df)
        away_N, away_T, _ = build_global_matrices(aligned_away_df)

        # Conjugate update uses the dynamic loop alpha
        home_q_matrix, _ = calculate_specific_q(home_N, home_T, , alpha, home_counts)
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
