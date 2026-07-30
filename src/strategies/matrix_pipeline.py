from ctmc.ctmc_builder import (
    standardise_possessions,
    align_team_perspective,
    create_full_team_df,
    build_global_matrices,
    calculate_specific_q,
)
import numpy as np


class MatrixPipeline:
    """
    Orchestrates the Bayesian prior and passes it through an assembly line of modular strategies.
    """

    def __init__(self, strategies: list):
        self.strategies = strategies

    def build_grid(
        self, global_N: np.ndarray, global_T: np.ndarray, ctx: dict
    ) -> np.ndarray:
        home_team, home_id = ctx["home_team"], ctx["home_id"]
        away_team, away_id = ctx["away_team"], ctx["away_id"]
        alpha = ctx["alpha"]

        full_home_df = standardise_possessions(create_full_team_df(home_team))
        full_away_df = standardise_possessions(create_full_team_df(away_team))

        aligned_home_df = align_team_perspective(full_home_df, home_id, sim_role="H")
        aligned_away_df = align_team_perspective(full_away_df, away_id, sim_role="A")

        home_N, home_T, _ = build_global_matrices(aligned_home_df)
        away_N, away_T, _ = build_global_matrices(aligned_away_df)

        _, _, Q_updated = calculate_specific_q(
            global_N, global_T, home_N, home_T, away_N, away_T, alpha
        )
        for strategy in self.strategies:
            Q_updated = strategy.apply(Q_updated, ctx)

        return Q_updated

    def build_grid_fast(
        self,
        global_N: np.ndarray,
        global_T: np.ndarray,
        home_N: np.ndarray,
        home_T: np.ndarray,
        away_N: np.ndarray,
        away_T: np.ndarray,
        ctx: dict,
    ) -> np.ndarray:
        """
        High-speed execution method using pre-cached team N and T values.
        """
        alpha = ctx["alpha"]

        _, _, Q_updated = calculate_specific_q(
            global_N, global_T, home_N, home_T, away_N, away_T, alpha
        )

        for strategy in self.strategies:
            strategy.apply(Q_updated, ctx)

        return Q_updated
