import torch
from typing import Dict, Any, Optional
from config.constants import HOME_ATTACK_IDX, AWAY_ATTACK_IDX

HOME_STATES_IDX = [0, 1, 2, 3, 4]
AWAY_STATES_IDX = [5, 6, 7, 8, 9]


def compute_raw_5d_features(
    n_matrix: torch.Tensor,
    T_vector: torch.Tensor,
    score_diff: torch.Tensor,
    epsilon: float = 1e-6,
):
    """
    5D feature extraction logic used by both TacticalVectoriser and the build_historical_db script.
    Returns a 1D PyTorch tensor: [Field Tilt, Home Prog, Away Prog, Match Tempo, Goal Diff].
    """

    device = n_matrix.device

    # 1. Field Tilt (Home share of Zone 4/5 attacking touches)
    home_att = n_matrix[HOME_ATTACK_IDX, :].sum()
    away_att = n_matrix[AWAY_ATTACK_IDX, :].sum()
    total_att = home_att + away_att

    v0_tilt = (
        home_att / total_att if total_att > 0 else torch.tensor(0.50, device=device)
    )

    # 2. Progression Ratios (Forward zone transitions / Total team transitions)
    def calc_prog_ratio(team_indices: list) -> torch.Tensor:
        progressions = torch.tensor(0.0, device=device)
        total_transitions = torch.tensor(0.0, device=device)
        for i in range(len(team_indices)):
            s_idx = team_indices[i]
            total_transitions += n_matrix[s_idx, :].sum()
            for j in range(i + 1, len(team_indices)):
                f_idx = team_indices[j]
                progressions += n_matrix[s_idx, f_idx]
        return (
            progressions / total_transitions
            if total_transitions > 0
            else torch.tensor(0.0, device=device)
        )

    v1_home_prog = calc_prog_ratio(HOME_STATES_IDX)
    v2_away_prog = calc_prog_ratio(AWAY_STATES_IDX)

    # 3. Match Tempo (Markovian departure rate sum across transient states 0..9)
    active_departures = n_matrix[:10, :].sum(dim=1)
    active_T = T_vector[:10] + epsilon
    v3_tempo = (active_departures / active_T).sum()

    # 4. Goal Differential
    v4_score_diff = score_diff.float()

    return torch.stack([v0_tilt, v1_home_prog, v2_away_prog, v3_tempo, v4_score_diff])


class TacticalVectoriser:
    """
    Transforms raw live CTMC ledgers from LiveEventScraper into normalised 5D vectors
    ready for native PyTorch Euclidean distance calculations (torch.cdist).
    """

    def __init__(
        self,
        historical_means: Optional[torch.Tensor] = None,
        historical_stds: Optional[torch.Tensor] = None,
    ):
        """
        Initialises standardisation parameters. In the final version, these will be loaded from the
        pre-compiled historical match database (.pt slices).
        """

        self.use_pinned = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_pinned else "cpu")

        # 5D base statistics: [Field Tilt, Home Prog, Away Prog, Match Tempo, Goal Diff]
        default_means = torch.tensor([0.50, 0.30, 0.30, 35.0, 0.0], dtype=torch.float32)
        default_stds = torch.tensor([0.15, 0.10, 0.10, 12.0, 1.2], dtype=torch.float32)

        self.mu = (
            historical_means.to(self.device)
            if historical_means is not None
            else default_means.to(self.device)
        )

        self.sigma = (
            historical_stds.to(self.device)
            if historical_stds is not None
            else default_stds.to(self.device)
        )

    def set_normalisation_params(self, mu: torch.Tensor, sigma: torch.Tensor):
        """
        Dynamically updates mu and sigma when switching minute-bucket .pt slices.
        """
        self.mu = mu.to(self.device)
        self.sigma = sigma.to(self.device)

    def vectorise(self, payload: Dict[str, Any], epsilon: float = 1e-6) -> torch.Tensor:
        """
        Ingests the dictionary exported by LiveEventScraper.export_engine_payload()
        and outputs a normalised 1D PyTorch tensor of shape (5,).
        """

        n_live = payload["n_live"].to(self.device, non_blocking=True)
        T_live = payload["T_live"].to(self.device, non_blocking=True)
        scoreboard = payload["scoreboard"].to(self.device, non_blocking=True)
        score_diff = scoreboard[0] - scoreboard[1]

        raw_tensor = compute_raw_5d_features(n_live, T_live, score_diff, epsilon)

        # standardise against historical baseline
        normalised_tensor = (raw_tensor - self.mu) / (self.sigma + epsilon)

        return normalised_tensor
