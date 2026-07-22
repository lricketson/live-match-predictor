import torch
from typing import Dict, Any, List, Tuple
from constants import HOME_ATTACK_IDX, AWAY_ATTACK_IDX

HOME_STATES_IDX = [0, 1, 2, 3, 4]
AWAY_STATES_IDX = [5, 6, 7, 8, 9]


class TacticalVectoriser:
    """
    Transforms raw live CTMC ledgers from LiveEventScraper into normalised 5D vectors containing
    aggregated match statistics (Field Tilt, Home Progression Ratio, Away Progression Ratio,
    Markovian Match Tempo, Goal Diff) ready for native PyTorch Euclidean distance calculations (torch.cdist).
    """

    def __init__(
        self, historical_means: List[float] = None, historical_stds: List[float] = None
    ):
        """
        Initialises standardisation parameters. In the final version, these will be loaded from the
        pre-compiled historical match database (.pt slices).
        """

        self.use_pinned = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_pinned else "cpu")

        # 5D base statistics: [Field Tilt, Home Prog, Away Prog, Match Tempo, Goal Diff]
        default_means = [0.50, 0.30, 0.30, 35.0, 0.0]
        default_stds = [0.15, 0.10, 0.10, 12.0, 1.2]

        self.mu = torch.tensor(
            historical_means or default_means,
            dtype=torch.float32,
            device=self.device,
            pin_memory=self.use_pinned if self.device.type == "cpu" else False,
        )

        self.sigma = torch.tensor(
            historical_stds or default_stds,
            dtype=torch.float32,
            device=self.device,
            pin_memory=self.use_pinned if self.device.type == "cpu" else False,
        )

    def _count_progressions(
        self, n_matrix: torch.Tensor, team_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A helper function to calculate both forward zone progressions and total team transitions.
        Returns: (progressions, total_transitions)
        """
        progressions = torch.tensor(0.0, device=self.device)
        total_transitions = torch.tensor(0.0, device=self.device)

        for i in range(len(team_indices)):
            start_idx = team_indices[i]
            # Total transitions departing this state
            total_transitions += n_matrix[start_idx, :].sum()
            # Forward progressions within team possession
            for j in range(i + 1, len(team_indices)):
                finish_idx = team_indices[j]
                progressions += n_matrix[start_idx, finish_idx]

        return progressions, total_transitions

    def vectorise(self, payload: Dict[str, Any], epsilon: float = 1e-6) -> torch.Tensor:
        """
        Ingests the dictionary exported by LiveEventScraper.export_engine_payload()
        and outputs a normalised 1D PyTorch tensor of shape (5,).
        """

        n_live = payload["n_live"].to(self.device, non_blocking=True)
        T_live = payload["T_live"].to(self.device, non_blocking=True)
        scoreboard = payload["scoreboard"].to(self.device, non_blocking=True)

        # --- V0: Field Tilt ---
        home_attack_touches = n_live[HOME_ATTACK_IDX, :].sum()
        away_attack_touches = n_live[AWAY_ATTACK_IDX, :].sum()
        total_attack_touches = home_attack_touches + away_attack_touches

        if total_attack_touches > 0:
            v0_tilt = home_attack_touches / total_attack_touches
        else:
            # for kickoff or early minutes with zero deep touches, default to 50-50 split
            v0_tilt = torch.tensor(0.50, device=self.device)

        # --- V1 & V2: Independent Progression Ratios ---
        p_prog_h, total_h = self._count_progressions(n_live, HOME_STATES_IDX)
        p_prog_a, total_a = self._count_progressions(n_live, AWAY_STATES_IDX)

        if total_h > 0:
            v1_home_prog = p_prog_h / total_h
        else:
            v1_home_prog = torch.tensor(0.0, device=self.device)

        if total_a > 0:
            v2_away_prog = p_prog_a / total_a
        else:
            v2_away_prog = torch.tensor(0.0, device=self.device)

        # --- V3: Match Tempo (Continuous-Time Markovian Rate) ---
        # isolate non-terminal starting states (indices 0-9)
        active_n = n_live[:10, :]
        active_T = T_live[:10].unsqueeze(1) + epsilon

        transition_rates = active_n / active_T
        v3_tempo = transition_rates.sum()

        # --- V4: Goal Diff ---
        v4_score_diff = (scoreboard[0] - scoreboard[1]).float()

        # stack into 5D vector and standardise against historical baseline
        raw_vector = torch.stack(
            [v0_tilt, v1_home_prog, v2_away_prog, v3_tempo, v4_score_diff]
        )
        normalised_vector = (raw_vector - self.mu) / (self.sigma + epsilon)

        return normalised_vector
