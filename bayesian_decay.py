import torch
from typing import Optional, Tuple
import math


class BayesianDecayEngine:
    """
    Fuses static historical baselines, K-NN tactical priors and live match ledgers into a
    valid continuous-time Markov chain (CTMC) infinitesimal generator matrix Q_final.
    """

    def __init__(
        self,
        historical_baseline: Optional[torch.Tensor] = None,
        total_match_seconds: float = 5400.0,
        half_life_seconds: float = 2700.0,
    ):
        self.use_pinned = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_pinned else "cpu")
        self.total_seconds = total_match_seconds
        self.half_life = half_life_seconds

        # decay constant
        self.lambda_decay = math.log(2.0) / self.half_life

        # If no offline baseline is loaded, default to a uniform 12x12 transition prior
        # historical_baseline is the overall prior (from all matches), not the k-NN one
        if historical_baseline is not None:
            self.q_hist = historical_baseline.to(
                self.device, dtype=torch.float32, non_blocking=True
            )
        else:
            # create a uniform prior with 0.1 transitions/sec to off-diagonal states
            self.q_hist = torch.full(
                (12, 12), 0.1, dtype=torch.float32, device=self.device
            )
            self.q_hist.fill_diagonal_(0.0)

    def compute_decay_weights(self, clock_seconds: float) -> Tuple[float, float, float]:
        """
        Computes exponential decay weights for time progression w in [0,1], where w
        is t/5400 (the fraction of the match that has so far been played).
        Guarantees the sum of all weights is 1.0 at all timestamps.
        """
        t = min(self.total_seconds, max(0.0, float(clock_seconds)))

        # w_pre is a function of t: e^(-l_d * t)
        w_pre = math.exp(-self.lambda_decay * t)
        progress_ratio = t / self.total_seconds
        w_live = (1.0 - w_pre) * (progress_ratio**1.5)
        w_knn = 1.0 - w_live - w_pre

        return w_pre, w_knn, w_live

    def blend(
        self,
        lambda_live: torch.Tensor,
        T_live: torch.Tensor,
        lambda_knn: torch.Tensor,
        clock_seconds: float,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """
        Executes row-wise Tri-Modal Bayesian blending and enforces CTMC rows-sum-to-zero validity.
        lambda_active(t) = w_pre * lambda_pre + w_KNN * lambda_KNN + w_live * lambda_live.
        Also performs row-wise dynamic reallocation for unvisited states.
        """

        # ensure incoming live/K-NN ledgers are on the active compute device
        l_live = lambda_live.to(device=self.device, non_blocking=True)
        l_knn = lambda_knn.to(device=self.device, non_blocking=True)
        t_live = T_live.to(device=self.device, non_blocking=True)

        w_pre, w_knn, w_live = self.compute_decay_weights(clock_seconds)

        # row-wise dynamic reallocation
        # if a team genuinely hasn't entered a certain state (e.g. Away hasn't entered Home's box)
        # then the exit rate from that state will be 0, and that will artificially reduce the overall
        # weighted sum exit rate, so we fall back to just our prior and pseudo-prior knowledge.
        # if T_live[i] > epsilon, visited_mask[i] is 1.0, otherwise it's 0.0.
        visited_mask = (t_live > epsilon).float().unsqueeze(1)

        prior_sum = w_pre + w_knn + epsilon
        w_pre_unvisited = w_pre / prior_sum
        w_knn_unvisited = w_knn / prior_sum

        # calculate effective weights
        # these are some very clever lines (props to Gemini)
        w_pre_eff = (visited_mask * w_pre) + ((1.0 - visited_mask) * w_pre_unvisited)
        w_knn_eff = (visited_mask * w_knn) + ((1.0 - visited_mask) * w_knn_unvisited)
        w_live_eff = visited_mask * w_live

        # tri-modal matrix blending
        lambda_blended = (
            (w_pre_eff * self.q_hist) + (w_knn_eff * l_knn) + (w_live_eff * l_live)
        )

        # enforce CTMC validity (rows must sum to 0)
        lambda_blended.fill_diagonal_(0.0)

        # sum departure rates across each row: shape (12,)
        row_departure_rates = lambda_blended.sum(dim=1)

        lambda_blended.diagonal().copy_(-row_departure_rates)

        return lambda_blended
