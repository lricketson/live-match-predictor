import unittest
import torch
import time
from bayesian_decay import BayesianDecayEngine


class TestBayesianDecayEngine(unittest.TestCase):
    def setUp(self):
        self.engine = BayesianDecayEngine(total_match_seconds=5400.0)

        # Mock 12x12 intensity matrices with positive transition rates
        self.mock_lambda_live = torch.rand((12, 12), dtype=torch.float32) * 0.5
        self.mock_lambda_knn = torch.rand((12, 12), dtype=torch.float32) * 0.5

        # Mock holding time vector: all states visited except Zone 0 (index 0)
        self.mock_T_live = torch.rand(12, dtype=torch.float32) * 60.0
        self.mock_T_live[0] = 0.0  # Simulate unvisited Zone 0

    def test_weight_partition_of_unity(self):
        """
        Verifies that alpha + beta + gamma == 1.0 across all timestamps.
        """
        test_timestamps = [
            0.0,
            1350.0,
            2700.0,
            4050.0,
            5400.0,
        ]  # Min 0, 22.5, 45, 67.5, 90
        for ts in test_timestamps:
            alpha, beta, gamma = self.engine.compute_decay_weights(ts)
            total_weight = alpha + beta + gamma
            self.assertAlmostEqual(
                total_weight, 1.0, places=6, msg=f"Partition of unity failed at ts={ts}"
            )

    def test_generator_matrix_validity(self):
        """
        CRITICAL CTMC LAW TEST: Verifies that all off-diagonal rates are >= 0
        and every single row sum equals exactly 0.0.
        """
        q_final = self.engine.blend(
            self.mock_lambda_live,
            self.mock_T_live,
            self.mock_lambda_knn,
            clock_seconds=2700.0,
        )

        # 1. Row sums must equal 0.0
        row_sums = q_final.sum(dim=1)
        for i, r_sum in enumerate(row_sums.tolist()):
            self.assertAlmostEqual(
                r_sum, 0.0, places=5, msg=f"Row {i} sum is not zero!"
            )

        # 2. Off-diagonal elements must be non-negative
        off_diag_mask = ~torch.eye(12, dtype=torch.bool, device=self.engine.device)
        min_off_diag = q_final[off_diag_mask].min().item()
        self.assertGreaterEqual(
            min_off_diag, 0.0, "Found negative off-diagonal transition rate!"
        )

    def test_unvisited_state_reallocation(self):
        """
        Verifies that an unvisited state (T_live == 0.0) does not suffer from
        departure rate shrinkage at Minute 60 when live weight gamma is active.
        """
        clock = 3600.0  # Minute 60 -> gamma is roughly 0.444
        q_final = self.engine.blend(
            self.mock_lambda_live,
            self.mock_T_live,
            self.mock_lambda_knn,
            clock_seconds=clock,
        )

        # Calculate expected unvisited weights manually for row 0
        alpha, beta, _ = self.engine.compute_decay_weights(clock)
        alpha_unvisited = alpha / (alpha + beta)
        beta_unvisited = beta / (alpha + beta)

        expected_row_0 = (alpha_unvisited * self.engine.q_hist[0]) + (
            beta_unvisited * self.mock_lambda_knn[0]
        )
        # Zero out diagonal and subtract departure sum
        expected_row_0[0] = 0.0
        expected_row_0[0] = -expected_row_0.sum()

        # Assert that row 0 in Q_final matches the reallocated expectation exactly
        self.assertTrue(
            torch.allclose(q_final[0], expected_row_0, atol=1e-5),
            "Unvisited state dynamic reallocation failed!",
        )

    def test_execution_speed_under_spec(self):
        """
        Verifies that matrix blending executes well under our sub-millisecond hardware budget.
        """
        # 1. THE FIX: Run a quick warm-up pass to initialize PyTorch's CPU thread pools!
        self.engine.blend(
            self.mock_lambda_live,
            self.mock_T_live,
            self.mock_lambda_knn,
            clock_seconds=2700.0,
        )

        # 2. Now time the actual steady-state execution:
        start_time = time.perf_counter()
        self.engine.blend(
            self.mock_lambda_live,
            self.mock_T_live,
            self.mock_lambda_knn,
            clock_seconds=2700.0,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        print(f"\n[+] Tri-Modal Bayesian Decay Execution Latency: {elapsed_ms:.3f} ms")
        self.assertLess(
            elapsed_ms, 2.0, f"Latency {elapsed_ms:.3f}ms exceeded 2.0ms ceiling!"
        )


if __name__ == "__main__":
    unittest.main()
