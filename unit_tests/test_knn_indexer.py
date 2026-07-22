import unittest
import torch
import time
from knn_indexer import TacticalKNNIndexer


class TestTacticalKNNIndexer(unittest.TestCase):
    def setUp(self):
        self.indexer = TacticalKNNIndexer(k_neighbours=50)
        self.mock_minute = 30
        self.num_historical_matches = 10000  # Spec requirement: M >= 10,000 matches

        # 1. Mock 10,000 historical feature vectors: Shape (10000, 4)
        mock_vectors = torch.randn(
            (self.num_historical_matches, 4), dtype=torch.float32
        )

        # 2. Mock 10,000 future transition count matrices: Shape (10000, 12, 12)
        # Using positive random numbers to simulate valid event counts
        mock_n_future = (
            torch.rand((self.num_historical_matches, 12, 12), dtype=torch.float32)
            * 10.0
        )

        # 3. Mock 10,000 future holding time vectors: Shape (10000, 12)
        mock_T_future = (
            torch.rand((self.num_historical_matches, 12), dtype=torch.float32) * 100.0
        )

        # Register the complete historical database slice
        self.indexer.register_historical_slice(
            self.mock_minute, mock_vectors, mock_n_future, mock_T_future
        )

    def test_output_shapes_and_types(self):
        mock_live_vector = torch.randn(4, dtype=torch.float32)
        distances, indices, bucket = self.indexer.find_nearest_neighbours(
            mock_live_vector, 1800.0
        )

        self.assertEqual(distances.shape, (50,))
        self.assertEqual(indices.shape, (50,))
        self.assertEqual(bucket, 30)

    def test_exact_match_zero_distance(self):
        exact_match_vector = self.indexer.db[self.mock_minute][4242].clone()
        distances, indices, _ = self.indexer.find_nearest_neighbours(
            exact_match_vector, 1800.0
        )

        self.assertEqual(indices[0].item(), 4242)
        self.assertAlmostEqual(distances[0].item(), 0.0, places=5)

    def test_pseudo_prior_construction_and_speed(self):
        """
        CRITICAL STAGE 2 HARDWARE TEST: Verifies that finding neighbors AND aggregating
        50 future matrices into Q_KNN completes well under our 45ms latency budget.
        """
        mock_live_vector = torch.randn(4, dtype=torch.float32)

        start_time = time.perf_counter()

        # Execute the full Stage 2 master method
        lambda_knn, distances, indices = self.indexer.get_pseudo_prior(
            mock_live_vector, 1800.0
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        print(
            f"\n[+] Full Stage 2 Pipeline Latency (Query + Matrix Aggregation): {elapsed_ms:.3f} ms"
        )

        # Verify output matrix dimensions and mathematical validity
        self.assertEqual(lambda_knn.shape, (12, 12))
        self.assertTrue(
            (lambda_knn >= 0.0).all(), "Transition intensities cannot be negative!"
        )

        # Assert end-to-end execution takes less than 45.0 milliseconds
        self.assertLess(
            elapsed_ms,
            45.0,
            f"Stage 2 Latency {elapsed_ms:.2f}ms exceeded 45ms specification budget!",
        )


if __name__ == "__main__":
    unittest.main()
