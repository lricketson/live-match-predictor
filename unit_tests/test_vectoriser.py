import unittest
import torch
from live_scraper import LiveEventScraper
from constants import STATE_TO_IDX
from vectoriser import TacticalVectoriser


class TestTacticalvectoriser(unittest.TestCase):
    def setUp(self):
        self.scraper = LiveEventScraper(home_team_id=100, away_team_id=200)
        self.vectoriser = TacticalVectoriser()

    def test_output_shape_and_type(self):
        """
        Verifies that the vectoriser returns a 1D PyTorch Tensor of exact length 5,
        residing on the correct compute device.
        """
        payload = self.scraper.export_engine_payload()
        vector = self.vectoriser.vectorise(payload)

        self.assertIsInstance(vector, torch.Tensor)
        self.assertEqual(vector.shape, (5,))
        self.assertEqual(vector.dtype, torch.float32)

    def test_neutral_prior_fallback(self):
        """
        At Minute 0 (zero touches), Field Tilt should default to 0.50 and Progression Ratio
        should default to 1.00 via Laplace smoothing.
        """
        payload = self.scraper.export_engine_payload()
        vector = self.vectoriser.vectorise(payload)

        # Un-normalise to check raw values
        raw_vector = (vector * self.vectoriser.sigma) + self.vectoriser.mu

        self.assertAlmostEqual(raw_vector[0].item(), 0.50, places=4)  # Field Tilt
        self.assertAlmostEqual(raw_vector[1].item(), 1.00, places=4)  # Prog Ratio
        self.assertAlmostEqual(raw_vector[3].item(), 0.00, places=4)  # Score Diff

    def test_active_match_flow_vectorisation(self):
        """
        Simulates Home team dominating territorial possession in Zone 4 and scoring a goal,
        asserting that Field Tilt and Score Differential shift appropriately.
        """
        # Event 1: Home pass in Zone 4 (Attacking third)
        self.scraper.process_event(
            {
                "eventId": 1,
                "expandedMinute": 10,
                "second": 0,
                "teamId": 100,
                "isTouch": True,
                "x": 90.0,
                "outcomeType": {"value": 1},
            }
        )
        # Event 2: Home Goal!
        self.scraper.process_event(
            {
                "eventId": 2,
                "expandedMinute": 11,
                "second": 0,
                "teamId": 100,
                "isTouch": True,
                "isGoal": True,
                "x": 95.0,
            }
        )

        payload = self.scraper.export_engine_payload()
        vector = self.vectoriser.vectorise(payload)
        raw_vector = (vector * self.vectoriser.sigma) + self.vectoriser.mu

        # 100% of touches occurred in Zone 4 -> Field Tilt must be 1.0
        self.assertAlmostEqual(raw_vector[0].item(), 1.00, places=4)
        # Home scored 1 goal -> Score Diff must be +1.0
        self.assertAlmostEqual(raw_vector[3].item(), 1.00, places=4)


if __name__ == "__main__":
    unittest.main()
