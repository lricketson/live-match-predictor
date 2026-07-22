import unittest
import torch
from live_scraper import LiveEventScraper
from constants import STATE_TO_IDX
from helpers import apply_stoppage_cap, resolve_goal_state, get_spatial_zone


class TestLiveEventScraper(unittest.TestCase):
    def setUp(self):
        """
        Runs before every single test. Initializes a fresh scraper with mock Team IDs:
        Home Team ID = 100 (e.g., Arsenal)
        Away Team ID = 200 (e.g., Chelsea)
        """
        self.home_id = 100
        self.away_id = 200
        self.scraper = LiveEventScraper(
            home_team_id=self.home_id,
            away_team_id=self.away_id,
            total_match_seconds=5400.0,
        )

    def test_initial_pinned_state(self):
        """
        Verifies that memory allocation complies with our low-latency hardware specs:
        Tensors must reside in Pinned CPU RAM when available, and CPU otherwise, and initialise
        at Home Kickoff (Index 2).
        """
        expected_pinned = torch.cuda.is_available()
        self.assertEqual(
            self.scraper.n_live.is_pinned(),
            expected_pinned,
            f"n_live must be in {expected_pinned}!",
        )
        self.assertEqual(
            self.scraper.T_live.is_pinned(),
            expected_pinned,
            f"T_live must be in {expected_pinned}!",
        )
        self.assertEqual(
            self.scraper.scoreboard.is_pinned(),
            expected_pinned,
            f"Scoreboard must be in {expected_pinned}!",
        )

        self.assertEqual(self.scraper.current_state_idx, STATE_TO_IDX["Z:2_P:H"])
        self.assertEqual(self.scraper.scoreboard.tolist(), [0, 0])

    def test_non_touch_events_ignored(self):
        """
        Events like yellow cards or VAR checks (isTouch=False) should be ignored
        and must not advance the match clock or mutate transition ledgers.
        """
        card_event = {
            "eventId": 1,
            "expandedMinute": 5,
            "second": 0,
            "teamId": self.home_id,
            "isTouch": False,
            "type": {"displayName": "Card"},
        }

        updated = self.scraper.process_event(card_event)
        self.assertFalse(updated)
        self.assertEqual(self.scraper.current_clock, 0.0)
        self.assertEqual(self.scraper.n_live.sum().item(), 0.0)

    def test_stoppage_time_capping(self):
        """
        CRITICAL QUANT TEST: If a player is injured for 60 seconds between touches,
        apply_stoppage_cap() must cap the holding time delta (T_live) at 3.0 seconds
        to prevent artificial dead-time from skewing our CTMC exit rates.
        """
        # Event 1: Normal pass at 00:10 (10 seconds elapsed from kickoff)
        event_1 = {
            "eventId": 1,
            "expandedMinute": 0,
            "second": 10,
            "teamId": self.home_id,
            "isTouch": True,
            "x": 50.0,
            "outcomeType": {"value": 1},  # Successful
        }
        self.scraper.process_event(event_1)

        # At kickoff (00:00 to 00:10), ball was in Z:2_P:H. 10s elapsed.
        kickoff_idx = STATE_TO_IDX["Z:2_P:H"]
        self.assertEqual(self.scraper.T_live[kickoff_idx].item(), 10.0)

        # Event 2: Next touch happens at 01:10 (60 seconds later due to injury/stoppage)
        event_2 = {
            "eventId": 2,
            "expandedMinute": 1,
            "second": 10,
            "teamId": self.home_id,
            "isTouch": True,
            "x": 70.0,
            "outcomeType": {"value": 1},
        }
        self.scraper.process_event(event_2)

        # The raw delta is 60.0s, but our rule caps anything > 15.0s down to 3.0s!
        zone_2_idx = STATE_TO_IDX["Z:2_P:H"]
        # Notice: T_live for Z:2_P:H should only increase by 3.0s, NOT 60.0s!
        expected_time = 10.0 + 3.0
        self.assertAlmostEqual(
            self.scraper.T_live[zone_2_idx].item(), expected_time, places=4
        )

    def test_possession_turnover_logic(self):
        """
        An unsuccessful pass (outcomeType.value == 0) by the Home team
        must immediately credit possession to the Away team in that spatial zone.
        """
        turnover_event = {
            "eventId": 1,
            "expandedMinute": 2,
            "second": 0,
            "teamId": self.home_id,
            "isTouch": True,
            "x": 80.0,  # Zone 3
            "outcomeType": {"value": 0},  # Unsuccessful -> Turnover!
        }
        self.scraper.process_event(turnover_event)

        # Since Home lost the ball in Zone 3, the finishing state must be Z:3_P:A
        expected_idx = STATE_TO_IDX["Z:3_P:A"]
        self.assertEqual(self.scraper.current_state_idx, expected_idx)
        self.assertEqual(self.scraper.current_state_str, "Z:3_P:A")

    def test_own_goal_resolution(self):
        """
        If an Away defender (teamId == away_id) touches the ball and scores an own goal,
        the engine MUST credit the goal to Goal_H and increment Home's scoreboard.
        """
        own_goal_event = {
            "eventId": 10,
            "expandedMinute": 15,
            "second": 30,
            "teamId": self.away_id,  # Away player acting
            "isTouch": True,
            "isGoal": True,
            "isOwnGoal": True,
            "x": 5.0,
        }
        self.scraper.process_event(own_goal_event)

        self.assertEqual(self.scraper.current_state_str, "Goal_H")
        self.assertEqual(self.scraper.scoreboard.tolist(), [1, 0])  # [Home, Away]

    def test_in_place_memory_reset(self):
        """
        Verifies that calling .reset() wipes the data clean between matches
        WITHOUT re-allocating memory or losing our Pinned RAM hardware lock.
        """
        # Populate ledgers with arbitrary data
        self.scraper.n_live += 5.0
        self.scraper.scoreboard[0] = 3

        # Grab physical memory address of the tensor before reset
        old_memory_address = self.scraper.n_live.data_ptr()

        # Wipe clean
        self.scraper.reset()

        # Assert data is zeroed out
        self.assertEqual(self.scraper.n_live.sum().item(), 0.0)
        self.assertEqual(self.scraper.scoreboard.tolist(), [0, 0])
        self.assertEqual(self.scraper.current_clock, 0.0)

        # Assert physical memory address did NOT change (zero OS garbage collection!)
        new_memory_address = self.scraper.n_live.data_ptr()
        self.assertEqual(
            old_memory_address,
            new_memory_address,
            "Memory address changed! In-place .zero_() failed.",
        )


if __name__ == "__main__":
    unittest.main()
