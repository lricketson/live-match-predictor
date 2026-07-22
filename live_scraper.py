from typing import Dict, List
import torch
from helpers import (
    apply_stoppage_cap,
    get_spatial_zone,
    resolve_goal_state,
    resolve_possession,
)
from constants import STATE_TO_IDX


class LiveEventScraper:
    """
    Ingests live match event streams, maintains real-time CTMC state ledgers in RAM, and exports clean
    snapshots for the Tri-Modal Bayesian Decay engine.
    """

    def __init__(
        self, home_team_id: int, away_team_id: int, total_match_seconds: float = 5400.0
    ):
        self.home_id = home_team_id
        self.away_id = away_team_id
        self.total_match_seconds = total_match_seconds

        # real-time state trackers
        self.current_clock: float = 0.0  # elapsed match time in seconds
        self.last_event_time: float = 0.0  # timestamp of the previous event

        self.use_pinned = torch.cuda.is_available()

        self.scoreboard = torch.zeros(2, dtype=torch.long, pin_memory=self.use_pinned)

        # initialise ball at home kickoff  (Z:2_P:H, which is Index 2)
        self.current_state_idx: int = STATE_TO_IDX["Z:2_P:H"]
        self.current_state_str: str = "Z:2_P:H"

        # continuous-time maths ledgers (in RAM)
        # n_live is a 12x12 matrix tracking exact transition counts from state i to j
        self.n_live = torch.zeros(
            (12, 12), dtype=torch.float32, pin_memory=self.use_pinned
        )

        # T_live is a 12-element vector tracking total cumulative seconds spent in state i
        self.T_live = torch.zeros(12, dtype=torch.float32, pin_memory=self.use_pinned)

    def process_event(self, event_packet: Dict[str, any]):
        """
        Takes in a single event and updates the scoreboard and the clock.
        """
        # skip non-touch events like cards
        if not event_packet.get("isTouch", False):
            return
        event_time = event_packet["expandedMinute"] * 60 + event_packet["second"]
        raw_delta = event_time - self.last_event_time
        delta_t = apply_stoppage_cap(raw_delta)

        is_goal = event_packet.get("isGoal", False) is True
        is_own = event_packet.get("isOwnGoal", False) is True
        is_home = event_packet["teamId"] == self.home_id

        goal_state = resolve_goal_state(is_goal, is_own, is_home)

        if goal_state:
            finishing_state_str = goal_state
            # update pinned scoreboard tensor directly
            if goal_state == "Goal_H":
                self.scoreboard[0] += 1
            else:
                self.scoreboard[1] += 1

        else:
            # not a goal
            end_x = event_packet.get("endX", event_packet["x"])
            zone = get_spatial_zone(end_x)
            outcome_val = event_packet.get("outcomeType", {}).get("value", 1)
            possession = resolve_possession(
                event_packet["teamId"], self.home_id, outcome_val
            )

            finishing_state_str = f"Z:{zone}_P:{possession}"

        next_state_idx = STATE_TO_IDX[finishing_state_str]

        self.n_live[self.current_state_idx, next_state_idx] += 1.0
        self.T_live[self.current_state_idx] += delta_t

        self.current_clock = event_time
        self.last_event_time = event_time
        self.current_state_idx = next_state_idx
        self.current_state_str = finishing_state_str

        return True

    def ingest_stream_chunk(self, events: List[Dict[str, any]]) -> bool:
        """
        Takes in a list/chunk of incoming event packets. Returns True if at least one
        touch event updated the live ledger.
        """
        state_updated = False
        for packet in events:
            if self.process_event(packet):
                state_updated = True
        return state_updated

    def export_engine_payload(self) -> Dict[str, any]:
        """
        Packages the active live state into a lightweight dictionary ready for the next stages. 
        All tensors remain in pinned CPU RAM ready for non-blocking GPU Direct Memory Access (DMA)
        transfer (.to(non_blocking=True)).
        """
        remaining_seconds = max(0.0, self.total_match_seconds - self.current_clock)
        payload = {
            "clock_seconds": self.current_clock,
            "remaining_seconds": remaining_seconds,
            "scoreboard": self.scoreboard,  # Pinned CPU Tensor (shape: [2])
            "active_ball_state_idx": self.current_state_idx,
            "n_live": self.n_live,  # Pinned CPU Tensor (shape: [12, 12])
            "T_live": self.T_live,  # Pinned CPU Tensor (shape: [12])
        }
        return payload

    def get_live_transition_rates(self, epsilon: float = 1e-6) -> torch.Tensor:
        """
        Helper: Computes live transition rates lambda_live = n_live / (T_live + epsilon).
        """
        return self.n_live / (self.T_live.unsqueeze(1) + epsilon)

    def reset(self):
        """
        Wipes ledgers clean between fixtures or half-times without triggering OS memory garbage collection.
        Uses PyTorch in-place .zero_() to preserve physical pinned RAM addresses.
        """
        self.current_clock = 0.0
        self.last_event_time = 0.0
        self.current_state_idx = STATE_TO_IDX["Z:2_P:H"]
        self.current_state_str = "Z:2_P:H"

        # calling .zero_() instead of torch.zeros() keeps the direct memory "highway" to the GPU open instead
        # of forcing PyTorch to ask the motherboard for a brand new physical memory block.
        self.scoreboard.zero_()
        self.n_live.zero_()
        self.T_live.zero_()
