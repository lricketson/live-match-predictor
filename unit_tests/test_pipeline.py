import torch
from live_scraper import LiveEventScraper
from vectoriser import TacticalVectoriser
from knn_indexer import TacticalKNNIndexer
from bayesian_decay import BayesianDecayEngine
from live_simulator import run_live_pytorch_monte_carlo


def run_pipeline_test():
    print("[*] Starting End-to-End Mathematical Verification...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Simulate Live Scraper with fake events
    scraper = LiveEventScraper(home_team_id=100, away_team_id=200)
    fake_events = [
        {
            "isTouch": True,
            "expandedMinute": 5,
            "second": 10,
            "x": 50.0,
            "teamId": 100,
            "outcomeType": {"value": 1},
        },
        {
            "isTouch": True,
            "expandedMinute": 5,
            "second": 25,
            "x": 85.0,
            "teamId": 100,
            "outcomeType": {"value": 1},
        },
        {
            "isTouch": True,
            "expandedMinute": 5,
            "second": 30,
            "x": 95.0,
            "teamId": 100,
            "isGoal": True,
            "outcomeType": {"value": 1},
        },
    ]
    scraper.ingest_stream_chunk(fake_events)
    payload = scraper.export_engine_payload()

    assert payload["clock_seconds"] == 330.0, "Clock scraping failed!"
    assert payload["scoreboard"][0] == 1, "Goal detection failed!"

    # 2. Test Tactical Vectoriser
    vectoriser = TacticalVectoriser()
    norm_vec = vectoriser.vectorise(payload)
    assert norm_vec.shape == (5,), f"Expected vector shape (5,), got {norm_vec.shape}"

    # 3. Test K-NN Indexer with a mock database slice
    indexer = TacticalKNNIndexer(k_neighbours=5)
    mock_vecs = torch.randn(20, 5, device=device)
    mock_n = torch.abs(torch.randn(20, 12, 12, device=device))
    mock_T = torch.abs(torch.randn(20, 12, device=device)) + 1.0
    indexer.register_historical_slice(10, mock_vecs, mock_n, mock_T)

    lambda_knn, dists, idxs = indexer.get_pseudo_prior(norm_vec, clock_seconds=600.0)
    assert lambda_knn.shape == (
        12,
        12,
    ), f"Expected K-NN matrix (12, 12), got {lambda_knn.shape}"

    # 4. Test Bayesian Decay Engine
    decay_engine = BayesianDecayEngine()
    lambda_live = scraper.get_live_transition_rates()
    q_blended = decay_engine.blend(
        lambda_live, payload["T_live"], lambda_knn, clock_seconds=330.0
    )

    assert q_blended.shape == (12, 12), "Blended matrix shape mismatch!"
    row_sums = q_blended.sum(dim=1)
    assert torch.allclose(
        row_sums, torch.zeros_like(row_sums), atol=1e-4
    ), "CTMC invalid: Rows do not sum to 0!"

    # 5. Test Live GPU Monte Carlo Simulator
    prob_h, prob_d, prob_a = run_live_pytorch_monte_carlo(
        q_matrix=q_blended,
        current_clock=payload["clock_seconds"],
        current_state_idx=payload["active_ball_state_idx"],
        live_home_goals=payload["scoreboard"][0],
        live_away_goals=payload["scoreboard"][1],
        num_simulations=1000,
        verbose=False,
    )
    total_prob = prob_h + prob_d + prob_a
    assert (
        abs(total_prob - 1.0) < 0.01
    ), f"Probabilities do not sum to 100%! Got {total_prob}"

    print("[+] All 5 pipeline verification assertions PASSED cleanly!")


if __name__ == "__main__":
    run_pipeline_test()
