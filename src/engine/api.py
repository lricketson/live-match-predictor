import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your existing persistent orchestration engine
from live_session import LiveMatchPredictorSession

# Global reference to hold the session in memory so we don't
# constantly re-load the 90 KNN slices into VRAM
active_session = None

app = FastAPI(title="PremLive Predictor API")

# Allow React (typically running on localhost:3000) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    home_team: str
    away_team: str


@app.post("/api/start-match")
async def start_match(req: MatchRequest):
    """
    Called when you click a fixture in the React UI.
    Instantiates the engine and loads the data into the RTX 4070 Ti.
    """
    global active_session

    # Initialize the session. The 90 KNN slices and CSV priors are loaded here.
    active_session = LiveMatchPredictorSession(
        home_team=req.home_team, away_team=req.away_team
    )

    return {
        "status": "success",
        "message": f"Loaded {req.home_team} vs {req.away_team} into 12GB VRAM",
    }


@app.get("/api/fixtures")
async def get_fixtures():
    """
    Returns upcoming fixtures for the UI grid.
    In the future, hook this up to the API-Football endpoint.
    """
    return [
        {"id": 1, "home": "Arsenal", "away": "Tottenham Hotspur", "kickoff": "10 mins"},
        {"id": 2, "home": "Manchester City", "away": "Chelsea", "kickoff": "2 hours"},
    ]


@app.websocket("/ws/match-stream")
async def match_stream(websocket: WebSocket):
    """
    The bidirectional WebSocket route that streams predictions to React.
    """
    await websocket.accept()
    global active_session

    if not active_session:
        await websocket.close(code=1008, reason="No active match session initialized.")
        return

    # In-memory paper trading ledger for version 1.0
    paper_trading_ledger = []

    # Buffer to hold live Opta events before sending them to the GPU
    event_buffer = []

    try:
        # This loop simulates an asynchronous listener for the live Opta feed.
        # You will replace the mock generation with your actual streaming consumer.
        while True:
            # 1. Simulate receiving a new Opta event packet
            mock_opta_event = {"type": "pass", "x": 50, "y": 50}
            event_buffer.append(mock_opta_event)

            # 2. Wait until we have a batch (e.g., 30 seconds worth of events)
            if len(event_buffer) >= 15:

                # Mock live bookmaker odds (Home, Draw, Away)
                current_bookie_odds = [2.10, 3.40, 3.60]

                # 3. Process the chunk to update the state and run Monte Carlo sims
                result = active_session.process_stream_chunk(
                    event_packets=event_buffer, bookie_odds=current_bookie_odds
                )

                if result:
                    # Append any detected +EV arbitrage opportunities to the ledger
                    if result.get("signals"):
                        paper_trading_ledger.extend(result["signals"])

                    # Attach the ledger and aggregated stats to the frontend payload
                    result["ledger"] = paper_trading_ledger

                    # (Optional) If LiveEventScraper tracks possession/xG, inject it here:
                    # result["stats"] = active_session.scraper.get_aggregated_stats()

                    # 4. Push the probabilities, market RMSE, and ledger to React
                    await websocket.send_json(result)

                # Clear the buffer for the next time window
                event_buffer.clear()

            # Yield control back to the event loop (simulating time between real events)
            await asyncio.sleep(2)

    except WebSocketDisconnect:
        print("React client disconnected from the match stream.")
