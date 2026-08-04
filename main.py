import time
import os
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import List, Dict, Any

from config.constants import CLUB_ID_MAP
from helpers import get_club_elos
from src.engine.live_session import LiveMatchPredictorSession
from src.engine.live_scraper import LiveEventScraper

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="EPL Live Predictor & Paper Trader",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for dark mode trading dashboard
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e222d; padding: 12px; border-radius: 8px; border: 1px solid #2a2e39; }
    .stButton>button { width: 100%; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 1. SIDEBAR CONFIGURATION & FIXTURE SELECTOR
# -----------------------------------------------------------------------------
st.sidebar.title("⚽ EPL Trading Terminal")
st.sidebar.markdown("---")

upcoming_fixtures = [
    {"home": "Arsenal", "away": "Chelsea", "odds": [1.95, 3.50, 4.00]},
    {"home": "Liverpool", "away": "Man City", "odds": [2.30, 3.40, 3.10]},
    {"home": "Tottenham", "away": "Arsenal", "odds": [3.20, 3.50, 2.20]},
    {"home": "Aston Villa", "away": "Everton", "odds": [1.75, 3.80, 4.80]},
    {"home": "Newcastle", "away": "West Ham", "odds": [1.85, 3.75, 4.20]},
    {"home": "Man Utd", "away": "Brentford", "odds": [2.05, 3.94, 3.93]},
    {"home": "Bournemouth", "away": "Leeds", "bookie_odds": [1.98, 4.00, 4.40]},
]

fixture_labels = [f"{f['home']} vs {f['away']}" for f in upcoming_fixtures]
selected_idx = st.sidebar.selectbox(
    "🎯 Select Fixture to Monitor",
    range(len(fixture_labels)),
    format_func=lambda i: fixture_labels[i],
)
active_fixture = upcoming_fixtures[selected_idx]

st.sidebar.markdown("### ⚙️ Paper Trading Risk Controls")
initial_bankroll = st.sidebar.number_input(
    "Starting Bankroll ($)",
    min_value=100.0,
    max_value=100000.0,
    value=1000.0,
    step=100.0,
)
min_ev_threshold = (
    st.sidebar.slider(
        "Min +EV Edge Threshold (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5
    )
    / 100.0
)
kelly_fraction = st.sidebar.slider(
    "Fractional Kelly Sizing", min_value=0.05, max_value=1.00, value=0.25, step=0.05
)

# Session State Initialization for Paper Trading Bankroll & Logs
if "bankroll" not in st.session_state:
    st.session_state.bankroll = initial_bankroll
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []
if "prob_history" not in st.session_state:
    st.session_state.prob_history = []

if st.sidebar.button("🔄 Reset Paper Trading Account"):
    st.session_state.bankroll = initial_bankroll
    st.session_state.trade_history = []
    st.session_state.prob_history = []
    st.rerun()

# -----------------------------------------------------------------------------
# 2. MAIN DASHBOARD HEADER
# -----------------------------------------------------------------------------
st.title(f"⚽ Match Monitor: {active_fixture['home']} vs {active_fixture['away']}")
st.caption(
    f"Continuous CTMC Simulation & +EV Arbitrage Auto-Trader | Engine Target Latency: < 20ms"
)


# Initialize Live Session Engine
@st.cache_resource
def get_live_session(home_team: str, away_team: str):
    return LiveMatchPredictorSession(
        home_team=home_team,
        away_team=away_team,
        min_ev_threshold=min_ev_threshold,
        kelly_fraction=kelly_fraction,
    )


try:
    session = get_live_session(active_fixture["home"], active_fixture["away"])
except Exception as e:
    st.error(f"[-] Engine Session Init Error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. LIVE MATCH STREAM SIMULATOR CONTROLS
# -----------------------------------------------------------------------------
st.markdown("---")
col_ctrl1, col_ctrl2 = st.columns([1, 3])

with col_ctrl1:
    sim_minute = st.slider(
        "⏱️ Match Clock (Minute)", min_value=0, max_value=90, value=35, step=5
    )

with col_ctrl2:
    home_g = st.number_input(
        f"{active_fixture['home']} Goals", min_value=0, max_value=10, value=1
    )
    away_g = st.number_input(
        f"{active_fixture['away']} Goals", min_value=0, max_value=10, value=0
    )

# Simulate Live Opta Event Payload for current Minute & Score
scraper = LiveEventScraper(
    home_team_id=CLUB_ID_MAP[active_fixture["home"]],
    away_team_id=CLUB_ID_MAP[active_fixture["away"]],
)
clock_seconds = float(sim_minute * 60)
payload = scraper.export_engine_payload()
payload["clock_seconds"] = clock_seconds
payload["scoreboard"] = torch.tensor([home_g, away_g], dtype=torch.long)
payload["lambda_live"] = scraper.get_live_transition_rates()

# Execute Live Engine Tick
live_odds = active_fixture["odds"]
tick_res = session.process_payload(
    payload=payload, bookie_odds=live_odds, num_simulations=2000
)

# Update Time-Series Probability History
st.session_state.prob_history.append(
    {
        "minute": sim_minute,
        "Home": tick_res["probs"]["home"] * 100.0,
        "Draw": tick_res["probs"]["draw"] * 100.0,
        "Away": tick_res["probs"]["away"] * 100.0,
    }
)

# -----------------------------------------------------------------------------
# 4. DASHBOARD METRICS CARDS
# -----------------------------------------------------------------------------
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

col_m1.metric(
    "Score", f"{active_fixture['home']} {home_g} - {away_g} {active_fixture['away']}"
)
col_m2.metric(
    f"{active_fixture['home']} Win %", f"{tick_res['probs']['home']*100:.1f}%"
)
col_m3.metric("Draw %", f"{tick_res['probs']['draw']*100:.1f}%")
col_m4.metric(
    f"{active_fixture['away']} Win %", f"{tick_res['probs']['away']*100:.1f}%"
)
col_m5.metric("Engine Latency", f"{tick_res['latency_ms']:.1f} ms")

# -----------------------------------------------------------------------------
# 5. REAL-TIME PROBABILITY FLUCTUATION CHART (PLOTLY)
# -----------------------------------------------------------------------------
st.markdown("### 📈 In-Play Probability Fluctuation Curve")

prob_df = pd.DataFrame(st.session_state.prob_history).drop_duplicates(subset=["minute"])

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=prob_df["minute"],
        y=prob_df["Home"],
        mode="lines+markers",
        name=f"{active_fixture['home']} Win",
        line=dict(color="#00E676", width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=prob_df["minute"],
        y=prob_df["Draw"],
        mode="lines+markers",
        name="Draw",
        line=dict(color="#FFD600", width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=prob_df["minute"],
        y=prob_df["Away"],
        mode="lines+markers",
        name=f"{active_fixture['away']} Win",
        line=dict(color="#FF1744", width=3),
    )
)

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Match Minute",
    yaxis_title="Probability (%)",
    yaxis_range=[0, 100],
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. PAPER TRADING AUTO-EXECUTION LEDGER
# -----------------------------------------------------------------------------
st.markdown("---")
col_t1, col_t2 = st.columns([2, 1])

with col_t1:
    st.markdown("### 💵 Paper Trading +EV Signal Executions")

    if tick_res["signals"]:
        for sig in tick_res["signals"]:
            stake_dollars = (sig.kelly_stake_pct / 100.0) * st.session_state.bankroll
            st.success(
                f"🔥 **+EV ARBITRAGE SIGNAL DETECTED!** | Selection: **{sig.outcome}** @ **{sig.bookie_odds}** | Edge: **+{sig.ev_percent}%** | Stake: **${stake_dollars:.2f}** ({sig.kelly_stake_pct}%)"
            )

            # Execute Paper Trade automatically if not already logged for this minute
            trade_id = f"{sim_minute}_{sig.outcome}"
            if not any(t["id"] == trade_id for t in st.session_state.trade_history):
                st.session_state.bankroll -= stake_dollars
                st.session_state.trade_history.append(
                    {
                        "id": trade_id,
                        "minute": sim_minute,
                        "outcome": sig.outcome,
                        "odds": sig.bookie_odds,
                        "edge": f"+{sig.ev_percent}%",
                        "stake": round(stake_dollars, 2),
                        "status": "OPEN ⏳",
                    }
                )
    else:
        st.info("ℹ️ No +EV mispricing detected at current bookmaker odds.")

    if st.session_state.trade_history:
        st.table(pd.DataFrame(st.session_state.trade_history).drop(columns=["id"]))

with col_t2:
    st.markdown("### 📊 Bankroll Performance Summary")
    pnl = st.session_state.bankroll - initial_bankroll
    pnl_pct = (pnl / initial_bankroll) * 100.0

    st.metric("Current Bankroll", f"${st.session_state.bankroll:.2f}")
    st.metric("Total PnL ($)", f"${pnl:+.2f}", delta=f"{pnl_pct:+.2f}%")
    st.metric("Market RMSE Consensus", f"{tick_res['market_rmse']:.4f}")
