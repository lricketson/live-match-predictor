from src.market.alpha_engine import AlphaEngine

engine = AlphaEngine(min_ev_threshold=0.02, kelly_fraction=0.25)

# Example: 60th minute in-play scenario
# Model predicts Home Win 45%, Draw 30%, Away Win 25%
model_probs = [0.45, 0.30, 0.25]

# Bookie quotes: Home 2.40, Draw 3.30, Away 3.20
bookie_odds = [2.40, 3.30, 2.70]

result = engine.evaluate(
    clock_seconds=3600.0, model_probs=model_probs, bookie_odds=bookie_odds
)

print("Market RMSE:", result["market_rmse"])
print("Devigged Probs:", result["devigged_market_probs"])
for sig in result["signals"]:
    print(
        f"[+EV SIGNAL] Outcome: {sig.outcome} | Odds: {sig.bookie_odds} | EV: {sig.ev_percent}% | Stake: {sig.kelly_stake_percent}%"
    )
