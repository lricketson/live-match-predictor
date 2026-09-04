import pandas as pd
from scripts.backtest import simulate_historical_match

# 1. Specify match details and path
match_path = "data/premier_league_26_27/match_1983549_Everton_vs_Crystal_Palace.json"
home_team = "Everton"
away_team = "Crystal Palace"

# 2. Provide closing/reference bookmaker odds: [Home, Draw, Away]
closing_odds = [2.33, 3.39, 3.12]

# 3. Run the live stream backtest simulation
signals_df = simulate_historical_match(
    match_json_path=match_path,
    home_team=home_team,
    away_team=away_team,
    closing_odds=closing_odds,
)

# 4. View identified +EV betting opportunities
print(f"\n[+] Total +EV Signals Generated: {len(signals_df)}")
if not signals_df.empty:
    print(signals_df.head(10).to_string(index=False))
