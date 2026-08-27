re-run hyperparameters tuning logic

investigate why there are only 5570 epl matches instead of 380 \* 15 = 5700

run A/B tests to see which of shin's method and power law devigging does a better job.

investigate why there are matches called 'home_vs_away.json', like match_1549627_Home_vs_Away.json

use inverse square distance scaling in knn
fine tune k
add a red card modifier strategy

investigate why laptop has 5570 matches and desktop has 5432

create main.py to easily launch the engine and get really familiar with what i need to do when a match is starting

think about using react instead of streamlit for the frontend

NOTE FOR THESE 2 RESULTS THAT I WAS MISTAKENLY USING PRESENT DAY ELOS FOR APR 2026 PREM MATCHES

current best hyperparams: (need to go further since gamma hit the wall)
BEST GAMMA: 0.12
BEST ALPHA: 0.015
BEST BETA: 1e-05
MINIMUM MARKET RMSE: 0.0966

r2: must take gamma even further

==================================================================
BEST GAMMA: 0.24
BEST ALPHA: 0.01
BEST BETA: 1e-05
MINIMUM MARKET RMSE: 0.0922
==================================================================

now with correct, at the time elo ratings:

==================================================================
BEST GAMMA: 0.27
BEST ALPHA: 0.005
BEST BETA: 1e-05
MINIMUM MARKET RMSE: 0.0915
==================================================================

we have found the ceiling for gamma:

---

[1/5] Gamma: 0.25 --> Best RMSE: 0.0917 (Alpha: 0.01, Beta: 1e-06)  
[2/5] Gamma: 0.27 --> Best RMSE: 0.0915 (Alpha: 0.005, Beta: 1e-06)  
[3/5] Gamma: 0.29 --> Best RMSE: 0.0931 (Alpha: 0.0025, Beta: 1e-06)  
[4/5] Gamma: 0.31 --> Best RMSE: 0.0937 (Alpha: 0.01, Beta: 1e-06)  
[5/5] Gamma: 0.33 --> Best RMSE: 0.0956 (Alpha: 0.005, Beta: 1e-06)

==================================================================
GAMMA TUNING COMPLETE: OVERALL OPTIMAL TRIPLET FOUND  
==================================================================
BEST GAMMA: 0.27
BEST ALPHA: 0.005
BEST BETA: 1e-06
MINIMUM MARKET RMSE: 0.0915
==================================================================

## FINAL HYPERPARAMS

==================================================================
BEST GAMMA: 0.255
BEST ALPHA: 0.0075
BEST BETA: 1e-09
MINIMUM MARKET RMSE: 0.0914
==================================================================

Since I likely won't be able to get a truly live Opta feed, here is what we'll do. If the live Opta feed is delayed by n seconds, we will run our model alongside the match and calculate alpha based on n-seconds-ago bookie odds. This will not make me money but it will still allow me to paper trade and make PnL graphs.
