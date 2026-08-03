change paths from /world_cup to /data

replace team_id_map with club_id_map and club elos

    - find if there is a reliable measure for club power ratings

add an entity resolution dictionary for clubs with &, -, etc. in their names

re-run hyperparameters tuning logic

choose new default normalisation parameters in tacticalvectoriser

investigate why there are only 5570 epl matches instead of 380 \* 15 = 5700

run A/B tests to see which of shin's method and power law devigging does a better job.

investigate why there are matches called 'home_vs_away.json', like match_1549627_Home_vs_Away.json

use inverse square distance scaling in knn
fine tune k
add a red card modifier strategy

investigate why laptop has 5570 matches and desktop has 5432
