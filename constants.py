BEST_ALPHA = "Not yet found"
BEST_BETA = "Not yet found"

STATE_TO_IDX = {
    "Z:0_P:H": 0,
    "Z:1_P:H": 1,
    "Z:2_P:H": 2,
    "Z:3_P:H": 3,
    "Z:4_P:H": 4,
    "Z:0_P:A": 5,
    "Z:1_P:A": 6,
    "Z:2_P:A": 7,
    "Z:3_P:A": 8,
    "Z:4_P:A": 9,
    "Goal_H": 10,
    "Goal_A": 11,
}


HOME_ATTACK_IDX = [3, 4]  # Z:3_P:H and Z:4_P:H
AWAY_ATTACK_IDX = [8, 9]  # Z:3_P:A and Z:4_P:A
