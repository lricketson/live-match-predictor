BEST_ALPHA = "Not yet found"
BEST_BETA = "Not yet found"


CLUB_ID_MAP = {
    "Arsenal": 13,
    "Aston Villa": 24,
    "Bournemouth": 183,
    "Brentford": 189,
    "Brighton": 211,
    "Burnley": 184,
    "Chelsea": 15,
    "Crystal Palace": 162,
    "Everton": 31,
    "Fulham": 170,
    "Ipswich": 165,
    "Leeds": 19,
    "Leicester": 14,
    "Liverpool": 26,
    "Luton": 95,
    "Man City": 167,
    "Man Utd": 32,
    "Newcastle": 23,
    "Nottingham Forest": 174,
    "Sheff Utd": 163,
    "Southampton": 18,
    "Sunderland": 16,
    "Tottenham": 30,
    "West Ham": 29,
    "Wolves": 161,
}

CLUB_NAME_RESOLVER = {
    # "The-Odds-API Name": "Understat Name"
    # Arsenal all good
    "Villa": "Aston Villa",  # just in case
    # Bournemouth all good
    # Brentford all good
    "Brighton and Hove Albion": "Brighton",
    # Burnley all good
    # Chelsea all good
    # Crystal Palace all good
    # Everton all good
    # Fulham all good
    "Leeds United": "Leeds",
    # Liverpool all good
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Manchester Utd": "Manchester United",
    "United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nottm Forest": "Nottingham Forest",
    # Sunderland all good
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolves": "Wolverhampton Wanderers",
    # ----- other teams -----
    "Leicester City": "Leicester",  # To keep it robust for next season
}

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
