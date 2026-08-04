BEST_ALPHA = 0.0075
BEST_BETA = 1e-9
BEST_GAMMA = 0.255


CLUB_ID_MAP = {
    "Arsenal": 13,
    "Aston Villa": 24,
    "Blackburn": 158,
    "Bolton": 92,
    "Bournemouth": 183,
    "Brentford": 189,
    "Brighton": 211,
    "Burnley": 184,
    "Cardiff": 188,
    "Chelsea": 15,
    "Crystal Palace": 162,
    "Everton": 31,
    "Fulham": 170,
    "Huddersfield": 166,
    "Hull": 214,
    "Ipswich": 165,
    "Leeds": 19,
    "Leicester": 14,
    "Liverpool": 26,
    "Luton": 95,
    "Man City": 167,
    "Man Utd": 32,
    "Middlesbrough": 21,
    "Newcastle": 23,
    "Norwich": 168,
    "Nottingham Forest": 174,
    "QPR": 171,
    "Reading": 94,
    "Sheff Utd": 163,
    "Southampton": 18,
    "Stoke": 96,
    "Sunderland": 16,
    "Swansea": 259,
    "Tottenham": 30,
    "WBA": 175,
    "Watford": 27,
    "West Ham": 29,
    "Wigan": 194,
    "Wolves": 161,
}
CLUB_NAME_RESOLVER = {
    # "The-Odds-API Name": "Understat Name"
    # Arsenal all good
    "Villa": "Aston Villa",  # just in case
    "AFC Bournemouth": "Bournemouth",
    # Brentford all good
    "Brighton and Hove Albion": "Brighton",
    # Burnley all good
    # Chelsea all good
    "Coventry City": "Coventry",
    # Crystal Palace all good
    # Everton all good
    # Fulham all good
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    # Liverpool all good
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Manchester Utd": "Man Utd",
    "United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Nottm Forest": "Nottingham Forest",
    "Forest": "Nottingham Forest",
    "Sheffield United": "Sheff Utd",
    "Sheffield Utd": "Sheff Utd",
    # Sunderland all good
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolves": "Wolverhampton Wanderers",
    # ----- other teams -----
    "Leicester City": "Leicester",  # To keep it robust for next season
}

STATES = [
    "Z:0_P:H",
    "Z:1_P:H",
    "Z:2_P:H",
    "Z:3_P:H",
    "Z:4_P:H",
    "Z:0_P:A",
    "Z:1_P:A",
    "Z:2_P:A",
    "Z:3_P:A",
    "Z:4_P:A",
    "Goal_H",
    "Goal_A",
]

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

DEFAULT_ELO = 1500.0
