import pandas as pd
from config.constants import CLUB_ID_MAP
from typing import Any, Dict
import os
import time
import json
import urllib.request
import io


def matchup_to_ctx_dict(
    home: str,
    away: str,
    bookie_odds: list[float],
    elo_df: pd.DataFrame,
    alpha: float,
    beta: float,
) -> dict:
    """
    Accepts alpha and beta as function arguments directly from the optimization loop.
    """
    elo_home = club_to_elo(elo_df, home)
    elo_away = club_to_elo(elo_df, away)

    ctx = {
        "home_team": home,
        "home_id": CLUB_ID_MAP[home],
        "away_team": away,
        "away_id": CLUB_ID_MAP[away],
        "elo_home": elo_home,
        "elo_away": elo_away,
        "alpha": alpha,  # Dynamic assignment
        "beta": beta,  # Dynamic assignment
        "bookie_odds": bookie_odds,
    }
    return ctx


CLUBELO_NAME_MAP = {
    "Arsenal": "Arsenal",
    "AstonVilla": "Aston Villa",
    "Blackburn": "Blackburn",
    "Bolton": "Bolton",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Cardiff": "Cardiff",
    "Chelsea": "Chelsea",
    "CrystalPalace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield",
    "Hull": "Hull",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "ManCity": "Man City",
    "ManUtd": "Man Utd",
    "Middlesbrough": "Middlesbrough",
    "Newcastle": "Newcastle",
    "Norwich": "Norwich",
    "NottmForest": "Nottingham Forest",
    "QPR": "QPR",
    "Reading": "Reading",
    "SheffieldUtd": "Sheff Utd",
    "Southampton": "Southampton",
    "Stoke": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea": "Swansea",
    "Spurs": "Tottenham",
    "WestHam": "West Ham",
    "Watford": "Watford",
    "Wigan": "Wigan",
    "Wolves": "Wolves",
}


def get_club_elos(
    match_date: str = None, cache_dir: str = "./cache/"
) -> Dict[str, float]:
    """
    Returns up-to-date Elo ratings for all clubs.
    If match_date is provided (YYYY-MM-DD), fetches exact historical ratings for that date.
    Otherwise, auto-refreshes today's live ratings if local cache is missing or >24h old.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "club_elos.json")

    # If historical date requested, query ClubElo for that exact date
    if match_date is not None:
        url = f"http://api.clubelo.com/{match_date}"
        return _fetch_from_url(url)

    # Check if local cache exists and was updated within the last 12 hours
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if (time.time() - mtime) < 43200:  # 12 hours in seconds
            with open(cache_path, "r") as f:
                return json.load(f)

    # Cache is stale or missing -> fetch today's live Elo ratings
    url = "http://api.clubelo.com/"
    elos = _fetch_from_url(url)

    # Save to cache
    with open(cache_path, "w") as f:
        json.dump(elos, f, indent=4)

    return elos


def _fetch_from_url(url: str) -> Dict[str, float]:
    try:
        # Pass standard browser User-Agent to prevent HTTP 403 Forbidden
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req) as response:
            csv_content = response.read().decode("utf-8")

        df = pd.read_csv(io.StringIO(csv_content))

        elos = {}
        for _, row in df.iterrows():
            club_name = str(row["Club"])
            if club_name in CLUBELO_NAME_MAP:
                canonical_name = CLUBELO_NAME_MAP[club_name]
                elos[canonical_name] = float(row["Elo"])

        # Default missing teams to 1500.0
        for team in CLUB_ID_MAP.keys():
            if team not in elos:
                elos[team] = 1500.0
        return elos

    except Exception as e:
        print(
            f"[!] Warning: Could not fetch from {url} ({e}). Returning default 1500.0 ratings."
        )
        return {team: 1500.0 for team in CLUB_ID_MAP.keys()}
