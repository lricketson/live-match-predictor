import pandas as pd
from config.constants import CLUB_ID_MAP


def club_to_elo(club: str):
    return "placeholder"


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
