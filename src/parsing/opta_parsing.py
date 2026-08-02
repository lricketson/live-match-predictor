import pandas as pd
import os
import json
import numpy as np


def get_spatial_zone(x_val: float) -> str:
    """
    Maps an Opta x-coordinate to one of the 5 spatial zones of the pitch.
    Caps edge cases to ensure a valid mapping.
    """
    x_val = max(0.0, min(float(x_val), 100.0))
    if x_val < 17.0:
        zone = "0"
    elif x_val < 39.0:
        zone = "1"
    elif x_val < 61.0:
        zone = "2"
    elif x_val < 83.0:
        zone = "3"
    else:
        zone = "4"
    return zone


def resolve_possession(team_id: int, home_id: int, outcome_val: int) -> str:
    """
    Determines current possession ('H' or 'A'). If outcome_val == 0 (action was unsuccessful),
    then possession flips to the opposing team.
    """
    possession = "H" if team_id == home_id else "A"
    if outcome_val == 0:
        possession = "A" if possession == "H" else "H"
    return possession


def resolve_goal_state(
    is_goal: bool, is_own_goal: bool, is_home_team: bool
) -> str | None:
    """
    Returns terminal absorbing states if a goal was scored, gracefully handling own goals.
    Returns None if the event was not a goal.
    """
    if not is_goal:
        return None
    if is_own_goal:
        return "Goal_A" if is_home_team else "Goal_H"
    return "Goal_H" if is_home_team else "Goal_A"


def apply_stoppage_cap(
    time_delta: float, max_stoppage: float = 15.0, default_fill: float = 2.0
) -> float:
    """
    Caps dead time between events to 15 seconds to preserve tactical realistic-ness.
    """
    if np.isnan(time_delta) or time_delta < 0:
        return default_fill
    return 3.0 if time_delta > max_stoppage else time_delta


vectorised_stoppage_cap = np.vectorize(apply_stoppage_cap)


def parse_match_to_dataframe(
    filepath: str, current_season_year: int = 2026
) -> pd.DataFrame:

    # open file and extract json
    with open(filepath, mode="r", encoding="utf-8") as f:
        match_data = json.load(f)

    # get ids of home and away teams
    home_id = match_data["home"]["teamId"]
    away_id = match_data["away"]["teamId"]

    parent_folder = os.path.basename(os.path.dirname(filepath))

    season_start_year = 2000 + int(parent_folder.split("_")[-2])

    season_delta = current_season_year - season_start_year

    # isolate the events array and load only that into Pandas
    events_list = match_data["events"]

    if not events_list:
        return pd.DataFrame()

    df = pd.DataFrame(events_list)

    # only take the rows where the ball was actually touched
    # this prevents taking events like red cards (which we'll come back to later, but this is a ledger of ball movement)
    if "isTouch" in df.columns:
        df = df[df["isTouch"] == True].copy()

    df["match_seconds"] = df["expandedMinute"] * 60 + df["second"]
    df = df.set_index("match_seconds")
    df = df.sort_index()

    def calculate_state(row, is_start_coordinate=True):
        possession = "H" if row["teamId"] == home_id else "A"

        # if we're looking at event end coordinates:
        if not is_start_coordinate:
            # Opta stores event outcomeTypes as a dict: {'value': 1, 'displayName': 'Successful'}
            # Value == 0 means unsuccessful. So if the event was unsuccessful, it means they lost
            # possession, so we give possession to the other team.
            outcome_val = row.get("outcomeType", {}).get("value")
            if outcome_val == 0:
                possession = "A" if possession == "H" else "H"

        # determine which coordinate to use for the event
        x_val = row["x"] if is_start_coordinate else row.get("endX", row["x"])

        zone = get_spatial_zone(x_val)  # this function handles edge cases cleanly

        state = f"Z:{zone}_P:{possession}"
        return state

    # get the events' starting states
    df["starting_state"] = df.apply(
        lambda row: calculate_state(row, is_start_coordinate=True), axis=1
    )
    # and the finishing states
    df["finishing_state"] = df.apply(
        lambda row: calculate_state(row, is_start_coordinate=False), axis=1
    )

    # fill NaN values
    is_goal = (
        df["isGoal"] == True
        if "isGoal" in df.columns
        else pd.Series(False, index=df.index)
    )
    # this produces a series of every event telling us whether it was an event done by the home team or not
    is_home = df["teamId"] == home_id
    is_own_goal = (
        df["isOwnGoal"] == True
        if "isOwnGoal" in df.columns
        else pd.Series(False, index=df.index)
    )

    # if the event was a goal and the event was done by the home team, it's a home goal
    df.loc[is_goal & is_home & ~is_own_goal, "finishing_state"] = "Goal_H"
    df.loc[is_goal & ~is_home & ~is_own_goal, "finishing_state"] = "Goal_A"

    # own goals logic
    df.loc[is_goal & is_home & is_own_goal, "finishing_state"] = "Goal_A"
    df.loc[is_goal & ~is_home & is_own_goal, "finishing_state"] = "Goal_H"

    # find time spent in the starting state (the difference between this event and the next)
    raw_time_deltas = df.index.to_series().diff().shift(-1)

    # fill the very last event of the match (which has no next event) with the standard 2 seconds
    raw_time_deltas = raw_time_deltas.fillna(2.0).values
    df["time_spent_seconds"] = vectorised_stoppage_cap(raw_time_deltas)
    df["season_delta"] = season_delta

    cols_to_keep = [
        "eventId",
        "teamId",
        "type",
        "outcomeType",
        "starting_state",
        "finishing_state",
        "time_spent_seconds",
        "season_delta",
    ]

    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols].copy()

    # change to snake_case from JS naming convention
    df = df.rename(
        columns={
            "eventId": "event_id",
            "teamId": "team_id",
            "outcomeType": "outcome_type",
        }
    )

    return df


def safe_parse(filepath):
    """
    Wrapper to catch corrupted files without crashing the whole script.
    """
    try:
        return parse_match_to_dataframe(filepath)
    except Exception as e:
        print(f"  [-] Failed to parse {os.path.basename(filepath)}: {str(e)}")
        return pd.DataFrame()  # Return empty df on failure
