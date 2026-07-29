import pandas as pd
import os
import glob
from helpers import (
    parse_match_to_dataframe,
    safe_parse,
)
from constants import STATES
import numpy as np


def create_master_df(folder_path="./data") -> pd.DataFrame:
    """
    Creates a master dataframe of every event that occurred in historical Premier League matches
    from 2011 to 2026 across nested season subfolders.
    """
    print(f"[*] Scanning '{folder_path}' for nested Premier League match data...")

    # 1. THE FIX: Look inside any subfolder starting with 'premier_league_' for .json files
    search_pattern = os.path.join(folder_path, "premier_league_*", "*.json")
    json_files = glob.glob(search_pattern)

    if not json_files:
        print(f"[-] No JSON files found matching pattern: {search_pattern}")
        print(
            "    Check your folder_path and ensure subfolders are named 'premier_league_YY_YY'."
        )
        return pd.DataFrame()

    # 2. Sort chronologically so we parse season-by-season instead of random OS order
    json_files = sorted(json_files)

    print(
        f"[*] Found {len(json_files)} match files across {len(set(os.path.dirname(p) for p in json_files))} season folders. Beginning parsing..."
    )

    df_list = []
    failed_files = []

    for i, file_path in enumerate(json_files, 1):
        try:
            # Parse the match using your existing parser function
            match_df = parse_match_to_dataframe(file_path)

            if not match_df.empty:
                df_list.append(match_df)

            # 3. Upgraded logging: Grab the parent folder name so you know WHICH season is parsing!
            parent_folder = os.path.basename(os.path.dirname(file_path))
            file_name = os.path.basename(file_path)

            print(
                f"  [+] ({i}/{len(json_files)}) Parsed: {parent_folder}/{file_name}\r",
                end="",
                flush=True,
            )
        except Exception as e:
            # If a file is formatted weirdly or corrupted, log it and continue
            print(
                f"\n  [-] ({i}/{len(json_files)}) Failed to parse {os.path.basename(file_path)}: {str(e)}"
            )
            failed_files.append(file_path)

    if not df_list:
        print("\n[-] Critical Error: All files failed to parse.")
        return pd.DataFrame()

    print("\n[*] Concatenating all parsed matches into Master DataFrame...")
    master_df = pd.concat(df_list, ignore_index=True)

    print(f"[+] Master DataFrame built successfully! Total Events: {len(master_df)}")

    if failed_files:
        print(f"[!] Warning: {len(failed_files)} files were skipped due to errors.")

    return master_df


def build_global_matrices(
    master_df: pd.DataFrame, gamma: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes a master DataFrame of all matches and returns rigid 12x12 NumPy matrices (N, T, Q) ready
    for Bayesian conjugate updating.
    """

    if master_df.empty:
        return np.zeros((12, 12)), np.zeros((12, 1)), np.zeros((12, 12))

    weights = np.exp(-gamma * master_df["season_delta"].values)
    master_df["decay_weight"] = weights
    master_df["weighted_time"] = master_df["time_spent_seconds"] * weights

    # calculate the weighted numerator n_ij
    # put events with same starting and finishing states into buckets
    # then sum up all the weights of those events (where older events have a lower weight)
    transition_counts = (
        master_df.groupby(["starting_state", "finishing_state"])["decay_weight"]
        .sum()
        .reset_index(name="n_ij")  # names the resulting column
    )

    # find total time spent in state i
    time_spent = (
        master_df.groupby("starting_state")["weighted_time"]
        .sum()
        .reset_index(name="T_i")
    )

    q_matrix = pd.merge(transition_counts, time_spent, on="starting_state")
    q_matrix["lambda_ij"] = q_matrix["n_ij"] / q_matrix["T_i"]

    n_grid = (
        transition_counts.pivot(
            index="starting_state", columns="finishing_state", values="n_ij"
        )
        .reindex(index=STATES, columns=STATES, fill_value=0.0)
        .fillna(0.0)
    )

    T_grid = (
        time_spent.set_index("starting_state")["T_i"]
        .reindex(index=STATES, fill_value=0.0)
        .fillna(0.0)
    )

    q_grid = (
        q_matrix.pivot(
            index="starting_state", columns="finishing_state", values="lambda_ij"
        )
        .reindex(index=STATES, columns=STATES, fill_value=0.0)
        .fillna(0.0)
    )

    N_mat = n_grid.to_numpy(dtype=np.float64, copy=True)
    T_mat = T_grid.to_numpy(dtype=np.float64, copy=True).reshape(-1, 1)
    Q_mat = q_grid.to_numpy(dtype=np.float64, copy=True)

    # ensure row validity (rows sum to 0)
    for i in range(10):
        Q_mat[i, i] = 0  # zero out any accidental self-transition counts
        Q_mat[i, i] = -np.nansum(Q_mat[i, :])  # set diagonal to negative row sum

    return N_mat, T_mat, Q_mat


def create_full_team_df(team_name: str, folder_path: str = "./data") -> pd.DataFrame:
    """
    Creates a master dataframe of every single event performed by a specific team across all
    historical seasons, strictly isolating their Home touches (P:H) and Away touches (P:A).
    """
    search_name = team_name.replace(" ", "_")

    # 1. Scan across all nested season folders for any filename matching the team name
    search_pattern = os.path.join(
        folder_path, "premier_league_*", f"*{search_name}*.json"
    )
    team_files = sorted(glob.glob(search_pattern))

    if not team_files:
        print(f"[-] No matches found for '{team_name}' in {folder_path}.")
        return pd.DataFrame()

    print(
        f"[*] Found {len(team_files)} match files for '{team_name}'. Beginning parsing..."
    )

    valid_dfs = []

    for i, file_path in enumerate(team_files, 1):
        try:
            df = parse_match_to_dataframe(file_path)
            if df.empty:
                continue

            filename = os.path.basename(file_path)

            # 2. THE GOLDEN RULE: Isolate the target club's actual touches!
            # If team_name comes BEFORE '_vs_', they were the Home team -> keep P:H touches (Rows 0-4)
            # If team_name comes AFTER '_vs_', they were the Away team -> keep P:A touches (Rows 5-9)
            if f"_{search_name}_vs_" in filename or filename.startswith(
                f"{search_name}_vs_"
            ):
                df_team = df[df["starting_state"].str.contains("_P:H")].copy()
            else:
                df_team = df[df["starting_state"].str.contains("_P:A")].copy()

            if not df_team.empty:
                valid_dfs.append(df_team)

            print(
                f"  [+] ({i}/{len(team_files)}) Filtered: {filename}\r",
                end="",
                flush=True,
            )
        except Exception as e:
            print(f"\n  [-] Failed on {os.path.basename(file_path)}: {str(e)}")

    if not valid_dfs:
        print(f"\n[-] All files for '{team_name}' resulted in empty DataFrames.")
        return pd.DataFrame()

    print(f"\n[*] Concatenating events for '{team_name}'...")
    merged_df = pd.concat(valid_dfs, ignore_index=True)

    print(
        f"[+] Successfully built team DataFrame for '{team_name}' ({len(merged_df)} events)."
    )
    return merged_df


def calculate_specific_q(
    global_N: np.ndarray,
    global_T: np.ndarray,
    home_team_N: np.ndarray,
    home_team_T: np.ndarray,
    away_team_N: np.ndarray,
    away_team_T: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs real-time Bayesian conjugate updating in <1ms latency by blending global league
    priors with specific club home/away baselines using hyperparameter alpha.
    """
    # 1. Assemble the pre-match Counts (N_fixture) and Holding Times (T_fixture)
    # initialise these arrays with zeros in the same shapes as global_N and global_T
    N_fixture = np.zeros_like(global_N, dtype=np.float64)
    T_fixture = np.zeros_like(global_T, dtype=np.float64)

    # Extract Home team's attacking profile (Rows 0 to 4: P:H) and input those numbers into the arrays
    # N_fixture is the array of counts for the fixture and contains both home and away counts
    # same goes for T_fixture
    N_fixture[0:5, :] = home_team_N[0:5, :]
    T_fixture[0:5, :] = home_team_T[0:5, :]

    # Extract Away team's attacking profile (Rows 5 to 9: P:A)
    N_fixture[5:10, :] = away_team_N[5:10, :]
    T_fixture[5:10, :] = away_team_T[5:10, :]

    # 2. Apply Bayesian Conjugate Update against Global Prior using hyperparameter alpha:
    # Formula: N_updated = (alpha * N_global) + N_fixture
    N_updated = (alpha * global_N) + N_fixture
    T_updated = (alpha * global_T) + T_fixture

    # 3. Compute continuous-time transition rates (lambda_ij = N_ij / T_i)
    Q_updated = np.zeros_like(N_updated, dtype=np.float64)

    # only divide across transient states (indices 0 to 9)
    # we ignore rows 10 and 11 because i represents the starting state, and we can't transition
    # from a goal state to another state because goal states are absorbing
    for i in range(10):
        # to prevent division by zero errors:
        if T_updated[i, 0] > 0:
            Q_updated[i, :] = N_updated[i, :] / T_updated[i, 0]
            # Zero out self-loop rates and set diagonal to negative row sum
            Q_updated[i, i] = 0.0
            Q_updated[i, i] = -np.sum(Q_updated[i, :])

    return N_updated, T_updated, Q_updated
