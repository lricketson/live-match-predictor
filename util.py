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
    master_df["decay_weights"] = weights
    master_df["weighted_time"] = master_df["time_spent_seconds"] * weights

    # calculate the weighted numerator n_ij
    # put events with same starting and finishing states into buckets
    # then sum up all the weights of those events (wehre older events have a lower weight)
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

    n_grid = transition_counts.pivot(
        index="starting_state", columns="finishing_state", values="n_ij"
    ).reindex(index=STATES, columns=STATES, fill_value=0.0)

    T_grid = time_spent.set_index("starting_state")["T_i"].reindex(
        index=STATES, fill_value=0.0
    )

    q_grid = q_matrix.pivot(
        index="starting_state", columns="finishing_state", values="lambda_ij"
    ).reindex(index=STATES, columns=STATES, fill_value=0.0)

    N_mat = n_grid.to_numpy(dtype=np.float64)
    T_mat = T_grid.to_numpy(dtype=np.float64)
    Q_mat = q_grid.to_numpy(dtype=np.float64)

    # ensure row validity (rows sum to 0)
    for i in range(12):
        Q_mat[i, i] = 0  # zero out any accidental self-transition counts
        Q_mat[i, i] = -np.sum(Q_mat[i, :])  # set diagonal to negative row sum

    return N_mat, T_mat, Q_mat


def create_full_team_df(team_name, folder_path="./data/world_cup_2026"):
    # switch from spaces to underscores for file names
    search_name = team_name.replace(" ", "_")

    # get all files
    all_files = glob.glob(os.path.join(folder_path, "*.json"))

    team_files = [f for f in all_files if search_name in os.path.basename(f)]

    if not team_files:
        print(f"[-] No matches found for {team_name} in {folder_path}.")
        return pd.DataFrame()
    print(f"[*] Found {len(team_files)} matches for {team_name}. Parsing...")

    # turn files to dfs
    dfs_list = [safe_parse(file) for file in team_files]
    valid_dfs = [df for df in dfs_list if not df.empty]

    if not valid_dfs:
        print(f"[-] All files for {team_name} failed to parse.")
        return pd.DataFrame()

    # concatenate
    merged_df = pd.concat(valid_dfs)
    print(
        f"[+] Successfully built dataframe for {team_name} ({len(merged_df)} events)."
    )
    return merged_df


def calculate_specific_q(
    global_q: pd.DataFrame, alpha: float, team_data_df: pd.DataFrame
):
    team_data_clean = team_data_df.rename(columns={"n_ij": "team_n", "T_i": "team_T"})
    merged = pd.merge(
        left=global_q,
        right=team_data_clean[
            [
                "starting_state",
                "finishing_state",
                "team_n",
                "team_T",
            ]
        ],
        on=["starting_state", "finishing_state"],
        how="left",
    )

    merged["team_n"] = merged["team_n"].fillna(0)
    merged["team_T"] = merged["team_T"].fillna(0)

    merged["updated_lambda_ij"] = (merged["n_ij"] * alpha + merged["team_n"]) / (
        merged["T_i"] * alpha + merged["team_T"]
    )
    updated_q_matrix = merged[
        ["starting_state", "finishing_state", "updated_lambda_ij"]
    ].copy()
    updated_q_grid = updated_q_matrix.pivot(
        index="starting_state", columns="finishing_state", values="updated_lambda_ij"
    ).fillna(0)

    return updated_q_matrix, updated_q_grid
