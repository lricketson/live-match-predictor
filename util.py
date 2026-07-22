import pandas as pd
import os
import glob
from helpers import (
    parse_match_to_dataframe,
    safe_parse,
)


def create_master_df(folder_path="./data/world_cup_2026"):
    print(f"[*] Scanning '{folder_path}' for match data...")

    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    if not json_files:
        print("[-] No JSON files found. Check your directory path.")
        return pd.DataFrame()

    print(f"[*] Found {len(json_files)} match files. Beginning parsing...")

    df_list = []
    failed_files = []

    for i, file_path in enumerate(json_files, 1):
        try:
            match_df = parse_match_to_dataframe(file_path)
            if not match_df.empty:
                df_list.append(match_df)
            print(
                f"  [+] ({i}/{len(json_files)}) Parsed: {os.path.basename(file_path)}"
            )
        except Exception as e:
            # if a file is formatted weirdly or corrupted, log it and continue
            print(
                f"  [-] ({i}/{len(json_files)}) Failed to parse {os.path.basename(file_path)}: {str(e)}"
            )
            failed_files.append(file_path)
    if not df_list:
        print("[-] Critical Error: All files failed to parse.")
        return pd.DataFrame()

    print("[*] Concatenating all parsed matches into Master DataFrame...")

    master_df = pd.concat(df_list)

    print(f"[+] Master DataFrame built successfully! Total Events: {len(master_df)}")

    if failed_files:
        print(f"[!] Warning: {len(failed_files)} files were skipped due to errors.")

    return master_df


def calculate_global_q(master_df: pd.DataFrame):
    """
    Takes a concatenated DataFrame of all matches and returns a global generator matrix Q.
    """

    # calculate the numerator n_ij
    # find events with same starting and finishing states and count the occurrences
    transition_counts = (
        master_df.groupby(["starting_state", "finishing_state"])
        .size()
        .reset_index(name="n_ij")
    )

    # find total time spent in state i
    time_spent = (
        master_df.groupby("starting_state")["time_spent_seconds"]
        .sum()
        .reset_index(name="T_i")
    )

    q_matrix = pd.merge(transition_counts, time_spent, on="starting_state")

    q_matrix["lambda_ij"] = q_matrix["n_ij"] / q_matrix["T_i"]

    q_grid = q_matrix.pivot(
        index="starting_state", columns="finishing_state", values="lambda_ij"
    ).fillna(0)

    return q_matrix, q_grid


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


def neutralise_global_prior(global_q: pd.DataFrame):
    """
    Strips home field advantage from the global prior matrix by averaging out mirror-image
    'H' and 'A' transition rates.
    """

    df = global_q.copy()

    def get_twin_state(state: str) -> str:
        possession = state[-1]
        if state.startswith("Goal"):
            twin_state = "Goal_H" if possession == "A" else "Goal_A"
            return twin_state
        zone_number = state[2]
        mirror_possession = "A" if possession == "H" else "H"
        return f"Z:{zone_number}_P:{mirror_possession}"

    df["twin_start"] = df["starting_state"].apply(get_twin_state)
    df["twin_finish"] = df["finishing_state"].apply(get_twin_state)

    twin_lookup = df[
        ["starting_state", "finishing_state", "lambda_ij", "n_ij", "T_i"]
    ].rename(columns={"lambda_ij": "twin_lambda", "n_ij": "twin_n", "T_i": "twin_T"})

    merged = pd.merge(
        df,
        twin_lookup,
        left_on=["twin_start", "twin_finish"],
        right_on=["starting_state", "finishing_state"],
        suffixes=("", "_drop"),
    )

    merged["lambda_ij"] = (merged["lambda_ij"] + merged["twin_lambda"]) / 2.0
    merged["n_ij"] = (merged["n_ij"] + merged["twin_n"]) / 2.0
    merged["T_i"] = (merged["T_i"] + merged["twin_T"]) / 2.0

    neutral_q = merged.drop(
        columns=[
            "twin_start",
            "twin_finish",
            "twin_lambda",
            "twin_n",
            "twin_T",
            "starting_state_drop",
            "finishing_state_drop",
        ]
    )

    return neutral_q
