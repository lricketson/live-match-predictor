import os
import glob
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from util import parse_match_to_dataframe
from constants import STATE_TO_IDX, HOME_ATTACK_IDX, AWAY_ATTACK_IDX

HOME_STATES_IDX = [0, 1, 2, 3, 4]
AWAY_STATES_IDX = [5, 6, 7, 8, 9]


def extract_features_at_minute(
    df_past: pd.DataFrame, elapsed_seconds: float, epsilon: float = 1e-6
) -> List[float]:
    """
    Calculates the raw 5D feature vector (V0, V1, V2, V3, V4) from kickoff up to time t.
    """
    if df_past.empty:
        return [0.5, 0.0, 0.0, 0.0, 0.0]

    # build historical n_past and T_past ledgers
    n_past = np.zeros((12, 12), dtype=np.float32)
    T_past = np.zeros(12, dtype=np.float32)

    for _, row in df_past.iterrows():
        start_idx = STATE_TO_IDX.get(row.get("starting_state"))
        finish_idx = STATE_TO_IDX.get(row.get("finishing_state"))
        t_spent = row.get("time_spent_seconds", 0.0)

        if start_idx is not None:
            T_past[start_idx] += float(t_spent)
            if finish_idx is not None:
                n_past[start_idx, finish_idx] += 1.0

    # V0: Field Tilt (Home share of attacking third touches)
    home_att = n_past[HOME_ATTACK_IDX, :].sum()
    away_att = n_past[AWAY_ATTACK_IDX, :].sum()

    total_att = home_att + away_att
    v0_tilt = float(home_att / total_att) if total_att > 0 else 0.50

    # V1 and V2: progression ratio (Home forward zone transitions / Total Home transitions)
    def calculate_progression_ratio(team_indices: List[int]) -> float:
        progressions = 0.0
        total_transitions = 0.0
        for i in range(len(team_indices)):
            s_idx = team_indices[i]
            total_transitions += n_past[s_idx, :].sum()
            for j in range(i + 1, len(team_indices)):
                f_idx = team_indices[j]
                progressions += n_past[s_idx, f_idx]
        return float(progressions / total_transitions) if total_transitions > 0 else 0.0

    v1_h_prog = calculate_progression_ratio(HOME_STATES_IDX)
    v2_a_prog = calculate_progression_ratio(AWAY_STATES_IDX)

    # V3: Markovian tempo (total touches per minute)
    # sum of departure rates across all the zones 0-9
    active_departures = n_past[:10, :].sum(axis=1)
    active_T = T_past[:10] + epsilon
    transition_rates = active_departures / active_T
    v3_tempo = float(np.sum(transition_rates))

    # V4: goal differential
    # n_past is the 12x12 matrix of transition counts, so [:, 10] gets the counts of an event that started
    # somewhere and ended in zone 10 (home goal). and then we sum up all those counts
    home_goals = n_past[:, 10].sum()
    away_goals = n_past[:, 11].sum()
    v4_score_diff = float(home_goals - away_goals)

    return [v0_tilt, v1_h_prog, v2_a_prog, v3_tempo, v4_score_diff]


def build_future_ledgers(df_future: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregates future transition counts (n) and holding times (T) from minute t to full time,
    for a given match dataframe.
    Returns: n_matrix shape (12, 12) and T_vector shape (12,).
    """
    n_matrix = np.zeros((12, 12), dtype=np.float32)
    T_vector = np.zeros(12, dtype=np.float32)

    if df_future.empty:
        return n_matrix, T_vector

    for _, row in df_future.iterrows():
        # for each event, get the starting state, and then get the idx of that state from STATE_TO_IDX
        s_idx = STATE_TO_IDX.get(row.get("starting_state"))
        f_idx = STATE_TO_IDX.get(row.get("finishing_state"))
        t_spent = row.get("time_spent_seconds")

        if s_idx is not None:
            T_vector[s_idx] += t_spent
            if f_idx is not None:
                n_matrix[s_idx, f_idx] += 1.0

    return n_matrix, T_vector


def compile_historical_database(
    input_dir: str = "./data/world_cup_2026",
    output_dir: str = "./compiled_db/",
    minute_slices: List[int] = list(range(5, 95, 5)),
    epsilon: float = 1e-6,
):
    """
    The master offline compiler: scans JSON files, slices by timestamp, standardises 5D vectors,
    and exports production .pt files for real-time K-NN ingestion.
    """

    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    if not json_files:
        print(f"[-] Error: no JSON files found in input directory {input_dir}.")
        return

    print(f"Found {len(json_files)} JSON match files. Beginning offline compilation...")

    # staging dictionaries: map minute -> list of vectors/matrices (e.g. n_matrices or T_vectors) across all matches
    staging_vectors = {m: [] for m in minute_slices}
    staging_n_future = {m: [] for m in minute_slices}
    staging_T_future = {m: [] for m in minute_slices}

    valid_matches_processed = 0

    for i, filepath in enumerate(json_files, 1):
        try:
            df = parse_match_to_dataframe(filepath)
            if df.empty or "starting_state" not in df.columns:
                continue

            for minute in minute_slices:
                elapsed_seconds = float(minute * 60)

                # time-based split
                df_past = df.loc[df.index <= elapsed_seconds]
                df_future = df.loc[df.index > elapsed_seconds]

                raw_vec = extract_features_at_minute(df_past, elapsed_seconds, epsilon)
                n_mat, T_vec = build_future_ledgers(df_future)

                staging_vectors[minute].append(raw_vec)
                staging_n_future[minute].append(n_mat)
                staging_T_future[minute].append(T_vec)

            valid_matches_processed += 1
            print(
                f"\r[+] Processed ({i}/{len(json_files)}): {os.path.basename(filepath)}",
                end="",
                flush=True,
            )

        except Exception as e:
            print(f"\n[-] Skipping corrupted file{filepath}: {e}")

    print(f"\n\n[+] Successfully parsed {valid_matches_processed} complete matches.")
    print("[+] Computing 5D global normalization parameters and saving .pt slices...")

    # standardise vectors and export PyTorch tensor files per minute slice
    for minute in minute_slices:
        # raw_matrix is the un-normalised 'master table' of all the 5D feature vectors up to (cumulatively)
        # the certain minute mark
        raw_matrix = np.array(staging_vectors[minute], dtype=np.float32)  # shape (M, 5)
        n_future_array = np.array(
            staging_n_future[minute], dtype=np.float32
        )  # shape (M, 12, 12)
        T_future_array = np.array(
            staging_T_future[minute], dtype=np.float32
        )  # shape (M, 12)

        raw_tensor = torch.tensor(raw_matrix, dtype=torch.float32)
        n_future_tensor = torch.tensor(n_future_array, dtype=torch.float32)
        T_future_tensor = torch.tensor(T_future_array, dtype=torch.float32)

        # mu is shape (5,) because it's the mean of each of the 5 elements of the vector
        mu = raw_tensor.mean(dim=0)

        sigma = raw_tensor.std(dim=0)  # shape (5,)

        norm_tensor = (raw_tensor - mu) / (sigma + epsilon)

        slice_payload = {
            "minute": minute,
            "num_matches": valid_matches_processed,
            "vectors_normalised": norm_tensor,
            "n_future": n_future_tensor,
            "T_future": T_future_tensor,
            "mu": mu,
            "sigma": sigma,
        }

        output_filename = os.path.join(output_dir, f"slice_min_{minute}.pt")
        torch.save(slice_payload, output_filename)

    print(f"[+] Complete! All 5D offline database slices saved to {output_dir}")


if __name__ == "__main__":
    compile_historical_database(
        input_dir="./data/world_cup_2026/",
        output_dir="./compiled_db/",
        minute_slices=[10 * i for i in range(1, 10)],
    )
