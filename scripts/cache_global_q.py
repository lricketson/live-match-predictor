import pandas as pd
import numpy as np
from util import create_master_df, build_global_matrices
from helpers import standardise_possessions
import os

# 1. Load all ~5700 Premier League matches (11/12 to 25/26)
print("[*] Compiling 15-season master dataset...")
master_df = create_master_df(folder_path="./data")

# maybe use this?
# master_df = standardise_possessions(master_df)

# 2. Extract the 12x12 Counts (N), Holding Times (T), and Rates (Q)
print("[*] Calculating global baseline matrices...")
N_global, T_global, Q_global = build_global_matrices(master_df, gamma=0.05)

# 3. Cache the exact mathematical components needed for the Bayesian conjugate update
print("[*] Saving global priors to disk...")

os.makedirs("cache", exist_ok=True)

# Save the 12x12 Transition Counts (n_ij^global)
pd.DataFrame(N_global).to_csv("cache/global_N_matrix.csv", index=False)

# Save the 12x1 Holding Times (T_i^global)
pd.DataFrame(T_global).to_csv("cache/global_T_vector.csv", index=False)

# Save the ready-to-read Q grid just for quick visual inspection or debugging
pd.DataFrame(Q_global).to_csv("cache/global_Q_grid.csv", index=False)

print("[+] Pipeline complete! Global priors cached and ready for runtime updating.")
