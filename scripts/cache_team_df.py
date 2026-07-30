from config.constants import CLUB_ID_MAP
from src.ctmc.ctmc_builder import create_full_team_df, build_global_matrices
import pandas as pd

for club in CLUB_ID_MAP.keys():

    club_df = create_full_team_df(f"{club}", folder_path="./data")
    N_club, T_club, Q_club = build_global_matrices(club_df, gamma=0.05)

    # Save to your cache folder
    pd.DataFrame(N_club).to_csv(
        f"cache/{club.replace(" ", "_").lower()}_N_matrix.csv", index=False
    )
    pd.DataFrame(T_club).to_csv(
        f"cache/{club.replace(" ", "_").lower()}_T_vector.csv", index=False
    )
