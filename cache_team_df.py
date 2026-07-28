# update this to loop through all clubs

arsenal_df = create_full_team_df("Arsenal", folder_path="./data")
N_arsenal, T_arsenal, Q_arsenal = build_global_matrices(arsenal_df, gamma=0.05)

# Save to your cache folder
pd.DataFrame(N_arsenal).to_csv("cache/arsenal_N_matrix.csv", index=False)
pd.DataFrame(T_arsenal).to_csv("cache/arsenal_T_vector.csv", index=False)
