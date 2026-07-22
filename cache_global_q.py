from util import calculate_global_q, create_master_df
from helpers import standardise_possessions

master_df = create_master_df()
master_df = standardise_possessions(master_df)

global_q_matrix, global_q_grid = calculate_global_q(master_df)

global_q_matrix.to_csv("global_q_matrix.csv", index=False)
global_q_grid.to_csv("global_q_grid.csv", index=False)
