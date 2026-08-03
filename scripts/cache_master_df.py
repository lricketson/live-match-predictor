from src.ctmc.ctmc_builder import create_master_df

master_df = create_master_df("./data")

master_df.to_parquet("./cache/master_df.parquet", index=False)
