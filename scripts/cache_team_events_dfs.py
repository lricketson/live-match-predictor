from config.constants import CLUB_ID_MAP
from src.ctmc.ctmc_builder import create_full_team_df

for club in CLUB_ID_MAP.keys():
    cleaned_name = club.replace(" ", "_").lower()
    raw_df = create_full_team_df(cleaned_name)
    raw_df.to_parquet(f"./cache/{cleaned_name}_events_df.parquet")
