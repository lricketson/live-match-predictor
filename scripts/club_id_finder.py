from pathlib import Path
import json

club_id_map = {}

# POINT THIS TO YOUR MASTER FOLDER CONTAINING ALL SEASON SUBFOLDERS
master_data_path = Path("./data")

print(f"[*] Recursively scanning '{master_data_path}' across all seasons...")

# .rglob() automatically dives into every season folder (11/12, 12/13, etc.)
for json_file in master_data_path.rglob("*.json"):
    try:
        with open(json_file, mode="r", encoding="utf-8") as f:
            data = json.load(f)

        for side in ["home", "away"]:
            if side in data:
                team_id = data[side].get("teamId")

                # Defend against 15 years of Opta schema drift ('name' vs 'teamName')
                team_name = (
                    data[side].get("name")
                    or data[side].get("teamName")
                    or f"Unknown_Club_{team_id}"
                )

                # Populate dictionary if valid and not already recorded
                if team_id and team_name not in club_id_map:
                    club_id_map[team_name] = team_id

    except Exception as e:
        print(f"  [-] Failed to read {json_file.name}: {e}")

# Sort alphabetically by club name for a clean printout
sorted_clubs = dict(sorted(club_id_map.items(), key=lambda item: item[0]))

print(
    f"[+] Successfully extracted {len(sorted_clubs)} unique clubs across all seasons!\n"
)
print("--- Copy the block below into your constants.py ---")
print("CLUB_ID_MAP = {")
for team_name, team_id in sorted_clubs.items():
    print(f'    "{team_name}": {team_id},')
print("}")
