#!/usr/bin/env python3
"""Check how many players appear in shift_data for 20212022/20252026 that are
NOT in the 2022+ pbp-derived position lookup."""
import pandas as pd
ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
SHIFT = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
POS = f"{ROOT}/NFI/output/player_positions.csv"
GAMES = f"{ROOT}/Data/game_ids.csv"

games = pd.read_csv(GAMES, dtype={"season":str})
games = games[games["game_type"]=="regular"]
gs_map = dict(zip(games["game_id"], games["season"]))

pos = pd.read_csv(POS)
known = set(pos["player_id"].astype(int).tolist())
print(f"Known positions: {len(known)}")

it = pd.read_csv(SHIFT, usecols=["game_id","player_id"], chunksize=500000)
by_season = {"20212022": set(), "20252026": set(), "20222023": set(), "20232024": set(), "20242025": set()}
for ch in it:
    ch = ch.dropna()
    ch["game_id"] = ch["game_id"].astype(int)
    ch["player_id"] = ch["player_id"].astype(int)
    for gid, pid in zip(ch["game_id"], ch["player_id"]):
        s = gs_map.get(gid)
        if s in by_season:
            by_season[s].add(pid)

for s in sorted(by_season):
    all_p = by_season[s]
    unk = all_p - known
    print(f"{s}: players={len(all_p):>5}  unknown={len(unk):>4}  ({100*len(unk)/max(1,len(all_p)):.1f}%)")

# Show what players are unknown in 20212022
unk_2122 = by_season["20212022"] - known
print(f"\n20212022 unknown player IDs (first 20): {sorted(list(unk_2122))[:20]}")
print(f"20252026 unknown player IDs (first 20): {sorted(list(by_season['20252026'] - known))[:20]}")
