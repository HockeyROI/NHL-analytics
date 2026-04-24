#!/usr/bin/env python3
"""Build player_id -> position lookup from pbp JSON roster spots."""
import os, json, csv, glob
from collections import defaultdict, Counter

PBP_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Zones/raw/pbp"
OUT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/NFI/output/player_positions.csv"

pos_counts = defaultdict(Counter)  # pid -> Counter of positionCode
name_map = {}

for path in glob.glob(os.path.join(PBP_DIR, "*.json")):
    try:
        d = json.load(open(path))
    except Exception:
        continue
    for p in d.get("rosterSpots", []):
        pid = p["playerId"]
        pos = p.get("positionCode", "")
        name = f"{p['firstName']['default']} {p['lastName']['default']}"
        pos_counts[pid][pos] += 1
        name_map[pid] = name

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["player_id", "player_name", "position", "pos_group"])
    for pid, ctr in sorted(pos_counts.items()):
        pos = ctr.most_common(1)[0][0]
        if pos == "D":
            grp = "D"
        elif pos == "G":
            grp = "G"
        elif pos in ("C", "L", "R"):
            grp = "F"
        else:
            grp = "U"
        w.writerow([pid, name_map.get(pid, ""), pos, grp])
print(f"Wrote {len(pos_counts)} players -> {OUT}")
