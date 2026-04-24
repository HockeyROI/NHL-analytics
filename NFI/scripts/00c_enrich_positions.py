#!/usr/bin/env python3
"""
Enrich positions for 20212022 unknown players via NHL player landing API.
Pulls https://api-web.nhle.com/v1/player/{pid}/landing and grabs positionCode.
Appends to player_positions.csv with pos_source tag.
Flags any players that still have no position to player_positions_unknown.csv.
"""
import os, json, urllib.request, urllib.error, time, csv
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
POS = f"{ROOT}/NFI/output/player_positions.csv"
SHIFT = f"{ROOT}/2026 posts/Geometry_post/Data/shift_data.csv"
GAMES = f"{ROOT}/Data/game_ids.csv"

UA = {"User-Agent": "Mozilla/5.0 (research)"}

def fetch(url, retries=2):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.load(r)
        except Exception as e:
            if i == retries-1:
                return None
            time.sleep(1)
    return None

# load existing positions
pos = pd.read_csv(POS)
known = dict(zip(pos["player_id"].astype(int), zip(pos["player_name"], pos["position"], pos["pos_group"])))
pos["pos_source"] = "pbp_rosterSpots_2022plus"

# find unknown players for 20212022
games = pd.read_csv(GAMES, dtype={"season":str})
games = games[games["game_type"]=="regular"]
gs_map = dict(zip(games["game_id"], games["season"]))

unk = set()
for ch in pd.read_csv(SHIFT, usecols=["game_id","player_id","first_name","last_name"], chunksize=500000):
    ch = ch.dropna(subset=["game_id","player_id"])
    ch["game_id"] = ch["game_id"].astype(int)
    ch["player_id"] = ch["player_id"].astype(int)
    for gid, pid, fn, ln in zip(ch["game_id"], ch["player_id"], ch["first_name"], ch["last_name"]):
        if gs_map.get(gid) == "20212022" and pid not in known:
            unk.add((pid, fn, ln))

print(f"Unknown 20212022 players to enrich: {len(unk)}")

# pull API per player
added = []
failed = []
for i, (pid, fn, ln) in enumerate(sorted(unk)):
    if i % 20 == 0:
        print(f"  {i}/{len(unk)}")
    d = fetch(f"https://api-web.nhle.com/v1/player/{pid}/landing")
    if d is None or "position" not in d:
        failed.append((pid, fn, ln))
        continue
    pcode = d.get("position", "")
    grp = "D" if pcode == "D" else ("G" if pcode == "G" else ("F" if pcode in ("C","L","R") else "U"))
    name = f"{fn} {ln}".strip() if fn and ln else d.get("firstName",{}).get("default","")+" "+d.get("lastName",{}).get("default","")
    added.append({"player_id": pid, "player_name": name, "position": pcode,
                  "pos_group": grp, "pos_source": "api_landing_2021_22"})

print(f"API success: {len(added)}, failed: {len(failed)}")

# merge
pos_out = pd.concat([pos, pd.DataFrame(added)], ignore_index=True)
pos_out = pos_out.drop_duplicates(subset="player_id", keep="first")
pos_out.to_csv(POS, index=False)
print(f"Total positions: {len(pos_out)}")

# flag any still unknown
if failed:
    with open(f"{ROOT}/NFI/output/player_positions_unknown.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_id","first_name","last_name","reason"])
        for pid,fn,ln in failed:
            w.writerow([pid, fn or "", ln or "", "api_lookup_failed"])
    print(f"Wrote {len(failed)} unconfirmed players to player_positions_unknown.csv")
