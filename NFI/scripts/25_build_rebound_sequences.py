#!/usr/bin/env python3
"""
Build rebound_sequences.csv from nhl_shot_events.csv.

Definition:
  A rebound sequence is any SOG or goal (the "original") followed by another
  shot attempt (SOG, missed-shot, or goal — the "rebound") by the SAME team,
  in the SAME period of the SAME game, within 3 seconds.

For each original event we capture every eligible follow-up within the window
(so a single original can spawn multiple rebound rows if multiple follow-ups
land within 3 sec).

Filters:
  - Original event_type in {shot-on-goal, goal}
  - Rebound event_type in {shot-on-goal, missed-shot, goal}
  - Same shooting_team_abbrev for both
  - Regulation only (period 1-3)
  - All seasons present in source

Output: /Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/
        Data/rebound_sequences.csv
Sorted by game_id, period, time_secs (of original).
"""
import time
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
OUT_CSV  = f"{ROOT}/Data/rebound_sequences.csv"

WINDOW_SEC = 3
ORIG_TYPES = {"shot-on-goal", "goal"}
REB_TYPES  = {"shot-on-goal", "missed-shot", "goal"}

cols = ["game_id","season","period","time_secs","event_type","situation_code",
        "shooting_team_abbrev","shooter_player_id","goalie_id","shot_type",
        "x_coord_norm","y_coord_norm","is_goal"]

print("Loading shots ...")
t0 = time.time()
df = pd.read_csv(SHOT_CSV, usecols=cols)
print(f"  {len(df):,} rows loaded in {time.time()-t0:.1f}s")

# Regulation only, restrict to events that could matter
df = df[df["period"].between(1, 3)].copy()
df = df[df["event_type"].isin(ORIG_TYPES | REB_TYPES)].copy()
df["time_secs"] = df["time_secs"].astype(int)
df["period"]    = df["period"].astype(int)
print(f"  after regulation + event-type filter: {len(df):,}")

# Sort once
df = df.sort_values(["game_id","period","time_secs"], kind="mergesort")\
       .reset_index(drop=True)

# Vectorized per-game scan: for each game we extract arrays once and loop in
# Python only for the in-window window (small).
out_rows = []
n_games = 0
t0 = time.time()
for gid, g in df.groupby("game_id", sort=False):
    n_games += 1
    period   = g["period"].values
    tsec     = g["time_secs"].values
    et       = g["event_type"].values
    team     = g["shooting_team_abbrev"].values
    n = len(g)

    # Cache columns we'll output
    season    = g["season"].values
    sit       = g["situation_code"].values
    shot_type = g["shot_type"].values
    x         = g["x_coord_norm"].values
    y         = g["y_coord_norm"].values
    shooter   = g["shooter_player_id"].values
    goalie    = g["goalie_id"].values
    is_goal   = g["is_goal"].values

    for i in range(n):
        if et[i] not in ORIG_TYPES:
            continue
        ti = tsec[i]; pi = period[i]; tmi = team[i]
        for j in range(i+1, n):
            if period[j] != pi:
                break  # sorted by period within game; left this period entirely
            dt = tsec[j] - ti
            if dt > WINDOW_SEC:
                break
            if dt < 0:
                continue  # safety, shouldn't happen due to sort
            if team[j] != tmi:
                continue
            if et[j] not in REB_TYPES:
                continue
            out_rows.append((
                gid, season[i], pi,
                et[i],  x[i],  y[i],  shot_type[i], tmi,
                shooter[i], goalie[i],
                et[j],  x[j],  y[j],  shot_type[j],
                shooter[j], int(is_goal[j]),
                int(dt),  sit[i],
            ))
    if n_games % 1000 == 0:
        print(f"  {n_games} games scanned, {len(out_rows):,} pairs so far "
              f"({time.time()-t0:.1f}s)")

print(f"\nTotal: {n_games} games, {len(out_rows):,} rebound pairs "
      f"in {time.time()-t0:.1f}s")

# Build dataframe
out = pd.DataFrame(out_rows, columns=[
    "game_id","season","period",
    "orig_event_type","orig_x","orig_y","orig_shot_type","orig_team",
    "orig_shooter_id","orig_goalie_id",
    "reb_event_type","reb_x","reb_y","reb_shot_type",
    "reb_shooter_id","reb_is_goal",
    "time_gap_secs","situation_code",
])
# Sort as requested
# Need orig time to sort within (game_id, period); recover from time_gap and
# the rebound's time isn't here, but we can recompute from a left join.
# Simpler: sort by game_id/period only — the natural insertion order already
# preserves ascending time_secs of the original event because we processed
# events in sorted order.
out = out.sort_values(["game_id","period"], kind="mergesort").reset_index(drop=True)

out.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}  ({len(out):,} rows)")

# ------- Reporting -------
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.width = 200

print("\n=== Row count ===")
print(f"  {len(out):,} rebound pairs")

print("\n=== Season distribution ===")
print(out["season"].value_counts().sort_index().to_string())

print("\n=== time_gap_secs distribution ===")
print(out["time_gap_secs"].value_counts().sort_index().to_string())
print("\n  Cumulative %:")
counts = out["time_gap_secs"].value_counts().sort_index()
cum_pct = (counts.cumsum() / counts.sum() * 100).round(2)
for k, v in cum_pct.items():
    print(f"    ≤ {k}s : {v}%")

print("\n=== reb_event_type breakdown ===")
vc = out["reb_event_type"].value_counts()
for k, v in vc.items():
    print(f"  {k}: {v:,}  ({v/len(out)*100:.2f}%)")
print(f"\n  Conversion (rebound goals / rebound attempts):")
g_cnt = (out["reb_event_type"]=="goal").sum()
print(f"    {g_cnt:,} / {len(out):,} = {g_cnt/len(out)*100:.2f}%")

print("\n=== situation_code distribution (top 15) ===")
sit_vc = out["situation_code"].value_counts().head(15)
for k, v in sit_vc.items():
    print(f"  {k}: {v:,}  ({v/len(out)*100:.2f}%)")
