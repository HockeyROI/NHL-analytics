#!/usr/bin/env python3
"""
P1a centrality-weighted: for each forward's CNFI/MNFI on-ice attempts,
weight each attempt by the y-band conversion rate from the y-band analysis.

Definitions (matched to the y-band analysis just produced):
  CNFI_wide = x_coord_norm in [74, 89]   (no y restriction)
  MNFI_wide = x_coord_norm in [55, 73]   (no y restriction)
  y bands on |y_coord_norm|:
    0-5, 5-10, 10-15, 15-20, 20-25, 25-30, 30+

Filters:
  - ES (state == 'ES')
  - Regulation only (period 1-3) -- already enforced upstream
  - Same 5-season pool as existing pillar_1 pipeline:
        20212022, 20222023, 20232024, 20242025, 20252026
  - Forwards only
  - Min 500 ES TOI minutes per forward (matches pillar gate)
  - Unblocked attempts (Fenwick: shot-on-goal + missed-shot + goal),
    matching the y-band analysis (blocked-shot coords are block, not origin)

For each forward we report on-ice CNFI and MNFI:
  P1a_raw          - on-ice attempts/60 ES (unweighted)
  P1a_weighted     - sum(attempts * y-band weight) per 60 ES
                     (units = expected goals/60 if shooter had league-avg
                      conversion at each y-band; relative metric)
  P1a_centrality   - mean |y| of on-ice attempts (lower = more central)
"""

import os, math
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
SHIFT_CSV = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
TOI_CSV = f"{ROOT}/NFI/output/player_toi.csv"
POS_CSV = f"{ROOT}/NFI/output/player_positions.csv"
GAME_CSV = f"{ROOT}/Data/game_ids.csv"
OUT_CSV = f"{ROOT}/NFI/output/P1a_centrality_weighted.csv"

SEASONS = {"20212022", "20222023", "20232024", "20242025", "20252026"}
MIN_ES_TOI_MIN = 500.0

# y-band weights from preceding analysis (CNFI x 74-89, MNFI x 55-73,
# ES regulation 5 seasons pooled, Wilson-validated, all bands met min 500)
CNFI_WEIGHTS = {
    "0-5":   0.1579,
    "5-10":  0.1075,
    "10-15": 0.0781,
    "15-20": 0.0513,
    "20-25": 0.0318,
    "25-30": 0.0209,
    "30+":   0.0123,
}
MNFI_WEIGHTS = {
    "0-5":   0.1109,
    "5-10":  0.0985,
    "10-15": 0.0887,
    "15-20": 0.0692,
    "20-25": 0.0492,
    "25-30": 0.0278,
    "30+":   0.0122,
}
BAND_ORDER = ["0-5","5-10","10-15","15-20","20-25","25-30","30+"]

def y_band(absy):
    if absy < 5:   return "0-5"
    if absy < 10:  return "5-10"
    if absy < 15:  return "10-15"
    if absy < 20:  return "15-20"
    if absy < 25:  return "20-25"
    if absy < 30:  return "25-30"
    return "30+"

# ---- Load positions (forwards only for output) ----
print("Loading positions...")
pos_df = pd.read_csv(POS_CSV, dtype={"player_id": int})
pos_map = dict(zip(pos_df["player_id"], pos_df["pos_group"]))

# ---- Load TOI (use existing ES TOI from pillar pipeline) ----
print("Loading TOI...")
toi_df = pd.read_csv(TOI_CSV, dtype={"player_id": int})
toi_es_min = dict(zip(toi_df["player_id"], toi_df["toi_ES_sec"] / 60.0))

# ---- Load game->season ----
g_df = pd.read_csv(GAME_CSV, dtype={"game_id": int, "season": str})
g_df = g_df[g_df["game_type"] == "regular"]
game_season = dict(zip(g_df["game_id"], g_df["season"]))

# ---- Load shots, filter to ES regulation in seasons, x in zones, unblocked ----
print("Loading shots...")
shot_cols = ["game_id","season","period","event_id","event_type","situation_code",
             "time_secs","home_team_id","shooting_team_id","shooting_team_abbrev",
             "home_team_abbrev","away_team_abbrev","x_coord_norm","y_coord_norm","is_goal"]
shots = pd.read_csv(SHOT_CSV, usecols=shot_cols,
                    dtype={"season": str, "situation_code": str})
shots = shots[shots["season"].isin(SEASONS)].copy()
shots = shots[shots["period"].between(1, 3)].copy()
shots = shots[shots["event_type"].isin(["shot-on-goal","missed-shot","goal"])].copy()  # Fenwick

# ES filter via situation code 1551
shots = shots[shots["situation_code"].astype(str) == "1551"].copy()

# Zone classification (wide)
x = shots["x_coord_norm"].values
absy = shots["y_coord_norm"].abs().values
in_cnfi = (x >= 74) & (x <= 89)
in_mnfi = (x >= 55) & (x <= 73)
shots = shots[in_cnfi | in_mnfi].copy()
shots["zone"] = np.where(shots["x_coord_norm"].between(74, 89), "CNFI", "MNFI")
shots["abs_y"] = shots["y_coord_norm"].abs()
shots["band"] = shots["abs_y"].apply(y_band)
shots["abs_time"] = shots["time_secs"].astype(int) + (shots["period"].astype(int)-1)*1200
shots["_shoot_home"] = shots["shooting_team_id"] == shots["home_team_id"]
print(f"  shots in scope: {len(shots):,}  (CNFI {(shots['zone']=='CNFI').sum():,}, MNFI {(shots['zone']=='MNFI').sum():,})")

shots_by_game = dict(tuple(shots.groupby("game_id")))
valid_gids = set(shots_by_game.keys())

# ---- Load shifts (only games we need) ----
print("Loading shifts (filtered)...")
shift_cols = ["game_id","player_id","period","team_abbrev","abs_start_secs","abs_end_secs"]
parts = []
for ch in pd.read_csv(SHIFT_CSV, usecols=shift_cols, chunksize=500_000):
    ch = ch.dropna(subset=shift_cols)
    ch["game_id"] = ch["game_id"].astype(int)
    ch["player_id"] = ch["player_id"].astype(int)
    ch["period"] = ch["period"].astype(int)
    ch["abs_start_secs"] = ch["abs_start_secs"].astype(int)
    ch["abs_end_secs"] = ch["abs_end_secs"].astype(int)
    ch = ch[ch["game_id"].isin(valid_gids) & ch["period"].between(1, 3)]
    if len(ch):
        parts.append(ch)
shifts = pd.concat(parts, ignore_index=True)
del parts
print(f"  shifts: {len(shifts):,}")

shifts_by_game = dict(tuple(shifts.groupby("game_id")))

# ---- Aggregators: per (player, zone, band) on-ice for attempts and goals ----
# attempts[(pid, zone, band)] = count
attempts = defaultdict(int)
goals    = defaultdict(int)
abs_y_sum = defaultdict(float)   # for centrality (mean |y|)
abs_y_n   = defaultdict(int)

# ---- Per-game shot-shift join ----
print("Per-game join...")
n_games = 0
for gid, gshots in shots_by_game.items():
    n_games += 1
    if n_games % 500 == 0:
        print(f"  {n_games} games...")
    gshifts = shifts_by_game.get(gid)
    if gshifts is None or len(gshifts) == 0:
        continue

    # Build per-team shift arrays once
    shifts_by_team = {}
    for team_ab, tsh in gshifts.groupby("team_abbrev"):
        starts = tsh["abs_start_secs"].values.astype(np.int32)
        ends   = tsh["abs_end_secs"].values.astype(np.int32)
        pids   = tsh["player_id"].values.astype(np.int64)
        order = np.argsort(starts)
        starts = starts[order]; ends = ends[order]; pids = pids[order]
        shifts_by_team[team_ab] = (starts, ends, pids)

    for r in gshots.itertuples(index=False):
        t = int(r.abs_time)
        zone = r.zone
        band = r.band
        is_goal = int(r.is_goal)
        absy = float(r.abs_y)
        shoot_ab = r.shooting_team_abbrev
        if shoot_ab not in shifts_by_team:
            continue
        st, en, pids = shifts_by_team[shoot_ab]
        # on-ice if start <= t < end. Find candidates with start<=t.
        idx_max = np.searchsorted(st, t, side="right")
        if idx_max == 0:
            continue
        on_mask = en[:idx_max] > t
        on_pids = pids[:idx_max][on_mask]
        for pid in on_pids:
            if pos_map.get(int(pid)) != "F":
                continue
            key = (int(pid), zone, band)
            attempts[key] += 1
            goals[key]    += is_goal
            abs_y_sum[(int(pid), zone)] += absy
            abs_y_n[(int(pid), zone)]   += 1

print("Aggregation done.")

# ---- Build per-forward summary ----
print("Building summary...")
all_forwards = sorted({pid for (pid, _, _) in attempts.keys()})

rows = []
for pid in all_forwards:
    toi_min = toi_es_min.get(pid, 0.0)
    if toi_min < MIN_ES_TOI_MIN:
        continue
    rec = {"player_id": pid, "es_toi_min": round(toi_min, 2)}

    for zone, weights in [("CNFI", CNFI_WEIGHTS), ("MNFI", MNFI_WEIGHTS)]:
        total_att = 0
        weighted_sum = 0.0
        for band in BAND_ORDER:
            n = attempts.get((pid, zone, band), 0)
            total_att += n
            weighted_sum += n * weights[band]
        n_cent = abs_y_n.get((pid, zone), 0)
        cent = (abs_y_sum.get((pid, zone), 0.0) / n_cent) if n_cent else float("nan")
        per60_raw = total_att / toi_min * 60.0
        per60_w   = weighted_sum / toi_min * 60.0
        rec[f"{zone}_attempts"]     = total_att
        rec[f"P1a_raw_{zone}"]      = round(per60_raw, 4)
        rec[f"P1a_weighted_{zone}"] = round(per60_w, 4)
        rec[f"P1a_centrality_{zone}"] = round(cent, 3) if not math.isnan(cent) else np.nan
    # add player name
    rows.append(rec)

out = pd.DataFrame(rows)

# Add names
name_map = dict(zip(pos_df["player_id"], pos_df["player_name"]))
out.insert(1, "player_name", out["player_id"].map(name_map))

# ---- Ranking & flag (>10 position swing) ----
def rank_with_flag(df, raw_col, w_col, prefix):
    df = df.copy()
    df[f"{prefix}_rank_raw"] = df[raw_col].rank(ascending=False, method="min").astype(int)
    df[f"{prefix}_rank_w"]   = df[w_col].rank(ascending=False, method="min").astype(int)
    df[f"{prefix}_rank_delta"] = df[f"{prefix}_rank_raw"] - df[f"{prefix}_rank_w"]
    df[f"{prefix}_centrality_flag"] = np.where(
        df[f"{prefix}_rank_delta"].abs() > 10,
        np.where(df[f"{prefix}_rank_delta"] > 0, "BOOSTED", "PENALIZED"),
        ""
    )
    return df

out = rank_with_flag(out, "P1a_raw_CNFI", "P1a_weighted_CNFI", "CNFI")
out = rank_with_flag(out, "P1a_raw_MNFI", "P1a_weighted_MNFI", "MNFI")

# Sort by P1a_weighted_CNFI desc as primary view
out = out.sort_values("P1a_weighted_CNFI", ascending=False).reset_index(drop=True)

cols = ["player_id","player_name","es_toi_min",
        "CNFI_attempts","P1a_raw_CNFI","P1a_weighted_CNFI","P1a_centrality_CNFI",
        "CNFI_rank_raw","CNFI_rank_w","CNFI_rank_delta","CNFI_centrality_flag",
        "MNFI_attempts","P1a_raw_MNFI","P1a_weighted_MNFI","P1a_centrality_MNFI",
        "MNFI_rank_raw","MNFI_rank_w","MNFI_rank_delta","MNFI_centrality_flag"]
out = out[cols]

out.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}  ({len(out)} forwards)")

# Console summary
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.max_columns = None
pd.options.display.width = 220

print("\n--- Top 25 by P1a_weighted_CNFI ---")
print(out.head(25)[["player_name","es_toi_min","CNFI_attempts",
                    "P1a_raw_CNFI","P1a_weighted_CNFI","P1a_centrality_CNFI",
                    "CNFI_rank_raw","CNFI_rank_w","CNFI_rank_delta",
                    "CNFI_centrality_flag"]].to_string(index=False))

print("\n--- CNFI BOOSTED (rank improved >10 by weighting) ---")
boost = out[out["CNFI_centrality_flag"]=="BOOSTED"].sort_values("CNFI_rank_delta", ascending=False)
print(boost[["player_name","CNFI_attempts","P1a_raw_CNFI","P1a_weighted_CNFI",
             "P1a_centrality_CNFI","CNFI_rank_raw","CNFI_rank_w","CNFI_rank_delta"]]
      .head(25).to_string(index=False))

print("\n--- CNFI PENALIZED (rank dropped >10 by weighting) ---")
pen = out[out["CNFI_centrality_flag"]=="PENALIZED"].sort_values("CNFI_rank_delta")
print(pen[["player_name","CNFI_attempts","P1a_raw_CNFI","P1a_weighted_CNFI",
           "P1a_centrality_CNFI","CNFI_rank_raw","CNFI_rank_w","CNFI_rank_delta"]]
      .head(25).to_string(index=False))

print("\n--- MNFI BOOSTED (rank improved >10 by weighting) ---")
mb = out[out["MNFI_centrality_flag"]=="BOOSTED"].sort_values("MNFI_rank_delta", ascending=False)
print(mb[["player_name","MNFI_attempts","P1a_raw_MNFI","P1a_weighted_MNFI",
          "P1a_centrality_MNFI","MNFI_rank_raw","MNFI_rank_w","MNFI_rank_delta"]]
      .head(15).to_string(index=False))

print("\n--- MNFI PENALIZED (rank dropped >10 by weighting) ---")
mp = out[out["MNFI_centrality_flag"]=="PENALIZED"].sort_values("MNFI_rank_delta")
print(mp[["player_name","MNFI_attempts","P1a_raw_MNFI","P1a_weighted_MNFI",
          "P1a_centrality_MNFI","MNFI_rank_raw","MNFI_rank_w","MNFI_rank_delta"]]
      .head(15).to_string(index=False))
