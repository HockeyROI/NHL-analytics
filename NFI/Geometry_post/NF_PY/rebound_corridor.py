#!/usr/bin/env python3
"""
HockeyROI - Rebound Corridor Model
Save to: NHL analysis/rebound_corridor.py

Usage:
  cd "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
  python3 rebound_corridor.py

Source: Data/nhl_shot_events.csv (ES only, situation_code=1551)

Outputs (all in Data/):
  rebound_sequences.csv         — every rebound pair (orig shot → rebound shot)
  rebound_corridor_map.csv      — 10x10ft grid: rebound generation + danger rates
  player_rebound_positioning.csv — per-player rebound stats + position + timing
"""

import csv
import math
import os
import time

import numpy as np
import pandas as pd
import requests

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"
BASE_URL    = "https://api-web.nhle.com/v1"
REBOUND_GAP = 3        # seconds — max gap to qualify as a rebound sequence
MIN_REB_SHOTS = 20     # minimum rebound shots for player table
Z95 = 1.96

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "HockeyROI-Analysis/2.0"


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def wilson_ci(k, n, z=Z95):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def fetch_player_name(player_id):
    try:
        r = SESSION.get(f"{BASE_URL}/player/{int(player_id)}/landing", timeout=12)
        if r.status_code == 200:
            d = r.json()
            first = d.get("firstName", {}).get("default", "")
            last  = d.get("lastName",  {}).get("default", "")
            return f"{first} {last}".strip()
    except Exception:
        pass
    return f"ID:{player_id}"


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def bin10(v):
    """Floor to nearest 10."""
    try:
        return int(math.floor(float(v) / 10.0) * 10)
    except (TypeError, ValueError):
        return None


# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading shot data...")
df = pd.read_csv(
    os.path.join(DATA_DIR, "nhl_shot_events.csv"),
    dtype=str,          # load everything as string first — safest for mixed types
)

# Filter to ES
df = df[df["situation_code"] == "1551"].copy()

# Cast numeric columns
for col in ("time_secs", "is_goal"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
for col in ("x_coord_norm", "y_coord_norm"):
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["game_id"] = df["game_id"].astype(str)
df["period"]  = pd.to_numeric(df["period"], errors="coerce")

# Rows eligible to be the ORIGINAL shot (reached the goalie)
orig_types = {"shot-on-goal", "goal"}
# Rows eligible to be the REBOUND shot (attempt after original)
reb_types  = {"shot-on-goal", "goal", "missed-shot"}

df_orig = df[df["event_type"].isin(orig_types)].copy()
df_all  = df[df["event_type"].isin(reb_types)].copy()

print(f"  ES original-eligible (sog+goal): {len(df_orig):,}")
print(f"  ES rebound-eligible  (sog+goal+missed): {len(df_all):,}")

# ─── STEP 1: REBOUND SEQUENCE DETECTION ────────────────────────────────────────
print("\n── Step 1: Detecting rebound sequences ──")

# Sort once globally — groupby preserves sort order within groups
df_all_sorted = df_all.sort_values(["game_id", "period", "time_secs"]).reset_index(drop=True)

sequences = []

# Group the full set (orig + missed) by game+period; process each group
for (gid, period), grp in df_all_sorted.groupby(["game_id", "period"], sort=False):
    grp = grp.reset_index(drop=True)
    n   = len(grp)

    for i in range(n):
        # Original shot must be sog or goal
        if grp.at[i, "event_type"] not in orig_types:
            continue
        # Must have valid coords and time
        if pd.isna(grp.at[i, "time_secs"]):
            continue

        t0   = grp.at[i, "time_secs"]
        team = grp.at[i, "shooting_team_abbrev"]

        # Scan forward for the first qualifying rebound by same team within gap
        for j in range(i + 1, n):
            dt = grp.at[j, "time_secs"] - t0
            if pd.isna(dt) or dt > REBOUND_GAP:
                break
            if grp.at[j, "shooting_team_abbrev"] != team:
                continue

            # Found a rebound
            sequences.append({
                "game_id"          : gid,
                "season"           : grp.at[i, "season"],
                "period"           : period,
                "orig_event_type"  : grp.at[i, "event_type"],
                "orig_x"           : grp.at[i, "x_coord_norm"],
                "orig_y"           : grp.at[i, "y_coord_norm"],
                "orig_shot_type"   : grp.at[i, "shot_type"],
                "orig_team"        : grp.at[i, "shooting_team_abbrev"],
                "orig_shooter_id"  : grp.at[i, "shooter_player_id"],
                "reb_event_type"   : grp.at[j, "event_type"],
                "reb_x"            : grp.at[j, "x_coord_norm"],
                "reb_y"            : grp.at[j, "y_coord_norm"],
                "reb_shot_type"    : grp.at[j, "shot_type"],
                "reb_shooter_id"   : grp.at[j, "shooter_player_id"],
                "reb_is_goal"      : int(grp.at[j, "is_goal"]) if not pd.isna(grp.at[j, "is_goal"]) else 0,
                "time_gap_secs"    : dt,
            })
            break   # one rebound per original shot

seq_df = pd.DataFrame(sequences)
print(f"  Rebound sequences found: {len(seq_df):,}")

out_seq = os.path.join(DATA_DIR, "rebound_sequences.csv")
seq_df.to_csv(out_seq, index=False)
print(f"  Saved: {out_seq}")


# ─── STEP 2: REBOUND CORRIDOR MAP (10x10 ft bins) ──────────────────────────────
print("\n── Step 2: Building rebound corridor map ──")

# Work from original-shot universe (sog + goal) in offensive zone
df_orig_oz = df_orig[df_orig["x_coord_norm"] > 25].copy()
df_orig_oz["x_bin"] = df_orig_oz["x_coord_norm"].apply(bin10)
df_orig_oz["y_bin"] = df_orig_oz["y_coord_norm"].apply(bin10)
df_orig_oz = df_orig_oz.dropna(subset=["x_bin", "y_bin"])

# Tag which original shots generated a rebound
reb_orig_set = set(zip(
    seq_df["game_id"].astype(str),
    seq_df["season"],
    seq_df["period"],
    seq_df["orig_x"],
    seq_df["orig_y"],
))
# Note: using (game_id, season, period, x, y) as proxy key — robust enough
# Build a set of sequence row indices keyed by orig shot identifier
# Simpler: merge seq_df back onto df_orig_oz

# Create a join key on both sides
def make_key(row):
    return (str(row["game_id"]), str(row["season"]), str(row["period"]),
            str(row.get("x_coord_norm", row.get("orig_x", ""))),
            str(row.get("y_coord_norm", row.get("orig_y", ""))))

# Mark original shots that have a rebound
seq_keys = set(
    (str(r["game_id"]), str(r["season"]), str(r["period"]),
     str(r["orig_x"]), str(r["orig_y"]))
    for _, r in seq_df.iterrows()
)
df_orig_oz["had_rebound"] = df_orig_oz.apply(
    lambda r: (str(r["game_id"]), str(r["season"]), str(r["period"]),
               str(r["x_coord_norm"]), str(r["y_coord_norm"])) in seq_keys,
    axis=1,
).astype(int)

# Rebound sequences with valid orig coords in OZ and their rebound destination
seq_oz = seq_df[
    pd.to_numeric(seq_df["orig_x"], errors="coerce") > 25
].copy()
seq_oz["orig_x_bin"] = seq_oz["orig_x"].apply(bin10)
seq_oz["orig_y_bin"] = seq_oz["orig_y"].apply(bin10)
seq_oz["reb_x_bin"]  = seq_oz["reb_x"].apply(bin10)
seq_oz["reb_y_bin"]  = seq_oz["reb_y"].apply(bin10)

# Per-bin: shots, rebounds generated, rebound rate, goal rate on rebounds, top dest
corridor_rows = []

for (xb, yb), bin_grp in df_orig_oz.groupby(["x_bin", "y_bin"]):
    shots       = len(bin_grp)
    reb_gen     = int(bin_grp["had_rebound"].sum())
    reb_rate    = reb_gen / shots if shots > 0 else np.nan

    # Rebounds generated from this bin
    reb_from = seq_oz[(seq_oz["orig_x_bin"] == xb) & (seq_oz["orig_y_bin"] == yb)]
    reb_goals    = int(reb_from["reb_is_goal"].sum()) if len(reb_from) > 0 else 0
    reb_goal_rate = reb_goals / len(reb_from) if len(reb_from) > 0 else np.nan

    # Most common rebound destination bin
    if len(reb_from) > 0:
        dest_counts = reb_from.groupby(["reb_x_bin", "reb_y_bin"]).size()
        top_dest    = dest_counts.idxmax()
        top_dest_str = f"({top_dest[0]},{top_dest[1]})"
    else:
        top_dest_str = ""

    corridor_rows.append({
        "orig_x_bin"       : xb,
        "orig_y_bin"       : yb,
        "orig_shots"       : shots,
        "rebound_sequences": reb_gen,
        "rebound_rate"     : round(reb_rate, 4) if not np.isnan(reb_rate) else "",
        "rebound_goals"    : reb_goals,
        "rebound_goal_rate": round(reb_goal_rate, 4) if not np.isnan(reb_goal_rate) else "",
        "top_rebound_dest" : top_dest_str,
    })

corridor_df = pd.DataFrame(corridor_rows).sort_values(
    ["orig_x_bin", "orig_y_bin"]
).reset_index(drop=True)

out_corr = os.path.join(DATA_DIR, "rebound_corridor_map.csv")
corridor_df.to_csv(out_corr, index=False)
print(f"  Bins computed: {len(corridor_df)}")
print(f"  Saved: {out_corr}")


# ─── STEP 3: PLAYER REBOUND POSITIONING ────────────────────────────────────────
print("\n── Step 3: Player rebound positioning ──")

player_rows_raw = []
reb_shooter_grps = seq_df.groupby("reb_shooter_id")

for pid, grp in reb_shooter_grps:
    total_reb   = len(grp)
    if total_reb < MIN_REB_SHOTS:
        continue
    reb_goals   = int(grp["reb_is_goal"].sum())
    _, gl_lo, gl_hi = wilson_ci(reb_goals, total_reb)
    reb_gr      = reb_goals / total_reb if total_reb > 0 else np.nan

    avg_x = grp["reb_x"].apply(to_float).mean()
    avg_y = grp["reb_y"].apply(to_float).mean()
    avg_gap = grp["time_gap_secs"].mean()

    player_rows_raw.append({
        "shooter_player_id"  : pid,
        "player_name"        : "",   # filled after API lookup
        "rebound_shots"      : total_reb,
        "rebound_goals"      : reb_goals,
        "rebound_goal_rate"  : round(reb_gr, 4),
        "reb_gr_ci_lo"       : round(gl_lo, 4),
        "reb_gr_ci_hi"       : round(gl_hi, 4),
        "avg_x_coord_norm"   : round(avg_x, 1) if not np.isnan(avg_x) else "",
        "avg_y_coord_norm"   : round(avg_y, 1) if not np.isnan(avg_y) else "",
        "avg_time_gap_secs"  : round(avg_gap, 2),
    })

print(f"  Players with >= {MIN_REB_SHOTS} rebound shots: {len(player_rows_raw)}")

# Fetch player names
unique_ids = [r["shooter_player_id"] for r in player_rows_raw]
print(f"  Fetching {len(unique_ids)} player names from NHL API...")
name_map = {}
for i, pid in enumerate(unique_ids):
    name_map[pid] = fetch_player_name(pid)
    if (i + 1) % 30 == 0:
        print(f"    {i+1}/{len(unique_ids)} fetched...")
    time.sleep(0.05)
print(f"  Done.")

for r in player_rows_raw:
    r["player_name"] = name_map.get(r["shooter_player_id"], f"ID:{r['shooter_player_id']}")

player_df = pd.DataFrame(player_rows_raw).sort_values(
    "rebound_goals", ascending=False
).reset_index(drop=True)

out_player = os.path.join(DATA_DIR, "player_rebound_positioning.csv")
player_df.to_csv(out_player, index=False)
print(f"  Saved: {out_player}")


# ─── TERMINAL SUMMARY ──────────────────────────────────────────────────────────
# Overall goal rates: rebound vs non-rebound
total_es_sog   = len(df_orig)
total_es_goals = int(df_orig["is_goal"].sum())
non_reb_goal_rate = total_es_goals / total_es_sog if total_es_sog > 0 else np.nan

reb_goal_count  = int(seq_df["reb_is_goal"].sum())
reb_goal_rate_overall = reb_goal_count / len(seq_df) if len(seq_df) > 0 else np.nan
multiplier = reb_goal_rate_overall / non_reb_goal_rate if non_reb_goal_rate else np.nan

# Top 5 bins by rebound goal rate (min 5 rebound sequences)
top_bins = corridor_df[
    (corridor_df["rebound_sequences"] >= 5) &
    (corridor_df["rebound_goal_rate"] != "")
].copy()
top_bins["reb_goal_rate_f"] = pd.to_numeric(top_bins["rebound_goal_rate"], errors="coerce")
top_bins = top_bins.nlargest(5, "reb_goal_rate_f")

print("\n" + "═" * 68)
print("  HOCKEYROI — REBOUND CORRIDOR MODEL SUMMARY")
print("═" * 68)
print(f"\n  Total ES rebound sequences found : {len(seq_df):,}")
print(f"  ES shots-on-goal + goals (base) : {total_es_sog:,}")
print(f"\n  GOAL RATES")
print(f"  Non-rebound (all ES sog+goal)    : {non_reb_goal_rate:.4f}  "
      f"({total_es_goals:,} goals / {total_es_sog:,} shots)")
print(f"  Rebound shot goal rate           : {reb_goal_rate_overall:.4f}  "
      f"({reb_goal_count:,} goals / {len(seq_df):,} rebound sequences)")
print(f"  Multiplier                       : {multiplier:.2f}x")

print(f"\n  TOP 15 PLAYERS BY REBOUND GOALS")
print(f"  {'Rk':<4}  {'Player':<24}  {'Reb G':>6}  {'Reb Sh':>7}  {'G Rate':>7}  "
      f"{'CI lo':>7}  {'CI hi':>7}  {'Avg X':>6}  {'Avg Y':>6}  {'Avg Gap':>8}")
print(f"  {'-'*88}")
for rank, row in player_df.head(15).iterrows():
    print(f"  {rank+1:<4}  {str(row['player_name']):<24}  {row['rebound_goals']:>6}  "
          f"{row['rebound_shots']:>7}  {row['rebound_goal_rate']:>7.4f}  "
          f"{row['reb_gr_ci_lo']:>7.4f}  {row['reb_gr_ci_hi']:>7.4f}  "
          f"{str(row['avg_x_coord_norm']):>6}  {str(row['avg_y_coord_norm']):>6}  "
          f"{row['avg_time_gap_secs']:>8.2f}s")

print(f"\n  TOP 5 ORIGINAL SHOT LOCATIONS BY REBOUND GOAL RATE (min 5 rebounds)")
print(f"  {'Orig bin (x,y)':<18}  {'Orig shots':>10}  {'Reb seqs':>9}  "
      f"{'Reb rate':>9}  {'Reb goals':>10}  {'Reb G rate':>10}  {'Top dest':>12}")
print(f"  {'-'*80}")
for _, row in top_bins.iterrows():
    print(f"  ({row['orig_x_bin']:>3},{row['orig_y_bin']:>4})          "
          f"  {row['orig_shots']:>10,}  {row['rebound_sequences']:>9}  "
          f"  {float(row['rebound_rate']):>8.4f}  {row['rebound_goals']:>10}  "
          f"  {float(row['rebound_goal_rate']):>9.4f}  {str(row['top_rebound_dest']):>12}")

print("\n" + "═" * 68)
print("  Output files written to Data/")
print("  - rebound_sequences.csv")
print("  - rebound_corridor_map.csv")
print("  - player_rebound_positioning.csv")
print("═" * 68 + "\n")
