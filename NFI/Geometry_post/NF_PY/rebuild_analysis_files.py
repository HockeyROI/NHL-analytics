#!/usr/bin/env python3
"""
HockeyROI — Rebuild three analysis files to include 20252026.

Step 1 — Append 20252026 rebound sequences to rebound_sequences.csv
Step 2 — Rebuild player_rebound_positioning.csv (all 6 seasons)
          current_team = primary team in 20252026 ES shots
Step 3 — Rebuild elite_netfront_players.csv with trend column
          trend = 20252026 NF rate vs prior-5-season avg (rising/declining/stable)

Outputs:
  Data/rebound_sequences.csv         (appended)
  Data/player_rebound_positioning.csv (full rebuild)
  Data/elite_netfront_players.csv     (full rebuild, adds trend column)
"""

import csv, math, os, time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

DATA_DIR    = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"
BASE_URL    = "https://api-web.nhle.com/v1"
REBOUND_GAP = 3
MIN_REB_SHOTS = 20
Z95           = 1.96
SLEEP_API     = 0.10

NEW_SEASON  = "20252026"
OLD_SEASONS = {"20202021","20212022","20222023","20232024","20242025"}

# Elite ranking filters / weights (identical to elite_netfront_ranking.py)
MIN_ELITE_SHOTS = 30
NF_X_MIN = 74.0;  NF_Y_MIN = -8.0;  NF_Y_MAX = 8.0
IDEAL_X  = 80.0;  IDEAL_Y  = 0.0
W_GOAL_RATE = 0.40;  W_TIME_GAP = 0.30;  W_POSITION = 0.30

TREND_THRESHOLD = 0.20   # ±20% change in NF rate vs prior avg → rising/declining

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "HockeyROI-Analysis/2.0"

SEQ_COLS = [
    "game_id","season","period","orig_event_type","orig_x","orig_y",
    "orig_shot_type","orig_team","orig_shooter_id",
    "reb_event_type","reb_x","reb_y","reb_shot_type",
    "reb_shooter_id","reb_is_goal","time_gap_secs",
]

def wilson_ci(k, n, z=Z95):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)

def fetch_player_name(pid):
    try:
        r = SESSION.get(f"{BASE_URL}/player/{int(pid)}/landing", timeout=12)
        if r.status_code == 200:
            d = r.json()
            fn = d.get("firstName",{}).get("default","")
            ln = d.get("lastName", {}).get("default","")
            return f"{fn} {ln}".strip()
    except Exception:
        pass
    return f"ID:{pid}"

def minmax_norm(vals, invert=False):
    mn, mx = min(vals), max(vals)
    rng = mx - mn
    if rng == 0:
        return [0.5] * len(vals)
    normed = [(v - mn) / rng for v in vals]
    return [1.0 - n for n in normed] if invert else normed


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — Load ES shot events (one full pass)
# ════════════════════════════════════════════════════════════════════════════
print("Phase 0: Loading nhl_shot_events.csv (ES only, situation_code=1551)...")

ES_CODE   = "1551"
ORIG_TYPES = {"shot-on-goal", "goal"}
REB_TYPES  = {"shot-on-goal", "goal", "missed-shot"}

# Accumulators for player-season NF stats (needed for trend + current_team)
# (pid, season) → {es_shots, nf_shots, team_counts}
p_es    = defaultdict(int)   # (pid, season)
p_nf    = defaultdict(int)   # (pid, season)
p_teams = defaultdict(lambda: defaultdict(dict))  # pid → season → {team: count}

# Per-game ES events for sequence detection (20252026 only)
new_rows_for_seq = []    # list of dicts for pd.DataFrame

# For league-wide rebound goal rate (all seasons, for elite ranking)
all_reb_shots = 0
all_reb_goals = 0

shot_file = os.path.join(DATA_DIR, "nhl_shot_events.csv")
total_loaded = 0

with open(shot_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["situation_code"] != ES_CODE:
            continue
        season = row["season"]
        pid    = row["shooter_player_id"]
        team   = row["shooting_team_abbrev"]

        try:
            x = float(row["x_coord_norm"])
            y = float(row["y_coord_norm"])
        except (ValueError, TypeError):
            x = y = float("nan")

        is_goal = row["is_goal"] == "1"
        is_nf   = (not math.isnan(x) and x >= NF_X_MIN and NF_Y_MIN <= y <= NF_Y_MAX)

        if pid and team:
            p_es[(pid, season)]  += 1
            if is_nf:
                p_nf[(pid, season)] += 1
            p_teams[pid][season][team] = p_teams[pid][season].get(team, 0) + 1

        # Collect rows for new season sequence detection
        if season == NEW_SEASON and row["event_type"] in REB_TYPES:
            try:
                ts = int(row["time_secs"]) if row["time_secs"] else None
            except ValueError:
                ts = None
            new_rows_for_seq.append({
                "game_id"              : row["game_id"],
                "season"               : season,
                "period"               : row["period"],
                "time_secs"            : ts,
                "event_type"           : row["event_type"],
                "shooting_team_abbrev" : team,
                "shooter_player_id"    : pid,
                "x_coord_norm"         : x if not math.isnan(x) else None,
                "y_coord_norm"         : y if not math.isnan(y) else None,
                "shot_type"            : row["shot_type"],
                "is_goal"              : 1 if is_goal else 0,
            })
        total_loaded += 1

print(f"  ES rows loaded: {total_loaded:,}  |  {NEW_SEASON} rows for seq detection: {len(new_rows_for_seq):,}")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Append 20252026 rebound sequences
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── Step 1: Appending {NEW_SEASON} rebound sequences ──")

seq_file = os.path.join(DATA_DIR, "rebound_sequences.csv")

# Find existing game_ids so we don't duplicate
existing_seq_gids = set()
if os.path.exists(seq_file):
    with open(seq_file, newline="") as f:
        for row in csv.DictReader(f):
            existing_seq_gids.add(str(row["game_id"]))
print(f"  Existing sequences file: {len(existing_seq_gids):,} unique game_ids already processed")

# Build DataFrame for new-season events
df_new = pd.DataFrame(new_rows_for_seq)
df_new["time_secs"] = pd.to_numeric(df_new["time_secs"], errors="coerce")
df_new["period"]    = pd.to_numeric(df_new["period"],    errors="coerce")
df_new["is_goal"]   = pd.to_numeric(df_new["is_goal"],   errors="coerce")

# Filter out game_ids already in file (resume-safe)
df_new = df_new[~df_new["game_id"].isin(existing_seq_gids)].copy()
new_game_ids = df_new["game_id"].nunique()
print(f"  New {NEW_SEASON} games to process for sequences: {new_game_ids:,}")

# Run sequence detection
df_sorted = df_new.sort_values(["game_id","period","time_secs"]).reset_index(drop=True)
new_sequences = []

for (gid, period), grp in df_sorted.groupby(["game_id","period"], sort=False):
    grp = grp.reset_index(drop=True)
    n   = len(grp)
    for i in range(n):
        if grp.at[i,"event_type"] not in ORIG_TYPES:
            continue
        if pd.isna(grp.at[i,"time_secs"]):
            continue
        t0   = grp.at[i,"time_secs"]
        team = grp.at[i,"shooting_team_abbrev"]
        for j in range(i+1, n):
            dt = grp.at[j,"time_secs"] - t0
            if pd.isna(dt) or dt > REBOUND_GAP:
                break
            if grp.at[j,"shooting_team_abbrev"] != team:
                continue
            new_sequences.append({
                "game_id"        : gid,
                "season"         : grp.at[i,"season"],
                "period"         : int(period),
                "orig_event_type": grp.at[i,"event_type"],
                "orig_x"         : grp.at[i,"x_coord_norm"],
                "orig_y"         : grp.at[i,"y_coord_norm"],
                "orig_shot_type" : grp.at[i,"shot_type"],
                "orig_team"      : team,
                "orig_shooter_id": grp.at[i,"shooter_player_id"],
                "reb_event_type" : grp.at[j,"event_type"],
                "reb_x"          : grp.at[j,"x_coord_norm"],
                "reb_y"          : grp.at[j,"y_coord_norm"],
                "reb_shot_type"  : grp.at[j,"shot_type"],
                "reb_shooter_id" : grp.at[j,"shooter_player_id"],
                "reb_is_goal"    : int(grp.at[j,"is_goal"]) if not pd.isna(grp.at[j,"is_goal"]) else 0,
                "time_gap_secs"  : round(float(dt), 1),
            })
            break

print(f"  New sequences detected: {len(new_sequences):,}")

# Append to file
file_is_new = not os.path.exists(seq_file) or os.path.getsize(seq_file) == 0
with open(seq_file, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=SEQ_COLS)
    if file_is_new:
        w.writeheader()
    w.writerows(new_sequences)

# Count total sequences in file
total_seqs = 0
with open(seq_file, newline="") as f:
    total_seqs = sum(1 for _ in csv.DictReader(f))
print(f"  Total sequences in file: {total_seqs:,}  (was {total_seqs - len(new_sequences):,})")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Rebuild player_rebound_positioning.csv (all 6 seasons)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── Step 2: Rebuilding player_rebound_positioning.csv (all 6 seasons) ──")

# Load full rebound_sequences.csv
print("  Loading full rebound_sequences.csv...")
seq_all = []
with open(seq_file, newline="") as f:
    for row in csv.DictReader(f):
        seq_all.append(row)

seq_df = pd.DataFrame(seq_all)
seq_df["reb_is_goal"]   = pd.to_numeric(seq_df["reb_is_goal"],   errors="coerce").fillna(0)
seq_df["time_gap_secs"] = pd.to_numeric(seq_df["time_gap_secs"], errors="coerce")
seq_df["reb_x"]         = pd.to_numeric(seq_df["reb_x"],         errors="coerce")
seq_df["reb_y"]         = pd.to_numeric(seq_df["reb_y"],         errors="coerce")
print(f"  Total sequences: {len(seq_df):,}")

# League-wide rebound goal rate (for CI-confirmed flag in elite ranking)
all_reb_shots = len(seq_df)
all_reb_goals = int(seq_df["reb_is_goal"].sum())
league_reb_gr = all_reb_goals / all_reb_shots if all_reb_shots else 0.0
print(f"  League rebound goal rate: {league_reb_gr:.4f}  ({all_reb_goals}/{all_reb_shots:,})")

# Current team: primary team in 20252026 ES shots
current_team = {}   # pid → team in 20252026
for pid, season_teams in p_teams.items():
    teams_2526 = season_teams.get(NEW_SEASON, {})
    if teams_2526:
        current_team[pid] = max(teams_2526, key=teams_2526.get)

# Existing name lookup (avoid re-hitting API for known players)
existing_names = {}
pos_file = os.path.join(DATA_DIR, "player_rebound_positioning.csv")
if os.path.exists(pos_file):
    with open(pos_file, newline="") as f:
        for row in csv.DictReader(f):
            pid = str(row["shooter_player_id"])
            if not pid.startswith("ID:") and row["player_name"] and not row["player_name"].startswith("ID:"):
                existing_names[pid] = row["player_name"]
print(f"  Names cached from existing file: {len(existing_names):,}")

# Build player stats
player_raw = []
for pid, grp in seq_df.groupby("reb_shooter_id"):
    total_reb = len(grp)
    if total_reb < MIN_REB_SHOTS:
        continue
    reb_goals = int(grp["reb_is_goal"].sum())
    gr, ci_lo, ci_hi = wilson_ci(reb_goals, total_reb)
    avg_x   = float(grp["reb_x"].mean()) if not grp["reb_x"].isna().all() else float("nan")
    avg_y   = float(grp["reb_y"].mean()) if not grp["reb_y"].isna().all() else float("nan")
    avg_gap = float(grp["time_gap_secs"].mean())
    player_raw.append({
        "shooter_player_id" : str(pid),
        "player_name"       : "",
        "rebound_shots"     : total_reb,
        "rebound_goals"     : reb_goals,
        "rebound_goal_rate" : round(gr, 4),
        "reb_gr_ci_lo"      : round(ci_lo, 4),
        "reb_gr_ci_hi"      : round(ci_hi, 4),
        "avg_x_coord_norm"  : round(avg_x, 1) if not math.isnan(avg_x) else "",
        "avg_y_coord_norm"  : round(avg_y, 1) if not math.isnan(avg_y) else "",
        "avg_time_gap_secs" : round(avg_gap, 2),
        "current_team"      : current_team.get(str(pid), "—"),
    })

print(f"  Players with ≥{MIN_REB_SHOTS} rebound shots: {len(player_raw):,}")

# Fetch names for new players only
new_pids = [r["shooter_player_id"] for r in player_raw
            if r["shooter_player_id"] not in existing_names]
print(f"  Fetching {len(new_pids)} new player names from NHL API...")
for i, pid in enumerate(new_pids):
    existing_names[pid] = fetch_player_name(pid)
    if (i+1) % 20 == 0:
        print(f"    {i+1}/{len(new_pids)} fetched...", flush=True)
    time.sleep(SLEEP_API)
if new_pids:
    print(f"  Done fetching names.")

for r in player_raw:
    r["player_name"] = existing_names.get(r["shooter_player_id"], f"ID:{r['shooter_player_id']}")

# Sort by rebound goals desc, save
player_raw.sort(key=lambda r: -r["rebound_goals"])
pos_cols = [
    "shooter_player_id","player_name","rebound_shots","rebound_goals",
    "rebound_goal_rate","reb_gr_ci_lo","reb_gr_ci_hi",
    "avg_x_coord_norm","avg_y_coord_norm","avg_time_gap_secs","current_team"
]
with open(pos_file, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=pos_cols)
    w.writeheader()
    w.writerows(player_raw)
print(f"  Saved: {pos_file}  ({len(player_raw)} players)")


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Rebuild elite_netfront_players.csv with trend
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── Step 3: Rebuilding elite_netfront_players.csv ──")

# Compute per-player per-season NF attempt rate (for trend)
# prior avg = mean NF rate over OLD_SEASONS they played in
# current   = NF rate in 20252026

def nf_rate(pid, season):
    key = (pid, season)
    es = p_es.get(key, 0)
    nf = p_nf.get(key, 0)
    return nf / es if es >= 10 else None   # ≥10 ES shots to count a season

def compute_trend(pid):
    prior_rates = [nf_rate(pid, s) for s in OLD_SEASONS if nf_rate(pid, s) is not None]
    curr        = nf_rate(pid, NEW_SEASON)
    if curr is None or not prior_rates:
        return "stable"    # not enough data → call stable
    prior_avg = sum(prior_rates) / len(prior_rates)
    if prior_avg == 0:
        return "stable"
    change = (curr - prior_avg) / prior_avg
    if change > TREND_THRESHOLD:
        return "rising"
    if change < -TREND_THRESHOLD:
        return "declining"
    return "stable"

# Filter eligible players (doorstep zone, ≥30 rebound shots)
eligible = []
for r in player_raw:
    try:
        reb_shots = int(r["rebound_shots"])
        reb_goals = int(r["rebound_goals"])
        reb_gr    = float(r["rebound_goal_rate"])
        ci_lo     = float(r["reb_gr_ci_lo"])
        ci_hi     = float(r["reb_gr_ci_hi"])
        avg_x     = float(r["avg_x_coord_norm"]) if r["avg_x_coord_norm"] != "" else float("nan")
        avg_y     = float(r["avg_y_coord_norm"]) if r["avg_y_coord_norm"] != "" else float("nan")
        avg_gap   = float(r["avg_time_gap_secs"])
        pid       = r["shooter_player_id"]
    except (ValueError, KeyError):
        continue
    if math.isnan(avg_x) or math.isnan(avg_y):
        continue
    if reb_shots < MIN_ELITE_SHOTS:
        continue
    if avg_x < NF_X_MIN or not (NF_Y_MIN <= avg_y <= NF_Y_MAX):
        continue
    eligible.append({
        "player_id"         : pid,
        "player_name"       : r["player_name"],
        "current_team"      : r["current_team"],
        "rebound_shots"     : reb_shots,
        "rebound_goals"     : reb_goals,
        "rebound_goal_rate" : reb_gr,
        "reb_gr_ci_lo"      : ci_lo,
        "reb_gr_ci_hi"      : ci_hi,
        "avg_x"             : avg_x,
        "avg_y"             : avg_y,
        "avg_time_gap"      : avg_gap,
        "trend"             : compute_trend(pid),
    })

print(f"  Eligible (≥{MIN_ELITE_SHOTS} shots, doorstep zone): {len(eligible)}")

# Composite score
for p in eligible:
    p["pos_dist"] = math.sqrt((p["avg_x"] - IDEAL_X)**2 + (p["avg_y"] - IDEAL_Y)**2)

gr_norm   = minmax_norm([p["rebound_goal_rate"] for p in eligible], invert=False)
gap_norm  = minmax_norm([p["avg_time_gap"]      for p in eligible], invert=True)
dist_norm = minmax_norm([p["pos_dist"]          for p in eligible], invert=True)

for i, p in enumerate(eligible):
    p["norm_goal_rate"] = gr_norm[i]
    p["norm_time_gap"]  = gap_norm[i]
    p["norm_position"]  = dist_norm[i]
    p["composite"]      = round(
        W_GOAL_RATE * gr_norm[i] + W_TIME_GAP * gap_norm[i] + W_POSITION * dist_norm[i], 4)
    p["ci_confirmed"]   = p["reb_gr_ci_lo"] > league_reb_gr

ranked = sorted(eligible, key=lambda x: -x["composite"])
for i, p in enumerate(ranked):
    p["rank"] = i + 1

elite_file = os.path.join(DATA_DIR, "elite_netfront_players.csv")
elite_cols = [
    "rank","player_name","player_id","current_team",
    "rebound_shots","rebound_goals","rebound_goal_rate",
    "reb_gr_ci_lo","reb_gr_ci_hi",
    "avg_x","avg_y","avg_time_gap","pos_dist",
    "norm_goal_rate","norm_time_gap","norm_position",
    "composite","ci_confirmed","trend",
]
with open(elite_file, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=elite_cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(ranked)
print(f"  Saved: {elite_file}  ({len(ranked)} players)")


# ════════════════════════════════════════════════════════════════════════════
#  TERMINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
ci_confirmed = [p for p in ranked if p["ci_confirmed"]]
trend_counts = {"rising":0, "declining":0, "stable":0}
for p in ranked:
    trend_counts[p["trend"]] = trend_counts.get(p["trend"], 0) + 1

SEP = "═" * 108
print(f"\n{SEP}")
print("  HOCKEYROI — REBUILT ANALYSIS FILES  (6 seasons: 20202021–20252026)")
print(SEP)
print(f"  rebound_sequences.csv:          {total_seqs:,} sequences  "
      f"(+{len(new_sequences):,} from {NEW_SEASON})")
print(f"  player_rebound_positioning.csv: {len(player_raw):,} players  "
      f"(≥{MIN_REB_SHOTS} rebound shots)")
print(f"  elite_netfront_players.csv:     {len(ranked):,} doorstep-zone players  "
      f"(≥{MIN_ELITE_SHOTS} shots)")
print(f"  CI-confirmed elite:             {len(ci_confirmed)}")
print(f"  Trend distribution:             rising={trend_counts['rising']}  "
      f"stable={trend_counts['stable']}  declining={trend_counts['declining']}")
print(f"  League rebound goal rate:       {league_reb_gr:.4f}")

TREND_ICON = {"rising": "↑", "declining": "↓", "stable": "—"}

print(f"\n  TOP 20 — COMPOSITE NET-FRONT RANKING  (current_team from {NEW_SEASON})")
print(f"  {'Rk':<4} {'Player':<22} {'Tm':<5} {'Reb':>5} {'Goals':>6} "
      f"{'G Rate':>7} {'CI lo':>7} {'CI hi':>7} "
      f"{'Avg X':>6} {'Avg Y':>6} {'Gap':>5} "
      f"{'Score':>7}  {'CI':>3}  {'Trend':>8}")
print(f"  {'-'*106}")
for p in ranked[:20]:
    ci_flag = "✓" if p["ci_confirmed"] else " "
    tr_icon = TREND_ICON.get(p["trend"], "—")
    print(f"  {p['rank']:<4} {p['player_name']:<22} {p['current_team']:<5} "
          f"{p['rebound_shots']:>5} {p['rebound_goals']:>6} "
          f"{p['rebound_goal_rate']:>7.4f} {p['reb_gr_ci_lo']:>7.4f} {p['reb_gr_ci_hi']:>7.4f} "
          f"{p['avg_x']:>6.1f} {p['avg_y']:>6.1f} {p['avg_time_gap']:>4.2f}s "
          f"{p['composite']:>7.4f}  {ci_flag:>3}  {tr_icon} {p['trend']}")

# Trend movers of note
rising   = sorted([p for p in ranked if p["trend"] == "rising"],   key=lambda x: x["composite"])
declining= sorted([p for p in ranked if p["trend"] == "declining"], key=lambda x: -x["composite"])

if rising:
    print(f"\n  ↑ RISING TREND (NF rate up >20% vs prior 5-season avg):")
    for p in rising:
        pid = p["player_id"]
        prior_rates = [nf_rate(pid, s) for s in OLD_SEASONS if nf_rate(pid, s) is not None]
        curr_r = nf_rate(pid, NEW_SEASON)
        prior_avg = sum(prior_rates)/len(prior_rates) if prior_rates else 0
        print(f"    {p['player_name']:<22} {p['current_team']:<5} "
              f"composite={p['composite']:.4f}  prior_avg={prior_avg:.3f}  "
              f"current={curr_r:.3f}  Δ={((curr_r-prior_avg)/prior_avg*100) if prior_avg else 0:+.0f}%")

if declining:
    print(f"\n  ↓ DECLINING TREND (NF rate down >20% vs prior 5-season avg):")
    for p in declining:
        pid = p["player_id"]
        prior_rates = [nf_rate(pid, s) for s in OLD_SEASONS if nf_rate(pid, s) is not None]
        curr_r = nf_rate(pid, NEW_SEASON)
        prior_avg = sum(prior_rates)/len(prior_rates) if prior_rates else 0
        print(f"    {p['player_name']:<22} {p['current_team']:<5} "
              f"composite={p['composite']:.4f}  prior_avg={prior_avg:.3f}  "
              f"current={curr_r:.3f}  Δ={((curr_r-prior_avg)/prior_avg*100) if prior_avg else 0:+.0f}%")

print(f"\n  Trend key: ↑ rising = NF rate up >20% vs prior 5yr avg  |  "
      f"↓ declining = down >20%  |  — stable = within ±20%")
print(f"  CI✓ = Wilson 95% CI lower bound > league avg ({league_reb_gr:.4f})")
print(SEP + "\n")
