#!/usr/bin/env python3
"""
HockeyROI - Elite Net-Front Player Ranking (Composite Score)
Save to: NHL analysis/elite_netfront_ranking.py

Usage:
  cd "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
  python3 elite_netfront_ranking.py

Sources:
  Data/player_rebound_positioning.csv  (from rebound_corridor.py)
  Data/nhl_shot_events.csv             (for current-team lookup)
  Data/rebound_sequences.csv           (for league avg rebound goal rate)

Composite score (players with ≥30 rebound shots in doorstep zone x≥74, -8≤y≤8):
  40% — rebound goal rate
  30% — avg time gap inverted  (faster = better, min-max normalized)
  30% — positioning score      (proximity to x=80, y=0, min-max normalized)

Output:
  Data/elite_netfront_players.csv
"""

import csv, math, os
from collections import defaultdict

DATA_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MIN_REBOUND_SHOTS = 30
DOORSTEP_X_MIN    = 74.0
DOORSTEP_Y_MIN    = -8.0
DOORSTEP_Y_MAX    =  8.0
IDEAL_X           = 80.0
IDEAL_Y           =  0.0

W_GOAL_RATE  = 0.40
W_TIME_GAP   = 0.30
W_POSITION   = 0.30

CURRENT_SEASON = "20242025"
OILERS_ABBREV  = "EDM"


# ─── STEP 1: CURRENT TEAM LOOKUP FROM SHOT DATA ────────────────────────────────
print("Building current-team lookup from 20242025 shot data...")
team_shot_counts = defaultdict(lambda: defaultdict(int))  # pid -> {team: count}

with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["season"] != CURRENT_SEASON:
            continue
        if row["game_type"] != "regular":
            continue
        pid  = row["shooter_player_id"]
        team = row["shooting_team_abbrev"]
        if pid and team:
            team_shot_counts[pid][team] += 1

# Most frequent team in 20242025 regular season for each player
current_team = {}
for pid, teams in team_shot_counts.items():
    current_team[pid] = max(teams, key=teams.get)

print(f"  {len(current_team):,} players with 20242025 regular season shots")


# ─── STEP 2: LEAGUE-WIDE REBOUND GOAL RATE ────────────────────────────────────
print("Computing league-wide rebound goal rate...")
total_reb_shots = 0
total_reb_goals = 0

with open(os.path.join(DATA_DIR, "rebound_sequences.csv"), newline="") as f:
    for row in csv.DictReader(f):
        total_reb_shots += 1
        if row["reb_is_goal"] == "1":
            total_reb_goals += 1

league_reb_gr = total_reb_goals / total_reb_shots if total_reb_shots else 0.0
print(f"  League avg rebound goal rate: {league_reb_gr:.4f}  "
      f"({total_reb_goals} goals / {total_reb_shots:,} sequences)")


# ─── STEP 3: LOAD & FILTER PLAYER DATA ────────────────────────────────────────
print("Loading player rebound positioning data...")
players = []

with open(os.path.join(DATA_DIR, "player_rebound_positioning.csv"), newline="") as f:
    for row in csv.DictReader(f):
        try:
            reb_shots = int(row["rebound_shots"])
            reb_goals = int(row["rebound_goals"])
            reb_gr    = float(row["rebound_goal_rate"])
            ci_lo     = float(row["reb_gr_ci_lo"])
            ci_hi     = float(row["reb_gr_ci_hi"])
            avg_x     = float(row["avg_x_coord_norm"])
            avg_y     = float(row["avg_y_coord_norm"])
            avg_gap   = float(row["avg_time_gap_secs"])
            pid       = str(row["shooter_player_id"])
            name      = row["player_name"]
        except (ValueError, KeyError):
            continue

        players.append({
            "player_id"        : pid,
            "player_name"      : name,
            "rebound_shots"    : reb_shots,
            "rebound_goals"    : reb_goals,
            "rebound_goal_rate": reb_gr,
            "reb_gr_ci_lo"     : ci_lo,
            "reb_gr_ci_hi"     : ci_hi,
            "avg_x"            : avg_x,
            "avg_y"            : avg_y,
            "avg_time_gap"     : avg_gap,
            "current_team"     : current_team.get(pid, "—"),
        })

print(f"  Total players in file: {len(players)}")

# Filter: minimum shots + doorstep zone
eligible = [
    p for p in players
    if p["rebound_shots"] >= MIN_REBOUND_SHOTS
    and p["avg_x"] >= DOORSTEP_X_MIN
    and DOORSTEP_Y_MIN <= p["avg_y"] <= DOORSTEP_Y_MAX
]
print(f"  After filter (≥{MIN_REBOUND_SHOTS} shots, x≥{DOORSTEP_X_MIN}, "
      f"{DOORSTEP_Y_MIN}≤y≤{DOORSTEP_Y_MAX}): {len(eligible)} players")


# ─── STEP 4: COMPOSITE SCORE ──────────────────────────────────────────────────
def minmax_norm(vals, invert=False):
    """Normalize a list of values to [0,1]. invert=True flips the scale."""
    mn, mx = min(vals), max(vals)
    rng = mx - mn
    if rng == 0:
        return [0.5] * len(vals)
    normed = [(v - mn) / rng for v in vals]
    if invert:
        normed = [1.0 - n for n in normed]
    return normed

# Compute raw positioning distance from (IDEAL_X, IDEAL_Y)
for p in eligible:
    p["pos_dist"] = math.sqrt((p["avg_x"] - IDEAL_X)**2 + (p["avg_y"] - IDEAL_Y)**2)

gr_vals   = [p["rebound_goal_rate"] for p in eligible]
gap_vals  = [p["avg_time_gap"]      for p in eligible]
dist_vals = [p["pos_dist"]          for p in eligible]

# Normalize: goal rate (higher=better), time gap (lower=better), dist (lower=better)
gr_norm   = minmax_norm(gr_vals,   invert=False)
gap_norm  = minmax_norm(gap_vals,  invert=True)   # invert: faster gap → higher score
dist_norm = minmax_norm(dist_vals, invert=True)   # invert: closer to ideal → higher score

for i, p in enumerate(eligible):
    p["norm_goal_rate"] = gr_norm[i]
    p["norm_time_gap"]  = gap_norm[i]
    p["norm_position"]  = dist_norm[i]
    p["composite"]      = round(
        W_GOAL_RATE * gr_norm[i] +
        W_TIME_GAP  * gap_norm[i] +
        W_POSITION  * dist_norm[i],
        4
    )
    p["ci_confirmed"]   = p["reb_gr_ci_lo"] > league_reb_gr
    p["is_oiler"]       = p["current_team"] == OILERS_ABBREV


# ─── STEP 5: RANK & SAVE ──────────────────────────────────────────────────────
ranked = sorted(eligible, key=lambda x: -x["composite"])

# Assign rank
for i, p in enumerate(ranked):
    p["rank"] = i + 1

# Save full list
out_cols = [
    "rank", "player_name", "player_id", "current_team",
    "rebound_shots", "rebound_goals", "rebound_goal_rate",
    "reb_gr_ci_lo", "reb_gr_ci_hi",
    "avg_x", "avg_y", "avg_time_gap", "pos_dist",
    "norm_goal_rate", "norm_time_gap", "norm_position",
    "composite", "ci_confirmed", "is_oiler",
]

out_path = os.path.join(DATA_DIR, "elite_netfront_players.csv")
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(ranked)
print(f"\nSaved: {out_path}  ({len(ranked)} players)")


# ─── STEP 6: TERMINAL SUMMARY ─────────────────────────────────────────────────
SEP = "═" * 102

print(f"\n{SEP}")
print("  HOCKEYROI — ELITE NET-FRONT PLAYER RANKING")
print(f"  Composite: {int(W_GOAL_RATE*100)}% goal rate  |  "
      f"{int(W_TIME_GAP*100)}% time gap (inverted)  |  "
      f"{int(W_POSITION*100)}% position (proximity to x=80, y=0)")
print(f"  Filter: ≥{MIN_REBOUND_SHOTS} rebound shots, avg_x≥{DOORSTEP_X_MIN}, "
      f"{DOORSTEP_Y_MIN}≤avg_y≤{DOORSTEP_Y_MAX}   |   "
      f"League avg rebound goal rate: {league_reb_gr:.4f}")
print(SEP)
print(f"  {'Rk':<4} {'Player':<22} {'Tm':<4} {'Reb':>5} {'Goals':>6} "
      f"{'G Rate':>7} {'CI lo':>7} {'CI hi':>7} "
      f"{'Avg X':>6} {'Avg Y':>6} {'Gap':>6} "
      f"{'Score':>7}  {'CI✓':>4}  {'OIL':>4}")
print(f"  {'-'*100}")

for p in ranked[:30]:
    ci_flag  = "✓" if p["ci_confirmed"] else ""
    oil_flag = "◄" if p["is_oiler"]    else ""
    print(f"  {p['rank']:<4} {p['player_name']:<22} {p['current_team']:<4} "
          f"{p['rebound_shots']:>5} {p['rebound_goals']:>6} "
          f"{p['rebound_goal_rate']:>7.4f} {p['reb_gr_ci_lo']:>7.4f} {p['reb_gr_ci_hi']:>7.4f} "
          f"{p['avg_x']:>6.1f} {p['avg_y']:>6.1f} {p['avg_time_gap']:>5.2f}s "
          f"{p['composite']:>7.4f}  {ci_flag:>4}  {oil_flag:>4}")

# Any Oilers outside top 30?
oiler_outside = [p for p in ranked[30:] if p["is_oiler"]]
if oiler_outside:
    print(f"\n  OILERS players outside top 30:")
    print(f"  {'-'*100}")
    for p in oiler_outside:
        ci_flag = "✓" if p["ci_confirmed"] else ""
        print(f"  {p['rank']:<4} {p['player_name']:<22} {p['current_team']:<4} "
              f"{p['rebound_shots']:>5} {p['rebound_goals']:>6} "
              f"{p['rebound_goal_rate']:>7.4f} {p['reb_gr_ci_lo']:>7.4f} {p['reb_gr_ci_hi']:>7.4f} "
              f"{p['avg_x']:>6.1f} {p['avg_y']:>6.1f} {p['avg_time_gap']:>5.2f}s "
              f"{p['composite']:>7.4f}  {ci_flag:>4}")

print(f"\n  CI✓ = CI lower bound > league avg ({league_reb_gr:.4f}) — statistically confirmed")
print(f"  ◄   = Current Oilers player (20242025 regular season)")
print(f"  Score components: norm values × weights, all in [0,1]")
print(SEP + "\n")
