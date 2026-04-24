#!/usr/bin/env python3
"""
HockeyROI - Team Net-Front Roster Analysis
Save to: NHL analysis/team_netfront_roster.py

Usage:
  cd "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
  python3 team_netfront_roster.py

Elite = CI-confirmed AND avg_x>=74 AND -8<=avg_y<=8 AND >=30 rebound shots
(all three criteria are already baked into elite_netfront_players.csv filter;
 ci_confirmed column distinguishes confirmed from borderline)
"""

import csv, math, os
from collections import defaultdict
from scipy import stats

DATA_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"
SEASONS  = ["20202021","20212022","20222023","20232024","20242025"]

# ─── LOAD ELITE PLAYERS ────────────────────────────────────────────────────────
print("Loading elite net-front players...")
elite_ids = set()
elite_info = {}   # player_id -> row dict
with open(os.path.join(DATA_DIR, "elite_netfront_players.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["ci_confirmed"] == "True":
            pid = row["player_id"]
            elite_ids.add(pid)
            elite_info[pid] = row
print(f"  CI-confirmed elite players: {len(elite_ids)}")
for pid, r in sorted(elite_info.items(), key=lambda x: int(x[1]["rank"])):
    print(f"    #{r['rank']:<3} {r['player_name']:<24} gr={r['rebound_goal_rate']}  "
          f"ci_lo={r['reb_gr_ci_lo']}  shots={r['rebound_shots']}")

# ─── LOAD STANDINGS ────────────────────────────────────────────────────────────
print("\nLoading standings...")
standings = {}   # (season, team) -> {points_pct, points, gp, ...}
with open(os.path.join(DATA_DIR, "standings_5seasons.csv"), newline="") as f:
    for row in csv.DictReader(f):
        standings[(row["season"], row["team"])] = row

# ─── BUILD PLAYOFF DEPTH PER TEAM-SEASON ──────────────────────────────────────
print("Computing playoff depth per team-season...")
DEPTH_LABEL = {1: "R1_exit", 2: "R2_exit", 3: "CF_exit", 4: "Final_loss"}

# Load playoff game_ids
playoff_meta = {}
with open(os.path.join(DATA_DIR, "game_ids.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["game_type"] == "playoff":
            playoff_meta[row["game_id"]] = {
                "season": row["season"],
                "home": row["home_abbrev"],
                "away": row["away_abbrev"],
            }

# Game winners from shot data
game_goals = defaultdict(lambda: defaultdict(int))
with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["game_type"] == "playoff" and row["is_goal"] == "1":
            game_goals[row["game_id"]][row["shooting_team_abbrev"]] += 1

game_winner = {gid: max(tg, key=tg.get) for gid, tg in game_goals.items() if tg}

series_wins = defaultdict(lambda: defaultdict(int))
for gid, winner in game_winner.items():
    if gid not in playoff_meta: continue
    season, rnd, ser = playoff_meta[gid]["season"], int(gid[6:8]), gid[8]
    series_wins[(season, rnd, ser)][winner] += 1

series_winner_map = {k: max(v, key=v.get) for k, v in series_wins.items()}

team_max_round  = defaultdict(int)
team_rounds_won = defaultdict(set)

for gid, meta in playoff_meta.items():
    season, rnd = meta["season"], int(gid[6:8])
    for tm in [meta["home"], meta["away"]]:
        key = (season, tm)
        team_max_round[key] = max(team_max_round.get(key, 0), rnd)

for (season, rnd, ser), winner in series_winner_map.items():
    team_rounds_won[(season, winner)].add(rnd)

team_depth = {}   # (season, team) -> depth label
for (season, team), max_rnd in team_max_round.items():
    won = team_rounds_won.get((season, team), set())
    if max_rnd in won and max_rnd == 4:
        team_depth[(season, team)] = "Cup_winner"
    elif max_rnd not in won:
        team_depth[(season, team)] = DEPTH_LABEL.get(max_rnd, f"R{max_rnd}_exit")
    else:
        team_depth[(season, team)] = DEPTH_LABEL.get(max_rnd + 1, f"R{max_rnd+1}_exit")

# Teams that missed playoffs: in standings but not in team_depth
for (season, team) in standings:
    if (season, team) not in team_depth:
        team_depth[(season, team)] = "missed_playoffs"

# ─── MAP ELITE PLAYERS TO TEAMS PER SEASON ────────────────────────────────────
print("Mapping elite players to teams per season from shot data...")

# For every elite player, count ES shots per team per season (regular only)
player_team_shots = defaultdict(lambda: defaultdict(int))
# (player_id, season) -> {team: shot_count}

with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["situation_code"] != "1551": continue
        if row["game_type"] != "regular":   continue
        pid = row["shooter_player_id"]
        if pid not in elite_ids:            continue
        player_team_shots[(pid, row["season"])][row["shooting_team_abbrev"]] += 1

# Primary team = team with most ES shots that season
elite_player_season_team = {}   # (pid, season) -> team
for (pid, season), team_counts in player_team_shots.items():
    elite_player_season_team[(pid, season)] = max(team_counts, key=team_counts.get)

# ─── COUNT ELITE PLAYERS PER TEAM-SEASON ──────────────────────────────────────
print("Counting elite net-front players per team-season...")

team_season_elite = defaultdict(list)   # (season, team) -> [player_ids]
for (pid, season), team in elite_player_season_team.items():
    team_season_elite[(season, team)].append(pid)

# ─── REBOUND STATS PER PLAYER PER SEASON (regular season) ─────────────────────
print("Aggregating rebound stats per player per season...")

# Tag rebound shots: keys built from rebound_sequences.csv (regular season only)
reb_key_data = {}   # (game_id, period, shooter_id, x, y) -> {is_goal, season, team}
with open(os.path.join(DATA_DIR, "rebound_sequences.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["game_id"][4:6] != "02": continue   # regular season only
        key = (row["game_id"], row["period"], row["reb_shooter_id"], row["reb_x"], row["reb_y"])
        reb_key_data[key] = {
            "is_goal": row["reb_is_goal"],
            "season" : row["season"],
            "team"   : row["orig_team"],
        }

# Per team-season: combined rebound shots/goals from elite players
team_season_reb = defaultdict(lambda: {"reb_shots": 0, "reb_goals": 0})
with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["situation_code"] != "1551": continue
        if row["game_type"] != "regular":   continue
        pid = row["shooter_player_id"]
        if pid not in elite_ids:            continue
        rkey = (row["game_id"], row["period"], pid, row["x_coord_norm"], row["y_coord_norm"])
        if rkey not in reb_key_data:        continue
        season = row["season"]
        team   = elite_player_season_team.get((pid, season))
        if not team: continue
        team_season_reb[(season, team)]["reb_shots"] += 1
        if row["is_goal"] == "1":
            team_season_reb[(season, team)]["reb_goals"] += 1

# ─── BUILD MAIN RESULTS TABLE ─────────────────────────────────────────────────
result_rows = []
for (season, team), std in standings.items():
    elite_pids   = team_season_elite.get((season, team), [])
    elite_names  = [elite_info[p]["player_name"] for p in elite_pids]
    reb          = team_season_reb.get((season, team), {"reb_shots": 0, "reb_goals": 0})
    depth        = team_depth.get((season, team), "missed_playoffs")
    pts_pct      = float(std["points_pct"]) if std["points_pct"] else float("nan")

    result_rows.append({
        "season"            : season,
        "team"              : team,
        "n_elite_nf"        : len(elite_pids),
        "elite_player_names": "; ".join(sorted(elite_names)),
        "elite_reb_shots"   : reb["reb_shots"],
        "elite_reb_goals"   : reb["reb_goals"],
        "points_pct"        : round(pts_pct, 4),
        "points"            : std["points"],
        "gp"                : std["gp"],
        "playoff_depth"     : depth,
    })

result_rows.sort(key=lambda x: (-x["n_elite_nf"], -x["points_pct"]))

out_path = os.path.join(DATA_DIR, "team_netfront_roster_analysis.csv")
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
    w.writeheader()
    w.writerows(result_rows)
print(f"  Saved: {out_path}  ({len(result_rows)} team-seasons)")

# ─── STEP 5: PLAYOFF PERFORMANCE — ELITE vs NON-ELITE ─────────────────────────
print("Computing playoff rebound performance (elite vs non-elite)...")

# Identify all NF players in doorstep zone (not just CI-confirmed) from elite file
doorstep_ids = set()
with open(os.path.join(DATA_DIR, "elite_netfront_players.csv"), newline="") as f:
    for row in csv.DictReader(f):
        doorstep_ids.add(row["player_id"])   # all 299 passed position filter

# Build rebound shot sets for regular season vs playoff
def collect_reb_stats(game_type_code):
    """Return {player_id: {shots, goals}} for given game_type ('02'=reg, '03'=playoff)."""
    # Load relevant reb sequences
    reb_keys = {}
    with open(os.path.join(DATA_DIR, "rebound_sequences.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["game_id"][4:6] != game_type_code: continue
            key = (row["game_id"], row["period"], row["reb_shooter_id"], row["reb_x"], row["reb_y"])
            reb_keys[key] = row["reb_is_goal"]

    stats = defaultdict(lambda: {"shots": 0, "goals": 0})
    with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["situation_code"] != "1551": continue
            if row["game_type"] != ("regular" if game_type_code == "02" else "playoff"): continue
            pid  = row["shooter_player_id"]
            rkey = (row["game_id"], row["period"], pid, row["x_coord_norm"], row["y_coord_norm"])
            if rkey not in reb_keys: continue
            stats[pid]["shots"] += 1
            if row["is_goal"] == "1":
                stats[pid]["goals"] += 1
    return stats

reg_stats    = collect_reb_stats("02")
playoff_stats = collect_reb_stats("03")

def pool_stats(pid_set, stats_dict):
    shots = sum(stats_dict[p]["shots"]  for p in pid_set if p in stats_dict)
    goals = sum(stats_dict[p]["goals"] for p in pid_set if p in stats_dict)
    return shots, goals, goals/shots if shots else float("nan")

# All doorstep players (299): elite (22) vs non-elite-doorstep (277)
non_elite_door = doorstep_ids - elite_ids

elite_reg_sh, elite_reg_g, elite_reg_gr    = pool_stats(elite_ids, reg_stats)
elite_ply_sh, elite_ply_g, elite_ply_gr    = pool_stats(elite_ids, playoff_stats)
nonelite_reg_sh, nonelite_reg_g, ne_reg_gr = pool_stats(non_elite_door, reg_stats)
nonelite_ply_sh, nonelite_ply_g, ne_ply_gr = pool_stats(non_elite_door, playoff_stats)

# ─── TERMINAL SUMMARY ─────────────────────────────────────────────────────────
SEP  = "═" * 78
sep2 = "─" * 78

print(f"\n{SEP}")
print("  HOCKEYROI — TEAM NET-FRONT ROSTER ANALYSIS")
print(SEP)

# Distribution of elite players per team-season
dist = defaultdict(int)
for r in result_rows: dist[r["n_elite_nf"]] += 1
print(f"\n  DISTRIBUTION: elite net-front players per team-season  (n=159)")
for k in sorted(dist): print(f"    {k} elite players: {dist[k]:>3} team-seasons")

# Correlation: n_elite vs points_pct
xs = [r["n_elite_nf"] for r in result_rows]
ys = [r["points_pct"] for r in result_rows]
r_val, p_val = stats.pearsonr(xs, ys)
sig = "YES ***" if p_val < 0.05 else "no"
print(f"\n  Pearson r (n_elite vs points_pct): {r_val:.4f}  p={p_val:.5f}  {sig}")

# Step 4: avg elite count by playoff depth
DEPTH_ORDER = ["missed_playoffs","R1_exit","R2_exit","CF_exit","Final_loss","Cup_winner"]
print(f"\n  STEP 4 — AVG ELITE NET-FRONT PLAYERS BY PLAYOFF OUTCOME")
print(f"  {'Exit stage':<18}  {'N':>5}  {'Avg Elite NF':>13}  {'Avg Pts%':>10}  {'Teams (elite>0)':}")
print(f"  {sep2}")
for depth in DEPTH_ORDER:
    sub = [r for r in result_rows if r["playoff_depth"] == depth]
    if not sub: continue
    avg_elite = sum(r["n_elite_nf"] for r in sub) / len(sub)
    avg_pts   = sum(r["points_pct"] for r in sub) / len(sub)
    with_elite = [r["team"] for r in sub if r["n_elite_nf"] > 0]
    print(f"  {depth:<18}  {len(sub):>5}  {avg_elite:>13.3f}  {avg_pts*100:>9.1f}%  "
          f"{', '.join(sorted(set(with_elite)))[:55]}")

# Teams with 3+ elite NF players in a single season
print(f"\n  TEAMS WITH 3+ ELITE NET-FRONT PLAYERS IN A SINGLE SEASON")
top_teams = [r for r in result_rows if r["n_elite_nf"] >= 3]
if top_teams:
    print(f"  {'Season':<10} {'Team':<6} {'N':>3}  {'Pts%':>6}  {'Depth':<18}  Players")
    print(f"  {sep2}")
    for r in sorted(top_teams, key=lambda x: -x["n_elite_nf"]):
        print(f"  {r['season']:<10} {r['team']:<6} {r['n_elite_nf']:>3}  "
              f"{r['points_pct']*100:>5.1f}%  {r['playoff_depth']:<18}  {r['elite_player_names']}")
else:
    print("  None found.")

# Teams with exactly 2
two_elite = sorted([r for r in result_rows if r["n_elite_nf"] == 2],
                    key=lambda x: -x["points_pct"])
print(f"\n  TEAMS WITH 2 ELITE NET-FRONT PLAYERS ({len(two_elite)} team-seasons)")
print(f"  {'Season':<10} {'Team':<6} {'Pts%':>6}  {'Depth':<18}  Players")
print(f"  {sep2}")
for r in two_elite:
    print(f"  {r['season']:<10} {r['team']:<6} {r['points_pct']*100:>5.1f}%  "
          f"{r['playoff_depth']:<18}  {r['elite_player_names']}")

# Step 5: Playoff vs regular season performance
print(f"\n  STEP 5 — PLAYOFF vs REGULAR SEASON REBOUND PERFORMANCE")
print(f"  {'Group':<30}  {'Reg shots':>10}  {'Reg gr':>8}  {'Ply shots':>10}  {'Ply gr':>8}  {'Delta':>8}")
print(f"  {sep2}")
rows_p5 = [
    ("Elite NF (n=22)",     elite_reg_sh, elite_reg_g, elite_reg_gr, elite_ply_sh, elite_ply_g, elite_ply_gr),
    ("Non-elite doorstep",  nonelite_reg_sh, nonelite_reg_g, ne_reg_gr, nonelite_ply_sh, nonelite_ply_g, ne_ply_gr),
]
for label, r_sh, r_g, r_gr, p_sh, p_g, p_gr in rows_p5:
    delta = p_gr - r_gr if p_gr == p_gr and r_gr == r_gr else float("nan")
    delta_str = f"{delta:+.4f}" if delta == delta else "n/a"
    print(f"  {label:<30}  {r_sh:>10,}  {r_gr:>8.4f}  {p_sh:>10,}  {p_gr:>8.4f}  {delta_str:>8}")

elite_gap_reg = elite_reg_gr - ne_reg_gr
elite_gap_ply = elite_ply_gr - ne_ply_gr
print(f"\n  Gap (elite - non-elite):  regular season = {elite_gap_reg:+.4f}   playoffs = {elite_gap_ply:+.4f}")
print(f"  Gap {'WIDENS' if abs(elite_gap_ply) > abs(elite_gap_reg) else 'NARROWS/HOLDS'} in playoffs")

print(f"\n{SEP}")
print(f"  Saved: Data/team_netfront_roster_analysis.csv")
print(SEP + "\n")
