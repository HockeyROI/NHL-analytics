#!/usr/bin/env python3
"""
HockeyROI - Net-Front Efficiency vs Winning Correlation
Save to: NHL analysis/netfront_wins_correlation.py

Usage:
  cd "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
  python3 netfront_wins_correlation.py

Sources:
  Data/nhl_shot_events.csv   (ES, regular season only)
  Data/rebound_sequences.csv (for rebound-shot tagging)
  NHL standings API          (end-of-season dates for 5 seasons)

Outputs (Data/):
  standings_5seasons.csv
  team_netfront_metrics.csv
  netfront_wins_correlation.csv
  playoff_depth_nf_rates.csv
"""

import csv, math, os, time
from collections import defaultdict
from scipy import stats
import requests

DATA_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"
BASE_URL = "https://api-web.nhle.com/v1"

NF_SHOT_TYPES = {"tip-in", "deflected", "bat"}
ALL_ATTEMPTS  = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}

# Season-end dates for final standings
STANDINGS_DATES = {
    "20202021": "2021-05-19",
    "20212022": "2022-05-01",
    "20222023": "2023-04-13",
    "20232024": "2024-04-18",
    "20242025": "2025-04-17",
}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "HockeyROI-Analysis/2.0"


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def fetch(url, timeout=12):
    try:
        r = SESSION.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    time.sleep(1.5)
    try:
        r = SESSION.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def abbrev(d):
    """Extract string from NHL API abbrev/name dicts like {'default': 'EDM'}."""
    if isinstance(d, dict):
        return d.get("default", "")
    return str(d)


def pearson(xs, ys, label):
    """Return Pearson r, p-value, n, significance flag."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x == x and y == y and x is not None and y is not None]
    if len(pairs) < 5:
        return float("nan"), float("nan"), 0, False
    xv, yv = zip(*pairs)
    r, p = stats.pearsonr(xv, yv)
    return r, p, len(pairs), p < 0.05


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — STANDINGS
# ════════════════════════════════════════════════════════════════════════════════
print("── Step 1: Fetching standings ──")
standings_rows = []   # list of dicts

for season, date in STANDINGS_DATES.items():
    data = fetch(f"{BASE_URL}/standings/{date}")
    if not data:
        print(f"  ERROR: Could not fetch standings for {season}")
        continue
    teams = data.get("standings", [])
    for t in teams:
        team_abbrev = abbrev(t.get("teamAbbrev", {}))
        gp     = t.get("gamesPlayed", 0)
        pts    = t.get("points", 0)
        pts_pct = pts / (gp * 2) if gp > 0 else 0
        standings_rows.append({
            "season"    : season,
            "team"      : team_abbrev,
            "gp"        : gp,
            "points"    : pts,
            "points_pct": round(pts_pct, 4),
            "wins"      : t.get("wins", 0),
            "reg_wins"  : t.get("regulationWins", 0),
            "goal_diff" : t.get("goalDifferential", 0),
        })
    print(f"  {season}: {len(teams)} teams fetched")

# Build lookup: (season, team) → standings row
standings_map = {(r["season"], r["team"]): r for r in standings_rows}

out_standings = os.path.join(DATA_DIR, "standings_5seasons.csv")
with open(out_standings, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(standings_rows[0].keys()))
    w.writeheader(); w.writerows(standings_rows)
print(f"  Saved: {out_standings}  ({len(standings_rows)} team-seasons)")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — NET-FRONT METRICS FROM SHOT DATA
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Step 2: Computing net-front metrics ──")

# Load rebound sequences → build rebound-shot key set
print("  Loading rebound sequences...")
reb_keys = set()  # (game_id, period, shooter_id, x, y) for rebound shots
reb_teams = {}    # same key → orig_team

with open(os.path.join(DATA_DIR, "rebound_sequences.csv"), newline="") as f:
    for row in csv.DictReader(f):
        # Only regular season rebounding (game_type encoded in game_id via 02)
        if row["game_id"][4:6] != "02":
            continue
        key = (row["game_id"], row["period"], row["reb_shooter_id"],
               row["reb_x"], row["reb_y"])
        reb_keys.add(key)
        reb_teams[key] = row["orig_team"]

print(f"  Rebound keys loaded: {len(reb_keys):,}")

# Aggregate ES regular season shots per (team, season)
print("  Scanning shot events (ES, regular season only)...")

# Counters: [team_season] -> {att, nf_att, nf_goals, total_goals}
counters = defaultdict(lambda: {
    "att": 0, "nf_att": 0, "nf_goals": 0, "total_goals": 0
})

with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["situation_code"] != "1551":
            continue
        if row["game_type"] != "regular":
            continue
        if row["event_type"] not in ALL_ATTEMPTS:
            continue

        team   = row["shooting_team_abbrev"]
        season = row["season"]
        goal   = row["is_goal"] == "1"
        stype  = row["shot_type"]
        key_ts = (season, team)

        counters[key_ts]["att"] += 1
        if goal:
            counters[key_ts]["total_goals"] += 1

        # Net-front: shot type match OR rebound sequence
        is_nf_type = stype in NF_SHOT_TYPES
        reb_key = (row["game_id"], row["period"], row["shooter_player_id"],
                   row["x_coord_norm"], row["y_coord_norm"])
        is_rebound = reb_key in reb_keys

        if is_nf_type or is_rebound:
            counters[key_ts]["nf_att"] += 1
            if goal:
                counters[key_ts]["nf_goals"] += 1

print(f"  Team-seasons computed: {len(counters)}")

# Build metrics rows
metrics_rows = []
for (season, team), c in sorted(counters.items()):
    att      = c["att"]
    nf_att   = c["nf_att"]
    nf_goals = c["nf_goals"]
    nf_rate  = nf_att / att       if att    > 0 else float("nan")
    nf_gr    = nf_goals / nf_att  if nf_att > 0 else float("nan")
    # NF goals per 60 proxy: NF goals / (total_att / 30)
    nf_g60   = nf_goals / (att / 30) if att > 0 else float("nan")

    # Merge standings
    std = standings_map.get((season, team), {})
    pts_pct = std.get("points_pct", float("nan"))

    metrics_rows.append({
        "season"          : season,
        "team"            : team,
        "es_attempts"     : att,
        "nf_attempts"     : nf_att,
        "nf_goals"        : nf_goals,
        "total_es_goals"  : c["total_goals"],
        "nf_attempt_rate" : round(nf_rate, 4) if nf_rate == nf_rate else "",
        "nf_goal_rate"    : round(nf_gr,   4) if nf_gr   == nf_gr   else "",
        "nf_goals_per60"  : round(nf_g60,  4) if nf_g60  == nf_g60  else "",
        "points_pct"      : pts_pct,
        "points"          : std.get("points", ""),
        "gp"              : std.get("gp", ""),
    })

out_metrics = os.path.join(DATA_DIR, "team_netfront_metrics.csv")
with open(out_metrics, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
    w.writeheader(); w.writerows(metrics_rows)
print(f"  Saved: {out_metrics}")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — PEARSON CORRELATIONS
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Step 3: Correlations with points_pct ──")

def col(key):
    vals = []
    for r in metrics_rows:
        v = r.get(key, "")
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            vals.append(float("nan"))
    return vals

pts_pct_vals  = col("points_pct")
nf_rate_vals  = col("nf_attempt_rate")
nf_gr_vals    = col("nf_goal_rate")
nf_g60_vals   = col("nf_goals_per60")
total_gr_vals = [r["total_es_goals"] / r["es_attempts"]
                 if r["es_attempts"] > 0 else float("nan")
                 for r in metrics_rows]

corr_results = []
for label, x_vals in [
    ("nf_attempt_rate",  nf_rate_vals),
    ("nf_goal_rate",     nf_gr_vals),
    ("nf_goals_per60",   nf_g60_vals),
    ("overall_es_goal_rate", total_gr_vals),
]:
    r, p, n, sig = pearson(x_vals, pts_pct_vals, label)
    corr_results.append({
        "metric"      : label,
        "pearson_r"   : round(r, 4) if r == r else "nan",
        "p_value"     : round(p, 5) if p == p else "nan",
        "n"           : n,
        "significant" : "YES ***" if sig else "no",
        "r_squared"   : round(r**2, 4) if r == r else "nan",
    })

out_corr = os.path.join(DATA_DIR, "netfront_wins_correlation.csv")
with open(out_corr, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(corr_results[0].keys()))
    w.writeheader(); w.writerows(corr_results)
print(f"  Saved: {out_corr}")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — PLAYOFF DEPTH BY NF GOAL RATE
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Step 4: Playoff depth analysis ──")

# Load playoff game_ids and determine series winners from shot goals
print("  Loading playoff game metadata...")
playoff_meta = {}   # game_id -> {season, home, away}
with open(os.path.join(DATA_DIR, "game_ids.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["game_type"] == "playoff":
            playoff_meta[row["game_id"]] = {
                "season": row["season"],
                "home"  : row["home_abbrev"],
                "away"  : row["away_abbrev"],
            }

# Sum goals per team per playoff game (from shot events)
print("  Computing game outcomes from playoff shot events...")
game_goals = defaultdict(lambda: defaultdict(int))
with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["game_type"] != "playoff":
            continue
        if row["is_goal"] == "1":
            game_goals[row["game_id"]][row["shooting_team_abbrev"]] += 1

# Determine game winner
game_winner = {}
for gid, tg in game_goals.items():
    if tg:
        game_winner[gid] = max(tg, key=tg.get)

# Aggregate series wins: (season, round, series) -> {team: wins}
series_wins = defaultdict(lambda: defaultdict(int))
for gid, winner in game_winner.items():
    if gid not in playoff_meta:
        continue
    season   = playoff_meta[gid]["season"]
    round_n  = int(gid[6:8])
    series_n = gid[8]
    series_wins[(season, round_n, series_n)][winner] += 1

# Series winner = team with most wins
series_winner_map = {}
for (season, rnd, ser), wins in series_wins.items():
    series_winner_map[(season, rnd, ser)] = max(wins, key=wins.get)

# For each team+season: track max round appeared in and rounds won
team_max_round   = defaultdict(int)    # (season, team) -> max round appeared
team_rounds_won  = defaultdict(set)    # (season, team) -> set of rounds won

for gid, meta in playoff_meta.items():
    season  = meta["season"]
    round_n = int(gid[6:8])
    for tm in [meta["home"], meta["away"]]:
        key = (season, tm)
        team_max_round[key] = max(team_max_round.get(key, 0), round_n)

for (season, rnd, ser), winner in series_winner_map.items():
    team_rounds_won[(season, winner)].add(rnd)

# Determine playoff depth label
DEPTH_LABEL = {
    1: "R1_exit",
    2: "R2_exit",
    3: "CF_exit",
    4: "Final_loss",
}

team_depth = {}  # (season, team) -> label
for (season, team), max_rnd in team_max_round.items():
    rounds_won = team_rounds_won.get((season, team), set())
    if max_rnd in rounds_won and max_rnd == 4:
        team_depth[(season, team)] = "Cup_winner"
    elif max_rnd not in rounds_won:
        team_depth[(season, team)] = DEPTH_LABEL.get(max_rnd, f"R{max_rnd}_exit")
    else:
        # Won this round, should have appeared in next — still in playoffs
        team_depth[(season, team)] = DEPTH_LABEL.get(max_rnd + 1, f"R{max_rnd+1}_exit")

# Group metrics rows by playoff depth, compute avg NF goal rate per group
DEPTH_ORDER = ["R1_exit", "R2_exit", "CF_exit", "Final_loss", "Cup_winner", "missed_playoffs"]

depth_nf_rates = defaultdict(list)
depth_nf_att_rates = defaultdict(list)
depth_pts_pct  = defaultdict(list)

for r in metrics_rows:
    season = r["season"]
    team   = r["team"]
    key    = (season, team)

    nf_gr  = r["nf_goal_rate"]
    nf_rate = r["nf_attempt_rate"]
    pts    = r["points_pct"]
    try:
        nf_gr   = float(nf_gr)
        nf_rate = float(nf_rate)
        pts     = float(pts)
    except (ValueError, TypeError):
        continue

    depth = team_depth.get(key, "missed_playoffs")
    depth_nf_rates[depth].append(nf_gr)
    depth_nf_att_rates[depth].append(nf_rate)
    depth_pts_pct[depth].append(pts)

depth_rows = []
for depth in DEPTH_ORDER:
    gr_list  = depth_nf_rates.get(depth, [])
    att_list = depth_nf_att_rates.get(depth, [])
    pts_list = depth_pts_pct.get(depth, [])
    n = len(gr_list)
    if n == 0:
        continue
    depth_rows.append({
        "playoff_depth"      : depth,
        "n_team_seasons"     : n,
        "avg_nf_goal_rate"   : round(sum(gr_list) / n, 4),
        "avg_nf_attempt_rate": round(sum(att_list) / n, 4),
        "avg_points_pct"     : round(sum(pts_list) / n, 4),
    })

out_depth = os.path.join(DATA_DIR, "playoff_depth_nf_rates.csv")
with open(out_depth, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(depth_rows[0].keys()))
    w.writeheader(); w.writerows(depth_rows)
print(f"  Saved: {out_depth}")


# ════════════════════════════════════════════════════════════════════════════════
# TERMINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
SEP = "═" * 70

print(f"\n{SEP}")
print("  HOCKEYROI — NET-FRONT EFFICIENCY vs WINNING")
print(SEP)

print(f"\n  STEP 3 — PEARSON CORRELATIONS WITH POINTS_PCT")
print(f"  (n ≈ 160 team-seasons, 5 seasons pooled)")
print(f"  {'Metric':<26}  {'r':>8}  {'r²':>7}  {'p-value':>10}  {'n':>5}  {'Sig?':>8}")
print(f"  {'-'*68}")
for c in corr_results:
    print(f"  {c['metric']:<26}  {float(c['pearson_r']):>8.4f}  "
          f"{float(c['r_squared']):>7.4f}  {float(c['p_value']):>10.5f}  "
          f"{c['n']:>5}  {c['significant']:>8}")

print(f"\n  STEP 4 — PLAYOFF DEPTH BY NET-FRONT GOAL RATE")
print(f"  {'Exit stage':<18}  {'N':>5}  {'Avg NF G%':>10}  {'Avg NF Att%':>12}  {'Avg Pts%':>10}")
print(f"  {'-'*60}")
for d in depth_rows:
    print(f"  {d['playoff_depth']:<18}  {d['n_team_seasons']:>5}  "
          f"  {float(d['avg_nf_goal_rate'])*100:>8.2f}%"
          f"  {float(d['avg_nf_attempt_rate'])*100:>11.2f}%"
          f"  {float(d['avg_points_pct'])*100:>9.2f}%")

# Per-season league avg nf_goal_rate for reference
print(f"\n  LEAGUE-WIDE NF GOAL RATE BY SEASON (regular season)")
print(f"  {'Season':<12}  {'NF G Rate':>10}  {'NF Att Rate':>12}  {'n teams':>8}")
print(f"  {'-'*46}")
for season in sorted(set(r["season"] for r in metrics_rows)):
    sub = [r for r in metrics_rows if r["season"] == season]
    grs = [float(r["nf_goal_rate"]) for r in sub if r["nf_goal_rate"] not in ("", "nan")]
    ars = [float(r["nf_attempt_rate"]) for r in sub if r["nf_attempt_rate"] not in ("", "nan")]
    if grs:
        print(f"  {season:<12}  {sum(grs)/len(grs)*100:>9.2f}%  "
              f"{sum(ars)/len(ars)*100:>11.2f}%  {len(sub):>8}")

print(f"\n{SEP}")
print("  Output files:")
print("  - Data/standings_5seasons.csv")
print("  - Data/team_netfront_metrics.csv")
print("  - Data/netfront_wins_correlation.csv")
print("  - Data/playoff_depth_nf_rates.csv")
print(SEP + "\n")
