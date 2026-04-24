#!/usr/bin/env python3
"""
HockeyROI - Oilers Net-Front Analysis
Save to: NHL analysis/oilers_netfront.py

Usage:
  cd "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
  python3 oilers_netfront.py

Sources:
  Data/nhl_shot_events.csv   — master shot database (5 seasons, ES=1551)
  Data/rebound_sequences.csv — detected rebound pairs

NOTE on Part 3: No EDM vs COL game on 2025-04-13.
  EDM played @ WPG that night (game_id=2024021282).
  Most recent EDM vs COL: 2025-02-07 COL @ EDM (game_id=2024020874).
  Part 3 runs on BOTH games and flags the discrepancy.

Outputs (Data/):
  oilers_netfront_part1_team.csv     — team profiles EDM / FLA / WSH vs league
  oilers_netfront_part2_players.csv  — top 10 Oilers net-front players
  oilers_netfront_part3_game.csv     — shot-level net-front log for target game
"""

import csv
import math
import os
import time
from collections import defaultdict

import requests

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"
BASE_URL = "https://api-web.nhle.com/v1"
Z95      = 1.96

SHOTS_FILE = os.path.join(DATA_DIR, "nhl_shot_events.csv")
REB_FILE   = os.path.join(DATA_DIR, "rebound_sequences.csv")

TEAMS_OF_INTEREST = ["EDM", "FLA", "WSH"]
SEASONS_ORDER = ["20202021", "20212022", "20222023", "20232024", "20242025"]

NF_SHOT_TYPES = {"tip-in", "deflected", "bat"}
ALL_ATTEMPTS   = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}

# April 13 game IDs
GAME_APR13_EDM = "2024021282"   # EDM @ WPG  (EDM did NOT play COL this night)
GAME_APR13_LAST_EDM_COL = "2024020874"   # COL @ EDM, 2025-02-07 (most recent EDM-COL)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "HockeyROI-Analysis/2.0"


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def wilson_ci(k, n, z=Z95):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def safe_div(a, b):
    return a / b if b else float("nan")


def fmt_pct(v):
    return f"{v*100:.2f}%" if v == v else "n/a"


def fmt_f(v, dec=4):
    return f"{v:.{dec}f}" if v == v else "n/a"


_name_cache = {}
def fetch_name(pid):
    pid = str(pid)
    if pid in _name_cache:
        return _name_cache[pid]
    try:
        r = SESSION.get(f"{BASE_URL}/player/{pid}/landing", timeout=12)
        if r.status_code == 200:
            d = r.json()
            name = (d.get("firstName", {}).get("default", "") + " " +
                    d.get("lastName",  {}).get("default", "")).strip()
            _name_cache[pid] = name
            return name
    except Exception:
        pass
    _name_cache[pid] = f"ID:{pid}"
    return _name_cache[pid]


# ─── LOAD & TAG DATA ───────────────────────────────────────────────────────────
print("Loading shot events...")
shots = []
with open(SHOTS_FILE, newline="") as f:
    for row in csv.DictReader(f):
        if row["situation_code"] != "1551":
            continue
        if row["event_type"] not in ALL_ATTEMPTS:
            continue
        shots.append(row)
print(f"  ES attempts loaded: {len(shots):,}")

print("Loading rebound sequences...")
reb_seqs = []
with open(REB_FILE, newline="") as f:
    reb_seqs = list(csv.DictReader(f))
print(f"  Rebound sequences loaded: {len(reb_seqs):,}")

# Build rebound-shot lookup: key = (game_id, period, reb_shooter_id, reb_x, reb_y)
# → {time_gap_secs, orig_team, reb_is_goal}
reb_lookup = {}
for r in reb_seqs:
    key = (r["game_id"], r["period"], r["reb_shooter_id"],
           r["reb_x"], r["reb_y"])
    reb_lookup[key] = {
        "time_gap"  : r["time_gap_secs"],
        "team"      : r["orig_team"],
        "reb_is_goal": r["reb_is_goal"],
        "orig_x"    : r["orig_x"],
        "orig_y"    : r["orig_y"],
    }

# Tag each shot as net-front
def is_netfront(row):
    if row["shot_type"] in NF_SHOT_TYPES:
        return True
    key = (row["game_id"], row["period"], row["shooter_player_id"],
           row["x_coord_norm"], row["y_coord_norm"])
    return key in reb_lookup

def get_reb_info(row):
    key = (row["game_id"], row["period"], row["shooter_player_id"],
           row["x_coord_norm"], row["y_coord_norm"])
    return reb_lookup.get(key)

print("Tagging net-front shots...")
for row in shots:
    row["_nf"]  = is_netfront(row)
    row["_reb"] = get_reb_info(row)  # None if not a rebound shot


# ════════════════════════════════════════════════════════════════════════════════
# PART 1 — TEAM PROFILES (EDM, FLA, WSH vs league)
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Part 1: Team net-front profiles ──")

# Aggregate: per (team, season) and league per season
team_stats   = defaultdict(lambda: defaultdict(int))   # [team][season][stat]
league_stats = defaultdict(lambda: defaultdict(int))   # [season][stat]

for row in shots:
    season = row["season"]
    team   = row["shooting_team_abbrev"]
    nf     = row["_nf"]
    goal   = row["is_goal"] == "1"

    # league
    league_stats[season]["att"] += 1
    if nf:
        league_stats[season]["nf_att"] += 1
        if goal:
            league_stats[season]["nf_goals"] += 1

    # per team
    team_stats[team][season + "_att"] += 1
    if nf:
        team_stats[team][season + "_nf_att"] += 1
        if goal:
            team_stats[team][season + "_nf_goals"] += 1

part1_rows = []
for team in TEAMS_OF_INTEREST:
    for season in SEASONS_ORDER:
        att      = team_stats[team][season + "_att"]
        nf_att   = team_stats[team][season + "_nf_att"]
        nf_goals = team_stats[team][season + "_nf_goals"]
        nf_rate  = safe_div(nf_att, att)
        nf_gr    = safe_div(nf_goals, nf_att)

        lg_att      = league_stats[season]["att"]
        lg_nf_att   = league_stats[season]["nf_att"]
        lg_nf_goals = league_stats[season]["nf_goals"]
        lg_nf_rate  = safe_div(lg_nf_att, lg_att)
        lg_nf_gr    = safe_div(lg_nf_goals, lg_nf_att)

        part1_rows.append({
            "team"           : team,
            "season"         : season,
            "es_attempts"    : att,
            "nf_attempts"    : nf_att,
            "nf_attempt_rate": round(nf_rate, 4) if att else "",
            "nf_goals"       : nf_goals,
            "nf_goal_rate"   : round(nf_gr, 4) if nf_att else "",
            "lg_nf_rate"     : round(lg_nf_rate, 4) if lg_att else "",
            "lg_nf_goal_rate": round(lg_nf_gr, 4) if lg_nf_att else "",
            "rate_vs_lg"     : round(nf_rate - lg_nf_rate, 4) if att and lg_att else "",
        })

out1 = os.path.join(DATA_DIR, "oilers_netfront_part1_team.csv")
with open(out1, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(part1_rows[0].keys()))
    w.writeheader(); w.writerows(part1_rows)
print(f"  Saved: {out1}")


# ════════════════════════════════════════════════════════════════════════════════
# PART 2 — TOP OILERS NET-FRONT PLAYERS
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Part 2: Oilers individual net-front players ──")

edm_players = defaultdict(lambda: {
    "reb_shots": 0, "reb_goals": 0, "reb_time_gaps": [],
    "reb_xs": [], "reb_ys": [],
    "nf_type_shots": 0, "nf_type_goals": 0,
    "nf_type_xs": [], "nf_type_ys": [],
})

for row in shots:
    if row["shooting_team_abbrev"] != "EDM":
        continue
    if not row["_nf"]:
        continue
    pid  = row["shooter_player_id"]
    goal = row["is_goal"] == "1"

    try: x = float(row["x_coord_norm"])
    except: x = None
    try: y = float(row["y_coord_norm"])
    except: y = None

    reb_info = row["_reb"]
    if reb_info:
        edm_players[pid]["reb_shots"] += 1
        if goal: edm_players[pid]["reb_goals"] += 1
        try: edm_players[pid]["reb_time_gaps"].append(float(reb_info["time_gap"]))
        except: pass
        if x is not None: edm_players[pid]["reb_xs"].append(x)
        if y is not None: edm_players[pid]["reb_ys"].append(y)
    elif row["shot_type"] in NF_SHOT_TYPES:
        edm_players[pid]["nf_type_shots"] += 1
        if goal: edm_players[pid]["nf_type_goals"] += 1
        if x is not None: edm_players[pid]["nf_type_xs"].append(x)
        if y is not None: edm_players[pid]["nf_type_ys"].append(y)

# Build player rows
p2_list = []
for pid, d in edm_players.items():
    total_nf    = d["reb_shots"] + d["nf_type_shots"]
    total_goals = d["reb_goals"] + d["nf_type_goals"]
    if total_nf < 5:
        continue
    all_xs = d["reb_xs"] + d["nf_type_xs"]
    all_ys = d["reb_ys"] + d["nf_type_ys"]
    avg_x  = sum(all_xs) / len(all_xs) if all_xs else float("nan")
    avg_y  = sum(all_ys) / len(all_ys) if all_ys else float("nan")
    avg_gap = (sum(d["reb_time_gaps"]) / len(d["reb_time_gaps"])
               if d["reb_time_gaps"] else float("nan"))
    nf_gr = safe_div(total_goals, total_nf)
    _, ci_lo, ci_hi = wilson_ci(total_goals, total_nf)
    p2_list.append({
        "shooter_player_id": pid,
        "player_name"      : "",
        "total_nf_shots"   : total_nf,
        "rebound_shots"    : d["reb_shots"],
        "nf_type_shots"    : d["nf_type_shots"],
        "nf_goals"         : total_goals,
        "rebound_goals"    : d["reb_goals"],
        "nf_goal_rate"     : round(nf_gr, 4),
        "nf_gr_ci_lo"      : round(ci_lo, 4),
        "nf_gr_ci_hi"      : round(ci_hi, 4),
        "avg_x_norm"       : round(avg_x, 1) if avg_x == avg_x else "",
        "avg_y_norm"       : round(avg_y, 1) if avg_y == avg_y else "",
        "avg_time_gap"     : round(avg_gap, 2) if avg_gap == avg_gap else "",
    })

p2_list.sort(key=lambda r: (-r["nf_goals"], -r["total_nf_shots"]))

# Fetch names for top 15 to be safe
print(f"  Fetching names for {min(15, len(p2_list))} Oilers players...")
for r in p2_list[:15]:
    r["player_name"] = fetch_name(r["shooter_player_id"])
    time.sleep(0.05)

out2 = os.path.join(DATA_DIR, "oilers_netfront_part2_players.csv")
with open(out2, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(p2_list[0].keys()))
    w.writeheader(); w.writerows(p2_list)
print(f"  Saved: {out2}")


# ════════════════════════════════════════════════════════════════════════════════
# PART 3 — GAME-LEVEL NET-FRONT LOG
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Part 3: Game net-front shot log ──")
print(f"  NOTE: No EDM vs COL game on 2025-04-13.")
print(f"  Using: EDM @ WPG  on 2025-04-13  (game_id={GAME_APR13_EDM})")
print(f"  Also:  Most recent EDM vs COL: 2025-02-07 (game_id={GAME_APR13_LAST_EDM_COL})")

def build_game_log(game_id, shots_data, label):
    game_rows = [r for r in shots_data if r["game_id"] == game_id and r["_nf"]]

    # Get unique player IDs for name lookup
    pids = {r["shooter_player_id"] for r in game_rows}
    print(f"  [{label}] Fetching {len(pids)} player names...")
    for pid in pids:
        fetch_name(pid)
        time.sleep(0.04)

    log = []
    for r in sorted(game_rows, key=lambda x: (x["period"], x["time_secs"])):
        reb_info = r["_reb"]
        log.append({
            "game_id"       : game_id,
            "game_label"    : label,
            "period"        : r["period"],
            "time_in_period": r["time_in_period"],
            "team"          : r["shooting_team_abbrev"],
            "player_name"   : fetch_name(r["shooter_player_id"]),
            "player_id"     : r["shooter_player_id"],
            "event_type"    : r["event_type"],
            "shot_type"     : r["shot_type"],
            "nf_category"   : "rebound" if reb_info else r["shot_type"],
            "x_coord_norm"  : r["x_coord_norm"],
            "y_coord_norm"  : r["y_coord_norm"],
            "time_gap_secs" : reb_info["time_gap"] if reb_info else "",
            "is_goal"       : r["is_goal"],
        })
    return log

log_apr13    = build_game_log(GAME_APR13_EDM,          shots, "EDM@WPG 2025-04-13")
log_edm_col  = build_game_log(GAME_APR13_LAST_EDM_COL, shots, "COL@EDM 2025-02-07")

all_game_logs = log_apr13 + log_edm_col

out3 = os.path.join(DATA_DIR, "oilers_netfront_part3_game.csv")
if all_game_logs:
    with open(out3, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_game_logs[0].keys()))
        w.writeheader(); w.writerows(all_game_logs)
print(f"  Saved: {out3}")


# ════════════════════════════════════════════════════════════════════════════════
# TERMINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
SEP = "═" * 72

print(f"\n{SEP}")
print("  HOCKEYROI — OILERS NET-FRONT ANALYSIS")
print(SEP)

# ── Part 1 ────────────────────────────────────────────────────────────────────
print(f"\n  PART 1 — TEAM NET-FRONT PROFILES (ES, all 5 seasons)")
print(f"  Net-front = tip-in / deflected / bat / rebound sequence")
print()
print(f"  {'Team':<5} {'Season':<10} {'ES Att':>7} {'NF Att':>7} "
      f"{'NF Rate':>8} {'Lg Rate':>8} {'vs Lg':>7} {'NF Goals':>9} {'NF G%':>7} {'Lg G%':>7}")
print(f"  {'-'*72}")

for row in part1_rows:
    vs_lg  = row["rate_vs_lg"]
    vs_str = f"{float(vs_lg):+.4f}" if vs_lg != "" else "  n/a"
    print(f"  {row['team']:<5} {row['season']:<10} "
          f"{row['es_attempts']:>7,} {row['nf_attempts']:>7,} "
          f"{fmt_pct(float(row['nf_attempt_rate'])) if row['nf_attempt_rate'] != '' else 'n/a':>8} "
          f"{fmt_pct(float(row['lg_nf_rate'])) if row['lg_nf_rate'] != '' else 'n/a':>8} "
          f"{vs_str:>7} "
          f"{row['nf_goals']:>9,} "
          f"{fmt_pct(float(row['nf_goal_rate'])) if row['nf_goal_rate'] != '' else 'n/a':>7} "
          f"{fmt_pct(float(row['lg_nf_goal_rate'])) if row['lg_nf_goal_rate'] != '' else 'n/a':>7}")
    # Add separator between teams
    if row["season"] == "20242025":
        print(f"  {'-'*72}")

# ── Part 2 ────────────────────────────────────────────────────────────────────
print(f"\n  PART 2 — TOP 10 OILERS NET-FRONT PLAYERS (5 seasons pooled)")
print(f"  {'Rk':<4} {'Player':<22} {'NF Tot':>7} {'Reb Sh':>7} {'NF Tip':>7} "
      f"{'NF G':>6} {'Reb G':>6} {'G%':>7} {'Avg X':>7} {'Avg Y':>7} {'Gap':>6}")
print(f"  {'-'*85}")
for i, r in enumerate(p2_list[:10]):
    gap_s = f"{float(r['avg_time_gap']):.2f}s" if r["avg_time_gap"] != "" else "  n/a"
    print(f"  {i+1:<4} {r['player_name']:<22} {r['total_nf_shots']:>7} "
          f"{r['rebound_shots']:>7} {r['nf_type_shots']:>7} "
          f"{r['nf_goals']:>6} {r['rebound_goals']:>6} "
          f"{fmt_pct(float(r['nf_goal_rate'])):>7} "
          f"{str(r['avg_x_norm']):>7} {str(r['avg_y_norm']):>7} "
          f"{gap_s:>6}")

# ── Part 3 ────────────────────────────────────────────────────────────────────
def print_game_section(log, label, team_a, team_b):
    print(f"\n  {label}")
    if not log:
        print(f"  (no net-front shots found for game_id — check data)")
        return

    print(f"  {'Per':<4} {'Time':<7} {'Team':<5} {'Player':<22} {'NF Cat':<12} "
          f"{'Shot Type':<12} {'X':>5} {'Y':>5} {'Gap':>5} {'Goal':>5}")
    print(f"  {'-'*90}")
    for r in log:
        gap_s = f"{float(r['time_gap_secs']):.1f}s" if r["time_gap_secs"] != "" else "  —"
        goal_s = "GOAL" if r["is_goal"] == "1" else ""
        print(f"  {r['period']:<4} {r['time_in_period']:<7} {r['team']:<5} "
              f"{r['player_name']:<22} {r['nf_category']:<12} "
              f"{r['shot_type']:<12} {r['x_coord_norm']:>5} {r['y_coord_norm']:>5} "
              f"{gap_s:>5} {goal_s:>5}")

    # Positional summary per team
    print(f"\n  Positional summary:")
    for tm in [team_a, team_b]:
        tm_rows = [r for r in log if r["team"] == tm]
        if not tm_rows:
            print(f"    {tm}: no net-front shots")
            continue
        xs = [float(r["x_coord_norm"]) for r in tm_rows if r["x_coord_norm"]]
        ys = [float(r["y_coord_norm"]) for r in tm_rows if r["y_coord_norm"]]
        avg_x = sum(xs)/len(xs) if xs else float("nan")
        avg_y = sum(ys)/len(ys) if ys else float("nan")
        goals = sum(1 for r in tm_rows if r["is_goal"] == "1")
        print(f"    {tm}: {len(tm_rows)} NF shots  avg pos ({avg_x:.1f}, {avg_y:.1f})  "
              f"{goals} goals")

print(f"\n  PART 3 — GAME NET-FRONT SHOT LOGS")
print(f"  *** NOTE: EDM did NOT play COL on 2025-04-13. ***")
print(f"  *** EDM played @ WPG. COL played @ ANA. ***")
print(f"  *** Showing EDM@WPG (Apr 13) + most recent EDM-COL (Feb 7) below. ***")

print_game_section(log_apr13,   "EDM @ WPG — 2025-04-13  (game_id=2024021282)", "EDM", "WPG")
print_game_section(log_edm_col, "COL @ EDM — 2025-02-07  (game_id=2024020874)", "COL", "EDM")

print(f"\n{SEP}")
print("  Output files: Data/oilers_netfront_part1_team.csv")
print("                Data/oilers_netfront_part2_players.csv")
print("                Data/oilers_netfront_part3_game.csv")
print(SEP + "\n")
