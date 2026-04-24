#!/usr/bin/env python3
"""
HockeyROI — Three Analyses
  Analysis 1: Three-pillar complete model (NF rate + GA/G + ES save%)
  Analysis 2: Cap hit vs composite score (NHL API — salary availability flagged)
  Analysis 3: Declining NF deployment detection

Save: Data/three_pillar_model.csv
      Data/netfront_value_rankings.csv
      Data/declining_nf_deployment.csv
"""

import csv, json, math, os, time, urllib.request
from collections import defaultdict
import numpy as np

DATA_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"

SEASONS   = ["20202021","20212022","20222023","20232024","20242025"]
CURRENT_S = "20242025"
REGULAR   = "regular"

NF_X_MIN = 74.0;  NF_Y_MIN = -8.0;  NF_Y_MAX = 8.0
SOG_TYPES = {"shot-on-goal", "goal"}

SEP  = "═" * 110
SEP2 = "─" * 110

def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    if p < 0.10:  return ".  "
    return "   "


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — BUILD PER-TEAM-SEASON AND PER-PLAYER-SEASON STATS
# ════════════════════════════════════════════════════════════════════════════
print("Phase 0: Reading nhl_shot_events.csv (this takes ~30s)...")

# Team-season accumulators
t_es_shots    = defaultdict(int)   # (season, team) → total ES shots by team
t_nf_shots    = defaultdict(int)   # (season, team) → NF zone shots by team
t_sog_against = defaultdict(int)   # (season, team) → SOG faced
t_ga          = defaultdict(int)   # (season, team) → goals conceded
t_game_ids    = defaultdict(set)   # (season, team) → unique game_ids

# Player-season accumulators
p_es_shots = defaultdict(int)    # (pid, season) → total ES shots
p_nf_shots = defaultdict(int)    # (pid, season) → NF zone shots
p_team     = defaultdict(lambda: defaultdict(dict))  # pid → season → {team: count}
p_name     = {}                  # pid → name (from elite file, filled later)

# League-season accumulators (for league-avg NF rate per season)
lg_es_shots = defaultdict(int)   # season → total ES shots
lg_nf_shots = defaultdict(int)   # season → NF zone shots

with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["game_type"] != REGULAR:
            continue
        season = row["season"]
        team   = row["shooting_team_abbrev"]
        gid    = row["game_id"]
        etype  = row["event_type"]
        pid    = row["shooter_player_id"]

        try:
            x = float(row["x_coord_norm"])
            y = float(row["y_coord_norm"])
        except ValueError:
            x = y = 0.0

        is_goal   = row["is_goal"] == "1"
        is_nf     = (x >= NF_X_MIN and NF_Y_MIN <= y <= NF_Y_MAX)

        t_key = (season, team)
        t_es_shots[t_key] += 1
        if is_nf:
            t_nf_shots[t_key] += 1
        t_game_ids[t_key].add(gid)

        # Defense: opponent concedes this shot
        home_abbrev = row["home_team_abbrev"]
        away_abbrev = row["away_team_abbrev"]
        opponent    = away_abbrev if team == home_abbrev else home_abbrev
        opp_key = (season, opponent)
        if etype in SOG_TYPES:
            t_sog_against[opp_key] += 1
        if is_goal:
            t_ga[opp_key] += 1

        # Player-season stats
        if pid:
            p_es_shots[(pid, season)] += 1
            if is_nf:
                p_nf_shots[(pid, season)] += 1
            p_team[pid][season][team] = p_team[pid][season].get(team, 0) + 1

        # League-season
        lg_es_shots[season] += 1
        if is_nf:
            lg_nf_shots[season] += 1

print("  Done.")

# League-avg NF rate per season
league_nf_rate = {s: lg_nf_shots[s] / lg_es_shots[s] for s in SEASONS if lg_es_shots[s] > 0}
print("  League-avg NF rate by season:")
for s in SEASONS:
    print(f"    {s}: {league_nf_rate.get(s,0):.4f}")


# ════════════════════════════════════════════════════════════════════════════
#  LOAD STANDINGS
# ════════════════════════════════════════════════════════════════════════════
print("\nLoading standings...")
standings = {}
with open(os.path.join(DATA_DIR, "standings_5seasons.csv"), newline="") as f:
    for row in csv.DictReader(f):
        standings[(row["season"], row["team"])] = {
            "points_pct": float(row["points_pct"]),
            "points"    : int(row["points"]),
            "gp"        : int(row["gp"]),
        }

# Load playoff depth from roster analysis
playoff_depth = {}
with open(os.path.join(DATA_DIR, "team_netfront_roster_analysis.csv"), newline="") as f:
    for row in csv.DictReader(f):
        playoff_depth[(row["season"], row["team"])] = row["playoff_depth"]


# ════════════════════════════════════════════════════════════════════════════
#  BUILD FULL 159-ROW DATASET
# ════════════════════════════════════════════════════════════════════════════
dataset = []
for key in sorted(standings.keys()):
    season, team = key
    stand = standings[key]
    gp        = stand["gp"]
    pts_pct   = stand["points_pct"]

    total_es  = t_es_shots.get(key, 0)
    total_nf  = t_nf_shots.get(key, 0)
    if total_es < 100:
        continue
    nf_rate   = total_nf / total_es

    sog_ag    = t_sog_against.get(key, 0)
    ga        = t_ga.get(key, 0)
    if sog_ag < 50:
        continue
    save_pct  = 1.0 - (ga / sog_ag)
    ga_pg     = ga / gp if gp > 0 else 0.0

    dataset.append({
        "season"      : season,
        "team"        : team,
        "points_pct"  : pts_pct,
        "nf_rate"     : nf_rate,
        "save_pct"    : save_pct,
        "ga_pg"       : ga_pg,
        "gp"          : gp,
        "playoff_depth": playoff_depth.get(key, "unknown"),
    })

n = len(dataset)
print(f"  Full dataset: {n} team-seasons")


# ════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1 — THREE-PILLAR COMPLETE MODEL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 1 — THREE-PILLAR MODEL")
print("=" * 60)

# Compute top-third thresholds (67th percentile)
def percentile(vals, pct):
    sorted_v = sorted(vals)
    idx = (pct / 100) * (len(sorted_v) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (idx - lo) * (sorted_v[hi] - sorted_v[lo])

nf_vals   = [r["nf_rate"]  for r in dataset]
sv_vals   = [r["save_pct"] for r in dataset]
ga_vals   = [r["ga_pg"]    for r in dataset]

thresh_nf = percentile(nf_vals, 67)   # top-third: ≥ this value
thresh_sv = percentile(sv_vals, 67)   # top-third: ≥ this value
thresh_ga = percentile(ga_vals, 33)   # bottom-third goals against: ≤ this value

print(f"\n  TOP-THIRD THRESHOLDS (n={n}):")
print(f"    NF attempt rate  ≥ {thresh_nf:.4f}  (top third)")
print(f"    ES save pct      ≥ {thresh_sv:.4f}  (top third)")
print(f"    GA per game      ≤ {thresh_ga:.4f}  (bottom third = fewest GA)")

# Tag each team-season
def pillar_count(r):
    p_nf  = r["nf_rate"]  >= thresh_nf
    p_sv  = r["save_pct"] >= thresh_sv
    p_ga  = r["ga_pg"]    <= thresh_ga
    return int(p_nf) + int(p_sv) + int(p_ga), p_nf, p_sv, p_ga

for r in dataset:
    cnt, p_nf, p_sv, p_ga = pillar_count(r)
    r["pillars"] = cnt
    r["p_nf"]    = p_nf
    r["p_sv"]    = p_sv
    r["p_ga"]    = p_ga

# Group by pillar count
from collections import Counter

groups = {0: [], 1: [], 2: [], 3: []}
for r in dataset:
    groups[r["pillars"]].append(r)

def depth_summary(rows):
    depth_ct = Counter(r["playoff_depth"] for r in rows)
    total    = len(rows)
    cup_win  = depth_ct.get("Cup_winner", 0)
    final    = depth_ct.get("Final_loss", 0)
    cf       = depth_ct.get("CF_exit", 0)
    r2       = depth_ct.get("R2_exit", 0)
    r1       = depth_ct.get("R1_exit", 0)
    missed   = depth_ct.get("missed_playoffs", 0)
    avg_pts  = sum(r["points_pct"] for r in rows) / total if total else 0
    return {
        "n": total, "avg_pts_pct": avg_pts,
        "Cup": cup_win, "Final": final, "CF": cf, "R2": r2, "R1": r1, "missed": missed,
        "finals+": cup_win + final,
        "early_exit+missed": r1 + missed,
    }

print(f"\n{SEP}")
print("  PILLAR COMBINATION RESULTS")
print(SEP)
print(f"  {'Pillars':<10} {'N':>4}  {'Avg Pts%':>9}  {'Cup':>4}  {'Final':>6}  {'CF':>4}  {'R2':>4}  {'R1':>4}  {'Missed':>7}  {'Cup+Final':>10}  {'Pillar combo'}")
print(f"  {SEP2}")

for cnt in [3, 2, 1, 0]:
    rows = groups[cnt]
    if not rows:
        print(f"  {cnt} pillar{'s' if cnt!=1 else '':1}    {'0':>4}  {'—':>9}")
        continue
    d = depth_summary(rows)
    label = {3: "NF + SV + GA", 2: "any 2 of 3", 1: "any 1 of 3", 0: "none"}[cnt]
    print(f"  {cnt} pillar{'s' if cnt!=1 else '':1}  {d['n']:>4}  {d['avg_pts_pct']*100:>8.1f}%  "
          f"{d['Cup']:>4}  {d['Final']:>6}  {d['CF']:>4}  {d['R2']:>4}  {d['R1']:>4}  {d['missed']:>7}  "
          f"{d['finals+']:>5}/{d['n']:<4} ({d['finals+']/d['n']*100:.0f}%)  {label}")

# Detail: all 3-pillar team-seasons
three_pillar = sorted(groups[3], key=lambda r: -r["points_pct"])
print(f"\n  ALL {len(three_pillar)} THREE-PILLAR TEAM-SEASONS (NF + save% + GA all top-third):")
print(f"  {'-'*90}")
print(f"  {'Season':<10} {'Team':<5}  {'Pts%':>6}  {'NF rate':>8}  {'SV%':>7}  {'GA/G':>6}  {'Result'}")
print(f"  {'-'*90}")
for r in three_pillar:
    print(f"  {r['season']:<10} {r['team']:<5}  {r['points_pct']*100:>5.1f}%  "
          f"{r['nf_rate']:>8.4f}  {r['save_pct']:>7.4f}  {r['ga_pg']:>6.3f}  {r['playoff_depth']}")

# Save
out1_path = os.path.join(DATA_DIR, "three_pillar_model.csv")
with open(out1_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["season","team","pillars","p_nf","p_sv","p_ga",
                                       "nf_rate","save_pct","ga_pg","points_pct","gp","playoff_depth"])
    w.writeheader()
    w.writerows(sorted(dataset, key=lambda r: (-r["pillars"], -r["points_pct"])))
print(f"\n  Saved: {out1_path}")


# ════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2 — CAP HIT vs COMPOSITE SCORE (NHL API)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("ANALYSIS 2 — COMPOSITE SCORE vs CONTRACT DATA (NHL API)")
print("=" * 60)

# Load elite players
elite_rows = []
with open(os.path.join(DATA_DIR, "elite_netfront_players.csv"), newline="") as f:
    for row in csv.DictReader(f):
        elite_rows.append(row)

ci_confirmed = [r for r in elite_rows if r["ci_confirmed"] == "True"]
print(f"\n  Loading {len(ci_confirmed)} CI-confirmed elite players from NHL API...")
print("  ⚠  NOTE: NHL public API (api-web.nhle.com) does NOT expose cap hit,")
print("           AAV, contract length, or UFA/RFA status.")
print("           For salary data use: puckpedia.com / capfriendly.com / spotrac.com")
print("           Extracting available biographical & draft data only.\n")

def fetch_player(pid):
    url = f"https://api-web.nhle.com/v1/player/{pid}/landing"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return None

player_api = {}
for i, row in enumerate(ci_confirmed):
    pid  = row["player_id"]
    name = row["player_name"]
    data = fetch_player(pid)
    if data:
        fn = data.get("firstName", {}).get("default", "")
        ln = data.get("lastName",  {}).get("default", "")
        pos = data.get("position", "?")
        bd  = data.get("birthDate", "?")
        age = 0
        if bd and bd != "?":
            birth_yr = int(bd[:4])
            age = 2025 - birth_yr
        draft = data.get("draftDetails", {})
        draft_str = f"{draft.get('year','?')} Rd{draft.get('round','?')} #{draft.get('overallPick','?')}" if draft else "undrafted"
        active  = data.get("isActive", "?")
        team_ab = data.get("currentTeamAbbrev", "—")
        # Check for any salary-related keys
        raw = json.dumps(data)
        has_salary = any(k in raw.lower() for k in ["salary","cap hit","aav","capnum","capvalue"])
        player_api[pid] = {
            "full_name"   : f"{fn} {ln}".strip() or name,
            "position"    : pos,
            "age"         : age,
            "draft"       : draft_str,
            "active"      : active,
            "current_team": team_ab,
            "has_salary"  : has_salary,
        }
    time.sleep(0.12)
    if (i + 1) % 5 == 0:
        print(f"    {i+1}/{len(ci_confirmed)} players fetched...", flush=True)

salary_found = sum(1 for v in player_api.values() if v["has_salary"])
print(f"\n  API results: {len(player_api)}/{len(ci_confirmed)} players returned data")
print(f"  Salary/cap data found in responses: {salary_found} players")
if salary_found == 0:
    print("  ✗ CONFIRMED: NHL public API returns no salary/contract data.")
    print("    Cap hit analysis requires external data source (PuckPedia, CapFriendly).")
    print("    Proceeding with composite score ranking + available API fields.\n")

# Print composite ranking with available data
print(f"\n{SEP}")
print("  CI-CONFIRMED ELITE NET-FRONT PLAYERS — COMPOSITE RANKING + API PROFILE")
print(f"  ⚠  No salary data available from NHL API")
print(SEP)
print(f"  {'Rk':<4} {'Player':<22} {'Pos':<4} {'Age':>4} {'Team':<5} {'Composite':>10} "
      f"{'Reb GR':>8} {'Shots':>7} {'Draft'}")
print(f"  {SEP2}")

sorted_elite = sorted(ci_confirmed, key=lambda r: -float(r["composite"]))
value_rows   = []
for r in sorted_elite:
    pid   = r["player_id"]
    api   = player_api.get(pid, {})
    comp  = float(r["composite"])
    gr    = float(r["rebound_goal_rate"])
    shots = int(r["rebound_shots"])
    pos   = api.get("position", "?")
    age   = api.get("age", "?")
    team  = api.get("current_team", r["current_team"])
    draft = api.get("draft", "?")
    active= api.get("active", True)
    print(f"  {r['rank']:<4} {r['player_name']:<22} {pos:<4} {str(age):>4} {team:<5} "
          f"{comp:>10.4f} {gr:>8.4f} {shots:>7}  {draft}")
    value_rows.append({
        "rank"              : r["rank"],
        "player_name"       : r["player_name"],
        "player_id"         : pid,
        "position"          : pos,
        "age_approx"        : age,
        "current_team"      : team,
        "composite"         : comp,
        "rebound_goal_rate" : gr,
        "rebound_shots"     : shots,
        "avg_x"             : r["avg_x"],
        "avg_y"             : r["avg_y"],
        "avg_time_gap"      : r["avg_time_gap"],
        "ci_confirmed"      : True,
        "draft_info"        : draft,
        "is_active"         : active,
        "cap_hit"           : "N/A — not in NHL API",
        "contract_expiry"   : "N/A — not in NHL API",
        "ufa_rfa_status"    : "N/A — not in NHL API",
        "value_tier"        : "",
        "change_of_scenery" : "",
    })

# Flag UFA candidates (age 27+ without contract clarity = likely in prime)
print(f"\n  NOTE: Without salary data, 'best value' and 'UFA targets' cannot be computed.")
print(f"  Players aged 28-32 with composite ≥ 0.55 are prime offseason targets:")
print(f"  {'Player':<22} {'Age':>4} {'Team':<5} {'Composite':>10}")
print(f"  {'-'*50}")
for r in value_rows:
    age = r["age_approx"]
    if isinstance(age, int) and 28 <= age <= 33 and r["composite"] >= 0.55:
        print(f"  {r['player_name']:<22} {age:>4} {r['current_team']:<5} {r['composite']:>10.4f}")

# Save
out2_path = os.path.join(DATA_DIR, "netfront_value_rankings.csv")
with open(out2_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(value_rows[0].keys()))
    w.writeheader()
    w.writerows(value_rows)
print(f"\n  Saved: {out2_path}")
print(f"\n  ⚠  To complete Analysis 2 with cap data:")
print(f"     Provide a CSV with columns [player_name, cap_hit, expiry_year, ufa_rfa]")
print(f"     from puckpedia.com or capfriendly.com and we can compute value rankings.")


# ════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3 — DECLINING NF DEPLOYMENT DETECTION
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("ANALYSIS 3 — DECLINING NF DEPLOYMENT DETECTION")
print("=" * 60)

# Build per-player per-season NF rate
# Only players with 4+ seasons of data in regular season
# AND 30+ ES shots in 20242025 (proxy for 300+ ES minutes)

all_pids   = set()
p_seasons  = defaultdict(dict)   # pid → {season: {'es': n, 'nf': n, 'team': str}}

for (pid, season), es_ct in p_es_shots.items():
    nf_ct = p_nf_shots.get((pid, season), 0)
    # Primary team this season for this player
    teams = p_team[pid].get(season, {})
    primary_team = max(teams, key=teams.get) if teams else "?"
    if es_ct >= 10:  # filter noise: at least 10 ES shots this season
        p_seasons[pid][season] = {
            "es"  : es_ct,
            "nf"  : nf_ct,
            "rate": nf_ct / es_ct,
            "team": primary_team,
        }
    all_pids.add(pid)

# Load player name lookup from elite file (augment with any rebound positioning file)
print("\n  Building player name lookup...")
name_lookup = {}
for r in elite_rows:
    name_lookup[r["player_id"]] = r["player_name"]

try:
    with open(os.path.join(DATA_DIR, "player_rebound_positioning.csv"), newline="") as f:
        for row in csv.DictReader(f):
            pid = str(row["shooter_player_id"])
            if pid not in name_lookup:
                name_lookup[pid] = row["player_name"]
    print(f"  {len(name_lookup)} player names loaded")
except FileNotFoundError:
    print("  player_rebound_positioning.csv not found — using elite list names only")

# Filter: 4+ seasons of data
MIN_SEASONS = 4
MIN_ES_CURRENT = 30    # proxy for 300+ ES minutes in 20242025

flagged = []

for pid, seasons_data in p_seasons.items():
    valid_seasons = [s for s in SEASONS if s in seasons_data]
    if len(valid_seasons) < MIN_SEASONS:
        continue

    # Must have data in 20242025 with enough shots
    if CURRENT_S not in seasons_data:
        continue
    curr = seasons_data[CURRENT_S]
    if curr["es"] < MIN_ES_CURRENT:
        continue

    # Compute peak NF rate across all seasons
    rates = {s: seasons_data[s]["rate"] for s in valid_seasons}
    peak_season = max(rates, key=rates.get)
    peak_rate   = rates[peak_season]
    curr_rate   = curr["rate"]

    # Peak must have been above league average that season
    lg_avg_peak = league_nf_rate.get(peak_season, 0)
    if peak_rate <= lg_avg_peak:
        continue

    # Current must have dropped 30%+ from peak
    if peak_rate == 0:
        continue
    drop_pct = (peak_rate - curr_rate) / peak_rate
    if drop_pct < 0.30:
        continue

    # Compute full seasonal history
    history = []
    for s in SEASONS:
        if s in seasons_data:
            sd = seasons_data[s]
            history.append(f"{s[-4:]}:{sd['rate']:.3f}({sd['es']}es)")
        else:
            history.append(f"{s[-4:]}:—")

    name = name_lookup.get(pid, f"ID:{pid}")
    flagged.append({
        "player_id"      : pid,
        "player_name"    : name,
        "current_team"   : curr["team"],
        "seasons_in_db"  : len(valid_seasons),
        "peak_season"    : peak_season,
        "peak_nf_rate"   : round(peak_rate, 4),
        "league_avg_peak": round(lg_avg_peak, 4),
        "current_nf_rate": round(curr_rate, 4),
        "drop_pct"       : round(drop_pct, 4),
        "current_es_shots": curr["es"],
        "season_history" : " | ".join(history),
    })

# Sort by drop magnitude
flagged.sort(key=lambda r: -r["drop_pct"])

print(f"\n  Players with 4+ seasons, peak NF rate > league avg, ")
print(f"  current (20242025) NF rate dropped ≥30%, ≥{MIN_ES_CURRENT} ES shots this season:")
print(f"  Found: {len(flagged)} change-of-scenery candidates\n")

print(f"{SEP}")
print("  ANALYSIS 3 — DECLINING NF DEPLOYMENT: CHANGE-OF-SCENERY CANDIDATES")
print(SEP)
print(f"  {'Player':<22} {'Team':<5} {'PkSzn':<9} {'PkRate':>7} {'LgAvg':>7} {'CurRate':>8} {'Drop%':>6}  {'ESshots':>8}  Seasonal history")
print(f"  {SEP2}")

for r in flagged:
    print(f"  {r['player_name']:<22} {r['current_team']:<5} {r['peak_season']:<9} "
          f"{r['peak_nf_rate']:>7.4f} {r['league_avg_peak']:>7.4f} {r['current_nf_rate']:>8.4f} "
          f"{r['drop_pct']*100:>5.1f}%  {r['current_es_shots']:>8}  {r['season_history']}")

# Highlight any who also made the elite list
elite_pids = {r["player_id"] for r in ci_confirmed}
in_elite   = [r for r in flagged if r["player_id"] in elite_pids]
if in_elite:
    print(f"\n  ↳ Of these, {len(in_elite)} are also on the CI-confirmed elite list (regression risk):")
    for r in in_elite:
        print(f"    {r['player_name']:<22} composite: {float(next(e['composite'] for e in ci_confirmed if e['player_id']==r['player_id'])):.4f}")

# Save
out3_path = os.path.join(DATA_DIR, "declining_nf_deployment.csv")
with open(out3_path, "w", newline="") as f:
    cols = ["player_id","player_name","current_team","seasons_in_db",
            "peak_season","peak_nf_rate","league_avg_peak",
            "current_nf_rate","drop_pct","current_es_shots","season_history"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(flagged)
print(f"\n  Saved: {out3_path}")


# ════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  SUMMARY")
print(SEP)
d3 = depth_summary(groups[3])
d2 = depth_summary(groups[2])
d1 = depth_summary(groups[1])
d0 = depth_summary(groups[0])
print(f"  Analysis 1:")
print(f"    Thresholds: NF rate ≥ {thresh_nf:.4f} | save% ≥ {thresh_sv:.4f} | GA/G ≤ {thresh_ga:.4f}")
print(f"    3-pillar teams: {d3['n']}  avg pts% {d3['avg_pts_pct']*100:.1f}%  "
      f"Cup+Final: {d3['finals+']}/{d3['n']} ({d3['finals+']/max(d3['n'],1)*100:.0f}%)  "
      f"Early/missed: {d3['early_exit+missed']}/{d3['n']} ({d3['early_exit+missed']/max(d3['n'],1)*100:.0f}%)")
print(f"    2-pillar teams: {d2['n']}  avg pts% {d2['avg_pts_pct']*100:.1f}%  "
      f"Cup+Final: {d2['finals+']}/{d2['n']} ({d2['finals+']/max(d2['n'],1)*100:.0f}%)")
print(f"    1-pillar teams: {d1['n']}  avg pts% {d1['avg_pts_pct']*100:.1f}%  "
      f"Cup+Final: {d1['finals+']}/{d1['n']} ({d1['finals+']/max(d1['n'],1)*100:.0f}%)")
print(f"    0-pillar teams: {d0['n']}  avg pts% {d0['avg_pts_pct']*100:.1f}%  "
      f"Cup+Final: {d0['finals+']}/{d0['n']} ({d0['finals+']/max(d0['n'],1)*100:.0f}%)")
print(f"\n  Analysis 2: NHL API returns NO salary/contract data.")
print(f"    {len(ci_confirmed)} CI-confirmed elite players profiled with biographical data.")
print(f"    Provide a salary CSV (PuckPedia/CapFriendly) for full value analysis.")
print(f"\n  Analysis 3: {len(flagged)} change-of-scenery candidates flagged.")
print(f"    {len(in_elite)} overlap with CI-confirmed elite list (deployed below their rebound-skill potential).")
print(SEP + "\n")
