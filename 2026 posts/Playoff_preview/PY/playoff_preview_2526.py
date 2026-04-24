#!/usr/bin/env python3
"""
HockeyROI — 2025-26 Playoff Preview
Steps:
  1. Identify 16 playoff teams from NHL API standings
  2. Three-pillar assessment (NF rate, ES save%, GA/G) vs all 32 teams
  3. Tier 1 / Tier 2 Medium+ net-front players per playoff team
  4. First-round matchup analysis
  Save: Data/playoff_preview_2526.csv
"""

import csv, json, os, math, urllib.request
from collections import defaultdict

DATA_DIR   = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"
SEASON     = "20252026"
ES_CODE    = "1551"
NF_X_MIN   = 74.0; NF_Y_MIN = -8.0; NF_Y_MAX = 8.0
PILLAR_PCT = 1/3   # top third

# ── STEP 1: STANDINGS → PLAYOFF TEAMS & BRACKET ──────────────────────────────
print("\nFetching standings from NHL API...")
url = "https://api-web.nhle.com/v1/standings/2026-04-15"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        standings_raw = json.load(r)
except Exception:
    # Fallback: use cached standings if available
    cached = "/tmp/standings.json"
    if os.path.exists(cached):
        print("  (using cached standings)")
        with open(cached) as f:
            standings_raw = json.load(f)
    else:
        raise

standings = standings_raw["standings"]

# Playoff clinch codes: p=Presidents, z=div+conf, y=div, x=clinched, e=eliminated
PLAYOFF_CODES = {"p", "z", "y", "x"}

teams_info = {}   # abbrev → full info dict
for s in standings:
    abbrev  = s["teamAbbrev"]["default"]
    clinch  = s.get("clinchIndicator", "")
    in_playoffs = clinch in PLAYOFF_CODES
    teams_info[abbrev] = {
        "abbrev":   abbrev,
        "name":     s["teamCommonName"]["default"],
        "conf":     s["conferenceAbbrev"],
        "div":      s["divisionAbbrev"],
        "div_seq":  s["divisionSequence"],
        "wc_seq":   s["wildcardSequence"],   # 0 = div team, 1-2 = wildcard
        "pts":      s["points"],
        "gp":       s["gamesPlayed"],
        "ga":       s["goalAgainst"],
        "in_playoffs": in_playoffs,
    }

playoff_teams = {a: info for a, info in teams_info.items() if info["in_playoffs"]}
print(f"  {len(playoff_teams)} playoff teams identified")

# ── DETERMINE FIRST-ROUND MATCHUPS ───────────────────────────────────────────
# NHL format: div 2 vs div 3 within division; div 1 vs wildcard cross-division.
# Better conf 1st-seed (more pts among the two div 1sts) gets WC2 (worse WC).
# Worse conf 1st-seed gets WC1 (better WC).

def bracket_for_conference(conf):
    conf_teams = {a: i for a, i in playoff_teams.items() if i["conf"] == conf}
    divs = sorted(set(i["div"] for i in conf_teams.values()))

    div_groups = {}
    for div in divs:
        members = [(a, i) for a, i in conf_teams.items() if i["div"] == div]
        div_members = [x for x in members if x[1]["wc_seq"] == 0]  # non-wildcard
        div_members.sort(key=lambda x: x[1]["div_seq"])
        div_groups[div] = div_members

    wildcards = [(a, i) for a, i in conf_teams.items() if i["wc_seq"] > 0]
    wildcards.sort(key=lambda x: x[1]["wc_seq"])  # WC1 first (better)
    wc1 = wildcards[0] if len(wildcards) > 0 else None
    wc2 = wildcards[1] if len(wildcards) > 1 else None

    matchups = []
    div1_seeds = []  # (abbrev, info) for each division's 1st seed
    for div in divs:
        dm = div_groups[div]
        if len(dm) >= 3:
            # 2 vs 3 within division
            matchups.append(("div_series", dm[1], dm[2], div))
        if dm:
            div1_seeds.append(dm[0])

    # Assign wildcards to division 1 seeds
    # Better div 1 seed (more pts) → gets WC2 (worse wildcard)
    if len(div1_seeds) == 2 and wc1 and wc2:
        div1_seeds.sort(key=lambda x: x[1]["pts"], reverse=True)
        better_div1  = div1_seeds[0]
        worse_div1   = div1_seeds[1]
        matchups.append(("wc_series", better_div1, wc2, "WC"))
        matchups.append(("wc_series", worse_div1,  wc1, "WC"))

    return matchups

east_matchups = bracket_for_conference("E")
west_matchups = bracket_for_conference("W")
all_matchups  = east_matchups + west_matchups

# ── STEP 2: THREE-PILLAR METRICS FROM SHOT DATA (ALL 32 TEAMS) ───────────────
print(f"  Reading nhl_shot_events.csv for {SEASON}...")

# Per-team offensive NF stats (ES 5v5)
team_off = defaultdict(lambda: {"es": 0, "nf": 0})
# Per-team defensive ES stats (what's shot AGAINST each team)
team_def = defaultdict(lambda: {"sog_against": 0, "goals_against_es": 0})

with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["season"] != SEASON or row["situation_code"] != ES_CODE:
            continue
        shooter = row["shooting_team_abbrev"]
        home    = row["home_team_abbrev"]
        away    = row["away_team_abbrev"]
        etype   = row["event_type"]
        is_goal = row["is_goal"] == "1"

        # Offensive NF
        try:
            x = float(row["x_coord_norm"])
            y = float(row["y_coord_norm"])
        except (ValueError, TypeError):
            x = y = None

        team_off[shooter]["es"] += 1
        if x is not None and x >= NF_X_MIN and NF_Y_MIN <= y <= NF_Y_MAX:
            team_off[shooter]["nf"] += 1

        # Defensive (SOG and goals against the defending team)
        if etype in ("shot-on-goal", "goal"):
            defender = away if shooter == home else home
            team_def[defender]["sog_against"] += 1
            if is_goal:
                team_def[defender]["goals_against_es"] += 1

print(f"  Done. {len(team_off)} teams with offensive data, {len(team_def)} with defensive.")

# GA/G from standings (all situations, entire season)
team_ga_pg = {}
for abbrev, info in teams_info.items():
    if info["gp"] > 0:
        team_ga_pg[abbrev] = info["ga"] / info["gp"]

# Compute per-team metrics
def es_save_pct(abbrev):
    d = team_def[abbrev]
    total = d["sog_against"]  # sog already includes goals
    if total == 0:
        return float("nan")
    return 1.0 - d["goals_against_es"] / total

def nf_rate(abbrev):
    o = team_off[abbrev]
    if o["es"] == 0:
        return float("nan")
    return o["nf"] / o["es"]

# All-32-team vectors for percentile thresholds
all_teams_abbrev = list(teams_info.keys())
nf_rates   = {a: nf_rate(a)      for a in all_teams_abbrev}
es_saves   = {a: es_save_pct(a)  for a in all_teams_abbrev}
ga_pgs     = {a: team_ga_pg.get(a, float("nan")) for a in all_teams_abbrev}

def pct_threshold(vals_dict, top_fraction, higher_is_better=True):
    """Return threshold value: top_fraction of 32 teams."""
    vals = sorted([v for v in vals_dict.values() if not math.isnan(v)],
                  reverse=higher_is_better)
    cutoff_idx = max(0, math.ceil(len(vals) * top_fraction) - 1)
    return vals[cutoff_idx]

thr_nf_rate = pct_threshold(nf_rates,  PILLAR_PCT, higher_is_better=True)
thr_es_save = pct_threshold(es_saves,  PILLAR_PCT, higher_is_better=True)
thr_ga_pg   = pct_threshold(ga_pgs,    PILLAR_PCT, higher_is_better=False)  # lower = better

print(f"\n  Pillar thresholds (top 1/3 of 32 teams):")
print(f"    NF attempt rate ≥ {thr_nf_rate:.4f}")
print(f"    ES save%        ≥ {thr_es_save:.4f}")
print(f"    GA/G (total)    ≤ {thr_ga_pg:.4f}")

def pillars(abbrev):
    nr  = nf_rates.get(abbrev, float("nan"))
    es  = es_saves.get(abbrev, float("nan"))
    ga  = ga_pgs.get(abbrev, float("nan"))
    p1  = (not math.isnan(nr) and nr >= thr_nf_rate)
    p2  = (not math.isnan(es) and es >= thr_es_save)
    p3  = (not math.isnan(ga) and ga <= thr_ga_pg)
    return p1, p2, p3

# League rank for each metric
def rank_in_32(abbrev, vals_dict, higher_is_better=True):
    v = vals_dict.get(abbrev, float("nan"))
    if math.isnan(v):
        return "—"
    others = sorted([x for x in vals_dict.values() if not math.isnan(x)],
                    reverse=higher_is_better)
    return others.index(v) + 1

# ── STEP 3: NET-FRONT PLAYERS PER TEAM ───────────────────────────────────────
print("  Reading acquisition_targets.csv...")
nf_players = defaultdict(list)   # team → list of player dicts

with open(os.path.join(DATA_DIR, "acquisition_targets.csv"), newline="") as f:
    for row in csv.DictReader(f):
        tier  = row["tier"]
        conf  = row["sample_conf"]
        team  = row["team_2526"]
        name  = row["player_name"]
        ci    = row["ci_confirmed"] == "True"
        comp  = row.get("career_composite", "0")
        try:
            comp_f = float(comp)
        except ValueError:
            comp_f = 0.0

        is_t1 = tier == "Tier1_healthy"
        is_t2 = tier == "Tier2_watch"
        is_med_plus = conf in ("High", "Medium")

        if is_t1 or (is_t2 and is_med_plus):
            nf_players[team].append({
                "name": name, "tier": tier, "conf": conf,
                "ci": ci, "composite": comp_f,
                "label": "T1" if is_t1 else "T2",
            })

# ── STEP 4: FULL PLAYOFF TEAM SUMMARY TABLE ──────────────────────────────────
SEP  = "═" * 110
SEP2 = "─" * 110

print(f"\n{SEP}")
print(f"  2025-26 NHL PLAYOFF PREVIEW — HockeyROI Three-Pillar & Net-Front Analysis")
print(SEP)

# Sort playoff teams: East then West, by conference seed
def conf_seed(info):
    conf_order = {"E": 0, "W": 1}
    # Div 1 seeds first, then wildcards
    wc = info["wc_seq"]
    return (conf_order[info["conf"]], 0 if wc == 0 else 1,
            info["div_seq"] if wc == 0 else wc)

sorted_playoff = sorted(playoff_teams.values(), key=conf_seed)

print(f"\n  {'Team':<6} {'Conf':<5} {'Seed':<14} {'NF Rate':>8} {'Rk':>4}  "
      f"{'ES Sv%':>7} {'Rk':>4}  {'GA/G':>6} {'Rk':>4}  {'Pillars':>8}  NF Players")
print(f"  {SEP2}")

team_summary = {}   # abbrev → dict for CSV

for info in sorted_playoff:
    a = info["abbrev"]
    nr   = nf_rates.get(a, float("nan"))
    es   = es_saves.get(a, float("nan"))
    ga   = ga_pgs.get(a, float("nan"))
    p1,p2,p3 = pillars(a)
    pcnt = sum([p1,p2,p3])

    rk_nr = rank_in_32(a, nf_rates,  True)
    rk_es = rank_in_32(a, es_saves,  True)
    rk_ga = rank_in_32(a, ga_pgs,    False)

    players = nf_players.get(a, [])
    t1 = [p for p in players if p["label"] == "T1"]
    t2 = [p for p in players if p["label"] == "T2"]

    pillar_str = ("●" if p1 else "○") + ("●" if p2 else "○") + ("●" if p3 else "○")

    # Seed label
    if info["wc_seq"] == 0:
        seed_lbl = f"{info['div']} {info['div_seq']}"
    else:
        seed_lbl = f"{info['conf']}-WC{info['wc_seq']}"

    player_str = ""
    if t1:
        player_str += f"T1: {', '.join(p['name'] for p in t1[:3])}"
        if len(t1) > 3:
            player_str += f" +{len(t1)-3}"
    if t2:
        player_str += (" | " if player_str else "") + f"T2: {', '.join(p['name'] for p in t2[:2])}"
        if len(t2) > 2:
            player_str += f" +{len(t2)-2}"
    if not player_str:
        player_str = "none screened"

    nr_s = f"{nr:.4f}" if not math.isnan(nr) else " —   "
    es_s = f"{es:.4f}" if not math.isnan(es) else " —   "
    ga_s = f"{ga:.3f}"  if not math.isnan(ga) else " —  "

    print(f"  {a:<6} {info['conf']:<5} {seed_lbl:<14} {nr_s:>8} {str(rk_nr):>4}  "
          f"{es_s:>7} {str(rk_es):>4}  {ga_s:>6} {str(rk_ga):>4}  "
          f"  {pillar_str} {pcnt}/3   {player_str}")

    team_summary[a] = {
        "abbrev": a, "name": info["name"], "conf": info["conf"],
        "div": info["div"], "seed_label": seed_lbl,
        "pts": info["pts"], "gp": info["gp"],
        "nf_rate": nr, "nf_rate_rank": rk_nr,
        "es_save_pct": es, "es_save_rank": rk_es,
        "ga_per_game": ga, "ga_rank": rk_ga,
        "pillar_nf": int(p1), "pillar_save": int(p2), "pillar_ga": int(p3),
        "pillar_count": pcnt, "pillar_str": pillar_str,
        "t1_count": len(t1), "t2_medplus_count": len(t2),
        "t1_players": "; ".join(p["name"] for p in t1),
        "t2_players": "; ".join(p["name"] for p in t2),
    }

# Pillar legend
print(f"\n  ● = meets pillar (top 1/3 of 32 teams)  ○ = does not")
print(f"  Pillar order: NF-rate | ES-save% | GA/G")

# ── STEP 4: FIRST-ROUND MATCHUPS ─────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  FIRST-ROUND MATCHUPS — NET-FRONT EDGE ANALYSIS")
print(SEP)

matchup_rows = []  # for CSV

def nf_edge(a_abbrev, b_abbrev):
    """Return the team with stronger net-front profile, or 'push'."""
    sa = team_summary[a_abbrev]
    sb = team_summary[b_abbrev]
    # Score: pillar_count * 2 + t1_count + 0.5*t2_medplus_count + nf_rate rank bonus
    def score(s):
        rk = s["nf_rate_rank"] if isinstance(s["nf_rate_rank"], int) else 33
        return s["pillar_count"] * 2 + s["t1_count"] + 0.5 * s["t2_medplus_count"] + (33 - rk) * 0.1
    sa_score = score(sa)
    sb_score = score(sb)
    if abs(sa_score - sb_score) < 0.3:
        return "push", sa_score, sb_score
    winner = a_abbrev if sa_score > sb_score else b_abbrev
    return winner, sa_score, sb_score

matchup_counter = 0
for mtype, teamA, teamB, div_label in all_matchups:
    matchup_counter += 1
    aA, infoA = teamA
    aB, infoB = teamB
    sA = team_summary[aA]
    sB = team_summary[aB]

    conf = infoA["conf"]
    conf_name = "Eastern" if conf == "E" else "Western"

    if mtype == "div_series":
        series_lbl = f"{div_label} Division — #{infoA['div_seq']} vs #{infoB['div_seq']}"
    else:
        div_winner_div = infoA["div"] if infoA["wc_seq"] == 0 else infoB["div"]
        wc_team        = aB if infoA["wc_seq"] == 0 else aA
        wc_seed        = infoB["wc_seq"] if infoA["wc_seq"] == 0 else infoA["wc_seq"]
        series_lbl = f"{div_winner_div} Division winner vs {conf}-WC{wc_seed}"

    edge_team, scoreA, scoreB = nf_edge(aA, aB)

    print(f"\n  ── Matchup {matchup_counter}: {conf_name} Conference ──────────────────────────────────")
    print(f"     {series_lbl}")
    print(f"     {aA} ({sA['pts']} pts)  vs  {aB} ({sB['pts']} pts)")
    print()

    for abbr, s in [(aA, sA), (aB, sB)]:
        nr_s = f"{s['nf_rate']:.4f}" if not math.isnan(s['nf_rate']) else "  —  "
        es_s = f"{s['es_save_pct']:.4f}" if not math.isnan(s['es_save_pct']) else "  —  "
        ga_s = f"{s['ga_per_game']:.3f}"  if not math.isnan(s['ga_per_game']) else "  —  "
        print(f"     {abbr:<5}  Pillars: {s['pillar_str']} {s['pillar_count']}/3  "
              f"NF={nr_s}(#{s['nf_rate_rank']})  "
              f"ES-Sv={es_s}(#{s['es_save_rank']})  "
              f"GA/G={ga_s}(#{s['ga_rank']})")
        t1_str = ", ".join(s["t1_players"].split("; ")) if s["t1_players"] else "none"
        t2_str = ", ".join(s["t2_players"].split("; ")) if s["t2_players"] else "none"
        print(f"           T1 ({s['t1_count']}): {t1_str}")
        if s["t2_players"]:
            print(f"           T2-Med+ ({s['t2_medplus_count']}): {t2_str}")

    print()
    if edge_team == "push":
        verdict = f"PUSH — near-even net-front profiles (scores {scoreA:.2f} vs {scoreB:.2f})"
    else:
        loser = aB if edge_team == aA else aA
        margin = abs(scoreA - scoreB)
        strength = "CLEAR" if margin > 2 else "SLIGHT"
        verdict = f"{strength} NF EDGE → {edge_team}  (scores {scoreA:.2f} vs {scoreB:.2f})"
    print(f"     ▶  NET-FRONT VERDICT: {verdict}")

    # Watch list: CI-confirmed T1 players from both teams
    watch = []
    for abbr in [aA, aB]:
        for p in nf_players.get(abbr, []):
            if p["label"] == "T1" and p["ci"]:
                watch.append(f"{p['name']} ({abbr})")
    if watch:
        print(f"     ▶  WATCH (CI-confirmed elite): {', '.join(watch)}")

    matchup_rows.append({
        "matchup_num": matchup_counter,
        "conference": conf_name,
        "series_label": series_lbl,
        "team_a": aA, "team_b": aB,
        "pts_a": sA["pts"], "pts_b": sB["pts"],
        "pillars_a": sA["pillar_count"], "pillars_b": sB["pillar_count"],
        "pillar_str_a": sA["pillar_str"], "pillar_str_b": sB["pillar_str"],
        "nf_rate_a": round(sA["nf_rate"], 4) if not math.isnan(sA["nf_rate"]) else "",
        "nf_rate_b": round(sB["nf_rate"], 4) if not math.isnan(sB["nf_rate"]) else "",
        "nf_rank_a": sA["nf_rate_rank"], "nf_rank_b": sB["nf_rate_rank"],
        "es_save_a": round(sA["es_save_pct"], 4) if not math.isnan(sA["es_save_pct"]) else "",
        "es_save_b": round(sB["es_save_pct"], 4) if not math.isnan(sB["es_save_pct"]) else "",
        "es_save_rank_a": sA["es_save_rank"], "es_save_rank_b": sB["es_save_rank"],
        "ga_pg_a": round(sA["ga_per_game"], 3) if not math.isnan(sA["ga_per_game"]) else "",
        "ga_pg_b": round(sB["ga_per_game"], 3) if not math.isnan(sB["ga_per_game"]) else "",
        "ga_rank_a": sA["ga_rank"], "ga_rank_b": sB["ga_rank"],
        "t1_count_a": sA["t1_count"], "t1_count_b": sB["t1_count"],
        "t1_players_a": sA["t1_players"], "t1_players_b": sB["t1_players"],
        "t2_med_count_a": sA["t2_medplus_count"], "t2_med_count_b": sB["t2_medplus_count"],
        "t2_players_a": sA["t2_players"], "t2_players_b": sB["t2_players"],
        "nf_edge_score_a": round(scoreA, 3), "nf_edge_score_b": round(scoreB, 3),
        "nf_verdict": verdict,
        "ci_watch_players": ", ".join(watch),
    })

# ── QUICK SUMMARY TABLE ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  QUICK REFERENCE — ALL 8 FIRST-ROUND MATCHUPS")
print(SEP)
print(f"  {'#':<3} {'Matchup':<15} {'Pillars':^12} {'T1 NF players':^16}  NF Edge")
print(f"  {'':─<3} {'':─<15} {'':─<12} {'':─<16}  {'':─<30}")
for mr in matchup_rows:
    mm = f"{mr['team_a']} vs {mr['team_b']}"
    pp = f"{mr['pillars_a']}/3  vs  {mr['pillars_b']}/3"
    t1 = f"{mr['t1_count_a']} vs {mr['t1_count_b']}"
    vd = mr["nf_verdict"].replace("NET-FRONT VERDICT: ", "")
    # Truncate verdict for table
    edge_short = vd[:45].rstrip()
    print(f"  {mr['matchup_num']:<3} {mm:<15} {pp:^12} {t1:^16}  {edge_short}")

# ── SAVE CSV ──────────────────────────────────────────────────────────────────
# Team-level CSV
team_csv_path = os.path.join(DATA_DIR, "playoff_preview_2526.csv")
if matchup_rows:
    fieldnames = list(matchup_rows[0].keys())
    with open(team_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(matchup_rows)
    print(f"\n  ✓ Saved {len(matchup_rows)} matchup rows → {team_csv_path}")

print(f"\n{SEP}\n")
