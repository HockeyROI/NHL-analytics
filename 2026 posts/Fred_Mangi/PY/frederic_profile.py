#!/usr/bin/env python3
"""
HockeyROI — Trent Frederic: 6-season breakdown + 2025-26 half-season split
Player ID: 8479365
"""

import csv, math, os
from collections import defaultdict

DATA_DIR  = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"
PID       = "8479365"
NAME      = "Trent Frederic"
TARGET_S  = "20252026"
ES_CODE   = "1551"
NF_X_MIN  = 74.0; NF_Y_MIN = -8.0; NF_Y_MAX = 8.0
SEASONS   = ["20202021","20212022","20222023","20232024","20242025","20252026"]
SPLIT_GAME = 60   # first N games = first half, remainder = last half

# ── PASS 1: collect EDM's 2025-26 game schedule, sorted by date ──────────────
# Split is defined by the TEAM schedule: EDM games 1–60 = H1, 61–81 = H2
SPLIT_TEAM = "EDM"
edm_game_dates = {}   # game_id → game_date  (EDM games only)

with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["season"] != TARGET_S:
            continue
        if row["home_team_abbrev"] != SPLIT_TEAM and row["away_team_abbrev"] != SPLIT_TEAM:
            continue
        gid = row["game_id"]
        if gid not in edm_game_dates:
            edm_game_dates[gid] = row["game_date"]

# EDM games sorted by date → game_number 1..N
edm_sorted = sorted(edm_game_dates.keys(), key=lambda g: edm_game_dates[g])
edm_game_num = {gid: i+1 for i, gid in enumerate(edm_sorted)}
total_edm_games = len(edm_sorted)
print(f"  EDM played {total_edm_games} games in 20252026 (in shot data)")

# ── PASS 2: shot events ───────────────────────────────────────────────────────
# Per-season accumulators
season_data = defaultdict(lambda: {
    "es": 0, "nf": 0, "nf_goals": 0,
    "nf_x": [], "nf_y": [], "team": defaultdict(int)
})

# Split: H1 = EDM games 1–60, H2 = EDM games 61–end
h1_gids = {gid for gid, n in edm_game_num.items() if n <= SPLIT_GAME}
h2_gids = {gid for gid, n in edm_game_num.items() if n >  SPLIT_GAME}
n_h2_team = total_edm_games - SPLIT_GAME

print(f"  Split: EDM games 1–{SPLIT_GAME} = H1  |  EDM games {SPLIT_GAME+1}–{total_edm_games} = H2 ({n_h2_team} games)")

with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["shooter_player_id"] != PID:
            continue
        if row["situation_code"] != ES_CODE:
            continue
        season = row["season"]
        gid    = row["game_id"]

        try:
            x = float(row["x_coord_norm"])
            y = float(row["y_coord_norm"])
        except (ValueError, TypeError):
            continue

        is_goal = row["is_goal"] == "1"
        is_nf   = (x >= NF_X_MIN and NF_Y_MIN <= y <= NF_Y_MAX)
        team    = row["shooting_team_abbrev"]

        # Full season bucket
        sd = season_data[season]
        sd["es"] += 1
        sd["team"][team] = sd["team"].get(team, 0) + 1
        if is_nf:
            sd["nf"] += 1
            sd["nf_x"].append(x)
            sd["nf_y"].append(y)
            if is_goal:
                sd["nf_goals"] += 1

# ── PASS 3: shot events again for split buckets ───────────────────────────────
split_data = {
    "h1": {"es": 0, "nf": 0, "nf_goals": 0, "nf_x": [], "nf_y": [], "team": defaultdict(int)},
    "h2": {"es": 0, "nf": 0, "nf_goals": 0, "nf_x": [], "nf_y": [], "team": defaultdict(int)},
}

with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["shooter_player_id"] != PID:
            continue
        if row["situation_code"] != ES_CODE:
            continue
        if row["season"] != TARGET_S:
            continue
        gid = row["game_id"]
        bucket = "h1" if gid in h1_gids else ("h2" if gid in h2_gids else None)
        if bucket is None:
            continue

        try:
            x = float(row["x_coord_norm"])
            y = float(row["y_coord_norm"])
        except (ValueError, TypeError):
            continue

        is_goal = row["is_goal"] == "1"
        is_nf   = (x >= NF_X_MIN and NF_Y_MIN <= y <= NF_Y_MAX)
        team    = row["shooting_team_abbrev"]

        sd = split_data[bucket]
        sd["es"] += 1
        sd["team"][team] = sd["team"].get(team, 0) + 1
        if is_nf:
            sd["nf"] += 1
            sd["nf_x"].append(x)
            sd["nf_y"].append(y)
            if is_goal:
                sd["nf_goals"] += 1

# ── PASS 4: rebound sequences ─────────────────────────────────────────────────
season_gaps   = defaultdict(list)
split_gaps    = {"h1": [], "h2": []}

with open(os.path.join(DATA_DIR, "rebound_sequences.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["reb_shooter_id"] != PID:
            continue
        season = row["season"]
        gid    = row.get("game_id", "")
        try:
            gap = float(row["time_gap_secs"])
        except ValueError:
            continue
        season_gaps[season].append(gap)

        if season == TARGET_S:
            bucket = "h1" if gid in h1_gids else ("h2" if gid in h2_gids else None)
            if bucket:
                split_gaps[bucket].append(gap)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def safe_div(a, b):
    return a / b if b else 0.0

def fmt(sd, gaps):
    """Return display dict from a shot-data bucket + gap list."""
    es  = sd["es"]
    nf  = sd["nf"]
    nfg = sd["nf_goals"]
    nr  = safe_div(nf, es)
    gr  = safe_div(nfg, nf)
    ax  = sum(sd["nf_x"]) / len(sd["nf_x"]) if sd["nf_x"] else float("nan")
    ay  = sum(sd["nf_y"]) / len(sd["nf_y"]) if sd["nf_y"] else float("nan")
    ag  = sum(gaps) / len(gaps) if gaps else float("nan")
    team = max(sd["team"], key=sd["team"].get) if sd["team"] else "—"
    return {"es":es,"nf":nf,"nf_goals":nfg,"nf_rate":nr,"nf_gr":gr,
            "avg_x":ax,"avg_y":ay,"avg_gap":ag,"n_reb":len(gaps),"team":team}

def row_str(label, team, d, marker=""):
    x_s   = f"{d['avg_x']:6.1f}" if not math.isnan(d['avg_x'])  else "   —  "
    y_s   = f"{d['avg_y']:6.1f}" if not math.isnan(d['avg_y'])  else "   —  "
    g_s   = f"{d['avg_gap']:6.2f}s" if not math.isnan(d['avg_gap']) else "    —  "
    return (f"  {label:<18} {team:<5} {d['es']:>7,} {d['nf']:>7} "
            f"{d['nf_rate']:>8.4f} {d['nf_goals']:>9} {d['nf_gr']:>7.4f} "
            f"{x_s} {y_s} {g_s} {d['n_reb']:>10}{marker}")

# ── BUILD SEASON ROWS ─────────────────────────────────────────────────────────
rows = []
for season in SEASONS:
    sd   = season_data[season]
    gaps = season_gaps[season]
    if sd["es"] == 0 and not gaps:
        continue
    team = max(sd["team"], key=sd["team"].get) if sd["team"] else "—"
    rows.append((season, team, fmt(sd, gaps)))

# Career totals
tot_es  = sum(r[2]["es"]       for r in rows)
tot_nf  = sum(r[2]["nf"]       for r in rows)
tot_nfg = sum(r[2]["nf_goals"] for r in rows)
all_x   = [x for s in SEASONS for x in season_data[s]["nf_x"]]
all_y   = [y for s in SEASONS for y in season_data[s]["nf_y"]]
all_gaps= [g for s in SEASONS for g in season_gaps[s]]
car = {
    "es": tot_es, "nf": tot_nf, "nf_goals": tot_nfg,
    "nf_rate": safe_div(tot_nf, tot_es),
    "nf_gr":   safe_div(tot_nfg, tot_nf),
    "avg_x":   sum(all_x)/len(all_x)       if all_x    else float("nan"),
    "avg_y":   sum(all_y)/len(all_y)       if all_y    else float("nan"),
    "avg_gap": sum(all_gaps)/len(all_gaps)  if all_gaps else float("nan"),
    "n_reb":   len(all_gaps),
}

# ── PRINT ─────────────────────────────────────────────────────────────────────
SEP  = "═" * 100
SEP2 = "─" * 100

print(f"\n{SEP}")
print(f"  {NAME}  —  Full Seasonal Breakdown  (ES 5v5, situation_code={ES_CODE})")
print(f"  Player ID: {PID}  |  Current team: EDM  |  Acquisition: Tier 2 Watch, Low confidence, rising")
print(SEP)

header = (f"\n  {'Season':<18} {'Tm':<5} {'ES att':>7} {'NF att':>7} {'NF rate':>8} "
          f"{'NF goals':>9} {'NF GR':>7} {'Avg X':>7} {'Avg Y':>7} {'Avg Gap':>8} {'Reb shots':>10}")
print(header)
print(f"  {SEP2}")

for season, team, d in rows:
    marker = "  ◄" if season == TARGET_S else ""
    print(row_str(season, team, d, marker))

print(f"  {SEP2}")
print(row_str("CAREER (6 seasons)", "", car))

# ── 2025-26 HALF-SEASON SPLIT ─────────────────────────────────────────────────
h1d = fmt(split_data["h1"], split_gaps["h1"])
h2d = fmt(split_data["h2"], split_gaps["h2"])

print(f"\n{SEP}")
print(f"  2025-26 HALF-SEASON SPLIT  (EDM team schedule, ES 5v5)")
print(SEP)
print(f"\n  Split: EDM games 1–{SPLIT_GAME} = H1  |  EDM games {SPLIT_GAME+1}–{total_edm_games} = H2")
print(f"  (Frederic's shots are bucketed by which EDM game they occurred in)\n")
print(header)
print(f"  {SEP2}")
print(row_str(f"H1 (EDM gms 1–{SPLIT_GAME})",   h1d["team"], h1d))
print(row_str(f"H2 (EDM gms {SPLIT_GAME+1}–{total_edm_games})", h2d["team"], h2d))
print(f"  {SEP2}")
print(row_str("2025-26 FULL",   max(season_data[TARGET_S]["team"],
                                    key=season_data[TARGET_S]["team"].get),
              fmt(season_data[TARGET_S], season_gaps[TARGET_S])))

# ── HALF-SPLIT DELTA TABLE ────────────────────────────────────────────────────
print(f"\n  H2 vs H1 Delta:")
print(f"  {'Metric':<28}  {'H1':>10}  {'H2':>10}  {'Change':>18}")
print(f"  {'-'*72}")

def delta_row(label, v1, v2, unit="", pct=False):
    if pct:
        s1, s2 = f"{v1*100:.2f}%", f"{v2*100:.2f}%"
    elif unit == "s":
        s1, s2 = f"{v1:.2f}s", f"{v2:.2f}s"
    elif unit == "coord":
        s1, s2 = f"{v1:.1f}", f"{v2:.1f}"
    else:
        s1 = f"{v1:,.0f}" if v1 >= 10 else f"{v1:.4f}"
        s2 = f"{v2:,.0f}" if v2 >= 10 else f"{v2:.4f}"

    if v1 == 0:
        chg = "  n/a"
    else:
        d   = v2 - v1
        pct_chg = d / v1 * 100
        arrow = "▲" if d > 0 else "▼" if d < 0 else "—"
        if pct:
            chg = f"  {arrow}{abs(d)*100:.1f}pp  ({pct_chg:+.1f}%)"
        elif unit in ("s","coord"):
            chg = f"  {arrow}{abs(d):.2f}  ({pct_chg:+.1f}%)"
        else:
            chg = f"  {arrow}{abs(d):.1f}  ({pct_chg:+.1f}%)"
    print(f"  {label:<28}  {s1:>10}  {s2:>10}  {chg}")

delta_row("ES shot attempts",         h1d["es"],       h2d["es"])
delta_row("NF attempts",              h1d["nf"],       h2d["nf"])
delta_row("NF attempt rate",          h1d["nf_rate"],  h2d["nf_rate"],  pct=True)
delta_row("NF goals",                 h1d["nf_goals"], h2d["nf_goals"])
delta_row("NF goal rate",             h1d["nf_gr"],    h2d["nf_gr"],    pct=True)
delta_row("Avg x_coord",              h1d["avg_x"]  if not math.isnan(h1d["avg_x"])  else 0,
                                      h2d["avg_x"]  if not math.isnan(h2d["avg_x"])  else 0,  unit="coord")
delta_row("Avg y_coord",              h1d["avg_y"]  if not math.isnan(h1d["avg_y"])  else 0,
                                      h2d["avg_y"]  if not math.isnan(h2d["avg_y"])  else 0,  unit="coord")
delta_row("Avg time gap",             h1d["avg_gap"] if not math.isnan(h1d["avg_gap"]) else 0,
                                      h2d["avg_gap"] if not math.isnan(h2d["avg_gap"]) else 0, unit="s")
delta_row("Rebound shots",            h1d["n_reb"],    h2d["n_reb"])

# ── ACQUISITION STATUS ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  ACQUISITION STATUS — from Data/acquisition_targets.csv")
print(SEP)
print(f"""
  Tier:              Tier 2 Watch
  Sample confidence: Low  (16 rebound shots in 2025-26; minimum for screening = 15)
  CI confirmed:      No   (career composite 0.6556 — below CI threshold)
  Trend:             Rising (NF attempt rate 31.6% in 2526 vs 23.3% career — ▲36%)
  Active flags:      rate_flag = True
                     Reason: 2025-26 NF rate 0.316 vs career 0.233 (▲36%) — rate spike
  Gap flag:          False  (time gap 1.31s vs career 1.49s — within ±0.25s threshold)
  GR flag:           False  (only 16 reb shots — below 30-shot minimum for GR flag)
  Career composite:  0.6556
  2025-26 composite: 0.4279  (pulled down by low rebound volume and 12.5% GR on 8 shots)
""")
print(SEP)
print()
