"""Compute RelTNZI% — on-ice minus off-ice neutral-zone TNZI for every player.

RelTNZI% = TNZI of team while player is on ice
           – TNZI of team while teammates are on ice and player is off

Mirrors the standard RelCF% construction. Wilson 95% CIs applied to the OZ%
(lower) and DZ% (upper) components on the on-ice and off-ice halves separately,
using NZ-faceoff-shift-count as n. Min 50 on-ice NZ FO shifts and min 20 GP.

Reads:
  Zones/raw/pbp/{game}.json
  Zones/raw/shifts/{game}.json
  Zones/output/_player_meta.json
  Data/game_ids.csv
  Zones/adjusted_rankings/tnzi_winning_correlation.csv  (for comparison row)
  NFI/output/fully_adjusted/player_fully_adjusted.csv   (for combined model)

Writes:
  Zones/adjusted_rankings/rel_tnzi_forwards.csv
  Zones/adjusted_rankings/rel_tnzi_defense.csv

Run:
  python3 Zones/scripts/compute_rel_tnzi.py
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

# Optional pandas/numpy for OLS
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ZONES = HERE.parent
ROOT = ZONES.parent
RAW_PBP = ZONES / "raw" / "pbp"
RAW_SHIFTS = ZONES / "raw" / "shifts"
PLAYER_META = ZONES / "output" / "_player_meta.json"
GAME_IDS = ROOT / "Data" / "game_ids.csv"
OUT_DIR = ZONES / "adjusted_rankings"
TNZI_CORR_CSV = OUT_DIR / "tnzi_winning_correlation.csv"
NFI_FILE = ROOT / "NFI" / "output" / "fully_adjusted" / "player_fully_adjusted.csv"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
SEASONS = ["20222023", "20232024", "20242025", "20252026"]
SEASON_END_DATE = {
    "20222023": "2023-04-13",
    "20232024": "2024-04-18",
    "20242025": "2025-04-17",
    "20252026": "2026-04-17",
}
SEASON_LABEL = {"20222023": "22/23", "20232024": "23/24",
                "20242025": "24/25", "20252026": "25/26"}

ABBR_MAP = {"ARI": "UTA"}
def norm_team(a): return ABBR_MAP.get(a, a)

V1_EVENTS = {"faceoff", "hit", "shot-on-goal", "missed-shot",
             "blocked-shot", "goal", "giveaway", "takeaway"}

Z = 1.96
Z2 = Z * Z
MIN_ON_SHIFTS = 50
MIN_GP = 20

POS_FORWARD = {"C", "L", "R"}
POS_DEFENSE = {"D"}

FLIP = {"O": "D", "D": "O", "N": "N"}

KEY_PLAYERS = ["McDavid", "MacKinnon", "Draisaitl", "Makar",
               "Hughes", "Nurse", "Bouchard", "Ekholm", "Bedard"]

# -----------------------------------------------------------------------------
# Time / Wilson helpers
# -----------------------------------------------------------------------------
def mmss(s):
    if not s or ":" not in s: return 0
    m, ss = s.split(":")
    try: return int(m) * 60 + int(ss)
    except ValueError: return 0

def play_abs_time(p):
    period = (p.get("periodDescriptor") or {}).get("number", 1) or 1
    return (period - 1) * 1200 + mmss(p.get("timeInPeriod", "00:00"))

def wilson_lower(p, n):
    if n is None or n <= 0: return None
    p = max(0.0, min(1.0, p))
    denom = 1 + Z2 / n
    center = p + Z2 / (2 * n)
    margin = Z * math.sqrt(max(0.0, p * (1 - p) / n + Z2 / (4 * n * n)))
    return (center - margin) / denom

def wilson_upper(p, n):
    if n is None or n <= 0: return None
    p = max(0.0, min(1.0, p))
    denom = 1 + Z2 / n
    center = p + Z2 / (2 * n)
    margin = Z * math.sqrt(max(0.0, p * (1 - p) / n + Z2 / (4 * n * n)))
    return (center + margin) / denom

def pearson(x, y):
    n = len(x)
    if n < 2: return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")

# -----------------------------------------------------------------------------
# Standings
# -----------------------------------------------------------------------------
def fetch_standings(date_str):
    d0 = datetime.strptime(date_str, "%Y-%m-%d")
    for off in range(5):
        d_try = (d0 - timedelta(days=off)).strftime("%Y-%m-%d")
        out = subprocess.run(
            ["curl", "-s", "-m", "20",
             f"https://api-web.nhle.com/v1/standings/{d_try}"],
            capture_output=True, text=True, check=True,
        )
        try: d = json.loads(out.stdout)
        except json.JSONDecodeError: continue
        if d.get("standings"):
            return {norm_team(r["teamAbbrev"]["default"]): r["points"]
                    for r in d["standings"]}
    return {}

# -----------------------------------------------------------------------------
# Per-game RelTNZI processor
# -----------------------------------------------------------------------------
def build_shift_intervals(shifts_json):
    out = defaultdict(list)
    for s in shifts_json.get("data", []):
        pid = s.get("playerId")
        if not pid: continue
        period = s.get("period") or 1
        st = s.get("startTime") or "00:00"
        et = s.get("endTime") or "00:00"
        a = (period - 1) * 1200 + mmss(st)
        b = (period - 1) * 1200 + mmss(et)
        if b <= a: continue
        out[pid].append((a, b, s.get("teamId")))
    return out

def player_on_ice_at(intervals_by_pid, t):
    """Return dict team_id -> set(pid) on ice at absolute time t."""
    on = defaultdict(set)
    for pid, ivs in intervals_by_pid.items():
        for a, b, tm in ivs:
            if a <= t < b:
                on[tm].add(pid)
                break
    return on

def process_game(game_id, buckets, gp_counter):
    """Update buckets in place.

    buckets[(pid, season)] = {
       "on": {"shifts","total","oz","dz","nz"},
       "off":{"shifts","total","oz","dz","nz"},
       "team": team_id, "team_abbrev": str
    }
    gp_counter[(pid, season)] += 1 once per game player appeared in.
    """
    pbp_path = RAW_PBP / f"{game_id}.json"
    sh_path = RAW_SHIFTS / f"{game_id}.json"
    if not pbp_path.exists() or not sh_path.exists():
        return
    try:
        pbp = json.load(open(pbp_path))
        shifts = json.load(open(sh_path))
    except (json.JSONDecodeError, OSError):
        return

    season = str(pbp.get("season", ""))
    home_id = (pbp.get("homeTeam") or {}).get("id")
    away_id = (pbp.get("awayTeam") or {}).get("id")
    home_abbr = norm_team((pbp.get("homeTeam") or {}).get("abbrev") or "")
    away_abbr = norm_team((pbp.get("awayTeam") or {}).get("abbrev") or "")
    if home_id is None or away_id is None:
        return

    intervals_by_pid = build_shift_intervals(shifts)

    # Roster per team (anyone who took a shift)
    roster_by_team = defaultdict(set)
    pid_team = {}
    for pid, ivs in intervals_by_pid.items():
        if not ivs: continue
        tm = ivs[0][2]
        if tm in (home_id, away_id):
            roster_by_team[tm].add(pid)
            pid_team[pid] = tm

    for pid in roster_by_team[home_id] | roster_by_team[away_id]:
        gp_counter[(pid, season)] += 1

    plays = pbp.get("plays") or []

    # Walk plays, identify NZ 5v5 faceoff contexts; close at next faceoff or
    # when situation changes off "1551".
    ctx = None

    def close(ctx, close_t):
        if ctx is None: return
        events = ctx["events"]  # already V1-filtered with (t, ez_home)
        if not events: return

        # Compute zone-time per event-zone for HOME perspective
        # Then per-team flip when assigning to players
        # team_zone_time_home = {"O":sec,"D":sec,"N":sec}
        per_team_z = {home_id: {"O": 0.0, "D": 0.0, "N": 0.0},
                      away_id: {"O": 0.0, "D": 0.0, "N": 0.0}}
        # For each consecutive pair of V1 events within ctx, dwell time
        # attributed to the earlier event's zone (in HOME perspective).
        # Cap at close_t.
        for i, (t, ez_home) in enumerate(events):
            t_next = events[i + 1][0] if i + 1 < len(events) else close_t
            dt = max(0.0, t_next - t)
            if dt <= 0: continue
            # HOME perspective zone = ez_home; AWAY = FLIP[ez_home]
            per_team_z[home_id][ez_home] += dt
            per_team_z[away_id][FLIP[ez_home]] += dt

        # For each team, classify each rostered player as on-ice or off-ice
        for team_id in (home_id, away_id):
            tot = sum(per_team_z[team_id].values())
            if tot <= 0:
                continue
            oz = per_team_z[team_id]["O"]
            dz = per_team_z[team_id]["D"]
            nz = per_team_z[team_id]["N"]
            on_set = ctx["on_ice"].get(team_id, set())
            for pid in roster_by_team[team_id]:
                key = (pid, season)
                b = buckets[key]
                b["team"] = team_id
                b["team_abbrev"] = (home_abbr if team_id == home_id else away_abbr)
                target = b["on"] if pid in on_set else b["off"]
                target["shifts"] += 1
                target["total"] += tot
                target["oz"] += oz
                target["dz"] += dz
                target["nz"] += nz

    for p in plays:
        typ = p.get("typeDescKey") or ""
        details = p.get("details") or {}
        situation = p.get("situationCode") or ""
        t_abs = play_abs_time(p)

        if typ == "faceoff":
            # Close prior context
            if ctx is not None:
                close(ctx, t_abs)
                ctx = None

            if situation != "1551":
                continue
            zone = details.get("zoneCode")
            if zone != "N":  # Only NEUTRAL zone faceoffs for RelTNZI
                continue
            owner = details.get("eventOwnerTeamId")
            if owner not in (home_id, away_id):
                continue
            # Faceoff zone is N from any perspective (N == N flipped)
            on_ice = player_on_ice_at(intervals_by_pid, t_abs)
            ctx = {
                "fo_t": t_abs,
                "events": [(t_abs, "N")],  # the faceoff itself
                "on_ice": {home_id: set(on_ice.get(home_id, set())),
                           away_id: set(on_ice.get(away_id, set()))},
            }
            continue

        if ctx is None:
            continue

        if situation and situation != "1551":
            close(ctx, t_abs)
            ctx = None
            continue

        zone = details.get("zoneCode")
        if zone in ("O", "D", "N") and typ in V1_EVENTS:
            owner = details.get("eventOwnerTeamId")
            if owner in (home_id, away_id):
                ez_home = zone if owner == home_id else FLIP[zone]
                ctx["events"].append((t_abs, ez_home))

        if typ in ("period-end", "game-end"):
            close(ctx, t_abs)
            ctx = None

    if ctx is not None and plays:
        close(ctx, play_abs_time(plays[-1]))

# -----------------------------------------------------------------------------
# Aggregation -> RelTNZI per scenario per player
# -----------------------------------------------------------------------------
def empty_split():
    return {"shifts": 0, "total": 0.0, "oz": 0.0, "dz": 0.0, "nz": 0.0}

def empty_bucket():
    return {"on": empty_split(), "off": empty_split(),
            "team": None, "team_abbrev": ""}

def aggregate(buckets, gp_counter, seasons_filter):
    """Sum buckets across the requested seasons. Return:
       {pid: {"on":..., "off":..., "gp":int, "team_abbrev":str}}
    """
    out = defaultdict(empty_bucket)
    out = {}
    gp = defaultdict(int)
    for (pid, season), gpc in gp_counter.items():
        if season in seasons_filter:
            gp[pid] += gpc
    for (pid, season), b in buckets.items():
        if season not in seasons_filter:
            continue
        if pid not in out:
            out[pid] = empty_bucket()
        for half in ("on", "off"):
            for k in ("shifts", "total", "oz", "dz", "nz"):
                out[pid][half][k] += b[half][k]
        # team abbrev from latest season represented
        if b["team_abbrev"]:
            out[pid]["team_abbrev"] = b["team_abbrev"]
            out[pid]["team"] = b["team"]
    # Attach gp
    for pid in out:
        out[pid]["gp"] = gp.get(pid, 0)
    return out

def compute_reltnzi(b):
    """Given a player bucket {"on":{shifts,total,oz,dz,nz},"off":...},
    return (raw_on, raw_off, on_wilson, off_wilson, rel_raw, rel_wilson)
    or None if disqualified."""
    on = b["on"]; off = b["off"]
    if on["shifts"] < MIN_ON_SHIFTS: return None
    if on["total"] <= 0: return None

    p_on_oz = on["oz"] / on["total"]
    p_on_dz = on["dz"] / on["total"]
    raw_on = p_on_oz - p_on_dz
    on_wilson_lo = wilson_lower(p_on_oz, on["shifts"])
    on_wilson_hi = wilson_upper(p_on_dz, on["shifts"])
    on_wilson = (on_wilson_lo - on_wilson_hi) if (on_wilson_lo is not None and on_wilson_hi is not None) else None

    if off["shifts"] > 0 and off["total"] > 0:
        p_off_oz = off["oz"] / off["total"]
        p_off_dz = off["dz"] / off["total"]
        raw_off = p_off_oz - p_off_dz
        off_wilson_lo = wilson_lower(p_off_oz, off["shifts"])
        off_wilson_hi = wilson_upper(p_off_dz, off["shifts"])
        off_wilson = (off_wilson_lo - off_wilson_hi) if (off_wilson_lo is not None and off_wilson_hi is not None) else None
    else:
        raw_off = 0.0
        off_wilson = 0.0

    rel_raw = raw_on - raw_off
    rel_wilson = on_wilson - off_wilson if (on_wilson is not None and off_wilson is not None) else None
    return raw_on, raw_off, on_wilson, off_wilson, rel_raw, rel_wilson

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def load_game_ids():
    out = []
    with open(GAME_IDS) as f:
        for r in csv.DictReader(f):
            if r["season"] in SEASONS and r["game_type"] == "regular":
                out.append((int(r["game_id"]), r["season"]))
    return out

def normalize_to_score(rows_with_val):
    vals = [v for _, v in rows_with_val if v is not None]
    if not vals:
        return {k: None for k, _ in rows_with_val}
    lo, hi = min(vals), max(vals); span = hi - lo
    out = {}
    for k, v in rows_with_val:
        if v is None: out[k] = None
        elif span == 0: out[k] = 5.0
        else: out[k] = round((v - lo) / span * 10.0, 1)
    return out

def main():
    print("[1/8] loading player meta ...")
    player_meta = {int(k): v for k, v in json.load(open(PLAYER_META)).items()}
    print(f"    {len(player_meta)} players")

    print("[2/8] reading game_ids.csv ...")
    games = load_game_ids()
    print(f"    {len(games)} regular games (4 seasons)")

    print("[3/8] processing games to build on-ice / off-ice NZ-FO buckets ...")
    # buckets[(pid, season)] = on/off split bundle
    buckets = defaultdict(empty_bucket)
    gp_counter = defaultdict(int)
    total = len(games)
    for i, (gid, season) in enumerate(games):
        process_game(gid, buckets, gp_counter)
        if (i + 1) % 500 == 0 or i + 1 == total:
            print(f"    {i+1}/{total}")

    print("[4/8] aggregating RelTNZI per scenario ...")
    scenarios = {
        "pooled":  set(SEASONS),
        "current": {"20252026"},
    }
    per_season = {s: aggregate(buckets, gp_counter, {s}) for s in SEASONS}
    scen_data = {name: aggregate(buckets, gp_counter, ss) for name, ss in scenarios.items()}

    # Compute Rel values per scenario
    scen_metrics = {}
    for name, data in scen_data.items():
        rows = {}
        for pid, b in data.items():
            res = compute_reltnzi(b)
            if res is None: continue
            if b["gp"] < MIN_GP: continue
            raw_on, raw_off, on_w, off_w, rel_raw, rel_wilson = res
            rows[pid] = dict(
                raw_on=raw_on, raw_off=raw_off, on_wilson=on_w, off_wilson=off_w,
                rel_raw=rel_raw, rel_wilson=rel_wilson, gp=b["gp"],
                team=b["team_abbrev"],
                on_shifts=b["on"]["shifts"], off_shifts=b["off"]["shifts"])
        scen_metrics[name] = rows

    # Per-season RelTNZI for season-by-season correlation
    per_season_metrics = {}
    for season, data in per_season.items():
        rows = {}
        for pid, b in data.items():
            res = compute_reltnzi(b)
            if res is None: continue
            if b["gp"] < MIN_GP: continue
            raw_on, raw_off, on_w, off_w, rel_raw, rel_wilson = res
            rows[pid] = dict(rel_wilson=rel_wilson, team=b["team_abbrev"], gp=b["gp"])
        per_season_metrics[season] = rows

    print("[5/8] normalising and writing player CSVs ...")
    # For pooled scenario, write CSVs by position group, using existing TNZI files
    # for raw / TNZI_L lookup.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tnzi_pooled_lookup = {}
    for grp in ("forwards", "defense"):
        df = pd.read_csv(OUT_DIR / f"tnzi_adjusted_{grp}.csv",
                         usecols=["player_name", "team", "pos", "TNZI", "TNZI_L"])
        for _, r in df.iterrows():
            tnzi_pooled_lookup[(r["player_name"], r["team"])] = (r["TNZI"], r["TNZI_L"])

    pooled_rows = scen_metrics["pooled"]
    by_group = {"forwards": [], "defense": []}
    pid_to_score = {}
    # Score normalization within each group
    for grp_name, pos_set in (("forwards", POS_FORWARD), ("defense", POS_DEFENSE)):
        subset = []
        for pid, m in pooled_rows.items():
            meta = player_meta.get(pid, {})
            pos = (meta.get("position") or "").upper()
            if pos not in pos_set: continue
            subset.append((pid, m["rel_wilson"]))
        scores = normalize_to_score(subset)
        for pid, val in subset:
            pid_to_score[pid] = scores.get(pid)

        for pid, m in pooled_rows.items():
            meta = player_meta.get(pid, {})
            pos = (meta.get("position") or "").upper()
            if pos not in pos_set: continue
            tnzi_raw, tnzi_L = tnzi_pooled_lookup.get(
                (meta.get("name", ""), m["team"]), (None, None))
            row = {
                "player_id": pid,
                "player_name": meta.get("name", ""),
                "team": m["team"],
                "pos": pos,
                "GP": m["gp"],
                "on_NZ_fo_shifts": m["on_shifts"],
                "off_NZ_fo_shifts": m["off_shifts"],
                "TNZI_raw": tnzi_raw if tnzi_raw is not None else "",
                "TNZI_L": tnzi_L if tnzi_L is not None else "",
                "RelTNZI_raw": round(m["rel_raw"], 5) if m["rel_raw"] is not None else "",
                "RelTNZI_wilson": round(m["rel_wilson"], 5) if m["rel_wilson"] is not None else "",
                "RelTNZI_score": pid_to_score.get(pid, ""),
            }
            by_group[grp_name].append(row)

        path = OUT_DIR / f"rel_tnzi_{grp_name}.csv"
        rows_sorted = sorted(by_group[grp_name],
                              key=lambda r: (r["RelTNZI_score"]
                                             if r["RelTNZI_score"] != "" else -999),
                              reverse=True)
        # Tag rank in-place
        for rank, r in enumerate(rows_sorted, 1):
            r["rank"] = rank
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "player_name", "team", "pos", "GP",
                        "on_NZ_fo_shifts", "off_NZ_fo_shifts",
                        "TNZI_raw", "TNZI_L", "RelTNZI_raw", "RelTNZI_wilson",
                        "RelTNZI_score", "player_id"])
            for r in rows_sorted:
                w.writerow([r["rank"], r["player_name"], r["team"], r["pos"], r["GP"],
                            r["on_NZ_fo_shifts"], r["off_NZ_fo_shifts"],
                            r["TNZI_raw"], r["TNZI_L"],
                            r["RelTNZI_raw"], r["RelTNZI_wilson"],
                            r["RelTNZI_score"], r["player_id"]])
        by_group[grp_name] = rows_sorted
        print(f"    wrote {path}  ({len(rows_sorted)} players)")

    print("[6/8] correlating with team points ...")
    standings = {s: fetch_standings(SEASON_END_DATE[s]) for s in SEASONS}
    pooled_pts = defaultdict(list)
    for s in SEASONS:
        for tm, p in standings[s].items():
            pooled_pts[tm].append(p)
    pooled_pts = {tm: mean(v) for tm, v in pooled_pts.items()}

    # Read pre-existing TNZI per-season + pooled correlation rows
    tnzi_corr = defaultdict(dict)
    for r in csv.DictReader(open(TNZI_CORR_CSV)):
        tnzi_corr[(r["scenario"], r["variant"])] = {
            "r": float(r["pearson_r"]), "r2": float(r["r_squared"])}

    # Compute team-avg RelTNZI per scenario and per season
    def team_avg(rows):
        per_team = defaultdict(list)
        for pid, m in rows.items():
            v = m.get("rel_wilson")
            if v is not None:
                per_team[m["team"]].append(v)
        return {tm: mean(v) for tm, v in per_team.items() if v}

    rel_pooled_team = team_avg(pooled_rows)
    rel_per_season_team = {s: team_avg(per_season_metrics[s]) for s in SEASONS}

    def corr_team(team_metric, points_map):
        xs, ys = [], []
        for tm, v in team_metric.items():
            if tm in points_map:
                xs.append(v); ys.append(points_map[tm])
        return pearson(xs, ys), len(xs)

    rel_per_season_r = {}
    for s in SEASONS:
        r, _ = corr_team(rel_per_season_team[s], standings[s])
        rel_per_season_r[s] = r
    rel_pooled_r, n_pool = corr_team(rel_pooled_team, pooled_pts)

    # Print comparison table
    print("\n" + "=" * 90)
    print("COMPARISON — Pearson r vs team points (n=32 per season; pooled = 4-season-avg, n=32)")
    print("=" * 90)
    hdr = (f"{'Metric':<14} {'22/23 r':>9} {'23/24 r':>9} {'24/25 r':>9} "
           f"{'25/26 r':>9} {'Pooled r':>9}")
    print(hdr); print("-" * 90)

    def fmt(v):
        return f"{v:+.4f}" if isinstance(v, float) and not math.isnan(v) else "    -   "

    # TNZI raw / L from existing CSV
    for variant, label in [("raw", "TNZI raw"), ("L", "TNZI_L")]:
        per_s = [tnzi_corr.get((s, variant), {}).get("r", float("nan")) for s in SEASONS]
        pool_r = tnzi_corr.get(("pooled", variant), {}).get("r", float("nan"))
        print(f"{label:<14} {fmt(per_s[0]):>9} {fmt(per_s[1]):>9} {fmt(per_s[2]):>9} "
              f"{fmt(per_s[3]):>9} {fmt(pool_r):>9}")

    rel_per_s = [rel_per_season_r[s] for s in SEASONS]
    print(f"{'RelTNZI%':<14} {fmt(rel_per_s[0]):>9} {fmt(rel_per_s[1]):>9} "
          f"{fmt(rel_per_s[2]):>9} {fmt(rel_per_s[3]):>9} {fmt(rel_pooled_r):>9}")

    # ---------------------- Player rankings -----------------------------------
    print("\n" + "=" * 90)
    print("TOP 20 FORWARDS by RelTNZI (pooled)")
    print("=" * 90)
    def _fmt_num(v):
        if v == "" or v is None: return "   -   "
        try: return f"{float(v):+.4f}"
        except (TypeError, ValueError): return "   -   "

    print(f"{'#':>3} {'Player':<24} {'Team':<5} {'Pos':<4} {'GP':>4} "
          f"{'TNZI':>8} {'TNZI_L':>8} {'RelTNZI':>9} {'Score':>5}")
    for r in by_group["forwards"][:20]:
        print(f"{r['rank']:>3} {r['player_name'][:24]:<24} {r['team']:<5} {r['pos']:<4} "
              f"{r['GP']:>4} {_fmt_num(r['TNZI_raw']):>8} "
              f"{_fmt_num(r['TNZI_L']):>8} "
              f"{r['RelTNZI_wilson']:>+9.4f} {r['RelTNZI_score']:>5}")

    print("\n" + "=" * 90)
    print("TOP 20 DEFENSEMEN by RelTNZI (pooled)")
    print("=" * 90)
    print(f"{'#':>3} {'Player':<24} {'Team':<5} {'Pos':<4} {'GP':>4} "
          f"{'TNZI':>8} {'TNZI_L':>8} {'RelTNZI':>9} {'Score':>5}")
    for r in by_group["defense"][:20]:
        print(f"{r['rank']:>3} {r['player_name'][:24]:<24} {r['team']:<5} {r['pos']:<4} "
              f"{r['GP']:>4} {_fmt_num(r['TNZI_raw']):>8} "
              f"{_fmt_num(r['TNZI_L']):>8} "
              f"{r['RelTNZI_wilson']:>+9.4f} {r['RelTNZI_score']:>5}")

    print("\n" + "=" * 90)
    print("KEY PLAYERS (pooled)")
    print("=" * 90)
    all_rows = by_group["forwards"] + by_group["defense"]
    for label in KEY_PLAYERS:
        for r in all_rows:
            if label.lower() in r["player_name"].lower():
                print(f"  {r['player_name']:<24} {r['team']:<4} {r['pos']:<3} "
                      f"GP={r['GP']:>3} on_fs={r['on_NZ_fo_shifts']:>4} off_fs={r['off_NZ_fo_shifts']:>5} "
                      f"TNZI={r['TNZI_raw']} TNZI_L={r['TNZI_L']} "
                      f"RelTNZI={r['RelTNZI_wilson']:+.4f} score={r['RelTNZI_score']}")

    # =========================================================================
    # Step 6 — combined model with RelNFI%
    # =========================================================================
    print("\n[7/8] running combined model with RelNFI% ...")
    nfi = pd.read_csv(NFI_FILE,
                      usecols=["player_name", "team", "season",
                               "RelNFI_pct", "toi_min"])
    nfi["season"] = nfi["season"].astype(str)
    nfi["team"] = nfi["team"].map(norm_team)

    # Build per-season player-level RelTNZI in pct (rel_wilson * 100 to match RelNFI scale)
    rows_join = []
    for season, prs in per_season_metrics.items():
        # Need player_name per pid
        for pid, m in prs.items():
            meta = player_meta.get(pid, {})
            name = meta.get("name") or ""
            if not name: continue
            rows_join.append({
                "player_name": name,
                "team": m["team"],
                "season": season,
                "RelTNZI_pct": (m["rel_wilson"] or 0.0) * 100.0,
            })
    rel_df = pd.DataFrame(rows_join)
    print(f"    RelTNZI per-season rows: {len(rel_df):,}")

    merged = rel_df.merge(nfi, on=["player_name", "team", "season"], how="inner")
    print(f"    merged with RelNFI: {len(merged):,} player-seasons")

    # Pearson r between RelTNZI and RelNFI at the player-season level
    r_pp = pearson(merged["RelTNZI_pct"].tolist(), merged["RelNFI_pct"].tolist())
    print(f"    player-season r(RelTNZI%, RelNFI%) = {r_pp:+.4f}  (N={len(merged):,})")

    # Aggregate to team-season then to team-pooled
    def agg(season_filter=None):
        m = merged
        if season_filter is not None:
            m = m[m["season"] == season_filter]
        return m.groupby("team").agg(
            tnzi_avg=("RelTNZI_pct", "mean"),
            nfi_avg=("RelNFI_pct", "mean")).reset_index()

    # Pooled = all 4 seasons stacked then team-mean (one row per team)
    pooled_team = merged.groupby("team").agg(
        tnzi_avg=("RelTNZI_pct", "mean"),
        nfi_avg=("RelNFI_pct", "mean")).reset_index()
    pooled_team["points"] = pooled_team["team"].map(pooled_pts)
    pooled_team = pooled_team.dropna()

    def ols(X_cols, y, df=pooled_team):
        X = np.column_stack([np.ones(len(df))] + [df[c].values for c in X_cols])
        y = df[y].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return beta, r2

    print("\n" + "=" * 90)
    print("COMBINED MODEL — team points pooled (n={}) regressed on team RelTNZI / RelNFI".format(
        len(pooled_team)))
    print("=" * 90)

    bA, r2A = ols(["tnzi_avg"], "points")
    bB, r2B = ols(["nfi_avg"], "points")
    bC, r2C = ols(["tnzi_avg", "nfi_avg"], "points")

    # Pearson at team level
    rt = pearson(pooled_team["tnzi_avg"].tolist(), pooled_team["points"].tolist())
    rn = pearson(pooled_team["nfi_avg"].tolist(), pooled_team["points"].tolist())
    rcorr = pearson(pooled_team["tnzi_avg"].tolist(),
                     pooled_team["nfi_avg"].tolist())

    print(f"  Model A: points ~ RelTNZI%        β = [{bA[0]:+.2f}, {bA[1]:+.4f}]   R² = {r2A:.4f}   "
          f"(team-level r = {rt:+.4f})")
    print(f"  Model B: points ~ RelNFI%         β = [{bB[0]:+.2f}, {bB[1]:+.4f}]   R² = {r2B:.4f}   "
          f"(team-level r = {rn:+.4f})")
    print(f"  Model C: points ~ Rel TNZI + RelNFI%  "
          f"β = [{bC[0]:+.2f}, {bC[1]:+.4f}, {bC[2]:+.4f}]   R² = {r2C:.4f}")
    print()
    print(f"  Team-level r(RelTNZI%, RelNFI%) = {rcorr:+.4f}  "
          f"(low = independent signals; high = redundant)")
    print(f"  ΔR² of combined over RelNFI alone:   {r2C - r2B:+.4f}")
    print(f"  ΔR² of combined over RelTNZI alone:  {r2C - r2A:+.4f}")

    # Per-season R² for combined
    print("\n  Per-season combined-model R² (team-season frame, n=32 per season):")
    print(f"  {'Season':<8} {'r2_C(both)':>11} {'r2_RelTNZI':>11} {'r2_RelNFI':>11} "
          f"{'corr(T,N)':>11}")
    for s in SEASONS:
        ts = pooled_team  # placeholder, recompute
        team_s = merged[merged["season"] == s].groupby("team").agg(
            tnzi_avg=("RelTNZI_pct", "mean"),
            nfi_avg=("RelNFI_pct", "mean")).reset_index()
        team_s["points"] = team_s["team"].map(standings[s])
        team_s = team_s.dropna()
        if len(team_s) < 5:
            print(f"  {SEASON_LABEL[s]:<8} (insufficient teams)")
            continue
        _, r2c = ols(["tnzi_avg", "nfi_avg"], "points", df=team_s)
        _, r2t = ols(["tnzi_avg"], "points", df=team_s)
        _, r2n = ols(["nfi_avg"], "points", df=team_s)
        rcs = pearson(team_s["tnzi_avg"].tolist(), team_s["nfi_avg"].tolist())
        print(f"  {SEASON_LABEL[s]:<8} {r2c:>11.4f} {r2t:>11.4f} {r2n:>11.4f} {rcs:>+11.4f}")

    print("\n[8/8] done.")

if __name__ == "__main__":
    main()
