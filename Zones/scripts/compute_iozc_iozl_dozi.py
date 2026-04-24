"""IOZC / IOZL / DOZI adjustments on four V1 zone-time metrics.

Reads:
  Zones/_player_meta.json
  Zones/_overlap.pkl                          (pooled teammate/opponent shared seconds)
  Zones/zone_variations/*_V1_*_pooled.csv     (if present; else rebuild from raw)
  Zones/raw/pbp/{gameId}.json                 (per-season V1 metrics)
  Data/game_ids.csv
  NHL standings API via curl

Writes:
  Zones/adjusted_rankings/{ozi|dzi|nzi|tnzi}_adjusted_{forwards|defense}.csv
  Zones/adjusted_rankings/dozi_{forwards|defense}.csv
  Zones/adjusted_rankings/tnzi_winning_correlation.csv
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

import numpy as np

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW_PBP = HERE / "raw" / "pbp"
RAW_SHIFTS = HERE / "raw" / "shifts"
GAME_IDS = ROOT / "Data" / "game_ids.csv"
PLAYER_META = HERE / "_player_meta.json"
OVERLAP_PKL = HERE / "_overlap.pkl"
ZONE_VAR = HERE / "zone_variations"
OUT_DIR = HERE / "adjusted_rankings"

Z = 1.96
Z2 = Z * Z
MIN_SHIFTS = 50
MIN_GP = 20
MIN_SEASONS_FOR_DOZI = 2
DOZI_RISING = 0.05
DOZI_DECLINING = -0.05

SEASONS = ["20222023", "20232024", "20242025", "20252026"]
POOLED = "pooled"
SCENARIOS = SEASONS + [POOLED]
CURRENT_SEASON = "20252026"

SEASON_END_DATE = {
    "20222023": "2023-04-13",
    "20232024": "2024-04-18",
    "20242025": "2025-04-17",
    "20252026": "2026-04-17",
}

V1_EVENTS = {
    "faceoff", "hit", "shot-on-goal", "missed-shot", "blocked-shot",
    "goal", "giveaway", "takeaway",
}

METRICS = ["OZI", "DZI", "NZI", "TNZI"]
POS_FORWARD = {"C", "L", "R"}
POS_DEFENSE = {"D"}
FLIP = {"O": "D", "D": "O", "N": "N"}
ABBR_MAP = {"ARI": "UTA"}

KEY_PLAYERS = [
    ("McDavid",    8478402),
    ("MacKinnon",  8477492),
    ("Draisaitl",  8477934),
    ("Makar",      8480069),
    ("Hughes (Q)", 8480800),
    ("Nurse",      8477498),
    ("Bouchard",   8480803),
    ("Ekholm",     8475218),
    ("Henrique",   8474641),
    ("Barkov",     8477493),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mmss(s: str) -> int:
    if not s or ":" not in s: return 0
    try:
        m, se = s.split(":"); return int(m) * 60 + int(se)
    except ValueError:
        return 0

def play_abs_time(play: dict) -> int:
    p = (play.get("periodDescriptor") or {}).get("number", 1) or 1
    return (p - 1) * 1200 + mmss(play.get("timeInPeriod", "00:00"))

def norm_team(abbr: str) -> str:
    return ABBR_MAP.get(abbr, abbr)

def wilson_lower(p: float, n: float):
    if n is None or n <= 0: return None
    p = max(0.0, min(1.0, p))
    denom = 1.0 + Z2 / n
    center = p + Z2 / (2.0 * n)
    margin = Z * math.sqrt(max(0.0, p * (1 - p) / n + Z2 / (4 * n * n)))
    return (center - margin) / denom

def wilson_upper(p: float, n: float):
    if n is None or n <= 0: return None
    p = max(0.0, min(1.0, p))
    denom = 1.0 + Z2 / n
    center = p + Z2 / (2.0 * n)
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

def spearman(x, y):
    def rank(v):
        s = sorted((val, i) for i, val in enumerate(v))
        r = [0] * len(v); i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1][0] == s[i][0]:
                j += 1
            for k in range(i, j + 1):
                r[s[k][1]] = (i + j) / 2 + 1
            i = j + 1
        return r
    return pearson(rank(x), rank(y))

def normalize_0_1(pid_val_pairs):
    vals = [v for _, v in pid_val_pairs if v is not None]
    if not vals: return {pid: None for pid, _ in pid_val_pairs}
    lo, hi = min(vals), max(vals); span = hi - lo
    out = {}
    for pid, v in pid_val_pairs:
        if v is None: out[pid] = None
        elif span == 0: out[pid] = 0.5
        else: out[pid] = (v - lo) / span
    return out

def score_0_10(pid_val_pairs):
    n01 = normalize_0_1(pid_val_pairs)
    return {pid: (None if v is None else round(v * 10, 1)) for pid, v in n01.items()}

# ---------------------------------------------------------------------------
# Standings
# ---------------------------------------------------------------------------

def fetch_standings(date_str: str) -> dict:
    d0 = datetime.strptime(date_str, "%Y-%m-%d")
    for offset in range(0, 5):
        d_try = (d0 - timedelta(days=offset)).strftime("%Y-%m-%d")
        out = subprocess.run(
            ["curl", "-s", "-m", "20",
             f"https://api-web.nhle.com/v1/standings/{d_try}"],
            capture_output=True, text=True, check=True,
        )
        try:
            d = json.loads(out.stdout)
        except json.JSONDecodeError:
            continue
        rows = d.get("standings", [])
        if rows:
            res = {}
            for row in rows:
                abbr = norm_team(row["teamAbbrev"]["default"])
                res[abbr] = {"points": row["points"],
                             "gp": row["gamesPlayed"]}
            if d_try != date_str:
                print(f"    note: {date_str} empty; using {d_try}")
            return res
    return {}

# ---------------------------------------------------------------------------
# Raw-PBP processing -> per-season player buckets
# ---------------------------------------------------------------------------

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

def on_ice_at(intervals_by_pid, t):
    out = defaultdict(set)
    for pid, ivs in intervals_by_pid.items():
        for a, b, tm in ivs:
            if a <= t < b:
                out[tm].add(pid); break
    return out

def shift_end_for(intervals_by_pid, pid, t):
    for a, b, _ in intervals_by_pid.get(pid, []):
        if a <= t < b: return b
    return None

def process_game(gid, player_season_gp, player_bucket):
    pbp_path = RAW_PBP / f"{gid}.json"
    sh_path = RAW_SHIFTS / f"{gid}.json"
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
    if home_id is None or away_id is None:
        return

    intervals = build_shift_intervals(shifts)
    for pid in intervals:
        player_season_gp[(pid, season)] += 1

    plays = pbp.get("plays") or []
    ctx = None

    def emit(ctx, close_t):
        if ctx is None or not ctx["events"]:
            return
        fo_home = ctx["fo_zone_home"]
        for team_id, pids in ctx["players_at_fo"].items():
            if team_id not in (home_id, away_id):
                continue
            fo_from_team = fo_home if team_id == home_id else FLIP[fo_home]
            for pid in pids:
                sh_end = shift_end_for(intervals, pid, ctx["fo_t"])
                if sh_end is None: continue
                eff_end = min(close_t, sh_end)
                if eff_end <= ctx["fo_t"]: continue
                filtered = [(t, ez) for (t, ez, _ty) in ctx["events"]
                            if ctx["fo_t"] <= t < eff_end]
                if not filtered:
                    continue
                key = (pid, season, fo_from_team)
                b = player_bucket[key]
                b["shifts"] += 1
                for i, (t, ez) in enumerate(filtered):
                    tn = filtered[i + 1][0] if i + 1 < len(filtered) else eff_end
                    dt = max(0.0, tn - t)
                    ez_player = ez if team_id == home_id else FLIP[ez]
                    b["total_sec"] += dt
                    if ez_player == "O":   b["oz_sec"] += dt
                    elif ez_player == "D": b["dz_sec"] += dt
                    else:                  b["nz_sec"] += dt

    for p in plays:
        typ = p.get("typeDescKey") or ""
        det = p.get("details") or {}
        t_abs = play_abs_time(p)
        sit = p.get("situationCode") or ""

        if typ == "faceoff":
            if ctx is not None:
                emit(ctx, t_abs); ctx = None
            if sit != "1551": continue
            zc = det.get("zoneCode")
            if zc not in ("O", "D", "N"): continue
            owner = det.get("eventOwnerTeamId")
            if owner not in (home_id, away_id): continue
            fo_home = zc if owner == home_id else FLIP[zc]
            oi = on_ice_at(intervals, t_abs)
            ctx = {
                "fo_t": t_abs,
                "fo_zone_home": fo_home,
                "events": [(t_abs, fo_home, "faceoff")],
                "players_at_fo": {home_id: set(oi.get(home_id, set())),
                                  away_id: set(oi.get(away_id, set()))},
            }
            continue

        if ctx is None: continue
        if sit and sit != "1551":
            emit(ctx, t_abs); ctx = None; continue
        zc = det.get("zoneCode")
        if zc in ("O", "D", "N") and typ in V1_EVENTS:
            owner = det.get("eventOwnerTeamId")
            if owner in (home_id, away_id):
                ez_home = zc if owner == home_id else FLIP[zc]
                ctx["events"].append((t_abs, ez_home, typ))
        if typ in ("period-end", "game-end"):
            emit(ctx, t_abs); ctx = None
    if ctx is not None:
        last_t = play_abs_time(plays[-1]) if plays else ctx["fo_t"]
        emit(ctx, last_t)

# ---------------------------------------------------------------------------
# Compute metric from zone buckets
# ---------------------------------------------------------------------------

def compute_metrics_for_bucket_set(b_o, b_d, b_n):
    """Return dict metric->(raw, wilson_adjusted, qualifying_flag) using shift
    count as Wilson n. Returns (None, None, False) if below MIN_SHIFTS."""
    def get(b): return b if b else {"shifts": 0, "total_sec": 0.0,
                                     "oz_sec": 0.0, "dz_sec": 0.0, "nz_sec": 0.0}
    bo = get(b_o); bd = get(b_d); bn = get(b_n)
    res = {}

    # OZI: on OZ FO
    if bo["shifts"] >= MIN_SHIFTS and bo["total_sec"] > 0:
        p = bo["oz_sec"] / bo["total_sec"]
        res["OZI"] = (p, wilson_lower(p, bo["shifts"]), True)
    else:
        res["OZI"] = (None, None, False)

    # DZI: OZ% on DZ FO
    if bd["shifts"] >= MIN_SHIFTS and bd["total_sec"] > 0:
        p = bd["oz_sec"] / bd["total_sec"]
        res["DZI"] = (p, wilson_lower(p, bd["shifts"]), True)
    else:
        res["DZI"] = (None, None, False)

    # NZI: OZ% on NZ FO
    if bn["shifts"] >= MIN_SHIFTS and bn["total_sec"] > 0:
        p = bn["oz_sec"] / bn["total_sec"]
        res["NZI"] = (p, wilson_lower(p, bn["shifts"]), True)
    else:
        res["NZI"] = (None, None, False)

    # TNZI: OZ% minus DZ% on NZ FO (Wilson lower OZ% - Wilson upper DZ%)
    if bn["shifts"] >= MIN_SHIFTS and bn["total_sec"] > 0:
        p_oz = bn["oz_sec"] / bn["total_sec"]
        p_dz = bn["dz_sec"] / bn["total_sec"]
        lo = wilson_lower(p_oz, bn["shifts"])
        hi = wilson_upper(p_dz, bn["shifts"])
        if lo is not None and hi is not None:
            res["TNZI"] = (p_oz - p_dz, lo - hi, True)
        else:
            res["TNZI"] = (None, None, False)
    else:
        res["TNZI"] = (None, None, False)

    return res

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_game_ids():
    out = []
    with open(GAME_IDS) as f:
        for r in csv.DictReader(f):
            if r["season"] in set(SEASONS) and r["game_type"] == "regular":
                out.append((int(r["game_id"]), r["season"]))
    return out

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/9] fetching standings ...")
    standings = {s: fetch_standings(SEASON_END_DATE[s]) for s in SEASONS}
    for s in SEASONS:
        print(f"    {s}: {len(standings[s])} teams")

    print("[2/9] loading player meta & overlap ...")
    player_meta = {int(k): v for k, v in json.load(open(PLAYER_META)).items()}
    overlap = pickle.load(open(OVERLAP_PKL, "rb"))
    overlap_pool = overlap.get("pooled", {})
    teammate_map = overlap_pool.get("teammate", {})
    opponent_map = overlap_pool.get("opponent", {})
    print(f"    meta players: {len(player_meta)}")
    print(f"    pooled teammate pairs: {len(teammate_map)}  "
          f"opponent pairs: {len(opponent_map)}")

    print("[3/9] reading game_ids ...")
    games = load_game_ids()
    print(f"    {len(games)} regular games across {len(SEASONS)} seasons")

    print("[4/9] processing raw PBP + shifts ...")
    player_bucket = defaultdict(lambda: {"shifts": 0, "total_sec": 0.0,
                                         "oz_sec": 0.0, "dz_sec": 0.0,
                                         "nz_sec": 0.0})
    player_season_gp = defaultdict(int)
    total = len(games)
    for i, (gid, _s) in enumerate(games):
        process_game(gid, player_season_gp, player_bucket)
        if (i + 1) % 1000 == 0 or i + 1 == total:
            print(f"    {i+1}/{total}")

    # ---- Aggregate per scenario -----------------------------------------
    print("[5/9] aggregating per-scenario buckets & computing raw/adjusted ...")

    def per_scenario_buckets(seasons_filter: set):
        """Return {pid: {'O':bucket,'D':bucket,'N':bucket,'gp':int}}"""
        out = defaultdict(lambda: {"O": None, "D": None, "N": None, "gp": 0})
        for (pid, season, fz), b in player_bucket.items():
            if season not in seasons_filter: continue
            cur = out[pid][fz] or {"shifts": 0, "total_sec": 0.0,
                                    "oz_sec": 0.0, "dz_sec": 0.0, "nz_sec": 0.0}
            cur["shifts"] += b["shifts"]
            cur["total_sec"] += b["total_sec"]
            cur["oz_sec"] += b["oz_sec"]
            cur["dz_sec"] += b["dz_sec"]
            cur["nz_sec"] += b["nz_sec"]
            out[pid][fz] = cur
        for (pid, season), gp in player_season_gp.items():
            if season in seasons_filter:
                out[pid]["gp"] += gp
        return out

    # scenario_raw[scenario][metric][pid] = raw p
    # scenario_wilson[scenario][metric][pid] = wilson_adjusted
    scenario_raw = defaultdict(lambda: defaultdict(dict))
    scenario_wilson = defaultdict(lambda: defaultdict(dict))
    scenario_shifts = defaultdict(lambda: defaultdict(dict))
    scenario_gp = defaultdict(dict)

    scen_filter = {s: {s} for s in SEASONS}
    scen_filter[POOLED] = set(SEASONS)

    for scen in SCENARIOS:
        agg = per_scenario_buckets(scen_filter[scen])
        for pid, data in agg.items():
            if data["gp"] < MIN_GP: continue
            m = compute_metrics_for_bucket_set(data["O"], data["D"], data["N"])
            for metric in METRICS:
                raw, adj, ok = m[metric]
                if ok:
                    scenario_raw[scen][metric][pid] = raw
                    scenario_wilson[scen][metric][pid] = adj
            scenario_shifts[scen][pid] = {
                "O": (data["O"]["shifts"] if data["O"] else 0),
                "D": (data["D"]["shifts"] if data["D"] else 0),
                "N": (data["N"]["shifts"] if data["N"] else 0),
            }
            scenario_gp[scen][pid] = data["gp"]

    # ---- Normalize 0-1 per position group per scenario ------------------
    print("[6/9] normalising per position group per scenario ...")
    # norm01[scenario][metric][pid] = 0-1 score (None if disqualified)
    norm01 = defaultdict(lambda: defaultdict(dict))
    for scen in SCENARIOS:
        for metric in METRICS:
            pid_vals = list(scenario_wilson[scen][metric].items())
            # split by position
            for pos_set in (POS_FORWARD, POS_DEFENSE):
                subset = [(pid, v) for pid, v in pid_vals
                          if (player_meta.get(pid, {}).get("position")
                              or "").upper() in pos_set]
                grp = normalize_0_1(subset)
                for pid, val in grp.items():
                    norm01[scen][metric][pid] = val

    # ---- IOZC / IOZL on POOLED ------------------------------------------
    print("[7/9] computing IOZC / IOZL from pooled overlap ...")

    def overlap_weighted(pid, metric, pool_map, pos_set):
        """Weighted average of normalized0-1 score of partners (teammate or
        opponent) weighted by shared seconds. Same position group only."""
        my_pos = (player_meta.get(pid, {}).get("position") or "").upper()
        if my_pos not in pos_set:
            return None
        total_w = 0.0
        sum_w = 0.0
        for (a, b), secs in pool_map.items():
            if a != pid and b != pid: continue
            partner = b if a == pid else a
            partner_pos = (player_meta.get(partner, {}).get("position") or "").upper()
            if partner_pos not in pos_set: continue
            score = norm01[POOLED][metric].get(partner)
            if score is None: continue
            total_w += secs
            sum_w += score * secs
        return (sum_w / total_w) if total_w > 0 else None

    # iozc[metric][pid] = value, iozl[metric][pid] = value
    iozc = defaultdict(dict); iozl = defaultdict(dict)
    for metric in METRICS:
        for pid in norm01[POOLED][metric]:
            pos = (player_meta.get(pid, {}).get("position") or "").upper()
            pos_set = POS_FORWARD if pos in POS_FORWARD else (POS_DEFENSE if pos in POS_DEFENSE else None)
            if pos_set is None: continue
            iozc[metric][pid] = overlap_weighted(pid, metric, opponent_map, pos_set)
            iozl[metric][pid] = overlap_weighted(pid, metric, teammate_map, pos_set)

    # ---- OLS regressions at team level ----------------------------------
    print("[8/9] OLS regressions and adjusted scores ...")

    # Team points: use average across 4 seasons
    team_points_avg = defaultdict(list)
    for s in SEASONS:
        for tm, row in standings[s].items():
            team_points_avg[tm].append(row["points"])
    team_points = {tm: mean(vs) for tm, vs in team_points_avg.items() if vs}

    # Aggregate player pooled raw / IOZC / IOZL to team level
    # Use each player's current team from meta (ARI -> UTA already normalised)
    def player_team(pid):
        meta = player_meta.get(pid, {})
        return norm_team(meta.get("team_abbrev", "") or "")

    def team_mean(pid_map):
        by_team = defaultdict(list)
        for pid, v in pid_map.items():
            if v is None: continue
            tm = player_team(pid)
            if not tm: continue
            by_team[tm].append(v)
        return {tm: mean(vs) for tm, vs in by_team.items() if vs}

    # betas[metric] = {"single_C": (b_raw, b_iozc),
    #                  "single_L": (b_raw, b_iozl),
    #                  "both":     (b_raw, b_iozc, b_iozl)}
    betas = {}
    for metric in METRICS:
        t_raw = team_mean(norm01[POOLED][metric])
        t_iozc = team_mean(iozc[metric])
        t_iozl = team_mean(iozl[metric])
        # Shared teams
        common = set(team_points) & set(t_raw) & set(t_iozc) & set(t_iozl)
        teams = sorted(common)
        if len(teams) < 5:
            betas[metric] = {"single_C": (None, None),
                             "single_L": (None, None),
                             "both": (None, None, None),
                             "teams": 0}
            continue
        y = np.array([team_points[tm] for tm in teams])
        raw_v = np.array([t_raw[tm] for tm in teams])
        c_v = np.array([t_iozc[tm] for tm in teams])
        l_v = np.array([t_iozl[tm] for tm in teams])
        # Model 1: y ~ raw + IOZC
        X1 = np.column_stack([np.ones(len(teams)), raw_v, c_v])
        b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
        # Model 2: y ~ raw + IOZL
        X2 = np.column_stack([np.ones(len(teams)), raw_v, l_v])
        b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
        # Model 3: y ~ raw + IOZC + IOZL
        X3 = np.column_stack([np.ones(len(teams)), raw_v, c_v, l_v])
        b3, *_ = np.linalg.lstsq(X3, y, rcond=None)
        betas[metric] = {
            "single_C": (float(b1[1]), float(b1[2])),
            "single_L": (float(b2[1]), float(b2[2])),
            "both":     (float(b3[1]), float(b3[2]), float(b3[3])),
            "teams": len(teams),
        }

    # Adjusted per-player scores — use raw (0-1 normalised) as the base
    # OZI_C = raw + (β_iozc/β_raw) × IOZC   etc.
    adj_raw = {metric: {"C": {}, "L": {}, "CL": {}} for metric in METRICS}
    for metric in METRICS:
        bC = betas[metric].get("single_C") or (None, None)
        bL = betas[metric].get("single_L") or (None, None)
        b3 = betas[metric].get("both") or (None, None, None)
        bR_C, bC_C = bC
        bR_L, bL_L = bL
        bR_3, bC_3, bL_3 = b3

        def ratio(numer, denom):
            if numer is None or denom is None or abs(denom) < 1e-12:
                return None
            return numer / denom

        r_C = ratio(bC_C, bR_C)
        r_L = ratio(bL_L, bR_L)
        r_C3 = ratio(bC_3, bR_3)
        r_L3 = ratio(bL_3, bR_3)

        for pid, raw in norm01[POOLED][metric].items():
            if raw is None: continue
            c = iozc[metric].get(pid)
            l = iozl[metric].get(pid)
            adj_raw[metric]["C"][pid] = (raw + r_C * c) if (r_C is not None and c is not None) else None
            adj_raw[metric]["L"][pid] = (raw - r_L * l) if (r_L is not None and l is not None) else None
            adj_raw[metric]["CL"][pid] = (
                raw + r_C3 * c - r_L3 * l
                if (r_C3 is not None and r_L3 is not None
                    and c is not None and l is not None)
                else None
            )

    # Normalize adjusted scores 0-10 within position group
    adj_scores = {metric: {"C": {}, "L": {}, "CL": {}} for metric in METRICS}
    for metric in METRICS:
        for variant in ("C", "L", "CL"):
            for pos_set in (POS_FORWARD, POS_DEFENSE):
                subset = [(pid, v) for pid, v in adj_raw[metric][variant].items()
                          if (player_meta.get(pid, {}).get("position") or "").upper() in pos_set]
                s = score_0_10(subset)
                for pid, val in s.items():
                    adj_scores[metric][variant][pid] = val

    # Also produce raw pooled score 0-10 per position group (for output)
    raw_scores_pool_10 = {metric: {} for metric in METRICS}
    for metric in METRICS:
        for pos_set in (POS_FORWARD, POS_DEFENSE):
            subset = [(pid, v) for pid, v in norm01[POOLED][metric].items()
                      if v is not None and
                      (player_meta.get(pid, {}).get("position") or "").upper() in pos_set]
            s = score_0_10(subset)
            for pid, val in s.items():
                raw_scores_pool_10[metric][pid] = val

    # ---- DOZI: year-over-year of per-season 0-1 normalised -------------
    print("    computing DOZI ...")
    # per-season 0-1 scores already in norm01[season][metric][pid]
    dozi = {metric: {} for metric in METRICS}
    for metric in METRICS:
        for pid in set().union(*(set(norm01[s][metric]) for s in SEASONS)):
            per = {s: norm01[s][metric].get(pid) for s in SEASONS}
            qualified = {s: v for s, v in per.items() if v is not None}
            deltas = {}
            pairs = [("20222023", "20232024", "DOZI_23_24"),
                     ("20232024", "20242025", "DOZI_24_25"),
                     ("20242025", "20252026", "DOZI_25_26")]
            for a, b, label in pairs:
                if per.get(a) is not None and per.get(b) is not None:
                    deltas[label] = per[b] - per[a]
            if len(qualified) < MIN_SEASONS_FOR_DOZI:
                dozi[metric][pid] = {"deltas": deltas, "trend": None,
                                     "recent": None, "flag": None}
                continue
            vals = [v for v in deltas.values()]
            trend = mean(vals) if vals else None
            recent = (deltas.get("DOZI_25_26")
                      if "DOZI_25_26" in deltas
                      else deltas.get("DOZI_24_25"))
            if recent is None:
                flag = None
            elif recent > DOZI_RISING:
                flag = "RISING"
            elif recent < DOZI_DECLINING:
                flag = "DECLINING"
            else:
                flag = "STABLE"
            dozi[metric][pid] = {"deltas": deltas, "trend": trend,
                                 "recent": recent, "flag": flag}

    # ---- TNZI winning correlation ---------------------------------------
    tnzi_corr_rows = []
    def corr_team(pid_map, points_map):
        t = team_mean(pid_map)
        common = set(t) & set(points_map)
        if len(common) < 5:
            return float("nan"), float("nan")
        xs = [t[tm] for tm in common]; ys = [points_map[tm] for tm in common]
        r = pearson(xs, ys)
        r2 = r * r if not math.isnan(r) else float("nan")
        return r, r2

    # Per-season team points
    team_points_by_season = {}
    for s in SEASONS:
        team_points_by_season[s] = {tm: row["points"] for tm, row in standings[s].items()}
    team_points_by_season[POOLED] = team_points

    variants_for_tnzi = {
        "raw":  norm01[POOLED]["TNZI"],
        "C":    adj_raw["TNZI"]["C"],
        "L":    adj_raw["TNZI"]["L"],
        "CL":   adj_raw["TNZI"]["CL"],
    }

    # pooled correlations
    for variant_label, pid_map in variants_for_tnzi.items():
        r, r2 = corr_team(pid_map, team_points)
        tnzi_corr_rows.append({"scenario": "pooled", "variant": variant_label,
                               "pearson_r": r, "r_squared": r2,
                               "n_teams": sum(1 for tm in team_mean(pid_map)
                                              if tm in team_points)})

    # per-season: only raw available per season (adjusted uses pooled overlap/β)
    for s in SEASONS:
        r, r2 = corr_team(norm01[s]["TNZI"], team_points_by_season[s])
        tnzi_corr_rows.append({"scenario": s, "variant": "raw",
                               "pearson_r": r, "r_squared": r2,
                               "n_teams": sum(1 for tm in team_mean(norm01[s]["TNZI"])
                                              if tm in team_points_by_season[s])})

    # For per-season variant correlations: apply pooled betas to per-season raw + pooled IOZC/IOZL
    for s in SEASONS:
        for variant_label, pid_map in (("C", adj_raw["TNZI"]["C"]),
                                        ("L", adj_raw["TNZI"]["L"]),
                                        ("CL", adj_raw["TNZI"]["CL"])):
            # use only pids also present in per-season raw; substitute per-season raw then apply same beta offsets
            bC = betas["TNZI"]["single_C"]; bL = betas["TNZI"]["single_L"]; b3 = betas["TNZI"]["both"]
            def rat(n, d):
                return None if (n is None or d is None or abs(d) < 1e-12) else n / d
            r_C = rat(bC[1], bC[0])
            r_L = rat(bL[1], bL[0])
            r_C3 = rat(b3[1], b3[0])
            r_L3 = rat(b3[2], b3[0])
            season_map = {}
            for pid, raw_s in norm01[s]["TNZI"].items():
                if raw_s is None: continue
                c = iozc["TNZI"].get(pid)
                l = iozl["TNZI"].get(pid)
                if variant_label == "C" and r_C is not None and c is not None:
                    season_map[pid] = raw_s + r_C * c
                elif variant_label == "L" and r_L is not None and l is not None:
                    season_map[pid] = raw_s - r_L * l
                elif variant_label == "CL" and r_C3 is not None and r_L3 is not None and c is not None and l is not None:
                    season_map[pid] = raw_s + r_C3 * c - r_L3 * l
            r, r2 = corr_team(season_map, team_points_by_season[s])
            tnzi_corr_rows.append({"scenario": s, "variant": variant_label,
                                   "pearson_r": r, "r_squared": r2,
                                   "n_teams": sum(1 for tm in team_mean(season_map)
                                                  if tm in team_points_by_season[s])})

    # ---- Write CSV outputs ----------------------------------------------
    print("[9/9] writing CSVs ...")
    write_adjusted_csvs(
        OUT_DIR, player_meta, scenario_gp[POOLED], scenario_shifts[POOLED],
        norm01, raw_scores_pool_10, iozc, iozl, adj_scores, adj_raw, dozi,
    )
    write_dozi_csvs(OUT_DIR, player_meta, scenario_gp[POOLED], dozi,
                    raw_scores_pool_10)
    write_corr_csv(OUT_DIR, tnzi_corr_rows)

    # ---- Print reports --------------------------------------------------
    print_tnzi_corr_table(tnzi_corr_rows)
    print_top_tnzi(player_meta, scenario_gp[POOLED], adj_scores,
                   adj_raw, norm01, iozc, iozl, dozi)
    print_dozi_leaders(player_meta, scenario_gp[POOLED], dozi,
                       raw_scores_pool_10)
    print_key_players(player_meta, scenario_gp[POOLED], scenario_shifts[POOLED],
                      norm01, adj_scores, iozc, iozl, dozi)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUT_COLS = [
    "player_name", "team", "pos", "GP", "seasons_qualified",
    "IOZC", "IOZL",
    "OZI", "OZI_C", "OZI_L", "OZI_CL",
    "DZI", "DZI_C", "DZI_L", "DZI_CL",
    "NZI", "NZI_C", "NZI_L", "NZI_CL",
    "TNZI", "TNZI_C", "TNZI_L", "TNZI_CL",
    "DOZI_23_24", "DOZI_24_25", "DOZI_25_26",
    "DOZI_trend", "DOZI_recent", "DOZI_flag",
]

def _fmt(v, d=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    try:
        return round(float(v), d)
    except (TypeError, ValueError):
        return v

def write_adjusted_csvs(out_dir, meta, gp_map, shifts_map, norm01,
                        raw_scores_pool_10, iozc, iozl, adj_scores, adj_raw, dozi):
    """Write one adjusted CSV per metric per position group. Each CSV has all
    OUT_COLS so downstream inspection is easy."""
    # Identify qualifying players: those with at least one metric POOLED.
    qualifying = set()
    for metric in METRICS:
        qualifying |= set(pid for pid, v in norm01[POOLED][metric].items() if v is not None)

    # Build row per player
    def build_row(pid, sort_metric):
        pos = (meta.get(pid, {}).get("position") or "").upper()
        seasons_q = sum(
            1 for s in SEASONS
            if any(norm01[s][m].get(pid) is not None for m in METRICS)
        )
        r = {
            "player_name": meta.get(pid, {}).get("name", ""),
            "team": norm_team(meta.get(pid, {}).get("team_abbrev", "") or ""),
            "pos": pos,
            "GP": gp_map.get(pid, 0),
            "seasons_qualified": seasons_q,
            "IOZC": _fmt(iozc[sort_metric].get(pid)),
            "IOZL": _fmt(iozl[sort_metric].get(pid)),
        }
        for metric in METRICS:
            r[metric] = raw_scores_pool_10[metric].get(pid, "")
            r[f"{metric}_C"] = adj_scores[metric]["C"].get(pid, "")
            r[f"{metric}_L"] = adj_scores[metric]["L"].get(pid, "")
            r[f"{metric}_CL"] = adj_scores[metric]["CL"].get(pid, "")
        # DOZI — use the file's headline metric for the deltas
        d = dozi[sort_metric].get(pid, {"deltas": {}, "trend": None, "recent": None, "flag": None})
        r["DOZI_23_24"] = _fmt(d["deltas"].get("DOZI_23_24"), 4)
        r["DOZI_24_25"] = _fmt(d["deltas"].get("DOZI_24_25"), 4)
        r["DOZI_25_26"] = _fmt(d["deltas"].get("DOZI_25_26"), 4)
        r["DOZI_trend"] = _fmt(d["trend"])
        r["DOZI_recent"] = _fmt(d["recent"])
        r["DOZI_flag"] = d["flag"] or ""
        return r

    for metric in METRICS:
        for group_name, pos_set in (("forwards", POS_FORWARD),
                                     ("defense", POS_DEFENSE)):
            rows = []
            for pid in qualifying:
                pos = (meta.get(pid, {}).get("position") or "").upper()
                if pos not in pos_set: continue
                if pid not in norm01[POOLED][metric]:
                    continue
                rows.append(build_row(pid, metric))
            # Sort by <metric>_CL desc if present, else <metric>
            def sort_key(row):
                v = row.get(f"{metric}_CL") or row.get(metric) or 0
                try: return -float(v)
                except (TypeError, ValueError): return 0
            rows.sort(key=sort_key)
            path = out_dir / f"{metric.lower()}_adjusted_{group_name}.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(OUT_COLS)
                for r in rows:
                    w.writerow([r.get(c, "") for c in OUT_COLS])
    print(f"    wrote adjusted CSVs in {out_dir}")

def write_dozi_csvs(out_dir, meta, gp_map, dozi, raw_scores_pool_10):
    for group_name, pos_set in (("forwards", POS_FORWARD),
                                 ("defense", POS_DEFENSE)):
        path = out_dir / f"dozi_{group_name}.csv"
        rows = []
        seen = set()
        for metric in METRICS:
            for pid, d in dozi[metric].items():
                if pid in seen: continue
                pos = (meta.get(pid, {}).get("position") or "").upper()
                if pos not in pos_set: continue
                seen.add(pid)
                row = {
                    "player_name": meta.get(pid, {}).get("name", ""),
                    "team": norm_team(meta.get(pid, {}).get("team_abbrev", "") or ""),
                    "pos": pos,
                    "GP": gp_map.get(pid, 0),
                }
                for m in METRICS:
                    dx = dozi[m].get(pid, {"deltas": {}, "trend": None,
                                           "recent": None, "flag": None})
                    row[f"{m}_23_24"] = _fmt(dx["deltas"].get("DOZI_23_24"))
                    row[f"{m}_24_25"] = _fmt(dx["deltas"].get("DOZI_24_25"))
                    row[f"{m}_25_26"] = _fmt(dx["deltas"].get("DOZI_25_26"))
                    row[f"{m}_trend"] = _fmt(dx["trend"])
                    row[f"{m}_recent"] = _fmt(dx["recent"])
                    row[f"{m}_flag"] = dx["flag"] or ""
                    row[f"{m}_pooled_score"] = raw_scores_pool_10[m].get(pid, "")
                rows.append(row)
        cols = ["player_name", "team", "pos", "GP"]
        for m in METRICS:
            cols += [f"{m}_pooled_score",
                     f"{m}_23_24", f"{m}_24_25", f"{m}_25_26",
                     f"{m}_trend", f"{m}_recent", f"{m}_flag"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in cols])
    print(f"    wrote DOZI CSVs in {out_dir}")

def write_corr_csv(out_dir, rows):
    path = out_dir / "tnzi_winning_correlation.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "variant", "pearson_r", "r_squared", "n_teams"])
        for r in rows:
            w.writerow([r["scenario"], r["variant"],
                        _fmt(r["pearson_r"]), _fmt(r["r_squared"]),
                        r["n_teams"]])
    print(f"    wrote {path}")

# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_tnzi_corr_table(rows):
    print("\n" + "=" * 80)
    print("TNZI — winning correlation (Pearson r / R²) — all four variants")
    print("=" * 80)
    print(f"{'Scenario':<12} {'Variant':<8} {'Pearson r':>10} {'R²':>7} {'Teams':>6}")
    order = ["pooled"] + SEASONS
    var_order = ["raw", "C", "L", "CL"]
    for scen in order:
        for variant in var_order:
            for r in rows:
                if r["scenario"] == scen and r["variant"] == variant:
                    pr = r["pearson_r"]; r2 = r["r_squared"]
                    prs = f"{pr:+.3f}" if isinstance(pr, float) and not math.isnan(pr) else "   -  "
                    r2s = f"{r2:.3f}" if isinstance(r2, float) and not math.isnan(r2) else "  -  "
                    print(f"{scen:<12} {variant:<8} {prs:>10} {r2s:>7} {r['n_teams']:>6}")
    # Interpretation
    def get(scen, var):
        for r in rows:
            if r["scenario"] == scen and r["variant"] == var:
                return r["pearson_r"]
        return float("nan")
    raw_pool = get("pooled", "raw")
    c_pool = get("pooled", "C"); l_pool = get("pooled", "L"); cl_pool = get("pooled", "CL")
    def delta(a, b):
        if math.isnan(a) or math.isnan(b): return "n/a"
        return f"{b - a:+.3f}"
    print("\nInterpretation (pooled 4-season):")
    print(f"  raw Pearson r = {raw_pool:+.3f}")
    print(f"  + IOZC only   : r = {c_pool:+.3f}   Δ = {delta(raw_pool, c_pool)}")
    print(f"  + IOZL only   : r = {l_pool:+.3f}   Δ = {delta(raw_pool, l_pool)}")
    print(f"  + both (CL)   : r = {cl_pool:+.3f}   Δ = {delta(raw_pool, cl_pool)}")
    best = max([("raw", raw_pool), ("C", c_pool), ("L", l_pool), ("CL", cl_pool)],
               key=lambda x: -2 if math.isnan(x[1]) else x[1])
    print(f"  best variant  : {best[0]}  (r = {best[1]:+.3f})")
    corsi = 0.55
    print(f"\nCorsi benchmark ≈ {corsi:+.3f}")
    print(f"  best TNZI vs Corsi : Δ = {best[1] - corsi:+.3f}")

def print_top_tnzi(meta, gp_map, adj_scores, adj_raw, norm01, iozc, iozl, dozi):
    print("\n" + "=" * 80)
    print("Top 20 by TNZI_CL (both-adjusted, pooled)")
    print("=" * 80)
    for group_name, pos_set in (("forwards", POS_FORWARD),
                                 ("defense", POS_DEFENSE)):
        print(f"\n-- {group_name} --")
        pid_score = [(pid, v) for pid, v in adj_scores["TNZI"]["CL"].items()
                     if v is not None and
                     (meta.get(pid, {}).get("position") or "").upper() in pos_set]
        pid_score.sort(key=lambda x: x[1], reverse=True)
        print(f"  {'#':>3} {'Player':<22} {'Tm':<4} {'GP':>4} "
              f"{'TNZI':>5} {'TNZI_C':>7} {'TNZI_L':>7} {'TNZI_CL':>8} "
              f"{'IOZC':>6} {'IOZL':>6} {'flag':<10}")
        for i, (pid, v) in enumerate(pid_score[:20], 1):
            tm = norm_team(meta.get(pid, {}).get("team_abbrev", "") or "")
            raw = round((norm01[POOLED]["TNZI"].get(pid) or 0) * 10, 1)
            vC = adj_scores["TNZI"]["C"].get(pid) or 0
            vL = adj_scores["TNZI"]["L"].get(pid) or 0
            ic = iozc["TNZI"].get(pid); il = iozl["TNZI"].get(pid)
            fl = (dozi["TNZI"].get(pid) or {}).get("flag") or ""
            print(f"  {i:>3} {meta.get(pid, {}).get('name', '')[:22]:<22} {tm:<4} "
                  f"{gp_map.get(pid, 0):>4} {raw:>5.1f} {vC:>7.1f} {vL:>7.1f} {v:>8.1f} "
                  f"{(ic if ic is not None else 0):>6.3f} {(il if il is not None else 0):>6.3f} "
                  f"{fl:<10}")

def print_dozi_leaders(meta, gp_map, dozi, raw_scores_pool_10):
    print("\n" + "=" * 80)
    print("DOZI — top 10 RISING / DECLINING by DOZI_recent on TNZI (pooled)")
    print("=" * 80)
    for group_name, pos_set in (("forwards", POS_FORWARD),
                                 ("defense", POS_DEFENSE)):
        print(f"\n-- {group_name} --")
        rows = []
        for pid, d in dozi["TNZI"].items():
            if d.get("recent") is None: continue
            pos = (meta.get(pid, {}).get("position") or "").upper()
            if pos not in pos_set: continue
            rows.append((pid, d["recent"], d.get("flag")))
        rising = sorted((r for r in rows if r[1] > 0), key=lambda x: -x[1])[:10]
        declining = sorted((r for r in rows if r[1] < 0), key=lambda x: x[1])[:10]
        print("  RISING (top 10):")
        for pid, rec, fl in rising:
            tm = norm_team(meta.get(pid, {}).get("team_abbrev", "") or "")
            print(f"    {meta.get(pid, {}).get('name', '')[:24]:<24} {tm:<4} "
                  f"GP={gp_map.get(pid, 0):<4} "
                  f"TNZI_pooled={raw_scores_pool_10['TNZI'].get(pid, '-'):<4} "
                  f"DOZI_recent={rec:+.3f}  [{fl}]")
        print("  DECLINING (top 10):")
        for pid, rec, fl in declining:
            tm = norm_team(meta.get(pid, {}).get("team_abbrev", "") or "")
            print(f"    {meta.get(pid, {}).get('name', '')[:24]:<24} {tm:<4} "
                  f"GP={gp_map.get(pid, 0):<4} "
                  f"TNZI_pooled={raw_scores_pool_10['TNZI'].get(pid, '-'):<4} "
                  f"DOZI_recent={rec:+.3f}  [{fl}]")

def print_key_players(meta, gp_map, shifts_map, norm01, adj_scores, iozc, iozl, dozi):
    print("\n" + "=" * 80)
    print("Key players — full pooled rows (all four metrics + adjustments)")
    print("=" * 80)
    hdr = (f"{'Player':<14} {'Tm':<4} {'GP':>4}  "
           f"{'OZI':>5}/{'C':<4}/{'L':<4}/{'CL':<4}  "
           f"{'DZI':>5}/{'C':<4}/{'L':<4}/{'CL':<4}  "
           f"{'NZI':>5}/{'C':<4}/{'L':<4}/{'CL':<4}  "
           f"{'TNZI':>5}/{'C':<4}/{'L':<4}/{'CL':<4}  "
           f"{'flag':<10}")
    print(hdr)
    for label, pid in KEY_PLAYERS:
        tm = norm_team(meta.get(pid, {}).get("team_abbrev", "") or "")
        gp = gp_map.get(pid, 0)

        def fmt10(metric, variant):
            if variant == "raw":
                v = norm01[POOLED][metric].get(pid)
                return f"{v * 10:.1f}" if v is not None else " -- "
            v = adj_scores[metric][variant].get(pid)
            return f"{v:.1f}" if v is not None else " -- "

        cells = []
        for metric in METRICS:
            cells.append(f"{fmt10(metric, 'raw'):>5}/"
                         f"{fmt10(metric, 'C'):<4}/"
                         f"{fmt10(metric, 'L'):<4}/"
                         f"{fmt10(metric, 'CL'):<4}")
        fl = (dozi["TNZI"].get(pid) or {}).get("flag") or ""
        print(f"{label:<14} {tm:<4} {gp:>4}  " + "  ".join(cells) + f"  {fl:<10}")

if __name__ == "__main__":
    main()
