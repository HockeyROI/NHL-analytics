"""Zone-metric variation study.

Builds 30 combinations (5 metrics × 6 event-filter versions) of player-level
zone-time scores for two scenarios (current 2025-26 regular and pooled 4-season),
then correlates team averages with team standings points.

Reads:
  Zones/_player_meta.json
  Zones/raw/pbp/{gameId}.json
  Zones/raw/shifts/{gameId}.json
  Data/game_ids.csv
  NHL standings API (HTTP via curl)

Writes:
  Zones/zone_variations/{metric}_{version}_{forwards|defense}_{regular|pooled}.csv
  Zones/zone_variations/summary_correlations.csv
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

# ===========================================================================
# Constants
# ===========================================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW_PBP = HERE / "raw" / "pbp"
RAW_SHIFTS = HERE / "raw" / "shifts"
GAME_IDS = ROOT / "Data" / "game_ids.csv"
PLAYER_META = HERE / "_player_meta.json"
OUT_DIR = HERE / "zone_variations"

Z = 1.96
Z2 = Z * Z
MIN_SHIFTS = 50
MIN_GP = 20

POOLED_SEASONS = {"20222023", "20232024", "20242025", "20252026"}
CURRENT_SEASON = "20252026"

# Season-end standings dates
SEASON_END_DATE = {
    "20222023": "2023-04-13",
    "20232024": "2024-04-18",
    "20242025": "2025-04-17",
    "20252026": "2026-04-18",
}

# Event-filter versions — L variants restrict to lost-faceoff contexts
EVENT_SETS = {
    "V1": {"faceoff", "hit", "shot-on-goal", "missed-shot", "blocked-shot",
           "goal", "giveaway", "takeaway"},
    "V2": {"shot-on-goal", "missed-shot", "goal"},
    "V3": {"shot-on-goal", "missed-shot", "blocked-shot", "goal"},
}
BASE_VERSION = {"V1": "V1", "V2": "V2", "V3": "V3",
                "V1L": "V1", "V2L": "V2", "V3L": "V3"}
IS_LOST_ONLY = {v: v.endswith("L") for v in
                ("V1", "V2", "V3", "V1L", "V2L", "V3L")}
VERSIONS = ["V1", "V2", "V3", "V1L", "V2L", "V3L"]

METRICS = ["OZI", "DZI", "SDZI", "NZI", "TNZI"]

POS_FORWARD = {"C", "L", "R"}
POS_DEFENSE = {"D"}

FLIP = {"O": "D", "D": "O", "N": "N"}

ABBR_MAP = {"ARI": "UTA"}  # Arizona -> Utah (2024 relocation)

KEY_PLAYERS = [
    ("McDavid",    8478402),
    ("MacKinnon",  8477492),
    ("Draisaitl",  8477934),
    ("Barkov",     8477493),
    ("Makar",      8480069),
    ("Q. Hughes",  8480800),
    ("Nurse",      8477498),
    ("Bouchard",   8480803),
    ("Ekholm",     8475218),
]

# ===========================================================================
# Time helpers
# ===========================================================================

def mmss(s: str) -> int:
    if not s or ":" not in s:
        return 0
    m, se = s.split(":")
    try:
        return int(m) * 60 + int(se)
    except ValueError:
        return 0

def play_abs_time(play: dict) -> int:
    p = (play.get("periodDescriptor") or {}).get("number", 1) or 1
    return (p - 1) * 1200 + mmss(play.get("timeInPeriod", "00:00"))

def norm_team(abbr: str) -> str:
    return ABBR_MAP.get(abbr, abbr)

# ===========================================================================
# Wilson CI
# ===========================================================================

def wilson_lower(p: float, n: float) -> float | None:
    if n is None or n <= 0:
        return None
    p = max(0.0, min(1.0, p))
    denom = 1.0 + Z2 / n
    center = p + Z2 / (2.0 * n)
    margin = Z * math.sqrt(max(0.0, p * (1.0 - p) / n + Z2 / (4.0 * n * n)))
    return (center - margin) / denom

def wilson_upper(p: float, n: float) -> float | None:
    if n is None or n <= 0:
        return None
    p = max(0.0, min(1.0, p))
    denom = 1.0 + Z2 / n
    center = p + Z2 / (2.0 * n)
    margin = Z * math.sqrt(max(0.0, p * (1.0 - p) / n + Z2 / (4.0 * n * n)))
    return (center + margin) / denom

# ===========================================================================
# Correlation
# ===========================================================================

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

# ===========================================================================
# Standings
# ===========================================================================

def fetch_standings(date_str: str) -> dict:
    # The NHL standings-by-date endpoint sometimes returns empty lists for dates
    # that straddle the regular-season boundary (e.g. 2026-04-18 returns 0 rows
    # because regulars ended on 2026-04-17). Fall back one day at a time.
    from datetime import datetime, timedelta
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
            if d_try != date_str:
                print(f"    note: {date_str} returned empty; using {d_try}")
            res = {}
            for row in rows:
                abbr = norm_team(row["teamAbbrev"]["default"])
                res[abbr] = {
                    "points": row["points"],
                    "wins": row["wins"],
                    "gp": row["gamesPlayed"],
                }
            return res
    return {}

def load_all_standings() -> dict[str, dict]:
    return {s: fetch_standings(d) for s, d in SEASON_END_DATE.items()}

# ===========================================================================
# Per-game processing
# ===========================================================================

def build_shift_intervals(shifts_json: dict):
    """Return dict: player_id -> list of (abs_start, abs_end, team_id)."""
    out = defaultdict(list)
    for s in shifts_json.get("data", []):
        pid = s.get("playerId")
        if not pid:
            continue
        period = s.get("period") or 1
        st = s.get("startTime") or "00:00"
        et = s.get("endTime") or "00:00"
        a = (period - 1) * 1200 + mmss(st)
        b = (period - 1) * 1200 + mmss(et)
        if b <= a:
            continue
        out[pid].append((a, b, s.get("teamId")))
    return out

def on_ice_at(intervals_by_pid, t):
    """dict: team_id -> set(player_id) on ice at absolute time t."""
    out = defaultdict(set)
    for pid, ivs in intervals_by_pid.items():
        for a, b, tm in ivs:
            if a <= t < b:
                out[tm].add(pid)
                break
    return out

def shift_end_for(intervals_by_pid, pid, t):
    """Return end time of the shift covering time t for player pid (or None)."""
    for a, b, _tm in intervals_by_pid.get(pid, []):
        if a <= t < b:
            return b
    return None

def process_game(game_id: int,
                 player_season_gp,
                 player_bucket):
    """Process a single game into player buckets.

    player_bucket[(player_id, season, version, fo_zone)] = {
        'shifts': int,
        'lost_shifts': int,
        'oz_sec': float, 'dz_sec': float, 'nz_sec': float, 'total_sec': float,
    }
    """
    pbp_path = RAW_PBP / f"{game_id}.json"
    sh_path = RAW_SHIFTS / f"{game_id}.json"
    if not pbp_path.exists() or not sh_path.exists():
        return
    try:
        pbp = json.load(open(pbp_path))
        shifts_json = json.load(open(sh_path))
    except (json.JSONDecodeError, OSError):
        return

    season = str(pbp.get("season", ""))
    home_id = (pbp.get("homeTeam") or {}).get("id")
    away_id = (pbp.get("awayTeam") or {}).get("id")
    if home_id is None or away_id is None:
        return

    # Build shift intervals
    intervals_by_pid = build_shift_intervals(shifts_json)

    # Track GP: every player with at least one shift this game
    appeared = set(intervals_by_pid.keys())
    for pid in appeared:
        player_season_gp[(pid, season)] += 1

    plays = pbp.get("plays") or []

    # Walk plays, segmenting into contexts starting at each 5v5 faceoff
    # Context dict:
    #   fo_t, fo_zone_home (O/D/N from home perspective),
    #   winning_team_id, losing_team_id,
    #   events: [(t, ez_home, type)],  # ez_home already normalised
    #   players_at_fo: {team_id: set(pids)},
    #   close_t (set when context closes)
    ctx = None

    def close_and_emit(ctx, close_t):
        if ctx is None or not ctx.get("players_at_fo"):
            return
        fo_home = ctx["fo_zone_home"]
        events = ctx["events"]  # list of (t, ez_home, typ)

        # For each version, filter events and attribute time
        # Build per-version event list (t, ez_home)
        per_version_events = {}
        for v in ("V1", "V2", "V3"):
            per_version_events[v] = [
                (t, ez) for (t, ez, typ) in events if typ in EVENT_SETS[v]
            ]

        for team_id, pids in ctx["players_at_fo"].items():
            if team_id not in (home_id, away_id):
                continue
            fo_from_team = fo_home if team_id == home_id else FLIP[fo_home]
            lost = (team_id != ctx["winning_team_id"])

            for pid in pids:
                shift_end = shift_end_for(intervals_by_pid, pid, ctx["fo_t"])
                if shift_end is None:
                    continue
                eff_end = min(close_t, shift_end)
                if eff_end <= ctx["fo_t"]:
                    continue

                for vers in VERSIONS:
                    base = BASE_VERSION[vers]
                    if IS_LOST_ONLY[vers] and not lost:
                        continue
                    key = (pid, season, vers, fo_from_team)
                    bucket = player_bucket[key]

                    # Shift count: once per (player, context, version, fo_zone)
                    bucket["shifts"] = bucket.get("shifts", 0) + 1
                    if lost:
                        bucket["lost_shifts"] = bucket.get("lost_shifts", 0) + 1

                    # Attribute time per filtered event
                    vevents = per_version_events[base]
                    # Only events within [fo_t, eff_end) count
                    filtered = [(t, ez) for (t, ez) in vevents
                                if ctx["fo_t"] <= t < eff_end]
                    if not filtered:
                        continue

                    for i, (t, ez) in enumerate(filtered):
                        t_next = filtered[i + 1][0] if i + 1 < len(filtered) else eff_end
                        dt = max(0.0, t_next - t)
                        ez_player = ez if team_id == home_id else FLIP[ez]
                        bucket["total_sec"] = bucket.get("total_sec", 0.0) + dt
                        key_zone = {"O": "oz_sec", "D": "dz_sec", "N": "nz_sec"}[ez_player]
                        bucket[key_zone] = bucket.get(key_zone, 0.0) + dt

    for p in plays:
        typ = p.get("typeDescKey") or ""
        details = p.get("details") or {}
        t_abs = play_abs_time(p)
        situation = p.get("situationCode") or ""

        if typ == "faceoff":
            # Close prior context at this faceoff time
            if ctx is not None:
                close_and_emit(ctx, t_abs)
                ctx = None

            if situation != "1551":
                continue
            zone_code = details.get("zoneCode")
            if zone_code not in ("O", "D", "N"):
                continue
            event_owner = details.get("eventOwnerTeamId")
            if event_owner not in (home_id, away_id):
                continue
            # zoneCode is from event owner (winner) perspective
            fo_zone_home = zone_code if event_owner == home_id else FLIP[zone_code]
            winner = event_owner
            loser = away_id if winner == home_id else home_id

            on_ice = on_ice_at(intervals_by_pid, t_abs)
            ctx = {
                "fo_t": t_abs,
                "fo_zone_home": fo_zone_home,
                "winning_team_id": winner,
                "losing_team_id": loser,
                "events": [],
                "players_at_fo": {home_id: set(on_ice.get(home_id, set())),
                                  away_id: set(on_ice.get(away_id, set()))},
            }
            # Include the faceoff itself as an event in the context
            ctx["events"].append((t_abs, fo_zone_home, "faceoff"))
            continue

        if ctx is None:
            continue

        # If situation changes mid-context, close context
        if situation and situation != "1551":
            close_and_emit(ctx, t_abs)
            ctx = None
            continue

        # Events with zoneCode & xCoord contribute
        zone_code = details.get("zoneCode")
        if zone_code in ("O", "D", "N") and typ in EVENT_SETS["V1"]:
            event_owner = details.get("eventOwnerTeamId")
            if event_owner not in (home_id, away_id):
                continue
            ez_home = zone_code if event_owner == home_id else FLIP[zone_code]
            ctx["events"].append((t_abs, ez_home, typ))

        # Period end / game end triggers
        if typ in ("period-end", "game-end"):
            close_and_emit(ctx, t_abs)
            ctx = None

    # Close any trailing context at end of game
    if ctx is not None:
        last_t = play_abs_time(plays[-1]) if plays else ctx["fo_t"]
        close_and_emit(ctx, last_t)

# ===========================================================================
# Aggregation -> per-scenario player metrics
# ===========================================================================

def build_scenario_rows(player_bucket, player_season_gp, player_meta, season_filter: set):
    """Return {pid: {version: {fo_zone: bucket_summed}, 'gp': int}}."""
    out = defaultdict(lambda: {"gp": 0, "versions": defaultdict(lambda: defaultdict(
        lambda: {"shifts": 0, "lost_shifts": 0, "total_sec": 0.0,
                 "oz_sec": 0.0, "dz_sec": 0.0, "nz_sec": 0.0}))})
    # GP
    for (pid, season), gp in player_season_gp.items():
        if season in season_filter:
            out[pid]["gp"] += gp
    # Buckets
    for (pid, season, vers, fo_zone), b in player_bucket.items():
        if season not in season_filter:
            continue
        dst = out[pid]["versions"][vers][fo_zone]
        dst["shifts"] += b.get("shifts", 0)
        dst["lost_shifts"] += b.get("lost_shifts", 0)
        dst["total_sec"] += b.get("total_sec", 0.0)
        dst["oz_sec"] += b.get("oz_sec", 0.0)
        dst["dz_sec"] += b.get("dz_sec", 0.0)
        dst["nz_sec"] += b.get("nz_sec", 0.0)
    return out

def compute_metric(b_ozfo, b_dzfo, b_nzfo, metric: str, n_kind: str):
    """Given zone-specific buckets for a player, compute raw p and Wilson-adjusted
    value for the requested metric. n_kind is 'shifts' (all) or 'lost_shifts'."""
    def prop(b, num_key):
        if not b or b["total_sec"] <= 0:
            return None
        return b[num_key] / b["total_sec"]

    def n(b):
        if not b:
            return 0
        return b["shifts"] if n_kind == "shifts" else b["lost_shifts"]

    if metric == "OZI":
        p = prop(b_ozfo, "oz_sec"); N = n(b_ozfo)
        if p is None or N < MIN_SHIFTS: return None, None
        return p, wilson_lower(p, N)
    if metric == "DZI":
        p = prop(b_dzfo, "oz_sec"); N = n(b_dzfo)
        if p is None or N < MIN_SHIFTS: return None, None
        return p, wilson_lower(p, N)
    if metric == "SDZI":
        if not b_dzfo or b_dzfo["total_sec"] <= 0: return None, None
        p = (b_dzfo["oz_sec"] + b_dzfo["nz_sec"]) / b_dzfo["total_sec"]
        N = n(b_dzfo)
        if N < MIN_SHIFTS: return None, None
        return p, wilson_lower(p, N)
    if metric == "NZI":
        p = prop(b_nzfo, "oz_sec"); N = n(b_nzfo)
        if p is None or N < MIN_SHIFTS: return None, None
        return p, wilson_lower(p, N)
    if metric == "TNZI":
        if not b_nzfo or b_nzfo["total_sec"] <= 0: return None, None
        p_oz = b_nzfo["oz_sec"] / b_nzfo["total_sec"]
        p_dz = b_nzfo["dz_sec"] / b_nzfo["total_sec"]
        N = n(b_nzfo)
        if N < MIN_SHIFTS: return None, None
        lo = wilson_lower(p_oz, N)
        hi = wilson_upper(p_dz, N)
        if lo is None or hi is None:
            return None, None
        return (p_oz - p_dz), (lo - hi)
    return None, None

def normalize_to_score(rows_with_val):
    """rows_with_val: list of (player_key, val_or_None); returns dict key->score (0-10 or None)."""
    vals = [v for _, v in rows_with_val if v is not None]
    if not vals:
        return {k: None for k, _ in rows_with_val}
    lo, hi = min(vals), max(vals); span = hi - lo
    out = {}
    for k, v in rows_with_val:
        if v is None:
            out[k] = None
        elif span == 0:
            out[k] = 5.0
        else:
            out[k] = round((v - lo) / span * 10.0, 1)
    return out

# ===========================================================================
# Main pipeline
# ===========================================================================

def load_game_ids():
    """Return list of (game_id, season) for regular games of 4 target seasons."""
    out = []
    with open(GAME_IDS) as f:
        for r in csv.DictReader(f):
            if r["season"] in POOLED_SEASONS and r["game_type"] == "regular":
                out.append((int(r["game_id"]), r["season"]))
    return out

def main():
    print("[1/8] fetching standings ...")
    standings = load_all_standings()
    for s in standings:
        print(f"    {s}: {len(standings[s])} teams")

    print("[2/8] loading player meta ...")
    player_meta = {int(k): v for k, v in json.load(open(PLAYER_META)).items()}

    print("[3/8] reading game_ids.csv ...")
    games = load_game_ids()
    print(f"    {len(games)} regular games across 4 seasons")

    print("[4/8] processing raw PBP + shifts per game ...")
    player_bucket = defaultdict(
        lambda: {"shifts": 0, "lost_shifts": 0, "total_sec": 0.0,
                 "oz_sec": 0.0, "dz_sec": 0.0, "nz_sec": 0.0})
    player_season_gp = defaultdict(int)

    total = len(games)
    for i, (gid, season) in enumerate(games):
        process_game(gid, player_season_gp, player_bucket)
        if (i + 1) % 500 == 0 or i + 1 == total:
            print(f"    processed {i+1}/{total}")

    print("[5/8] computing metrics per scenario ...")
    scenarios = {
        "regular": {CURRENT_SEASON},
        "pooled": POOLED_SEASONS,
    }

    # scenario_player_metrics[scenario][(metric, version)] = {pid: (raw, adj)}
    scenario_player_metrics = {}
    scenario_player_rows = {}
    for scen_name, season_set in scenarios.items():
        bundle = build_scenario_rows(player_bucket, player_season_gp,
                                     player_meta, season_set)
        per_mv = defaultdict(dict)  # (metric,vers) -> pid -> (raw, adj)
        for pid, data in bundle.items():
            meta = player_meta.get(pid, {})
            pos = (meta.get("position") or "").upper()
            if pos in ("G", ""):
                continue
            if data["gp"] < MIN_GP:
                continue
            for vers in VERSIONS:
                vbuckets = data["versions"].get(vers, {})
                b_o = vbuckets.get("O")
                b_d = vbuckets.get("D")
                b_n = vbuckets.get("N")
                n_kind = "lost_shifts" if IS_LOST_ONLY[vers] else "shifts"
                for metric in METRICS:
                    raw, adj = compute_metric(b_o, b_d, b_n, metric, n_kind)
                    if raw is None:
                        continue
                    per_mv[(metric, vers)][pid] = (raw, adj)
        scenario_player_metrics[scen_name] = per_mv
        scenario_player_rows[scen_name] = bundle

    print("[6/8] normalising scores by position group and writing CSVs ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # all_rows[scen][(metric, vers)][group] = list[dict]
    all_rows = defaultdict(lambda: defaultdict(dict))

    for scen_name, per_mv in scenario_player_metrics.items():
        bundle = scenario_player_rows[scen_name]
        for (metric, vers), pid_map in per_mv.items():
            for group, pos_set in (("forwards", POS_FORWARD),
                                    ("defense", POS_DEFENSE)):
                subset = [(pid, adj) for pid, (raw, adj) in pid_map.items()
                          if (player_meta.get(pid, {}).get("position") or "").upper() in pos_set
                          and adj is not None]
                scores = normalize_to_score([(pid, val) for pid, val in subset])

                rows = []
                for pid, (raw, adj) in pid_map.items():
                    meta = player_meta.get(pid, {})
                    pos = (meta.get("position") or "").upper()
                    if pos not in pos_set:
                        continue
                    gp = bundle[pid]["gp"]
                    vbuckets = bundle[pid]["versions"].get(vers, {})
                    b_o = vbuckets.get("O") or {"shifts": 0, "lost_shifts": 0}
                    b_d = vbuckets.get("D") or {"shifts": 0, "lost_shifts": 0}
                    b_n = vbuckets.get("N") or {"shifts": 0, "lost_shifts": 0}
                    if IS_LOST_ONLY[vers]:
                        oz_fs, dz_fs, nz_fs = b_o["lost_shifts"], b_d["lost_shifts"], b_n["lost_shifts"]
                    else:
                        oz_fs, dz_fs, nz_fs = b_o["shifts"], b_d["shifts"], b_n["shifts"]
                    row = {
                        "player_id": pid,
                        "player_name": meta.get("name", ""),
                        "team": norm_team(meta.get("team_abbrev", "") or ""),
                        "pos": pos,
                        "GP": gp,
                        "oz_faceoff_shifts": oz_fs,
                        "dz_faceoff_shifts": dz_fs,
                        "nz_faceoff_shifts": nz_fs,
                        "raw_metric": round(raw, 5) if raw is not None else "",
                        "wilson_adjusted": round(adj, 5) if adj is not None else "",
                        "score": scores.get(pid, ""),
                    }
                    rows.append(row)

                # Rank by score
                rows_sorted = sorted(rows, key=lambda r: (r["score"] if r["score"] != "" else -999),
                                     reverse=True)
                for rank, r in enumerate(rows_sorted, 1):
                    r["rank"] = rank if r["score"] != "" else ""

                all_rows[scen_name][(metric, vers)][group] = rows_sorted

                path = OUT_DIR / f"{metric}_{vers}_{group}_{scen_name}.csv"
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["rank", "player_name", "team", "pos", "GP",
                                "oz_faceoff_shifts", "dz_faceoff_shifts", "nz_faceoff_shifts",
                                "raw_metric", "wilson_adjusted", "score", "player_id"])
                    for r in rows_sorted:
                        w.writerow([r["rank"], r["player_name"], r["team"], r["pos"], r["GP"],
                                    r["oz_faceoff_shifts"], r["dz_faceoff_shifts"], r["nz_faceoff_shifts"],
                                    r["raw_metric"], r["wilson_adjusted"], r["score"], r["player_id"]])

    print("[7/8] computing team-level correlations ...")
    # Team points targets
    team_points_current = {tm: st["points"] for tm, st in standings[CURRENT_SEASON].items()}
    # Average across 4 seasons (per user spec)
    team_points_pooled = defaultdict(list)
    for s in POOLED_SEASONS:
        for tm, st in standings[s].items():
            team_points_pooled[tm].append(st["points"])
    team_points_pooled_avg = {tm: mean(vs) for tm, vs in team_points_pooled.items()}

    target_by_scen = {
        "regular": team_points_current,
        "pooled": team_points_pooled_avg,
    }

    corr_results = []  # list of dicts for summary
    for scen_name in ("regular", "pooled"):
        targets = target_by_scen[scen_name]
        for metric in METRICS:
            for vers in VERSIONS:
                # Combine forward+defense rows for team averages
                combined = (all_rows[scen_name][(metric, vers)].get("forwards", []) +
                            all_rows[scen_name][(metric, vers)].get("defense", []))
                team_scores = defaultdict(list)
                for r in combined:
                    if r["score"] == "":
                        continue
                    team_scores[r["team"]].append(r["score"])
                # Team average
                team_avg = {tm: mean(sc) for tm, sc in team_scores.items() if sc}
                xs, ys, teams = [], [], []
                for tm, avg in team_avg.items():
                    if tm in targets:
                        xs.append(avg); ys.append(targets[tm]); teams.append(tm)
                if len(xs) >= 5:
                    pr = pearson(xs, ys); sp = spearman(xs, ys)
                else:
                    pr = float("nan"); sp = float("nan")
                corr_results.append({
                    "scenario": scen_name,
                    "metric": metric,
                    "version": vers,
                    "pearson": pr,
                    "spearman": sp,
                    "teams": len(xs),
                })

    # Build summary dict keyed by (metric,version)
    summary = {(c["metric"], c["version"]): {} for c in corr_results}
    for c in corr_results:
        summary[(c["metric"], c["version"])][c["scenario"]] = c

    # Top forward / top defender in current scenario
    def top_name(scen, metric, vers, group):
        lst = all_rows[scen][(metric, vers)].get(group, [])
        for r in lst:
            if r["score"] != "":
                return f"{r['player_name']} ({r['team']})"
        return ""

    summary_rows = []
    for (metric, vers), sc in summary.items():
        cur = sc.get("regular", {})
        pool = sc.get("pooled", {})
        row = {
            "metric": metric,
            "version": vers,
            "pearson_r_current": cur.get("pearson", float("nan")),
            "spearman_current": cur.get("spearman", float("nan")),
            "pearson_r_pooled": pool.get("pearson", float("nan")),
            "spearman_pooled": pool.get("spearman", float("nan")),
            "top_F_current": top_name("regular", metric, vers, "forwards"),
            "top_D_current": top_name("regular", metric, vers, "defense"),
        }
        summary_rows.append(row)

    summary_path = OUT_DIR / "summary_correlations.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "version",
                    "pearson_r_current", "spearman_current",
                    "pearson_r_pooled", "spearman_pooled",
                    "top_F_current", "top_D_current"])
        for r in summary_rows:
            w.writerow([r["metric"], r["version"],
                        round(r["pearson_r_current"], 4) if not math.isnan(r["pearson_r_current"]) else "",
                        round(r["spearman_current"], 4) if not math.isnan(r["spearman_current"]) else "",
                        round(r["pearson_r_pooled"], 4) if not math.isnan(r["pearson_r_pooled"]) else "",
                        round(r["spearman_pooled"], 4) if not math.isnan(r["spearman_pooled"]) else "",
                        r["top_F_current"], r["top_D_current"]])
    print(f"    wrote {summary_path}")

    print("[8/8] printing reports ...")
    print_summary_tables(summary_rows)
    print_key_players(all_rows, summary_rows)

# ===========================================================================
# Reporting
# ===========================================================================

def print_summary_tables(summary_rows):
    print("\n" + "=" * 110)
    print("MASTER SUMMARY — 30 combinations ranked by Pearson r (CURRENT 2025-26)")
    print("=" * 110)
    hdr = (f"{'Metric':<6} {'Ver':<4} {'Pr_cur':>7} {'Sp_cur':>7} "
           f"{'Pr_pool':>8} {'Sp_pool':>8}  {'Top F (current)':<30} {'Top D (current)':<30}")
    print(hdr)
    ranked_cur = sorted(summary_rows,
                        key=lambda r: (r["pearson_r_current"]
                                       if not math.isnan(r["pearson_r_current"]) else -2),
                        reverse=True)
    def fmt(v):
        return f"{v:+.3f}" if isinstance(v, float) and not math.isnan(v) else "   -  "
    for r in ranked_cur:
        print(f"{r['metric']:<6} {r['version']:<4} {fmt(r['pearson_r_current']):>7} "
              f"{fmt(r['spearman_current']):>7} {fmt(r['pearson_r_pooled']):>8} "
              f"{fmt(r['spearman_pooled']):>8}  "
              f"{(r['top_F_current'] or '')[:30]:<30} {(r['top_D_current'] or '')[:30]:<30}")

    print("\n" + "=" * 110)
    print("MASTER SUMMARY — 30 combinations ranked by Pearson r (POOLED 4-season avg)")
    print("=" * 110)
    print(hdr)
    ranked_pool = sorted(summary_rows,
                         key=lambda r: (r["pearson_r_pooled"]
                                        if not math.isnan(r["pearson_r_pooled"]) else -2),
                         reverse=True)
    for r in ranked_pool:
        print(f"{r['metric']:<6} {r['version']:<4} {fmt(r['pearson_r_current']):>7} "
              f"{fmt(r['spearman_current']):>7} {fmt(r['pearson_r_pooled']):>8} "
              f"{fmt(r['spearman_pooled']):>8}  "
              f"{(r['top_F_current'] or '')[:30]:<30} {(r['top_D_current'] or '')[:30]:<30}")

def print_key_players(all_rows, summary_rows):
    # Top 5 combinations by current Pearson r
    ranked_cur = sorted(
        summary_rows,
        key=lambda r: (r["pearson_r_current"] if not math.isnan(r["pearson_r_current"]) else -2),
        reverse=True,
    )[:5]
    print("\n" + "=" * 110)
    print("KEY PLAYER RANKINGS — top 5 combinations by Pearson r (CURRENT)")
    print("=" * 110)
    for r in ranked_cur:
        m, v = r["metric"], r["version"]
        print(f"\n[{m} / {v}]  Pearson_cur={r['pearson_r_current']:+.3f}  "
              f"Pearson_pool={r['pearson_r_pooled']:+.3f}")
        for scen_name in ("regular", "pooled"):
            print(f"  -- {scen_name} --")
            f_rows = all_rows[scen_name][(m, v)].get("forwards", [])
            d_rows = all_rows[scen_name][(m, v)].get("defense", [])
            # Index by player_id
            idx_all = {r2["player_id"]: r2 for r2 in (f_rows + d_rows)}
            for label, pid in KEY_PLAYERS:
                r2 = idx_all.get(pid)
                if not r2 or r2["score"] == "":
                    print(f"     {label:<12} — did not qualify")
                else:
                    print(f"     {label:<12} rank={r2['rank']:<4} team={r2['team']:<4} "
                          f"GP={r2['GP']:<3} score={r2['score']:<4} "
                          f"adj={r2['wilson_adjusted']:<8}")

if __name__ == "__main__":
    main()
