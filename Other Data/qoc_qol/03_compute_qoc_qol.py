"""Compute 5v5 QOC and QOL for EDM skaters, 2025-26 regular season.

Pipeline
--------
1. For each EDM game (81 games): parse shift chart into 5v5 intervals and
   accumulate, for every EDM skater, pairwise overlap seconds with
     - every teammate    -> QOL weight
     - every opponent    -> QOC weight

2. For ALL 2025-26 regular games (1306 games): compute per-player
     - 5v5 TOI total and games played   -> 5v5 TOI/game
     - On-ice 5v5 CF and CA             -> CF% = CF / (CF + CA)
   CF/CA is derived by joining 5v5 shot events with the per-game on-ice
   intervals built from shift charts.

3. Normalise both metrics league-wide to [0, 1] using min-max on players
   with >= MIN_TOI_SEC season 5v5 TOI; composite quality = mean of the two.

4. For each EDM skater, QOC = weighted mean of opponents' quality (weights =
   head-to-head overlap seconds) and QOL = weighted mean of teammates'
   quality (weights = shared-ice overlap seconds).

5v5 detection
-------------
Identify goalies per game with the heuristic: a player is a goalie iff one
of their shift durations exceeds 300 seconds (verified against PBP
rosterSpots on 40 EDM games - zero mismatches). An interval is 5v5 iff
each team has exactly 5 non-goalie skaters on the ice.
"""
from __future__ import annotations
import csv, json, os
from collections import defaultdict
from bisect import bisect_right

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(ROOT)
SHIFTS_DIR = os.path.join(ROOT, "shifts")
PBP_DIR = os.path.join(ROOT, "pbp")
GAME_IDS_CSV = os.path.join(DATA_ROOT, "game_ids.csv")
SHOT_EVENTS_CSV = os.path.join(DATA_ROOT, "nhl_shot_events.csv")

SEASON = "20252026"
GAME_TYPE = "regular"
TEAM = "EDM"
MIN_TOI_SEC_FOR_NORM = 5 * 60 * 10          # require ~50 min 5v5 TOI to enter the normalisation pool
SHOT_EVENTS = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}
SITCODE_5V5 = "1551"

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def hms_to_sec(s: str) -> int:
    mm, ss = s.split(":")
    return int(mm) * 60 + int(ss)

def abs_time(period: int, period_seconds: int) -> int:
    """Convert (period, seconds-into-period) to absolute game seconds."""
    return (period - 1) * 1200 + period_seconds

def load_games():
    with open(GAME_IDS_CSV) as f:
        return [row for row in csv.DictReader(f)
                if row["season"] == SEASON and row["game_type"] == GAME_TYPE]

# ----------------------------------------------------------------------
# per-game 5v5 interval builder
# ----------------------------------------------------------------------
def build_game_intervals(shift_json: dict, pbp_goalies: set[int] | None = None):
    """Return (intervals, per_player_5v5_toi, goalies, home_team_id, away_team_id).

    intervals is list of (start, end, home_on_ice_set, away_on_ice_set) for
    every 5v5 sub-interval in the game.
    """
    shifts_raw = [s for s in shift_json["data"]
                  if s.get("typeCode") == 517 and s.get("duration")]

    # Identify goalies by max-shift heuristic (verified elsewhere).
    max_dur = defaultdict(int)
    for s in shifts_raw:
        d = hms_to_sec(s["duration"])
        if d > max_dur[s["playerId"]]:
            max_dur[s["playerId"]] = d
    goalies = {pid for pid, d in max_dur.items() if d > 300}
    if pbp_goalies:
        goalies |= pbp_goalies

    # Determine which teamId is home. In the shift chart, teamId is fine; we
    # just need two sides. Pick them by first-seen order.
    team_ids = []
    for s in shifts_raw:
        if s["teamId"] not in team_ids:
            team_ids.append(s["teamId"])
        if len(team_ids) == 2:
            break
    side_of = {team_ids[0]: "A", team_ids[1]: "B"} if len(team_ids) == 2 else {}

    # Build (+/-) change-point events.
    events = []           # (abs_time, delta, player_id, side)
    for s in shifts_raw:
        if s["teamId"] not in side_of:
            continue
        side = side_of[s["teamId"]]
        p = s["period"]
        start = abs_time(p, hms_to_sec(s["startTime"]))
        end = abs_time(p, hms_to_sec(s["endTime"]))
        if end <= start:
            continue
        pid = s["playerId"]
        events.append((start, +1, pid, side))
        events.append((end,   -1, pid, side))

    events.sort(key=lambda e: (e[0], e[1]))   # process exits before entries at same t? Actually either works, below is agnostic

    on_A: set[int] = set()
    on_B: set[int] = set()
    intervals = []
    player_5v5_toi = defaultdict(int)

    i = 0
    n = len(events)
    while i < n:
        t = events[i][0]
        # Compute the on-ice state just before t (already in on_A/on_B)
        # Record interval [prev_t, t] using current sets
        if intervals or i > 0:
            pass
        # Actually we need prev_t. Track it.
        i += 1
    # Rewrite with prev_t tracking:
    on_A.clear(); on_B.clear()
    intervals.clear()
    player_5v5_toi.clear()

    prev_t = events[0][0] if events else 0
    i = 0
    while i < n:
        t = events[i][0]
        if t > prev_t:
            skaters_A = on_A - goalies
            skaters_B = on_B - goalies
            if len(skaters_A) == 5 and len(skaters_B) == 5:
                dur = t - prev_t
                intervals.append((prev_t, t, frozenset(skaters_A), frozenset(skaters_B)))
                for pid in skaters_A:
                    player_5v5_toi[pid] += dur
                for pid in skaters_B:
                    player_5v5_toi[pid] += dur
            prev_t = t
        # apply all events at t
        while i < n and events[i][0] == t:
            _, delta, pid, side = events[i]
            target = on_A if side == "A" else on_B
            if delta == +1:
                target.add(pid)
            else:
                target.discard(pid)
            i += 1

    team_of_side = {"A": team_ids[0], "B": team_ids[1]} if len(team_ids) == 2 else {}
    return intervals, player_5v5_toi, goalies, side_of, team_of_side

# ----------------------------------------------------------------------
# main pipeline
# ----------------------------------------------------------------------
def main():
    games = load_games()
    print(f"[load] {len(games)} regular-season games in {SEASON}")

    # EDM games for QOC/QOL overlap
    edm_games = [g for g in games if TEAM in (g["home_abbrev"], g["away_abbrev"])]
    edm_gids = {g["game_id"] for g in edm_games}
    print(f"[load] {len(edm_games)} EDM games")

    # player -> team appearance counter (to pick the most frequent team)
    player_team_games = defaultdict(lambda: defaultdict(int))
    player_name = {}

    # Load PBP rosterSpots for EDM games to seed names and goalie info.
    for g in edm_games:
        gid = g["game_id"]
        p_path = os.path.join(PBP_DIR, f"{gid}.json")
        if os.path.exists(p_path):
            pbp = json.load(open(p_path))
            for p in pbp.get("rosterSpots", []):
                pid = p["playerId"]
                name = f"{p['firstName']['default']} {p['lastName']['default']}"
                player_name[pid] = (name, p["positionCode"])

    # Per-player season 5v5 TOI totals and games played
    season_toi = defaultdict(int)
    season_gp  = defaultdict(int)          # unique games with any 5v5 TOI
    season_cf  = defaultdict(int)
    season_ca  = defaultdict(int)
    # Also remember team per game for CF/CA orientation
    # team_id_of_player_in_game[(gid, pid)] = team_id (from shifts)

    # Pre-load shot events keyed by game_id for 5v5
    print("[shots] loading shot events CSV ...")
    shots_by_game = defaultdict(list)
    with open(SHOT_EVENTS_CSV) as f:
        for row in csv.DictReader(f):
            if row["season"] != SEASON or row["game_type"] != GAME_TYPE:
                continue
            if row["situation_code"] != SITCODE_5V5:
                continue
            if row["event_type"] not in SHOT_EVENTS:
                continue
            gid = row["game_id"]
            try:
                tsecs = int(row["time_secs"])
            except ValueError:
                tsecs = hms_to_sec(row["time_in_period"])
            t_abs = abs_time(int(row["period"]), tsecs)
            shots_by_game[gid].append((t_abs, int(row["shooting_team_id"])))
    print(f"[shots] total 5v5 shot events: {sum(len(v) for v in shots_by_game.values())}")

    # EDM pairwise overlap accumulators:
    # overlap[edm_pid][other_pid] = seconds of 5v5 overlap
    # We track (teammate/opponent) separately.
    edm_teammate_overlap = defaultdict(lambda: defaultdict(int))   # edm_pid -> edm_teammate_pid -> secs
    edm_opponent_overlap = defaultdict(lambda: defaultdict(int))   # edm_pid -> opp_pid -> secs

    print("[process] walking all shift charts ...")
    for idx, g in enumerate(games, 1):
        gid = g["game_id"]
        shift_path = os.path.join(SHIFTS_DIR, f"{gid}.json")
        if not os.path.exists(shift_path):
            continue
        shift_json = json.load(open(shift_path))
        pbp_goalies = None
        p_path = os.path.join(PBP_DIR, f"{gid}.json")
        if os.path.exists(p_path):
            pbp = json.load(open(p_path))
            pbp_goalies = {p["playerId"] for p in pbp.get("rosterSpots", [])
                           if p["positionCode"] == "G"}

        intervals, p5v5_toi, goalies, side_of, team_of_side = build_game_intervals(
            shift_json, pbp_goalies
        )
        if not intervals:
            continue

        # record team of each non-goalie player in this game
        # (take side from intervals themselves - we have the sets, not sides;
        #  but we have team_of_side from the builder)
        # season totals
        for pid, toi in p5v5_toi.items():
            season_toi[pid] += toi
            season_gp[pid] += 1
        # track player team affiliation across games
        # derive team per player in this game from the first interval they appear in
        for start, end, on_A, on_B in intervals:
            for pid in on_A:
                player_team_games[pid][team_of_side.get("A")] += 1
            for pid in on_B:
                player_team_games[pid][team_of_side.get("B")] += 1
            # Only need one pass to tag team; break would miss someone appearing later, so continue all
            # But counts blow up linearly. Acceptable - we take argmax after.

        # --- CF / CA attribution ---
        starts = [iv[0] for iv in intervals]
        shots = shots_by_game.get(gid, [])
        for t_abs, shooting_team_id in shots:
            # find the interval whose [start, end) contains t_abs
            i = bisect_right(starts, t_abs) - 1
            if i < 0 or i >= len(intervals):
                continue
            s, e, on_A, on_B = intervals[i]
            if not (s <= t_abs < e):
                continue
            team_A = team_of_side.get("A")
            team_B = team_of_side.get("B")
            for pid in on_A:
                if team_A == shooting_team_id:
                    season_cf[pid] += 1
                else:
                    season_ca[pid] += 1
            for pid in on_B:
                if team_B == shooting_team_id:
                    season_cf[pid] += 1
                else:
                    season_ca[pid] += 1

        # --- EDM QOC / QOL accumulation ---
        if gid in edm_gids:
            edm_team_id = None
            # Find EDM's team id in this game by looking up the game's abbrevs
            if g["home_abbrev"] == TEAM:
                # the home team's teamId — we don't know without PBP; use PBP
                pass
            # Simpler: use PBP rosterSpots' teamId grouping
            if os.path.exists(p_path):
                pbp = json.load(open(p_path))
                home_id = pbp["homeTeam"]["id"]
                away_id = pbp["awayTeam"]["id"]
                edm_team_id = home_id if pbp["homeTeam"]["abbrev"] == TEAM else away_id
            else:
                # fallback: infer by which side contained a known EDM player
                edm_team_id = None
            if edm_team_id is None:
                # fallback: identify EDM side from team_of_side + expected teamAbbrev
                # (not perfect without PBP but all EDM games have PBP)
                continue

            for start, end, on_A, on_B in intervals:
                dur = end - start
                team_A = team_of_side.get("A")
                team_B = team_of_side.get("B")
                if edm_team_id == team_A:
                    edm_set, opp_set = on_A, on_B
                elif edm_team_id == team_B:
                    edm_set, opp_set = on_B, on_A
                else:
                    continue
                edm_list = list(edm_set)
                opp_list = list(opp_set)
                # teammate overlap (unordered but stored per EDM player)
                for i, p1 in enumerate(edm_list):
                    for p2 in edm_list:
                        if p1 == p2: continue
                        edm_teammate_overlap[p1][p2] += dur
                    for p2 in opp_list:
                        edm_opponent_overlap[p1][p2] += dur
        if idx % 200 == 0:
            print(f"  processed {idx}/{len(games)} games")

    # Player -> most common teamId across all games
    player_team = {}
    for pid, counts in player_team_games.items():
        if not counts: continue
        player_team[pid] = max(counts.items(), key=lambda x: x[1])[0]

    # Map teamId -> abbrev using game_ids table (home/away abbrev + id via pbp)
    team_abbrev = {}
    # easiest: for each EDM game's pbp, extract team ids and abbrevs
    for g in edm_games:
        p_path = os.path.join(PBP_DIR, f"{g['game_id']}.json")
        if os.path.exists(p_path):
            pbp = json.load(open(p_path))
            team_abbrev[pbp["homeTeam"]["id"]] = pbp["homeTeam"]["abbrev"]
            team_abbrev[pbp["awayTeam"]["id"]] = pbp["awayTeam"]["abbrev"]

    # ------------------------------------------------------------------
    # Quality score: normalise TOI/gp and CF% league-wide
    # ------------------------------------------------------------------
    print("[quality] computing league-wide quality scores ...")
    pool = [pid for pid, toi in season_toi.items() if toi >= MIN_TOI_SEC_FOR_NORM]
    print(f"[quality] qualified player pool: {len(pool)} (>= {MIN_TOI_SEC_FOR_NORM/60:.0f} min 5v5 TOI)")

    toi_per_gp = {pid: season_toi[pid] / season_gp[pid] for pid in pool if season_gp[pid] > 0}
    cfpct = {}
    for pid in pool:
        cf, ca = season_cf[pid], season_ca[pid]
        if cf + ca > 0:
            cfpct[pid] = cf / (cf + ca)

    # Use only players with both metrics
    final_pool = [pid for pid in pool if pid in toi_per_gp and pid in cfpct]
    t_vals = [toi_per_gp[p] for p in final_pool]
    c_vals = [cfpct[p] for p in final_pool]
    t_min, t_max = min(t_vals), max(t_vals)
    c_min, c_max = min(c_vals), max(c_vals)
    print(f"[quality] TOI/game range: {t_min:.1f}s - {t_max:.1f}s")
    print(f"[quality] CF% range: {c_min:.3f} - {c_max:.3f}")

    def norm(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else 0.5

    quality = {}
    for pid in final_pool:
        t_n = norm(toi_per_gp[pid], t_min, t_max)
        c_n = norm(cfpct[pid], c_min, c_max)
        quality[pid] = (t_n + c_n) / 2.0

    # ------------------------------------------------------------------
    # QOC and QOL for EDM players
    # ------------------------------------------------------------------
    print("[qoc/qol] computing for EDM skaters ...")
    # Identify current EDM skaters: in player_team, team_abbrev[...] == 'EDM' AND positionCode != 'G'
    edm_skaters = []
    for pid, tid in player_team.items():
        if team_abbrev.get(tid) != TEAM:
            continue
        nm = player_name.get(pid)
        if not nm: continue
        name, pos = nm
        if pos == "G":
            continue
        edm_skaters.append(pid)

    rows = []
    for pid in edm_skaters:
        name, pos = player_name[pid]
        # Teammate weighted quality
        def weighted_q(overlap_map):
            tot_w = 0.0
            tot_wq = 0.0
            for other_pid, secs in overlap_map.items():
                q = quality.get(other_pid)
                if q is None:  # insufficient sample; skip
                    continue
                tot_w += secs
                tot_wq += secs * q
            return (tot_wq / tot_w, tot_w) if tot_w > 0 else (float("nan"), 0)

        qol, w_teammate = weighted_q(edm_teammate_overlap.get(pid, {}))
        qoc, w_opp = weighted_q(edm_opponent_overlap.get(pid, {}))
        toi = season_toi.get(pid, 0)
        gp = season_gp.get(pid, 0)
        own_q = quality.get(pid, float("nan"))
        cf = season_cf.get(pid, 0); ca = season_ca.get(pid, 0)
        own_cfpct = cf / (cf + ca) if cf + ca > 0 else float("nan")
        rows.append({
            "player_id": pid,
            "name": name,
            "position": pos,
            "games_played": gp,
            "toi_5v5_total_sec": toi,
            "toi_5v5_per_game_sec": toi / gp if gp else 0,
            "cf_5v5": cf,
            "ca_5v5": ca,
            "cfpct_5v5": round(own_cfpct, 4) if own_cfpct == own_cfpct else "",
            "quality_self": round(own_q, 4) if own_q == own_q else "",
            "qoc_quality": round(qoc, 4) if qoc == qoc else "",
            "qol_quality": round(qol, 4) if qol == qol else "",
            "qoc_weight_sec": int(w_opp),
            "qol_weight_sec": int(w_teammate),
        })

    rows.sort(key=lambda r: r["qoc_quality"] if isinstance(r["qoc_quality"], float) else -1,
              reverse=True)

    out_all = os.path.join(ROOT, "edm_qoc_qol.csv")
    with open(out_all, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {out_all}  ({len(rows)} EDM skaters)")

    # Forwards table (ranked by QOC, and a second by QOL)
    forwards = [r for r in rows if r["position"] in ("C", "L", "R")
                and isinstance(r["qoc_quality"], float)]
    forwards_by_qoc = sorted(forwards, key=lambda r: r["qoc_quality"], reverse=True)
    forwards_by_qol = sorted(forwards, key=lambda r: r["qol_quality"], reverse=True)

    def _fmt(v, spec=">8.4f", empty=">8"):
        return format(v, spec) if isinstance(v, float) else format("-", empty)

    def _print_table(title, rows):
        print("\n=============================================================")
        print(title)
        print("=============================================================")
        print(f"{'Rank':<5}{'Name':<25}{'Pos':<5}{'GP':>4}{'TOI/G':>8}{'CF%':>8}"
              f"{'QOC':>8}{'QOL':>8}{'Self':>8}")
        for i, r in enumerate(rows, 1):
            toi_g = r["toi_5v5_per_game_sec"]
            cfp = r["cfpct_5v5"]
            cf_str = f"{cfp*100:>7.1f}%" if isinstance(cfp, float) else format("-", ">8")
            print(f"{i:<5}{r['name'][:24]:<25}{r['position']:<5}{r['games_played']:>4}"
                  f"{toi_g/60:>7.2f}m{cf_str}"
                  f"{_fmt(r['qoc_quality'])}{_fmt(r['qol_quality'])}{_fmt(r['quality_self'])}")

    _print_table("EDM Forwards ranked by QOC (hardest opponents first)", forwards_by_qoc)
    _print_table("EDM Forwards ranked by QOL (best linemates first)", forwards_by_qol)

if __name__ == "__main__":
    main()
