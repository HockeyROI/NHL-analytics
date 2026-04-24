"""compute_zone_and_overlap.py
-------------------------------
Single-pass through every shift chart + play-by-play for seasons 2022-23
through 2025-26 (regular + playoff). Produces intermediate artefacts used by
compute_pqr_roc_rol.py.

NEW METHODOLOGY (faceoff-start shifts):
  OER and DER (and NZ_ratio) now measure ONLY shifts that began with the
  player on the ice for a faceoff. Line-change shifts are ignored. This
  script therefore tracks, per player, per scenario:

    total_5v5_toi_sec        overall 5v5 TOI across all shifts (for PQR TOI)
    games_played             games with any 5v5 ice time
    team_id / team_abbrev    most recent team seen

    oz_fo_shifts             count of 5v5 OZ-faceoff start shifts
    oz_fo_5v5_sec            total 5v5 ice seconds during those shifts
    oz_fo_oz_sec             5v5 OZ time during those shifts
    oz_fo_dz_sec, oz_fo_nz_sec

    dz_fo_shifts, dz_fo_5v5_sec, dz_fo_oz_sec, dz_fo_dz_sec, dz_fo_nz_sec
    nz_fo_shifts, nz_fo_5v5_sec, nz_fo_oz_sec, nz_fo_dz_sec, nz_fo_nz_sec

  A shift "started with a faceoff" iff the faceoff's abs time is within
  FACEOFF_MATCH_SEC of the shift's start. The faceoff's zone must itself
  be at 5v5 (situationCode == "1551") for the shift to be counted.

Outputs (under Player_Ranking/):

  Player_Ranking/_player_meta.json
  Player_Ranking/_zone_time.json
  Player_Ranking/_overlap.pkl

Puck zone between events is the zone of the last event with xCoord - a
known approximation. `homeTeamDefendingSide` normalises coordinates per
period. 5v5 is enforced via situationCode == "1551" AND a 5-skater check
from the shift chart.
"""
from __future__ import annotations
import csv, json, os, pickle, time
from collections import defaultdict
from bisect import bisect_right, bisect_left
from itertools import combinations

ROOT       = os.path.dirname(os.path.abspath(__file__))
PROJECT    = os.path.dirname(ROOT)
SHIFTS_DIR = os.path.join(ROOT, "raw", "shifts")
PBP_DIR    = os.path.join(ROOT, "raw", "pbp")
GAME_IDS   = os.path.join(PROJECT, "Data", "game_ids.csv")
OUT_DIR    = ROOT
os.makedirs(OUT_DIR, exist_ok=True)

SEASONS_POOL       = {"20222023", "20232024", "20242025", "20252026"}
CURRENT_SEASON     = "20252026"
SITCODE_5V5        = "1551"
BLUE_LINE_X        = 25
FACEOFF_MATCH_SEC  = 2   # a shift is "faceoff-started" if the faceoff is within +/- this of shift start

SCENARIOS = ("pooled", "current_regular", "current_playoffs")

# --------------------------------------------------------------------------
def hms(s): mm, ss = s.split(":"); return int(mm)*60 + int(ss)
def abs_t(period, t): return (period-1)*1200 + t

# --------------------------------------------------------------------------
def build_goalies_and_shifts(shift_json, pbp):
    home_id = pbp["homeTeam"]["id"]
    away_id = pbp["awayTeam"]["id"]
    pbp_goalies = {p["playerId"] for p in pbp.get("rosterSpots", []) if p["positionCode"] == "G"}
    shifts_raw = [s for s in shift_json["data"] if s.get("typeCode") == 517 and s.get("duration")]
    max_dur = defaultdict(int)
    for s in shifts_raw:
        d = hms(s["duration"])
        if d > max_dur[s["playerId"]]: max_dur[s["playerId"]] = d
    goalies = pbp_goalies | {pid for pid, d in max_dur.items() if d > 300}
    shifts = []
    for s in shifts_raw:
        tid = s["teamId"]
        if tid == home_id: side = "H"
        elif tid == away_id: side = "A"
        else: continue
        pid = s["playerId"]
        start = abs_t(s["period"], hms(s["startTime"]))
        end   = abs_t(s["period"], hms(s["endTime"]))
        if end <= start: continue
        shifts.append({"pid": pid, "side": side, "start": start, "end": end, "goalie": pid in goalies})
    return shifts, goalies, home_id, away_id

def build_intervals_from_shifts(shifts):
    """Return a list of 5v5 (both teams 5 non-goalie skaters) intervals as
    (start, end, home_on_ice, away_on_ice)."""
    events = []
    for s in shifts:
        if s["goalie"]: continue
        events.append((s["start"], +1, s["pid"], s["side"]))
        events.append((s["end"],   -1, s["pid"], s["side"]))
    events.sort(key=lambda e: (e[0], e[1]))
    intervals = []
    on_H, on_A = set(), set()
    if not events: return intervals
    prev_t = events[0][0]
    i, n = 0, len(events)
    while i < n:
        t = events[i][0]
        if t > prev_t:
            if len(on_H) == 5 and len(on_A) == 5:
                intervals.append((prev_t, t, frozenset(on_H), frozenset(on_A)))
            prev_t = t
        while i < n and events[i][0] == t:
            _, delta, pid, side = events[i]
            target = on_H if side == "H" else on_A
            if delta == +1: target.add(pid)
            else: target.discard(pid)
            i += 1
    return intervals

def build_timelines(pbp):
    """Return four sorted parallel lists and the faceoff event list.
       zone_times, zone_vals   : puck zone from HOME perspective (last-known event loc)
       sit_times,  sit_vals    : situationCode at each moment
       faceoffs                : list of (t_abs, zone_home, situation)
    """
    zt, zv, st, sv, fos = [], [], [], [], []
    for p in pbp.get("plays", []):
        if "timeInPeriod" not in p: continue
        period = p["periodDescriptor"]["number"]
        t_abs = abs_t(period, hms(p["timeInPeriod"]))
        if p.get("situationCode"):
            st.append(t_abs); sv.append(p["situationCode"])
        d = p.get("details") or {}
        x = d.get("xCoord")
        h_def = p.get("homeTeamDefendingSide")
        if x is not None and h_def is not None:
            x_h = -x if h_def == "right" else x
            if   x_h >  BLUE_LINE_X: zone = "OZ_home"
            elif x_h < -BLUE_LINE_X: zone = "DZ_home"
            else: zone = "NZ"
            zt.append(t_abs); zv.append(zone)
            if p["typeDescKey"] == "faceoff":
                fos.append((t_abs, zone, p.get("situationCode", "")))
    def _sort(ts, vs):
        if not ts: return [], []
        order = sorted(range(len(ts)), key=lambda i: ts[i])
        return [ts[i] for i in order], [vs[i] for i in order]
    zt, zv = _sort(zt, zv)
    st, sv = _sort(st, sv)
    fos.sort()
    return zt, zv, st, sv, fos

def zone_from_player(zone_home, side):
    """Translate home-perspective zone to the player's team perspective."""
    if side == "H":
        return {"OZ_home": "OZ", "DZ_home": "DZ"}.get(zone_home, "NZ")
    return {"OZ_home": "DZ", "DZ_home": "OZ"}.get(zone_home, "NZ")

# --------------------------------------------------------------------------
def blank():
    return {
        "toi_sec": 0, "games_played": 0,
        "team_id": None, "team_abbrev": None,
        "oz_fo_shifts": 0, "dz_fo_shifts": 0, "nz_fo_shifts": 0,
        "oz_fo_5v5_sec": 0, "oz_fo_oz_sec": 0, "oz_fo_dz_sec": 0, "oz_fo_nz_sec": 0,
        "dz_fo_5v5_sec": 0, "dz_fo_oz_sec": 0, "dz_fo_dz_sec": 0, "dz_fo_nz_sec": 0,
        "nz_fo_5v5_sec": 0, "nz_fo_oz_sec": 0, "nz_fo_dz_sec": 0, "nz_fo_nz_sec": 0,
    }

def main():
    with open(GAME_IDS) as f:
        all_games = [r for r in csv.DictReader(f) if r["season"] in SEASONS_POOL]
    print(f"[zone] {len(all_games)} games in scope")

    player_meta = {}
    team_abbrev = {}

    zone = {sc: defaultdict(blank) for sc in SCENARIOS}
    overlap = {sc: {"teammate": defaultdict(int), "opponent": defaultdict(int)}
               for sc in SCENARIOS}

    t0 = time.time()
    for gi, g in enumerate(all_games, 1):
        gid = g["game_id"]
        season = g["season"]
        gtype = g["game_type"]
        spath = os.path.join(SHIFTS_DIR, f"{gid}.json")
        ppath = os.path.join(PBP_DIR, f"{gid}.json")
        if not (os.path.exists(spath) and os.path.exists(ppath)): continue
        try:
            pbp = json.load(open(ppath))
            sj  = json.load(open(spath))
        except Exception as e:
            print(f"  skip {gid}: {e}"); continue

        home_abbrev = pbp["homeTeam"]["abbrev"]; away_abbrev = pbp["awayTeam"]["abbrev"]
        team_abbrev[pbp["homeTeam"]["id"]] = home_abbrev
        team_abbrev[pbp["awayTeam"]["id"]] = away_abbrev

        # roster metadata
        for p in pbp.get("rosterSpots", []):
            pid = p["playerId"]
            m = player_meta.setdefault(pid, {
                "name": f"{p['firstName']['default']} {p['lastName']['default']}",
                "position": p["positionCode"],
                "team_id": p["teamId"],
                "team_abbrev": team_abbrev.get(p["teamId"], ""),
                "seasons": set(),
                "first_seen": g["game_date"],
                "last_seen":  g["game_date"],
            })
            m["seasons"].add(season)
            if g["game_date"] > m["last_seen"]:
                m["last_seen"] = g["game_date"]
                m["team_id"] = p["teamId"]
                m["team_abbrev"] = team_abbrev.get(p["teamId"], "")
            if g["game_date"] < m["first_seen"]:
                m["first_seen"] = g["game_date"]

        shifts, goalies, home_id, away_id = build_goalies_and_shifts(sj, pbp)
        intervals = build_intervals_from_shifts(shifts)
        if not intervals: continue
        zt, zv, st, sv, fos = build_timelines(pbp)
        faceoff_times = [f[0] for f in fos]

        sc_list = ["pooled"]
        if season == CURRENT_SEASON and gtype == "regular":  sc_list.append("current_regular")
        if season == CURRENT_SEASON and gtype == "playoff":  sc_list.append("current_playoffs")

        interval_starts = [iv[0] for iv in intervals]
        interval_ends   = [iv[1] for iv in intervals]

        # ---------- 5v5 TOI per player + pair overlap (same as before) ----------
        game_players_side = {}   # pid -> side ('H'|'A'), for team tagging at end
        all_break_times = sorted(set(zt + st))

        for s, e, hset, aset in intervals:
            dur_full = e - s
            # 5v5 TOI per player (count all 5v5 skater-time via on-ice sets)
            for pid in hset:
                for sc in sc_list:
                    zone[sc][pid]["toi_sec"] += dur_full
                game_players_side[pid] = "H"
            for pid in aset:
                for sc in sc_list:
                    zone[sc][pid]["toi_sec"] += dur_full
                game_players_side[pid] = "A"
            # overlap pairs
            hlist = sorted(hset); alist = sorted(aset)
            for p1, p2 in combinations(hlist, 2):
                key = (p1, p2)
                for sc in sc_list: overlap[sc]["teammate"][key] += dur_full
            for p1, p2 in combinations(alist, 2):
                key = (p1, p2)
                for sc in sc_list: overlap[sc]["teammate"][key] += dur_full
            for p1 in hlist:
                for p2 in alist:
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    for sc in sc_list: overlap[sc]["opponent"][key] += dur_full

        # games_played + team for this game, per scenario
        side_team = {"H": home_id, "A": away_id}
        for pid, side in game_players_side.items():
            for sc in sc_list:
                c = zone[sc][pid]
                c["games_played"] += 1
                c["team_id"] = side_team[side]
                c["team_abbrev"] = team_abbrev.get(side_team[side], "")

        # ---------- NEW: faceoff-start shift zone time per player ----------
        for sh in shifts:
            if sh["goalie"]: continue
            pid = sh["pid"]; side = sh["side"]
            shift_start = sh["start"]; shift_end = sh["end"]
            # Find a faceoff within +/- FACEOFF_MATCH_SEC of shift start
            lo = bisect_left(faceoff_times,  shift_start - FACEOFF_MATCH_SEC)
            hi = bisect_right(faceoff_times, shift_start + FACEOFF_MATCH_SEC)
            if lo >= hi:
                continue
            # pick closest
            best = min(range(lo, hi), key=lambda i: abs(faceoff_times[i] - shift_start))
            fo_t, fo_zone_home, fo_sit = fos[best]
            if fo_sit != SITCODE_5V5:
                continue
            # zone from player's perspective
            fo_zone_player = zone_from_player(fo_zone_home, side)  # 'OZ'|'DZ'|'NZ'

            # Walk the shift's 5v5 portion, sub-divided by zone changes
            # 5v5 portion = intersection of [shift_start, shift_end] with the union of intervals
            # We'll iterate intervals that overlap this shift and sub-divide them.
            # Find first interval whose end > shift_start
            a_idx = bisect_right(interval_ends, shift_start)
            # Find last interval whose start < shift_end
            b_idx = bisect_left(interval_starts, shift_end) - 1
            if a_idx > b_idx:
                continue
            # Sum seconds in each zone during 5v5 portion of shift
            oz_sec = dz_sec = nz_sec = 0
            for iv_idx in range(a_idx, b_idx + 1):
                iv_s, iv_e, hset, aset = intervals[iv_idx]
                # the player must actually be on ice in this interval
                if (side == "H" and pid not in hset) or (side == "A" and pid not in aset):
                    continue
                s_clip = max(iv_s, shift_start)
                e_clip = min(iv_e, shift_end)
                if e_clip <= s_clip: continue
                # Sub-divide by zone timeline within [s_clip, e_clip)
                # Find break points within the interval's zone timeline that fall inside [s_clip, e_clip)
                z_lo = bisect_right(zt, s_clip) - 1   # index of zone at or before s_clip
                # Walk forward through zone change points
                cursor = s_clip
                while cursor < e_clip:
                    cur_zone = zv[z_lo] if z_lo >= 0 else "NZ"
                    next_change = zt[z_lo + 1] if (z_lo + 1) < len(zt) else float("inf")
                    seg_end = min(next_change, e_clip)
                    seg_dur = seg_end - cursor
                    if seg_dur > 0:
                        player_zone = zone_from_player(cur_zone, side)
                        if   player_zone == "OZ": oz_sec += seg_dur
                        elif player_zone == "DZ": dz_sec += seg_dur
                        else: nz_sec += seg_dur
                    cursor = seg_end
                    z_lo += 1

            total_5v5 = oz_sec + dz_sec + nz_sec
            if total_5v5 <= 0:
                # Shift started with a faceoff but had no 5v5 time on ice (rare; skip)
                continue

            # accumulate into scenario buckets under the faceoff's player-perspective zone
            shift_prefix = {"OZ": "oz", "DZ": "dz", "NZ": "nz"}[fo_zone_player]
            for sc in sc_list:
                c = zone[sc][pid]
                c[f"{shift_prefix}_fo_shifts"]  += 1
                c[f"{shift_prefix}_fo_5v5_sec"] += total_5v5
                c[f"{shift_prefix}_fo_oz_sec"]  += oz_sec
                c[f"{shift_prefix}_fo_dz_sec"]  += dz_sec
                c[f"{shift_prefix}_fo_nz_sec"]  += nz_sec

        if gi % 500 == 0:
            print(f"  zone: {gi}/{len(all_games)} games  ({time.time()-t0:.1f}s)")

    # Write artefacts
    meta_out = {
        str(pid): {
            "name": m["name"], "position": m["position"],
            "team_id": m["team_id"], "team_abbrev": m["team_abbrev"],
            "seasons": sorted(m["seasons"]),
            "first_seen": m["first_seen"], "last_seen": m["last_seen"],
        }
        for pid, m in player_meta.items()
    }
    with open(os.path.join(OUT_DIR, "_player_meta.json"), "w") as f:
        json.dump(meta_out, f)
    print(f"[zone] wrote _player_meta.json ({len(meta_out)} players)")

    zone_out = {sc: {str(pid): c for pid, c in zone[sc].items()} for sc in SCENARIOS}
    with open(os.path.join(OUT_DIR, "_zone_time.json"), "w") as f:
        json.dump(zone_out, f)
    sz = os.path.getsize(os.path.join(OUT_DIR, "_zone_time.json")) / 1e6
    print(f"[zone] wrote _zone_time.json ({sz:.1f} MB)")

    with open(os.path.join(OUT_DIR, "_overlap.pkl"), "wb") as f:
        pickle.dump({sc: {k: dict(v) for k, v in overlap[sc].items()} for sc in SCENARIOS}, f)
    sz = os.path.getsize(os.path.join(OUT_DIR, "_overlap.pkl")) / 1e6
    print(f"[zone] wrote _overlap.pkl ({sz:.1f} MB)")
    for sc in SCENARIOS:
        t = len(overlap[sc]["teammate"]); o = len(overlap[sc]["opponent"])
        print(f"  {sc}: {t} teammate pairs, {o} opponent pairs")

if __name__ == "__main__":
    main()
