"""Zone-tracking smoke test on ONE 2025-26 EDM game.

Goals:
  1. Prove we can build 5v5 on-ice intervals from the shift chart.
  2. Prove we can compute puck zone over time from PBP x/y + homeTeamDefendingSide
     and correctly orient OZ/DZ per team per period.
  3. Prove the 5v5 filter works via situationCode == '1551'.
  4. Print sample output for 5 players: OZ/DZ/NZ time % and oZS%.

Notes (approximation):
  Between PBP events the puck's zone is assumed to be the zone of the last
  event with coordinates (the "last known event location" method in the spec).
  This is an approximation - the puck can change zones between events.
"""
from __future__ import annotations
import json, os
from collections import defaultdict
from bisect import bisect_right

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIFT_PATH = os.path.join(ROOT, "raw", "shifts", "2025020006.json")
PBP_PATH   = os.path.join(ROOT, "raw", "pbp",    "2025020006.json")

BLUE_LINE_X = 25   # NHL blue lines at x = +/- 25 in the standard coord system
SITCODE_5V5 = "1551"

def hms(s): mm, ss = s.split(":"); return int(mm)*60 + int(ss)
def abs_t(period, t): return (period-1)*1200 + t

# ----------------------------------------------------------------------
# 1. build 5v5 intervals from shift chart (side A = home, side B = away by convention)
# ----------------------------------------------------------------------
def build_intervals(shift_json, pbp, skaters_only=True):
    home_id = pbp["homeTeam"]["id"]
    away_id = pbp["awayTeam"]["id"]
    pbp_goalies = {p["playerId"] for p in pbp["rosterSpots"] if p["positionCode"] == "G"}

    shifts = [s for s in shift_json["data"] if s.get("typeCode") == 517 and s.get("duration")]
    # Goalie safety net via max-shift heuristic + PBP
    max_dur = defaultdict(int)
    for s in shifts:
        d = hms(s["duration"])
        if d > max_dur[s["playerId"]]: max_dur[s["playerId"]] = d
    goalies = pbp_goalies | {pid for pid, d in max_dur.items() if d > 300}

    events = []
    for s in shifts:
        side = "H" if s["teamId"] == home_id else ("A" if s["teamId"] == away_id else None)
        if side is None: continue
        p = s["period"]
        st = abs_t(p, hms(s["startTime"]))
        en = abs_t(p, hms(s["endTime"]))
        if en <= st: continue
        events.append((st, +1, s["playerId"], side))
        events.append((en, -1, s["playerId"], side))
    events.sort(key=lambda e: (e[0], e[1]))

    intervals = []  # (start, end, home_on_ice, away_on_ice)
    on_H, on_A = set(), set()
    if not events: return intervals, goalies, home_id, away_id

    prev_t = events[0][0]
    i, n = 0, len(events)
    while i < n:
        t = events[i][0]
        if t > prev_t:
            sH = on_H - goalies
            sA = on_A - goalies
            if len(sH) == 5 and len(sA) == 5:
                intervals.append((prev_t, t, frozenset(sH), frozenset(sA)))
            prev_t = t
        while i < n and events[i][0] == t:
            _, delta, pid, side = events[i]
            target = on_H if side == "H" else on_A
            if delta == +1: target.add(pid)
            else: target.discard(pid)
            i += 1
    return intervals, goalies, home_id, away_id

# ----------------------------------------------------------------------
# 2. build puck-zone timeline (from home perspective)
#    zone: 'OZ_home' -> puck in home's offensive zone
#          'DZ_home' -> home's defensive zone
#          'NZ'      -> neutral
# ----------------------------------------------------------------------
def build_puck_zone_timeline(pbp):
    """Return (list of (t_abs, zone_home), list of (t_abs, situationCode))."""
    zone_tl = []
    sit_tl  = []
    for p in pbp["plays"]:
        pd = p["periodDescriptor"]
        period = pd["number"]
        if "timeInPeriod" not in p:
            continue
        t = hms(p["timeInPeriod"])
        t_abs = abs_t(period, t)
        if p.get("situationCode"):
            sit_tl.append((t_abs, p["situationCode"]))
        d = p.get("details") or {}
        x = d.get("xCoord")
        if x is None:
            continue
        home_def_side = p.get("homeTeamDefendingSide")
        if home_def_side is None:
            continue
        # Normalize so home always defends left (x < 0): flip x if defending right.
        x_h = -x if home_def_side == "right" else x
        if x_h >  BLUE_LINE_X: zone = "OZ_home"
        elif x_h < -BLUE_LINE_X: zone = "DZ_home"
        else: zone = "NZ"
        zone_tl.append((t_abs, zone))
    zone_tl.sort(key=lambda r: r[0])
    sit_tl.sort(key=lambda r: r[0])
    return zone_tl, sit_tl

# ----------------------------------------------------------------------
# 3. walk intervals and attribute zone time per player at 5v5
# ----------------------------------------------------------------------
def zone_time_per_player(intervals, zone_tl, sit_tl):
    """Return {pid: {'OZ':s, 'DZ':s, 'NZ':s}} where OZ/DZ are FROM that player's team perspective."""
    # Need to know each player's team. Build from intervals.
    # Also need situationCode per time for 5v5 filtering.
    z_times = [t for t, _ in zone_tl]
    z_vals  = [z for _, z in zone_tl]
    s_times = [t for t, _ in sit_tl]
    s_vals  = [s for _, s in sit_tl]

    def zone_at(t):
        i = bisect_right(z_times, t) - 1
        return z_vals[i] if i >= 0 else "NZ"
    def sit_at(t):
        i = bisect_right(s_times, t) - 1
        return s_vals[i] if i >= 0 else None

    counters = defaultdict(lambda: {"OZ": 0, "DZ": 0, "NZ": 0})
    # Remember each player's side (home vs away) — same across the game.
    side_of = {}

    # Sub-divide each interval by zone-timeline change points and situationCode change points
    all_breaks = sorted(set(z_times + s_times))
    for s, e, home_set, away_set in intervals:
        # collect break points inside [s, e)
        # find slice of all_breaks within (s, e)
        lo = bisect_right(all_breaks, s)
        hi = bisect_right(all_breaks, e - 1)  # last index whose time < e
        pts = [s] + all_breaks[lo:hi+1] + [e]
        pts = [p for p in pts if s <= p <= e]
        pts = sorted(set(pts))
        # iterate sub-intervals
        for a, b in zip(pts, pts[1:]):
            if b <= a: continue
            # require 5v5
            if sit_at(a) != SITCODE_5V5:
                continue
            zhome = zone_at(a)
            dur = b - a
            for pid in home_set:
                side_of[pid] = "H"
                if zhome == "OZ_home":
                    counters[pid]["OZ"] += dur
                elif zhome == "DZ_home":
                    counters[pid]["DZ"] += dur
                else:
                    counters[pid]["NZ"] += dur
            for pid in away_set:
                side_of[pid] = "A"
                # For away players OZ/DZ are flipped
                if zhome == "OZ_home":
                    counters[pid]["DZ"] += dur
                elif zhome == "DZ_home":
                    counters[pid]["OZ"] += dur
                else:
                    counters[pid]["NZ"] += dur
    return counters, side_of

# ----------------------------------------------------------------------
# 4. oZS% from faceoffs that start a shift (within 2 sec of shift start)
# ----------------------------------------------------------------------
def ozs_per_player(pbp, intervals, side_home_id):
    """Return {pid: {'oz_starts':n, 'dz_starts':n}} for 5v5 faceoff shift starts."""
    # Collect faceoffs: (t_abs, zone_home, situationCode)
    fos = []
    for p in pbp["plays"]:
        if p["typeDescKey"] != "faceoff": continue
        t_abs = abs_t(p["periodDescriptor"]["number"], hms(p["timeInPeriod"]))
        if p.get("situationCode") != SITCODE_5V5: continue
        d = p.get("details") or {}
        x = d.get("xCoord")
        home_def_side = p.get("homeTeamDefendingSide")
        if x is None or home_def_side is None:
            continue
        x_h = -x if home_def_side == "right" else x
        if   x_h >  BLUE_LINE_X: zone_home = "OZ_home"
        elif x_h < -BLUE_LINE_X: zone_home = "DZ_home"
        else: zone_home = "NZ"
        fos.append((t_abs, zone_home))

    # For each faceoff, find players on the ice at that moment (use intervals)
    starts = [iv[0] for iv in intervals]
    counters = defaultdict(lambda: {"oz_starts": 0, "dz_starts": 0})
    for t_abs, zone_home in fos:
        if zone_home == "NZ":
            continue
        i = bisect_right(starts, t_abs) - 1
        if i < 0 or i >= len(intervals): continue
        s, e, home_set, away_set = intervals[i]
        if not (s <= t_abs < e): continue
        # For each player: OZ start if zone == their OZ
        for pid in home_set:
            if zone_home == "OZ_home": counters[pid]["oz_starts"] += 1
            elif zone_home == "DZ_home": counters[pid]["dz_starts"] += 1
        for pid in away_set:
            if zone_home == "OZ_home": counters[pid]["dz_starts"] += 1
            elif zone_home == "DZ_home": counters[pid]["oz_starts"] += 1
    return counters

# ----------------------------------------------------------------------
# orchestrate
# ----------------------------------------------------------------------
def main():
    pbp = json.load(open(PBP_PATH))
    shift_json = json.load(open(SHIFT_PATH))
    home_abbr = pbp["homeTeam"]["abbrev"]
    away_abbr = pbp["awayTeam"]["abbrev"]
    print(f"Game: {away_abbr} @ {home_abbr} ({pbp['gameDate']})")

    intervals, goalies, home_id, away_id = build_intervals(shift_json, pbp)
    total_5v5 = sum(e - s for s, e, _, _ in intervals)
    print(f"5v5 intervals (skater-count based): {len(intervals)}  total time: {total_5v5}s ({total_5v5/60:.1f} min)")

    zone_tl, sit_tl = build_puck_zone_timeline(pbp)
    print(f"Zone change points: {len(zone_tl)}  Situation change points: {len(sit_tl)}")

    counters, side_of = zone_time_per_player(intervals, zone_tl, sit_tl)
    ozs = ozs_per_player(pbp, intervals, home_id)

    names = {p["playerId"]: f"{p['firstName']['default']} {p['lastName']['default']}"
             for p in pbp["rosterSpots"]}
    team_of = {p["playerId"]: p["teamId"] for p in pbp["rosterSpots"]}

    # Orientation sanity check: after normalisation, HOME OZ should be on the
    # side opposite to where home defends in period 1. For game 2025020006 home
    # defends left in P1, so HOME OZ is x>25. Sum of HOME OZ seconds should be
    # > 0 and should roughly equal sum of AWAY DZ seconds (same puck).
    sum_home_oz = sum(counters[p]["OZ"] for p in counters if team_of.get(p) == home_id)
    sum_away_dz = sum(counters[p]["DZ"] for p in counters if team_of.get(p) == away_id)
    print(f"Sanity: home-OZ skater-seconds = {sum_home_oz}  away-DZ skater-seconds = {sum_away_dz}  (should be equal)")

    # Pick 5 home (EDM) skaters sorted by 5v5 TOI descending and print
    edm_players = [pid for pid, t in team_of.items() if t == home_id and pid not in goalies]
    edm_players = [p for p in edm_players if sum(counters[p].values()) > 0]
    edm_players.sort(key=lambda p: -sum(counters[p].values()))
    print(f"\n{'Name':<24}{'TOI5v5':>8}{'OZ%':>7}{'DZ%':>7}{'NZ%':>7}{'oZS':>5}{'dZS':>5}{'oZS%':>7}")
    for pid in edm_players[:5]:
        c = counters[pid]
        tot = c["OZ"] + c["DZ"] + c["NZ"]
        ozp = c["OZ"]/tot*100 if tot else 0
        dzp = c["DZ"]/tot*100 if tot else 0
        nzp = c["NZ"]/tot*100 if tot else 0
        oz = ozs[pid]["oz_starts"]; dz = ozs[pid]["dz_starts"]
        ozsp = oz/(oz+dz)*100 if oz+dz>0 else 0.0
        print(f"{names.get(pid,str(pid))[:23]:<24}{tot:>7}s{ozp:>6.1f}%{dzp:>6.1f}%{nzp:>6.1f}%{oz:>5}{dz:>5}{ozsp:>6.1f}%")

if __name__ == "__main__":
    main()
