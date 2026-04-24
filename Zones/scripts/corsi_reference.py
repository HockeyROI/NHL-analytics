"""corsi_reference.py
----------------------
PRESERVED reference computation of 5v5 Corsi (CF, CA, CF%) for every player,
now pooled across the four target seasons: 2022-23, 2023-24, 2024-25, 2025-26,
regular season AND playoffs combined.

Corsi is stored here only as a reference column joined into the new PQR /
ROC / ROL pipeline - it is NOT an input to PQR scoring.

Source data (on disk, do NOT delete):
  Data/qoc_qol/shifts/{gameId}.json  - shift charts for every game in scope
  Data/qoc_qol/pbp/{gameId}.json     - play-by-play for every game in scope
                                        (used to identify goalies precisely)
  Data/nhl_shot_events.csv           - pre-extracted shot events table

5v5 detection:
  - Identify goalies per game using PBP rosterSpots (positionCode == 'G')
    plus a max-shift-duration > 300s safety net.
  - An interval is 5v5 iff each team has exactly 5 non-goalie skaters on ice.

CF / CA attribution:
  - For each 5v5 shot event (shot-on-goal, goal, missed-shot, blocked-shot)
    with situation_code == '1551', look up which players were on the ice at
    that moment and credit CF or CA based on shooting team.

Output: Player_Ranking/corsi_reference.csv
  columns: player_id, games_played, toi_5v5_sec, cf_5v5, ca_5v5, cfpct_5v5
"""
from __future__ import annotations
import csv, json, os
from collections import defaultdict
from bisect import bisect_right

ROOT    = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
SHIFTS_DIR = os.path.join(ROOT, "raw", "shifts")
PBP_DIR    = os.path.join(ROOT, "raw", "pbp")
GAME_IDS   = os.path.join(PROJECT, "Data", "game_ids.csv")
SHOTS_CSV  = os.path.join(PROJECT, "Data", "nhl_shot_events.csv")
OUT_DIR    = ROOT
OUT_CSV    = os.path.join(OUT_DIR, "corsi_reference.csv")

SEASONS     = {"20222023", "20232024", "20242025", "20252026"}
GAME_TYPES  = {"regular", "playoff"}
SITCODE_5V5 = "1551"
SHOT_EVENTS = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}

def hms(s): mm, ss = s.split(":"); return int(mm)*60 + int(ss)
def abs_t(period, t): return (period-1)*1200 + t

def build_intervals(shift_json, pbp_goalies=None):
    shifts = [s for s in shift_json["data"] if s.get("typeCode") == 517 and s.get("duration")]
    max_dur = defaultdict(int)
    for s in shifts:
        d = hms(s["duration"])
        if d > max_dur[s["playerId"]]:
            max_dur[s["playerId"]] = d
    goalies = {pid for pid, d in max_dur.items() if d > 300}
    if pbp_goalies:
        goalies |= pbp_goalies
    team_ids = []
    for s in shifts:
        if s["teamId"] not in team_ids:
            team_ids.append(s["teamId"])
        if len(team_ids) == 2: break
    if len(team_ids) != 2:
        return [], {}, {}
    side_of = {team_ids[0]: "A", team_ids[1]: "B"}
    events = []
    for s in shifts:
        if s["teamId"] not in side_of: continue
        st = abs_t(s["period"], hms(s["startTime"]))
        en = abs_t(s["period"], hms(s["endTime"]))
        if en <= st: continue
        events.append((st, +1, s["playerId"], side_of[s["teamId"]]))
        events.append((en, -1, s["playerId"], side_of[s["teamId"]]))
    events.sort(key=lambda e: (e[0], e[1]))
    on_A, on_B = set(), set()
    intervals = []
    player_toi = defaultdict(int)
    if not events:
        return intervals, player_toi, {"A": team_ids[0], "B": team_ids[1]}
    prev_t = events[0][0]
    i, n = 0, len(events)
    while i < n:
        t = events[i][0]
        if t > prev_t:
            sA = on_A - goalies
            sB = on_B - goalies
            if len(sA) == 5 and len(sB) == 5:
                dur = t - prev_t
                intervals.append((prev_t, t, frozenset(sA), frozenset(sB)))
                for pid in sA: player_toi[pid] += dur
                for pid in sB: player_toi[pid] += dur
            prev_t = t
        while i < n and events[i][0] == t:
            _, delta, pid, side = events[i]
            target = on_A if side == "A" else on_B
            if delta == +1: target.add(pid)
            else: target.discard(pid)
            i += 1
    return intervals, player_toi, {"A": team_ids[0], "B": team_ids[1]}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GAME_IDS) as f:
        games = [r for r in csv.DictReader(f)
                 if r["season"] in SEASONS and r["game_type"] in GAME_TYPES]
    print(f"[corsi] {len(games)} games in scope")

    shots_by_game = defaultdict(list)
    with open(SHOTS_CSV) as f:
        for row in csv.DictReader(f):
            if (row["season"] in SEASONS and row["game_type"] in GAME_TYPES
                and row["situation_code"] == SITCODE_5V5
                and row["event_type"] in SHOT_EVENTS):
                t_abs = abs_t(int(row["period"]), int(row["time_secs"]))
                shots_by_game[row["game_id"]].append((t_abs, int(row["shooting_team_id"])))
    print(f"[corsi] total 5v5 shot events: {sum(len(v) for v in shots_by_game.values())}")

    season_toi = defaultdict(int)
    season_gp  = defaultdict(int)
    season_cf  = defaultdict(int)
    season_ca  = defaultdict(int)

    for gi, g in enumerate(games, 1):
        gid = g["game_id"]
        spath = os.path.join(SHIFTS_DIR, f"{gid}.json")
        if not os.path.exists(spath): continue
        shift_json = json.load(open(spath))
        pbp_goalies = None
        ppath = os.path.join(PBP_DIR, f"{gid}.json")
        if os.path.exists(ppath):
            pbp = json.load(open(ppath))
            pbp_goalies = {p["playerId"] for p in pbp.get("rosterSpots", [])
                           if p["positionCode"] == "G"}
        intervals, p_toi, team_of_side = build_intervals(shift_json, pbp_goalies)
        for pid, toi in p_toi.items():
            season_toi[pid] += toi
            season_gp[pid] += 1
        starts = [iv[0] for iv in intervals]
        for t_abs, shooting_team_id in shots_by_game.get(gid, []):
            i = bisect_right(starts, t_abs) - 1
            if 0 <= i < len(intervals):
                s, e, on_A, on_B = intervals[i]
                if s <= t_abs < e:
                    tA, tB = team_of_side["A"], team_of_side["B"]
                    for pid in on_A:
                        (season_cf if tA == shooting_team_id else season_ca)[pid] += 1
                    for pid in on_B:
                        (season_cf if tB == shooting_team_id else season_ca)[pid] += 1
        if gi % 500 == 0:
            print(f"  corsi: processed {gi}/{len(games)}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "games_played", "toi_5v5_sec", "cf_5v5", "ca_5v5", "cfpct_5v5"])
        all_pids = set(season_toi) | set(season_cf) | set(season_ca)
        for pid in sorted(all_pids):
            cf, ca = season_cf[pid], season_ca[pid]
            cfp = cf / (cf + ca) if cf + ca > 0 else ""
            w.writerow([pid, season_gp[pid], season_toi[pid], cf, ca,
                        round(cfp, 5) if isinstance(cfp, float) else ""])
    print(f"[corsi_reference] wrote {OUT_CSV} ({len(set(season_toi)|set(season_cf)|set(season_ca))} players)")

if __name__ == "__main__":
    main()
