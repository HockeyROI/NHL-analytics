"""compute_zone_start_per_season.py
------------------------------------
D/N/O Start% (faceoff-started 5v5 shifts), broken out PER SEASON instead of
pooled across all 4 years. This is a presentation-layer stat, not a rating —
"who starts where" — so unlike compute_pqr_roc_rol.py's OUER/DUER/PAR family
it carries NO minimum-sample floor. Every player-season with at least one
qualifying faceoff-shift gets a row; small samples are just noisy, not
excluded.

Classification logic (build_goalies_and_shifts, build_intervals_from_shifts,
build_timelines, zone_from_player, the faceoff-start-shift walk) is a
verbatim copy of compute_zone_and_overlap.py's validated version — copied,
not imported, so this script can key results by (player_id, season) without
touching or depending on that script's own pooled-scenario execution.

Reads:  Zones/raw/pbp/{gid}.json, Zones/raw/shifts/{gid}.json, Data/game_ids.csv
Writes: Zones/output/zone_start_per_season.csv
"""
from __future__ import annotations
import csv, json, os, time
from collections import defaultdict
from bisect import bisect_right, bisect_left
from itertools import combinations

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ZONES_DIR   = os.path.dirname(SCRIPTS_DIR)          # .../HockeyROI/Zones
PROJECT     = os.path.dirname(ZONES_DIR)            # .../HockeyROI
SHIFTS_DIR  = os.path.join(ZONES_DIR, "raw", "shifts")
PBP_DIR     = os.path.join(ZONES_DIR, "raw", "pbp")
GAME_IDS    = os.path.join(PROJECT, "Data", "game_ids.csv")
OUT_DIR     = os.path.join(ZONES_DIR, "output")

SEASONS_POOL       = {"20222023", "20232024", "20242025", "20252026"}
SITCODE_5V5        = "1551"
BLUE_LINE_X        = 25
FACEOFF_MATCH_SEC  = 2


def hms(s): mm, ss = s.split(":"); return int(mm) * 60 + int(ss)
def abs_t(period, t): return (period - 1) * 1200 + t


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
    if side == "H":
        return {"OZ_home": "OZ", "DZ_home": "DZ"}.get(zone_home, "NZ")
    return {"OZ_home": "DZ", "DZ_home": "OZ"}.get(zone_home, "NZ")


def blank():
    return {"games_played": 0, "team_id": None, "team_abbrev": None,
            "oz_fo_shifts": 0, "dz_fo_shifts": 0, "nz_fo_shifts": 0}


def main():
    with open(GAME_IDS) as f:
        all_games = [r for r in csv.DictReader(f) if r["season"] in SEASONS_POOL]
    print(f"[zone-per-season] {len(all_games)} games in scope")

    player_meta = {}
    team_abbrev = {}
    # keyed by (season, player_id) — the whole point of this script
    zone = defaultdict(blank)

    t0 = time.time()
    n_processed = 0
    n_missing = 0
    for gi, g in enumerate(all_games, 1):
        gid = g["game_id"]
        season = g["season"]
        spath = os.path.join(SHIFTS_DIR, f"{gid}.json")
        ppath = os.path.join(PBP_DIR, f"{gid}.json")
        if not (os.path.exists(spath) and os.path.exists(ppath)):
            n_missing += 1
            continue
        try:
            pbp = json.load(open(ppath))
            sj  = json.load(open(spath))
        except Exception as e:
            print(f"  skip {gid}: {e}"); continue

        home_abbrev = pbp["homeTeam"]["abbrev"]; away_abbrev = pbp["awayTeam"]["abbrev"]
        team_abbrev[pbp["homeTeam"]["id"]] = home_abbrev
        team_abbrev[pbp["awayTeam"]["id"]] = away_abbrev

        for p in pbp.get("rosterSpots", []):
            pid = p["playerId"]
            m = player_meta.setdefault(pid, {
                "name": f"{p['firstName']['default']} {p['lastName']['default']}",
                "position": p["positionCode"],
            })

        shifts, goalies, home_id, away_id = build_goalies_and_shifts(sj, pbp)
        intervals = build_intervals_from_shifts(shifts)
        if not intervals: continue
        zt, zv, st, sv, fos = build_timelines(pbp)
        faceoff_times = [f[0] for f in fos]

        interval_starts = [iv[0] for iv in intervals]
        interval_ends   = [iv[1] for iv in intervals]

        game_players_side = {}
        for s, e, hset, aset in intervals:
            for pid in hset: game_players_side[pid] = "H"
            for pid in aset: game_players_side[pid] = "A"
        side_team = {"H": home_id, "A": away_id}
        for pid, side in game_players_side.items():
            key = (season, pid)
            c = zone[key]
            c["games_played"] += 1
            c["team_id"] = side_team[side]
            c["team_abbrev"] = team_abbrev.get(side_team[side], "")

        for sh in shifts:
            if sh["goalie"]: continue
            pid = sh["pid"]; side = sh["side"]
            shift_start = sh["start"]; shift_end = sh["end"]
            lo = bisect_left(faceoff_times,  shift_start - FACEOFF_MATCH_SEC)
            hi = bisect_right(faceoff_times, shift_start + FACEOFF_MATCH_SEC)
            if lo >= hi: continue
            best = min(range(lo, hi), key=lambda i: abs(faceoff_times[i] - shift_start))
            fo_t, fo_zone_home, fo_sit = fos[best]
            if fo_sit != SITCODE_5V5: continue
            fo_zone_player = zone_from_player(fo_zone_home, side)

            a_idx = bisect_right(interval_ends, shift_start)
            b_idx = bisect_left(interval_starts, shift_end) - 1
            if a_idx > b_idx: continue
            total_5v5 = 0
            for iv_idx in range(a_idx, b_idx + 1):
                iv_s, iv_e, hset, aset = intervals[iv_idx]
                if (side == "H" and pid not in hset) or (side == "A" and pid not in aset):
                    continue
                s_clip = max(iv_s, shift_start)
                e_clip = min(iv_e, shift_end)
                if e_clip > s_clip:
                    total_5v5 += e_clip - s_clip
            if total_5v5 <= 0: continue

            shift_prefix = {"OZ": "oz", "DZ": "dz", "NZ": "nz"}[fo_zone_player]
            zone[(season, pid)][f"{shift_prefix}_fo_shifts"] += 1

        n_processed += 1
        if gi % 1000 == 0:
            print(f"  {gi}/{len(all_games)} games  ({time.time()-t0:.1f}s)")

    print(f"[zone-per-season] processed {n_processed} games, {n_missing} missing raw files")

    rows = []
    for (season, pid), c in zone.items():
        total = c["oz_fo_shifts"] + c["dz_fo_shifts"] + c["nz_fo_shifts"]
        m = player_meta.get(pid, {})
        rows.append({
            "player_id": pid,
            "player_name": m.get("name", ""),
            "position": m.get("position", ""),
            "team": c["team_abbrev"],
            "season": season,
            "games_played": c["games_played"],
            "oz_faceoff_shifts": c["oz_fo_shifts"],
            "dz_faceoff_shifts": c["dz_fo_shifts"],
            "nz_faceoff_shifts": c["nz_fo_shifts"],
            "OZ Start%": round(c["oz_fo_shifts"] / total * 100, 2) if total else None,
            "DZ Start%": round(c["dz_fo_shifts"] / total * 100, 2) if total else None,
            "NZ Start%": round(c["nz_fo_shifts"] / total * 100, 2) if total else None,
        })

    out_path = os.path.join(OUT_DIR, "zone_start_per_season.csv")
    fieldnames = ["player_id", "player_name", "position", "team", "season", "games_played",
                  "oz_faceoff_shifts", "dz_faceoff_shifts", "nz_faceoff_shifts",
                  "OZ Start%", "DZ Start%", "NZ Start%"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[zone-per-season] wrote {out_path} ({len(rows)} player-season rows)")


if __name__ == "__main__":
    main()
