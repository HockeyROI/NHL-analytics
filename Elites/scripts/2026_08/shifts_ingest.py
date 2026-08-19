"""shifts_ingest.py -- Step 0 of the Elites pipeline.

Ingest NHL shift-chart data (already fetched into committed per-season parquets
at Data/shift_data_by_season/) and distil it into every EVEN-STRENGTH on-ice
interval across all six seasons in the database.

Even strength here == strict 5v5: BOTH teams have exactly 5 skaters on the ice
AND BOTH teams have a goalie on the ice. Requiring a goalie on each side is what
excludes empty-net situations (a pulled goalie ends that goalie's shift, so the
interval no longer has a goalie for that team and is dropped). All special-teams
strengths (4v5, 5v4, 4v4, 3v3 OT, ...) are dropped because a side != 5 skaters.

Scope: REGULAR season only (game_type == 'regular' in Data/game_ids.csv), matching
the rest of the project's convention that playoffs live in separate *_playoffs
artifacts. Goalie vs skater is resolved from NFI/output/player_positions.csv.

Method -- per game we segment the timeline at every shift start/end boundary.
A shift occupies the half-open interval (start, end] (the project-wide convention,
see fa_linemate_without_me.py: exclusive start = the incoming line change is not
yet "on" at the instant it starts; inclusive end = a shift-ending timestamp lands
in the same second as the stoppage/goal that ended it). So between two adjacent
boundaries (a, b] a shift is active iff start <= a and end >= b.

Output: Elites/Output/ev_shifts.csv with one row per 5v5 interval:
  game_id, season, period, start_time, end_time, duration,
  home_team, away_team, home_skaters, away_skaters
(start_time/end_time are ABSOLUTE game seconds; skater columns are ';'-joined
player_ids of the FIVE skaters per side -- goalies excluded.)

Prints total EV (5v5) minutes per season for sanity-checking against known
league totals (~44 min of 5v5 per game).
"""
from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))          # .../Elites/scripts/2026_08
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))
DATA = os.path.join(PROJECT, "Data")
SHIFT_DIR = os.path.join(DATA, "shift_data_by_season")
GAME_IDS = os.path.join(DATA, "game_ids.csv")
POS_CSV = os.path.join(PROJECT, "NFI", "output", "player_positions.csv")
OUT = os.path.join(PROJECT, "Elites", "Output", "ev_shifts.csv")


def load_positions():
    pos = pd.read_csv(POS_CSV)
    return dict(zip(pos["player_id"].astype(int), pos["pos_group"].astype(str)))


def segment_game(shifts, is_goalie, side_has_goalie_data):
    """Yield 5v5 (start, end] intervals for one game.

    shifts: dict side -> list of (player_id, start, end), side in {'H','A'}.
    side_has_goalie_data: dict side -> bool. If a side has NO goalie shift rows
      in the whole game (a data-logging gap, seen in ~67 games of 2020-21), we
      cannot use goalie-on-ice to detect an empty net for that side, so we don't
      require one -- otherwise the entire game is wrongly dropped. When the side
      DOES have goalie data we require a goalie on ice, which is what excludes
      genuine empty-net time (a real empty net is almost always 6v5/5v6 and is
      already excluded by the 5-skater count anyway).
    Yields (start, end, home_sk_ids, away_sk_ids).
    """
    starts = []
    ends = []
    for rows in shifts.values():
        for _pid, s, e in rows:
            starts.append(s)
            ends.append(e)
    bounds = np.array(sorted(set(starts) | set(ends)))
    for k in range(len(bounds) - 1):
        a = int(bounds[k])
        b = int(bounds[k + 1])
        if b <= a:
            continue
        onice = {"H": set(), "A": set()}    # DISTINCT skater ids
        goalies = {"H": set(), "A": set()}  # DISTINCT goalie ids
        for side, rows in shifts.items():
            for pid, s, e in rows:
                if s <= a and e >= b:        # active during (a, b]
                    # dedup by pid: a real shift + an overlapping zero-length
                    # marker row can both match, which would phantom-inflate the
                    # on-ice count (same guard as fa_linemate_without_me.py).
                    if is_goalie(pid):
                        goalies[side].add(pid)
                    else:
                        onice[side].add(pid)
        goalie_ok_h = bool(goalies["H"]) or not side_has_goalie_data["H"]
        goalie_ok_a = bool(goalies["A"]) or not side_has_goalie_data["A"]
        if (len(onice["H"]) == 5 and len(onice["A"]) == 5
                and goalie_ok_h and goalie_ok_a):
            yield a, b, sorted(onice["H"]), sorted(onice["A"])


def main():
    pos_map = load_positions()

    def is_goalie(pid):
        return pos_map.get(int(pid)) == "G"

    gi = pd.read_csv(GAME_IDS)
    gi["game_id"] = gi["game_id"].astype(int)
    gi = gi[gi["game_type"] == "regular"]
    g2home = dict(zip(gi["game_id"], gi["home_abbrev"]))
    g2away = dict(zip(gi["game_id"], gi["away_abbrev"]))
    g2season = dict(zip(gi["game_id"], gi["season"].astype(str)))
    reg_games = set(gi["game_id"])

    out_rows = []
    ev_secs_by_season = {}
    unmapped_goalie_warn = 0

    for pq in sorted(glob.glob(os.path.join(SHIFT_DIR, "*.parquet"))):
        season = os.path.basename(pq)[:-8]
        sd = pd.read_parquet(pq)
        sd["game_id"] = sd["game_id"].astype(int)
        sd = sd[sd["game_id"].isin(reg_games)]
        sd = sd.dropna(subset=["player_id", "abs_start_secs", "abs_end_secs"])
        sd["player_id"] = sd["player_id"].astype(int)
        sd["abs_start_secs"] = sd["abs_start_secs"].astype(int)
        sd["abs_end_secs"] = sd["abs_end_secs"].astype(int)
        season_secs = 0

        for gid, g in sd.groupby("game_id", sort=False):
            home_ab = g2home.get(gid)
            away_ab = g2away.get(gid)
            if home_ab is None:
                continue
            shifts = {"H": [], "A": []}
            for pid, ab, s, e in zip(g["player_id"], g["team_abbrev"],
                                     g["abs_start_secs"], g["abs_end_secs"]):
                if e <= s:
                    continue
                if ab == home_ab:
                    shifts["H"].append((pid, s, e))
                elif ab == away_ab:
                    shifts["A"].append((pid, s, e))
                # else: abbrev not matching either side -> skip (should not happen)
            if not shifts["H"] or not shifts["A"]:
                continue
            side_has_goalie_data = {
                side: any(is_goalie(pid) for pid, _s, _e in rows)
                for side, rows in shifts.items()
            }
            for a, b, hids, aids in segment_game(shifts, is_goalie,
                                                 side_has_goalie_data):
                dur = b - a
                season_secs += dur
                out_rows.append({
                    "game_id": gid,
                    "season": g2season.get(gid, season),
                    "period": min(a // 1200 + 1, 5),
                    "start_time": a,
                    "end_time": b,
                    "duration": dur,
                    "home_team": home_ab,
                    "away_team": away_ab,
                    "home_skaters": ";".join(str(x) for x in sorted(hids)),
                    "away_skaters": ";".join(str(x) for x in sorted(aids)),
                })
        ev_secs_by_season[season] = season_secs
        print(f"  [{season}] {sd['game_id'].nunique()} games -> "
              f"{season_secs / 60:,.0f} EV(5v5) minutes", flush=True)

    df = pd.DataFrame(out_rows)
    df.to_csv(OUT, index=False)
    print("\n" + "=" * 60)
    print("TOTAL EV (5v5) MINUTES PER SEASON")
    print("=" * 60)
    for s in sorted(ev_secs_by_season):
        mins = ev_secs_by_season[s] / 60
        ng = df[df["season"] == s]["game_id"].nunique() if len(df) else 0
        per_g = mins / ng if ng else 0
        print(f"  {s}: {mins:>10,.0f} min   ({ng} games, {per_g:.1f} min/game)")
    print(f"\n[done] {len(df):,} EV intervals -> {OUT}")


if __name__ == "__main__":
    main()
