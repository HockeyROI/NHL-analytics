#!/usr/bin/env python3
"""
Per-player, per-situation TOI — the all-situations foundation file.

Generalizes NFI/NFI_Score_Adj/2026/05/scripts/02_per_state_toi.py:
  - universe: ALL skaters (that script did 379 forwards only)
  - situations: full skater matchups (5v5/5v4/4v5/5v3/3v5/4v4/4v3/3v4/3v3
    + empty-net 6v* / *v6, else 'other'), NOT just collapsed ES/PP/PK
  - all periods INCLUDING overtime (that script capped at regulation 3600s;
    3v3 only exists in regular-season OT, so OT must be kept), and playoff OT
  - grouped per (player_id, season, game_type) so per-season per-60 rates work

Method (unchanged core): per game, walk events in (abs_time, event_id) order to
build piecewise-constant segments carrying (home_skaters, away_skaters) from
situation_code. For each shift, intersect with segments, label the situation
from THAT player's team's perspective (own vs opponent skater count), and
accumulate seconds. Reconciles: per-player sum over situations == total on-ice.

Output (long): player_id, player_name, position, season, game_type, situation,
               toi_min, gp
  gp = distinct games the player had a shift in, for that season+game_type
       (repeated on each situation row for convenience).

Inputs:
  Data/nhl_shot_events.csv                    (events + situation_code)
  NFI/Geometry_post/Data/shift_data.csv       (shift intervals, abs seconds)
  Data/game_ids.csv                           (game_id -> season, game_type)
  NFI/Output/player_positions.csv             (player_id -> name, position)
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import _data_sources as _ds

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))
SHOT_CSV = ROOT / "Data" / "nhl_shot_events.csv"
SHIFT_CSV = ROOT / "NFI" / "Geometry_post" / "Data" / "shift_data.csv"
GAMES = ROOT / "Data" / "game_ids.csv"
POSITIONS = ROOT / "NFI" / "Output" / "player_positions.csv"

OUT_CSV = Path(os.environ.get("SITUATION_TOI_OUT", ROOT / "Data" / "player_situation_toi.csv"))

REQUIRED_SHIFT_COLS = ["game_id", "player_id", "period", "team_abbrev",
                       "abs_start_secs", "abs_end_secs"]

# Keep skater counts in {3,4,5,6}; anything else (penalty shot 1v0, shootout,
# data glitches) buckets to 'other' so seconds are never dropped.
_KEEP = {3, 4, 5, 6}


def situation_label(own: int, opp: int) -> str:
    if own in _KEEP and opp in _KEEP:
        return f"{own}v{opp}"
    return "other"


def stop(msg: str, code: int = 2) -> int:
    print(f"[situation_toi] STOP — {msg}", file=sys.stderr)
    return code


def main() -> int:
    for p in (GAMES, POSITIONS):
        if not p.exists():
            return stop(f"input not found: {p}")

    # ---------- player identity (names, positions; drop goalies) ----------
    pos = pd.read_csv(POSITIONS)
    name_map = dict(zip(pos["player_id"], pos["player_name"]))
    posn_map = dict(zip(pos["player_id"], pos["position"]))
    goalie_ids = set(pos.loc[pos["position"] == "G", "player_id"].astype(int))
    print(f"[situation_toi] known players: {len(name_map):,}  (goalies excluded: {len(goalie_ids):,})")

    # ---------- games -> season / game_type ----------
    g = pd.read_csv(GAMES, dtype={"game_id": int, "season": str})
    g_season = dict(zip(g["game_id"], g["season"]))
    g_type = dict(zip(g["game_id"], g["game_type"]))
    print(f"[situation_toi] games: {len(g):,}  "
          f"({(g['game_type'] == 'regular').sum():,} regular, "
          f"{(g['game_type'] == 'playoff').sum():,} playoff)")

    # ---------- events: per-game timeline with skater counts ----------
    print(f"[situation_toi] reading events {SHOT_CSV} ...")
    use_shots = ["game_id", "period", "time_secs", "event_id", "is_goal",
                 "situation_code", "home_team_abbrev", "away_team_abbrev"]
    shots = _ds.load_shot_events(usecols=use_shots,
                                 dtype={"situation_code": str})
    shots = shots.dropna(subset=["situation_code", "period", "time_secs"])
    shots["abs_time"] = (shots["time_secs"].astype(int)
                         + (shots["period"].astype(int) - 1) * 1200)
    sc = shots["situation_code"].astype(str).str.zfill(4)
    shots["away_sk"] = sc.str[1].astype(int)   # situation_code = [ag, ask, hsk, hg]
    shots["home_sk"] = sc.str[2].astype(int)
    shots = shots.sort_values(["game_id", "abs_time", "event_id"]).reset_index(drop=True)
    print(f"[situation_toi]   {len(shots):,} events across {shots['game_id'].nunique():,} games")

    shots_by_game = dict(tuple(shots.groupby("game_id")))
    team_abbrevs = (shots.groupby("game_id")
                    .agg(home_ab=("home_team_abbrev", "first"),
                         away_ab=("away_team_abbrev", "first"))
                    .to_dict(orient="index"))

    # ---------- shifts: stream, keep skaters only ----------
    print(f"[situation_toi] streaming shifts {SHIFT_CSV} ...")
    shifts = _ds.load_shift_data(usecols=REQUIRED_SHIFT_COLS)
    shifts = shifts.dropna(subset=REQUIRED_SHIFT_COLS)
    for _c in ("game_id", "player_id", "abs_start_secs", "abs_end_secs"):
        shifts[_c] = shifts[_c].astype(int)
    shifts = shifts[~shifts["player_id"].isin(goalie_ids)]
    print(f"[situation_toi]   skater shifts: {len(shifts):,}")
    shifts_by_game = dict(tuple(shifts.groupby("game_id")))

    # ---------- per-game segment build + shift intersection ----------
    # accumulators keyed by (player_id, season, game_type)
    sit_sec: dict = defaultdict(lambda: defaultdict(float))   # key -> {situation: seconds}
    games_played: dict = defaultdict(set)                     # key -> {game_id}

    n_games = 0
    n_no_shifts = 0
    for gid, gshots in shots_by_game.items():
        season = g_season.get(gid)
        gtype = g_type.get(gid)
        if season is None or gtype is None:
            continue
        n_games += 1
        if n_games % 1000 == 0:
            print(f"[situation_toi]   processed {n_games:,} games")
        if gid not in shifts_by_game:
            n_no_shifts += 1
            continue
        gshifts = shifts_by_game[gid]
        home_ab = team_abbrevs[gid]["home_ab"]
        away_ab = team_abbrevs[gid]["away_ab"]

        # game end = latest of last event / last shift end (covers OT length)
        game_end = int(max(int(gshots["abs_time"].max()),
                           int(gshifts["abs_end_secs"].max())))

        ev = gshots[["abs_time", "home_sk", "away_sk"]].to_numpy()
        seg_starts, seg_ends, seg_home_sk, seg_away_sk = [], [], [], []
        prev_t = 0
        cur_home_sk, cur_away_sk = 5, 5   # assume 5v5 before first event
        for row in ev:
            t = int(row[0])
            if t > game_end:
                t = game_end
            if t > prev_t:
                seg_starts.append(prev_t); seg_ends.append(t)
                seg_home_sk.append(cur_home_sk); seg_away_sk.append(cur_away_sk)
            cur_home_sk = int(row[1]); cur_away_sk = int(row[2])
            prev_t = t
        if prev_t < game_end:
            seg_starts.append(prev_t); seg_ends.append(game_end)
            seg_home_sk.append(cur_home_sk); seg_away_sk.append(cur_away_sk)
        if not seg_starts:
            continue

        seg_starts_a = np.array(seg_starts, dtype=np.int64)
        seg_ends_a = np.array(seg_ends, dtype=np.int64)
        seg_home_a = np.array(seg_home_sk, dtype=np.int64)
        seg_away_a = np.array(seg_away_sk, dtype=np.int64)

        sh_st = gshifts["abs_start_secs"].to_numpy(dtype=np.int64)
        sh_en = gshifts["abs_end_secs"].to_numpy(dtype=np.int64)
        sh_pid = gshifts["player_id"].to_numpy(dtype=np.int64)
        sh_team = gshifts["team_abbrev"].to_numpy()

        for i in range(len(sh_pid)):
            s, e = sh_st[i], sh_en[i]
            if e > game_end:
                e = game_end
            if e <= s:
                continue
            team = sh_team[i]
            is_home = (team == home_ab)
            if not is_home and team != away_ab:
                continue
            pid = int(sh_pid[i])
            key = (pid, season, gtype)
            games_played[key].add(gid)
            lo = np.searchsorted(seg_ends_a, s, side="right")
            hi = np.searchsorted(seg_starts_a, e, side="left")
            for j in range(lo, hi):
                ovl = min(e, seg_ends_a[j]) - max(s, seg_starts_a[j])
                if ovl <= 0:
                    continue
                if is_home:
                    own, opp = int(seg_home_a[j]), int(seg_away_a[j])
                else:
                    own, opp = int(seg_away_a[j]), int(seg_home_a[j])
                sit_sec[key][situation_label(own, opp)] += ovl

    print(f"[situation_toi]   processed {n_games:,} games  (skipped, no shifts: {n_no_shifts:,})")

    # ---------- assemble long output ----------
    rows = []
    for key, sits in sit_sec.items():
        pid, season, gtype = key
        gp = len(games_played.get(key, ()))
        for sit, sec in sits.items():
            rows.append({
                "player_id": pid,
                "player_name": name_map.get(pid, ""),
                "position": posn_map.get(pid, ""),
                "season": season,
                "game_type": gtype,
                "situation": sit,
                "toi_min": round(sec / 60.0, 4),
                "gp": gp,
            })
    out = pd.DataFrame(rows).sort_values(
        ["season", "game_type", "player_id", "situation"]).reset_index(drop=True)

    # ---------- sanity + summary ----------
    print(f"\n[situation_toi] rows: {len(out):,}  "
          f"players: {out['player_id'].nunique():,}  "
          f"seasons: {sorted(out['season'].unique())}")
    reg = out[out["game_type"] == "regular"]
    share = (reg.groupby("situation")["toi_min"].sum()
             .sort_values(ascending=False))
    tot = share.sum()
    print("[situation_toi] regular-season league TOI share by situation:")
    for sit, mins in share.items():
        print(f"   {sit:<6} {mins:>14,.1f} min   {mins / tot * 100:5.2f}%")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[situation_toi] wrote {OUT_CSV}  ({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
