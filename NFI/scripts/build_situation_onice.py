#!/usr/bin/env python3
"""
Per-player, per-situation ON-ICE + INDIVIDUAL counts — the Phase-2 engine.

Companion to build_situation_toi.py. Same validated segment/shift-overlap core,
but also attributes every shot event to the players on ice, so we get the full
possession/xG suite per situation:

  on-ice FOR : CF FF xGF GF     (shooting team's skaters)
  on-ice AG  : CA FA xGA GA     (defending team's skaters)
  individual : iCF iFF ixG iG   (the shooter)
  toi_min    : recomputed here for internal per-60 consistency (cross-checks
               against Data/player_situation_toi.csv)

Keyed per (player_id, season, game_type, situation). Each player's row uses
THEIR OWN team's perspective: a 5v4 event for the PP team is a 4v5 event for the
killing team, so the same shot lands under 5v4 for one and 4v5 for the other.

Event sets (repo convention, see docs/METHODOLOGY.md):
  Corsi   = shot-on-goal + missed-shot + goal + blocked-shot
  Fenwick = shot-on-goal + missed-shot + goal            (blocked excluded)
  xG defined only on SOG+goal (from NFI/output/shot_xg_per_event.csv); misses
  and blocks contribute 0 xG. Goals from is_goal.

Downstream (app) derives rates by dividing counts by toi_min (per-60) and shares
(CF% = CF/(CF+CA), xGF%, etc.), plus RelNFI-style on/off per situation.

Inputs:
  Data/nhl_shot_events.csv
  NFI/output/shot_xg_per_event.csv
  NFI/Geometry_post/Data/shift_data.csv
  Data/game_ids.csv
  NFI/output/player_positions.csv
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
XG_CSV = ROOT / "xG" / "output" / "shot_xg_per_event.csv"  # v2 model (xG/build_xg.py)
SHIFT_CSV = ROOT / "NFI" / "Geometry_post" / "Data" / "shift_data.csv"
GAMES = ROOT / "Data" / "game_ids.csv"
POSITIONS = ROOT / "NFI" / "output" / "player_positions.csv"

OUT_CSV = Path(os.environ.get("SITUATION_ONICE_OUT", ROOT / "Data" / "player_situation_onice.csv"))

REQUIRED_SHIFT_COLS = ["game_id", "player_id", "period", "team_abbrev",
                       "abs_start_secs", "abs_end_secs"]
CORSI = {"shot-on-goal", "missed-shot", "goal", "blocked-shot"}
FENWICK = {"shot-on-goal", "missed-shot", "goal"}
SOG = {"shot-on-goal", "goal"}          # shots on goal (for PDO's SH%/SV%)
_KEEP = {3, 4, 5, 6}

# per-player accumulator field order. SOGF/SOGA + xGFs/xGAs (xG on the SOG
# subset) feed per-situation PDO and PDOxG (both SOG-based, like build_pdo_sog).
FIELDS = ["toi_min", "CF", "FF", "xGF", "GF", "CA", "FA", "xGA", "GA",
          "SOGF", "SOGA", "xGFs", "xGAs", "iCF", "iFF", "ixG", "iG"]


def situation_label(own: int, opp: int) -> str:
    if own in _KEEP and opp in _KEEP:
        return f"{own}v{opp}"
    return "other"


def stop(msg: str, code: int = 2) -> int:
    print(f"[situation_onice] STOP — {msg}", file=sys.stderr)
    return code


def main() -> int:
    for p in (XG_CSV, GAMES, POSITIONS):
        if not p.exists():
            return stop(f"input not found: {p}")

    pos = pd.read_csv(POSITIONS)
    name_map = dict(zip(pos["player_id"], pos["player_name"]))
    posn_map = dict(zip(pos["player_id"], pos["position"]))
    goalie_ids = set(pos.loc[pos["position"] == "G", "player_id"].astype(int))

    g = pd.read_csv(GAMES, dtype={"game_id": int, "season": str})
    g_season = dict(zip(g["game_id"], g["season"]))
    g_type = dict(zip(g["game_id"], g["game_type"]))

    # ---------- events (+ xG merge) ----------
    print(f"[situation_onice] reading events ...")
    use_shots = ["game_id", "period", "time_secs", "event_id", "event_type",
                 "is_goal", "situation_code", "shooting_team_id", "home_team_id",
                 "shooting_team_abbrev", "home_team_abbrev", "away_team_abbrev"]
    ev = _ds.load_shot_events(usecols=use_shots, dtype={"situation_code": str})
    ev = ev[ev["event_type"].isin(CORSI)].copy()
    ev = ev.dropna(subset=["situation_code", "period", "time_secs"])
    xg = pd.read_csv(XG_CSV)
    ev = ev.merge(xg, on=["game_id", "event_id"], how="left")
    ev["xg"] = ev["xg"].fillna(0.0)
    ev["abs_time"] = ev["time_secs"].astype(int) + (ev["period"].astype(int) - 1) * 1200
    ev["is_fen"] = ev["event_type"].isin(FENWICK)
    ev["is_sog"] = ev["event_type"].isin(SOG)
    ev["is_goal_i"] = ev["is_goal"].astype(int)
    sc = ev["situation_code"].astype(str).str.zfill(4)
    ev["away_sk"] = sc.str[1].astype(int)
    ev["home_sk"] = sc.str[2].astype(int)
    ev["shoot_home"] = ev["shooting_team_id"] == ev["home_team_id"]
    ev = ev.sort_values(["game_id", "abs_time", "event_id"]).reset_index(drop=True)
    print(f"[situation_onice]   {len(ev):,} Corsi events "
          f"({int(ev['is_fen'].sum()):,} Fenwick, {int(ev['is_goal_i'].sum()):,} goals)")
    ev_by_game = dict(tuple(ev.groupby("game_id")))
    team_abbrevs = (ev.groupby("game_id")
                    .agg(home_ab=("home_team_abbrev", "first"),
                         away_ab=("away_team_abbrev", "first")).to_dict(orient="index"))

    # ---------- shifts ----------
    print(f"[situation_onice] streaming shifts ...")
    shifts = _ds.load_shift_data(usecols=REQUIRED_SHIFT_COLS)
    shifts = shifts.dropna(subset=REQUIRED_SHIFT_COLS)
    for _c in ("game_id", "player_id", "abs_start_secs", "abs_end_secs"):
        shifts[_c] = shifts[_c].astype(int)
    shifts = shifts[~shifts["player_id"].isin(goalie_ids)]
    print(f"[situation_onice]   skater shifts: {len(shifts):,}")
    shifts_by_game = dict(tuple(shifts.groupby("game_id")))

    # acc[(pid, season, gtype, situation)] -> np.array over FIELDS
    acc: dict = defaultdict(lambda: np.zeros(len(FIELDS), dtype=np.float64))
    games_played: dict = defaultdict(set)   # (pid, season, gtype) -> {gid}
    IDX = {f: i for i, f in enumerate(FIELDS)}

    n_games = 0
    for gid, ge in ev_by_game.items():
        season, gtype = g_season.get(gid), g_type.get(gid)
        if season is None or gtype is None or gid not in shifts_by_game:
            continue
        n_games += 1
        if n_games % 1000 == 0:
            print(f"[situation_onice]   processed {n_games:,} games")
        gs = shifts_by_game[gid]
        home_ab = team_abbrevs[gid]["home_ab"]
        away_ab = team_abbrevs[gid]["away_ab"]
        game_end = int(max(int(ge["abs_time"].max()), int(gs["abs_end_secs"].max())))

        sh_st = gs["abs_start_secs"].to_numpy(np.int64)
        sh_en = gs["abs_end_secs"].to_numpy(np.int64)
        sh_pid = gs["player_id"].to_numpy(np.int64)
        sh_home = (gs["team_abbrev"].to_numpy() == home_ab)
        sh_valid_team = sh_home | (gs["team_abbrev"].to_numpy() == away_ab)

        # ---------- TOI via situation segments (same core as build_situation_toi) ----------
        e_arr = ge[["abs_time", "home_sk", "away_sk"]].to_numpy()
        seg_s, seg_e, seg_h, seg_a = [], [], [], []
        prev_t, cur_h, cur_a = 0, 5, 5
        for row in e_arr:
            t = min(int(row[0]), game_end)
            if t > prev_t:
                seg_s.append(prev_t); seg_e.append(t); seg_h.append(cur_h); seg_a.append(cur_a)
            cur_h, cur_a = int(row[1]), int(row[2])
            prev_t = t
        if prev_t < game_end:
            seg_s.append(prev_t); seg_e.append(game_end); seg_h.append(cur_h); seg_a.append(cur_a)
        seg_s_a = np.array(seg_s, np.int64); seg_e_a = np.array(seg_e, np.int64)
        seg_h_a = np.array(seg_h, np.int64); seg_a_a = np.array(seg_a, np.int64)
        for i in range(len(sh_pid)):
            if not sh_valid_team[i]:
                continue
            s, e = sh_st[i], min(sh_en[i], game_end)
            if e <= s:
                continue
            pid = int(sh_pid[i])
            games_played[(pid, season, gtype)].add(gid)
            lo = np.searchsorted(seg_e_a, s, side="right")
            hi = np.searchsorted(seg_s_a, e, side="left")
            for j in range(lo, hi):
                ovl = min(e, seg_e_a[j]) - max(s, seg_s_a[j])
                if ovl <= 0:
                    continue
                own, opp = (seg_h_a[j], seg_a_a[j]) if sh_home[i] else (seg_a_a[j], seg_h_a[j])
                acc[(pid, season, gtype, situation_label(int(own), int(opp)))][IDX["toi_min"]] += ovl / 60.0

        # ---------- event attribution ----------
        for row in ge.itertuples(index=False):
            t = int(row.abs_time)
            # (start, end] — include shifts ending exactly at the event (goals
            # end shifts), exclude ones starting at it. Fixes on-ice GF/GA
            # undercount vs the [start, end) convention.
            on = (sh_st < t) & (sh_en >= t) & sh_valid_team
            if not on.any():
                continue
            shoot_home = bool(row.shoot_home)
            # own = shooting team, opp = defending
            if shoot_home:
                own_sk, opp_sk = int(row.home_sk), int(row.away_sk)
            else:
                own_sk, opp_sk = int(row.away_sk), int(row.home_sk)
            for_lab = situation_label(own_sk, opp_sk)
            ag_lab = situation_label(opp_sk, own_sk)
            is_fen = bool(row.is_fen)
            is_sog = bool(row.is_sog)
            xgv = float(row.xg)
            gl = int(row.is_goal_i)
            on_idx = np.nonzero(on)[0]
            for k in on_idx:
                pid = int(sh_pid[k])
                p_home = bool(sh_home[k])
                if p_home == shoot_home:
                    a = acc[(pid, season, gtype, for_lab)]
                    a[IDX["CF"]] += 1
                    if is_sog:
                        a[IDX["SOGF"]] += 1; a[IDX["xGFs"]] += xgv
                    if is_fen:
                        a[IDX["FF"]] += 1; a[IDX["xGF"]] += xgv; a[IDX["GF"]] += gl
                else:
                    a = acc[(pid, season, gtype, ag_lab)]
                    a[IDX["CA"]] += 1
                    if is_sog:
                        a[IDX["SOGA"]] += 1; a[IDX["xGAs"]] += xgv
                    if is_fen:
                        a[IDX["FA"]] += 1; a[IDX["xGA"]] += xgv; a[IDX["GA"]] += gl

    print(f"[situation_onice]   processed {n_games:,} games")

    # ---------- individual counts (shooter) — separate vectorized pass ----------
    print("[situation_onice] individual (shooter) counts ...")
    iev = pd.read_csv(SHOT_CSV, usecols=["game_id", "event_id", "event_type", "is_goal",
                                         "situation_code", "shooting_team_id", "home_team_id",
                                         "shooter_player_id"],
                      dtype={"situation_code": str})
    iev = iev[iev["event_type"].isin(CORSI)].dropna(subset=["shooter_player_id", "situation_code"])
    iev = iev.merge(xg, on=["game_id", "event_id"], how="left")
    iev["xg"] = iev["xg"].fillna(0.0)
    iev["season"] = iev["game_id"].map(g_season)
    iev["game_type"] = iev["game_id"].map(g_type)
    iev = iev.dropna(subset=["season", "game_type"])
    iev = iev[~iev["shooter_player_id"].astype(int).isin(goalie_ids)]
    sc = iev["situation_code"].astype(str).str.zfill(4)
    iev["away_sk"] = sc.str[1].astype(int); iev["home_sk"] = sc.str[2].astype(int)
    sh_home = iev["shooting_team_id"] == iev["home_team_id"]
    own = np.where(sh_home, iev["home_sk"], iev["away_sk"])
    opp = np.where(sh_home, iev["away_sk"], iev["home_sk"])
    iev["situation"] = [situation_label(int(o), int(p)) for o, p in zip(own, opp)]
    iev["is_fen"] = iev["event_type"].isin(FENWICK)
    iev["is_goal_i"] = iev["is_goal"].astype(int)
    for r in iev.itertuples(index=False):
        pid = int(r.shooter_player_id)
        a = acc[(pid, r.season, r.game_type, r.situation)]
        a[IDX["iCF"]] += 1
        if r.is_fen:
            a[IDX["iFF"]] += 1; a[IDX["ixG"]] += float(r.xg); a[IDX["iG"]] += int(r.is_goal_i)

    # ---------- assemble ----------
    rows = []
    for (pid, season, gtype, sit), vec in acc.items():
        gp = len(games_played.get((pid, season, gtype), ()))
        d = {"player_id": pid, "player_name": name_map.get(pid, ""),
             "position": posn_map.get(pid, ""), "season": season,
             "game_type": gtype, "situation": sit, "gp": gp}
        for f in FIELDS:
            v = vec[IDX[f]]
            d[f] = round(v, 4) if f in ("toi_min", "xGF", "xGA", "xGFs", "xGAs", "ixG") else int(round(v))
        rows.append(d)
    out = pd.DataFrame(rows).sort_values(
        ["season", "game_type", "player_id", "situation"]).reset_index(drop=True)

    print(f"\n[situation_onice] rows: {len(out):,}  players: {out['player_id'].nunique():,}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[situation_onice] wrote {OUT_CSV}  ({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
