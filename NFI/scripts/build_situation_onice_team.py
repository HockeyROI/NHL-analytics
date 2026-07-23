#!/usr/bin/env python3
"""Per-TEAM, per-situation on-ice counts + TOI — team analog of the player
situation engine (build_situation_onice.py).

Simpler than the player version: team for/against counts don't need the shift
join (every shot belongs to a team regardless of who's on the ice), and a
team's time in a situation is just game-clock in that strength state (no
per-player attribution). So this reads only the shot events + xG.

Per game:
  - build piecewise-constant strength segments from event situation_code and
    add each segment's duration to BOTH teams' situation TOI (home's 5v4 is the
    away team's 4v5).
  - attribute each Fenwick event once to the shooting team (for) under its
    own-perspective situation, and once to the defending team (against) under
    the mirror situation.

Output (long): team, season, game_type, situation, toi_min, gp,
               CF FF xGF GF (for), CA FA xGA GA (against)
-> Data/team_situation_onice.csv

Feeds: the Teams-tab Situation family, and per-situation RelNFI/RelxG (the
team-without-player on/off baseline uses these team totals).
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))
SHOT_CSV = ROOT / "Data" / "nhl_shot_events.csv"
XG_CSV = ROOT / "xG" / "output" / "shot_xg_per_event.csv"
GAMES = ROOT / "Data" / "game_ids.csv"
OUT_CSV = Path(os.environ.get("TEAM_SITUATION_ONICE_OUT", ROOT / "Data" / "team_situation_onice.csv"))

CORSI = {"shot-on-goal", "missed-shot", "goal", "blocked-shot"}
FENWICK = {"shot-on-goal", "missed-shot", "goal"}
_KEEP = {3, 4, 5, 6}
FIELDS = ["toi_min", "CF", "FF", "xGF", "GF", "CA", "FA", "xGA", "GA"]


def situation_label(own: int, opp: int) -> str:
    return f"{own}v{opp}" if own in _KEEP and opp in _KEEP else "other"


def main() -> int:
    for p in (SHOT_CSV, XG_CSV, GAMES):
        if not p.exists():
            print(f"[team_situation] missing {p}", file=sys.stderr)
            return 2

    g = pd.read_csv(GAMES, dtype={"game_id": int, "season": str})
    g_season = dict(zip(g["game_id"], g["season"]))
    g_type = dict(zip(g["game_id"], g["game_type"]))

    print("[team_situation] reading events ...")
    use = ["game_id", "period", "time_secs", "event_id", "event_type", "is_goal",
           "situation_code", "shooting_team_id", "home_team_id",
           "home_team_abbrev", "away_team_abbrev"]
    ev = pd.read_csv(SHOT_CSV, usecols=use, dtype={"situation_code": str})
    ev = ev[ev["event_type"].isin(CORSI)].dropna(subset=["situation_code", "period", "time_secs"])
    xg = pd.read_csv(XG_CSV)
    ev = ev.merge(xg, on=["game_id", "event_id"], how="left")
    ev["xg"] = ev["xg"].fillna(0.0)
    ev["abs_time"] = ev["time_secs"].astype(int) + (ev["period"].astype(int) - 1) * 1200
    ev["is_fen"] = ev["event_type"].isin(FENWICK)
    ev["is_goal_i"] = ev["is_goal"].astype(int)
    sc = ev["situation_code"].astype(str).str.zfill(4)
    ev["away_sk"], ev["home_sk"] = sc.str[1].astype(int), sc.str[2].astype(int)
    ev["shoot_home"] = ev["shooting_team_id"] == ev["home_team_id"]
    ev = ev.sort_values(["game_id", "abs_time", "event_id"]).reset_index(drop=True)
    print(f"[team_situation]   {len(ev):,} Corsi events")

    IDX = {f: i for i, f in enumerate(FIELDS)}
    acc: dict = defaultdict(lambda: np.zeros(len(FIELDS)))   # (team,season,gt,sit)->vec
    games_played: dict = defaultdict(set)                     # (team,season,gt)->{gid}

    for gid, ge in ev.groupby("game_id"):
        season, gtype = g_season.get(gid), g_type.get(gid)
        if season is None or gtype is None:
            continue
        home = ge["home_team_abbrev"].iloc[0]
        away = ge["away_team_abbrev"].iloc[0]
        games_played[(home, season, gtype)].add(gid)
        games_played[(away, season, gtype)].add(gid)
        game_end = int(ge["abs_time"].max())

        # strength segments -> team situation TOI (both perspectives)
        arr = ge[["abs_time", "home_sk", "away_sk"]].to_numpy()
        prev_t, cur_h, cur_a = 0, 5, 5
        for row in arr:
            t = min(int(row[0]), game_end)
            if t > prev_t:
                dur = (t - prev_t) / 60.0
                acc[(home, season, gtype, situation_label(cur_h, cur_a))][IDX["toi_min"]] += dur
                acc[(away, season, gtype, situation_label(cur_a, cur_h))][IDX["toi_min"]] += dur
            cur_h, cur_a = int(row[1]), int(row[2])
            prev_t = t

        # event counts: for -> shooting team, against -> defending team
        for r in ge.itertuples(index=False):
            sh_home = bool(r.shoot_home)
            own_sk, opp_sk = (int(r.home_sk), int(r.away_sk)) if sh_home else (int(r.away_sk), int(r.home_sk))
            for_team, ag_team = (home, away) if sh_home else (away, home)
            for_lab, ag_lab = situation_label(own_sk, opp_sk), situation_label(opp_sk, own_sk)
            af = acc[(for_team, season, gtype, for_lab)]
            aa = acc[(ag_team, season, gtype, ag_lab)]
            af[IDX["CF"]] += 1
            aa[IDX["CA"]] += 1
            if r.is_fen:
                af[IDX["FF"]] += 1; af[IDX["xGF"]] += float(r.xg); af[IDX["GF"]] += int(r.is_goal_i)
                aa[IDX["FA"]] += 1; aa[IDX["xGA"]] += float(r.xg); aa[IDX["GA"]] += int(r.is_goal_i)

    rows = []
    for (team, season, gtype, sit), vec in acc.items():
        gp = len(games_played.get((team, season, gtype), ()))
        d = {"team": team, "season": season, "game_type": gtype, "situation": sit, "gp": gp}
        for f in FIELDS:
            v = vec[IDX[f]]
            d[f] = round(v, 4) if f in ("toi_min", "xGF", "xGA") else int(round(v))
        rows.append(d)
    out = pd.DataFrame(rows).sort_values(["season", "game_type", "team", "situation"]).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[team_situation] wrote {OUT_CSV}  ({len(out):,} rows, {out['team'].nunique()} teams)")
    # sanity: 5v5 for/against league-symmetric
    reg = out[(out.game_type == "regular") & (out.situation == "5v5")]
    print(f"[team_situation] 5v5 ΣxGF={reg.xGF.sum():,.0f} ΣxGA={reg.xGA.sum():,.0f} "
          f"(diff {abs(reg.xGF.sum()-reg.xGA.sum())/reg.xGF.sum()*100:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
