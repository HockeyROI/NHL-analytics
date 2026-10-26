#!/usr/bin/env python3
"""Per-game, all-situations box-score events from the cached PBP.

Richer per-game counting stats for the game-log Box Score family (the season
counterpart is build_box_score_pbp.py). One row per (player_id, game_id):

  G      goals (scoringPlayerId)
  A1 A2  primary / secondary assists (goal assist1/assist2)
  SOG    shots on goal (shot-on-goal + goal)
  Hits   hits given (hittingPlayerId)
  Blocks blocked shots (blockingPlayerId)
  TK GV  takeaways / giveaways (playerId)
  PIM    penalty minutes (committedByPlayerId, summed duration)
  MinPen MajPen  minor / major penalties taken; PenDrawn penalties drawn
  PPG PPA SHG SHA  power-play / shorthanded goals + assists (man advantage from
                 the goal's situationCode, net of a pulled goalie)
  GWG    the winning team's goal that put it ahead for good (not shootouts)
  HitsTaken  times hit; FOW FOL faceoffs won / lost
  RebC   rebounds created (own shot followed within 3s by a teammate shot)
  SOG_SO SOA  shootout goals / attempts (shootout plays are excluded from
                 every other field, like the NHL box score)

All situations (not 5v5-only), matching how a real box score reads. Writes a
committed, CI-portable per-season parquet:

  Data/box_pbp_by_game/{season}.parquet

INCREMENTAL: games already in the parquet are kept; only PBP games not yet
present are parsed and appended (so CI/local re-runs add just the new games).
"""
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))
PBP_DIR = ROOT / "Zones" / "raw" / "pbp"
OUT_DIR = ROOT / "Data" / "box_pbp_by_game"
SHOTS_ON_GOAL = {"shot-on-goal", "goal"}
FIELDS = ["G", "A1", "A2", "SOG", "Hits", "Blocks", "TK", "GV", "PIM",
          "MinPen", "MajPen", "PenDrawn", "PPG", "PPA", "SHG", "SHA", "GWG",
          "HitsTaken", "FOW", "FOL", "RebC", "SOG_SO", "SOA"]
FENWICK = {"shot-on-goal", "missed-shot", "goal"}
REBOUND_SEC = 3


def _abs_time(p):
    per = (p.get("periodDescriptor", {}) or {}).get("number", 1)
    try:
        mm, ss = (p.get("timeInPeriod", "0:00") or "0:00").split(":")
        return (int(per) - 1) * 1200 + int(mm) * 60 + int(ss)
    except Exception:
        return (int(per) - 1) * 1200


def _strength(code, own_home):
    """(own, opp) skaters from a 4-digit situationCode [awayG, awaySk, homeSk,
    homeG], counting a pulled goalie's extra attacker as not a man advantage."""
    c = str(code or "").zfill(4)
    if not c.isdigit():
        return None
    ag, ask, hsk, hg = (int(x) for x in c)
    a_eff, h_eff = ask - (1 - ag), hsk - (1 - hg)
    return (h_eff, a_eff) if own_home else (a_eff, h_eff)


def main() -> int:
    files = sorted(glob.glob(str(PBP_DIR / "*.json")))
    if not files:
        print(f"no PBP in {PBP_DIR}")
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # existing (game_id) per season, to skip
    existing = {}
    for pq in glob.glob(str(OUT_DIR / "*.parquet")):
        s = Path(pq).stem
        try:
            _d = pd.read_parquet(pq)
            # a parquet from before the extra fields were added is re-parsed whole
            existing[s] = (set(_d["game_id"].unique())
                           if set(FIELDS).issubset(_d.columns) else set())
        except Exception:
            existing[s] = set()

    # acc[(pid, gid, season, gt)][field] = count
    acc = defaultdict(lambda: defaultdict(int))
    seen_games = set()
    added = 0
    print(f"scanning {len(files):,} games...")
    for n, fp in enumerate(files, 1):
        if n % 1000 == 0:
            print(f"  {n:,}")
        gid_s = Path(fp).stem
        try:
            gid = int(gid_s)
        except ValueError:
            continue
        start = gid_s[:4]
        season = f"{start}{int(start) + 1}"
        gt = ("regular" if gid_s[4:6] == "02"
              else ("playoff" if gid_s[4:6] == "03" else None))
        if gt is None or gid in existing.get(season, set()):
            continue
        try:
            _j = json.load(open(fp))
        except Exception:
            continue
        # Only finished games: an in-progress / pre-game snapshot would be
        # appended once and never revisited (the build is incremental).
        if _j.get("gameState") not in ("OFF", "FINAL"):
            continue
        plays = _j.get("plays", [])
        seen_games.add((gid, season, gt))
        added += 1

        def add(pid, field, n=1):
            if pid is not None:
                acc[(int(pid), gid, season, gt)][field] += n

        home_id = (_j.get("homeTeam") or {}).get("id")
        plays = sorted((p for p in plays if isinstance(p, dict)),
                       key=lambda p: p.get("sortOrder", 0))
        goals = []                      # (team_id, scorer) in order, non-shootout
        prev_shot = None                # (team, abs_time, shooter)
        for p in plays:
            et = p.get("typeDescKey")
            d = p.get("details", {}) or {}
            ptype = (p.get("periodDescriptor", {}) or {}).get("periodType")
            if ptype == "SO":           # shootout: its own two fields only
                if et in ("goal", "shot-on-goal", "missed-shot"):
                    sh = d.get("scoringPlayerId") or d.get("shootingPlayerId")
                    add(sh, "SOA")
                    if et == "goal":
                        add(sh, "SOG_SO")
                continue
            if et == "goal":
                sc, a1, a2 = (d.get("scoringPlayerId"), d.get("assist1PlayerId"),
                              d.get("assist2PlayerId"))
                add(sc, "G"); add(sc, "SOG"); add(a1, "A1"); add(a2, "A2")
                team = d.get("eventOwnerTeamId")
                stg = _strength(p.get("situationCode"), team == home_id)
                if stg:
                    kind = "PP" if stg[0] > stg[1] else ("SH" if stg[0] < stg[1] else None)
                    if kind:
                        add(sc, f"{kind}G"); add(a1, f"{kind}A"); add(a2, f"{kind}A")
                goals.append((team, sc))
            elif et == "shot-on-goal":
                add(d.get("shootingPlayerId"), "SOG")
            elif et == "hit":
                add(d.get("hittingPlayerId"), "Hits")
                add(d.get("hitteePlayerId"), "HitsTaken")
            elif et == "blocked-shot":
                # a teammate-blocked shot isn't a block (the NHL box skips it)
                if d.get("reason") == "blocked":
                    add(d.get("blockingPlayerId"), "Blocks")
            elif et == "takeaway":
                add(d.get("playerId"), "TK")
            elif et == "giveaway":
                add(d.get("playerId"), "GV")
            elif et == "faceoff":
                add(d.get("winningPlayerId"), "FOW")
                add(d.get("losingPlayerId"), "FOL")
            elif et == "penalty":
                pid = d.get("committedByPlayerId")
                add(pid, "PIM", int(d.get("duration") or 0))
                tc = str(d.get("typeCode") or "")
                if tc == "MIN":
                    add(pid, "MinPen")
                elif tc == "MAJ":
                    add(pid, "MajPen")
                add(d.get("drawnByPlayerId"), "PenDrawn")
            if et in FENWICK:
                team = d.get("eventOwnerTeamId")
                t = _abs_time(p)
                sh = d.get("shootingPlayerId") or d.get("scoringPlayerId")
                if (prev_shot and prev_shot[0] == team and prev_shot[2] is not None
                        and 0 <= t - prev_shot[1] <= REBOUND_SEC):
                    add(prev_shot[2], "RebC")
                prev_shot = (team, t, sh)
        # Game-winning goal: the winner's goal number (loser's total + 1), only
        # when regulation/OT decided it (a shootout win credits no GWG).
        if goals:
            tally = {}
            for tm, _ in goals:
                tally[tm] = tally.get(tm, 0) + 1
            if len(set(tally.values())) > 1 or len(tally) == 1:
                win = max(tally, key=tally.get)
                lose_n = sum(v for k, v in tally.items() if k != win)
                n = 0
                for tm, sc in goals:
                    if tm == win:
                        n += 1
                        if n == lose_n + 1:
                            add(sc, "GWG")
                            break

    if not seen_games:
        print("no new games to add.")
        return 0
    rows = []
    for (pid, gid, season, gt), dd in acc.items():
        rows.append({"player_id": pid, "game_id": gid, "season": season,
                     "game_type": gt, **{f: dd.get(f, 0) for f in FIELDS}})
    new = pd.DataFrame(rows)
    for season, sub in new.groupby("season"):
        pq = OUT_DIR / f"{season}.parquet"
        if pq.exists():
            old = pd.read_parquet(pq)
            df = pd.concat([old, sub], ignore_index=True)
            df = df.drop_duplicates(subset=["player_id", "game_id"], keep="last")
        else:
            df = sub
        df = df.sort_values(["game_id", "player_id"]).reset_index(drop=True)
        df.to_parquet(pq, index=False)
        print(f"  wrote {season}: +{sub['game_id'].nunique()} games "
              f"({len(df):,} rows) -> {pq}")
    print(f"\ndone: {added} new games parsed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
