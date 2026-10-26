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
FIELDS = ["G", "A1", "A2", "SOG", "Hits", "Blocks", "TK", "GV", "PIM"]


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
            existing[s] = set(pd.read_parquet(pq, columns=["game_id"])["game_id"].unique())
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

        def add(pid, field):
            if pid is not None:
                acc[(int(pid), gid, season, gt)][field] += 1

        for p in plays:
            if not isinstance(p, dict):
                continue
            et = p.get("typeDescKey")
            d = p.get("details", {}) or {}
            if et == "goal":
                add(d.get("scoringPlayerId"), "G")
                add(d.get("scoringPlayerId"), "SOG")
                add(d.get("assist1PlayerId"), "A1")
                add(d.get("assist2PlayerId"), "A2")
            elif et == "shot-on-goal":
                add(d.get("shootingPlayerId"), "SOG")
            elif et == "hit":
                add(d.get("hittingPlayerId"), "Hits")
            elif et == "blocked-shot":
                add(d.get("blockingPlayerId"), "Blocks")
            elif et == "takeaway":
                add(d.get("playerId"), "TK")
            elif et == "giveaway":
                add(d.get("playerId"), "GV")
            elif et == "penalty":
                pid = d.get("committedByPlayerId")
                dur = d.get("duration") or 0
                if pid is not None:
                    acc[(int(pid), gid, season, gt)]["PIM"] += int(dur)

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
