#!/usr/bin/env python3
"""Box-score fields the NHL summary reports DON'T expose, from the cached PBP.

Fills the gaps in build_box_score.py using Zones/raw/pbp (all event types):
  A1 / A2        primary / secondary assists (goal assist1/assist2 player ids)
  hits_taken     times the player was hit (hittee)
  fo_won/fo_lost faceoff wins / losses (backup to the API faceoffwins report)
  rebounds_created  a player's shot followed within 3s by another shot from the
                    same team (a rebound off their shot)

Cache covers 2022-23..2025-26 (regular + playoff); older seasons get no
supplement (those box-score columns stay blank there).

Output: Data/box_score_pbp.csv  (player_id, season, game_type, A1, A2,
        hits_taken, fo_won, fo_lost, rebounds_created)
"""
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))
PBP_DIR = ROOT / "Zones" / "raw" / "pbp"
OUT = ROOT / "Data" / "box_score_pbp.csv"
SHOTS = {"shot-on-goal", "missed-shot", "goal"}
REBOUND_SEC = 3.0


def _abs_time(p):
    per = (p.get("periodDescriptor", {}) or {}).get("number", 1)
    t = p.get("timeInPeriod", "0:00") or "0:00"
    try:
        mm, ss = t.split(":")
        return (int(per) - 1) * 1200 + int(mm) * 60 + int(ss)
    except Exception:
        return (int(per) - 1) * 1200


def main() -> int:
    files = sorted(glob.glob(str(PBP_DIR / "*.json")))
    if not files:
        print(f"no PBP in {PBP_DIR}")
        return 2
    # acc[(player_id, season, game_type)][field] = count
    acc = defaultdict(lambda: defaultdict(int))
    print(f"parsing {len(files):,} games...")
    for n, fp in enumerate(files, 1):
        if n % 1000 == 0:
            print(f"  {n:,}")
        gid = Path(fp).stem
        start = gid[:4]
        season = f"{start}{int(start) + 1}"
        gt = "regular" if gid[4:6] == "02" else ("playoff" if gid[4:6] == "03" else None)
        if gt is None:
            continue
        try:
            plays = json.load(open(fp)).get("plays", [])
        except Exception:
            continue
        plays = [p for p in plays if isinstance(p, dict)]
        plays.sort(key=lambda p: p.get("sortOrder", 0))

        def add(pid, field):
            if pid is not None:
                acc[(int(pid), season, gt)][field] += 1

        prev_shot = None   # (team_id, abs_time, shooter_id)
        for p in plays:
            et = p.get("typeDescKey")
            det = p.get("details", {}) or {}
            if et == "goal":
                add(det.get("assist1PlayerId"), "A1")
                add(det.get("assist2PlayerId"), "A2")
            elif et == "hit":
                add(det.get("hitteePlayerId"), "hits_taken")
            elif et == "faceoff":
                add(det.get("winningPlayerId"), "fo_won")
                add(det.get("losingPlayerId"), "fo_lost")
            if et in SHOTS:
                team = det.get("eventOwnerTeamId")
                t = _abs_time(p)
                shooter = det.get("shootingPlayerId") or det.get("scoringPlayerId")
                if (prev_shot and prev_shot[0] == team
                        and 0 <= t - prev_shot[1] <= REBOUND_SEC
                        and prev_shot[2] is not None):
                    acc[(int(prev_shot[2]), season, gt)]["rebounds_created"] += 1
                prev_shot = (team, t, shooter)

    fields = ["A1", "A2", "hits_taken", "fo_won", "fo_lost", "rebounds_created"]
    rows = []
    for (pid, season, gt), d in acc.items():
        rows.append({"player_id": pid, "season": season, "game_type": gt,
                     **{f: d.get(f, 0) for f in fields}})
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out):,} rows, {out['player_id'].nunique():,} players")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
