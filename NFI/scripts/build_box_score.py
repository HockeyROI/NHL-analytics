#!/usr/bin/env python3
"""Skater box-score stats from the NHL Stats REST API (in-house pull).

Pulls the public NHL stats reports per (season, game_type) and merges them into
one per-(player, season, game_type) box-score table:
  summary  -> G, A, Pts, Sh, Sh%, PPG, PPP, SHG, SHP, EVG, EVP, PIM,
              faceoffWinPct, +/-, GWG, GP, position, team, shoots
  realtime -> hits (given), blockedShots, takeaways, giveaways, TOI/GP
  penalties-> minorPenalties, majorPenalties, penaltiesDrawn, misconducts
  faceoffwins -> faceoffsWon / faceoffsLost / totalFaceoffs (best-effort)

Gaps NOT in these reports (primary/secondary assist split, hits TAKEN,
rebounds created) are filled separately from the cached play-by-play by
build_box_score_pbp.py. Cap hit is a separate contract-data source.

Output: Data/box_score_skaters.csv
"""
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))
OUT = ROOT / "Data" / "box_score_skaters.csv"
BASE = "https://api.nhle.com/stats/rest/en/skater"
SEASONS = ["20202021", "20212022", "20222023", "20232024", "20242025", "20252026"]
GAME_TYPES = {"regular": 2, "playoff": 3}
REPORTS = ["summary", "realtime", "penalties", "faceoffwins", "shootout"]
HDRS = {"User-Agent": "Mozilla/5.0 (HockeyROI box-score pull)"}
# fields to keep per report (playerId always kept for the merge)
KEEP = {
    "summary": ["skaterFullName", "positionCode", "teamAbbrevs", "shootsCatches",
                "gamesPlayed", "goals", "assists", "points", "shots", "shootingPct",
                "ppGoals", "ppPoints", "shGoals", "shPoints", "evGoals", "evPoints",
                "penaltyMinutes", "plusMinus", "gameWinningGoals", "faceoffWinPct"],
    "realtime": ["hits", "blockedShots", "takeaways", "giveaways", "timeOnIcePerGame"],
    "penalties": ["minorPenalties", "majorPenalties", "misconductPenalties",
                  "penaltiesDrawn"],
    "faceoffwins": ["totalFaceoffs", "totalFaceoffWins", "totalFaceoffLosses"],
    # Shootout is its own report (regular season only; playoffs have no
    # shootout, so gt2 returns zeros/empties there and that's fine).
    "shootout": ["shootoutGoals", "shootoutShots", "shootoutShootingPct"],
}


def _fetch(report, season, gt_id):
    rows, start, limit = [], 0, 100
    while True:
        exp = f"seasonId={season} and gameTypeId={gt_id}"
        url = (f"{BASE}/{report}?isAggregate=false&isGame=false&start={start}"
               f"&limit={limit}&cayenneExp={urllib.parse.quote(exp)}")
        try:
            req = urllib.request.Request(url, headers=HDRS)
            r = json.load(urllib.request.urlopen(req, timeout=25))
        except Exception as e:
            print(f"    {report} {season} gt{gt_id} start{start}: {e}")
            break
        data = r.get("data", [])
        rows.extend(data)
        if len(data) < limit:
            break
        start += limit
        time.sleep(0.3)
    return rows


def main() -> int:
    frames = []
    for season in SEASONS:
        for gt_name, gt_id in GAME_TYPES.items():
            print(f"{season} {gt_name} ...")
            base = None
            for report in REPORTS:
                rows = _fetch(report, season, gt_id)
                if not rows:
                    continue
                df = pd.DataFrame(rows)
                cols = ["playerId"] + [c for c in KEEP[report] if c in df.columns]
                # some reports (e.g. faceoffwins) return duplicate rows per
                # player -> dedupe before the playerId merge to avoid a
                # cartesian blow-up (centers were multiplied x4).
                df = df[cols].drop_duplicates("playerId")
                if base is None:
                    base = df
                else:
                    dup = [c for c in df.columns if c in base.columns and c != "playerId"]
                    df = df.drop(columns=dup)
                    base = base.merge(df, on="playerId", how="outer")
                time.sleep(0.2)
            if base is None or base.empty:
                continue
            base["season"] = season
            base["game_type"] = gt_name
            frames.append(base)
            print(f"    {len(base)} skaters")

    if not frames:
        print("no data pulled")
        return 2
    out = pd.concat(frames, ignore_index=True)
    # tidy names / faceoff W-L
    out = out.rename(columns={"playerId": "player_id", "skaterFullName": "player_name",
                              "positionCode": "position", "teamAbbrevs": "team",
                              "shootsCatches": "shoots", "gamesPlayed": "GP",
                              "timeOnIcePerGame": "toi_per_game_sec",
                              "totalFaceoffWins": "faceoffs_won",
                              "totalFaceoffLosses": "faceoffs_lost"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out):,} player-season-type rows, "
          f"{out['player_id'].nunique():,} players, cols={len(out.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
