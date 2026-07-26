#!/usr/bin/env python3
"""Team box-score + special-teams stats from the NHL Stats REST API (in-house pull).

Pulls the public NHL team stats reports per (season, game_type) and merges them
into one per-(team, season, game_type) table:
  summary   -> GF, GA, points, wins/losses, PP%/PK% (official), faceoffWinPct
  powerplay -> PP goals for, PP opportunities, SH goals against (on the PP)
  penaltykill -> PK%, PP goals against, times shorthanded, SH goals for
  realtime  -> hits, blockedShots, takeaways, giveaways, shots, empty-net goals

Unlike aggregating the skater box (Data/box_score_skaters.csv), these are the
league's OWN team totals, so a mid-season trade doesn't smear a player's
season line across a "EDM,COL" pseudo-team.

teamId -> triCode from the /team meta endpoint; ARI folded into UTA for
franchise continuity (matches the rest of the app).

Output: Data/team_box_score.csv
"""
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))
OUT = ROOT / "Data" / "team_box_score.csv"
BASE = "https://api.nhle.com/stats/rest/en/team"
SEASONS = ["20202021", "20212022", "20222023", "20232024", "20242025", "20252026"]
GAME_TYPES = {"regular": 2, "playoff": 3}
HDRS = {"User-Agent": "Mozilla/5.0 (HockeyROI team box-score pull)"}

# fields to keep per report (teamId always kept for the merge)
KEEP = {
    "summary": ["gamesPlayed", "goalsFor", "goalsAgainst", "points", "wins",
                "losses", "otLosses", "powerPlayPct", "penaltyKillPct",
                "faceoffWinPct", "shotsForPerGame", "shotsAgainstPerGame"],
    "powerplay": ["powerPlayGoalsFor", "ppOpportunities", "shGoalsAgainst",
                  "ppNetGoals", "ppTimeOnIcePerGame"],
    "penaltykill": ["penaltyKillPct", "ppGoalsAgainst", "timesShorthanded",
                    "shGoalsFor", "pkNetGoals"],
    "realtime": ["hits", "blockedShots", "takeaways", "giveaways", "shots",
                 "emptyNetGoals"],
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


def _team_map():
    """teamId -> triCode (current tricode). ARI folded into UTA downstream."""
    req = urllib.request.Request(f"{BASE}", headers=HDRS)
    d = json.load(urllib.request.urlopen(req, timeout=25)).get("data", [])
    return {int(r["id"]): r["triCode"] for r in d if r.get("id") and r.get("triCode")}


def main() -> int:
    tmap = _team_map()
    frames = []
    for season in SEASONS:
        for gt_name, gt_id in GAME_TYPES.items():
            print(f"{season} {gt_name} ...")
            base = None
            for report in KEEP:
                rows = _fetch(report, season, gt_id)
                if not rows:
                    continue
                df = pd.DataFrame(rows)
                cols = ["teamId"] + [c for c in KEEP[report] if c in df.columns]
                df = df[cols].drop_duplicates("teamId")
                if base is None:
                    base = df
                else:
                    dup = [c for c in df.columns if c in base.columns and c != "teamId"]
                    base = base.merge(df.drop(columns=dup), on="teamId", how="outer")
                time.sleep(0.2)
            if base is None or base.empty:
                continue
            base["season"] = season
            base["game_type"] = gt_name
            frames.append(base)
            print(f"    {len(base)} teams")

    if not frames:
        print("no data pulled")
        return 2
    out = pd.concat(frames, ignore_index=True)
    out["team"] = out["teamId"].map(tmap).replace({"ARI": "UTA"})
    out = out.drop(columns=["teamId"])
    # move id columns to the front
    front = ["team", "season", "game_type"]
    out = out[front + [c for c in out.columns if c not in front]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out):,} team-season-type rows, "
          f"{out['team'].nunique()} teams, cols={len(out.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
