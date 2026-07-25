#!/usr/bin/env python3
"""Append PLAYOFF shots to NFI/output/shots_tagged.csv.

shots_tagged.csv is the per-shot table (zone, state, score bucket) that the
playoff pipeline (build_playoff_data.py, 03p_player_counts_playoffs.py,
21p_goalie_gsax_playoffs.py, Quality_Games QG_SCOPE=playoff) reads. Its only
regular writer — 03_onice_attribution_pillars.py — is game_type=="regular" only
and would WIPE the playoff rows if run. The playoff rows currently in the file
(2021-2025) came from a tagger that isn't in the pipeline, so a freshly-fetched
postseason (e.g. 2025-26) never gets tagged.

This replicates 03_onice_attribution_pillars.py's shots_tagged logic EXACTLY
(same columns, same state/zone/score derivation), but for playoff games only
(game_id digits 4-5 == "03"), and APPENDS any playoff seasons missing from the
file — regular-season and already-present playoff rows are left untouched.

Idempotent: re-running adds nothing once a season is present.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
sys.path.insert(0, f"{ROOT}/NFI/scripts")
import _data_sources as _ds

SHOTS_TAGGED = f"{ROOT}/NFI/output/shots_tagged.csv"

# Match 03_onice_attribution_pillars.py's SEASONS scope (starts 2021-22, so
# 2020-21 is intentionally excluded from shots_tagged, regular AND playoff).
SEASONS = {"20212022", "20222023", "20232024", "20242025", "20252026"}

INFL1 = 55   # MNFI/FNFI boundary  (matches 03_onice_attribution_pillars.py)
BLUE = 25


def classify_zone(x, y):
    if pd.isna(x) or pd.isna(y):
        return "unk"
    if 74 <= x <= 89 and -9 <= y <= 9:
        return "CNFI"
    if -15 <= y <= 15:
        if INFL1 <= x < 74:
            return "MNFI"
        if BLUE <= x < INFL1:
            return "FNFI"
        return "lane_other"
    return "Wide"


def score_bucket(d):
    if d <= -2:
        return "trail2plus"
    if d == -1:
        return "trail1"
    if d == 0:
        return "tied"
    if d == 1:
        return "lead1"
    return "lead2plus"


OUT_COLS = ["game_id", "season", "period", "event_id", "abs_time", "event_type",
            "x_coord_norm", "y_coord_norm", "shooting_team_id",
            "shooting_team_abbrev", "home_team_abbrev", "away_team_abbrev",
            "shooter_player_id", "goalie_id", "state", "zone", "is_goal_i",
            "score_bucket", "shoot_diff", "shoot_home"]


def main() -> int:
    # Playoff seasons already tagged -> skip them. Key on the game_id start-year
    # prefix (e.g. "2024" for 2024030xxx) on BOTH sides so the skip actually
    # matches — comparing this prefix against the "20242025" season code (an
    # earlier bug) never matched and re-tagged every season, duplicating rows.
    have = set()
    if os.path.exists(SHOTS_TAGGED):
        st = pd.read_csv(SHOTS_TAGGED, usecols=["game_id"])
        gp = st["game_id"].astype(str)
        have = set(s[:4] for s in gp[gp.str[4:6] == "03"])
    print(f"playoff start-years already in shots_tagged: {sorted(have)}")

    cols = ["game_id", "season", "period", "event_id", "event_type",
            "situation_code", "time_secs", "home_team_id", "shooting_team_id",
            "home_team_abbrev", "away_team_abbrev", "shooting_team_abbrev",
            "shooter_player_id", "goalie_id", "x_coord_norm", "y_coord_norm",
            "is_goal"]
    shots = _ds.load_shot_events(usecols=cols, dtype={"season": str,
                                                      "situation_code": str})
    shots = shots[(shots["game_id"].astype(str).str[4:6] == "03")
                  & shots["period"].between(1, 3)
                  & shots["event_type"].isin(
                      ["shot-on-goal", "missed-shot", "blocked-shot", "goal"])].copy()
    shots["season"] = shots["season"].astype(str)
    shots["_startyr"] = shots["game_id"].astype(str).str[:4]
    shots = shots[shots["season"].isin(SEASONS) & ~shots["_startyr"].isin(have)].copy()
    if shots.empty:
        print("Nothing to add — every playoff season is already tagged.")
        return 0
    print(f"tagging {len(shots):,} playoff shots for seasons "
          f"{sorted(shots['season'].unique())}")

    shots["abs_time"] = (shots["time_secs"].astype(int)
                         + (shots["period"].astype(int) - 1) * 1200)
    shots["_shoot_home"] = shots["shooting_team_id"] == shots["home_team_id"]

    sc = shots["situation_code"].astype(str).str.zfill(4)
    ag, ask = sc.str[0].astype(int), sc.str[1].astype(int)
    hsk, hg = sc.str[2].astype(int), sc.str[3].astype(int)
    empty_net = (ag == 0) | (hg == 0)
    sh = np.where(shots["_shoot_home"], hsk, ask)
    op = np.where(shots["_shoot_home"], ask, hsk)
    shots["state"] = np.where((sh == op) & (sh == 5), "ES",
                     np.where((sh == op) & (sh == 4), "4v4",
                     np.where((sh == op) & (sh == 3), "3v3",
                     np.where(sh > op, "PP", "PK"))))
    shots = shots[~empty_net].copy()

    shots["zone"] = [classify_zone(x, y) for x, y in
                     zip(shots["x_coord_norm"].values, shots["y_coord_norm"].values)]
    shots["is_goal_i"] = shots["is_goal"].astype(int)

    # running pre-event score -> shoot_diff -> bucket (identical to 03_onice)
    shots = shots.sort_values(["game_id", "abs_time", "event_id"]).reset_index(drop=True)
    shots["home_goal"] = ((shots["event_type"] == "goal")
                          & (shots["shooting_team_id"] == shots["home_team_id"])).astype(int)
    shots["away_goal"] = ((shots["event_type"] == "goal")
                          & (shots["shooting_team_id"] != shots["home_team_id"])).astype(int)
    shots["home_score"] = shots.groupby("game_id")["home_goal"].cumsum() - shots["home_goal"]
    shots["away_score"] = shots.groupby("game_id")["away_goal"].cumsum() - shots["away_goal"]
    shots["shoot_diff"] = np.where(shots["_shoot_home"],
                                   shots["home_score"] - shots["away_score"],
                                   shots["away_score"] - shots["home_score"])
    shots["score_bucket"] = shots["shoot_diff"].apply(score_bucket)
    shots = shots.rename(columns={"_shoot_home": "shoot_home"})

    out = shots[OUT_COLS]
    out.to_csv(SHOTS_TAGGED, mode="a", header=False, index=False)
    print(f"Appended {len(out):,} playoff shot rows to {SHOTS_TAGGED}")
    for s in sorted(out["season"].unique()):
        print(f"  {s}: {(out['season'] == s).sum():,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
