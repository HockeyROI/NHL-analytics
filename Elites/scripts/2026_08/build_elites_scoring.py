"""build_elites_scoring.py -- committed, CI-portable 5v5 scoring for the Elites tiers.

tiers.py needs 5v5 goals + PRIMARY ASSISTS per player, which only exist in the raw
play-by-play (the committed shot parquets have goals but no assists). Raw PBP is
gitignored (absent on CI), so -- exactly like Zones/scripts/build_ccg_events.py --
we distil the minimal per-goal stream tiers needs into a committed per-season
parquet the runner can read without the raw cache:

  Data/elites_scoring_by_season/{season}.parquet
    columns: game_id, season, game_type, scorer, assist1   (one row per 5v5 goal)

INCREMENTAL: only games in the raw PBP cache that are NOT already in the parquet
are added. On CI, update_current_season fetches the week's new games into
Zones/raw/pbp; this appends just those. Locally it backfills the whole cache the
first time. tiers.py then aggregates goals-per-scorer / A1-per-assister from these
parquets -- identical to parsing the raw PBP directly, but portable.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))
PBP_DIR = os.path.join(PROJECT, "Zones", "raw", "pbp")
OUT_DIR = os.path.join(PROJECT, "Data", "elites_scoring_by_season")
os.makedirs(OUT_DIR, exist_ok=True)


def extract(path):
    g = json.load(open(path))
    if g.get("gameType") != 2:                      # regular season only
        return None, []
    season = str(g.get("season"))
    gid = int(g.get("id"))
    rows = []
    for p in g.get("plays") or []:
        if p.get("typeDescKey") != "goal":
            continue
        if str(p.get("situationCode")) != "1551":   # strict 5v5, goalies in
            continue
        d = p.get("details") or {}
        rows.append({"game_id": gid, "season": season, "game_type": "regular",
                     "scorer": d.get("scoringPlayerId"),
                     "assist1": d.get("assist1PlayerId")})
    return season, rows


def main():
    files = sorted(glob.glob(os.path.join(PBP_DIR, "*.json")))
    if not files:
        print(f"[elites_scoring] no raw PBP in {PBP_DIR} -- nothing to add "
              f"(kept committed parquets).", flush=True)
        return
    existing = {}
    for pq in glob.glob(os.path.join(OUT_DIR, "*.parquet")):
        s = os.path.basename(pq)[:-8]
        try:
            existing[s] = set(pd.read_parquet(pq, columns=["game_id"])["game_id"].unique())
        except Exception:
            existing[s] = set()
    new_by_season = {}
    added = 0
    for path in files:
        gid = int(os.path.basename(path)[:-5])
        yr = int(str(gid)[:4])
        season_key = f"{yr}{yr + 1}"
        if gid in existing.get(season_key, set()):
            continue
        try:
            season, rows = extract(path)
        except Exception as ex:
            print(f"  !! {os.path.basename(path)}: {ex}", flush=True)
            continue
        if season is None:                          # non-regular-season game
            continue
        if gid in existing.get(season, set()):
            continue
        new_by_season.setdefault(season, []).extend(rows)
        added += 1
        if added % 2000 == 0:
            print(f"  extracted {added} new games", flush=True)
    if not new_by_season:
        print("[elites_scoring] no new games to add.", flush=True)
        return
    for season, rows in new_by_season.items():
        pq = os.path.join(OUT_DIR, f"{season}.parquet")
        df_new = pd.DataFrame(rows)
        if os.path.exists(pq):
            old = pd.read_parquet(pq)
            df = pd.concat([old, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=["game_id", "scorer", "assist1"], keep="last")
        else:
            df = df_new
        df = df.sort_values(["game_id"]).reset_index(drop=True)
        df.to_parquet(pq, index=False)
        print(f"  wrote {season}: +{df_new['game_id'].nunique()} games "
              f"({len(df):,} goals total) -> {pq}", flush=True)


if __name__ == "__main__":
    main()
