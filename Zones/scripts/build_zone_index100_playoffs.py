"""Playoff companion to build_zone_index100.py — builds the 0-100 Zone-Impact
index (50 = position-group average) pooled across ALL playoff games, for
OZI / DZI / NZI / TZI. Same V1 methodology (imported from
compute_zone_variations.py); only the game set differs (game_type == 'playoff').

Writes:
  Zones/adjusted_rankings/zone_index100/playoffs_{forwards|defense}.csv
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import compute_zone_variations as zv
import build_zone_index100 as base   # reuse build_index / write_scope machinery


def load_playoff_game_ids():
    out = []
    with open(zv.GAME_IDS) as f:
        for r in csv.DictReader(f):
            if r["game_type"] == "playoff":
                out.append((int(r["game_id"]), r["season"]))
    return out


def main():
    base.OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] loading player meta ...")
    player_meta = {int(k): v for k, v in json.load(open(zv.PLAYER_META)).items()}

    print("[2/3] processing playoff PBP + shifts ...")
    games = load_playoff_game_ids()
    player_bucket = defaultdict(
        lambda: {"shifts": 0, "lost_shifts": 0, "total_sec": 0.0,
                 "oz_sec": 0.0, "dz_sec": 0.0, "nz_sec": 0.0})
    player_season_gp = defaultdict(int)
    seasons = set()
    for gid, season in games:
        seasons.add(season)
        zv.process_game(gid, player_season_gp, player_bucket)
    print(f"    {len(games)} playoff games across {len(seasons)} seasons")

    print("[3/3] building + writing pooled playoff index ...")
    bundle = zv.build_scenario_rows(player_bucket, player_season_gp,
                                    player_meta, seasons)
    # Playoff samples are short (median ~11 GP), so don't apply the 20-GP regular
    # floor — the 50-shift gate in compute_metric qualifies players instead.
    base.write_scope("playoffs", bundle, player_meta, min_gp=1)
    print("done.")


if __name__ == "__main__":
    main()
