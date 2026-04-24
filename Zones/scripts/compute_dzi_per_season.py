"""Compute per-(season, team) DZI Fenwick (V2) team-average scores.

Reuses process_game/build_scenario_rows/compute_metric from compute_zone_variations.
Writes: Zones/zone_variations/DZI_V2_team_per_season.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent))
from compute_zone_variations import (  # type: ignore
    PLAYER_META, OUT_DIR, MIN_GP, VERSIONS, METRICS, IS_LOST_ONLY,
    POS_FORWARD, POS_DEFENSE, POOLED_SEASONS, norm_team,
    process_game, build_scenario_rows, compute_metric, normalize_to_score,
    load_game_ids,
)

TARGET_METRIC = "DZI"
TARGET_VERSION = "V2"


def main():
    print("[1/4] loading player meta ...")
    player_meta = {int(k): v for k, v in json.load(open(PLAYER_META)).items()}

    print("[2/4] reading game_ids.csv ...")
    games = load_game_ids()
    print(f"    {len(games)} regular games across {len(POOLED_SEASONS)} seasons")

    print("[3/4] processing raw PBP + shifts ...")
    player_bucket = defaultdict(
        lambda: {"shifts": 0, "lost_shifts": 0, "total_sec": 0.0,
                 "oz_sec": 0.0, "dz_sec": 0.0, "nz_sec": 0.0})
    player_season_gp = defaultdict(int)

    total = len(games)
    for i, (gid, season) in enumerate(games):
        process_game(gid, player_season_gp, player_bucket)
        if (i + 1) % 500 == 0 or i + 1 == total:
            print(f"    processed {i+1}/{total}")

    print("[4/4] computing per-season DZI V2 team averages ...")
    vers = TARGET_VERSION
    metric = TARGET_METRIC
    n_kind = "lost_shifts" if IS_LOST_ONLY[vers] else "shifts"

    out_rows = []  # (season, team, dzi_mean, n_players)
    for season in sorted(POOLED_SEASONS):
        bundle = build_scenario_rows(
            player_bucket, player_season_gp, player_meta, {season})

        # Compute per-player metric
        per_player = {}  # pid -> adj
        for pid, data in bundle.items():
            meta = player_meta.get(pid, {})
            pos = (meta.get("position") or "").upper()
            if pos in ("G", ""):
                continue
            if data["gp"] < MIN_GP:
                continue
            vbuckets = data["versions"].get(vers, {})
            b_o = vbuckets.get("O"); b_d = vbuckets.get("D"); b_n = vbuckets.get("N")
            raw, adj = compute_metric(b_o, b_d, b_n, metric, n_kind)
            if adj is None:
                continue
            per_player[pid] = adj

        # Normalize per position group (matches compute_zone_variations output
        # for team-level averaging semantics), then average F+D scores per team.
        team_scores = defaultdict(list)
        for pos_set in (POS_FORWARD, POS_DEFENSE):
            subset = [(pid, adj) for pid, adj in per_player.items()
                      if (player_meta.get(pid, {}).get("position") or "").upper() in pos_set]
            scores = normalize_to_score(subset)
            for pid, sc in scores.items():
                if sc is None:
                    continue
                team = norm_team(player_meta.get(pid, {}).get("team_abbrev", "") or "")
                if not team:
                    continue
                team_scores[team].append(sc)

        for team, scs in team_scores.items():
            if not scs:
                continue
            out_rows.append({
                "season": season,
                "team": team,
                "DZI_Fenwick_team_avg": mean(scs),
                "n_players": len(scs),
            })

    out_path = OUT_DIR / "DZI_V2_team_per_season.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["season", "team", "DZI_Fenwick_team_avg", "n_players"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"    wrote {len(out_rows)} team-season rows -> {out_path}")


if __name__ == "__main__":
    main()
