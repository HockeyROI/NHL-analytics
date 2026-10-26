"""build_zone_index_per_game.py -- per-game OZI / DZI / NZI / TZI on the season's
0-100 scale, for the app's game log.

The season index (build_zone_index100.py) is, per player and metric,
    index = clip(50 + raw_pct - avg_pct, 0, 100)
where raw_pct is the player's share of post-faceoff on-ice time spent in the
offensive zone (V1 events, faceoff-anchored contexts, compute_zone_variations
methodology) and avg_pct is the mean raw_pct of that season's qualifying
players in the same position group (MIN_GP / MIN_SHIFTS gates).

Per game we take the SAME raw share from that game's shifts only and place it on
the SAME season scale (same avg_pct), so a game value of 60 means "10 points
better than the season's average player", directly comparable with the season
number. No shift minimum per game (a game has only a handful of faceoff-started
shifts per zone) -- n_* columns carry the shift counts so small samples are
visible; a metric is blank when the player had no shift of that kind.

  OZI  offensive-zone time share after OFFENSIVE-zone faceoffs
  DZI  offensive-zone time share after DEFENSIVE-zone faceoffs
  NZI  offensive-zone time share after NEUTRAL-zone faceoffs
  TZI  (OZ share - DZ share) after neutral-zone faceoffs (transition)

Output: Zones/output/zone_index_by_game/{season}.parquet
  player_id, game_id, OZI, DZI, NZI, TZI, n_oz, n_dz, n_nz
Needs the local raw PBP + shift cache (Zones/raw), like the season index -- run
it in the local refresh after build_zone_index100.py.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import pandas as pd

import compute_zone_variations as zv
import build_zone_index100 as zi

SEASONS = ["20222023", "20232024", "20242025", "20252026", "20262027"]
LIVE = "20262027"
OUT_DIR = zv.ZONES / "output" / "zone_index_by_game"
# metric -> (faceoff zone bucket, numerator key)
RAW = {"OZI": ("O", "oz_sec"), "DZI": ("D", "oz_sec"), "NZI": ("N", "oz_sec")}


def game_raw(b):
    """Per-game raw shares (0-1) from one player's V1 buckets {fo_zone: bucket}."""
    out = {}
    for m, (fz, key) in RAW.items():
        bb = b.get(fz)
        out[m] = (bb[key] / bb["total_sec"]) if bb and bb["total_sec"] > 0 else None
    bn = b.get("N")
    out["TZI"] = ((bn["oz_sec"] - bn["dz_sec"]) / bn["total_sec"]
                  if bn and bn["total_sec"] > 0 else None)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {int(k): v for k, v in json.load(open(zv.PLAYER_META)).items()}

    def grp_of(pid):
        pos = (meta.get(pid, {}).get("position") or "").upper()
        return ("forwards" if pos in zv.POS_FORWARD else
                "defense" if pos in zv.POS_DEFENSE else None)

    games = defaultdict(list)
    with open(zv.GAME_IDS) as f:
        for r in csv.DictReader(f):
            if r["season"] in SEASONS and r["game_type"] == "regular":
                games[r["season"]].append(int(r["game_id"]))

    for season in SEASONS:
        season_bucket = defaultdict(lambda: {"shifts": 0, "lost_shifts": 0, "total_sec": 0.0,
                                             "oz_sec": 0.0, "dz_sec": 0.0, "nz_sec": 0.0})
        season_gp = defaultdict(int)
        rows = []
        for gid in games[season]:
            gb = defaultdict(lambda: {"shifts": 0, "lost_shifts": 0, "total_sec": 0.0,
                                      "oz_sec": 0.0, "dz_sec": 0.0, "nz_sec": 0.0})
            ggp = defaultdict(int)
            zv.process_game(gid, ggp, gb)
            if not ggp:
                continue                       # unplayed / missing raw files
            for k, v in ggp.items():
                season_gp[k] += v
            per_pid = defaultdict(dict)
            for (pid, ssn, vers, fz), b in gb.items():
                sb = season_bucket[(pid, ssn, vers, fz)]
                for f in sb:
                    sb[f] += b.get(f, 0)
                if vers == "V1":
                    per_pid[pid][fz] = b
            for pid, b in per_pid.items():
                r = game_raw(b)
                rows.append({"player_id": pid, "game_id": gid, **r,
                             "n_oz": b.get("O", {}).get("shifts", 0),
                             "n_dz": b.get("D", {}).get("shifts", 0),
                             "n_nz": b.get("N", {}).get("shifts", 0)})
        if not rows:
            continue
        # Season averages exactly as build_zone_index100 computes them.
        bundle = zv.build_scenario_rows(season_bucket, season_gp, meta, {season})
        min_gp = 1 if season == LIVE else zv.MIN_GP
        raw = {g: defaultdict(list) for g in ("forwards", "defense")}
        for pid, data in bundle.items():
            g = grp_of(pid)
            if g is None or data["gp"] < min_gp:
                continue
            for disp, internal in zi.METRICS:
                rv = zi.raw_for_player(data, internal)
                if rv is not None:
                    raw[g][disp].append(rv * 100.0)
        avg = {(g, m): sum(v) / len(v) for g in raw for m, v in raw[g].items() if v}
        df = pd.DataFrame(rows)
        df["_g"] = df["player_id"].map(grp_of)
        df = df.dropna(subset=["_g"])
        for m in ("OZI", "DZI", "NZI", "TZI"):
            a = df["_g"].map(lambda g: avg.get((g, m)))
            df[m] = (50.0 + df[m].astype(float) * 100.0 - a).clip(0, 100).round(1)
        df = df.drop(columns=["_g"]).sort_values(["game_id", "player_id"])
        df.to_parquet(OUT_DIR / f"{season}.parquet", index=False)
        print(f"[zone-per-game] {season}: {df['game_id'].nunique()} games, "
              f"{len(df):,} player-games; season avgs "
              + ", ".join(f"{g[0]}/{m}={v:.1f}" for (g, m), v in sorted(avg.items())))


if __name__ == "__main__":
    main()
