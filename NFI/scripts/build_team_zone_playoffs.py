#!/usr/bin/env python3
"""Playoff team Zone Impact — playoff companion to build_team_zone.py.

TOI-weighted team OZI/DZI/NZI/TZI + composite from the per-player 0-100 playoff
Zone Impact index (Zones/adjusted_rankings/zone_index100/playoffs_{forwards,
defense}.csv — the pooled all-playoffs index, 50 = position-group average, built
by Zones/scripts/build_zone_index100_playoffs.py). TOI weights from
player_fully_adjusted_playoffs.csv (playoff toi_min summed across playoff
seasons), joined by player_name. Forwards + defense pooled at team level.

Only the 'all_playoffs' window is emitted — the only playoff scope the app shows.
No floors. Output: NFI/output/team_zone_playoffs.csv
"""
import os
import numpy as np
import pandas as pd

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
IDX = f"{ROOT}/Zones/adjusted_rankings/zone_index100"
NFI_PLAYER = f"{ROOT}/NFI/output/fully_adjusted/player_fully_adjusted_playoffs.csv"
OUT_FP = f"{ROOT}/NFI/output/team_zone_playoffs.csv"
METRICS = ["OZI", "DZI", "NZI", "TZI"]
COMPOSITE_METRICS = ["OZI", "DZI", "NZI"]   # TZI (a split) kept out of composite

# Per-player 0-100 playoff index (pooled all-playoffs), forwards + defense.
zf = pd.read_csv(f"{IDX}/playoffs_forwards.csv")
zd = pd.read_csv(f"{IDX}/playoffs_defense.csv")
zone = pd.concat([zf, zd], ignore_index=True)

# Playoff TOI per player_name, summed across all playoff seasons, for weighting.
nfi = pd.read_csv(NFI_PLAYER)
toi = nfi.groupby("player_name")["toi_min"].sum().rename("w").reset_index()

zw = zone.merge(toi, on="player_name", how="left")
# Players missing TOI: fall back to GP as the weight (so they still count).
zw["w"] = zw["w"].fillna(zw.get("GP")).fillna(0)
zw = zw[zw["w"] > 0]

agg = {}
for m in METRICS:
    sub = zw[["team", m, "w"]].dropna(subset=[m])  # skip players with no score
    sub = sub[sub["w"] > 0]
    agg[m] = sub.groupby("team").apply(
        lambda x: np.average(x[m], weights=x["w"]), include_groups=False).rename(m)
comp = pd.concat(agg.values(), axis=1)
comp["composite"] = comp[COMPOSITE_METRICS].mean(axis=1)
comp = comp.reset_index()
for c in METRICS + ["composite"]:
    comp[f"{c}_rank"] = comp[c].rank(ascending=False, method="min").astype("Int64")
comp.insert(0, "window", "all_playoffs")

print(f"[all_playoffs] {len(comp)} teams. composite top-5:")
top = comp.dropna(subset=["composite"]).sort_values("composite", ascending=False).head(5)
for _, r in top.iterrows():
    print(f"   #{int(r['composite_rank'])} {r['team']:<4} comp={r['composite']:.1f} "
          f"(OZI {r['OZI']:.1f} DZI {r['DZI']:.1f} NZI {r['NZI']:.1f} TZI {r['TZI']:.1f})")

out = comp[["window", "team", "OZI", "DZI", "NZI", "TZI", "composite",
            "OZI_rank", "DZI_rank", "NZI_rank", "TZI_rank", "composite_rank"]]
out = out.sort_values(["window", "composite_rank"]).reset_index(drop=True)
out.to_csv(OUT_FP, index=False)
print(f"\nWrote {OUT_FP} — {len(out)} rows, windows {sorted(out['window'].unique())}")
