#!/usr/bin/env python3
"""Playoff team Zone Impact — playoff companion to build_team_zone.py.

TOI-weighted team NZI/OZI/DZI + composite from the playoff PLAYER zone scores
(Zones/output/playoffs/tnzi_adjusted_{forwards,defense}_playoffs.csv, which
already carry per-player NZI/DZI/OZI per playoff season + an all_playoffs row).
TOI weights from player_fully_adjusted_playoffs.csv (playoff toi_min), joined by
player_name. Forwards + defense pooled at team level.

Windows = each playoff season (2022-25) + all_playoffs (the pooled view the app
shows). No floors. Output: NFI/output/team_zone_playoffs.csv
"""
import os
import numpy as np
import pandas as pd

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
ZP = f"{ROOT}/Zones/output/playoffs"
NFI_PLAYER = f"{ROOT}/NFI/output/fully_adjusted/player_fully_adjusted_playoffs.csv"
OUT_FP = f"{ROOT}/NFI/output/team_zone_playoffs.csv"
METRICS = ["NZI", "OZI", "DZI"]

# Player zone (name-keyed) per playoff season + all_playoffs.
zf = pd.read_csv(f"{ZP}/tnzi_adjusted_forwards_playoffs.csv")
zd = pd.read_csv(f"{ZP}/tnzi_adjusted_defense_playoffs.csv")
zone = pd.concat([zf, zd], ignore_index=True)
zone["season"] = zone["season"].astype(str)

# Playoff TOI per (player_name, season-scope) for weighting.
nfi = pd.read_csv(NFI_PLAYER)
nfi["season"] = nfi["season"].astype(str)
toi = (nfi.groupby(["player_name", "season"])["toi_min"].sum().reset_index()
       .rename(columns={"toi_min": "w"}))

windows = sorted(zone["season"].unique())  # per-season + all_playoffs
rows = []
for win in windows:
    zw = zone[zone["season"] == win].copy()
    zw = zw.merge(toi[toi["season"] == win][["player_name", "w"]], on="player_name", how="left")
    # Players missing TOI: fall back to GP as the weight (so they still count).
    zw["w"] = zw["w"].fillna(zw.get("GP")).fillna(0)
    zw = zw[zw["w"] > 0]
    if zw.empty:
        continue
    agg = {}
    for m in METRICS:
        sub = zw[["team", m, "w"]].dropna(subset=[m])  # skip players with no score
        sub = sub[sub["w"] > 0]
        g = sub.groupby("team").apply(
            lambda x: np.average(x[m], weights=x["w"]), include_groups=False).rename(m)
        agg[m] = g
    comp = pd.concat(agg.values(), axis=1)
    comp["composite"] = comp[METRICS].mean(axis=1)
    comp = comp.reset_index()
    for c in METRICS + ["composite"]:
        comp[f"{c}_rank"] = comp[c].rank(ascending=False, method="min").astype("Int64")
    comp.insert(0, "window", win)
    rows.append(comp)
    if win == "all_playoffs":
        print(f"[{win}] {len(comp)} teams. composite top-5:")
        top = comp.dropna(subset=["composite"]).sort_values("composite", ascending=False).head(5)
        for _, r in top.iterrows():
            print(f"   #{int(r['composite_rank'])} {r['team']:<4} comp={r['composite']:.2f} "
                  f"(NZI {r['NZI']:.2f} OZI {r['OZI']:.2f} DZI {r['DZI']:.2f})")

out = pd.concat(rows, ignore_index=True)[
    ["window", "team", "NZI", "OZI", "DZI", "composite",
     "NZI_rank", "OZI_rank", "DZI_rank", "composite_rank"]]
out = out.sort_values(["window", "composite_rank"]).reset_index(drop=True)
out.to_csv(OUT_FP, index=False)
print(f"\nWrote {OUT_FP} — {len(out)} rows, windows {sorted(out['window'].unique())}")
