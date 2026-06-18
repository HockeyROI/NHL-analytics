#!/usr/bin/env python3
"""Build team-level Zone Impact (NZI/OZI/DZI + composite) for the Teams tab.

Materializes a CSV (NFI/output/team_zone.csv) so the Streamlit app reads it
directly instead of recomputing on the fly. Reproduces the VERIFIED published
team-zone aggregation — the 4yr composite gives CAR 8.13, EDM 8.09, VGK 8.07,
MIN 7.99, UTA 7.96 (matches the published post).

Method (per window):
  team_score[metric] = TOI-weighted mean of player raw_score, forwards+defense
                       pooled (raw scores are already position-normalized 0-10).
  composite          = mean(NZI, OZI, DZI) of the team scores.
  TOI weight         = player_fully_adjusted.toi_min summed by player_name over
                       the window's seasons (the canonical TOI source; the 4yr
                       player-zone files carry no TOI column).
  Players with no/zero TOI are dropped from the weighted mean (a handful of
  call-ups outside the NFI dataset), exactly as the published reproduction did.

Windows (no raw single-season team zone — 2024-25 single-season is hit-zoneCode
distorted; the Teams tab maps single seasons onto these pools):
  4y_pool : publication_filtered/4yr_{metric}_{pos}.csv ; TOI over 2022-26
  2y_2426 : per_season/2yr_recent_{metric}_{pos}.csv    ; TOI over 2024-25+2025-26
            (uncapped per-season "2yr_recent" files — the same player-2yr-zone
             set used by the Player tab; publication_filtered/2yr_* also exist
             but are the GP-strict subset, not used here.)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/ashgarg/Documents/HockeyROI")
ADJ = ROOT / "Zones" / "adjusted_rankings"
NFI_PLAYER = ROOT / "NFI" / "output" / "fully_adjusted" / "player_fully_adjusted.csv"
OUT_CSV = ROOT / "NFI" / "output" / "team_zone.csv"

METRICS = ["NZI", "OZI", "DZI"]

WINDOWS = {
    "4y_pool": {
        "dir": ADJ / "publication_filtered",
        "tmpl": "4yr_{metric}_{pos}.csv",
        "seasons": ["20222023", "20232024", "20242025", "20252026"],
    },
    "2y_2426": {
        "dir": ADJ / "per_season",
        "tmpl": "2yr_recent_{metric}_{pos}.csv",
        "seasons": ["20242025", "20252026"],
    },
}


def load_toi(seasons):
    """player_name -> summed toi_min over the given seasons."""
    nfi = pd.read_csv(NFI_PLAYER, dtype={"season": str})
    return nfi[nfi["season"].isin(seasons)].groupby("player_name")["toi_min"].sum()


def team_metric(window_dir, tmpl, metric, toi):
    """TOI-weighted team mean of one metric (forwards + defense pooled)."""
    frames = []
    for pos in ("forwards", "defense"):
        fp = window_dir / tmpl.format(metric=metric, pos=pos)
        if not fp.exists():
            raise FileNotFoundError(fp)
        d = pd.read_csv(fp)
        frames.append(d[["player_name", "team", "raw_score"]])
    a = pd.concat(frames, ignore_index=True)
    a["w"] = a["player_name"].map(toi)
    a = a.dropna(subset=["w", "raw_score"])
    a = a[a["w"] > 0]
    return (a.groupby("team")
             .apply(lambda x: np.average(x["raw_score"], weights=x["w"]),
                    include_groups=False)
             .rename(metric))


def build_window(window, cfg):
    toi = load_toi(cfg["seasons"])
    per = [team_metric(cfg["dir"], cfg["tmpl"], m, toi) for m in METRICS]
    comp = pd.concat(per, axis=1)
    comp["composite"] = comp[METRICS].mean(axis=1)
    comp = comp.reset_index()
    for col in METRICS + ["composite"]:
        comp[f"{col}_rank"] = comp[col].rank(ascending=False, method="min").astype(int)
    comp.insert(0, "window", window)
    return comp


def main():
    # ----- input existence checks -----
    if not NFI_PLAYER.exists():
        sys.exit(f"MISSING input: {NFI_PLAYER}")
    for window, cfg in WINDOWS.items():
        for metric in METRICS:
            for pos in ("forwards", "defense"):
                fp = cfg["dir"] / cfg["tmpl"].format(metric=metric, pos=pos)
                if not fp.exists():
                    sys.exit(f"MISSING input ({window}): {fp}")
    print("All input files present.\n")

    out_rows = []
    for window, cfg in WINDOWS.items():
        comp = build_window(window, cfg)
        out_rows.append(comp)
        top = comp.sort_values("composite", ascending=False).head(5)
        print(f"[{window}] {len(comp)} teams · TOI seasons {cfg['seasons']}")
        print(f"   files: {cfg['tmpl']}")
        for _, r in top.iterrows():
            print(f"   #{int(r['composite_rank'])}  {r['team']:<4} "
                  f"composite={r['composite']:.2f} "
                  f"(NZI {r['NZI']:.2f}  OZI {r['OZI']:.2f}  DZI {r['DZI']:.2f})")
        print()

    out = pd.concat(out_rows, ignore_index=True)
    cols = ["window", "team", "NZI", "OZI", "DZI", "composite",
            "NZI_rank", "OZI_rank", "DZI_rank", "composite_rank"]
    out = out[cols].sort_values(["window", "composite_rank"]).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows ({out['window'].nunique()} windows) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
