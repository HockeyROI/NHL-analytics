#!/usr/bin/env python3
"""Build team-level Zone Impact (OZI/DZI/NZI/TZI + composite) for the Teams tab.

Materializes a CSV (NFI/output/team_zone.csv) so the Streamlit app reads it
directly. Team scores are the TOI-weighted mean of the per-player 0-100 Zone
Impact index (50 = position-group average, built by
Zones/scripts/build_zone_index100.py). Because each player value is
`50 + (player% - league-avg%)`, a TOI-weighted team mean equals
`50 + (team-TOI-weighted% - league-avg%)` — i.e. the team score is already on
the "50 = league-average team" scale, no extra recentring. A team of exactly
average players scores 50; stronger possession teams sit a few points above.
(Team territorial tilt varies less than an individual's, so the team spread is
naturally tighter than the player spread — the same honest compression a team
save% shows vs an individual's.)

Method (per window):
  team_score[metric] = TOI-weighted mean of the player 0-100 index, forwards +
                       defense pooled.
  composite          = mean(OZI, DZI, NZI) of the team scores (TZI, a split, is
                       reported on its own and NOT folded into the composite).
  TOI weight         = player_fully_adjusted.toi_min summed by player_name over
                       the window's seasons. Players with no/zero TOI are dropped.

Windows (the Teams tab maps single seasons onto these pools):
  4y_pool : zone_index100/pooled_{forwards,defense}.csv ; TOI over 2022-26
  2y_2426 : zone_index100/2yr_{forwards,defense}.csv    ; TOI over 2024-25+2025-26
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/ashgarg/Documents/HockeyROI")
IDX = ROOT / "Zones" / "adjusted_rankings" / "zone_index100"
NFI_PLAYER = ROOT / "NFI" / "output" / "fully_adjusted" / "player_fully_adjusted.csv"
OUT_CSV = ROOT / "NFI" / "output" / "team_zone.csv"

METRICS = ["OZI", "DZI", "NZI", "TZI"]
COMPOSITE_METRICS = ["OZI", "DZI", "NZI"]   # TZI is a split, kept out of composite

WINDOWS = {
    "4y_pool": {"scope": "pooled",
                "seasons": ["20222023", "20232024", "20242025", "20252026"]},
    "2y_2426": {"scope": "2yr",
                "seasons": ["20242025", "20252026"]},
}


def load_toi(seasons):
    """player_name -> summed toi_min over the given seasons."""
    nfi = pd.read_csv(NFI_PLAYER, dtype={"season": str})
    return nfi[nfi["season"].isin(seasons)].groupby("player_name")["toi_min"].sum()


def load_index(scope):
    """Per-player 0-100 index (forwards + defense pooled) for one scope."""
    frames = []
    for pos in ("forwards", "defense"):
        fp = IDX / f"{scope}_{pos}.csv"
        if not fp.exists():
            raise FileNotFoundError(fp)
        frames.append(pd.read_csv(fp))
    return pd.concat(frames, ignore_index=True)


def team_metric(idx, metric, toi):
    """TOI-weighted team mean of one 0-100 metric (skip players missing it)."""
    a = idx[["player_name", "team", metric]].copy()
    a["w"] = a["player_name"].map(toi)
    a = a.dropna(subset=["w", metric])
    a = a[a["w"] > 0]
    return (a.groupby("team")
             .apply(lambda x: np.average(x[metric], weights=x["w"]),
                    include_groups=False)
             .rename(metric))


def build_window(window, cfg):
    toi = load_toi(cfg["seasons"])
    idx = load_index(cfg["scope"])
    per = [team_metric(idx, m, toi) for m in METRICS]
    comp = pd.concat(per, axis=1)
    comp["composite"] = comp[COMPOSITE_METRICS].mean(axis=1)
    comp = comp.reset_index()
    for col in METRICS + ["composite"]:
        comp[f"{col}_rank"] = comp[col].rank(ascending=False, method="min").astype(int)
    comp.insert(0, "window", window)
    return comp


def main():
    if not NFI_PLAYER.exists():
        sys.exit(f"MISSING input: {NFI_PLAYER}")
    for window, cfg in WINDOWS.items():
        for pos in ("forwards", "defense"):
            fp = IDX / f"{cfg['scope']}_{pos}.csv"
            if not fp.exists():
                sys.exit(f"MISSING input ({window}): {fp}")
    print("All input files present.\n")

    out_rows = []
    for window, cfg in WINDOWS.items():
        comp = build_window(window, cfg)
        out_rows.append(comp)
        top = comp.sort_values("composite", ascending=False).head(5)
        print(f"[{window}] {len(comp)} teams · TOI seasons {cfg['seasons']}")
        for _, r in top.iterrows():
            print(f"   #{int(r['composite_rank'])}  {r['team']:<4} "
                  f"composite={r['composite']:.1f} "
                  f"(OZI {r['OZI']:.1f}  DZI {r['DZI']:.1f}  NZI {r['NZI']:.1f}  "
                  f"TZI {r['TZI']:.1f})")
        print()

    out = pd.concat(out_rows, ignore_index=True)
    cols = ["window", "team", "OZI", "DZI", "NZI", "TZI", "composite",
            "OZI_rank", "DZI_rank", "NZI_rank", "TZI_rank", "composite_rank"]
    out = out[cols].sort_values(["window", "composite_rank"]).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows ({out['window'].nunique()} windows) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
