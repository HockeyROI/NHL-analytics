#!/usr/bin/env python3
"""Rebuild the pooled goalie GSAx file from the per-season file.

Single source of truth — the per-season pipeline uses CNFI + MNFI only
(FNFI dropped per framework update). Aggregating up keeps the Pooled view
and per-season views in the Streamlit app measuring the same thing.

Source : NFI/output/goalie_nfi_gsax_by_season.csv  (script 21)
Aux    : NFI/output/player_toi.csv                 (pooled ES TOI per goalie)
Output : NFI/output/goalie_nfi_gsax_pooled_v2.csv  (NEW — does not overwrite
         the original goalie_nfi_gsax.csv this run)

Aggregation per goalie_id:
  goalie_name  → most recent non-null value (latest season wins)
  team         → modal team across all season rows (most frequent)
  games        → sum across seasons
  total_faced  → sum across seasons
  GSAx         → sum across seasons (additive — same denominator throughout)
  GSAx_per60   → sum(GSAx) / pooled_es_toi_min * 60
                 where pooled_es_toi_min comes from player_toi.csv. If a
                 goalie isn't in player_toi.csv, fall back to back-deriving
                 TOI from the per-season GSAx_per60 (TOI = GSAx / per60 * 60
                 per row, then summed).

Qualifying threshold: pooled total_faced >= 300 (matches the old script 20
threshold so the v2 file is comparable to the original).
"""
import os
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
OUT = f"{ROOT}/NFI/output"
SRC_FP   = f"{OUT}/goalie_nfi_gsax_by_season.csv"
TOI_FP   = f"{OUT}/player_toi.csv"
ORIG_FP  = f"{OUT}/goalie_nfi_gsax.csv"     # for comparison only — read-only
OUT_FP   = f"{OUT}/goalie_nfi_gsax_pooled_v2.csv"

MIN_TOTAL_FACED = 300


def main() -> None:
    season_df = pd.read_csv(SRC_FP)
    season_df["season"] = season_df["season"].astype(int)
    print(f"Source: {SRC_FP}  rows={len(season_df)}  "
          f"goalies={season_df['goalie_id'].nunique()}  "
          f"seasons={sorted(season_df['season'].unique().tolist())}")

    toi = pd.read_csv(TOI_FP)
    pooled_toi_map = (
        toi[toi["position"] == "G"]
            .set_index("player_id")["toi_ES_sec"].mul(1.0 / 60.0).to_dict()
    )

    # ---------------- Aggregate per goalie ----------------
    rows = []
    for gid, g in season_df.groupby("goalie_id"):
        g_sorted = g.sort_values("season")
        recent_name = (
            g_sorted["goalie_name"].dropna().iloc[-1]
            if g_sorted["goalie_name"].notna().any() else ""
        )

        team_series = g_sorted["team"].dropna()
        if team_series.empty:
            modal_team = ""
        else:
            counts = Counter(team_series.tolist())
            # modal team = most frequent; tie → most recent season's team
            top_count = max(counts.values())
            tied = [t for t, c in counts.items() if c == top_count]
            if len(tied) == 1:
                modal_team = tied[0]
            else:
                modal_team = team_series.iloc[-1]

        games_sum  = int(g_sorted["games"].fillna(0).sum())
        faced_sum  = float(g_sorted["total_faced"].fillna(0).sum())
        gsax_sum   = float(g_sorted["GSAx"].fillna(0).sum())

        # Pooled TOI: prefer player_toi.csv. Fall back to back-deriving from
        # per-season rows (TOI_min_row = GSAx_row / GSAx_per60_row * 60).
        pooled_toi_min = pooled_toi_map.get(gid)
        if pooled_toi_min is None or pooled_toi_min <= 0 or np.isnan(pooled_toi_min):
            # Back-derive: skip rows with 0 / NaN per60 (would imply 0 GSAx
            # exactly, which is rare; just sum the well-defined ones).
            mask = (g_sorted["GSAx_per60"].abs() > 1e-9) & g_sorted["GSAx"].notna()
            if mask.any():
                derived = (
                    g_sorted.loc[mask, "GSAx"]
                    / g_sorted.loc[mask, "GSAx_per60"] * 60.0
                )
                pooled_toi_min = float(derived.sum())
            else:
                pooled_toi_min = float("nan")

        gsax_per60 = (
            gsax_sum / pooled_toi_min * 60.0
            if pooled_toi_min and pooled_toi_min > 0 else float("nan")
        )

        rows.append({
            "goalie_id":    int(gid),
            "goalie_name":  recent_name,
            "team":         modal_team,
            "games":        games_sum,
            "total_faced":  faced_sum,
            "GSAx":         round(gsax_sum, 2),
            "GSAx_per60":   round(gsax_per60, 3) if pd.notna(gsax_per60) else np.nan,
            "es_toi_min":   round(pooled_toi_min, 2) if pd.notna(pooled_toi_min) else np.nan,
            "n_seasons":    int(g_sorted["season"].nunique()),
        })

    pooled = pd.DataFrame(rows)
    print(f"\nPre-threshold pooled goalies: {len(pooled)}")
    pooled = pooled[pooled["total_faced"] >= MIN_TOTAL_FACED].copy()
    pooled = pooled.sort_values("GSAx", ascending=False).reset_index(drop=True)
    print(f"Post-threshold (faced >= {MIN_TOTAL_FACED}): {len(pooled)}")

    # ---------------- Save ----------------
    keep_cols = ["goalie_id", "goalie_name", "team", "n_seasons",
                 "games", "total_faced", "es_toi_min",
                 "GSAx", "GSAx_per60"]
    pooled[keep_cols].to_csv(OUT_FP, index=False)
    print(f"\nWrote {OUT_FP} — shape {pooled.shape}")

    # ---------------- Top 10 by GSAx_per60 (faced ≥ 1000 to filter noise) ----------------
    print("\n=== TOP 10 by GSAx_per60 (pooled v2, total_faced ≥ 1000) ===")
    top_per60 = (
        pooled[pooled["total_faced"] >= 1000]
            .nlargest(10, "GSAx_per60")
            [["goalie_name", "team", "n_seasons", "games",
              "total_faced", "GSAx", "GSAx_per60"]]
    )
    print(top_per60.to_string(index=False))

    # ---------------- Locate Shesterkin / Sorokin ----------------
    print("\n=== Shesterkin & Sorokin in pooled v2 ===")
    pooled_per60_rank = (
        pooled[pooled["total_faced"] >= 1000]
            .sort_values("GSAx_per60", ascending=False)
            .reset_index(drop=True)
    )
    pooled_per60_rank["rank_per60"] = pooled_per60_rank.index + 1
    for name in ("Shesterkin", "Sorokin"):
        rows = pooled_per60_rank[
            pooled_per60_rank["goalie_name"].str.contains(name, na=False)
        ]
        if rows.empty:
            print(f"  {name}: not in pooled (faced ≥ 1000) view")
        else:
            r = rows.iloc[0]
            print(f"  {name}: rank #{int(r['rank_per60'])}  "
                  f"team={r['team']}  faced={r['total_faced']:.0f}  "
                  f"GSAx={r['GSAx']:+.2f}  per60={r['GSAx_per60']:+.3f}")

    # ---------------- Comparison to original goalie_nfi_gsax.csv ----------------
    # The legacy goalie_nfi_gsax.csv was retired in April 2026; this block
    # is a one-time-migration sanity check kept for reference. If the legacy
    # file is no longer on disk, skip the comparison silently.
    print("\n=== Top 10 — new pooled v2 (CNFI+MNFI, min 300) by GSAx ===")
    new_top = (
        pooled.sort_values("GSAx", ascending=False).head(10)
            [["goalie_name", "team", "total_faced", "GSAx"]]
            .reset_index(drop=True)
    )
    new_top.index = new_top.index + 1
    print(new_top.to_string())

    try:
        print("\n=== Top 10 — original goalie_nfi_gsax.csv (CNFI+MNFI+FNFI, "
              "pooled, min 300) ===")
        orig = pd.read_csv(ORIG_FP)
        print("Original columns:", orig.columns.tolist())
        orig_top = (
            orig.sort_values("NFI_GSAx_calibrated", ascending=False).head(10)
                [["goalie_name", "total_faced", "NFI_GSAx_calibrated"]]
                .reset_index(drop=True)
        )
        orig_top.index = orig_top.index + 1
        print(orig_top.to_string())

        # ---------------- Diff: who moved most ----------------
        cmp = orig[["goalie_name", "total_faced", "NFI_GSAx_calibrated"]].copy()
        cmp = cmp.rename(columns={"NFI_GSAx_calibrated": "GSAx_orig",
                                  "total_faced": "faced_orig"})
        cmp = cmp.merge(
            pooled[["goalie_name", "team", "total_faced", "GSAx"]]
                  .rename(columns={"total_faced": "faced_v2", "GSAx": "GSAx_v2"}),
            on="goalie_name", how="outer"
        )
        cmp["delta"] = cmp["GSAx_v2"] - cmp["GSAx_orig"]
        print("\n=== Largest |Δ| in GSAx vs original (top 10) ===")
        print(cmp.dropna(subset=["GSAx_orig", "GSAx_v2"])
                  .reindex(cmp.dropna(subset=["GSAx_orig", "GSAx_v2"])["delta"]
                           .abs().sort_values(ascending=False).index)
                  .head(10)
                  [["goalie_name", "team", "faced_orig", "faced_v2",
                    "GSAx_orig", "GSAx_v2", "delta"]]
                  .to_string(index=False))
        print(f"\nFiles touched (read-only): {SRC_FP}, {TOI_FP}, {ORIG_FP}")
        print(f"Files written: {OUT_FP}")
        print("Original goalie_nfi_gsax.csv NOT modified.")
    except FileNotFoundError:
        print(f"\n[skip comparison] legacy {ORIG_FP} not on disk — "
              f"the v2 file is now the canonical pooled goalie source.")
        print(f"\nFiles touched (read-only): {SRC_FP}, {TOI_FP}")
        print(f"Files written: {OUT_FP}")


if __name__ == "__main__":
    main()
