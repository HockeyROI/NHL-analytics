"""
QNFS_pct vs goalie frame size.

  - Height/weight from goalie_frame_gsax_2023now.csv
  - QNFS_pct from qnfs_2022-2026.csv (filtered to qualified == True)
  - bucket label from goalie_archetypes_4bucket.csv (joined on goalie_id)

Frame <-> QNFS are merged on goalie name via fuzzy matching (normalized name,
difflib fallback). Three correlations (height, weight, weight/height ratio) are
run against QNFS_pct, mirroring the prior frame-analysis summary-table format.
"""

import difflib
import os

import pandas as pd

from goalie_nfi_size_analysis import normalize_name
from goalie_frame_2023now_analysis import run_three

BASE      = "/Users/ashgarg/Documents/HockeyROI"
FRAME     = f"{BASE}/Goalies/sizes/output/goalie_frame_gsax_2023now.csv"
QNFS      = f"{BASE}/NFI/goalie_consistency/output/qnfs_2022-2026.csv"
ARCHE     = f"{BASE}/NFI/goalie_consistency/output/goalie_archetypes_4bucket.csv"
OUT       = f"{BASE}/Goalies/sizes/output/goalie_qnfs_size.csv"


def fuzzy_merge(frame, qnfs):
    """Match frame.display_name -> qnfs.goalie_name. Returns merged DataFrame."""
    frame = frame.copy()
    qnfs = qnfs.copy()
    frame["_nm"] = frame["display_name"].map(normalize_name)
    qnfs["_nm"] = qnfs["goalie_name"].map(normalize_name)

    qnfs_by_nm = {nm: i for i, nm in zip(qnfs.index, qnfs["_nm"])}
    qnfs_names = list(qnfs["_nm"])

    rows, unmatched = [], []
    for _, fr in frame.iterrows():
        nm = fr["_nm"]
        score = 1.0
        idx = qnfs_by_nm.get(nm)
        if idx is None:  # fuzzy fallback
            cand = difflib.get_close_matches(nm, qnfs_names, n=1, cutoff=0.84)
            if cand:
                idx = qnfs_by_nm[cand[0]]
                score = difflib.SequenceMatcher(None, nm, cand[0]).ratio()
        if idx is None:
            unmatched.append(fr["display_name"])
            continue
        q = qnfs.loc[idx]
        rows.append({
            "goalie_name": q["goalie_name"],
            "display_name": fr["display_name"],
            "goalie_id": q["goalie_id"],
            "QNFS_pct": q["QNFS_pct"],
            "QNFS_lo": q.get("QNFS_lo"),
            "QNFS_hi": q.get("QNFS_hi"),
            "GP": q.get("GP"),
            "quality_games": q.get("quality_games"),
            "height_in": fr["height_in"],
            "weight_lbs": fr["weight_lbs"],
            "weight_height_ratio": fr["weight_height_ratio"],
            "match_score": round(score, 3),
        })
    if unmatched:
        print(f"  Unmatched frame goalies ({len(unmatched)}): {unmatched}")
    return pd.DataFrame(rows)


def main():
    frame = pd.read_csv(FRAME)
    qnfs = pd.read_csv(QNFS)
    arche = pd.read_csv(ARCHE)

    qnfs = qnfs[qnfs["qualified"] == True].copy()
    print(f"Frame goalies: {len(frame)}")
    print(f"QNFS qualified==True: {len(qnfs)}")

    merged = fuzzy_merge(frame, qnfs)
    print(f"Merged (frame n QNFS): {len(merged)}")

    # bucket label via goalie_id (exact join)
    merged = merged.merge(arche[["goalie_id", "bucket"]], on="goalie_id", how="left")
    merged["bucket"] = merged["bucket"].fillna("(no bucket)")

    merged = merged.sort_values("QNFS_pct", ascending=False).reset_index(drop=True)

    # report any non-exact fuzzy matches for transparency
    fuzzy = merged[merged["match_score"] < 1.0]
    if len(fuzzy):
        print("  Fuzzy (non-exact) matches:")
        for _, r in fuzzy.iterrows():
            print(f"    {r['display_name']!r} -> {r['goalie_name']!r} (score {r['match_score']})")

    run_three(merged, "QNFS_pct", "goalie_name", ["bucket"], "QNFS_pct vs FRAME SIZE (qualified goalies)")

    out_cols = ["goalie_name", "bucket", "height_in", "weight_lbs", "weight_height_ratio",
                "QNFS_pct", "QNFS_lo", "QNFS_hi", "GP", "quality_games",
                "goalie_id", "display_name", "match_score"]
    merged[out_cols].to_csv(OUT, index=False)
    print(f"\n  Saved -> {OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
