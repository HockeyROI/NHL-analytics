#!/usr/bin/env python3
"""Committable per-season parquet copies of the shot events, for the app.

The raw Data/nhl_shot_events.csv (~134MB, 29 cols) is gitignored — the deployed
Streamlit app can't read it. This writes ONE parquet PER SEASON with the FULL
column set (~2MB each, ~13MB total for 6 seasons), which the app loads a season
at a time for the shot charts (and any future per-shot feature).

ADDITIVE + NON-DESTRUCTIVE: the raw CSV monolith is left exactly as-is; every
existing pipeline script keeps reading it. These parquets are new files only.

Output: Data/shot_events_by_season/{season}.parquet   (season e.g. "20252026")
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))
SHOT_CSV = ROOT / "Data" / "nhl_shot_events.csv"
OUT_DIR = ROOT / "Data" / "shot_events_by_season"


def main() -> int:
    if not SHOT_CSV.exists():
        print(f"missing {SHOT_CSV}")
        return 2
    print("loading full shot events...")
    df = pd.read_csv(SHOT_CSV, dtype={"season": str, "situation_code": str})
    # shrink a couple of wide dtypes so the parquet stays tiny (values unchanged)
    for c in ("x_coord", "y_coord", "x_coord_norm", "y_coord_norm"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for season, g in df.groupby("season"):
        fp = OUT_DIR / f"{season}.parquet"
        g.to_parquet(fp, index=False)
        mb = fp.stat().st_size / 1e6
        total += mb
        print(f"  {season}: {len(g):,} rows -> {fp.name} ({mb:.1f} MB)")
    print(f"wrote {df['season'].nunique()} season files, {total:.1f} MB total, to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
