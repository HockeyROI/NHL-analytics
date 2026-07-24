#!/usr/bin/env python3
"""Committable per-season parquet copies of the shift data.

Same trick as build_shot_chart_data.py does for shot events. The raw
NFI/Geometry_post/Data/shift_data.csv is ~426MB and gitignored, which is what
stopped the weekly GitHub Action from rebuilding the situation engines: the
runner simply didn't have the file. Split per season and stored as parquet it
is ~3.5MB/season (~21MB total) — small enough to live in the repo, so CI can
rebuild everything unattended.

Keeps only the columns the pipeline actually reads. The dropped ones are
either redundant (first/last name -> player_id) or derivable string forms of
the absolute seconds (start_time/end_time/start_secs/end_secs), and they
dominate the file size: keeping everything costs ~54MB, these 6 cost ~21MB.

ADDITIVE + NON-DESTRUCTIVE: the raw CSV is untouched and still works locally.

Output: Data/shift_data_by_season/{season}.parquet   (season e.g. "20252026")
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))
SHIFT_CSV = ROOT / "NFI" / "Geometry_post" / "Data" / "shift_data.csv"
OUT_DIR = ROOT / "Data" / "shift_data_by_season"
# exactly what build_situation_toi.py / build_situation_onice.py read
KEEP = ["game_id", "player_id", "period", "team_abbrev",
        "abs_start_secs", "abs_end_secs"]


def main() -> int:
    if not SHIFT_CSV.exists():
        print(f"missing {SHIFT_CSV}")
        return 2
    print("loading shift data (large — streaming in chunks)...")
    parts = []
    for ch in pd.read_csv(SHIFT_CSV, usecols=KEEP, chunksize=1_000_000):
        ch = ch.dropna(subset=["game_id"])
        ch["game_id"] = ch["game_id"].astype("int64")
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    del parts
    # season start year is the first 4 digits of the game_id
    df["season"] = (df["game_id"] // 1_000_000).astype(int)
    for c in ("player_id", "period", "abs_start_secs", "abs_end_secs",
              "start_secs", "end_secs"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for yr, g in df.groupby("season"):
        season = f"{int(yr)}{int(yr) + 1}"
        g = g.drop(columns=["season"])
        fp = OUT_DIR / f"{season}.parquet"
        g.to_parquet(fp, index=False)
        mb = fp.stat().st_size / 1e6
        total += mb
        print(f"  {season}: {len(g):,} shifts -> {fp.name} ({mb:.1f} MB)")
    print(f"wrote {df['season'].nunique()} season files, {total:.1f} MB total, to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
