"""Shared readers for the two big raw inputs, so builders run BOTH locally and
on the GitHub runner.

Locally the full CSVs exist (Data/nhl_shot_events.csv ~128MB,
NFI/Geometry_post/Data/shift_data.csv ~426MB) but both are gitignored, so a CI
checkout doesn't have them. The committable per-season parquets
(Data/shot_events_by_season/, Data/shift_data_by_season/) carry the same rows
at ~12MB and ~21MB, which is what lets the weekly Action rebuild everything
unattended.

Prefer the CSV when present (it's the freshest, and local runs regenerate it),
otherwise fall back to the parquets.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))
SHOT_CSV = ROOT / "Data" / "nhl_shot_events.csv"
SHOT_PARQUET_DIR = ROOT / "Data" / "shot_events_by_season"
SHIFT_CSV = ROOT / "NFI" / "Geometry_post" / "Data" / "shift_data.csv"
SHIFT_PARQUET_DIR = ROOT / "Data" / "shift_data_by_season"


def _from_parquets(directory: Path, usecols=None, dtype=None) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquets in {directory}")
    parts = [pd.read_parquet(f, columns=list(usecols) if usecols else None)
             for f in files]
    df = pd.concat(parts, ignore_index=True)
    if dtype:
        for c, t in dtype.items():
            if c in df.columns:
                df[c] = df[c].astype(t)
    return df


def load_shot_events(usecols=None, dtype=None) -> pd.DataFrame:
    """Shot events from the CSV if present, else the per-season parquets."""
    if SHOT_CSV.exists():
        return pd.read_csv(SHOT_CSV, usecols=usecols, dtype=dtype)
    print(f"[data_sources] {SHOT_CSV.name} absent -> reading {SHOT_PARQUET_DIR.name}/")
    return _from_parquets(SHOT_PARQUET_DIR, usecols, dtype)


def load_shift_data(usecols=None, dtype=None) -> pd.DataFrame:
    """Shift data from the CSV if present, else the per-season parquets.
    Returned whole (the parquets are small); callers that used chunked reads
    can just iterate the frame."""
    if SHIFT_CSV.exists():
        return pd.read_csv(SHIFT_CSV, usecols=usecols, dtype=dtype)
    print(f"[data_sources] {SHIFT_CSV.name} absent -> reading {SHIFT_PARQUET_DIR.name}/")
    return _from_parquets(SHIFT_PARQUET_DIR, usecols, dtype)
