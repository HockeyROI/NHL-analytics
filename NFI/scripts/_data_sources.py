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


# Our own per-event xG (xG/build_xg.py). Keyed (game_id, event_id).
XG_CSV = ROOT / "xG" / "output" / "shot_xg_per_event.csv"

# Our shot-event event_type -> MoneyPuck's `event` code. MoneyPuck's shots file
# is Fenwick-only (SHOT/MISS/GOAL); blocked shots aren't in it, so we drop them.
_MP_EVENT = {"shot-on-goal": "SHOT", "missed-shot": "MISS", "goal": "GOAL"}


def load_shots_mp_schema(fenwick_only: bool = True) -> pd.DataFrame:
    """Our shot events remapped to MoneyPuck's column names, with our own xG
    merged on, so the goalie-consistency and Quality-Games builders can drop
    MoneyPuck with a one-line source swap instead of a rewrite.

    Emitted columns (MoneyPuck spelling):
      game_id, season (int, e.g. 20242025), isPlayoffGame (0/1), period,
      homeSkatersOnIce, awaySkatersOnIce, goalieIdForShot, shooterPlayerId,
      event (SHOT/MISS/GOAL), goal (0/1), xGoal (our xg; 0 where unscored).

    Empty-net shots keep goalieIdForShot = NaN (the defending goalie is pulled),
    so the same `goalieIdForShot.notna()` filter the MoneyPuck builders already
    use still excludes them — GSAx/save% can't be charged on an empty net.
    """
    ev = load_shot_events(
        usecols=["game_id", "event_id", "season", "game_type", "period",
                 "event_type", "situation_code", "goalie_id",
                 "shooter_player_id", "is_goal"],
        dtype={"situation_code": str})
    ev = ev[ev["event_type"].isin(_MP_EVENT)] if fenwick_only else ev
    sc = ev["situation_code"].astype(str).str.zfill(4)
    out = pd.DataFrame({
        "game_id": ev["game_id"].astype(int),
        "event_id": ev["event_id"].astype(int),
        "season": ev["season"].astype(int),
        "isPlayoffGame": (ev["game_type"] == "playoff").astype(int),
        "period": ev["period"].astype(int),
        "awaySkatersOnIce": sc.str[1].astype(int),   # code = [ag, ask, hsk, hg]
        "homeSkatersOnIce": sc.str[2].astype(int),
        "goalieIdForShot": ev["goalie_id"],
        "shooterPlayerId": ev["shooter_player_id"],
        "event": ev["event_type"].map(_MP_EVENT),
        "goal": ev["is_goal"].astype(int),
    })
    xg = pd.read_csv(XG_CSV)
    out = out.merge(xg, on=["game_id", "event_id"], how="left")
    out["xGoal"] = out["xg"].fillna(0.0)
    return out.drop(columns=["xg"])
