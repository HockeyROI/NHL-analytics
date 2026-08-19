"""build_pp_pk_app.py -- transform the SpecialTeams PP/PK ratings into the
app-friendly per-season files the Streamlit Player-Ratings family reads.

Reads (whichever case the SpecialTeams dir is on disk):
  <SpecialTeams>/Output/pp_rating.csv   -- PP Rating (0-100, locked EV-style formula)
  <SpecialTeams>/Output/pk_rating.csv   -- PK Rating (signed delta) + Wilson CI

Writes (committed, read by app_final.py exactly like player_ratings/elite_context):
  Elites/Output/pp_pk_ratings/{season}_{forwards|defense}.csv  for 2022-23..2025-26
    cols: player_name, team, pos, PP Rating, PP Tier, PK Rating, PK ci_low,
          PK ci_high, PK Flag

PP Tier = top 15% / next 55% / bottom 30% of PP Rating within (season, position).
PK Flag: PK is ASYMMETRIC -- only the liability side is reliable (the good side is
a soft-deployment artifact). So NO tier / NO "best PK" ranking:
  "Confirmed Liability" = clears_ci AND ci_high < 0 (whole CI below zero)
  "Repeat Liability"    = a Confirmed Liability whose player is confirmed in >= 2
                          seasons across the full 6-season dataset
  ""                    = everything else (incl. the unreliable positive side)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))
OUTDIR = os.path.join(PROJECT, "Elites", "Output", "pp_pk_ratings")
os.makedirs(OUTDIR, exist_ok=True)

APP_SEASONS = ["20222023", "20232024", "20242025", "20252026"]
SEASON_LABEL = {"20222023": "2022-23", "20232024": "2023-24",
                "20242025": "2024-25", "20252026": "2025-26"}


def _src(name):
    # The folder is canonically SpecialTeams (matches every pk_/pp_ script).
    # Absent on the CI runner -> return None so main() keeps the committed snapshot.
    p = os.path.join(PROJECT, "SpecialTeams", "Output", name)
    return p if os.path.exists(p) else None


def tier(vals):
    """top 15% Elite / next 55% Middle / bottom 30% Poor by within-group rank."""
    p = vals.rank(pct=True)
    out = pd.Series("Middle", index=vals.index)
    out[p >= 0.85] = "Elite"
    out[p <= 0.30] = "Poor"
    return out


def main():
    _pp_src, _pk_src = _src("pp_rating.csv"), _src("pk_rating.csv")
    if not _pp_src or not _pk_src:
        # CI runner has no local SpecialTeams pipeline yet -> keep the committed
        # pp_pk_ratings files as-is (a snapshot) instead of crashing the update.
        print("[pp_pk] SpecialTeams source CSVs not found -- keeping committed "
              "pp_pk_ratings files.", flush=True)
        return
    pp = pd.read_csv(_pp_src)
    pk = pd.read_csv(_pk_src)
    pp["season"] = pp["season"].astype(str)
    pk["season"] = pk["season"].astype(str)

    # PP Tier within (season, position)
    pp["PP Tier"] = (pp.groupby(["season", "position"])["PP Rating"]
                     .transform(tier))

    # PK Flag: confirmed liabilities + repeat offenders (across ALL seasons)
    pk["_confirmed"] = (pk["clears_ci"].astype(bool)) & (pk["ci_high"] < 0)
    _repeat = (pk[pk["_confirmed"]].groupby("player")["season"].nunique())
    _repeat = set(_repeat[_repeat >= 2].index)
    def _flag(r):
        if not r["_confirmed"]:
            return ""
        return "Repeat Liability" if r["player"] in _repeat else "Confirmed Liability"
    pk["PK Flag"] = pk.apply(_flag, axis=1)

    ppk = pp[["player", "season", "team", "position", "PP Rating", "PP Tier"]].merge(
        pk[["player", "season", "team", "position", "PK Rating", "ci_low", "ci_high", "PK Flag"]],
        on=["player", "season", "position"], how="outer", suffixes=("", "_pk"))
    ppk["team"] = ppk["team"].fillna(ppk["team_pk"])     # coalesce PP/PK team
    ppk = ppk.rename(columns={"player": "player_name", "position": "pos",
                              "ci_low": "PK ci_low", "ci_high": "PK ci_high"})
    ppk["PP Rating"] = ppk["PP Rating"].round(1)
    ppk["PK Rating"] = ppk["PK Rating"].round(2)
    ppk["PP Tier"] = ppk["PP Tier"].fillna("")
    ppk["PK Flag"] = ppk["PK Flag"].fillna("")

    cols = ["player_name", "team", "pos", "PP Rating", "PP Tier",
            "PK Rating", "PK ci_low", "PK ci_high", "PK Flag"]
    n = 0
    for ssn in APP_SEASONS:
        sub = ppk[ppk["season"] == ssn]
        for grp, posfile in (("F", "forwards"), ("D", "defense")):
            q = sub[sub["pos"] == grp]
            if q.empty:
                continue
            q[cols].to_csv(os.path.join(OUTDIR, f"{SEASON_LABEL[ssn]}_{posfile}.csv"),
                           index=False)
            n += 1
    print(f"[pp_pk] wrote {n} files -> {OUTDIR}", flush=True)
    # sanity
    latest = ppk[ppk["season"] == "20252026"]
    print("2025-26 confirmed/repeat liabilities:",
          latest["PK Flag"].value_counts().to_dict())
    print("2025-26 top PP forwards:",
          latest[latest.pos == "F"].nlargest(5, "PP Rating")[["player_name", "PP Rating", "PP Tier"]]
          .to_dict("records"))


if __name__ == "__main__":
    main()
