#!/usr/bin/env python3
"""Per-season goalie NFI-GSAx — companion to 20_fix_gsax_denominator.py.

Same source data (NFI/output/shots_tagged.csv) and same shots-faced
denominator correction as script 20, but:

  * Loops by season — one row per (goalie_id, season)
  * Uses CNFI + MNFI only (drops FNFI from both threshold and zone math)
  * Qualifying threshold: 100 dangerous-zone shots-faced per season
  * Adds per-row games + primary team derived from shots_tagged

Per-60 estimation: shots_tagged doesn't carry minutes; player_toi.csv is
pooled across seasons. ES TOI per goalie-season is therefore allocated
proportionally — each goalie's pooled toi_ES_min times the share of their
career faced shots that fell in the season. This treats the faced/minute
rate as approximately constant across seasons for a given goalie, which
is the standard simplification when shifts data isn't season-keyed.

Output: NFI/output/goalie_nfi_gsax_by_season.csv

Does NOT modify script 20 or any existing files.
"""
import os
import math

import numpy as np
import pandas as pd

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
OUT  = f"{ROOT}/NFI/output"
SHOT_FP = f"{OUT}/shots_tagged.csv"
POS_FP  = f"{OUT}/player_positions.csv"
TOI_FP  = f"{OUT}/player_toi.csv"
OUT_FP  = f"{OUT}/goalie_nfi_gsax_by_season.csv"

MIN_SHOTS = 100   # dangerous-zone shots-faced per season — was 300 pooled
DANGER_ZONES = ["CNFI", "MNFI"]   # FNFI dropped per spec
DEFAULT_LEAGUE_DANGER_PER60 = 12.0  # NHL ~12 ES dangerous SA/60 fallback


def faced_mask(df: pd.DataFrame) -> pd.Series:
    return df["event_type"].isin(["shot-on-goal", "goal"])


def main() -> None:
    print("Loading shots_tagged.csv ...")
    sh = pd.read_csv(SHOT_FP)
    # Bug 2 fix (May 2026): restrict to regular-season games. shots_tagged.csv
    # includes playoff data for build_playoff_data.py; goalie GSAx must match
    # the regular-season-only convention. game_id digits 4-5 == "02" = regular.
    sh = sh[sh["game_id"].astype(str).str[4:6] == "02"].copy()
    es = sh[sh["state"] == "ES"].copy()
    es["season"] = es["season"].astype(int)
    print(f"ES rows (5v5 regular season): {len(es):,}")
    print(f"Seasons: {sorted(es['season'].unique().tolist())}")

    # Goalie name lookup
    pos = pd.read_csv(POS_FP)
    name_map = dict(zip(pos["player_id"].astype("Int64"),
                         pos["player_name"].astype(str)))

    # Pooled ES TOI per goalie (used to allocate per-season minutes by faced share)
    toi = pd.read_csv(TOI_FP)
    toi_g = toi[toi["position"] == "G"][["player_id", "toi_ES_sec"]].copy()
    toi_g["pooled_toi_min"] = toi_g["toi_ES_sec"] / 60.0
    pooled_toi_map = dict(zip(toi_g["player_id"].astype("Int64"),
                               toi_g["pooled_toi_min"]))

    season_frames = []
    for season in sorted(es["season"].unique()):
        ss = es[es["season"] == season].copy()

        # ------ per-season per-zone faced goal rate (corrected denominator) ------
        rate_faced = {}
        for z in DANGER_ZONES:
            zone_rows = ss[(ss["zone"] == z) & faced_mask(ss)]
            n = len(zone_rows)
            k = int(zone_rows["is_goal_i"].sum())
            rate_faced[z] = (k / n) if n else 0.0

        # ------ goalie aggregation: faced + goals per zone in the season ------
        f = ss[faced_mask(ss) & ss["goalie_id"].notna()].copy()
        f["goalie_id"] = f["goalie_id"].astype(int)

        # Per (goalie, zone) faced + goals (danger zones only)
        danger = f[f["zone"].isin(DANGER_ZONES)]
        agg = (
            danger.groupby(["goalie_id", "zone"])
                  .agg(faced=("is_goal_i", "size"),
                       goals=("is_goal_i", "sum"))
                  .reset_index()
        )
        wide = agg.pivot_table(index="goalie_id", columns="zone",
                                values=["faced", "goals"], fill_value=0)
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide = wide.reset_index()

        for c in ["faced_CNFI", "faced_MNFI", "goals_CNFI", "goals_MNFI"]:
            if c not in wide.columns:
                wide[c] = 0

        wide["total_faced"] = wide["faced_CNFI"] + wide["faced_MNFI"]
        wide["total_goals"] = wide["goals_CNFI"] + wide["goals_MNFI"]

        # Do NOT drop sub-threshold goalies: emit every goalie-season and flag
        # whether it clears the per-season net-front-shot bar via `qualified`.
        # The app shows all values but only ranks qualified goalies (others "UR");
        # downstream analysis should filter on `qualified == True`.
        wide["qualified"] = wide["total_faced"] >= MIN_SHOTS
        if wide.empty:
            print(f"  season {season}: no goalies")
            continue

        # xG_calibrated using corrected per-faced rates (CNFI + MNFI only)
        wide["xG"] = (wide["faced_CNFI"] * rate_faced["CNFI"]
                       + wide["faced_MNFI"] * rate_faced["MNFI"])
        wide["GSAx"] = (wide["xG"] - wide["total_goals"]).round(2)
        # Raw (unadjusted) save% on the NFI danger-zone shot set — a
        # complement to GSAx, not a replacement. GSAx already accounts for
        # shot difficulty within CNFI/MNFI; this doesn't.
        wide["NFI_save_pct"] = np.where(
            wide["total_faced"] > 0,
            (wide["total_faced"] - wide["total_goals"]) / wide["total_faced"],
            np.nan,
        ).round(4)

        # ------ games + primary team per goalie (from shots_tagged) ------
        # Defending team for each shot: home if shooter is away, else away.
        ss_full = ss[ss["goalie_id"].notna()].copy()
        ss_full["goalie_id"] = ss_full["goalie_id"].astype(int)
        ss_full["defending_team"] = np.where(
            ss_full["shooting_team_abbrev"] == ss_full["home_team_abbrev"],
            ss_full["away_team_abbrev"],
            ss_full["home_team_abbrev"],
        )
        team_mode = (
            ss_full.groupby("goalie_id")["defending_team"]
                  .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "")
                  .rename("team")
                  .reset_index()
        )
        games = (
            ss_full.groupby("goalie_id")["game_id"]
                  .nunique().rename("games").reset_index()
        )

        wide = wide.merge(team_mode, on="goalie_id", how="left")
        wide = wide.merge(games, on="goalie_id", how="left")

        # ------ per-60 via pooled-TOI-share allocation ------
        # season_toi_min = pooled_toi_min[goalie] * (season_faced / pooled_faced)
        # If pooled TOI is missing fall back to the league-rate proxy.
        # Pooled faced is computed across all seasons in shots_tagged.
        if "_pooled_faced_cache" not in globals():
            pooled_danger = es[
                faced_mask(es) & es["zone"].isin(DANGER_ZONES) & es["goalie_id"].notna()
            ].copy()
            pooled_danger["goalie_id"] = pooled_danger["goalie_id"].astype(int)
            globals()["_pooled_faced_cache"] = (
                pooled_danger.groupby("goalie_id").size().to_dict()
            )
        pooled_faced_map = globals()["_pooled_faced_cache"]

        def _toi_for(gid: int, season_faced: float) -> float:
            pooled_min = pooled_toi_map.get(gid)
            pooled_f = pooled_faced_map.get(gid, 0)
            if pooled_min and pooled_f and pooled_f > 0:
                return float(pooled_min) * (season_faced / pooled_f)
            # Fallback: league-rate proxy (~12 dangerous SA per 60 ES min)
            return season_faced / DEFAULT_LEAGUE_DANGER_PER60 * 60.0

        wide["es_toi_min"] = [
            _toi_for(int(gid), float(f_))
            for gid, f_ in zip(wide["goalie_id"], wide["total_faced"])
        ]
        wide["GSAx_per60"] = np.where(
            wide["es_toi_min"] > 0,
            wide["GSAx"] / wide["es_toi_min"] * 60.0,
            np.nan,
        ).round(3)

        # ------ goalie name + final tidy ------
        wide["goalie_name"] = wide["goalie_id"].astype("Int64").map(name_map).fillna("")
        wide["season"] = int(season)

        keep = ["goalie_id", "goalie_name", "season",
                "GSAx", "GSAx_per60", "NFI_save_pct",
                "total_faced", "total_goals", "games", "team", "qualified"]
        season_frames.append(wide[keep].copy())
        print(f"  season {season}: {len(wide)} goalies, "
              f"{int(wide['qualified'].sum())} qualified (>= {MIN_SHOTS} shots)")

    out = pd.concat(season_frames, ignore_index=True)
    out = out.sort_values(["season", "GSAx"], ascending=[True, False]).reset_index(drop=True)
    out.to_csv(OUT_FP, index=False)
    print(f"\nWrote {OUT_FP} — shape {out.shape}")

    # ------ Sanity reports ------
    print("\n=== Goalies per season ===")
    print(out.groupby("season").size().to_string())

    print("\n=== Top 5 by GSAx_per60 (2024-25, faced ≥ 200 to filter noise) ===")
    s2425 = out[(out["season"] == 20242025) & (out["total_faced"] >= 200)]
    print(s2425.nlargest(5, "GSAx_per60")[
        ["goalie_name", "team", "total_faced", "games", "GSAx", "GSAx_per60"]
    ].to_string(index=False))

    print("\n=== Shesterkin & Sorokin multi-season check ===")
    for name in ("Shesterkin", "Sorokin"):
        rows = out[out["goalie_name"].str.contains(name, na=False)]
        print(f"{name}: {len(rows)} season rows")
        print(rows[["season", "team", "total_faced", "games",
                    "GSAx", "GSAx_per60"]].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
