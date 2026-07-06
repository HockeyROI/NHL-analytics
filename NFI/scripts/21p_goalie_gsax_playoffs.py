#!/usr/bin/env python3
"""Playoff goalie NFI-GSAx — playoff companion to 21_goalie_gsax_by_season.py.

Same definition (CNFI+MNFI ES faced, corrected per-faced rate, GSAx = xG −
goals, per-60 via faced-share TOI allocation) but on PLAYOFF games
(game_id digits 4-5 == "03"). Playoff shots exist only for 2022-23..2024-25.

NO qualifying floor — every goalie with any playoff danger-zone shot is
emitted, per playoff season PLUS an `all_playoffs` pooled row. `total_faced`
is included so the Streamlit Min-Shots slider does the thresholding. The app
shows the pooled row by default.

Per-60 TOI: there is no playoff goalie TOI file, so playoff ES minutes are
allocated from each goalie's REGULAR faced-rate — playoff_toi = regular_pooled
ES TOI × (playoff_faced / regular_pooled_faced). Same "faced/min ≈ constant"
simplification script 21 already uses across seasons.

Output: NFI/output/goalie_nfi_gsax_by_season_playoffs.csv
Does NOT modify the regular file or any existing output.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
OUT = f"{ROOT}/NFI/output"
SHOT_FP = f"{OUT}/shots_tagged.csv"
POS_FP = f"{OUT}/player_positions.csv"
TOI_FP = f"{OUT}/player_toi.csv"
OUT_FP = f"{OUT}/goalie_nfi_gsax_by_season_playoffs.csv"

DANGER = ["CNFI", "MNFI"]
DEFAULT_LEAGUE_DANGER_PER60 = 12.0


def faced_mask(df):
    return df["event_type"].isin(["shot-on-goal", "goal"])


def _season_block(ss, rate_faced, name_map, reg_toi, reg_faced, label):
    """Build one output block (a single playoff season or the pooled set)."""
    f = ss[faced_mask(ss) & ss["goalie_id"].notna()].copy()
    if f.empty:
        return None
    f["goalie_id"] = f["goalie_id"].astype(int)
    danger = f[f["zone"].isin(DANGER)]
    agg = (danger.groupby(["goalie_id", "zone"])
           .agg(faced=("is_goal_i", "size"), goals=("is_goal_i", "sum")).reset_index())
    if agg.empty:
        return None
    wide = agg.pivot_table(index="goalie_id", columns="zone",
                           values=["faced", "goals"], fill_value=0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for c in ["faced_CNFI", "faced_MNFI", "goals_CNFI", "goals_MNFI"]:
        if c not in wide.columns:
            wide[c] = 0
    wide["total_faced"] = wide["faced_CNFI"] + wide["faced_MNFI"]
    wide["total_goals"] = wide["goals_CNFI"] + wide["goals_MNFI"]
    # NO floor — keep every goalie with any danger-zone faced shot.
    wide = wide[wide["total_faced"] > 0].copy()
    if wide.empty:
        return None
    wide["xG"] = wide["faced_CNFI"] * rate_faced["CNFI"] + wide["faced_MNFI"] * rate_faced["MNFI"]
    wide["GSAx"] = (wide["xG"] - wide["total_goals"]).round(2)
    wide["NFI_save_pct"] = np.where(
        wide["total_faced"] > 0,
        (wide["total_faced"] - wide["total_goals"]) / wide["total_faced"],
        np.nan,
    ).round(4)

    ffull = f.copy()
    ffull["defending_team"] = np.where(
        ffull["shooting_team_abbrev"] == ffull["home_team_abbrev"],
        ffull["away_team_abbrev"], ffull["home_team_abbrev"])
    team_mode = (ffull.groupby("goalie_id")["defending_team"]
                 .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "")
                 .rename("team").reset_index())
    games = ffull.groupby("goalie_id")["game_id"].nunique().rename("games").reset_index()
    wide = wide.merge(team_mode, on="goalie_id", how="left").merge(games, on="goalie_id", how="left")

    def _toi(gid, faced):
        pm, pf = reg_toi.get(gid), reg_faced.get(gid, 0)
        if pm and pf and pf > 0:
            return float(pm) * (faced / pf)
        return faced / DEFAULT_LEAGUE_DANGER_PER60 * 60.0

    wide["es_toi_min"] = [_toi(int(g), float(x)) for g, x in zip(wide["goalie_id"], wide["total_faced"])]
    wide["GSAx_per60"] = np.where(wide["es_toi_min"] > 0, wide["GSAx"] / wide["es_toi_min"] * 60.0, np.nan).round(3)
    wide["goalie_name"] = wide["goalie_id"].astype("Int64").map(name_map).fillna("")
    wide["season"] = label
    return wide[["goalie_id", "goalie_name", "season", "GSAx", "GSAx_per60",
                 "NFI_save_pct", "total_faced", "total_goals", "games", "team"]].copy()


def main():
    print("Loading shots_tagged.csv ...")
    sh = pd.read_csv(SHOT_FP)
    sh["_dig"] = sh["game_id"].astype(str).str[4:6]
    es = sh[sh["state"] == "ES"].copy()
    reg = es[es["_dig"] == "02"]
    po = es[es["_dig"] == "03"].copy()
    po["season"] = po["season"].astype(str)
    print(f"  ES regular rows: {len(reg):,} | ES playoff rows: {len(po):,}")
    print(f"  playoff seasons: {sorted(po['season'].unique())}")

    pos = pd.read_csv(POS_FP)
    name_map = dict(zip(pos["player_id"].astype("Int64"), pos["player_name"].astype(str)))

    # Regular baselines for TOI allocation: pooled regular ES TOI + faced/goalie.
    toi = pd.read_csv(TOI_FP)
    tg = toi[toi["position"] == "G"][["player_id", "toi_ES_sec"]].copy()
    reg_toi = dict(zip(tg["player_id"].astype(int), tg["toi_ES_sec"] / 60.0))
    rd = reg[faced_mask(reg) & reg["zone"].isin(DANGER) & reg["goalie_id"].notna()].copy()
    rd["goalie_id"] = rd["goalie_id"].astype(int)
    reg_faced = rd.groupby("goalie_id").size().to_dict()

    def _rate(scope):
        r = {}
        for z in DANGER:
            zr = scope[(scope["zone"] == z) & faced_mask(scope)]
            n = len(zr)
            r[z] = (int(zr["is_goal_i"].sum()) / n) if n else 0.0
        return r

    blocks = []
    for season in sorted(po["season"].unique()):
        ss = po[po["season"] == season]
        b = _season_block(ss, _rate(ss), name_map, reg_toi, reg_faced, season)
        if b is not None:
            blocks.append(b)
            print(f"  [{season}] {len(b)} goalies")
    # Pooled: all playoff seasons combined, own pooled rate.
    pooled = _season_block(po, _rate(po), name_map, reg_toi, reg_faced, "all_playoffs")
    if pooled is not None:
        blocks.append(pooled)
        print(f"  [all_playoffs] {len(pooled)} goalies")

    out = pd.concat(blocks, ignore_index=True).sort_values(
        ["season", "GSAx"], ascending=[True, False]).reset_index(drop=True)
    out.to_csv(OUT_FP, index=False)
    print(f"\nWrote {OUT_FP} — shape {out.shape}")
    print("\n=== all_playoffs top 6 by GSAx (faced >= 100) ===")
    ap = out[(out["season"] == "all_playoffs") & (out["total_faced"] >= 100)]
    print(ap.nlargest(6, "GSAx")[["goalie_name", "team", "total_faced", "games",
                                  "GSAx", "GSAx_per60"]].to_string(index=False))


if __name__ == "__main__":
    main()
