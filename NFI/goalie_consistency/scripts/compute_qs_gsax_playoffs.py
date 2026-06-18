"""Playoff QS-GSAx — playoff companion to compute_qs_gsax.py.

Same MoneyPuck all-shot 5v5-ES regulation methodology (per-game GSAx = xGoal −
goal; quality game = GSAx >= 0; per-game >=10 shots to count), but on PLAYOFF
games (isPlayoffGame == 1). Playoff shots exist in MoneyPuck for 2022-25.

NO goalie-level qualifying floor (regular uses GP>=25) — every goalie with a
qualifying game appears, per playoff season PLUS an `all_playoffs` pooled row.
GP is kept so the Streamlit slider thresholds at display. The per-game >=10
shots rule is the metric's definition and is retained.

Output: NFI/goalie_consistency/output/qs_gsax_per_season_playoffs.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/ashgarg/Documents/HockeyROI/Quality_Games/Data/Money_puck")
NAMES_FILE = Path("/Users/ashgarg/Documents/HockeyROI/NFI/output/player_positions.csv")
OUT_DIR = Path("/Users/ashgarg/Documents/HockeyROI/NFI/goalie_consistency/output")
OUT_FILE = OUT_DIR / "qs_gsax_per_season_playoffs.csv"

SEASONS = {"shots_2022.csv": "20222023", "shots_2023.csv": "20232024",
           "shots_2024.csv": "20242025", "shots_2025.csv": "20252026"}
MIN_SHOTS_PER_GAME = 10   # metric definition (NOT a goalie qualifying floor)


def wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (center - margin) / denom


print("[compute_qs_gsax_playoffs] playoff QS-GSAx, no goalie floor")
per_game_all = []
for fname, season in SEASONS.items():
    fp = DATA_DIR / fname
    if not fp.exists():
        print(f"  MISSING {fp}, skip"); continue
    df = pd.read_csv(fp)
    df = df[(df["homeSkatersOnIce"] == 5) & (df["awaySkatersOnIce"] == 5)
            & (df["isPlayoffGame"] == 1) & (df["goalieIdForShot"].notna())].copy()
    if "period" in df.columns:
        df = df[df["period"] <= 3]
    if df.empty:
        print(f"  {fname} (season {season}): no playoff shots"); continue
    pg = (df.groupby(["game_id", "goalieIdForShot", "goalieNameForShot"])
          .agg(shots_faced=("xGoal", "size"), xG=("xGoal", "sum"), goals=("goal", "sum")).reset_index())
    pg["GSAx"] = pg["xG"] - pg["goals"]
    pg["season"] = season
    pg = pg.rename(columns={"goalieIdForShot": "goalie_id", "goalieNameForShot": "goalie_name"})
    pg = pg[pg["shots_faced"] >= MIN_SHOTS_PER_GAME]
    per_game_all.append(pg)
    print(f"  {fname} (season {season}): {len(pg)} qualifying game-goalie rows")

per_game = pd.concat(per_game_all, ignore_index=True)
per_game["goalie_id"] = per_game["goalie_id"].astype(int)
per_game["quality"] = (per_game["GSAx"] >= 0).astype(int)

_names = pd.read_csv(NAMES_FILE)
_canon = dict(zip(_names["player_id"].astype(int), _names["player_name"].astype(str)))
_mp = per_game.drop_duplicates("goalie_id").set_index("goalie_id")["goalie_name"].to_dict()
canon = {g: _canon.get(g, _mp.get(g)) for g in per_game["goalie_id"].unique()}


def agg(group_cols, label=None):
    a = (per_game.groupby(group_cols)
         .agg(GP=("game_id", "size"), quality_games=("quality", "sum"),
              GSAx_total=("GSAx", "sum")).reset_index())
    if label is not None:
        a["season"] = label
    return a


out = pd.concat([agg(["goalie_id", "season"]), agg(["goalie_id"], "all_playoffs")],
                ignore_index=True)
out["goalie_name"] = out["goalie_id"].map(canon)
out["QS_GSAx_pct"] = out["quality_games"] / out["GP"] * 100
out["QS_GSAx_lo"] = out.apply(lambda r: wilson_lower(r["quality_games"], r["GP"]) * 100, axis=1)
out = out[["goalie_id", "goalie_name", "season", "GP", "quality_games",
           "GSAx_total", "QS_GSAx_pct", "QS_GSAx_lo"]]
out = out.sort_values(["season", "QS_GSAx_pct"], ascending=[True, False]).reset_index(drop=True)
out.to_csv(OUT_FILE, index=False)
print(f"\nWrote {OUT_FILE} — {len(out)} rows, {out['goalie_id'].nunique()} goalies")
ap = out[(out["season"] == "all_playoffs") & (out["GP"] >= 15)]
print("\n=== all_playoffs top 6 by QS-GSAx% (GP>=15) ===")
print(ap.nlargest(6, "QS_GSAx_pct")[["goalie_name", "GP", "quality_games",
                                     "QS_GSAx_pct", "GSAx_total"]].round(1).to_string(index=False))
