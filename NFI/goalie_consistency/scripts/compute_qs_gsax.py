"""
Build QS-GSAx from HockeyROI's own per-shot xG (xG/build_xg.py).

QS-GSAx = share of qualifying games where per-game GSAx >= 0.
Companion metric to QNFS%: same binary methodology, all-shot scope.

MoneyPuck retired (2026): reads our shot events + our xG via the MP-schema shim
(NFI/scripts/_data_sources.load_shots_mp_schema), so the MoneyPuck-shaped
filter/agg logic below is unchanged — only the data source swapped.

Methodology:
  - 5v5 even-strength regulation only (homeSkatersOnIce == 5 AND awaySkatersOnIce == 5)
  - Regular season only (isPlayoffGame == 0)
  - All Fenwick shots (SHOT/MISS/GOAL; blocked shots not included)
  - Empty-net excluded (goalieIdForShot must be valid)
  - Min 10 shots faced per game for game to count
  - Min 25 qualifying games per season for goalie-season to qualify
  - Per-game GSAx = sum(xGoal) - sum(goal)
  - Wilson 95% lower bound for ranking
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _data_sources as _ds

# ---- CONFIG ----
NAMES_FILE = Path("/Users/ashgarg/Documents/HockeyROI/NFI/output/player_positions.csv")
OUT_DIR = Path("/Users/ashgarg/Documents/HockeyROI/NFI/goalie_consistency/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SHOTS_PER_GAME = 10
MIN_GP_PER_SEASON = 25
# Same 4-season window the "2022-2026" outputs have always covered (the shim
# also carries 2020-21/2021-22, which we exclude to keep the pool unchanged).
SEASONS = {20222023, 20232024, 20242025, 20252026}

# ---- BUILD PER-GAME GSAx (regular-season, 5v5 ES, all Fenwick) ----
# One MP-schema frame for every season; goalie name comes from player_positions
# (attached later by canonical id), so goalieNameForShot isn't needed here.
allshots = _ds.load_shots_mp_schema()
df = allshots[
    allshots["season"].isin(SEASONS)
    & (allshots["homeSkatersOnIce"] == 5)
    & (allshots["awaySkatersOnIce"] == 5)
    & (allshots["isPlayoffGame"] == 0)
    & (allshots["period"] <= 3)
    & (allshots["goalieIdForShot"].notna())
].copy()
print(f"5v5 ES reg, goalie present: {len(df):,} shots across {df['season'].nunique()} seasons")

pg = (
    df.groupby(["game_id", "goalieIdForShot", "season"])
    .agg(shots_faced=("xGoal", "size"), xG=("xGoal", "sum"), goals=("goal", "sum"))
    .reset_index()
    .rename(columns={"goalieIdForShot": "goalie_id"})
)
pg["GSAx"] = pg["xG"] - pg["goals"]
pg["goalie_name"] = ""   # filled from canonical map below
n_pre = len(pg)
pg = pg[pg["shots_faced"] >= MIN_SHOTS_PER_GAME]
print(f"per-game rows: {n_pre:,} -> {len(pg):,} after min-{MIN_SHOTS_PER_GAME}-shots filter")

per_game = pg
per_game["goalie_id"] = per_game["goalie_id"].astype(int)
print(f"\nTotal per-game rows across all seasons: {len(per_game):,}")
print(f"Unique goalies: {per_game['goalie_id'].nunique()}")

# ---- CANONICAL NAME MAP ----
# Group on goalie_id, attach a single canonical name from player_positions.csv.
_names = pd.read_csv(NAMES_FILE)
_canon = dict(zip(_names["player_id"].astype(int), _names["player_name"].astype(str)))
canon_name = {gid: _canon.get(gid, str(gid)) for gid in per_game["goalie_id"].unique()}

# ---- SPOT-CHECK: Vasilevskiy 4-season overall GSAx (sanity check) ----
vasi = per_game[per_game["goalie_id"] == 8476883]
if len(vasi):
    print(f"\n  Spot-check: Vasilevskiy GSAx (all-shot, 5v5 reg) = {vasi['GSAx'].sum():+.2f} across {len(vasi)} games")

# ---- COMPUTE PER-SEASON QS-GSAx + WILSON FLOOR ----
def wilson_lower(k, n, z=1.96):
    """Wilson 95% lower bound for a binomial proportion."""
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (center - margin) / denom

per_game["quality"] = (per_game["GSAx"] >= 0).astype(int)

per_season = (
    per_game.groupby(["goalie_id", "season"])
    .agg(GP=("game_id", "size"), quality_games=("quality", "sum"),
         GSAx_total=("GSAx", "sum"))
    .reset_index()
)
per_season["goalie_name"] = per_season["goalie_id"].map(canon_name)
per_season["QS_GSAx_pct"] = per_season["quality_games"] / per_season["GP"] * 100
per_season["QS_GSAx_lo"] = per_season.apply(
    lambda r: wilson_lower(r["quality_games"], r["GP"]) * 100, axis=1
)

# Emit every goalie-season; flag whether it clears the 25-GP sample-size bar.
per_season_qual = per_season.copy()
per_season_qual["qualified"] = per_season_qual["GP"] >= MIN_GP_PER_SEASON
n_q = int(per_season_qual["qualified"].sum())
print(f"\nPer-season: {len(per_season_qual)} rows emitted, {n_q} qualified (GP >= {MIN_GP_PER_SEASON})")

# ---- COMPUTE POOLED 4-SEASON ----
qualified_goalies = set(per_season_qual.loc[per_season_qual["qualified"], "goalie_id"])
pool_data = per_game

pooled = (
    pool_data.groupby(["goalie_id"])
    .agg(GP=("game_id", "size"), quality_games=("quality", "sum"),
         GSAx_total=("GSAx", "sum"))
    .reset_index()
)
pooled["goalie_name"] = pooled["goalie_id"].map(canon_name)
pooled["QS_GSAx_pct"] = pooled["quality_games"] / pooled["GP"] * 100
pooled["QS_GSAx_lo"] = pooled.apply(
    lambda r: wilson_lower(r["quality_games"], r["GP"]) * 100, axis=1
)
pooled["qualified"] = pooled["goalie_id"].isin(qualified_goalies)
pooled = pooled.sort_values(["qualified", "QS_GSAx_lo"], ascending=False).reset_index(drop=True)
pooled["rank"] = pooled.index + 1

print(f"\nPooled: {len(pooled)} goalies, {int(pooled['qualified'].sum())} qualified")

# ---- WRITE OUTPUTS ----
out_per_season = OUT_DIR / "qs_gsax_per_season_2022-2026.csv"
out_pooled = OUT_DIR / "qs_gsax_2022-2026.csv"
per_season_qual.to_csv(out_per_season, index=False)
pooled.to_csv(out_pooled, index=False)
print(f"\nWritten:\n  {out_per_season}\n  {out_pooled}")

# ---- VALIDATION: print top 20 pooled ----
print("\n=== TOP 20 POOLED QS-GSAx ===")
print(pooled.head(20)[["rank", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo", "GSAx_total"]].to_string(index=False))

print("\nDONE.")
