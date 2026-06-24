"""
Build QS-GSAx from MoneyPuck per-shot data.

QS-GSAx = share of qualifying games where per-game GSAx >= 0.
Companion metric to QNFS%: same binary methodology, all-shot scope, MoneyPuck xG model.

Methodology:
  - 5v5 even-strength regulation only (5v5 derived from homeSkatersOnIce == 5 AND awaySkatersOnIce == 5)
  - Regular season only (isPlayoffGame == 0)
  - All Fenwick shots (MoneyPuck shots file = SHOT/MISS/GOAL only, no blocked shots; no shotWasOnGoal filter)
  - Empty-net excluded (goalieIdForShot must be valid)
  - Min 10 shots faced per game for game to count
  - Min 25 qualifying games per season for goalie-season to qualify
  - Per-game GSAx = sum(xGoal) - sum(goal)
  - Quality game = per-game GSAx >= 0
  - Wilson 95% lower bound for ranking
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---- CONFIG ----
DATA_DIR = Path("/Users/ashgarg/Documents/HockeyROI/Quality_Games/Data/Money_puck")
NAMES_FILE = Path("/Users/ashgarg/Documents/HockeyROI/NFI/output/player_positions.csv")
OUT_DIR = Path("/Users/ashgarg/Documents/HockeyROI/NFI/goalie_consistency/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = {
    "shots_2022.csv": 20222023,
    "shots_2023.csv": 20232024,
    "shots_2024.csv": 20242025,
    "shots_2025.csv": 20252026,
}

MIN_SHOTS_PER_GAME = 10
MIN_GP_PER_SEASON = 25

# ---- DIAGNOSTIC FIRST ----
print("=== DIAGNOSTIC ===")
for fname in SEASONS:
    fp = DATA_DIR / fname
    if not fp.exists():
        raise FileNotFoundError(f"Missing: {fp}")
    sample = pd.read_csv(fp, nrows=5)
    needed = {"game_id", "goalieIdForShot", "goalieNameForShot", "xGoal", "goal",
              "season", "isPlayoffGame", "homeSkatersOnIce", "awaySkatersOnIce"}
    missing = needed - set(sample.columns)
    if missing:
        raise ValueError(f"{fname} missing required columns: {missing}")
    print(f"  {fname}: header OK, required columns present")
print()

# ---- BUILD PER-GAME GSAx ACROSS ALL 4 SEASONS ----
per_game_all = []
for fname, season_int in SEASONS.items():
    fp = DATA_DIR / fname
    print(f"Loading {fname} (season {season_int})...")
    df = pd.read_csv(fp)
    n_total = len(df)

    # Filter: 5v5 ES regulation, regular season, valid goalie
    df = df[
        (df["homeSkatersOnIce"] == 5)
        & (df["awaySkatersOnIce"] == 5)
        & (df["isPlayoffGame"] == 0)
        & (df["goalieIdForShot"].notna())
    ].copy()

    # If a period column exists, filter to regulation only (period 1-3)
    if "period" in df.columns:
        df = df[df["period"] <= 3]
    else:
        print(f"    NOTE: no 'period' column in {fname} — relying on 5v5 filter to exclude most OT")

    n_after = len(df)
    print(f"    rows: {n_total:,} -> {n_after:,} after 5v5/reg/season filters")

    # Aggregate per (game, goalie)
    pg = (
        df.groupby(["game_id", "goalieIdForShot", "goalieNameForShot"])
        .agg(shots_faced=("xGoal", "size"), xG=("xGoal", "sum"), goals=("goal", "sum"))
        .reset_index()
    )
    pg["GSAx"] = pg["xG"] - pg["goals"]
    pg["season"] = season_int
    pg = pg.rename(columns={"goalieIdForShot": "goalie_id", "goalieNameForShot": "goalie_name"})

    # Game qualification: min shots faced
    n_games_pre = len(pg)
    pg = pg[pg["shots_faced"] >= MIN_SHOTS_PER_GAME]
    n_games_post = len(pg)
    print(f"    per-game rows: {n_games_pre:,} -> {n_games_post:,} after min-{MIN_SHOTS_PER_GAME}-shots filter")

    per_game_all.append(pg)

per_game = pd.concat(per_game_all, ignore_index=True)
per_game["goalie_id"] = per_game["goalie_id"].astype(int)
print(f"\nTotal per-game rows across 4 seasons: {len(per_game):,}")
print(f"Unique goalies: {per_game['goalie_id'].nunique()}")

# ---- CANONICAL NAME MAP (same pattern as compute_qnfs.py) ----
# MoneyPuck spells some goalies inconsistently across seasons (e.g. "Sam" vs
# "Samuel" Montembeault), which would split one goalie into multiple rows if we
# grouped on name. Group on goalie_id ONLY, then attach a single canonical name
# from player_positions.csv, falling back to the first MoneyPuck spelling.
_names = pd.read_csv(NAMES_FILE)
_canon = dict(zip(_names["player_id"].astype(int), _names["player_name"].astype(str)))
_mp_name = per_game.drop_duplicates("goalie_id").set_index("goalie_id")["goalie_name"].to_dict()
canon_name = {gid: _canon.get(gid, _mp_name.get(gid)) for gid in per_game["goalie_id"].unique()}

# ---- SPOT-CHECK: Vasilevskiy 4-season overall GSAx (sanity check) ----
vasi = per_game[per_game["goalie_id"] == 8476883]
if len(vasi):
    print(f"\n  Spot-check: Vasilevskiy 4-season GSAx (all-shot, 5v5 reg) = {vasi['GSAx'].sum():+.2f} across {len(vasi)} games")
    print(f"  (NF-only 4-season GSAx was +33.07 — overall should be larger in magnitude)")
else:
    print("  WARNING: Vasilevskiy not found in aggregated data — check goalie_id")

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
    # GP = count of per-game rows (one row per goalie-game). NOT nunique(game_id):
    # MoneyPuck game_id is bare (e.g. 20001 recurs every season), so nunique would
    # collapse same-numbered games across seasons. (Harmless within a single season,
    # but use size everywhere for consistency.)
    .agg(GP=("game_id", "size"), quality_games=("quality", "sum"),
         GSAx_total=("GSAx", "sum"))
    .reset_index()
)
per_season["goalie_name"] = per_season["goalie_id"].map(canon_name)
per_season["QS_GSAx_pct"] = per_season["quality_games"] / per_season["GP"] * 100
per_season["QS_GSAx_lo"] = per_season.apply(
    lambda r: wilson_lower(r["quality_games"], r["GP"]) * 100, axis=1
)

# DO NOT drop sub-25-GP seasons. Emit every goalie-season and flag whether it
# clears the 25-GP sample-size bar via a `qualified` column. The Streamlit app
# shows every value but only ranks qualified goalie-seasons (others render "UR");
# downstream analysis should filter on `qualified == True`.
per_season_qual = per_season.copy()
per_season_qual["qualified"] = per_season_qual["GP"] >= MIN_GP_PER_SEASON
n_q = int(per_season_qual["qualified"].sum())
print(f"\nPer-season: {len(per_season_qual)} rows emitted, {n_q} qualified (GP >= {MIN_GP_PER_SEASON})")

# ---- COMPUTE POOLED 4-SEASON ----
# Pool EVERY goalie (no gate); flag pooled `qualified` as having >=1 season that
# cleared the 25-GP bar (matches the QNFS pooled qualified definition).
qualified_goalies = set(per_season_qual.loc[per_season_qual["qualified"], "goalie_id"])
pool_data = per_game

pooled = (
    pool_data.groupby(["goalie_id"])
    # GP = count of per-game rows (true games). nunique(game_id) would undercount
    # because MoneyPuck game_id is bare and recurs across seasons (see per_season note).
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

print(f"\nPooled 4-season: {len(pooled)} goalies, {int(pooled['qualified'].sum())} qualified")

# ---- WRITE OUTPUTS ----
out_per_season = OUT_DIR / "qs_gsax_per_season_2022-2026.csv"
out_pooled = OUT_DIR / "qs_gsax_2022-2026.csv"

per_season_qual.to_csv(out_per_season, index=False)
pooled.to_csv(out_pooled, index=False)
print(f"\nWritten:")
print(f"  {out_per_season}")
print(f"  {out_pooled}")

# ---- VALIDATION: print top 20 pooled + spot-check the six ----
print("\n=== TOP 20 POOLED QS-GSAx (4-season) ===")
print(pooled.head(20)[["rank", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo", "GSAx_total"]].to_string(index=False))

print("\n=== SIX VALIDATION ===")
SIX_NAMES = ["Jeremy Swayman", "Connor Hellebuyck", "Igor Shesterkin", "Ilya Sorokin", "John Gibson", "Logan Thompson"]
SIX_IDS = {  # NHL player IDs as fallback if names don't match MoneyPuck's spelling
    # (corrected against MoneyPuck id->name map before run; originals for Swayman/Sorokin/Thompson were wrong)
    "Jeremy Swayman": 8480280,
    "Connor Hellebuyck": 8476945,
    "Igor Shesterkin": 8478048,
    "Ilya Sorokin": 8478009,
    "John Gibson": 8476434,
    "Logan Thompson": 8480313,
}
for name in SIX_NAMES:
    row = pooled[pooled["goalie_name"] == name]
    if len(row) == 0:
        # Fallback: lookup by ID in case name spelling differs
        gid = SIX_IDS[name]
        row = pooled[pooled["goalie_id"] == gid]
        if len(row):
            actual_name = row.iloc[0]["goalie_name"]
            print(f"  {name} (MoneyPuck name: '{actual_name}'): rank {int(row.iloc[0]['rank'])}, "
                  f"QS-GSAx% {row.iloc[0]['QS_GSAx_pct']:.2f}, Wilson {row.iloc[0]['QS_GSAx_lo']:.2f}, GP {int(row.iloc[0]['GP'])}")
        else:
            print(f"  {name}: NOT FOUND by name or by ID {gid}")
    else:
        r = row.iloc[0]
        print(f"  {name}: rank {int(r['rank'])}, QS-GSAx% {r['QS_GSAx_pct']:.2f}, "
              f"Wilson {r['QS_GSAx_lo']:.2f}, GP {int(r['GP'])}")

print("\nDONE.")
