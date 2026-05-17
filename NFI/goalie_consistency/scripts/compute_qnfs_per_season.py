# NFI/goalie_consistency/scripts/compute_qnfs_per_season.py
#
# Per-(goalie, season) Quality Net-Front Save percentage with Wilson 95% CIs.
# Companion to compute_qnfs.py (pooled-window version).
#
# Locked design (same as compute_qnfs.py):
#   - Fenwick base (shot-on-goal + goal + missed-shot)
#   - Regular season, strict 5v5 ES (situation_code 1551), period_type REG
#   - Net-front zone = CNFI ∪ MNFI (per NFI/scripts/02_zones_and_rebound_confirm.py)
#   - All goalie appearances counted (relief included)
#   - MIN 3 NF shots per game-goalie row to count
#   - xG = per-season per-zone league Fenwick goal rate × shots faced
#   - Threshold: GSAx_NF >= 0 = quality save game
#
# Per-season additions:
#   - One row per (goalie_id, season)
#   - Skip seasons with GP < 10 for that goalie

import pandas as pd
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint

# ---- CONFIG ----
SHOT_FILE  = Path("/Users/ashgarg/Documents/HockeyROI/Data/nhl_shot_events.v2.csv")
NAMES_FILE = Path("/Users/ashgarg/Documents/HockeyROI/NFI/output/player_positions.csv")
OUT_FILE   = Path(__file__).parent.parent / "output" / "qnfs_per_season_2022-2026.csv"

SEASONS = [20222023, 20232024, 20242025, 20252026]
MIN_NF_SHOTS_PER_GAME = 3
QUALITY_THRESHOLD = 0.0
MIN_GP_PER_SEASON = 10

SAMPLE_NAMES = [
    "Connor Hellebuyck",
    "Igor Shesterkin",
    "Ilya Sorokin",
    "Filip Gustavsson",
    "Lukas Dostal",   # Top-third Wilson lo despite prior negative HockeyROI analysis
]

print("[compute_qnfs_per_season] per-(goalie, season) QNFS% with Wilson CIs")
print(f"Reading: {SHOT_FILE.name}")

# ---- LOAD ----
df = pd.read_csv(SHOT_FILE, low_memory=False)
print(f"Rows: {len(df):,}")

# ---- FILTER: REG SEASON, 5v5 ES, FENWICK, has goalie ----
df = df[df["season"].isin(SEASONS)]
df = df[df["game_type"] == "regular"]
df = df[df["period_type"] == "REG"]
df = df[df["situation_code"] == 1551]
df = df[df["event_type"].isin(["shot-on-goal", "goal", "missed-shot"])]
df = df[df["goalie_id"].notna()].copy()
df["goalie_id"] = df["goalie_id"].astype(int)
print(f"After filters: {len(df):,} rows")

# ---- ZONE TAGGING (canonical CNFI/MNFI) ----
def tag_zone(x_norm, y_norm):
    in_cnfi = (x_norm >= 74) & (x_norm <= 89) & (y_norm.abs() <= 9)
    in_mnfi = (x_norm >= 55) & (x_norm <  74) & (y_norm.abs() <= 15)
    z = pd.Series("OTHER", index=x_norm.index)
    z[in_mnfi] = "MNFI"
    z[in_cnfi] = "CNFI"
    return z

df["zone"] = tag_zone(df["x_coord_norm"], df["y_coord_norm"])
nf = df[df["zone"].isin(["CNFI", "MNFI"])].copy()

# ---- PER-SEASON PER-ZONE LEAGUE FENWICK GOAL RATES ----
zone_rates = (
    nf.groupby(["season", "zone"])
      .agg(faced=("is_goal", "size"), goals=("is_goal", "sum"))
      .assign(rate=lambda d: d["goals"] / d["faced"])
      .reset_index()
)
rates = zone_rates.pivot(index="season", columns="zone", values="rate").reset_index()
rates.columns.name = None
rates = rates.rename(columns={"CNFI": "rate_CNFI", "MNFI": "rate_MNFI"})

# ---- PER-GAME, PER-GOALIE NF FACED / GOALS / xG / GSAx ----
gg = (
    nf.groupby(["season", "game_id", "goalie_id", "zone"])
      .agg(faced=("is_goal", "size"), goals=("is_goal", "sum"))
      .reset_index()
)
gg = gg.pivot_table(
    index=["season", "game_id", "goalie_id"],
    columns="zone",
    values=["faced", "goals"],
    fill_value=0,
).reset_index()
gg.columns = ["_".join([c for c in col if c]).strip("_") for col in gg.columns]
for col in ["faced_CNFI", "faced_MNFI", "goals_CNFI", "goals_MNFI"]:
    if col not in gg.columns:
        gg[col] = 0

gg = gg.merge(rates, on="season", how="left")
gg["xG_NF"]    = gg["faced_CNFI"] * gg["rate_CNFI"] + gg["faced_MNFI"] * gg["rate_MNFI"]
gg["goals_NF"] = gg["goals_CNFI"] + gg["goals_MNFI"]
gg["GSAx_NF"]  = gg["xG_NF"] - gg["goals_NF"]
gg["NF_shots"] = gg["faced_CNFI"] + gg["faced_MNFI"]

gg = gg[gg["NF_shots"] >= MIN_NF_SHOTS_PER_GAME].copy()
gg["is_quality"] = (gg["GSAx_NF"] >= QUALITY_THRESHOLD).astype(int)
print(f"Game-goalie rows (NF_shots >= {MIN_NF_SHOTS_PER_GAME}): {len(gg):,}")

# ---- PER-(GOALIE, SEASON) AGGREGATION ----
per_season = (
    gg.groupby(["goalie_id", "season"])
      .agg(
          GP=("game_id", "size"),
          quality_games=("is_quality", "sum"),
          NF_shots=("NF_shots", "sum"),
      )
      .reset_index()
)
print(f"Goalie-season rows before GP filter: {len(per_season):,}")

per_season = per_season[per_season["GP"] >= MIN_GP_PER_SEASON].copy()
print(f"Goalie-season rows after GP >= {MIN_GP_PER_SEASON}: {len(per_season):,}")

per_season["QNFS_pct"] = per_season["quality_games"] / per_season["GP"]

# ---- WILSON 95% CI ----
lo, hi = proportion_confint(
    count=per_season["quality_games"].astype(int),
    nobs=per_season["GP"].astype(int),
    alpha=0.05,
    method="wilson",
)
per_season["QNFS_lo"] = lo
per_season["QNFS_hi"] = hi

# ---- NAME JOIN ----
names = pd.read_csv(NAMES_FILE)[["player_id", "player_name"]].rename(
    columns={"player_id": "goalie_id", "player_name": "goalie_name"}
)
per_season = per_season.merge(names, on="goalie_id", how="left")

# ---- Express proportions as percentages ----
for c in ("QNFS_pct", "QNFS_lo", "QNFS_hi"):
    per_season[c] = per_season[c] * 100.0

# ---- COLUMN ORDER ----
per_season = per_season[[
    "goalie_id", "goalie_name", "season", "GP",
    "quality_games", "QNFS_pct", "QNFS_lo", "QNFS_hi", "NF_shots",
]]

# ---- SORT: goalie_id, season asc (chronological per goalie) ----
per_season = per_season.sort_values(["goalie_id", "season"]).reset_index(drop=True)

# ---- WRITE ----
per_season.to_csv(OUT_FILE, index=False)
print(f"\nWrote: {OUT_FILE}")
print(f"Rows in CSV: {len(per_season):,}")
print(f"Unique goalies with at least one qualifying season: "
      f"{per_season['goalie_id'].nunique()}")

# ---- SAMPLE GOALIE YEAR-BY-YEAR TABLES ----
for name in SAMPLE_NAMES:
    rows = per_season[per_season["goalie_name"] == name].copy()
    print(f"\n=== {name} ===")
    if rows.empty:
        print("  (no qualifying seasons in window)")
        continue
    for c in ("QNFS_pct", "QNFS_lo", "QNFS_hi"):
        rows[c] = rows[c].round(2)
    print(rows[[
        "season", "GP", "quality_games",
        "QNFS_pct", "QNFS_lo", "QNFS_hi", "NF_shots",
    ]].to_string(index=False))
