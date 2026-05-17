# NFI/goalie_consistency/scripts/compute_qnfs.py
#
# Production: Quality Net-Front Save percentage (QNFS%).
# Share of 5v5 ES regular-season goalie appearances with NF GSAx >= 0.
#
# Locked design (same as net_front_consistency_design_diagnostic.py):
#   - Fenwick base (shot-on-goal + goal + missed-shot)
#   - Regular season, strict 5v5 ES (situation_code 1551), period_type REG
#   - Net-front zone = CNFI ∪ MNFI (per NFI/scripts/02_zones_and_rebound_confirm.py)
#   - All goalie appearances counted (relief included)
#   - MIN 3 NF shots per game-goalie row to count
#   - xG = per-season per-zone league Fenwick goal rate × shots faced
#   - Threshold: GSAx_NF >= 0 = quality save game (binary metric)
#   - Qualified flag: GP >= 25 in any single season within the window

import pandas as pd
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint

# ---- CONFIG ----
SHOT_FILE  = Path("/Users/ashgarg/Documents/HockeyROI/Data/nhl_shot_events.v2.csv")
NAMES_FILE = Path("/Users/ashgarg/Documents/HockeyROI/NFI/output/player_positions.csv")
OUT_FILE   = Path(__file__).parent.parent / "output" / "qnfs_2022-2026.csv"

SEASONS = [20222023, 20232024, 20242025, 20252026]
MIN_NF_SHOTS_PER_GAME = 3
QUALITY_THRESHOLD = 0.0
QUALIFIED_GP_PER_SEASON = 25

print("[compute_qnfs] computing QNFS% (net-front Fenwick GSAx >= 0)")
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

# ---- PER-GOALIE AGGREGATION (pooled across all seasons in window) ----
per_goalie = (
    gg.groupby("goalie_id")
      .agg(
          GP=("game_id", "size"),
          quality_games=("is_quality", "sum"),
          NF_shots=("NF_shots", "sum"),
      )
      .reset_index()
)
per_goalie["QNFS_pct"] = per_goalie["quality_games"] / per_goalie["GP"]

# ---- WILSON 95% CI on the proportion (quality_games / GP) ----
lo, hi = proportion_confint(
    count=per_goalie["quality_games"].astype(int),
    nobs=per_goalie["GP"].astype(int),
    alpha=0.05,
    method="wilson",
)
per_goalie["QNFS_lo"] = lo
per_goalie["QNFS_hi"] = hi

# ---- QUALIFIED FLAG: GP >= 25 in any single season ----
per_season_gp = (
    gg.groupby(["goalie_id", "season"])["game_id"]
      .nunique()
      .reset_index(name="GP_season")
)
max_season_gp = (
    per_season_gp.groupby("goalie_id")["GP_season"]
                 .max()
                 .reset_index(name="max_season_GP")
)
per_goalie = per_goalie.merge(max_season_gp, on="goalie_id", how="left")
per_goalie["qualified"] = per_goalie["max_season_GP"] >= QUALIFIED_GP_PER_SEASON
per_goalie = per_goalie.drop(columns=["max_season_GP"])

# ---- NAME JOIN (left merge — unmatched goalies keep blank name) ----
names = pd.read_csv(NAMES_FILE)[["player_id", "player_name"]].rename(
    columns={"player_id": "goalie_id", "player_name": "goalie_name"}
)
per_goalie = per_goalie.merge(names, on="goalie_id", how="left")

# ---- Express proportions as percentages ----
for c in ("QNFS_pct", "QNFS_lo", "QNFS_hi"):
    per_goalie[c] = per_goalie[c] * 100.0

# ---- SORT: qualified by QNFS_lo desc; unqualified by QNFS_pct desc ----
qualified   = per_goalie[ per_goalie["qualified"]].sort_values("QNFS_lo",  ascending=False)
unqualified = per_goalie[~per_goalie["qualified"]].sort_values("QNFS_pct", ascending=False)
per_goalie = pd.concat([qualified, unqualified], ignore_index=True)

# ---- Column order (goalie_name right after goalie_id) ----
per_goalie = per_goalie[[
    "goalie_id", "goalie_name", "GP", "quality_games", "NF_shots",
    "QNFS_pct", "QNFS_lo", "QNFS_hi", "qualified",
]]

# ---- WRITE ----
per_goalie.to_csv(OUT_FILE, index=False)
print(f"\nWrote: {OUT_FILE}")
print(f"Rows in CSV: {len(per_goalie):,}")
print(f"Qualified goalies: {int(per_goalie['qualified'].sum())}")

# ---- TOP 25 QUALIFIED (sorted by Wilson lower bound) ----
print("\n=== Top 25 qualified goalies by QNFS_lo (Wilson 95% lower bound) ===")
top25 = per_goalie[per_goalie["qualified"]].head(25).copy()
for c in ("QNFS_pct", "QNFS_lo", "QNFS_hi"):
    top25[c] = top25[c].round(2)
print(top25[[
    "goalie_name", "GP", "quality_games", "QNFS_pct",
    "QNFS_lo", "QNFS_hi", "NF_shots",
]].to_string(index=False))
