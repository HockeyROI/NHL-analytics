# NFI/goalie_consistency/scripts/net_front_consistency_design_diagnostic.py
#
# Purpose: Lock the design of NFI Net-Front Save Consistency.
# Run 1 (this version): print GP floor sweep, exit. User picks floor.
# Run 2: with chosen floor, fit GMM on filtered pool, report bucket edges,
#        then compute per-goalie bucket counts on FULL pool.
#
# NOT part of canonical NFI pipeline. Diagnostic only.
#
# Locked design specs:
#   - Fenwick base (zone rates and faced counts use shot-on-goal + goal + missed-shot)
#   - Count all goalie appearances including relief
#   - 3 NF shots minimum per game-goalie row to count
#   - GMM fit on workload-filtered pool only
#   - Per-goalie bucket counts computed on full pool (every goalie)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture

print("[diagnostic] net_front_consistency_design_diagnostic.py")

# ---- CONFIG ----
SHOT_FILE = Path("/Users/ashgarg/Documents/HockeyROI/Data/nhl_shot_events.v2.csv")
SEASONS = [20222023, 20232024, 20242025, 20252026]
GP_FLOORS = [15, 20, 25, 30, 35, 40]
MIN_NF_SHOTS_PER_GAME = 3

# Set to None for Run 1 (sweep only, exit).
# Set to a GP floor (e.g. 30) for Run 2 (full diagnostic).
GP_FLOOR_PER_SEASON_CHOSEN = 35

# ---- LOAD ----
print(f"Reading: {SHOT_FILE.name}")
df = pd.read_csv(SHOT_FILE, low_memory=False)
print(f"Rows: {len(df):,}")

required = {
    "season", "game_id", "game_type", "period_type",
    "situation_code", "event_type",
    "goalie_id", "x_coord_norm", "y_coord_norm", "is_goal",
}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing expected columns: {missing}.")

# ---- FILTER: REG SEASON, 5v5 ES, FENWICK, has goalie ----
# Fenwick = shot-on-goal + goal + missed-shot (excludes blocks)
before = len(df)
df = df[df["season"].isin(SEASONS)]
df = df[df["game_type"] == "regular"]
df = df[df["period_type"] == "REG"]
df = df[df["situation_code"] == 1551]
df = df[df["event_type"].isin(["shot-on-goal", "goal", "missed-shot"])]
df = df[df["goalie_id"].notna()]
print(f"After filters (reg, 5v5 ES, Fenwick, has goalie): "
      f"{len(df):,} rows ({before-len(df):,} dropped)")

# ---- ZONE TAGGING (canonical CNFI/MNFI from 02_zones_and_rebound_confirm.py) ----
def tag_zone(x_norm, y_norm):
    in_cnfi = (x_norm >= 74) & (x_norm <= 89) & (y_norm.abs() <= 9)
    in_mnfi = (x_norm >= 55) & (x_norm < 74) & (y_norm.abs() <= 15)
    z = pd.Series("OTHER", index=x_norm.index)
    z[in_mnfi] = "MNFI"
    z[in_cnfi] = "CNFI"
    return z

df["zone"] = tag_zone(df["x_coord_norm"], df["y_coord_norm"])
nf = df[df["zone"].isin(["CNFI", "MNFI"])].copy()
print(f"Net-front Fenwick shots (CNFI+MNFI): "
      f"{len(nf):,} ({len(nf)/len(df):.1%} of all shots)\n")

# ---- LEAGUE ZONE GOAL RATES, FENWICK DENOMINATOR (per season, per zone) ----
zone_rates = (
    nf.groupby(["season", "zone"])
      .agg(faced=("is_goal", "size"), goals=("is_goal", "sum"))
      .assign(rate=lambda d: d["goals"] / d["faced"])
      .reset_index()
)
print("=== LEAGUE FENWICK GOAL RATES (per season, per zone) ===")
print(zone_rates.to_string(index=False), "\n")

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

rates = zone_rates.pivot(index="season", columns="zone", values="rate").reset_index()
rates.columns.name = None
rates = rates.rename(columns={"CNFI": "rate_CNFI", "MNFI": "rate_MNFI"})
gg = gg.merge(rates, on="season", how="left")

gg["xG_NF"] = gg["faced_CNFI"] * gg["rate_CNFI"] + gg["faced_MNFI"] * gg["rate_MNFI"]
gg["goals_NF"] = gg["goals_CNFI"] + gg["goals_MNFI"]
gg["GSAx_NF"] = gg["xG_NF"] - gg["goals_NF"]
gg["NF_shots"] = gg["faced_CNFI"] + gg["faced_MNFI"]

# Apply per-game minimum NF shots threshold
before_min = len(gg)
gg = gg[gg["NF_shots"] >= MIN_NF_SHOTS_PER_GAME].copy()
print(f"Game-goalie rows: {len(gg):,} "
      f"({before_min - len(gg):,} dropped by NF_shots >= {MIN_NF_SHOTS_PER_GAME})\n")

print("=== PER-GAME NF GSAx — LEAGUE-WIDE (full pool) ===")
print(gg["GSAx_NF"].describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]))
print(f"  mean: {gg['GSAx_NF'].mean():+.4f}")
print(f"  std:  {gg['GSAx_NF'].std():.4f}\n")

# ---- QUESTION 1: PER-SEASON GP FLOOR SWEEP ----
# A goalie qualifies if their max GP in any single season is >= floor.
per_season_gp = (
    gg.groupby(["goalie_id", "season"])["game_id"]
      .nunique()
      .reset_index(name="GP")
)
print("=== PER-SEASON GP DISTRIBUTION (one row per goalie-season) ===")
print(f"  rows: {len(per_season_gp):,} goalie-season pairs")
print(per_season_gp["GP"].describe(percentiles=[.1, .25, .5, .75, .9]))
print()

hist_bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, float("inf")]
hist_labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30",
               "30-40", "40-50", "50-60", "60-70", "70+"]
per_season_gp["bin"] = pd.cut(
    per_season_gp["GP"], bins=hist_bins, labels=hist_labels,
    right=False, include_lowest=True,
)
hist = per_season_gp["bin"].value_counts().reindex(hist_labels, fill_value=0)
print("=== PER-SEASON GP HISTOGRAM (goalie-seasons per bin) ===")
for b, n in hist.items():
    print(f"  {b:>6}: {n}")
print()

goalie_max_gp = (
    per_season_gp.groupby("goalie_id")["GP"]
                 .max()
                 .reset_index(name="max_GP_in_a_season")
)
print("=== PER-SEASON GP FLOOR SWEEP ===")
print(f"  Goalies total in pool: {len(goalie_max_gp)}")
for floor in GP_FLOORS:
    qualified = goalie_max_gp[goalie_max_gp["max_GP_in_a_season"] >= floor]
    print(f"  GP-in-any-single-season >= {floor:>2}: {len(qualified)} goalies")
print()

# ---- GATE: if no floor chosen, exit here (Run 1) ----
if GP_FLOOR_PER_SEASON_CHOSEN is None:
    print("=== RUN 1 COMPLETE ===")
    print("Pick a GP floor from the sweep above, set GP_FLOOR_PER_SEASON_CHOSEN, rerun.")
    raise SystemExit(0)

# ---- RUN 2: GMM FIT ON WORKLOAD-FILTERED POOL ----
print(f"=== RUN 2: GP_FLOOR_PER_SEASON_CHOSEN = {GP_FLOOR_PER_SEASON_CHOSEN} ===\n")

qualified_ids = set(goalie_max_gp[
    goalie_max_gp["max_GP_in_a_season"] >= GP_FLOOR_PER_SEASON_CHOSEN
]["goalie_id"])
print(f"Qualified goalies for GMM fit: {len(qualified_ids)}")

gg_filt = gg[gg["goalie_id"].isin(qualified_ids)].copy()
print(f"Game-goalie rows in filtered pool: {len(gg_filt):,} "
      f"(of {len(gg):,} total)\n")

x = gg_filt["GSAx_NF"].values.reshape(-1, 1)

results = {}
for k in [2, 3]:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    gmm.fit(x)
    results[k] = {
        "bic": gmm.bic(x),
        "means": sorted(gmm.means_.flatten()),
        "weights": [w for _, w in sorted(zip(gmm.means_.flatten(),
                                              gmm.weights_))],
        "model": gmm,
    }

print("=== GMM FIT (filtered pool) ===")
for k in [2, 3]:
    r = results[k]
    print(f"  k={k}: BIC={r['bic']:,.0f} | "
          f"means={[f'{m:+.3f}' for m in r['means']]} | "
          f"weights={[f'{w:.2%}' for w in r['weights']]}")
better = 3 if results[3]["bic"] < results[2]["bic"] else 2
print(f"  --> BIC prefers k={better}\n")

# ---- BUCKET EDGES ----
if better == 3:
    print("=== BUCKET EDGES (from k=3 fit) ===")
    gmm3 = results[3]["model"]
    sorted_x = np.sort(x.flatten())
    grid = np.linspace(sorted_x[0], sorted_x[-1], 5000).reshape(-1, 1)
    labels = gmm3.predict(grid)
    order = np.argsort(gmm3.means_.flatten())
    relabel = {old: new for new, old in enumerate(order)}
    labels = np.array([relabel[l] for l in labels])
    transitions = np.where(np.diff(labels) != 0)[0]
    edges = sorted([grid[t][0] for t in transitions])
    bad_edge, qual_edge = edges[0], edges[-1]
    print(f"  Bad/Neutral boundary:     {bad_edge:+.3f}")
    print(f"  Neutral/Quality boundary: {qual_edge:+.3f}\n")

    # Bucket split on filtered pool (sanity check)
    bad = (gg_filt["GSAx_NF"] <= bad_edge).mean()
    qual = (gg_filt["GSAx_NF"] >= qual_edge).mean()
    neutral = 1 - bad - qual
    print(f"  Filtered pool split: bad {bad:.1%} | "
          f"neutral {neutral:.1%} | quality {qual:.1%}\n")

    # ---- PER-GOALIE BUCKET COUNTS ON FULL POOL ----
    gg["bucket"] = "neutral"
    gg.loc[gg["GSAx_NF"] <= bad_edge, "bucket"] = "bad"
    gg.loc[gg["GSAx_NF"] >= qual_edge, "bucket"] = "quality"

    goalie_buckets = (
        gg.groupby(["goalie_id", "bucket"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )
    for col in ["quality", "neutral", "bad"]:
        if col not in goalie_buckets.columns:
            goalie_buckets[col] = 0
    goalie_buckets["GP"] = (
        goalie_buckets["quality"] + goalie_buckets["neutral"] + goalie_buckets["bad"]
    )
    goalie_buckets["QNFS_pct"] = (
        goalie_buckets["quality"] / goalie_buckets["GP"]
    )
    goalie_buckets["qualified"] = goalie_buckets["goalie_id"].isin(qualified_ids)

    # Sort: qualified first, by QNFS% descending
    goalie_buckets = goalie_buckets.sort_values(
        ["qualified", "QNFS_pct"], ascending=[False, False]
    )

    out_path = Path(__file__).parent.parent / "output" / "nf_consistency_diagnostic.csv"
    goalie_buckets.to_csv(out_path, index=False)
    print(f"=== PER-GOALIE BUCKET COUNTS (full pool) ===")
    print(f"  Saved to: {out_path}")
    print(f"  Rows: {len(goalie_buckets)} (qualified: {goalie_buckets['qualified'].sum()})\n")
    print("Top 10 qualified goalies by QNFS%:")
    print(goalie_buckets[goalie_buckets['qualified']].head(10).to_string(index=False))
else:
    print("=== BUCKET EDGES: SKIPPED ===")
    print("  BIC prefers 2 buckets — three-bucket design not supported by data.")
    print("  Honest move: binary metric on NF GSAx (still better than QS%).")
