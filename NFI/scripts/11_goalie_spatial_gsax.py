#!/usr/bin/env python3
"""
Two goalie metrics (ES regulation, 5-season pool, min 300 shots faced TOTAL):

  Option 1 - Spatial Save% Index: CNFI + MNFI combined save% per goalie with Wilson CIs.
  Option 2 - NFI-GSAx: goals saved above expected using empirically derived zone rates:
             CNFI 13.08%, MNFI 10.02%, FNFI 3.37%.

Min threshold: 300 shots faced TOTAL across CNFI+MNFI (for Option 1) or CNFI+MNFI+FNFI
(for Option 2, since all three contribute).
"""
import math
import pandas as pd

OUT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/NFI/output"
MIN_SHOTS = 300

# zone conversion rates from zone_conversion_rates.csv (ES regulation 5-season, per attempt).
# These are goals/attempts where attempts = SOG+miss+block+goal.
RATE = {"CNFI": 0.13075, "MNFI": 0.10020, "FNFI": 0.03370}
# Per-faced-shot rates for a properly calibrated xG (goals/(SOG+goal)). Higher than RATE
# because missed/blocked shots that never reach the goalie are excluded from denom.
RATE_FACED = {"CNFI": 0.18130, "MNFI": 0.15180, "FNFI": 0.05120}

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0.0,c-h), min(1.0,c+h))

# ---- load ----
sh = pd.read_csv(f"{OUT}/shots_tagged.csv")
pos = pd.read_csv(f"{OUT}/player_positions.csv")
name_map = dict(zip(pos["player_id"].astype(int), pos["player_name"]))

# ES regulation, goalie-faced shots (SOG + goal; goalie present)
f = sh[(sh["state"]=="ES") & sh["event_type"].isin(["shot-on-goal","goal"]) & sh["goalie_id"].notna()].copy()
f["goalie_id"] = f["goalie_id"].astype(int)
print(f"ES faced-net shots: {len(f):,}")

# ---- Option 1: Spatial Save% Index (CNFI+MNFI) ----
dz = f[f["zone"].isin(["CNFI","MNFI"])]
sp = dz.groupby("goalie_id").agg(
    faced=("is_goal_i","size"),
    goals=("is_goal_i","sum"),
).reset_index()
sp = sp[sp["faced"]>=MIN_SHOTS].copy()

def row_wilson(r):
    p, lo, hi = wilson(r["faced"]-r["goals"], r["faced"])
    return pd.Series({"spatial_save_pct":round(p,4),
                      "sv_lo":round(lo,4), "sv_hi":round(hi,4)})
sp[["spatial_save_pct","sv_lo","sv_hi"]] = sp.apply(row_wilson, axis=1)
sp["goalie_name"] = sp["goalie_id"].map(name_map)
sp = sp.sort_values("spatial_save_pct", ascending=False).reset_index(drop=True)
sp["rank_spatial"] = sp.index + 1
sp = sp[["rank_spatial","goalie_id","goalie_name","faced","goals",
        "spatial_save_pct","sv_lo","sv_hi"]]
sp.to_csv(f"{OUT}/goalie_spatial_savepct.csv", index=False)
print(f"\nOption 1 Spatial Save% Index: {len(sp)} qualifying goalies (min {MIN_SHOTS} CNFI+MNFI)")
print(sp.head(15).to_string(index=False))

# ---- Option 2: NFI-GSAx (CNFI+MNFI+FNFI) ----
zs = f[f["zone"].isin(["CNFI","MNFI","FNFI"])].copy()
# per goalie per zone faced + goals
g = zs.groupby(["goalie_id","zone"]).agg(
    faced=("is_goal_i","size"),
    goals=("is_goal_i","sum"),
).reset_index()
wide = g.pivot_table(index="goalie_id", columns="zone", values=["faced","goals"], fill_value=0)
wide.columns = [f"{a}_{b}" for a,b in wide.columns]
wide = wide.reset_index()
# ensure all zone cols exist
for c in ["faced_CNFI","faced_MNFI","faced_FNFI","goals_CNFI","goals_MNFI","goals_FNFI"]:
    if c not in wide.columns: wide[c] = 0

wide["total_faced"] = wide["faced_CNFI"]+wide["faced_MNFI"]+wide["faced_FNFI"]
wide["total_goals"] = wide["goals_CNFI"]+wide["goals_MNFI"]+wide["goals_FNFI"]
wide = wide[wide["total_faced"]>=MIN_SHOTS].copy()

# Primary GSAx using user-specified per-attempt rates
wide["xG"] = (wide["faced_CNFI"]*RATE["CNFI"]
              + wide["faced_MNFI"]*RATE["MNFI"]
              + wide["faced_FNFI"]*RATE["FNFI"])
wide["NFI_GSAx"] = wide["xG"] - wide["total_goals"]   # positive = goals saved vs expected

# Calibrated GSAx using per-faced rates (goals/(SOG+goal)) — more suitable for
# goalie-only benchmarking since goalies never "face" blocks/misses.
wide["xG_calibrated"] = (wide["faced_CNFI"]*RATE_FACED["CNFI"]
                          + wide["faced_MNFI"]*RATE_FACED["MNFI"]
                          + wide["faced_FNFI"]*RATE_FACED["FNFI"])
wide["NFI_GSAx_calibrated"] = wide["xG_calibrated"] - wide["total_goals"]

wide["goalie_name"] = wide["goalie_id"].map(name_map)
wide = wide.sort_values("NFI_GSAx", ascending=False).reset_index(drop=True)
wide["rank_gsax"] = wide.index + 1
wide["rank_gsax_calibrated"] = wide["NFI_GSAx_calibrated"].rank(method="min", ascending=False).astype(int)
gs = wide[["rank_gsax","rank_gsax_calibrated","goalie_id","goalie_name",
           "faced_CNFI","faced_MNFI","faced_FNFI",
           "total_faced","total_goals","xG","NFI_GSAx",
           "xG_calibrated","NFI_GSAx_calibrated"]].copy()
for c in ["xG","NFI_GSAx","xG_calibrated","NFI_GSAx_calibrated"]:
    gs[c] = gs[c].round(2)
gs.to_csv(f"{OUT}/goalie_nfi_gsax.csv", index=False)
print(f"\nOption 2 NFI-GSAx: {len(gs)} qualifying goalies (min {MIN_SHOTS} TNFI)")
print(gs.head(15).to_string(index=False))

# ---- Side-by-side comparison ----
cmp = sp[["goalie_id","goalie_name","rank_spatial","faced","goals","spatial_save_pct","sv_lo","sv_hi"]]\
        .merge(gs[["goalie_id","rank_gsax","rank_gsax_calibrated","total_faced","total_goals",
                   "xG","NFI_GSAx","xG_calibrated","NFI_GSAx_calibrated"]],
               on="goalie_id", how="outer")
cmp = cmp.rename(columns={"faced":"faced_CM","goals":"goals_CM","total_faced":"faced_TNFI","total_goals":"goals_TNFI"})

# Fill name for rows that only appear in gsax (faced<300 CNFI+MNFI but enough TNFI)
cmp["goalie_name"] = cmp["goalie_name"].fillna(cmp["goalie_id"].map(name_map))
cmp["rank_diff"] = (cmp["rank_spatial"] - cmp["rank_gsax"]).abs()
# Flag significant disagreement: rank differs by > 15 positions OR one metric qualifies + other doesn't
cmp["significant_disagreement"] = (
    (cmp["rank_diff"] > 15) |
    cmp["rank_spatial"].isna() |
    cmp["rank_gsax"].isna()
)

cmp = cmp.sort_values("rank_spatial", na_position="last").reset_index(drop=True)
cmp = cmp[["goalie_id","goalie_name","rank_spatial","spatial_save_pct","sv_lo","sv_hi",
           "faced_CM","goals_CM","rank_gsax","xG","NFI_GSAx",
           "rank_gsax_calibrated","xG_calibrated","NFI_GSAx_calibrated",
           "faced_TNFI","goals_TNFI",
           "rank_diff","significant_disagreement"]]
cmp.to_csv(f"{OUT}/goalie_metric_comparison.csv", index=False)

print(f"\nComparison: {len(cmp)} goalies (union of both lists)")
print(f"Both metrics qualify: {cmp[['rank_spatial','rank_gsax']].notna().all(axis=1).sum()}")
print(f"Significant disagreement (|rank_diff|>15 or only-one-qualifies): {cmp['significant_disagreement'].sum()}")

# Show flagged disagreements
flagged = cmp[cmp["significant_disagreement"]].copy()
print("\n=== Significant disagreements ===")
print(flagged[["goalie_name","rank_spatial","rank_gsax","spatial_save_pct",
               "NFI_GSAx","faced_CM","faced_TNFI","rank_diff"]].to_string(index=False))
