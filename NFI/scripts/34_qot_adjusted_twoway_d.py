#!/usr/bin/env python3
"""
QoT-adjusted P4/P5 and two-way D score (D only).

Method:
  - For each qualifying D, regress P5_weighted on QoT_spatial_TNFI (OLS).
    Residual = how much offense the D produces beyond what their teammate
    quality predicts. Positive = better than expected.
  - Same for P4_weighted (defense; lower = better).
    Residual = how much defense beyond teammate prediction. Negative = better.
  - Two-way QoT-adj = z(P5_resid) - z(P4_resid)

Output: NFI/output/twoway_D_score_QoT_adjusted.csv  (NEW file, no overwrites)
"""
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"

twd = pd.read_csv(f"{OUT}/twoway_D_score.csv")
qct = pd.read_csv(f"{OUT}/player_qoc_qot.csv")
qct_d = qct[qct["position"]=="D"][["player_id","QoT_spatial_TNFI","QoT_spatial_CNFI"]]

# Merge — restrict to D in both files
m = twd.merge(qct_d, on="player_id", how="inner")
print(f"D in twoway_D_score.csv: {len(twd)}")
print(f"D in player_qoc_qot.csv: {len(qct_d)}")
print(f"D matched in both:       {len(m)}")

# OLS regression utility
def ols_residuals(x, y):
    """Returns residuals, slope, intercept, R^2."""
    n = len(x)
    x_mean = x.mean(); y_mean = y.mean()
    ss_xy = ((x-x_mean)*(y-y_mean)).sum()
    ss_xx = ((x-x_mean)**2).sum()
    slope = ss_xy / ss_xx if ss_xx>0 else 0.0
    intercept = y_mean - slope*x_mean
    yhat = intercept + slope*x
    resid = y - yhat
    ss_res = (resid**2).sum()
    ss_tot = ((y-y_mean)**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0.0
    return resid, slope, intercept, r2

x_qot = m["QoT_spatial_TNFI"].values

# P5 (offense, higher = better)
p5 = m["P5_weighted"].values
p5_resid, p5_slope, p5_intercept, p5_r2 = ols_residuals(x_qot, p5)
m["P5_QoT_adj"] = p5_resid

# P4 (defense, lower = better)
p4 = m["P4_weighted"].values
p4_resid, p4_slope, p4_intercept, p4_r2 = ols_residuals(x_qot, p4)
m["P4_QoT_adj"] = p4_resid

# Two-way (z of residuals)
def z(s): return (s - s.mean()) / s.std(ddof=0)

m["z_P5_QoT_adj"]      = z(m["P5_QoT_adj"])
m["z_P4_QoT_adj"]      = z(m["P4_QoT_adj"])
m["twoway_D_QoT_adj"]  = m["z_P5_QoT_adj"] - m["z_P4_QoT_adj"]

# Ranks
m["off_rank_QoT_adj"]    = m["P5_QoT_adj"].rank(ascending=False, method="min").astype(int)
m["def_rank_QoT_adj"]    = m["P4_QoT_adj"].rank(ascending=True,  method="min").astype(int)
m["twoway_D_rank_QoT_adj"]  = m["twoway_D_QoT_adj"].rank(ascending=False, method="min").astype(int)
m["twoway_D_rank_unadj"] = m["twoway_D_score"].rank(ascending=False, method="min").astype(int)
m["rank_delta"] = m["twoway_D_rank_unadj"] - m["twoway_D_rank_QoT_adj"]

m_sorted = m.sort_values("twoway_D_QoT_adj", ascending=False).reset_index(drop=True)

cols_out = ["player_id","player_name","es_toi_min","QoT_spatial_TNFI",
            "P5_weighted","P5_QoT_adj","z_P5_QoT_adj",
            "P4_weighted","P4_QoT_adj","z_P4_QoT_adj",
            "twoway_D_QoT_adj","twoway_D_score",
            "off_rank_QoT_adj","def_rank_QoT_adj",
            "twoway_D_rank_QoT_adj","twoway_D_rank_unadj","rank_delta"]
m_out = m_sorted[cols_out].copy()
m_out.to_csv(f"{OUT}/twoway_D_score_QoT_adjusted.csv", index=False)
print(f"\nWrote {OUT}/twoway_D_score_QoT_adjusted.csv  ({len(m_out)} D)")

# ============================================================
# Reporting
# ============================================================
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.width = 220
pd.options.display.max_columns = None

print("\n=== Regression statistics ===")
print(f"P5_weighted ~ QoT_spatial_TNFI:")
print(f"  slope:     {p5_slope:+.4f}")
print(f"  intercept: {p5_intercept:+.4f}")
print(f"  R²:        {p5_r2:.4f}  (variance in P5 explained by QoT)")
print(f"P4_weighted ~ QoT_spatial_TNFI:")
print(f"  slope:     {p4_slope:+.4f}")
print(f"  intercept: {p4_intercept:+.4f}")
print(f"  R²:        {p4_r2:.4f}  (variance in P4 explained by QoT)")

print("\n=== TOP 20 TWO-WAY D — QoT-adjusted ===")
top20 = m_out.head(20)
print(top20[["twoway_D_rank_QoT_adj","player_name","es_toi_min","QoT_spatial_TNFI",
              "P5_weighted","P5_QoT_adj","P4_weighted","P4_QoT_adj",
              "twoway_D_QoT_adj","twoway_D_score","twoway_D_rank_unadj","rank_delta"]]
      .to_string(index=False))

print("\n=== Spot-check: named D players ===")
named = ["Quinn Hughes","Roman Josi","Erik Karlsson","Adam Fox",
         "Cale Makar","Devon Toews","Evan Bouchard"]
for n in named:
    r = m_out[m_out["player_name"]==n]
    if len(r)==0:
        print(f"  {n}: NOT FOUND in matched data")
        continue
    r = r.iloc[0]
    print(f"\n  {n}:")
    print(f"    QoT_spatial_TNFI: {r['QoT_spatial_TNFI']:.3f}  (league mean ~3.275)")
    print(f"    P5: {r['P5_weighted']:.3f} → QoT-adj resid: {r['P5_QoT_adj']:+.4f}  (z {r['z_P5_QoT_adj']:+.2f})")
    print(f"    P4: {r['P4_weighted']:.3f} → QoT-adj resid: {r['P4_QoT_adj']:+.4f}  (z {r['z_P4_QoT_adj']:+.2f})")
    print(f"    Two-way unadjusted: {r['twoway_D_score']:+.3f}  rank {int(r['twoway_D_rank_unadj'])}")
    print(f"    Two-way QoT-adj:    {r['twoway_D_QoT_adj']:+.3f}  rank {int(r['twoway_D_rank_QoT_adj'])}")
    print(f"    Δ rank (unadj → adj): {int(r['rank_delta']):+d}  ({'improved' if r['rank_delta']>0 else 'dropped' if r['rank_delta']<0 else 'unchanged'})")

# Biggest movers overall
print("\n=== BIGGEST UPWARD MOVERS (rank improved most by QoT adj) ===")
movers_up = m_out.sort_values("rank_delta", ascending=False).head(15)
print(movers_up[["player_name","es_toi_min","QoT_spatial_TNFI",
                  "twoway_D_score","twoway_D_QoT_adj",
                  "twoway_D_rank_unadj","twoway_D_rank_QoT_adj","rank_delta"]]
      .to_string(index=False))

print("\n=== BIGGEST DOWNWARD MOVERS (rank dropped most by QoT adj) ===")
movers_dn = m_out.sort_values("rank_delta").head(15)
print(movers_dn[["player_name","es_toi_min","QoT_spatial_TNFI",
                  "twoway_D_score","twoway_D_QoT_adj",
                  "twoway_D_rank_unadj","twoway_D_rank_QoT_adj","rank_delta"]]
      .to_string(index=False))
