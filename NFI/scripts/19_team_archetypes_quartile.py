#!/usr/bin/env python3
"""
Quartile-split version of team construction archetypes.

Uses the existing team_construction_model.csv (160 team-seasons with
F_top6_twoway, D_top4_twoway, goalie GSAx/60, etc.).

Splits:
  High = top 25% of F_top6_twoway / D_top4_twoway across the 160 team-seasons
  Low  = bottom 25%
  Middle = unclassified, dropped

Archetypes:
  A: HighF + HighD
  B: HighF + LowD
  C: LowF  + HighD
  D: LowF  + LowD

Outputs:
  NFI/output/team_construction_model_quartile.csv  (full per-team-season
      with quartile flags + archetype assignment, including 'MIDDLE' rows)
  NFI/output/team_archetypes_quartile.csv          (archetype summary)
"""
import os, math
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT = f"{ROOT}/NFI/output"
SRC = f"{OUT}/team_construction_model.csv"

m = pd.read_csv(SRC)
print(f"Loaded {len(m)} team-seasons from {SRC}")

# Quartile thresholds
f_q25, f_q75 = m["F_top6_twoway"].quantile([0.25, 0.75])
d_q25, d_q75 = m["D_top4_twoway"].quantile([0.25, 0.75])
print(f"F_top6_twoway: Q25={f_q25:.4f}, Q75={f_q75:.4f}")
print(f"D_top4_twoway: Q25={d_q25:.4f}, Q75={d_q75:.4f}")

m["F_band"] = np.where(m["F_top6_twoway"] >= f_q75, "High",
              np.where(m["F_top6_twoway"] <= f_q25, "Low", "Middle"))
m["D_band"] = np.where(m["D_top4_twoway"] >= d_q75, "High",
              np.where(m["D_top4_twoway"] <= d_q25, "Low", "Middle"))

def archetype_q(row):
    f, d = row["F_band"], row["D_band"]
    if f == "Middle" or d == "Middle":
        return "MIDDLE"
    if f == "High" and d == "High": return "A_HighF_HighD"
    if f == "High" and d == "Low":  return "B_HighF_LowD"
    if f == "Low"  and d == "High": return "C_LowF_HighD"
    return "D_LowF_LowD"  # Low + Low

m["archetype_q"] = m.apply(archetype_q, axis=1)

# Goalie high/low (median split, recomputed within full 160)
g_med = m["starter_NFI_GSAx_per60"].median()
m["g_high"] = m["starter_NFI_GSAx_per60"] >= g_med
m["arche_g_q"] = m["archetype_q"] + "_" + np.where(m["g_high"], "Ghi", "Glo")

# Save full per-team-season output
keep_cols = ["season","team","conference","gp","points","points_pct","wins",
             "goal_diff","goal_diff_per_game","TNFI_pct","NFI_composite",
             "F_top6_twoway","F_band","D_top4_twoway","D_band",
             "starter_goalie_name","starter_NFI_GSAx_per60","g_high",
             "archetype_q","arche_g_q",
             "made_playoffs","round1_win","round2_win","cf_win","cup_win",
             "conf_rank","contender_top4_conf"]
m[keep_cols].to_csv(f"{OUT}/team_construction_model_quartile.csv", index=False)

# Archetype summary (drop MIDDLE)
arche = m[m["archetype_q"]!="MIDDLE"].copy()
summary = arche.groupby("archetype_q").agg(
    n=("team","count"),
    mean_points=("points","mean"),
    mean_goal_diff_per_game=("goal_diff_per_game","mean"),
    mean_TNFI_pct=("TNFI_pct","mean"),
    mean_starter_GSAx=("starter_NFI_GSAx_per60","mean"),
    playoff_rate=("made_playoffs","mean"),
    contender_rate=("contender_top4_conf","mean"),
    cup_rate=("cup_win","mean"),
).round(3)
summary.to_csv(f"{OUT}/team_archetypes_quartile.csv")

# ANOVA
groups = [arche[arche["archetype_q"]==a]["points"].values
          for a in sorted(arche["archetype_q"].unique())]
F_stat, anova_p = stats.f_oneway(*groups)

# Goalie split within archetype
g_split = arche.groupby(["archetype_q","g_high"]).agg(
    n=("team","count"),
    mean_points=("points","mean"),
    mean_goal_diff_per_game=("goal_diff_per_game","mean"),
    playoff_rate=("made_playoffs","mean"),
    contender_rate=("contender_top4_conf","mean"),
    cup_rate=("cup_win","mean"),
).round(3).reset_index()

# Hyman OLS: A vs B controlling for TNFI%
ab = arche[arche["archetype_q"].isin(["A_HighF_HighD","B_HighF_LowD"])].copy()
ab["is_A"] = (ab["archetype_q"]=="A_HighF_HighD").astype(int)
X = np.column_stack([np.ones(len(ab)), ab["is_A"].values, ab["TNFI_pct"].values])
y = ab["points"].values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
y_hat = X @ beta
resid = y - y_hat
n_, k_ = X.shape
sig2 = (resid**2).sum() / (n_ - k_)
cov = sig2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))
t = beta / se
from scipy.stats import t as t_dist
p_val = 2 * (1 - t_dist.cdf(np.abs(t), df=n_-k_))
hyman = pd.DataFrame({
    "var": ["intercept","is_A_vs_B","TNFI_pct"],
    "beta": beta, "se": se, "t": t, "p": p_val,
}).round(4)

# ---- Console ----
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 240
pd.options.display.max_columns = None

print("\n=== Quartile-split archetype summary ===")
print(summary.to_string())
print(f"\nClassified: {len(arche)} of 160 (Middle dropped: {(m['archetype_q']=='MIDDLE').sum()})")
print(f"ANOVA on points across 4 archetypes: F={F_stat:.3f}, p={anova_p:.5f}")

print("\n=== Goalie split within archetype ===")
print(g_split.to_string(index=False))

print("\n=== Hyman OLS (A vs B controlling for TNFI%) ===")
print(f"n_A={(ab['is_A']==1).sum()}, n_B={(ab['is_A']==0).sum()}, total={len(ab)}")
print(hyman.to_string(index=False))

# ---- Team-by-team listing per archetype ----
def fmt_season(s): return f"{str(s)[:4]}-{str(s)[6:]}"
def stage(r):
    if r["cup_win"]:        return "CUP"
    if r["cf_win"]:         return "Final"
    if r["round2_win"]:     return "CF"
    if r["round1_win"]:     return "R2"
    if r["made_playoffs"]:  return "R1"
    return "—"
arche["season_d"] = arche["season"].apply(fmt_season)
arche["stage"]    = arche.apply(stage, axis=1)

show_cols = ["season_d","team","points","conf_rank","stage",
             "F_top6_twoway","D_top4_twoway","TNFI_pct",
             "starter_goalie_name","starter_NFI_GSAx_per60"]

for arche_name in ["A_HighF_HighD","B_HighF_LowD","C_LowF_HighD","D_LowF_LowD"]:
    sub = arche[arche["archetype_q"]==arche_name]\
              .sort_values(["season","points"], ascending=[True, False])
    print(f"\n--- {arche_name} (n={len(sub)}) ---")
    print(sub[show_cols].to_string(index=False))

# Florida specific check
print("\n=== Where do FLA 2023-24 and 2024-25 land? ===")
fla = m[(m["team"]=="FLA") & (m["season"].isin([20232024, 20242025]))]
print(fla[["season","team","F_top6_twoway","F_band","D_top4_twoway","D_band",
           "archetype_q","points","cup_win"]].to_string(index=False))

print("\nFiles written:")
for f in ["team_construction_model_quartile.csv","team_archetypes_quartile.csv"]:
    print(f"  {OUT}/{f}")
