#!/usr/bin/env python3
"""
Step 8 - Horse race: team-level metrics vs standings points.
  - Univariate Pearson correlation (R, p, var explained) for every metric
  - Multivariate regression: all 7 pillars together (betas, R2, p-values)
  - Head-to-head: best TNFI variant vs best Corsi variant vs HD CF% vs each pillar

Also builds empirically derived composite NFI score from multivariate betas
at player + team level.
"""
import os, math
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT = f"{ROOT}/NFI/output"

team = pd.read_csv(f"{OUT}/metrics_team.csv")
player = pd.read_csv(f"{OUT}/metrics_player.csv")

# drop rows without points
team = team.dropna(subset=["points"])
print(f"Team-seasons with points: {len(team)}")

# Build team-level PILLAR aggregates (ES).
# Pillar proxies at team level:
#   P1: CNFI_CF per 60 (CNFI_CF / ES_TOI * 60) ~ scale with gp (points per game measure)
#   P2: CNFI_CA per 60 (defensive)
#   P3: MNFI_CF + FNFI_CF per 60 (offensive lane generation)
#   P4: all-zone against per 60 (team's defensive on-ice context)
#   P5: (MNFI+FNFI)_CF per 60 for defensemen - but team level uses team total
#   P6: goalie save% FNFI+MNFI weighted by faced (team goalie spread)
#   P7: goalie save% CNFI
# For team horse race, we'll use team-level proxies:

# Need team TOI — approximate as gp * 60 (minutes). But ES TOI < 60 min per game.
# Use gp * ~50 min (typical ES). We'll use CF-standardized per-60.
# Actually, for correlations, we can use per-game rates using gp.
team["ES_min"] = team["gp"] * 48.0  # approx ES minutes per game regulation

for col in ["CNFI_CF","MNFI_CF","FNFI_CF","CNFI_CA","MNFI_CA","FNFI_CA","TNFI_CF","TNFI_CA"]:
    team[f"{col}_per60"] = team[col] / team["ES_min"] * 60.0

# Pillar proxies at team level (each distinct to avoid collinearity).
# P3 and P5 both involve MNFI+FNFI individual shots but split by position;
# at team level we approximate them with different combinations.
team["P1_NF_F"]    = team["CNFI_CF_per60"]                          # net-front attempts taken
team["P2_DefF"]    = team["TNFI_CA_per60"]                          # TNFI against (lower better)
team["P3_OffF"]    = team["MNFI_CF_per60"]                          # medium-lane generation (F proxy)
team["P4_DefD"]    = team["CA"] / team["ES_min"] * 60.0             # all-zone CA
team["P5_OffD"]    = team["FNFI_CF_per60"]                          # far-lane generation (D proxy)
# goalie save pillars from pillar CSVs
g6 = pd.read_csv(f"{OUT}/pillar_6_goalie_FNFI_MNFI.csv")
g7 = pd.read_csv(f"{OUT}/pillar_7_goalie_CNFI.csv")
# aggregate team-season goalie save% weighted by faced: need team affiliation for each goalie
# Load goalie_team_lookup if available
goalie_team_path = f"{ROOT}/2026 posts/Goalies All/Benchmarks Goalies/Data/goalie_team_lookup.csv"
if os.path.exists(goalie_team_path):
    gt = pd.read_csv(goalie_team_path)
else:
    gt = None

# Without reliable team affiliation, approximate: use team-level save% from team_counts_by_state_zone
tc = pd.read_csv(f"{OUT}/team_counts_by_state_zone.csv")
tc_es = tc[tc["state"]=="ES"]
tc_agg = tc_es.groupby(["season","team"]).agg(
    mnfi_fa=("ag_att", lambda x: 0),  # placeholder
).reset_index()
# Properly compute: team saves = team goalie saves against team shots against (from shooting_team's perspective)
# Use tc where team is the DEFENDING team; ag_att = shots faced; ag_gl = goals allowed
tc_g6 = tc_es[tc_es["zone"].isin(["FNFI","MNFI"])].groupby(["season","team"]).agg(
    faced6=("ag_att","sum"), goals6=("ag_gl","sum")
).reset_index()
tc_g6["P6_SvPct"] = 1 - tc_g6["goals6"]/tc_g6["faced6"]
tc_g7 = tc_es[tc_es["zone"]=="CNFI"].groupby(["season","team"]).agg(
    faced7=("ag_att","sum"), goals7=("ag_gl","sum")
).reset_index()
tc_g7["P7_SvPct"] = 1 - tc_g7["goals7"]/tc_g7["faced7"]
team = team.merge(tc_g6[["season","team","P6_SvPct"]], on=["season","team"], how="left")
team = team.merge(tc_g7[["season","team","P7_SvPct"]], on=["season","team"], how="left")

# Univariate correlation
print("\n=== Univariate correlations with points ===")
metrics = [
    "CF_pct","CF_score_adj_pct","CF_zone_adj_pct","CF_score_zone_adj_pct",
    "HD_CF_pct","HD_CF_score_adj_pct","HD_CF_score_zone_adj_pct",
    "TNFI_pct","TNFI_score_adj_pct","TNFI_zone_adj_pct","TNFI_score_zone_adj_pct",
    "CNFI_pct","MNFI_pct","FNFI_pct",
    "P1_NF_F","P2_DefF","P3_OffF","P4_DefD","P5_OffD","P6_SvPct","P7_SvPct",
]
corr_rows = []
for m in metrics:
    if m not in team.columns: continue
    x = team[m].values.astype(float)
    y = team["points"].values.astype(float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 10: continue
    r, p = stats.pearsonr(x[mask], y[mask])
    corr_rows.append({"metric":m,"R":round(r,4),"p":round(p,6),"var_explained":round(r*r,4),"n":int(mask.sum())})
corr_df = pd.DataFrame(corr_rows).sort_values("var_explained", ascending=False)
corr_df.to_csv(f"{OUT}/horse_race_univariate.csv", index=False)
print(corr_df.to_string(index=False))

# Multivariate: 7 pillars
print("\n=== Multivariate: 7 pillars ===")
pillar_cols = ["P1_NF_F","P2_DefF","P3_OffF","P4_DefD","P5_OffD","P6_SvPct","P7_SvPct"]
sub = team[["season","team","points"] + pillar_cols].dropna()
X = sub[pillar_cols].values
# z-score
Xz = (X - X.mean(axis=0))/X.std(axis=0,ddof=0)
y = sub["points"].values
Xz1 = np.column_stack([np.ones(len(Xz)), Xz])
# OLS via pseudo-inverse to handle any near-singularities
XtX = Xz1.T @ Xz1
# ridge regularization (tiny) for stability
lam = 1e-6 * np.trace(XtX) / XtX.shape[0]
XtX_reg = XtX + lam*np.eye(XtX.shape[0])
beta = np.linalg.solve(XtX_reg, Xz1.T @ y)
yhat = Xz1 @ beta
ss_res = ((y-yhat)**2).sum()
ss_tot = ((y-y.mean())**2).sum()
r2 = 1 - ss_res/ss_tot
n, k = len(y), len(pillar_cols)
r2_adj = 1 - (1-r2)*(n-1)/(n-k-1)
# standard errors, t, p
mse = ss_res / (n-k-1)
var_beta = mse * np.linalg.inv(XtX_reg)
se = np.sqrt(np.diag(var_beta))
tstat = beta/se
pval = 2*(1-stats.t.cdf(np.abs(tstat), df=n-k-1))

print(f"R^2 = {r2:.4f}   adj R^2 = {r2_adj:.4f}   n = {n}")
print(f"{'term':<12}{'beta':>10}{'se':>10}{'t':>8}{'p':>10}")
pnames = ["intercept"] + pillar_cols
mv_rows = []
for i,(nm,b,s,t_,pv) in enumerate(zip(pnames,beta,se,tstat,pval)):
    print(f"{nm:<12}{b:>10.4f}{s:>10.4f}{t_:>8.3f}{pv:>10.4f}")
    mv_rows.append({"term":nm,"beta":b,"se":s,"t":t_,"p":pv})
mv_df = pd.DataFrame(mv_rows)
mv_df.attrs["r2"] = r2
mv_df.to_csv(f"{OUT}/horse_race_multivariate.csv", index=False)

# ---- Composite NFI score (player + team) using standardized betas ----
# empirical weights = coefficients from pillar regression (excluding intercept)
w = dict(zip(pillar_cols, beta[1:]))
# team composite
team_mean = {c: team[c].mean() for c in pillar_cols}
team_std  = {c: team[c].std(ddof=0) for c in pillar_cols}
team["NFI_composite"] = 0.0
for c in pillar_cols:
    z = (team[c] - team_mean[c]) / (team_std[c] if team_std[c]>0 else 1)
    team["NFI_composite"] += w[c] * z

team[["season","team","points","NFI_composite"] + pillar_cols].to_csv(
    f"{OUT}/team_composite_NFI.csv", index=False)

# player composite using analogous pillars at player level
# pl pillars:
toi = pd.read_csv(f"{OUT}/player_toi.csv")
pos = pd.read_csv(f"{OUT}/player_positions.csv")
counts = pd.read_csv(f"{OUT}/player_counts_by_state_zone.csv")
# Compute per-player per-60 proxies (ES)
es = counts[counts["state"]=="ES"].copy()
es = es.merge(toi[["player_id","toi_ES_sec"]], on="player_id", how="left")
es["toi_ES_min"] = es["toi_ES_sec"]/60.0
# pivot by zone
wide = es.pivot_table(index=["player_id","toi_ES_min"],
                     columns="zone",
                     values=["ind_att","onice_for_att","onice_ag_att"],
                     fill_value=0).reset_index()
wide.columns = ["_".join([str(c) for c in col if c]).strip("_") for col in wide.columns.values]

def safe(col):
    return wide[col] if col in wide.columns else 0

# Normalize per 60 ES minutes
mins = wide["toi_ES_min"].replace(0, np.nan)
wide["P1_NF_F"] = safe("ind_att_CNFI") / mins * 60
wide["P2_DefF"] = (safe("onice_ag_att_CNFI")+safe("onice_ag_att_MNFI")+safe("onice_ag_att_FNFI")) / mins * 60
wide["P3_OffF"] = (safe("ind_att_MNFI")+safe("ind_att_FNFI")) / mins * 60
wide["P4_DefD"] = (safe("onice_ag_att_CNFI")+safe("onice_ag_att_MNFI")+safe("onice_ag_att_FNFI")
                  + safe("onice_ag_att_Wide") + safe("onice_ag_att_lane_other")) / mins * 60
wide["P5_OffD"] = (safe("onice_for_att_MNFI")+safe("onice_for_att_FNFI")) / mins * 60

# Merge position and TOI
wide = wide.merge(pos[["player_id","pos_group","player_name"]], on="player_id", how="left")
wide = wide.merge(toi[["player_id","toi_total_min"]], on="player_id", how="left")
wide = wide[wide["toi_total_min"]>=500].copy()

# For P6/P7 at player level for goalies
g6 = pd.read_csv(f"{OUT}/pillar_6_goalie_FNFI_MNFI.csv")
g7 = pd.read_csv(f"{OUT}/pillar_7_goalie_CNFI.csv")
if len(g6):
    g6m = g6.groupby("goalie_id").apply(lambda x: (x["save_pct"]*x["faced"]).sum()/x["faced"].sum()).reset_index(name="P6_SvPct")
else:
    g6m = pd.DataFrame(columns=["goalie_id","P6_SvPct"])
if len(g7):
    g7m = g7.groupby("goalie_id").apply(lambda x: (x["save_pct"]*x["faced"]).sum()/x["faced"].sum()).reset_index(name="P7_SvPct")
else:
    g7m = pd.DataFrame(columns=["goalie_id","P7_SvPct"])
g6m.columns = ["player_id","P6_SvPct"]; g7m.columns = ["player_id","P7_SvPct"]
wide = wide.merge(g6m, on="player_id", how="left").merge(g7m, on="player_id", how="left")
wide["P6_SvPct"] = wide["P6_SvPct"].fillna(np.nan)
wide["P7_SvPct"] = wide["P7_SvPct"].fillna(np.nan)

# composite (z-score players, weighted by team-level betas)
pl_mean = {c: wide[c].mean(skipna=True) for c in pillar_cols}
pl_std = {c: wide[c].std(ddof=0,skipna=True) for c in pillar_cols}
wide["NFI_composite"] = 0.0
for c in pillar_cols:
    if pl_std[c] and pl_std[c]>0:
        z = (wide[c] - pl_mean[c]) / pl_std[c]
    else:
        z = 0
    wide["NFI_composite"] = wide["NFI_composite"] + w[c] * z.fillna(0)

wide[["player_id","player_name","pos_group","toi_total_min","NFI_composite"] + pillar_cols].to_csv(
    f"{OUT}/player_composite_NFI.csv", index=False)

print(f"\nWrote team/player composite NFI CSVs.")
print("Weights (from multivariate regression):")
for k,v in w.items(): print(f"  {k}: {v:.4f}")
