#!/usr/bin/env python3
"""
Step 10 - Finalize outputs:
  - team_level_all_metrics.csv (all pillars + NFI + traditional per season+team)
  - heatmap_savepct_with_inflection.csv (5x5 grid + flagged inflection points)
  - horse_race_summary.csv (ranked by R^2 with multivariate + head-to-head)
  - composite_player_team.csv (composite scores)
  - pillar_ci_flagging.csv (pillars failing 50% CI threshold)
"""
import os, math
import pandas as pd
import numpy as np
from scipy import stats

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT = f"{ROOT}/NFI/output"

# ---------- heat map + inflection flag ----------
heat = pd.read_csv(f"{OUT}/heatmap_save_pct_5x5.csv")
infl = pd.read_csv(f"{OUT}/inflection_points.csv")
# Flag rows that match center-lane inflection x-bins (y in [-15,15])
infl_xs = set(infl["x_bin"].tolist())
heat["is_inflection"] = heat.apply(
    lambda r: int(r["xb"] in infl_xs and -15 <= r["yb"] <= 15),
    axis=1
)
heat.to_csv(f"{OUT}/heatmap_savepct_with_inflection.csv", index=False)
print(f"heatmap: {len(heat)} bins  (inflection-flagged: {heat['is_inflection'].sum()})")

# ---------- team level all metrics ----------
team = pd.read_csv(f"{OUT}/metrics_team.csv")
comp = pd.read_csv(f"{OUT}/team_composite_NFI.csv")
# composite already includes pillar columns
pillar_cols = ["P1_NF_F","P2_DefF","P3_OffF","P4_DefD","P5_OffD","P6_SvPct","P7_SvPct"]
merge_cols = ["season","team","NFI_composite"] + pillar_cols
team = team.merge(comp[merge_cols], on=["season","team"], how="left")
team.to_csv(f"{OUT}/team_level_all_metrics.csv", index=False)
print(f"team_level_all_metrics: {len(team)} rows, {team.shape[1]} cols")

# ---------- pillar CI flagging ----------
# For each pillar CSV with per60_lo/per60_hi, check % of players where
# CI excludes zero (per60_lo > 0) -> "CI-confirmed above zero".
def ci_check(path, pillar_label, qualified=True):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    if "per60_lo" not in df.columns: return None
    # Group by unique player
    # "clears" = CI half-width small enough to be informative (per60_hi - per60_lo < per60)
    df["informative"] = ((df["per60_hi"] - df["per60_lo"]) < df["per60"]).fillna(False)
    per_pl = df.groupby("player_id")["informative"].any().reset_index()
    total = len(per_pl)
    clears = per_pl["informative"].sum()
    return {"pillar": pillar_label, "players": total, "ci_clear": int(clears),
            "ci_clear_pct": round(clears/total*100, 1) if total else 0}

flags = []
for p, lbl in [("pillar_1_netfront_F.csv","P1_NF_F"),
               ("pillar_2_defensive_F.csv","P2_DefF"),
               ("pillar_3_offensive_F.csv","P3_OffF"),
               ("pillar_4_defensive_D.csv","P4_DefD"),
               ("pillar_5_offensive_D.csv","P5_OffD")]:
    r = ci_check(f"{OUT}/{p}", lbl)
    if r: flags.append(r)
# Goalies: pillar 6/7 use sv_lo/sv_hi
for p, lbl in [("pillar_6_goalie_FNFI_MNFI.csv","P6_Goalie_FM"),
               ("pillar_7_goalie_CNFI.csv","P7_Goalie_CNFI")]:
    pf = f"{OUT}/{p}"
    if not os.path.exists(pf): continue
    df = pd.read_csv(pf)
    if "sv_lo" in df.columns:
        df["informative"] = ((df["sv_hi"]-df["sv_lo"]) < 0.04)  # tight CI
        per_pl = df.groupby("goalie_id")["informative"].any().reset_index()
        total = len(per_pl); clears = per_pl["informative"].sum()
        flags.append({"pillar": lbl, "players": total, "ci_clear": int(clears),
                      "ci_clear_pct": round(clears/total*100, 1) if total else 0})
flag_df = pd.DataFrame(flags)
flag_df["below_50pct"] = flag_df["ci_clear_pct"] < 50
flag_df.to_csv(f"{OUT}/pillar_ci_flagging.csv", index=False)
print("\nPillar CI flagging:")
print(flag_df.to_string(index=False))

# ---------- Horse race summary (expanded) ----------
univ = pd.read_csv(f"{OUT}/horse_race_univariate.csv").sort_values("var_explained", ascending=False)
mv = pd.read_csv(f"{OUT}/horse_race_multivariate.csv")
# Head-to-head
best_tnfi = univ[univ["metric"].str.startswith("TNFI_") | univ["metric"].isin(["CNFI_pct","MNFI_pct","FNFI_pct"])].iloc[0]
best_corsi = univ[univ["metric"].str.startswith("CF_")].iloc[0] if len(univ[univ["metric"].str.startswith("CF_")]) else None
hd_row = univ[univ["metric"]=="HD_CF_pct"]
pillars = univ[univ["metric"].str.startswith("P")]

summary_rows = []
for _,r in univ.iterrows():
    summary_rows.append({"rank":None,"metric":r["metric"],"R":r["R"],"p":r["p"],
                         "var_explained":r["var_explained"],"n":r["n"],"type":"univariate"})
# multivariate R^2 (from saved file: r2 might not be there; recompute from residual to fit)
# r2 stored as last attribute; we re-derive by taking coefficient row + sum
# Simpler: rebuild R^2 from scratch by running mv on team data
team = pd.read_csv(f"{OUT}/team_level_all_metrics.csv").dropna(subset=["points"])
pillar_cols = ["P1_NF_F","P2_DefF","P3_OffF","P4_DefD","P5_OffD","P6_SvPct","P7_SvPct"]
sub = team[["points"]+pillar_cols].dropna()
if len(sub)>0:
    X = sub[pillar_cols].values
    Xz = (X - X.mean(axis=0))/X.std(axis=0,ddof=0)
    y = sub["points"].values
    Xz1 = np.column_stack([np.ones(len(Xz)), Xz])
    XtX = Xz1.T @ Xz1
    lam = 1e-6 * np.trace(XtX) / XtX.shape[0]
    beta = np.linalg.solve(XtX + lam*np.eye(XtX.shape[0]), Xz1.T @ y)
    yhat = Xz1 @ beta
    mv_r2 = 1 - ((y-yhat)**2).sum() / ((y-y.mean())**2).sum()
    summary_rows.append({"rank":None,"metric":"MULTIVARIATE (all 7 pillars)","R":math.sqrt(mv_r2),"p":0.0,
                         "var_explained":mv_r2,"n":len(sub),"type":"multivariate"})

sum_df = pd.DataFrame(summary_rows).sort_values("var_explained", ascending=False).reset_index(drop=True)
sum_df["rank"] = sum_df.index + 1
sum_df.to_csv(f"{OUT}/horse_race_summary.csv", index=False)
print(f"\nHorse race summary top 10:")
print(sum_df.head(10).to_string(index=False))

# ---------- Head-to-head table ----------
# Best TNFI variant, best Corsi variant, HD CF%, each pillar ranked by R^2
candidates = []
candidates.append(("Best_TNFI_variant", best_tnfi.metric, best_tnfi["R"], best_tnfi["var_explained"]))
if best_corsi is not None:
    candidates.append(("Best_Corsi_variant", best_corsi.metric, best_corsi["R"], best_corsi["var_explained"]))
if len(hd_row):
    hdr = hd_row.iloc[0]
    candidates.append(("HD_CF%", "HD_CF_pct", hdr["R"], hdr["var_explained"]))
for _,r in pillars.iterrows():
    candidates.append((r["metric"], r["metric"], r["R"], r["var_explained"]))
h2h = pd.DataFrame(candidates, columns=["slot","metric","R","var_explained"]).sort_values("var_explained", ascending=False)
h2h.to_csv(f"{OUT}/horse_race_head_to_head.csv", index=False)
print("\nHead-to-head:")
print(h2h.to_string(index=False))

# ---------- composite combined player + team ----------
p_comp = pd.read_csv(f"{OUT}/player_composite_NFI.csv")
t_comp = pd.read_csv(f"{OUT}/team_composite_NFI.csv")
print(f"\nComposite files: player={len(p_comp)}, team={len(t_comp)}")

print("\nAll outputs in:", OUT)
for f in sorted(os.listdir(OUT)):
    path = f"{OUT}/{f}"
    sz = os.path.getsize(path)
    print(f"  {f:<48}  {sz:>10,} bytes")
