#!/usr/bin/env python3
"""
Investigate whether Fenwick zone-adjustment factor is structurally larger
than Corsi's, using best-available proxies in the absence of faceoff data.

Approach:
  1. Compute league-wide block rate (Block / Corsi total).
  2. Break down block rate by shot zone (CNFI/MNFI/FNFI/Wide) — does block
     rate vary by location, which would proxy for "shift context"?
  3. Compute per-team-season CF / CA / FF / FA, then CF% and FF%.
  4. The MATHEMATICAL test: under idealized symmetric blocking, CF% and FF%
     should be IDENTICAL (per team-game). Compute both per team-season and
     show the divergence.
  5. Use shot's zone_code as a context proxy: split each team's shots into
     "in their OZ" (shooter is the team) vs "in their DZ" (shooter is opp).
     For each split, compute block rate.
  6. Compare CF% vs FF% with asymmetric block rates and quantify the
     zone-factor effect.

This isn't the textbook faceoff-based zone adjustment, but it tests the
underlying claim — does block prevalence differ in a way that would create
a Fenwick > Corsi zone-factor gap.
"""
import math
import pandas as pd
import numpy as np

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
SEASONS = {"20212022","20222023","20232024","20242025","20252026"}

print("Loading shots ...")
sh = pd.read_csv(SHOT_CSV,
                 usecols=["game_id","season","period","situation_code","event_type",
                          "shooting_team_abbrev","x_coord_norm","y_coord_norm",
                          "home_team_abbrev","away_team_abbrev","zone_code","is_goal"],
                 dtype={"season":str,"situation_code":str})
sh = sh[sh["season"].isin(SEASONS)]
sh = sh[sh["period"].between(1,3)]
sh = sh[sh["situation_code"].astype(str)=="1551"]
sh = sh[sh["event_type"].isin(["shot-on-goal","missed-shot","blocked-shot","goal"])]
sh = sh.dropna(subset=["x_coord_norm","y_coord_norm"])

# Apply blocked-shot coord fix
blk = sh["event_type"]=="blocked-shot"
sh.loc[blk,"x_coord_norm"] = sh.loc[blk,"x_coord_norm"].abs()
sh.loc[blk,"y_coord_norm"] = sh.loc[blk,"y_coord_norm"].abs()
sh["abs_y"] = sh["y_coord_norm"].abs()
sh["is_block"] = (sh["event_type"]=="blocked-shot").astype(int)
sh["is_fenwick"] = (~blk).astype(int)
print(f"  total ES regulation Corsi events: {len(sh):,}")

# ===== 1. League-wide block rate =====
n_total = len(sh)
n_block = int(sh["is_block"].sum())
print("\n=== Q1: League-wide block prevalence ===")
print(f"  Total Corsi events:  {n_total:,}")
print(f"  Total blocked:       {n_block:,}  ({n_block/n_total*100:.2f}% of Corsi)")
print(f"  Total Fenwick:       {n_total-n_block:,}  ({(n_total-n_block)/n_total*100:.2f}%)")

# ===== 2. Block rate by shot zone =====
def zone_label(x, ay):
    if 74 <= x <= 89 and ay <= 9:   return "CNFI"
    if 55 <= x <= 73 and ay <= 15:  return "MNFI"
    if 25 <= x <= 54 and ay <= 15:  return "FNFI"
    if ay > 15: return "Wide"
    return "Other"
sh["zone"] = [zone_label(x, ay) for x, ay in zip(sh["x_coord_norm"], sh["abs_y"])]

print("\n=== Q2: Block rate by shot zone (where the shot occurred) ===")
zg = sh.groupby("zone").agg(total=("is_block","size"),
                              blocks=("is_block","sum")).reset_index()
zg["block_rate_pct"] = (zg["blocks"]/zg["total"]*100).round(2)
print(zg.to_string(index=False))
print("\n  Interpretation: if block_rate_pct is roughly UNIFORM across zones,")
print("  then Fenwick (= Corsi minus blocks) cannot have a structurally")
print("  larger zone-adjustment factor than Corsi. If it varies STRONGLY,")
print("  the asymmetry can drive a Fenwick > Corsi zone factor.")

# ===== 3. Team-season CF / CA / FF / FA =====
print("\n=== Q3: Team-season CF, CA, FF, FA aggregate ===")
# For each team-season, sum its CF (its shots) and CA (opponents' shots)
sh["fenwick_event"] = (~blk).astype(int)
ts_cf = sh.groupby(["season","shooting_team_abbrev"]).agg(
    CF=("is_block","size"),
    CF_blocks=("is_block","sum"),
).reset_index().rename(columns={"shooting_team_abbrev":"team"})
ts_cf["FF"] = ts_cf["CF"] - ts_cf["CF_blocks"]

# Need defending-team perspective: for each shot, the defending team is the
# OTHER abbrev. We need a per-team sum of "shots taken against me".
sh_h = sh.copy()
sh_h["def_team"] = np.where(sh_h["shooting_team_abbrev"]==sh_h["home_team_abbrev"],
                              sh_h["away_team_abbrev"], sh_h["home_team_abbrev"])
ts_ca = sh_h.groupby(["season","def_team"]).agg(
    CA=("is_block","size"),
    CA_blocks=("is_block","sum"),
).reset_index().rename(columns={"def_team":"team"})
ts_ca["FA"] = ts_ca["CA"] - ts_ca["CA_blocks"]

ts = ts_cf.merge(ts_ca, on=["season","team"])
ts["CF_pct"] = ts["CF"] / (ts["CF"] + ts["CA"])
ts["FF_pct"] = ts["FF"] / (ts["FF"] + ts["FA"])
ts["block_rate_for"]  = ts["CF_blocks"] / ts["CF"]
ts["block_rate_ag"]   = ts["CA_blocks"] / ts["CA"]
ts["block_rate_diff"] = ts["block_rate_for"] - ts["block_rate_ag"]
ts["CF_FF_diff"] = ts["CF_pct"] - ts["FF_pct"]

print(f"  team-seasons: {len(ts)}")
print(f"\n  Mean CF%:  {ts['CF_pct'].mean()*100:.3f}%   (sd {ts['CF_pct'].std()*100:.3f})")
print(f"  Mean FF%:  {ts['FF_pct'].mean()*100:.3f}%   (sd {ts['FF_pct'].std()*100:.3f})")
print(f"  Mean (CF% - FF%):     {ts['CF_FF_diff'].mean()*100:+.4f} pp")
print(f"  Std  (CF% - FF%):     {ts['CF_FF_diff'].std()*100:.4f} pp")
print(f"\n  Mean block rate FOR (own shots):     {ts['block_rate_for'].mean()*100:.3f}%")
print(f"  Mean block rate AGAINST (opp shots): {ts['block_rate_ag'].mean()*100:.3f}%")
print(f"  Mean (For - Ag) block rate diff:     {ts['block_rate_diff'].mean()*100:+.4f} pp")

# ===== 4. Correlate (CF%-FF%) with team's offensive bias =====
print("\n=== Q4: Does CF%-FF% gap depend on team's offensive bias? ===")
# A team that is dominantly in OZ has high CF% (high CF, low CA). If Fenwick
# zone factor > Corsi, then high-CF% teams should have CF% > FF% by more
# than low-CF% teams.
ts["off_bias"] = ts["CF"] / (ts["CF"]+ts["CA"])  # same as CF%
from scipy import stats
r, p = stats.pearsonr(ts["off_bias"], ts["CF_FF_diff"])
print(f"  Pearson r (off_bias, CF%-FF%) = {r:+.4f}, p = {p:.4e}")
print(f"  If r > 0: CF% systematically exceeds FF% for offense-heavy teams")
print(f"            → Fenwick zone factor > Corsi zone factor is structurally true")
print(f"  If r ≈ 0: removing blocks doesn't asymmetrically affect zone-heavy teams")
print(f"            → Fenwick and Corsi zone factors should be SIMILAR")

# Also: regress CF% on off_bias, FF% on off_bias separately. The slopes are
# the implied zone-adjustment factors at the team-season level.
def ols(x, y):
    x_m = x.mean(); y_m = y.mean()
    ss_xy = ((x-x_m)*(y-y_m)).sum()
    ss_xx = ((x-x_m)**2).sum()
    slope = ss_xy/ss_xx if ss_xx > 0 else 0
    intercept = y_m - slope*x_m
    yhat = intercept + slope*x
    ss_res = ((y-yhat)**2).sum(); ss_tot = ((y-y_m)**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    return slope, intercept, r2

# This is a degenerate self-regression (off_bias IS CF%). The right test
# requires an INDEPENDENT measure of "OZ-context pressure". Without faceoff
# data we don't have one. But we can construct a proxy: for each team-season,
# the team's CNFI+MNFI shot SHARE (i.e., what fraction of their attempts
# are central-zone) is a proxy for sustained offensive pressure.
ts_cm = sh[sh["zone"].isin(["CNFI","MNFI"])].groupby(["season","shooting_team_abbrev"])\
        .size().reset_index(name="CM_for").rename(columns={"shooting_team_abbrev":"team"})
ts = ts.merge(ts_cm, on=["season","team"])
ts["cm_share"] = ts["CM_for"] / ts["CF"]

slope_cf, _, r2_cf = ols(ts["cm_share"].values, ts["CF_pct"].values)
slope_ff, _, r2_ff = ols(ts["cm_share"].values, ts["FF_pct"].values)
print(f"\n=== Q5: Independent OLS — central-zone share (proxy for sustained OZ pressure) → CF% vs FF% ===")
print(f"  CF% ~ cm_share:  slope = {slope_cf:+.4f},  R² = {r2_cf:.4f}")
print(f"  FF% ~ cm_share:  slope = {slope_ff:+.4f},  R² = {r2_ff:.4f}")
print(f"  Slope ratio FF/CF: {slope_ff/slope_cf:.4f}  (≈1.0 means equal sensitivity)")

# ===== 6. Direct empirical proxy of zone factor: split shots into 'team-A-in-OZ'
# (zone_code='O' for shots taken by team A) and 'team-A-in-DZ' (zone_code='O'
# for shots taken by opp). Compute team's CF% and FF% in each context. =====
print("\n=== Q6: Block rate split by shooting team perspective ===")
# Block rate for shots taken BY each team
tr = sh.groupby("shooting_team_abbrev").agg(
    n=("is_block","size"), blocks=("is_block","sum")).reset_index()
tr["block_rate_for"] = tr["blocks"]/tr["n"]
print(f"  Team block-rate FOR (own shots):  range {tr['block_rate_for'].min()*100:.2f}% - "
      f"{tr['block_rate_for'].max()*100:.2f}%, mean {tr['block_rate_for'].mean()*100:.3f}%")

# Block rate for shots taken AGAINST each team
sh_def = sh.copy()
sh_def["def_team"] = np.where(sh_def["shooting_team_abbrev"]==sh_def["home_team_abbrev"],
                                sh_def["away_team_abbrev"], sh_def["home_team_abbrev"])
tr_ag = sh_def.groupby("def_team").agg(
    n=("is_block","size"), blocks=("is_block","sum")).reset_index()
tr_ag["block_rate_ag"] = tr_ag["blocks"]/tr_ag["n"]
print(f"  Team block-rate AGAINST (opp shots): range {tr_ag['block_rate_ag'].min()*100:.2f}% - "
      f"{tr_ag['block_rate_ag'].max()*100:.2f}%, mean {tr_ag['block_rate_ag'].mean()*100:.3f}%")

# ===== Summary tables =====
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print("\n=== Top-5 / bottom-5 teams by (CF% - FF%) ===")
ts2 = ts.sort_values("CF_FF_diff", ascending=False)
print("Top 5 (CF% larger than FF% by most):")
print(ts2.head(5)[["season","team","CF","FF","CA","FA","CF_pct","FF_pct",
                    "CF_FF_diff","block_rate_for","block_rate_ag","block_rate_diff"]].to_string(index=False))
print("\nBottom 5 (FF% larger than CF% by most):")
print(ts2.tail(5)[["season","team","CF","FF","CA","FA","CF_pct","FF_pct",
                    "CF_FF_diff","block_rate_for","block_rate_ag","block_rate_diff"]].to_string(index=False))
