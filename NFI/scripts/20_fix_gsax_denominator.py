#!/usr/bin/env python3
"""
FIX 1 — GSAx denominator correction.

Existing zone_conversion_rates.csv uses goals/attempts (Fenwick or Corsi
denominator including missed/blocked shots that never reach the goalie).
For goalie expected-goals work the correct denominator is goals/(SOG + goals)
— shots that actually faced the goalie.

This script:
  1. Recomputes per-zone rates with BOTH denominators from shots_tagged.csv
  2. Overwrites NFI/output/zone_conversion_rates.csv with corrected per-faced
     columns added.
  3. Rebuilds calibrated NFI-GSAx using the corrected per-faced rates.
  4. Overwrites NFI/output/goalie_nfi_gsax.csv with corrected calibrated GSAx.
  5. Overwrites NFI/output/goalie_metric_comparison.csv with corrected
     xG_calibrated / NFI_GSAx_calibrated and corrected per-60 derivative.
  6. Reports rank changes vs prior file (>5 position swing).
"""
import math
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
MIN_SHOTS = 300

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0.0, c-h), min(1.0, c+h))

# Cache the OLD goalie_metric_comparison rank columns before overwrite.
gm_old = pd.read_csv(f"{OUT}/goalie_metric_comparison.csv")
old_calib_rank = dict(zip(gm_old["goalie_id"],
                          gm_old["NFI_GSAx_cumulative"]))  # we'll re-rank vs new
old_per60_rank = dict(zip(gm_old["goalie_id"], gm_old["rank_per60"]))

sh = pd.read_csv(f"{OUT}/shots_tagged.csv")
es = sh[sh["state"]=="ES"].copy()
print(f"ES regulation rows in shots_tagged: {len(es):,}")

# ----- Step 1: recompute zone rates with both denominators -----
def faced_mask(df):
    """SOG + goal events (i.e., shots that reached the goalie)."""
    return df["event_type"].isin(["shot-on-goal","goal"])

def all_attempts_mask(df):
    """All Corsi attempts (SOG + missed + blocked + goal)."""
    return df["event_type"].isin(["shot-on-goal","missed-shot","blocked-shot","goal"])

zones = ["CNFI","MNFI","FNFI","Wide","lane_other"]
rows = []

# HD conventional
hd = es[(es["x_coord_norm"]>69) & (es["y_coord_norm"].between(-22,22))]
hd_a = hd[all_attempts_mask(hd)]
hd_f = hd[faced_mask(hd)]
gl_a = int(hd_a["is_goal_i"].sum()); n_a = len(hd_a)
gl_f = int(hd_f["is_goal_i"].sum()); n_f = len(hd_f)
p_a, lo_a, hi_a = wilson(gl_a, n_a)
p_f, lo_f, hi_f = wilson(gl_f, n_f)
rows.append({"zone":"HD_conventional (x>69, y in ±22)",
             "attempts":n_a,"goals":gl_a,
             "conversion_pct":round(p_a*100,3),"ci_lo_pct":round(lo_a*100,3),"ci_hi_pct":round(hi_a*100,3),
             "faced":n_f,"faced_goals":gl_f,
             "faced_conversion_pct":round(p_f*100,3),
             "faced_ci_lo_pct":round(lo_f*100,3),"faced_ci_hi_pct":round(hi_f*100,3)})

for z in zones:
    sub = es[es["zone"]==z]
    sa = sub[all_attempts_mask(sub)]; sf = sub[faced_mask(sub)]
    n_a, gl_a = len(sa), int(sa["is_goal_i"].sum())
    n_f, gl_f = len(sf), int(sf["is_goal_i"].sum())
    p_a, lo_a, hi_a = wilson(gl_a, n_a)
    p_f, lo_f, hi_f = wilson(gl_f, n_f)
    rows.append({"zone":z,"attempts":n_a,"goals":gl_a,
                 "conversion_pct":round(p_a*100,3),
                 "ci_lo_pct":round(lo_a*100,3),"ci_hi_pct":round(hi_a*100,3),
                 "faced":n_f,"faced_goals":gl_f,
                 "faced_conversion_pct":round(p_f*100,3),
                 "faced_ci_lo_pct":round(lo_f*100,3),"faced_ci_hi_pct":round(hi_f*100,3)})

# TNFI combined
tn = es[es["zone"].isin(["CNFI","MNFI","FNFI"])]
ta = tn[all_attempts_mask(tn)]; tf = tn[faced_mask(tn)]
p_a, lo_a, hi_a = wilson(int(ta["is_goal_i"].sum()), len(ta))
p_f, lo_f, hi_f = wilson(int(tf["is_goal_i"].sum()), len(tf))
rows.append({"zone":"TNFI (CNFI+MNFI+FNFI)",
             "attempts":len(ta),"goals":int(ta["is_goal_i"].sum()),
             "conversion_pct":round(p_a*100,3),
             "ci_lo_pct":round(lo_a*100,3),"ci_hi_pct":round(hi_a*100,3),
             "faced":len(tf),"faced_goals":int(tf["is_goal_i"].sum()),
             "faced_conversion_pct":round(p_f*100,3),
             "faced_ci_lo_pct":round(lo_f*100,3),"faced_ci_hi_pct":round(hi_f*100,3)})

# All
ea = es[all_attempts_mask(es)]; ef = es[faced_mask(es)]
p_a, lo_a, hi_a = wilson(int(ea["is_goal_i"].sum()), len(ea))
p_f, lo_f, hi_f = wilson(int(ef["is_goal_i"].sum()), len(ef))
rows.append({"zone":"ALL ES REG",
             "attempts":len(ea),"goals":int(ea["is_goal_i"].sum()),
             "conversion_pct":round(p_a*100,3),
             "ci_lo_pct":round(lo_a*100,3),"ci_hi_pct":round(hi_a*100,3),
             "faced":len(ef),"faced_goals":int(ef["is_goal_i"].sum()),
             "faced_conversion_pct":round(p_f*100,3),
             "faced_ci_lo_pct":round(lo_f*100,3),"faced_ci_hi_pct":round(hi_f*100,3)})

zr = pd.DataFrame(rows)
zr.to_csv(f"{OUT}/zone_conversion_rates.csv", index=False)
print("\n=== zone_conversion_rates.csv (rebuilt) ===")
print(zr.to_string(index=False))

# Build the corrected per-faced rate dictionary
RATE_FACED = {}
for z in ["CNFI","MNFI","FNFI"]:
    r = zr[zr["zone"]==z].iloc[0]
    RATE_FACED[z] = r["faced_conversion_pct"] / 100.0
print(f"\nCorrected per-faced rates: {RATE_FACED}")

# ----- Step 2: recompute calibrated NFI-GSAx using corrected rates -----
# Goalie-faced shots only (SOG + goal)
f = es[faced_mask(es) & es["goalie_id"].notna()].copy()
f["goalie_id"] = f["goalie_id"].astype(int)

pos = pd.read_csv(f"{OUT}/player_positions.csv")
name_map = dict(zip(pos["player_id"].astype(int), pos["player_name"]))

# Per goalie per zone faced + goals (CNFI/MNFI/FNFI only for the calibrated GSAx)
g = f[f["zone"].isin(["CNFI","MNFI","FNFI"])].groupby(["goalie_id","zone"]).agg(
    faced=("is_goal_i","size"),
    goals=("is_goal_i","sum"),
).reset_index()
wide = g.pivot_table(index="goalie_id", columns="zone",
                     values=["faced","goals"], fill_value=0)
wide.columns = [f"{a}_{b}" for a,b in wide.columns]
wide = wide.reset_index()
for c in ["faced_CNFI","faced_MNFI","faced_FNFI",
          "goals_CNFI","goals_MNFI","goals_FNFI"]:
    if c not in wide.columns:
        wide[c] = 0
wide["total_faced"] = wide["faced_CNFI"]+wide["faced_MNFI"]+wide["faced_FNFI"]
wide["total_goals"] = wide["goals_CNFI"]+wide["goals_MNFI"]+wide["goals_FNFI"]
wide = wide[wide["total_faced"]>=MIN_SHOTS].copy()
wide["goalie_name"] = wide["goalie_id"].map(name_map).fillna("")

# Original (per-attempt) rates kept for reference (read from rebuilt zone file)
RATE_ATTEMPT = {z: zr[zr["zone"]==z]["conversion_pct"].iloc[0]/100.0
                for z in ["CNFI","MNFI","FNFI"]}

# Old (incorrect) calibrated xG used per-attempt rates labeled as faced.
# New corrected calibration uses the actual per-faced rates.
wide["xG"] = (wide["faced_CNFI"]*RATE_ATTEMPT["CNFI"]
              + wide["faced_MNFI"]*RATE_ATTEMPT["MNFI"]
              + wide["faced_FNFI"]*RATE_ATTEMPT["FNFI"])
wide["NFI_GSAx"] = wide["xG"] - wide["total_goals"]
wide["xG_calibrated"] = (wide["faced_CNFI"]*RATE_FACED["CNFI"]
                          + wide["faced_MNFI"]*RATE_FACED["MNFI"]
                          + wide["faced_FNFI"]*RATE_FACED["FNFI"])
wide["NFI_GSAx_calibrated"] = wide["xG_calibrated"] - wide["total_goals"]

# Ranks
wide = wide.sort_values("NFI_GSAx_calibrated", ascending=False).reset_index(drop=True)
wide["rank_gsax_calibrated"] = wide["NFI_GSAx_calibrated"].rank(method="min", ascending=False).astype(int)
wide["rank_gsax"] = wide["NFI_GSAx"].rank(method="min", ascending=False).astype(int)

gs = wide[["rank_gsax","rank_gsax_calibrated","goalie_id","goalie_name",
           "faced_CNFI","faced_MNFI","faced_FNFI",
           "total_faced","total_goals",
           "xG","NFI_GSAx","xG_calibrated","NFI_GSAx_calibrated"]].copy()
for c in ["xG","NFI_GSAx","xG_calibrated","NFI_GSAx_calibrated"]:
    gs[c] = gs[c].round(2)

gs.to_csv(f"{OUT}/goalie_nfi_gsax.csv", index=False)
print(f"\n=== goalie_nfi_gsax.csv rebuilt ({len(gs)} goalies) ===")

# ----- Step 3: rebuild goalie_metric_comparison.csv (preserve other cols) -----
# Rebuild using prior structure: replace xG/NFI_GSAx/xG_calibrated/NFI_GSAx_calibrated
# and reset NFI_GSAx_cumulative + NFI_GSAx_per60 to use the calibrated value
# (since the user requested that calibrated be the primary).
gm = gm_old.copy()
# Map fresh values
fresh = gs.set_index("goalie_id")
for col_target, col_source in [("xG","xG"),
                               ("NFI_GSAx","NFI_GSAx"),
                               ("xG_calibrated","xG_calibrated"),
                               ("NFI_GSAx","NFI_GSAx")]:
    pass

gm["xG"]                  = gm["goalie_id"].map(fresh["xG"])
gm["NFI_GSAx"]            = gm["goalie_id"].map(fresh["NFI_GSAx"])
gm["xG_calibrated"]       = gm["goalie_id"].map(fresh["xG_calibrated"])
# Promote calibrated to primary (NFI_GSAx_cumulative) and recompute per-60
gm["NFI_GSAx_cumulative"] = gm["goalie_id"].map(fresh["NFI_GSAx_calibrated"])
# per-60 uses ES TOI minutes (existing column)
gm["NFI_GSAx_per60"] = (gm["NFI_GSAx_cumulative"] / gm["toi_ES_min"]) * 60.0
gm["NFI_GSAx_per60"] = gm["NFI_GSAx_per60"].round(3)
gm["NFI_GSAx_cumulative"] = gm["NFI_GSAx_cumulative"].round(2)

# Rebuild ranks
gm["rank_cumulative"] = gm["NFI_GSAx_cumulative"].rank(method="min", ascending=False)
gm["rank_per60"]      = gm["NFI_GSAx_per60"].rank(method="min", ascending=False)
gm["rank_diff_per60_vs_cum"] = gm["rank_cumulative"] - gm["rank_per60"]

# tier-2 rank rebuild (4000-min subset)
mask4k = gm["meets_4000min"]==True
gm["rank_per60_4000min"] = np.nan
gm["rank_cumulative_4000min"] = np.nan
gm.loc[mask4k, "rank_per60_4000min"] = gm.loc[mask4k, "NFI_GSAx_per60"]\
    .rank(method="min", ascending=False).values
gm.loc[mask4k, "rank_cumulative_4000min"] = gm.loc[mask4k, "NFI_GSAx_cumulative"]\
    .rank(method="min", ascending=False).values

# Save
gm.to_csv(f"{OUT}/goalie_metric_comparison.csv", index=False)
print(f"=== goalie_metric_comparison.csv rebuilt ({len(gm)} goalies) ===")

# ----- Step 4: rank-change report (per-60 rank, since per-60 was the
# headline metric used by upstream team-construction work) -----
gm["old_rank_per60"] = gm["goalie_id"].map(old_per60_rank)
gm["rank_per60_delta"] = gm["old_rank_per60"] - gm["rank_per60"]
gm["flag_rank_change_gt5"] = gm["rank_per60_delta"].abs() > 5

pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 220
pd.options.display.max_columns = None

print("\n=== TOP 15 by corrected NFI-GSAx (cumulative) ===")
top15 = gm.sort_values("NFI_GSAx_cumulative", ascending=False).head(15)
print(top15[["goalie_name","total_faced","toi_ES_min",
             "NFI_GSAx_cumulative","NFI_GSAx_per60",
             "rank_cumulative","rank_per60",
             "old_rank_per60","rank_per60_delta","flag_rank_change_gt5"]]
      .to_string(index=False))

print("\n=== BOTTOM 5 by corrected NFI-GSAx (cumulative) ===")
bot5 = gm.sort_values("NFI_GSAx_cumulative", ascending=True).head(5)
print(bot5[["goalie_name","total_faced","toi_ES_min",
            "NFI_GSAx_cumulative","NFI_GSAx_per60",
            "rank_cumulative","rank_per60",
            "old_rank_per60","rank_per60_delta","flag_rank_change_gt5"]]
      .to_string(index=False))

print("\n=== Goalies with per-60 rank change > 5 positions ===")
flag = gm[gm["flag_rank_change_gt5"]==True].sort_values("rank_per60_delta")
print(flag[["goalie_name","total_faced","toi_ES_min",
            "NFI_GSAx_cumulative","NFI_GSAx_per60",
            "old_rank_per60","rank_per60","rank_per60_delta"]]
      .to_string(index=False))
print(f"\nTotal flagged: {len(flag)} of {len(gm)} goalies")

print(f"\nFiles overwritten:")
for f_name in ["zone_conversion_rates.csv","goalie_nfi_gsax.csv",
               "goalie_metric_comparison.csv"]:
    print(f"  {OUT}/{f_name}")
