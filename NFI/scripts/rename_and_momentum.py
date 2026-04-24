"""Rebuild all output files with new naming convention + build momentum metrics.

Naming convention (applied throughout):
  raw per-player NFI%       -> NFI_pct
  zone-adjusted              -> NFI_pct_ZA
  fully adjusted (3A)        -> NFI_pct_3A   (was FA_NFI_emp)
  Quality of Competition     -> NFQOC        (was IOC_NFI)
  Quality of Linemates       -> NFQOL        (was IOL_NFI)
  Relative For (rate delta)  -> RelNFI_F_pct (was TNFI_RF)
  Relative Against           -> RelNFI_A_pct (was TNFI_RA)
  Combined (net two-way)     -> RelNFI_pct   (was TNFI_combined)
  Momentum year-over-year    -> NFI_pct_3A_MOM
  Three-year avg momentum    -> NFI_pct_3A_MOM_3yr
  Equivalents for CF and FF follow same pattern.

Outputs to NFI/output/fully_adjusted/:
  player_fully_adjusted.csv
  current_season_player_fully_adjusted.csv
  team_fully_adjusted.csv
  current_season_team_fully_adjusted.csv
  horse_race_all_metrics.csv
  top30_RelNFI_pooled.csv, top30_RelNFI_F_pooled.csv, top30_RelNFI_A_pooled.csv
  top30_RelNFI_2526.csv
  top20_NFI_3A_MOM_ascending.csv
  top20_NFI_3A_MOM_descending.csv
  top20_NFI_3A_MOM_3yr_ascending.csv
  momentum_divergence_flags.csv
And updates NFI/output/zone_adjustment/complete_decision_tree/stage7_annual_ratings.csv.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "NFI/output/fully_adjusted"
OUT_DT = ROOT / "NFI/output/zone_adjustment/complete_decision_tree"
TEAM_METRICS = ROOT / "NFI/output/team_level_all_metrics.csv"

ABBR = {"ARI":"UTA"}
POOLED = {"20222023","20232024","20242025","20252026"}
CURRENT = "20252026"
SEASONS_ORDERED = ["20222023","20232024","20242025","20252026"]
def norm(a): return ABBR.get(a,a) if a else a

print("[0] loading ...")
pp = pd.read_csv(OUT / "player_fully_adjusted.csv")
pp["season"] = pp["season"].astype(str)

# Rename columns to new convention
rename_map = {
    # raw + ZA (already named with _pct / ZA_*_emp suffixes)
    "NFI_pct":         "NFI_pct",
    "ZA_NFI_emp":      "NFI_pct_ZA",
    "FA_NFI_emp":      "NFI_pct_3A",
    "IOC_NFI":         "NFQOC",
    "IOL_NFI":         "NFQOL",
    "CF_pct":          "CF_pct",
    "ZA_CF_emp":       "CF_pct_ZA",
    "ZA_CF_trad":      "CF_pct_ZA_trad",
    "FA_CF_emp":       "CF_pct_3A",
    "FA_CF_trad":      "CF_pct_3A_trad",
    "IOC_CF":          "CFQOC",
    "IOL_CF":          "CFQOL",
    "FF_pct":          "FF_pct",
    "ZA_FF_emp":       "FF_pct_ZA",
    "ZA_FF_trad":      "FF_pct_ZA_trad",
    "FA_FF_emp":       "FF_pct_3A",
    "FA_FF_trad":      "FF_pct_3A_trad",
    "IOC_FF":          "FFQOC",
    "IOL_FF":          "FFQOL",
    # Relatives
    "TNFI_RF":         "RelNFI_F_pct",
    "TNFI_RA":         "RelNFI_A_pct",
    "TNFI_combined":   "RelNFI_pct",
    "CF_RF":           "RelCF_F_pct",
    "CF_RA":           "RelCF_A_pct",
    "CF_combined":     "RelCF_pct",
    "FF_RF":           "RelFF_F_pct",
    "FF_RA":           "RelFF_A_pct",
    "FF_combined":     "RelFF_pct",
}
pp = pp.rename(columns=rename_map)

# =====================================================================
# PART 3 — Momentum metrics (year-over-year Δ in FA/3A ratings)
# =====================================================================
print("[PART 3] computing momentum metrics ...")
# For each player, sort by season and compute year-over-year diff in *_3A
pp = pp.sort_values(["player_id","season"]).reset_index(drop=True)
for metric in ["NFI","CF","FF"]:
    col_3a = f"{metric}_pct_3A"
    mom_col = f"{metric}_pct_3A_MOM"
    # Per-player diff (requires same player across adjacent seasons)
    pp[mom_col] = pp.groupby("player_id")[col_3a].diff()

# Require min ES TOI: 2000 pooled, 500 current. Compute flag.
pp["toi_min"] = pp["toi_sec"]/60
# For season-N momentum to be valid: player had ≥ threshold in BOTH season N and N-1
# Get previous-season toi_min per row via shift
pp["prev_toi_min"] = pp.groupby("player_id")["toi_min"].shift(1)
# Validity flags
pp["mom_valid_pooled"]  = (pp["toi_min"]>=2000) & (pp["prev_toi_min"]>=2000)
pp["mom_valid_current"] = (pp["toi_min"]>=500)  & (pp["prev_toi_min"]>=500)

# 3-year rolling momentum average (avg of YoY changes across last 3 transitions)
# Requires player has data in 4 consecutive seasons each with ≥ 2000 min.
# We implement as: NFI_pct_3A_MOM_3yr_i = mean(mom_i, mom_{i-1}, mom_{i-2})
# when all three preceding moms are valid (all intermediate seasons ≥ 2000 min).
for metric in ["NFI","CF","FF"]:
    mom_col = f"{metric}_pct_3A_MOM"
    pp[f"{metric}_pct_3A_MOM_3yr"] = (
        pp.groupby("player_id")[mom_col]
          .rolling(window=3, min_periods=3).mean()
          .reset_index(level=0, drop=True)
    )

# Direction consistency flag (all 3 moms same sign)
def rolling_sign_consistent(series, window=3):
    out = pd.Series(np.nan, index=series.index, dtype=object)
    vals = series.to_numpy()
    for i in range(len(vals)):
        if i+1 < window: continue
        window_vals = vals[i-window+1:i+1]
        if np.any(np.isnan(window_vals)): continue
        signs = np.sign(window_vals)
        if np.all(signs > 0):   out.iloc[i] = "consistent_up"
        elif np.all(signs < 0): out.iloc[i] = "consistent_down"
        else:                    out.iloc[i] = "mixed"
    return out
pp["NFI_MOM_consistency"] = (
    pp.groupby("player_id")["NFI_pct_3A_MOM"]
      .apply(lambda s: rolling_sign_consistent(s, 3))
      .reset_index(level=0, drop=True)
)

# =====================================================================
# PART 1 — Top 30 RelNFI% (and F/A components), pooled + current
# =====================================================================
print("[PART 1] Top 30 RelNFI% ...")
def grp_agg(df, col, min_toi_pooled=2000):
    g = df.groupby(["player_id","player_name","position"]).agg(
        n_seasons=("season","count"),
        toi_total=("toi_min","sum"),
        team_recent=("team","last"),
        metric_mean=(col,"mean"),
    ).reset_index()
    g = g[g["toi_total"]>=min_toi_pooled]
    return g.sort_values("metric_mean", ascending=False).reset_index(drop=True)

top_rel   = grp_agg(pp, "RelNFI_pct").head(30)
top_rel_f = grp_agg(pp, "RelNFI_F_pct").head(30)
top_rel_a = grp_agg(pp, "RelNFI_A_pct").head(30)
top_rel.to_csv(OUT/"top30_RelNFI_pooled.csv", index=False)
top_rel_f.to_csv(OUT/"top30_RelNFI_F_pooled.csv", index=False)
top_rel_a.to_csv(OUT/"top30_RelNFI_A_pooled.csv", index=False)

# Current season top 30 (500+ min)
curr = pp[(pp["season"]==CURRENT) & (pp["toi_min"]>=500)].copy()
curr_rel = curr.sort_values("RelNFI_pct", ascending=False).head(30)
curr_rel.to_csv(OUT/"top30_RelNFI_2526.csv", index=False)

def show(df, label, col):
    print(f"\n=== {label} ===")
    for i, r in df.iterrows():
        n = r.get("n_seasons", 1)
        toi = r.get("toi_total", r.get("toi_min"))
        mv = r.get("metric_mean", r.get(col))
        team = r.get("team_recent", r.get("team"))
        pos = r.get("position", r.get("pos"))
        print(f"{i+1:>3}  {r['player_name']:<26} {pos:<3} {team:<5}  "
              f"n={int(n)}  toi={toi:>6.0f}  {col}={mv:+.3f} /60")

show(top_rel,   "TOP 30 by RelNFI% POOLED (min 2000 ES min)",        "RelNFI_pct")
show(top_rel_f, "TOP 30 by RelNFI_F% POOLED (min 2000 ES min)",      "RelNFI_F_pct")
show(top_rel_a, "TOP 30 by RelNFI_A% POOLED (min 2000 ES min)",      "RelNFI_A_pct")
show(curr_rel,  "TOP 30 by RelNFI% 2025-26 (min 500 ES min)",        "RelNFI_pct")

# =====================================================================
# PART 2 — Horse race Rel-metrics vs standings
# =====================================================================
print("\n[PART 2] horse race Rel-metrics ...")
tm = pd.read_csv(TEAM_METRICS)
tm["season"] = tm["season"].astype(str)
tm = tm[tm["season"].isin(POOLED)][["season","team","points"]]

def team_agg_toi(df, col):
    d = df.dropna(subset=[col,"team"]).copy()
    d["w"] = d["toi_sec"]
    return d.groupby(["season","team"]).apply(
        lambda x: np.average(x[col], weights=x["w"])).rename(col).reset_index()

def r2pv(x, y):
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 5: return (np.nan, np.nan, 0, np.nan)
    r = np.corrcoef(x[mask], y[mask])[0,1]
    pv = st.pearsonr(x[mask], y[mask]).pvalue
    return r*r, r, int(mask.sum()), pv

horse_rows = []
cols_to_test = [
    # individual relatives
    "RelNFI_pct","RelNFI_F_pct","RelNFI_A_pct",
    "RelFF_pct","RelFF_F_pct","RelFF_A_pct",
    "RelCF_pct","RelCF_F_pct","RelCF_A_pct",
    # plus baseline FA/ZA for context
    "NFI_pct","NFI_pct_ZA","NFI_pct_3A",
    "FF_pct","FF_pct_ZA","FF_pct_3A",
    "CF_pct","CF_pct_ZA","CF_pct_3A",
]
for col in cols_to_test:
    tb = team_agg_toi(pp, col).merge(tm, on=["season","team"], how="inner")
    pooled = r2pv(tb[col].values.astype(float), tb["points"].values.astype(float))
    tb_c = tb[tb["season"]==CURRENT]
    curr_ = r2pv(tb_c[col].values.astype(float), tb_c["points"].values.astype(float))
    horse_rows.append({
        "metric": col,
        "R2_pooled": pooled[0], "r_pooled": pooled[1], "N_pooled": pooled[2], "p_pooled": pooled[3],
        "R2_2526": curr_[0],    "r_2526": curr_[1],    "N_2526": curr_[2],    "p_2526": curr_[3],
    })
horse = pd.DataFrame(horse_rows).sort_values("R2_pooled", ascending=False).reset_index(drop=True)
horse.to_csv(OUT/"horse_race_all_metrics.csv", index=False)
print("\n=== HORSE RACE ALL METRICS — R² vs points ===")
print(horse.to_string(index=False))

# =====================================================================
# PART 3 — Momentum top 20s
# =====================================================================
print("\n[PART 3b] momentum top 20 lists ...")
# Current-season (2025-26) momentum: players with 500+ min both seasons
mom_curr = pp[(pp["season"]==CURRENT) & pp["mom_valid_current"]].copy()
asc_curr = mom_curr.sort_values("NFI_pct_3A_MOM", ascending=False).head(20)
desc_curr = mom_curr.sort_values("NFI_pct_3A_MOM").head(20)
asc_curr.to_csv(OUT/"top20_NFI_3A_MOM_ascending.csv", index=False)
desc_curr.to_csv(OUT/"top20_NFI_3A_MOM_descending.csv", index=False)

# 3-year trend (players with valid 3yr computed in latest season)
mom_3yr = pp[(pp["season"]==CURRENT) & pp["NFI_pct_3A_MOM_3yr"].notna()].copy()
asc_3yr = mom_3yr.sort_values("NFI_pct_3A_MOM_3yr", ascending=False).head(20)
asc_3yr.to_csv(OUT/"top20_NFI_3A_MOM_3yr_ascending.csv", index=False)

# Divergence: NFI momentum and CF momentum opposite sign, current season
div = pp[(pp["season"]==CURRENT) & pp["mom_valid_current"] &
          pp["NFI_pct_3A_MOM"].notna() & pp["CF_pct_3A_MOM"].notna()].copy()
div["diverged"] = np.sign(div["NFI_pct_3A_MOM"]) != np.sign(div["CF_pct_3A_MOM"])
div_flagged = div[div["diverged"]].copy()
div_flagged["NFI_dir"] = np.where(div_flagged["NFI_pct_3A_MOM"]>0, "up", "down")
div_flagged["CF_dir"]  = np.where(div_flagged["CF_pct_3A_MOM"]>0, "up", "down")
div_flagged = div_flagged.sort_values("NFI_pct_3A_MOM", ascending=False)
div_flagged.to_csv(OUT/"momentum_divergence_flags.csv", index=False)

def show_mom(df, label):
    print(f"\n=== {label} ===")
    for i, r in df.iterrows():
        mom_n = r.get("NFI_pct_3A_MOM", np.nan)
        mom_c = r.get("CF_pct_3A_MOM", np.nan)
        mom_f = r.get("FF_pct_3A_MOM", np.nan)
        mom_3 = r.get("NFI_pct_3A_MOM_3yr", np.nan)
        print(f"{i+1:>3}  {r['player_name']:<26} {r['position']:<3} {r['team']:<5}  "
              f"toi={r['toi_min']:>5.0f}  prev={r['prev_toi_min']:>5.0f}  "
              f"NFI_MOM={mom_n:+.4f}  CF_MOM={mom_c:+.4f}  FF_MOM={mom_f:+.4f}  "
              f"3yr={mom_3:+.4f}" if not np.isnan(mom_3) else
              f"{i+1:>3}  {r['player_name']:<26} {r['position']:<3} {r['team']:<5}  "
              f"toi={r['toi_min']:>5.0f}  prev={r['prev_toi_min']:>5.0f}  "
              f"NFI_MOM={mom_n:+.4f}  CF_MOM={mom_c:+.4f}  FF_MOM={mom_f:+.4f}")

show_mom(asc_curr.reset_index(drop=True),  "TOP 20 ASCENDING by NFI_pct_3A_MOM (2025-26, min 500 min)")
show_mom(desc_curr.reset_index(drop=True), "TOP 20 DESCENDING by NFI_pct_3A_MOM (2025-26, min 500 min)")
show_mom(asc_3yr.reset_index(drop=True),   "TOP 20 ASCENDING by NFI_pct_3A_MOM_3yr (multi-season trend)")

print(f"\n-- Momentum divergence flags (NFI vs CF opposite direction, 2025-26) --")
print(f"  {len(div_flagged)} players flagged")
print(f"  First 10 (NFI up / CF down OR vice versa):")
for i, r in div_flagged.head(10).iterrows():
    print(f"    {r['player_name']:<25} {r['position']:<2} {r['team']:<4}  "
          f"NFI_MOM={r['NFI_pct_3A_MOM']:+.3f} ({r['NFI_dir']})  "
          f"CF_MOM={r['CF_pct_3A_MOM']:+.3f} ({r['CF_dir']})")

# =====================================================================
# PART 4/5 — Rebuild player/team CSVs with new naming
# =====================================================================
print("\n[PART 4/5] writing Streamlit CSVs ...")
output_cols = [
    "player_id","player_name","position","team","season","toi_min",
    "NFI_pct","NFI_pct_ZA","NFI_pct_3A","NFQOC","NFQOL",
    "RelNFI_F_pct","RelNFI_A_pct","RelNFI_pct",
    "NFI_pct_3A_MOM","NFI_pct_3A_MOM_3yr","NFI_MOM_consistency",
    "CF_pct","CF_pct_ZA","CF_pct_3A",
    "RelCF_F_pct","RelCF_A_pct","RelCF_pct",
    "CF_pct_3A_MOM",
    "FF_pct","FF_pct_ZA","FF_pct_3A",
    "RelFF_F_pct","RelFF_A_pct","RelFF_pct",
    "FF_pct_3A_MOM",
]
missing = [c for c in output_cols if c not in pp.columns]
if missing:
    print(f"    missing (skipping): {missing}")
available = [c for c in output_cols if c in pp.columns]
pp[available].to_csv(OUT/"player_fully_adjusted.csv", index=False)
pp[pp["season"]==CURRENT][available].to_csv(
    OUT/"current_season_player_fully_adjusted.csv", index=False)

# Team aggregates
team_rows = []
for (season, team), sub in pp.groupby(["season","team"]):
    if len(sub)==0: continue
    row = {"season":season, "team":team, "n_players":len(sub)}
    for col in ["NFI_pct","NFI_pct_ZA","NFI_pct_3A",
                "RelNFI_F_pct","RelNFI_A_pct","RelNFI_pct",
                "CF_pct","CF_pct_ZA","CF_pct_3A",
                "RelCF_F_pct","RelCF_A_pct","RelCF_pct",
                "FF_pct","FF_pct_ZA","FF_pct_3A",
                "RelFF_F_pct","RelFF_A_pct","RelFF_pct"]:
        vals = sub[[col,"toi_sec"]].dropna()
        row[col] = float(np.average(vals[col], weights=vals["toi_sec"])) if len(vals) else np.nan
    team_rows.append(row)
team_out = pd.DataFrame(team_rows).merge(tm, on=["season","team"], how="left")
team_out.to_csv(OUT/"team_fully_adjusted.csv", index=False)
team_out[team_out["season"]==CURRENT].to_csv(OUT/"current_season_team_fully_adjusted.csv", index=False)

# =====================================================================
# PART 6 — Stage 7 annual ratings with new naming
# =====================================================================
print("[PART 6] Stage 7 annual ratings ...")
s7_rows = []
for _, r in pp.iterrows():
    pid = int(r["player_id"]); season = r["season"]
    s7_rows.append({"player_id":pid,"player_name":r["player_name"],"team":r["team"],
                    "position":r["position"],"season":season,"toi_min":r["toi_min"],
                    "metric":"NFI","raw":r.get("NFI_pct"),"ZA":r.get("NFI_pct_ZA"),
                    "3A":r.get("NFI_pct_3A"),
                    "QOC":r.get("NFQOC"),"QOL":r.get("NFQOL"),
                    "Rel_F":r.get("RelNFI_F_pct"),"Rel_A":r.get("RelNFI_A_pct"),
                    "Rel":r.get("RelNFI_pct"),
                    "MOM":r.get("NFI_pct_3A_MOM"),"MOM_3yr":r.get("NFI_pct_3A_MOM_3yr")})
    s7_rows.append({"player_id":pid,"player_name":r["player_name"],"team":r["team"],
                    "position":r["position"],"season":season,"toi_min":r["toi_min"],
                    "metric":"FF","raw":r.get("FF_pct"),"ZA":r.get("FF_pct_ZA"),
                    "3A":r.get("FF_pct_3A"),
                    "QOC":r.get("FFQOC"),"QOL":r.get("FFQOL"),
                    "Rel_F":r.get("RelFF_F_pct"),"Rel_A":r.get("RelFF_A_pct"),
                    "Rel":r.get("RelFF_pct"),
                    "MOM":r.get("FF_pct_3A_MOM"),"MOM_3yr":r.get("FF_pct_3A_MOM_3yr")})
    s7_rows.append({"player_id":pid,"player_name":r["player_name"],"team":r["team"],
                    "position":r["position"],"season":season,"toi_min":r["toi_min"],
                    "metric":"CF","raw":r.get("CF_pct"),"ZA":r.get("CF_pct_ZA"),
                    "3A":r.get("CF_pct_3A"),
                    "QOC":r.get("CFQOC"),"QOL":r.get("CFQOL"),
                    "Rel_F":r.get("RelCF_F_pct"),"Rel_A":r.get("RelCF_A_pct"),
                    "Rel":r.get("RelCF_pct"),
                    "MOM":r.get("CF_pct_3A_MOM"),"MOM_3yr":r.get("CF_pct_3A_MOM_3yr")})
s7 = pd.DataFrame(s7_rows)
s7.to_csv(OUT_DT/"stage7_annual_ratings.csv", index=False)

print(f"\n[done] outputs in {OUT} and {OUT_DT}")
