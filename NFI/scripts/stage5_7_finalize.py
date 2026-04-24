"""Finalize Stage 5 R² summary + Stage 7 annual ratings + V1b FA variant.

V1b = forward-only on-ice CNFI+MNFI% (same per-player metric as V5/NFI%,
but IOC/IOL restricted to forward linemates/opponents, and player list filtered
to forwards).

Writes to NFI/output/zone_adjustment/complete_decision_tree/:
  stage5_r2_summary.csv       (revised)
  stage7_annual_ratings.csv   (revised)
Writes to NFI/output/fully_adjusted/:
  top30_FA_V1b_pooled.csv
  top30_FA_V1b_2526.csv
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DT  = ROOT / "NFI/output/zone_adjustment/complete_decision_tree"
OUT_FA  = ROOT / "NFI/output/fully_adjusted"
TEAM_METRICS = ROOT / "NFI/output/team_level_all_metrics.csv"
POS_CSV = ROOT / "NFI/output/player_positions.csv"

ABBR = {"ARI":"UTA"}
POOLED = {"20222023","20232024","20242025","20252026"}
CURRENT = "20252026"
def norm(a): return ABBR.get(a,a) if a else a

print("[0] loading ...")
pp_all = pd.read_csv(OUT_FA / "player_fully_adjusted.csv")  # has all FA columns
factors = json.load(open("/tmp/fa_factors.json"))
with open("/tmp/fa_shared.pkl","rb") as f:
    shared_minutes = pickle.load(f)

pos_df = pd.read_csv(POS_CSV)
pos_map = dict(zip(pos_df["player_id"].astype(int), pos_df["pos_group"].astype(str)))
name_map = dict(zip(pos_df["player_id"].astype(int), pos_df["player_name"].astype(str)))

# Re-load ppdf (pkl still has raw attribution with cf_fen etc)
ppdf_raw = pd.read_pickle("/tmp/s4_ppdf.pkl")
# Merge in the FA columns from player_fully_adjusted
keep_cols = ["player_id","season","NFI_pct","ZA_NFI_emp","IOC_NFI","IOL_NFI","FA_NFI_emp",
             "CF_pct","ZA_CF_emp","ZA_CF_trad","IOC_CF","IOL_CF","FA_CF_emp","FA_CF_trad",
             "FF_pct","ZA_FF_emp","ZA_FF_trad","IOC_FF","IOL_FF","FA_FF_emp","FA_FF_trad"]
pp_all_s = pp_all.rename(columns={"player_name":"name","position":"pos"})[keep_cols + ["name","pos","team","toi_sec"]]
ppdf = pp_all_s.copy()
ppdf["season"] = ppdf["season"].astype(str)

# ----------------------------------------------------------------------
# V1b: FORWARD-ONLY on-ice CNFI+MNFI%
# Use forward-restricted LM and OPP matrices from shared_minutes
# ----------------------------------------------------------------------
print("[1] computing V1b (forward-only on-ice CNFI+MNFI%) FA ...")
# Load shared_events from FA pipeline caching
# We need forward-only subset of M_for_cm, M_ag_cm. But we didn't persist shared_events.
# Approximation: V1b on per-player ratings = same raw NFI% (cm_pct) but with IOC/IOL
# restricted to forward pairs using the shared-minutes LM/OPP, and restricting
# the shared-event matrices to forward-forward subset.
#
# Since shared_events isn't cached, we'll use the existing IOL_NFI and IOC_NFI
# from the FA pipeline (computed over ALL skaters) as an approximation for V1b.
# To get true forward-only, we'd re-run shared-event accumulation with F filter.
#
# Alternative: recompute IOL/IOC using only forward linemates/opponents with the
# existing LM matrix masked to F-F pairs AND using already-computed
# "rating_without_i" from saved shared_events. Without shared_events persisted
# we approximate by restricting LM/OPP to F-F and using player's ZA_NFI as rating.
# This is effectively a Jacobi-style single-pass QoT using ZA ratings, forward
# subset.
#
# Build per-season forward masks
results_ioc = {}; results_iol = {}; results_fa = {}; betas = {}
for season in sorted(POOLED):
    info = shared_minutes[season]
    idx, pids_s = info["idx"], info["pids"]
    LM, OPP = info["LM"], info["OPP"]
    n = len(pids_s)
    is_fwd = np.array([pos_map.get(p) == "F" for p in pids_s])
    # Masked matrices: zero out rows and cols where player not forward
    LM_f  = LM.copy();  OPP_f  = OPP.copy()
    LM_f[~is_fwd, :] = 0; LM_f[:, ~is_fwd] = 0
    OPP_f[~is_fwd, :] = 0; OPP_f[:, ~is_fwd] = 0
    # Use ZA_NFI_emp as the rating for weighting
    sub = ppdf[ppdf["season"]==season]
    raw_vec = np.full(n, np.nan); za_vec = np.full(n, np.nan)
    for _, row in sub.iterrows():
        pid = int(row["player_id"])
        if pid in idx:
            i = idx[pid]
            raw_vec[i] = row["NFI_pct"]
            za_vec[i]  = row["ZA_NFI_emp"]
    # IOC and IOL from ZA ratings over forward-subset LM/OPP
    # (single-pass; no without-me correction since we don't have shared events cached)
    raw_use = np.where(np.isnan(raw_vec), 0.0, raw_vec)
    denom_lm  = LM_f.sum(axis=1);  denom_opp = OPP_f.sum(axis=1)
    IOL = np.where(denom_lm>0,  (LM_f  @ raw_use)/np.where(denom_lm>0, denom_lm, 1), np.nan)
    IOC = np.where(denom_opp>0, (OPP_f @ raw_use)/np.where(denom_opp>0, denom_opp, 1), np.nan)
    # Only forwards get valid FA
    valid = is_fwd & ~np.isnan(IOC) & ~np.isnan(IOL) & ~np.isnan(za_vec)
    if valid.sum() < 20: continue
    X = np.column_stack([np.ones(valid.sum()), IOC[valid], IOL[valid]])
    y = za_vec[valid]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    betas[season] = (float(beta[0]), float(beta[1]), float(beta[2]))
    mIOC = IOC[valid].mean(); mIOL = IOL[valid].mean()
    FA = za_vec - beta[1]*(IOC - mIOC) - beta[2]*(IOL - mIOL)
    for i, pid in enumerate(pids_s):
        if is_fwd[i]:
            key = (pid, season)
            if not np.isnan(IOC[i]): results_ioc[key] = float(IOC[i])
            if not np.isnan(IOL[i]): results_iol[key] = float(IOL[i])
            if not np.isnan(FA[i]):  results_fa[key]  = float(FA[i])

print(f"  V1b betas per season: { {s:(round(b[1],3),round(b[2],3)) for s,b in betas.items()} }")

def attach(df, col, rating_map):
    df[col] = df.apply(lambda r: rating_map.get((int(r["player_id"]), r["season"]), np.nan), axis=1)
    return df
ppdf = attach(ppdf, "IOC_V1b", results_ioc)
ppdf = attach(ppdf, "IOL_V1b", results_iol)
ppdf = attach(ppdf, "FA_V1b",  results_fa)
# Raw/ZA for V1b = same as NFI but only for forwards
ppdf["V1b_raw"] = np.where(ppdf["pos"]=="F", ppdf["NFI_pct"], np.nan)
ppdf["V1b_ZA"]  = np.where(ppdf["pos"]=="F", ppdf["ZA_NFI_emp"], np.nan)

# ----------------------------------------------------------------------
# Stage 5 R² summary — all metrics at every stage.
# ----------------------------------------------------------------------
print("[2] Stage 5 R² summary ...")
tm = pd.read_csv(TEAM_METRICS)
tm["season"] = tm["season"].astype(str)
tm = tm[tm["season"].isin(POOLED)][["season","team","points"]]

def team_agg(df, col):
    d = df.dropna(subset=[col,"team"]).copy()
    d["w"] = d["toi_sec"]
    return d.groupby(["season","team"]).apply(
        lambda x: np.average(x[col], weights=x["w"])).rename(col).reset_index()

def r2(x, y):
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 5: return np.nan, np.nan
    r = np.corrcoef(x[mask], y[mask])[0,1]
    return r*r, r

stage5 = []
for metric, raw_col, za_col, za_trad_col, fa_col, fa_trad_col, zone_method, factor_pp in [
    ("V5 (NFI% on-ice all skaters)", "NFI_pct", "ZA_NFI_emp", None, "FA_NFI_emp", None, "ZA empirical", 10.71),
    ("V1b (NFI% on-ice forwards)",    "V1b_raw", "V1b_ZA", None, "FA_V1b", None, "ZA empirical", 10.71),
    ("Fenwick CF%",                   "FF_pct", "ZA_FF_emp", "ZA_FF_trad", "FA_FF_emp", "FA_FF_trad", "ZA empirical", 11.91),
    ("Corsi CF%",                     "CF_pct", "ZA_CF_emp", "ZA_CF_trad", "FA_CF_emp", "FA_CF_trad", "ZA empirical", 3.89),
]:
    row = {"metric": metric, "zone_method": zone_method, "factor_pp": factor_pp}
    # Pooled team-aggregated R²
    for label, col in [("R2_raw",raw_col),("R2_ZA_emp",za_col),
                        ("R2_ZA_trad",za_trad_col),
                        ("R2_FA_emp",fa_col),("R2_FA_trad",fa_trad_col)]:
        if col is None:
            row[label] = np.nan; continue
        tb = team_agg(ppdf, col).merge(tm, on=["season","team"], how="inner")
        r2v, rv = r2(tb[col].values.astype(float), tb["points"].values.astype(float))
        row[label] = r2v
        row[label.replace("R2","r")] = rv
    stage5.append(row)

# Prior team-level Stage 1-3 results (for reference rows)
prior_rows = [
    {"metric":"V5 team-level (Stage 1-3)", "R2_raw":0.5955, "R2_DZ":0.4705,
     "R2_ZA_emp":0.6000, "zone_method":"ZA 2.5x", "factor_pp":26.78, "note":"team-agg"},
    {"metric":"HD Fenwick FF%", "R2_raw":0.4804, "R2_ZA_emp":0.4819,
     "R2_ZA_trad":0.4813, "zone_method":"ZA 1x", "factor_pp":10.52, "note":"team-agg"},
    {"metric":"HD Corsi CF%",   "R2_raw":0.2238, "R2_ZA_emp":0.2168,
     "R2_ZA_trad":0.2102, "zone_method":"ZA emp", "factor_pp":1.82, "note":"block-coord-fixed"},
    {"metric":"xG% (MP team 5v5)","R2_raw":0.5377,"R2_ZA_emp":0.5484,
     "zone_method":"ZA HD proxy", "factor_pp":None, "note":"team-agg"},
    {"metric":"PDO (ES regulation)","R2_raw":0.3360,
     "zone_method":"neg control", "factor_pp":None, "note":"team-agg"},
]
stage5_df = pd.DataFrame(stage5 + prior_rows)
stage5_df.to_csv(OUT_DT / "stage5_r2_summary.csv", index=False)

print("\n=== STAGE 5 R² SUMMARY (revised) ===")
cols = ["metric","R2_raw","R2_ZA_emp","R2_ZA_trad","R2_FA_emp","R2_FA_trad","zone_method","factor_pp"]
with pd.option_context("display.max_columns", None, "display.width", 150,
                       "display.float_format", "{:.4f}".format):
    print(stage5_df[[c for c in cols if c in stage5_df.columns]].to_string(index=False))

# ----------------------------------------------------------------------
# Stage 7 annual ratings CSV (with corrected FA)
# ----------------------------------------------------------------------
print("\n[3] Stage 7 annual ratings ...")
s7_rows = []
for _, r in ppdf.iterrows():
    pid = int(r["player_id"]); season = r["season"]
    # V5
    s7_rows.append({"player_id":pid,"name":r["name"],"team":r["team"],"pos":r["pos"],
                    "season":season,"metric":"V5","toi_min":r["toi_sec"]/60,
                    "raw":r["NFI_pct"],"ZA_emp":r["ZA_NFI_emp"],"ZA_trad":np.nan,
                    "FA_emp":r["FA_NFI_emp"],"FA_trad":np.nan,
                    "IOC":r["IOC_NFI"],"IOL":r["IOL_NFI"]})
    s7_rows.append({"player_id":pid,"name":r["name"],"team":r["team"],"pos":r["pos"],
                    "season":season,"metric":"Fenwick","toi_min":r["toi_sec"]/60,
                    "raw":r["FF_pct"],"ZA_emp":r["ZA_FF_emp"],"ZA_trad":r["ZA_FF_trad"],
                    "FA_emp":r["FA_FF_emp"],"FA_trad":r["FA_FF_trad"],
                    "IOC":r["IOC_FF"],"IOL":r["IOL_FF"]})
    s7_rows.append({"player_id":pid,"name":r["name"],"team":r["team"],"pos":r["pos"],
                    "season":season,"metric":"Corsi","toi_min":r["toi_sec"]/60,
                    "raw":r["CF_pct"],"ZA_emp":r["ZA_CF_emp"],"ZA_trad":r["ZA_CF_trad"],
                    "FA_emp":r["FA_CF_emp"],"FA_trad":r["FA_CF_trad"],
                    "IOC":r["IOC_CF"],"IOL":r["IOL_CF"]})
    if r["pos"] == "F":
        s7_rows.append({"player_id":pid,"name":r["name"],"team":r["team"],"pos":r["pos"],
                        "season":season,"metric":"V1b","toi_min":r["toi_sec"]/60,
                        "raw":r["NFI_pct"],"ZA_emp":r["ZA_NFI_emp"],"ZA_trad":np.nan,
                        "FA_emp":r["FA_V1b"],"FA_trad":np.nan,
                        "IOC":r["IOC_V1b"],"IOL":r["IOL_V1b"]})
s7 = pd.DataFrame(s7_rows)
s7.to_csv(OUT_DT / "stage7_annual_ratings.csv", index=False)
print(f"    wrote {len(s7)} rows to {OUT_DT/'stage7_annual_ratings.csv'}")

# V1b top-30 pooled
grp = ppdf[ppdf["pos"]=="F"].groupby(["player_id","name","team"]).agg(
    n_seasons=("season","count"),
    toi_total=("toi_sec", lambda s: s.sum()/60),
    raw_mean=("NFI_pct","mean"), ZA_mean=("ZA_NFI_emp","mean"),
    IOC_mean=("IOC_V1b","mean"), IOL_mean=("IOL_V1b","mean"),
    FA_mean=("FA_V1b","mean"),
).reset_index()
grp = grp[grp["toi_total"]>=500].sort_values("FA_mean", ascending=False).reset_index(drop=True)
grp["small_sample"] = grp["toi_total"] < 1500
grp.to_csv(OUT_FA / "top30_FA_V1b_pooled.csv", index=False)

# V1b 2025-26 top-30
curr = ppdf[(ppdf["season"]==CURRENT) & (ppdf["pos"]=="F")].copy()
curr["toi_min"] = curr["toi_sec"]/60
curr = curr.sort_values("FA_V1b", ascending=False)
curr["small_sample"] = curr["toi_min"] < 500
curr.to_csv(OUT_FA / "top30_FA_V1b_2526.csv", index=False)

print("\n[done]")
