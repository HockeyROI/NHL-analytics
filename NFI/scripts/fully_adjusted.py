"""Fully Adjusted (FA) metric pipeline — Steps 2-5.

Methodology (confirmed):
  FA_metric = ZA_metric − β_IOC × (IOC − mean IOC) − β_IOL × (IOL − mean IOL)
  where:
    IOC_i = shared-ES-TOI weighted avg of RAW ratings of opponents faced
    IOL_i = shared-ES-TOI weighted avg of RAW ratings of linemates
    β_IOC, β_IOL from single OLS: ZA_metric ~ IOC + IOL on all qualifying players
Single-pass, stable.

Inputs:
  /tmp/s4_ppdf.pkl (cached per-player attribution from Stage 4)
  /tmp/fa_factors.json (empirical zone factors from Step 1)
  shift_data.csv (for shared-minutes matrix)

Outputs:
  NFI/output/fully_adjusted/
    player_fully_adjusted.csv
    team_fully_adjusted.csv
    horse_race_fully_adjusted.csv
    current_season_player_fully_adjusted.csv
    current_season_team_fully_adjusted.csv
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.stats as st
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "NFI/output/fully_adjusted"
OUT.mkdir(parents=True, exist_ok=True)
SHIFT_CSV = ROOT / "NFI/Geometry_post/Data/shift_data.csv"
TEAM_METRICS = ROOT / "NFI/output/team_level_all_metrics.csv"
POS_CSV = ROOT / "NFI/output/player_positions.csv"
GAME_IDS = ROOT / "Data/game_ids.csv"

ABBR = {"ARI":"UTA"}
POOLED = {"20222023","20232024","20242025","20252026"}
CURRENT = "20252026"

def norm(a): return ABBR.get(a,a) if a else a

# ----------------------------------------------------------------------
print("[0] loading cached inputs ...")
ppdf = pd.read_pickle("/tmp/s4_ppdf.pkl")
print(f"    player-seasons cached: {len(ppdf)}")
factors = json.load(open("/tmp/fa_factors.json"))
print(f"    factors: {factors}")

# Load position
pos_df = pd.read_csv(POS_CSV)
pos_map = dict(zip(pos_df["player_id"].astype(int), pos_df["pos_group"].astype(str)))
name_map = dict(zip(pos_df["player_id"].astype(int), pos_df["player_name"].astype(str)))

# Primary team per (player, season) already in ppdf
# oz_ratio wasn't saved in s4_ppdf.pkl — recompute from cached faceoffs.
if "oz_ratio" not in ppdf.columns:
    print("    computing per-player oz_ratio from cached faceoffs ...")
    from bisect import bisect_right
    FLIP = {"O":"D","D":"O","N":"N"}
    fo_df = pd.read_pickle("/tmp/dt_faceoffs.pkl")
    fo_by_g = {g: grp[["t","winner","loser","zone_from_winner"]].to_numpy()
               for g, grp in fo_df.groupby("game_id", sort=False)}
    games_games = pd.read_csv(GAME_IDS)
    games_games = games_games[(games_games["season"].astype(str).isin(POOLED)) &
                              (games_games["game_type"]=="regular")]
    g2s_local = dict(zip(games_games["game_id"].astype(int), games_games["season"].astype(str)))
    # Load shifts for faceoff attribution (subset to games in scope)
    _sd = pd.read_csv(SHIFT_CSV,
        usecols=["game_id","player_id","team_abbrev","abs_start_secs","abs_end_secs"])
    _sd = _sd.dropna(subset=["player_id","abs_start_secs","abs_end_secs"])
    _sd["game_id"] = _sd["game_id"].astype("int64")
    _sd = _sd[_sd["game_id"].isin(set(g2s_local.keys()))]
    _sd["player_id"] = _sd["player_id"].astype("int64")
    _sd["abs_start_secs"] = _sd["abs_start_secs"].astype("int32")
    _sd["abs_end_secs"]   = _sd["abs_end_secs"].astype("int32")
    _sd["team_abbrev"] = _sd["team_abbrev"].astype(str).map(norm)
    _sd = _sd.sort_values(["game_id","abs_start_secs"]).reset_index(drop=True)
    _sbg = {gid: (g["player_id"].to_numpy(), g["team_abbrev"].to_numpy(),
                  g["abs_start_secs"].to_numpy(), g["abs_end_secs"].to_numpy())
            for gid, g in _sd.groupby("game_id", sort=False)}
    pfo = defaultdict(lambda: {"oz":0,"dz":0})
    for gid, fos in fo_by_g.items():
        if gid not in _sbg: continue
        season = g2s_local[gid]
        pids_arr, teams_arr, starts_arr, ends_arr = _sbg[gid]
        for t_face, winner, loser, zone_w in fos:
            hi = bisect_right(starts_arr, t_face)
            for i in range(hi):
                if ends_arr[i] <= t_face: continue
                p_team = teams_arr[i]
                zp = zone_w if p_team == winner else FLIP[zone_w]
                if zp == "O":   pfo[(int(pids_arr[i]), season)]["oz"] += 1
                elif zp == "D": pfo[(int(pids_arr[i]), season)]["dz"] += 1
    pfo_df = pd.DataFrame([{"player_id":k[0],"season":k[1],**v} for k,v in pfo.items()])
    pfo_df["oz_ratio"] = pfo_df["oz"] / (pfo_df["oz"]+pfo_df["dz"]).replace(0,np.nan)
    ppdf = ppdf.merge(pfo_df[["player_id","season","oz_ratio"]], on=["player_id","season"], how="left")
    ppdf["oz_ratio"] = ppdf["oz_ratio"].fillna(0.5)
    print(f"    oz_ratio computed for {ppdf['oz_ratio'].notna().sum()} player-seasons")

# Raw % metrics
ppdf["NFI_pct"] = ppdf["cm_pct"]
ppdf["CF_pct"]  = ppdf["cor_pct"]
ppdf["FF_pct"]  = ppdf["fen_pct"]
# Apply empirical ZA per metric (and traditional 3.5pp for CF/FF)
ppdf["ZA_NFI_emp"] = ppdf["NFI_pct"] - factors["NFI"]     * (ppdf["oz_ratio"] - 0.5)
ppdf["ZA_CF_emp"]  = ppdf["CF_pct"]  - factors["Corsi"]   * (ppdf["oz_ratio"] - 0.5)
ppdf["ZA_CF_trad"] = ppdf["CF_pct"]  - 0.035              * (ppdf["oz_ratio"] - 0.5)
ppdf["ZA_FF_emp"]  = ppdf["FF_pct"]  - factors["Fenwick"] * (ppdf["oz_ratio"] - 0.5)
ppdf["ZA_FF_trad"] = ppdf["FF_pct"]  - 0.035              * (ppdf["oz_ratio"] - 0.5)

# ----------------------------------------------------------------------
# Build shared-minutes matrices per season.
# ----------------------------------------------------------------------
print("[1] building shared-minutes matrices ...")
games_df = pd.read_csv(GAME_IDS)
games_df = games_df[(games_df["season"].astype(str).isin(POOLED)) &
                    (games_df["game_type"]=="regular")]
g2s = dict(zip(games_df["game_id"].astype(int), games_df["season"].astype(str)))
gids = sorted(g2s.keys())

sd = pd.read_csv(SHIFT_CSV,
    usecols=["game_id","player_id","team_abbrev","abs_start_secs","abs_end_secs"])
sd = sd.dropna(subset=["player_id","abs_start_secs","abs_end_secs"])
sd["game_id"] = sd["game_id"].astype("int64")
sd = sd[sd["game_id"].isin(gids)]
sd["player_id"] = sd["player_id"].astype("int64")
sd["abs_start_secs"] = sd["abs_start_secs"].astype("int32")
sd["abs_end_secs"]   = sd["abs_end_secs"].astype("int32")
sd["team_abbrev"] = sd["team_abbrev"].astype(str).map(norm)
sd = sd.sort_values(["game_id","abs_start_secs"]).reset_index(drop=True)

shifts_by_game = {}
for gid, g in sd.groupby("game_id", sort=False):
    shifts_by_game[gid] = {
        "pid": g["player_id"].to_numpy(),
        "team": g["team_abbrev"].to_numpy(),
        "s": g["abs_start_secs"].to_numpy(),
        "e": g["abs_end_secs"].to_numpy(),
    }

qualifying = {(int(r.player_id), r.season) for r in ppdf.itertuples()}

shared = {}
for season in sorted(POOLED):
    pids_s = sorted({pid for (pid, s) in qualifying if s == season})
    idx = {pid:i for i, pid in enumerate(pids_s)}
    n = len(pids_s)
    LM  = np.zeros((n,n), dtype=np.float32)
    OPP = np.zeros((n,n), dtype=np.float32)
    season_games = [g for g in gids if g2s[g]==season]
    for gi, gid in enumerate(season_games):
        s_arr = shifts_by_game.get(gid)
        if s_arr is None: continue
        pids, teams, starts, ends = s_arr["pid"], s_arr["team"], s_arr["s"], s_arr["e"]
        per_p_shifts = defaultdict(list)
        per_p_team = {}
        for i in range(len(pids)):
            pi = int(pids[i])
            if pi not in idx: continue
            per_p_shifts[pi].append((starts[i], ends[i]))
            per_p_team[pi] = teams[i]
        plist = list(per_p_shifts.keys())
        arrs = {p: np.array(per_p_shifts[p], dtype=np.int32) for p in plist}
        for a_i, pa in enumerate(plist):
            a_s, a_e = arrs[pa][:,0], arrs[pa][:,1]
            for b_i in range(a_i+1, len(plist)):
                pb = plist[b_i]
                b_s, b_e = arrs[pb][:,0], arrs[pb][:,1]
                ov = np.maximum(0, np.minimum(a_e[:,None], b_e[None,:])
                                    - np.maximum(a_s[:,None], b_s[None,:]))
                tot = ov.sum()
                if tot == 0: continue
                ia, ib = idx[pa], idx[pb]
                if per_p_team[pa] == per_p_team[pb]:
                    LM[ia, ib] += tot; LM[ib, ia] += tot
                else:
                    OPP[ia, ib] += tot; OPP[ib, ia] += tot
        if (gi+1) % 400 == 0 or gi+1 == len(season_games):
            print(f"    {season}: {gi+1}/{len(season_games)}")
    shared[season] = {"idx":idx, "LM":LM, "OPP":OPP, "pids":pids_s}
    print(f"  season {season}: n={n}")
# Save shared for future iterations
import pickle
with open("/tmp/fa_shared.pkl","wb") as f:
    pickle.dump({s:{k:v for k,v in d.items() if k in ("idx","pids","LM","OPP")}
                  for s,d in shared.items()}, f)
print("    saved shared matrices to /tmp/fa_shared.pkl")

# ----------------------------------------------------------------------
# Methodology A — compute IOC/IOL using RAW ratings, regress ZA on IOC/IOL,
# subtract mean-centered context.
# ----------------------------------------------------------------------
print("[2] computing IOC / IOL / FA per metric ...")
def compute_fa(metric_key, raw_col, za_col):
    """Return (IOC_map, IOL_map, FA_map, beta_IOC, beta_IOL) keyed by (pid, season)."""
    all_IOC, all_IOL, all_FA = {}, {}, {}
    betas = {}
    for season in sorted(POOLED):
        info = shared[season]; idx = info["idx"]; pids_s = info["pids"]
        LM, OPP = info["LM"], info["OPP"]
        n = len(pids_s)
        raw_vec = np.full(n, np.nan); za_vec = np.full(n, np.nan)
        sub = ppdf[ppdf["season"]==season]
        for _, row in sub.iterrows():
            pid = int(row["player_id"])
            if pid in idx:
                i = idx[pid]
                raw_vec[i] = row[raw_col]
                za_vec[i]  = row[za_col]
        valid = ~np.isnan(raw_vec) & ~np.isnan(za_vec)
        if valid.sum() < 20: continue
        # IOC / IOL from raw ratings
        raw_use = np.where(np.isnan(raw_vec), 0.0, raw_vec)
        denom_opp = OPP.sum(axis=1)
        denom_lm  = LM.sum(axis=1)
        IOC = np.divide(OPP @ raw_use, np.where(denom_opp>0, denom_opp, 1), out=np.full(n,np.nan), where=(denom_opp>0))
        IOL = np.divide(LM  @ raw_use, np.where(denom_lm>0,  denom_lm,  1), out=np.full(n,np.nan), where=(denom_lm>0))
        # Regression ZA ~ IOC + IOL on valid subset
        reg_valid = valid & ~np.isnan(IOC) & ~np.isnan(IOL)
        if reg_valid.sum() < 20: continue
        X = np.column_stack([np.ones(reg_valid.sum()), IOC[reg_valid], IOL[reg_valid]])
        y = za_vec[reg_valid]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        # Mean-center over regression sample
        mIOC = IOC[reg_valid].mean(); mIOL = IOL[reg_valid].mean()
        FA = za_vec - beta[1]*(IOC - mIOC) - beta[2]*(IOL - mIOL)
        # Store
        for i, pid in enumerate(pids_s):
            key = (pid, season)
            if not np.isnan(IOC[i]): all_IOC[key] = float(IOC[i])
            if not np.isnan(IOL[i]): all_IOL[key] = float(IOL[i])
            if not np.isnan(FA[i]):  all_FA[key]  = float(FA[i])
        betas[season] = (float(beta[0]), float(beta[1]), float(beta[2]))
    return all_IOC, all_IOL, all_FA, betas

# Compute for NFI, CF, FF — both empirical and traditional variants where applicable
results = {}
for metric_tag, raw_col, za_emp, za_trad in [
    ("NFI","NFI_pct","ZA_NFI_emp", None),  # no trad variant for NFI
    ("CF", "CF_pct", "ZA_CF_emp", "ZA_CF_trad"),
    ("FF", "FF_pct", "ZA_FF_emp", "ZA_FF_trad"),
]:
    ioc, iol, fa_emp, beta_emp = compute_fa(metric_tag, raw_col, za_emp)
    results[metric_tag] = {"IOC":ioc,"IOL":iol,"FA_emp":fa_emp,"beta_emp":beta_emp}
    if za_trad is not None:
        _, _, fa_trad, beta_trad = compute_fa(metric_tag, raw_col, za_trad)
        results[metric_tag]["FA_trad"] = fa_trad
        results[metric_tag]["beta_trad"] = beta_trad
    print(f"  {metric_tag}: betas(emp) per season: { {s:(round(b[1],3),round(b[2],3)) for s,b in beta_emp.items()} }")

# ----------------------------------------------------------------------
# Attach to ppdf
# ----------------------------------------------------------------------
def attach(df, col, rating_map):
    df[col] = df.apply(lambda r: rating_map.get((int(r["player_id"]), r["season"]), np.nan), axis=1)
    return df

for tag in ["NFI","CF","FF"]:
    ppdf = attach(ppdf, f"IOC_{tag}", results[tag]["IOC"])
    ppdf = attach(ppdf, f"IOL_{tag}", results[tag]["IOL"])
    ppdf = attach(ppdf, f"FA_{tag}_emp", results[tag]["FA_emp"])
    if "FA_trad" in results[tag]:
        ppdf = attach(ppdf, f"FA_{tag}_trad", results[tag]["FA_trad"])

# ----------------------------------------------------------------------
# STEP 3 — Horse race: team aggregation + R² vs points (pooled + current)
# ----------------------------------------------------------------------
print("[3] horse race ...")
tm = pd.read_csv(TEAM_METRICS)
tm["season"] = tm["season"].astype(str)
tm = tm[tm["season"].isin(POOLED)][["season","team","points"]]

# MoneyPuck xG% (team-level) for horse race baseline
print("    fetching MoneyPuck xG% ...")
xg_frames = []
for yr in [2022,2023,2024,2025]:
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{yr}/regular/teams.csv"
    subprocess.run(["curl","-sL","-A","Mozilla/5.0", url, "-o", f"/tmp/mp_{yr}.csv"], check=False)
    try:
        t = pd.read_csv(f"/tmp/mp_{yr}.csv")
        t5 = t[t["situation"]=="5on5"][["team","xGoalsPercentage"]].copy()
        t5["season"] = f"{yr}{yr+1}"
        xg_frames.append(t5)
    except Exception: pass
xg = pd.concat(xg_frames, ignore_index=True) if xg_frames else pd.DataFrame()
xg["team"] = xg["team"].map(norm) if len(xg) else []

def team_agg(df, col):
    d = df.dropna(subset=[col,"team"]).copy()
    d["w"] = d["toi_sec"]
    out = d.groupby(["season","team"]).apply(
        lambda x: np.average(x[col], weights=x["w"])).rename(col).reset_index()
    return out

def r2(x, y):
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 5: return np.nan, np.nan, 0
    r = np.corrcoef(x[mask], y[mask])[0,1]
    return r*r, r, int(mask.sum())

variants = [
    ("NFI% raw",           "NFI_pct"),
    ("NFI% ZA empirical",  "ZA_NFI_emp"),
    ("NFI% FA empirical",  "FA_NFI_emp"),
    ("CF% raw",            "CF_pct"),
    ("CF% ZA empirical",   "ZA_CF_emp"),
    ("CF% ZA traditional", "ZA_CF_trad"),
    ("CF% FA empirical",   "FA_CF_emp"),
    ("CF% FA traditional", "FA_CF_trad"),
    ("FF% raw",            "FF_pct"),
    ("FF% ZA empirical",   "ZA_FF_emp"),
    ("FF% ZA traditional", "ZA_FF_trad"),
    ("FF% FA empirical",   "FA_FF_emp"),
    ("FF% FA traditional", "FA_FF_trad"),
]

horse_rows = []
for label, col in variants:
    tb = team_agg(ppdf, col).merge(tm, on=["season","team"], how="inner")
    r_pool = r2(tb[col].values.astype(float), tb["points"].values.astype(float))
    tb_curr = tb[tb["season"]==CURRENT]
    r_curr = r2(tb_curr[col].values.astype(float), tb_curr["points"].values.astype(float))
    horse_rows.append({"metric":label, "R2_pooled":r_pool[0], "r_pooled":r_pool[1],
                       "N_pooled":r_pool[2], "R2_2526":r_curr[0], "r_2526":r_curr[1],
                       "N_2526":r_curr[2]})

# xG% (team-level)
if len(xg):
    xg_m = xg.merge(tm, on=["season","team"], how="inner")
    r_xg = r2(xg_m["xGoalsPercentage"].values.astype(float), xg_m["points"].values.astype(float))
    xg_curr = xg_m[xg_m["season"]==CURRENT]
    r_xg_c = r2(xg_curr["xGoalsPercentage"].values.astype(float), xg_curr["points"].values.astype(float))
    horse_rows.append({"metric":"xG% (MP team 5v5)",
                       "R2_pooled":r_xg[0], "r_pooled":r_xg[1], "N_pooled":r_xg[2],
                       "R2_2526":r_xg_c[0], "r_2526":r_xg_c[1], "N_2526":r_xg_c[2]})

horse = pd.DataFrame(horse_rows).sort_values("R2_pooled", ascending=False).reset_index(drop=True)
horse.to_csv(OUT/"horse_race_fully_adjusted.csv", index=False)
print("\n" + "="*90)
print("STEP 3 — HORSE RACE R² (pooled N=126, current-season N=32)")
print("="*90)
print(horse.to_string(index=False))

# ----------------------------------------------------------------------
# STEP 4 — Player lists
# ----------------------------------------------------------------------
print("\n[4] player lists ...")

def top30(df, col, n=30):
    d = df.dropna(subset=[col,"name"]).copy()
    return d.sort_values(col, ascending=False).head(n)

# Pooled top 30 (career mean weighted by TOI; min 500 min total)
for metric in ["NFI","CF","FF"]:
    grp = ppdf.groupby(["player_id","name","pos"]).agg(
        n_seasons=("season","count"),
        toi_total=("toi_sec", lambda s: s.sum()/60),
        team_recent=("team","last"),
        raw_mean=({"NFI":"NFI_pct","CF":"CF_pct","FF":"FF_pct"}[metric],"mean"),
        ZA_mean=({"NFI":"ZA_NFI_emp","CF":"ZA_CF_emp","FF":"ZA_FF_emp"}[metric],"mean"),
        FA_mean=({"NFI":"FA_NFI_emp","CF":"FA_CF_emp","FF":"FA_FF_emp"}[metric],"mean"),
    ).reset_index()
    grp = grp[grp["toi_total"]>=500].sort_values("FA_mean", ascending=False).reset_index(drop=True)
    grp["small_sample"] = grp["toi_total"] < 1500
    grp.to_csv(OUT/f"top30_FA_{metric}_pooled.csv", index=False)

# Current season top 30 (FA-NFI%)
curr_df = ppdf[ppdf["season"]==CURRENT].copy()
curr_df["toi_min"] = curr_df["toi_sec"]/60
for metric in ["NFI","CF","FF"]:
    dset = curr_df.sort_values(f"FA_{metric}_emp", ascending=False).reset_index(drop=True)
    dset["small_sample"] = dset["toi_min"] < 500
    dset.to_csv(OUT/f"top30_FA_{metric}_2526.csv", index=False)

# Cross-metric consistency (pooled)
top30_nfi = set(pd.read_csv(OUT/"top30_FA_NFI_pooled.csv").head(30)["player_id"].astype(int))
top30_cf  = set(pd.read_csv(OUT/"top30_FA_CF_pooled.csv" ).head(30)["player_id"].astype(int))
top30_ff  = set(pd.read_csv(OUT/"top30_FA_FF_pooled.csv" ).head(30)["player_id"].astype(int))
elite = top30_nfi & top30_cf & top30_ff
elite_df = pd.DataFrame([{"player_id":pid,"name":name_map.get(pid,""),
                           "pos":pos_map.get(pid,"")} for pid in elite])
elite_df.to_csv(OUT/"cross_metric_elite.csv", index=False)

# ----------------------------------------------------------------------
# STEP 5 — Streamlit CSVs
# ----------------------------------------------------------------------
print("[5] writing Streamlit CSVs ...")
player_cols = ["player_id","name","pos","team","season","toi_sec",
               "NFI_pct","ZA_NFI_emp","IOC_NFI","IOL_NFI","FA_NFI_emp",
               "CF_pct","ZA_CF_emp","ZA_CF_trad","IOC_CF","IOL_CF","FA_CF_emp","FA_CF_trad",
               "FF_pct","ZA_FF_emp","ZA_FF_trad","IOC_FF","IOL_FF","FA_FF_emp","FA_FF_trad"]
player_out = ppdf[[c for c in player_cols if c in ppdf.columns]].copy()
player_out["es_toi_min"] = player_out["toi_sec"]/60
player_out = player_out.rename(columns={"name":"player_name","pos":"position"})
player_out.to_csv(OUT/"player_fully_adjusted.csv", index=False)

# Team aggregated
team_rows = []
for season in sorted(POOLED):
    for team in sorted(ppdf[ppdf["season"]==season]["team"].dropna().unique()):
        sub = ppdf[(ppdf["season"]==season) & (ppdf["team"]==team)]
        if len(sub) == 0: continue
        row = {"season":season, "team":team, "n_players":len(sub),
               "team_toi_min":sub["toi_sec"].sum()/60}
        for col in ["NFI_pct","ZA_NFI_emp","FA_NFI_emp",
                    "CF_pct","ZA_CF_emp","ZA_CF_trad","FA_CF_emp","FA_CF_trad",
                    "FF_pct","ZA_FF_emp","ZA_FF_trad","FA_FF_emp","FA_FF_trad"]:
            vals = sub[[col,"toi_sec"]].dropna()
            if len(vals) == 0:
                row[col] = np.nan
            else:
                row[col] = float(np.average(vals[col], weights=vals["toi_sec"]))
        team_rows.append(row)
team_out = pd.DataFrame(team_rows)
# Merge points
team_out = team_out.merge(tm, on=["season","team"], how="left")
team_out.to_csv(OUT/"team_fully_adjusted.csv", index=False)

# Current season variants
player_out[player_out["season"]==CURRENT].to_csv(
    OUT/"current_season_player_fully_adjusted.csv", index=False)
team_out[team_out["season"]==CURRENT].to_csv(
    OUT/"current_season_team_fully_adjusted.csv", index=False)

print(f"\n[done] outputs in {OUT}")
