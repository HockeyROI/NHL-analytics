"""FA methodology with linemate-without-me IOL correction.

Problem: naive IOL_i = weighted avg of linemates' on-ice ratings suffers
from shared-event collinearity — at 5v5 ES, each shot event counts in BOTH
player i's AND linemate j's on-ice stats. So IOL_i ≈ rating_i, β_IOL → 1.

Fix: when computing linemate j's contribution to IOL_i, use j's rating
EXCLUDING events where i was also on the ice.

For each ordered pair (i, j) we need:
  M_for[i, j]     = events where i was on the shooter's team AND j on ice
  M_against[i, j] = events where i was on the opp-of-shooter side AND j on ice

Then j's rating without i = (cf_f_j − M_for[j, i]) / (cf_f_j − M_for[j, i] + ca_j − M_against[j, i]).

Outputs:
  NFI/output/fully_adjusted/
    player_fully_adjusted.csv         (revised with corrected FA)
    team_fully_adjusted.csv           (revised)
    horse_race_fully_adjusted.csv     (revised)
    top30_FA_{NFI,CF,FF}_pooled.csv   (revised)
    top30_FA_{NFI,CF,FF}_2526.csv     (revised)
    current_season_player_fully_adjusted.csv
    current_season_team_fully_adjusted.csv
    cross_metric_elite.csv
"""
from __future__ import annotations

import json, pickle
from pathlib import Path
from bisect import bisect_left, bisect_right
from collections import defaultdict
import numpy as np
import pandas as pd
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "NFI/output/fully_adjusted"
OUT.mkdir(parents=True, exist_ok=True)
SHOTS_CSV = ROOT / "NFI/output/shots_tagged.csv"
SHIFT_CSV = ROOT / "NFI/Geometry_post/Data/shift_data.csv"
TEAM_METRICS = ROOT / "NFI/output/team_level_all_metrics.csv"
POS_CSV = ROOT / "NFI/output/player_positions.csv"
GAME_IDS = ROOT / "Data/game_ids.csv"

ABBR = {"ARI":"UTA"}
POOLED = {"20222023","20232024","20242025","20252026"}
CURRENT = "20252026"
FEN = {"shot-on-goal","missed-shot","goal"}
COR = FEN | {"blocked-shot"}
def norm(a): return ABBR.get(a,a) if a else a

print("[0] loading cached inputs ...")
ppdf = pd.read_pickle("/tmp/s4_ppdf.pkl")
factors = json.load(open("/tmp/fa_factors.json"))
with open("/tmp/fa_shared.pkl","rb") as f:
    shared_minutes = pickle.load(f)

pos_df = pd.read_csv(POS_CSV)
pos_map = dict(zip(pos_df["player_id"].astype(int), pos_df["pos_group"].astype(str)))
name_map = dict(zip(pos_df["player_id"].astype(int), pos_df["player_name"].astype(str)))
print(f"    ppdf: {len(ppdf)} player-seasons; factors: {factors}")

# Recompute oz_ratio (wasn't saved in s4_ppdf)
if "oz_ratio" not in ppdf.columns:
    from bisect import bisect_right as br
    FLIP = {"O":"D","D":"O","N":"N"}
    fo_df = pd.read_pickle("/tmp/dt_faceoffs.pkl")
    fo_by_g_fo = {g: grp[["t","winner","loser","zone_from_winner"]].to_numpy()
                  for g, grp in fo_df.groupby("game_id", sort=False)}
    games_games = pd.read_csv(GAME_IDS)
    games_games = games_games[(games_games["season"].astype(str).isin(POOLED)) &
                              (games_games["game_type"]=="regular")]
    g2s_fo = dict(zip(games_games["game_id"].astype(int),
                      games_games["season"].astype(str)))
    _sd = pd.read_csv(SHIFT_CSV,
        usecols=["game_id","player_id","team_abbrev","abs_start_secs","abs_end_secs"])
    _sd = _sd.dropna(subset=["player_id","abs_start_secs","abs_end_secs"])
    _sd["game_id"] = _sd["game_id"].astype("int64")
    _sd = _sd[_sd["game_id"].isin(g2s_fo.keys())]
    _sd["player_id"] = _sd["player_id"].astype("int64")
    _sd["abs_start_secs"] = _sd["abs_start_secs"].astype("int32")
    _sd["abs_end_secs"] = _sd["abs_end_secs"].astype("int32")
    _sd["team_abbrev"] = _sd["team_abbrev"].astype(str).map(norm)
    _sd = _sd.sort_values(["game_id","abs_start_secs"]).reset_index(drop=True)
    _sbg = {gid:(g["player_id"].to_numpy(), g["team_abbrev"].to_numpy(),
                 g["abs_start_secs"].to_numpy(), g["abs_end_secs"].to_numpy())
            for gid,g in _sd.groupby("game_id", sort=False)}
    pfo = defaultdict(lambda: {"oz":0,"dz":0})
    for gid, fos in fo_by_g_fo.items():
        if gid not in _sbg: continue
        season = g2s_fo[gid]
        pids_, teams_, starts_, ends_ = _sbg[gid]
        for t_face, winner, loser, zone_w in fos:
            hi = br(starts_, t_face)
            for i in range(hi):
                if ends_[i] <= t_face: continue
                p_team = teams_[i]
                zp = zone_w if p_team==winner else FLIP[zone_w]
                if zp=="O":   pfo[(int(pids_[i]), season)]["oz"]+=1
                elif zp=="D": pfo[(int(pids_[i]), season)]["dz"]+=1
    pfo_df = pd.DataFrame([{"player_id":k[0],"season":k[1],**v} for k,v in pfo.items()])
    pfo_df["oz_ratio"] = pfo_df["oz"] / (pfo_df["oz"]+pfo_df["dz"]).replace(0,np.nan)
    ppdf = ppdf.merge(pfo_df[["player_id","season","oz_ratio"]],
                      on=["player_id","season"], how="left")
    ppdf["oz_ratio"] = ppdf["oz_ratio"].fillna(0.5)

# Raw % and ZA columns
ppdf["NFI_pct"] = ppdf["cm_pct"]
ppdf["CF_pct"]  = ppdf["cor_pct"]
ppdf["FF_pct"]  = ppdf["fen_pct"]
ppdf["ZA_NFI_emp"] = ppdf["NFI_pct"] - factors["NFI"]     * (ppdf["oz_ratio"]-0.5)
ppdf["ZA_CF_emp"]  = ppdf["CF_pct"]  - factors["Corsi"]   * (ppdf["oz_ratio"]-0.5)
ppdf["ZA_CF_trad"] = ppdf["CF_pct"]  - 0.035              * (ppdf["oz_ratio"]-0.5)
ppdf["ZA_FF_emp"]  = ppdf["FF_pct"]  - factors["Fenwick"] * (ppdf["oz_ratio"]-0.5)
ppdf["ZA_FF_trad"] = ppdf["FF_pct"]  - 0.035              * (ppdf["oz_ratio"]-0.5)

# ---------------------------------------------------------------------
# Load shifts and shots for shared-event accumulation
# ---------------------------------------------------------------------
print("[1] loading shifts ...")
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
sd["abs_end_secs"] = sd["abs_end_secs"].astype("int32")
sd["team_abbrev"] = sd["team_abbrev"].astype(str).map(norm)
sd = sd.sort_values(["game_id","abs_start_secs"]).reset_index(drop=True)
shifts_by_game = {gid:(g["player_id"].to_numpy(), g["team_abbrev"].to_numpy(),
                       g["abs_start_secs"].to_numpy(), g["abs_end_secs"].to_numpy())
                  for gid, g in sd.groupby("game_id", sort=False)}

print("[2] loading shots ...")
xd = pd.read_csv(SHOTS_CSV,
    usecols=["game_id","season","abs_time","event_type","shooting_team_abbrev","zone","state"])
xd["season"] = xd["season"].astype(str)
xd = xd[(xd["season"].isin(POOLED)) & (xd["state"]=="ES")]
xd["shooting_team_abbrev"] = xd["shooting_team_abbrev"].astype(str).map(norm)
xd["is_cor"] = xd["event_type"].isin(COR)
xd["is_fen"] = xd["event_type"].isin(FEN)
xd["is_cm"]  = xd["zone"].isin(["CNFI","MNFI"])
xd = xd.sort_values(["game_id","abs_time"]).reset_index(drop=True)
shots_by_game = {gid:{"t":g["abs_time"].to_numpy(),
                      "team":g["shooting_team_abbrev"].to_numpy(),
                      "cor":g["is_cor"].to_numpy(),
                      "fen":g["is_fen"].to_numpy(),
                      "cm": g["is_cm"].to_numpy()}
                 for gid, g in xd.groupby("game_id", sort=False)}

# ---------------------------------------------------------------------
# Build shared-event matrices per season.
# M_for_X[i, j]  = events where i on shooter-side AND j on ice AND metric X (cor/fen/cm)
# M_ag_X[i, j]   = events where i on opp-of-shooter side AND j on ice AND metric X
# ---------------------------------------------------------------------
print("[3] building shared-event matrices (this is the main compute) ...")
qualifying = {(int(r.player_id), r.season) for r in ppdf.itertuples()}
season_idx = {s: shared_minutes[s]["idx"] for s in shared_minutes}

shared_events = {}
for s in sorted(POOLED):
    n = len(shared_minutes[s]["pids"])
    shared_events[s] = {
        "M_for_cor":     np.zeros((n,n), dtype=np.float32),
        "M_ag_cor":      np.zeros((n,n), dtype=np.float32),
        "M_for_fen":     np.zeros((n,n), dtype=np.float32),
        "M_ag_fen":      np.zeros((n,n), dtype=np.float32),
        "M_for_cm":      np.zeros((n,n), dtype=np.float32),
        "M_ag_cm":       np.zeros((n,n), dtype=np.float32),
    }

for gi, gid in enumerate(gids):
    if gid not in shifts_by_game or gid not in shots_by_game: continue
    season = g2s[gid]
    idx = season_idx[season]
    pids, teams, starts, ends = shifts_by_game[gid]
    shots = shots_by_game[gid]
    t_arr = shots["t"]
    n_shots = len(t_arr)
    if n_shots == 0: continue
    # For each shot: find on-ice players, partition by team
    me = shared_events[season]
    for si in range(n_shots):
        t = t_arr[si]
        # Active shifts at time t: start <= t < end
        hi = bisect_right(starts, t)
        # Collect on-ice player indices + teams
        A_list = []; B_list = []
        shooter = shots["team"][si]
        for i in range(hi):
            if ends[i] <= t: continue
            pid_i = int(pids[i])
            if pid_i not in idx: continue
            ii = idx[pid_i]
            if teams[i] == shooter:
                A_list.append(ii)
            else:
                B_list.append(ii)
        if not A_list and not B_list: continue
        A = np.array(A_list, dtype=np.int32) if A_list else np.empty(0, dtype=np.int32)
        B = np.array(B_list, dtype=np.int32) if B_list else np.empty(0, dtype=np.int32)
        all_ice = np.concatenate([A, B]) if (len(A) or len(B)) else np.empty(0, dtype=np.int32)
        is_cor = shots["cor"][si]; is_fen = shots["fen"][si]; is_cm = shots["cm"][si]
        # For-side: rows in A, cols in all
        if len(A) and len(all_ice):
            rows_A = A[:, None]; cols = all_ice[None, :]
            if is_cor: me["M_for_cor"][rows_A, cols] += 1
            if is_fen: me["M_for_fen"][rows_A, cols] += 1
            if is_cm:  me["M_for_cm"] [rows_A, cols] += 1
        # Against-side: rows in B, cols in all
        if len(B) and len(all_ice):
            rows_B = B[:, None]
            if is_cor: me["M_ag_cor"][rows_B, cols] += 1
            if is_fen: me["M_ag_fen"][rows_B, cols] += 1
            if is_cm:  me["M_ag_cm"] [rows_B, cols] += 1
    if (gi+1) % 500 == 0 or gi+1 == len(gids):
        print(f"    game {gi+1}/{len(gids)}")

# Sanity check: M_for_cor.sum(axis=1) diagonal should relate to player's cf_f
# (diagonal i=j: events where i on shooter side AND i on ice = i's cf_f)
for s in sorted(POOLED):
    m_sum = shared_events[s]["M_for_cor"].diagonal().sum()
    print(f"    sanity {s}: sum diagonals of M_for_cor = {int(m_sum):,}")

# ---------------------------------------------------------------------
# Compute IOL (linemate-without-me) and IOC (opponent-without-me).
# ---------------------------------------------------------------------
print("[4] computing linemate-without-me IOL and opponent-without-me IOC per metric ...")

def compute_ratings(metric_key, raw_col, za_col, shared_f_key, shared_a_key):
    """Return dict (pid, season) -> FA_value, plus season betas."""
    results_ioc = {}
    results_iol = {}
    results_fa  = {}
    betas = {}
    for season in sorted(POOLED):
        info = shared_minutes[season]
        idx, pids_s = info["idx"], info["pids"]
        LM, OPP = info["LM"], info["OPP"]
        M_for = shared_events[season][shared_f_key]
        M_ag  = shared_events[season][shared_a_key]
        n = len(pids_s)
        # Player totals for numerator / denominator
        total_f = np.zeros(n); total_a = np.zeros(n); raw_vec = np.full(n, np.nan)
        za_vec  = np.full(n, np.nan)
        f_col, a_col = {"cor":("cf_cor","ca_cor"), "fen":("cf_fen","ca_fen"),
                         "cm":("cf_cm","ca_cm")}[metric_key]
        sub = ppdf[ppdf["season"]==season]
        for _, row in sub.iterrows():
            pid = int(row["player_id"])
            if pid in idx:
                i = idx[pid]
                total_f[i] = row[f_col]
                total_a[i] = row[a_col]
                raw_vec[i] = row[raw_col]
                za_vec[i]  = row[za_col]

        # IOL: linemate-without-me
        # For each i: Σ_j LM[i,j] × rating_j_without_i / Σ_j LM[i,j]
        # rating_j_without_i = (total_f_j − M_for[j,i]) / (total_f_j − M_for[j,i] + total_a_j − M_ag[j,i])
        # Vectorized per i: compute numer & denom arrays for all j.
        IOL = np.full(n, np.nan)
        IOC = np.full(n, np.nan)
        for i in range(n):
            # Linemates (same team): LM[i,:] > 0
            LM_row = LM[i]
            # For linemate, the "without i" contribution to j's stats:
            # j_f_without_i = total_f[j] - M_for[j, i]
            # j_a_without_i = total_a[j] - M_ag[j, i]
            j_f = total_f - M_for[:, i]
            j_a = total_a - M_ag[:, i]
            denom_j = j_f + j_a
            with np.errstate(invalid="ignore", divide="ignore"):
                rating_wo_i = np.where(denom_j > 0, j_f / denom_j, np.nan)
            # IOL weights = LM[i,:] (excluding self)
            mask_lm = (LM_row > 0) & ~np.isnan(rating_wo_i)
            if mask_lm.any() and LM_row[mask_lm].sum() > 0:
                IOL[i] = float(np.sum(LM_row[mask_lm] * rating_wo_i[mask_lm]) / LM_row[mask_lm].sum())
            # IOC: opponent-without-me. Opponents j faced by i, weighted by OPP[i,j].
            # j is on different team from i. "j_rating_without_i" same formula.
            OPP_row = OPP[i]
            mask_opp = (OPP_row > 0) & ~np.isnan(rating_wo_i)
            if mask_opp.any() and OPP_row[mask_opp].sum() > 0:
                IOC[i] = float(np.sum(OPP_row[mask_opp] * rating_wo_i[mask_opp]) / OPP_row[mask_opp].sum())

        # Regress ZA ~ IOC + IOL on valid
        valid = ~np.isnan(IOC) & ~np.isnan(IOL) & ~np.isnan(za_vec)
        if valid.sum() < 20: continue
        X = np.column_stack([np.ones(valid.sum()), IOC[valid], IOL[valid]])
        y = za_vec[valid]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        betas[season] = (float(beta[0]), float(beta[1]), float(beta[2]))
        # Mean-center IOC, IOL on regression sample
        mIOC = IOC[valid].mean(); mIOL = IOL[valid].mean()
        FA = za_vec - beta[1]*(IOC - mIOC) - beta[2]*(IOL - mIOL)

        for i, pid in enumerate(pids_s):
            key = (pid, season)
            if not np.isnan(IOC[i]): results_ioc[key] = float(IOC[i])
            if not np.isnan(IOL[i]): results_iol[key] = float(IOL[i])
            if not np.isnan(FA[i]):  results_fa[key]  = float(FA[i])
    return results_ioc, results_iol, results_fa, betas

# NFI (uses cm counters, ZA empirical)
ioc_n, iol_n, fa_n_emp, beta_n_emp = compute_ratings("cm","NFI_pct","ZA_NFI_emp","M_for_cm","M_ag_cm")
print(f"  NFI emp betas per season: { {s:(round(b[1],3),round(b[2],3)) for s,b in beta_n_emp.items()} }")
# CF (Corsi)
ioc_c, iol_c, fa_c_emp, beta_c_emp = compute_ratings("cor","CF_pct","ZA_CF_emp","M_for_cor","M_ag_cor")
print(f"  CF emp betas per season:  { {s:(round(b[1],3),round(b[2],3)) for s,b in beta_c_emp.items()} }")
_,     _,     fa_c_trd, beta_c_trd = compute_ratings("cor","CF_pct","ZA_CF_trad","M_for_cor","M_ag_cor")
# FF (Fenwick)
ioc_f, iol_f, fa_f_emp, beta_f_emp = compute_ratings("fen","FF_pct","ZA_FF_emp","M_for_fen","M_ag_fen")
print(f"  FF emp betas per season:  { {s:(round(b[1],3),round(b[2],3)) for s,b in beta_f_emp.items()} }")
_,     _,     fa_f_trd, beta_f_trd = compute_ratings("fen","FF_pct","ZA_FF_trad","M_for_fen","M_ag_fen")

# Attach to ppdf
def attach(df, col, rating_map):
    df[col] = df.apply(lambda r: rating_map.get((int(r["player_id"]), r["season"]), np.nan), axis=1)
    return df

ppdf = attach(ppdf, "IOC_NFI", ioc_n);  ppdf = attach(ppdf, "IOL_NFI", iol_n)
ppdf = attach(ppdf, "FA_NFI_emp", fa_n_emp)
ppdf = attach(ppdf, "IOC_CF", ioc_c);   ppdf = attach(ppdf, "IOL_CF", iol_c)
ppdf = attach(ppdf, "FA_CF_emp", fa_c_emp);  ppdf = attach(ppdf, "FA_CF_trad", fa_c_trd)
ppdf = attach(ppdf, "IOC_FF", ioc_f);   ppdf = attach(ppdf, "IOL_FF", iol_f)
ppdf = attach(ppdf, "FA_FF_emp", fa_f_emp);  ppdf = attach(ppdf, "FA_FF_trad", fa_f_trd)

# ---------------------------------------------------------------------
# Horse race
# ---------------------------------------------------------------------
print("[5] horse race + player lists ...")
tm = pd.read_csv(TEAM_METRICS)
tm["season"] = tm["season"].astype(str)
tm = tm[tm["season"].isin(POOLED)][["season","team","points"]]

# xG%
print("    fetching MoneyPuck xG ...")
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
if len(xg): xg["team"] = xg["team"].map(norm)

def team_agg(df, col):
    d = df.dropna(subset=[col,"team"]).copy()
    d["w"] = d["toi_sec"]
    out = d.groupby(["season","team"]).apply(
        lambda x: np.average(x[col], weights=x["w"])).rename(col).reset_index()
    return out

def r2(x, y):
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 5: return (np.nan, np.nan, 0)
    r = np.corrcoef(x[mask], y[mask])[0,1]
    return r*r, r, int(mask.sum())

variants = [
    ("NFI% raw","NFI_pct"), ("NFI% ZA empirical","ZA_NFI_emp"), ("NFI% FA empirical","FA_NFI_emp"),
    ("CF% raw","CF_pct"), ("CF% ZA empirical","ZA_CF_emp"), ("CF% ZA traditional","ZA_CF_trad"),
    ("CF% FA empirical","FA_CF_emp"), ("CF% FA traditional","FA_CF_trad"),
    ("FF% raw","FF_pct"), ("FF% ZA empirical","ZA_FF_emp"), ("FF% ZA traditional","ZA_FF_trad"),
    ("FF% FA empirical","FA_FF_emp"), ("FF% FA traditional","FA_FF_trad"),
]
horse_rows = []
for lab, col in variants:
    tb = team_agg(ppdf, col).merge(tm, on=["season","team"], how="inner")
    rp = r2(tb[col].values.astype(float), tb["points"].values.astype(float))
    tc = tb[tb["season"]==CURRENT]
    rc = r2(tc[col].values.astype(float), tc["points"].values.astype(float))
    horse_rows.append({"metric":lab,"R2_pooled":rp[0],"r_pooled":rp[1],"N_pooled":rp[2],
                       "R2_2526":rc[0],"r_2526":rc[1],"N_2526":rc[2]})
if len(xg):
    xgm = xg.merge(tm, on=["season","team"], how="inner")
    rp = r2(xgm["xGoalsPercentage"].values.astype(float), xgm["points"].values.astype(float))
    xgc = xgm[xgm["season"]==CURRENT]
    rc = r2(xgc["xGoalsPercentage"].values.astype(float), xgc["points"].values.astype(float))
    horse_rows.append({"metric":"xG% (MP team 5v5)","R2_pooled":rp[0],"r_pooled":rp[1],
                       "N_pooled":rp[2],"R2_2526":rc[0],"r_2526":rc[1],"N_2526":rc[2]})

horse = pd.DataFrame(horse_rows).sort_values("R2_pooled", ascending=False).reset_index(drop=True)
horse.to_csv(OUT/"horse_race_fully_adjusted.csv", index=False)
print("\n" + "="*94)
print("HORSE RACE (corrected FA)")
print("="*94)
print(horse.to_string(index=False))

# ---------------------------------------------------------------------
# Player lists
# ---------------------------------------------------------------------
for metric in ["NFI","CF","FF"]:
    grp = ppdf.groupby(["player_id","name","pos"]).agg(
        n_seasons=("season","count"),
        toi_total=("toi_sec", lambda s: s.sum()/60),
        team_recent=("team","last"),
        raw_mean=({"NFI":"NFI_pct","CF":"CF_pct","FF":"FF_pct"}[metric],"mean"),
        ZA_mean=({"NFI":"ZA_NFI_emp","CF":"ZA_CF_emp","FF":"ZA_FF_emp"}[metric],"mean"),
        IOC_mean=({"NFI":"IOC_NFI","CF":"IOC_CF","FF":"IOC_FF"}[metric],"mean"),
        IOL_mean=({"NFI":"IOL_NFI","CF":"IOL_CF","FF":"IOL_FF"}[metric],"mean"),
        FA_mean= ({"NFI":"FA_NFI_emp","CF":"FA_CF_emp","FF":"FA_FF_emp"}[metric],"mean"),
    ).reset_index()
    grp = grp[grp["toi_total"]>=500].sort_values("FA_mean", ascending=False).reset_index(drop=True)
    grp["small_sample"] = grp["toi_total"] < 1500
    grp.to_csv(OUT/f"top30_FA_{metric}_pooled.csv", index=False)

# Current season top 30
curr = ppdf[ppdf["season"]==CURRENT].copy()
curr["toi_min"] = curr["toi_sec"]/60
for metric in ["NFI","CF","FF"]:
    curr.sort_values(f"FA_{metric}_emp", ascending=False).assign(
        small_sample=curr["toi_min"]<500
    ).to_csv(OUT/f"top30_FA_{metric}_2526.csv", index=False)

# Cross-metric consistency
tN = set(pd.read_csv(OUT/"top30_FA_NFI_pooled.csv").head(30)["player_id"].astype(int))
tC = set(pd.read_csv(OUT/"top30_FA_CF_pooled.csv").head(30)["player_id"].astype(int))
tF = set(pd.read_csv(OUT/"top30_FA_FF_pooled.csv").head(30)["player_id"].astype(int))
elite = tN & tC & tF
pd.DataFrame([{"player_id":p,"name":name_map.get(p,""),"pos":pos_map.get(p,"")}
              for p in elite]).to_csv(OUT/"cross_metric_elite.csv", index=False)
print(f"    cross-metric elite (in top 30 of all 3 FA metrics): {len(elite)}")

# ---------------------------------------------------------------------
# Streamlit CSVs
# ---------------------------------------------------------------------
player_cols = ["player_id","name","pos","team","season","toi_sec",
               "NFI_pct","ZA_NFI_emp","IOC_NFI","IOL_NFI","FA_NFI_emp",
               "CF_pct","ZA_CF_emp","ZA_CF_trad","IOC_CF","IOL_CF","FA_CF_emp","FA_CF_trad",
               "FF_pct","ZA_FF_emp","ZA_FF_trad","IOC_FF","IOL_FF","FA_FF_emp","FA_FF_trad"]
player_out = ppdf[[c for c in player_cols if c in ppdf.columns]].copy()
player_out["es_toi_min"] = player_out["toi_sec"]/60
player_out = player_out.rename(columns={"name":"player_name","pos":"position"})
player_out.to_csv(OUT/"player_fully_adjusted.csv", index=False)
player_out[player_out["season"]==CURRENT].to_csv(
    OUT/"current_season_player_fully_adjusted.csv", index=False)

# Team aggregated
team_rows = []
for season in sorted(POOLED):
    for team in sorted(ppdf[ppdf["season"]==season]["team"].dropna().unique()):
        sub = ppdf[(ppdf["season"]==season) & (ppdf["team"]==team)]
        if len(sub)==0: continue
        row = {"season":season, "team":team, "n_players":len(sub),
               "team_toi_min":sub["toi_sec"].sum()/60}
        for col in ["NFI_pct","ZA_NFI_emp","FA_NFI_emp",
                    "CF_pct","ZA_CF_emp","ZA_CF_trad","FA_CF_emp","FA_CF_trad",
                    "FF_pct","ZA_FF_emp","ZA_FF_trad","FA_FF_emp","FA_FF_trad"]:
            vals = sub[[col,"toi_sec"]].dropna()
            row[col] = float(np.average(vals[col], weights=vals["toi_sec"])) if len(vals) else np.nan
        team_rows.append(row)
team_out = pd.DataFrame(team_rows).merge(tm, on=["season","team"], how="left")
team_out.to_csv(OUT/"team_fully_adjusted.csv", index=False)
team_out[team_out["season"]==CURRENT].to_csv(
    OUT/"current_season_team_fully_adjusted.csv", index=False)

print(f"\n[done] outputs in {OUT}")
