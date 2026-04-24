#!/usr/bin/env python3
"""
Step 9 - Spatial QoC & QoT for NFI.
Efficient version using dense numpy matrices indexed by player idx.
"""
import os, math
from collections import defaultdict
import pandas as pd
import numpy as np

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
SHIFT = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
OUT = f"{ROOT}/NFI/output"
SEASONS = {"20212022","20222023","20232024","20242025","20252026"}

# game season map
games = pd.read_csv(f"{ROOT}/Data/game_ids.csv", dtype={"season":str})
games = games[games["game_type"]=="regular"]
valid_gids = set(int(g) for g,s in zip(games["game_id"], games["season"]) if s in SEASONS)
print(f"Valid games: {len(valid_gids):,}")

# positions
pos = pd.read_csv(f"{OUT}/player_positions.csv")
pos_map = dict(zip(pos["player_id"], pos["pos_group"]))

# TOI qualification
toi = pd.read_csv(f"{OUT}/player_toi.csv")
qual = set(toi[toi["toi_total_min"]>=500]["player_id"].tolist())
print(f"Qualifying players: {len(qual):,}")

# player rates
counts = pd.read_csv(f"{OUT}/player_counts_by_state_zone.csv")
counts = counts[counts["state"]=="ES"]
ind = counts.groupby(["player_id","zone"])["ind_att"].sum().unstack(fill_value=0).reset_index()
ind = ind.merge(toi[["player_id","toi_ES_sec"]], on="player_id", how="left")
ind["toi_ES_min"] = ind["toi_ES_sec"]/60.0
for z in ["CNFI","MNFI","FNFI"]:
    if z not in ind.columns: ind[z] = 0
ind["ind_TNFI_per60"] = (ind["CNFI"]+ind["MNFI"]+ind["FNFI"]) / ind["toi_ES_min"].replace(0,np.nan) * 60.0
ind["ind_CNFI_per60"] = ind["CNFI"] / ind["toi_ES_min"].replace(0,np.nan) * 60.0
oi = counts.groupby("player_id").agg(of=("onice_for_att","sum"), ag=("onice_ag_att","sum")).reset_index()
oi["CF_pct"] = oi["of"]/(oi["of"]+oi["ag"])

# Build universe of player_ids
all_pids = sorted(set(toi["player_id"].tolist()) | set(pos["player_id"].tolist()))
pid_to_idx = {p:i for i,p in enumerate(all_pids)}
N = len(all_pids)
print(f"Player universe: {N}")

rate_tnfi = np.zeros(N); rate_cnfi = np.zeros(N); rate_cfp = np.full(N, 0.5)
for _,r in ind.iterrows():
    i = pid_to_idx.get(r["player_id"])
    if i is None: continue
    rate_tnfi[i] = 0 if pd.isna(r["ind_TNFI_per60"]) else r["ind_TNFI_per60"]
    rate_cnfi[i] = 0 if pd.isna(r["ind_CNFI_per60"]) else r["ind_CNFI_per60"]
for _,r in oi.iterrows():
    i = pid_to_idx.get(r["player_id"])
    if i is None: continue
    rate_cfp[i] = r["CF_pct"] if not pd.isna(r["CF_pct"]) else 0.5

# Accumulator arrays: seconds with / vs each player per focal
# Rather than NxN, use dict per focal player (too big). Use sparse: accumulate per game into dicts.
# Instead, build co-occurrence via dict-of-dict, only for qualifying players.
qual_idx = set(pid_to_idx[p] for p in qual if p in pid_to_idx)

# stream shifts
print("Streaming shifts...")
shift_cols = ["game_id","player_id","period","team_abbrev","abs_start_secs","abs_end_secs"]
shifts_list = []
for ch in pd.read_csv(SHIFT, usecols=shift_cols, chunksize=500000):
    ch = ch.dropna(subset=["game_id","player_id","period","abs_start_secs","abs_end_secs"])
    ch["game_id"] = ch["game_id"].astype(int)
    ch["player_id"] = ch["player_id"].astype(int)
    ch = ch[ch["game_id"].isin(valid_gids) & ch["period"].between(1,3)]
    if len(ch): shifts_list.append(ch)
shifts = pd.concat(shifts_list, ignore_index=True)
del shifts_list
print(f"  shifts: {len(shifts):,}")

shifts_by_game = dict(tuple(shifts.groupby("game_id")))

# Accumulate seconds: teammate-wise aggregated rate
# For each focal player, we want: sum over (teammate rate * sec_with_teammate) / sum(sec)
# Instead of storing per-pair, directly accumulate per-focal numerator and denominator.

num_tnfi_tm = np.zeros(N); num_cnfi_tm = np.zeros(N); num_cf_tm = np.zeros(N); den_tm = np.zeros(N)
num_tnfi_op = np.zeros(N); num_cnfi_op = np.zeros(N); num_cf_op = np.zeros(N); den_op = np.zeros(N)

T = 3600
print("Processing games...")
ng = 0
for gid, gsh in shifts_by_game.items():
    ng += 1
    if ng % 500 == 0:
        print(f"  {ng}/{len(shifts_by_game)}")
    # per team occupancy
    team_data = {}
    for team_ab, tsh in gsh.groupby("team_abbrev"):
        pids = tsh["player_id"].values.astype(int)
        starts = np.clip(tsh["abs_start_secs"].values.astype(int), 0, T)
        ends = np.clip(tsh["abs_end_secs"].values.astype(int), 0, T)
        uniq = sorted(set(pids))
        idxs = [pid_to_idx[p] for p in uniq if p in pid_to_idx]
        pid_to_local = {p:i for i,p in enumerate(uniq)}
        occ = np.zeros((len(uniq), T), dtype=np.uint8)
        for p, s, e in zip(pids, starts, ends):
            if e > s: occ[pid_to_local[p], s:e] = 1
        idx_arr = np.array([pid_to_idx.get(p,-1) for p in uniq])
        team_data[team_ab] = (uniq, idx_arr, occ)

    # Same-team co-occurrence
    for team_ab, (uniq, idx_arr, occ) in team_data.items():
        if occ.shape[0] == 0: continue
        M = occ.astype(np.int32) @ occ.astype(np.int32).T  # (P,P) seconds together
        # subtract diagonal for "with others"
        np.fill_diagonal(M, 0)
        # teammate rates for each col
        r_tnfi = rate_tnfi[idx_arr]
        r_cnfi = rate_cnfi[idx_arr]
        r_cf   = rate_cfp[idx_arr]
        # per-focal: sum_j M[i,j] * r_tnfi[j]
        np.add.at(num_tnfi_tm, idx_arr, M @ r_tnfi)
        np.add.at(num_cnfi_tm, idx_arr, M @ r_cnfi)
        np.add.at(num_cf_tm,   idx_arr, M @ r_cf)
        np.add.at(den_tm,      idx_arr, M.sum(axis=1))

    # Cross-team
    teams = list(team_data.keys())
    if len(teams) == 2:
        u1, ix1, o1 = team_data[teams[0]]
        u2, ix2, o2 = team_data[teams[1]]
        X = o1.astype(np.int32) @ o2.astype(np.int32).T
        r1_tnfi = rate_tnfi[ix1]; r1_cnfi = rate_cnfi[ix1]; r1_cf = rate_cfp[ix1]
        r2_tnfi = rate_tnfi[ix2]; r2_cnfi = rate_cnfi[ix2]; r2_cf = rate_cfp[ix2]
        # for team1 focals: weight by team2 rates
        np.add.at(num_tnfi_op, ix1, X @ r2_tnfi)
        np.add.at(num_cnfi_op, ix1, X @ r2_cnfi)
        np.add.at(num_cf_op,   ix1, X @ r2_cf)
        np.add.at(den_op,      ix1, X.sum(axis=1))
        # for team2 focals: weight by team1 rates
        np.add.at(num_tnfi_op, ix2, X.T @ r1_tnfi)
        np.add.at(num_cnfi_op, ix2, X.T @ r1_cnfi)
        np.add.at(num_cf_op,   ix2, X.T @ r1_cf)
        np.add.at(den_op,      ix2, X.T.sum(axis=1))

print("Building output...")
rows = []
for pid, i in pid_to_idx.items():
    if pid not in qual: continue
    rows.append({
        "player_id": pid,
        "position": pos_map.get(pid,""),
        "tot_tm_sec": den_tm[i],
        "tot_op_sec": den_op[i],
        "QoT_spatial_TNFI": (num_tnfi_tm[i]/den_tm[i]) if den_tm[i]>0 else 0,
        "QoT_spatial_CNFI": (num_cnfi_tm[i]/den_tm[i]) if den_tm[i]>0 else 0,
        "QoT_corsi_CF":     (num_cf_tm[i]/den_tm[i])   if den_tm[i]>0 else 0,
        "QoC_spatial_TNFI": (num_tnfi_op[i]/den_op[i]) if den_op[i]>0 else 0,
        "QoC_spatial_CNFI": (num_cnfi_op[i]/den_op[i]) if den_op[i]>0 else 0,
        "QoC_corsi_CF":     (num_cf_op[i]/den_op[i])   if den_op[i]>0 else 0,
    })

qo = pd.DataFrame(rows)
# Merge TNFI%/CF% for adjusted metrics
pmet = pd.read_csv(f"{OUT}/metrics_player.csv")
qo = qo.merge(pmet[["player_id","TNFI_pct","CF_pct","HD_CF_pct","CNFI_pct","MNFI_pct","FNFI_pct"]],
              on="player_id", how="left")

# QoC/QoT adjustments (league-average difference, additive scaling)
# For TNFI%: subtract (QoT - league_QoT) contribution of teammates' generation rates,
# add (QoC - league_QoC) to credit harder opponents.
lQoT = qo["QoT_spatial_TNFI"].mean(); lQoC = qo["QoC_spatial_TNFI"].mean()
# scale factor: convert per-60 TNFI delta to % points. Use ratio of std(TNFI_pct)/std(QoT) for calibration.
if qo["QoT_spatial_TNFI"].std() > 0:
    k = qo["TNFI_pct"].std() / qo["QoT_spatial_TNFI"].std()
else:
    k = 0.01
qo["TNFI_QoTAdj_pct"]      = qo["TNFI_pct"] - (qo["QoT_spatial_TNFI"]-lQoT) * k
qo["TNFI_QoCAdj_pct"]      = qo["TNFI_pct"] + (qo["QoC_spatial_TNFI"]-lQoC) * k
qo["TNFI_QoCQoTAdj_pct"]   = qo["TNFI_pct"] + (qo["QoC_spatial_TNFI"]-lQoC)*k - (qo["QoT_spatial_TNFI"]-lQoT)*k

# also for CNFI/MNFI/FNFI
for zn, col_q, col_c in [("CNFI","QoT_spatial_CNFI","QoC_spatial_CNFI")]:
    lqt = qo[col_q].mean(); lqc = qo[col_c].mean()
    if qo[col_q].std() > 0:
        kk = qo["CNFI_pct"].std() / qo[col_q].std()
    else:
        kk = 0.01
    qo[f"{zn}_QoTAdj_pct"] = qo["CNFI_pct"] - (qo[col_q]-lqt)*kk
    qo[f"{zn}_QoCAdj_pct"] = qo["CNFI_pct"] + (qo[col_c]-lqc)*kk
    qo[f"{zn}_QoCQoTAdj_pct"] = qo["CNFI_pct"] + (qo[col_c]-lqc)*kk - (qo[col_q]-lqt)*kk

qo.to_csv(f"{OUT}/player_qoc_qot.csv", index=False)
print(f"Wrote player_qoc_qot.csv ({len(qo)} rows)")
