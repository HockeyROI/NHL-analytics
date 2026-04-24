#!/usr/bin/env python3
"""Top 50 for V3 (two-way forward) and V5 (on-ice CNFI+MNFI% all skaters).
Reuses the same logic as script 37 but outputs top 50 rather than 30."""
import os, math, time
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
TOP30_DIR = f"{OUT}/top30_variations"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
SHIFT_CSV = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
GAME_CSV = f"{ROOT}/Data/game_ids.csv"

POOLED_SEASONS = {"20212022","20222023","20232024","20242025","20252026"}
CURRENT_SEASON = "20252026"
MIN_POOLED_TOI = 500.0
MIN_CURRENT_TOI = 200.0
AGE_MAX = 40
AGE_OLD = 25
AGE_OLD_TOI_GATE = 1000.0

def wilson(k, n, z=1.96):
    if n == 0: return (0.0,0.0,0.0)
    p = k/n; denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return p, max(0.0,c-h), min(1.0,c+h)

# Lookups
pos_df = pd.read_csv(f"{OUT}/player_positions.csv", dtype={"player_id":int})
pos_grp = dict(zip(pos_df["player_id"], pos_df["pos_group"]))
pos_role = dict(zip(pos_df["player_id"], pos_df["position"]))
name_map = dict(zip(pos_df["player_id"], pos_df["player_name"]))
ages = pd.read_csv(f"{OUT}/player_ages.csv", dtype={"player_id":int})
age_map = dict(zip(ages["player_id"], ages["age"]))
toi_pool = pd.read_csv(f"{OUT}/player_toi.csv", dtype={"player_id":int})
toi_pool_es_min = dict(zip(toi_pool["player_id"], toi_pool["toi_ES_sec"]/60.0))
games = pd.read_csv(GAME_CSV, dtype={"game_id":int,"season":str})
games = games[games["game_type"]=="regular"]
game_season_map = dict(zip(games["game_id"], games["season"]))

# ES TOI per (player, season)
print("Building per-(player, season) ES TOI ...")
sit_shots = pd.read_csv(SHOT_CSV,
                         usecols=["game_id","season","period","time_secs",
                                  "situation_code","event_type","shooting_team_id",
                                  "home_team_id"],
                         dtype={"season":str,"situation_code":str})
sit_shots = sit_shots[sit_shots["season"].isin(POOLED_SEASONS)]
sit_shots = sit_shots[sit_shots["period"].between(1,3)]
sc = sit_shots["situation_code"].astype(str).str.zfill(4)
ag = sc.str[0].astype(int); ask = sc.str[1].astype(int)
hsk = sc.str[2].astype(int); hg = sc.str[3].astype(int)
empty = (ag==0) | (hg==0)
sh_h = sit_shots["shooting_team_id"]==sit_shots["home_team_id"]
sh_sk = np.where(sh_h, hsk, ask); opp_sk = np.where(sh_h, ask, hsk)
state = np.where(empty, "EN",
        np.where(sh_sk == opp_sk, "ES",
        np.where(sh_sk > opp_sk, "PP", "PK")))
sit_shots["state"] = state
sit_shots["abs_time"] = sit_shots["time_secs"].astype(int) + (sit_shots["period"].astype(int)-1)*1200
sit_by_game = dict(tuple(sit_shots.sort_values(["game_id","abs_time"]).groupby("game_id")))

print("Loading shifts ...")
shifts_parts = []
for ch in pd.read_csv(SHIFT_CSV,
                      usecols=["game_id","player_id","period","team_abbrev",
                                "abs_start_secs","abs_end_secs"],
                      chunksize=500_000):
    ch = ch.dropna()
    ch["game_id"] = ch["game_id"].astype(int); ch["player_id"] = ch["player_id"].astype(int)
    ch["period"] = ch["period"].astype(int)
    ch["abs_start_secs"] = ch["abs_start_secs"].astype(int)
    ch["abs_end_secs"]   = ch["abs_end_secs"].astype(int)
    ch = ch[ch["period"].between(1,3)]
    ch["season"] = ch["game_id"].map(game_season_map)
    ch = ch[ch["season"].isin(POOLED_SEASONS)]
    if len(ch): shifts_parts.append(ch)
shifts = pd.concat(shifts_parts, ignore_index=True); del shifts_parts
shifts_by_game = dict(tuple(shifts.groupby("game_id")))

ts_es = defaultdict(int)
print("Computing ES TOI per (player, season) ...")
n_games = 0; t0 = time.time()
for gid, gshifts in shifts_by_game.items():
    n_games += 1
    season = game_season_map.get(gid)
    if season is None or season not in POOLED_SEASONS: continue
    sg = sit_by_game.get(gid)
    if sg is None or len(sg)==0:
        intervals = [(0, 3600, "ES")]
    else:
        intervals = []; prev_t = 0; prev_state = "ES"
        for r in sg.itertuples(index=False):
            t = int(r.abs_time)
            if t > prev_t: intervals.append((prev_t, t, prev_state))
            prev_t = t; prev_state = r.state
        if prev_t < 3600: intervals.append((prev_t, 3600, prev_state))
    iv_starts = np.array([iv[0] for iv in intervals])
    iv_ends   = np.array([iv[1] for iv in intervals])
    iv_states = np.array([iv[2] for iv in intervals])
    for r in gshifts.itertuples(index=False):
        s = int(r.abs_start_secs); e = int(r.abs_end_secs); pid = int(r.player_id)
        if e <= s: continue
        lo = np.searchsorted(iv_ends, s, side="right")
        for j in range(lo, len(iv_starts)):
            if iv_starts[j] >= e: break
            if iv_states[j] == "ES":
                ovl = min(e, iv_ends[j]) - max(s, iv_starts[j])
                if ovl > 0: ts_es[(pid, season)] += ovl

toi_curr_min = {pid: ts_es.get((pid, CURRENT_SEASON), 0)/60.0
                for pid in {p for (p, s) in ts_es.keys()}}
print(f"  done in {time.time()-t0:.1f}s")

# Shots
print("Loading shots ...")
sh = pd.read_csv(SHOT_CSV,
                 usecols=["game_id","season","period","situation_code","event_type",
                          "shooting_team_abbrev","shooter_player_id","x_coord_norm",
                          "y_coord_norm","home_team_id","shooting_team_id",
                          "home_team_abbrev","away_team_abbrev","time_secs"],
                 dtype={"season":str,"situation_code":str})
sh = sh[sh["season"].isin(POOLED_SEASONS)]
sh = sh[sh["period"].between(1,3)]
sh = sh[sh["situation_code"].astype(str)=="1551"]
sh = sh[sh["event_type"].isin(["shot-on-goal","missed-shot","goal"])]
sh = sh.dropna(subset=["x_coord_norm","y_coord_norm","shooter_player_id"])
sh["abs_y"] = sh["y_coord_norm"].abs()
sh["shooter_player_id"] = sh["shooter_player_id"].astype(int)
sh["shooter_pos"] = sh["shooter_player_id"].map(pos_grp)
sh["zone"] = np.where(
    (sh["x_coord_norm"].between(74,89)) & (sh["abs_y"]<=9), "CNFI",
    np.where((sh["x_coord_norm"].between(55,73)) & (sh["abs_y"]<=15), "MNFI",
    np.where((sh["x_coord_norm"].between(25,54)) & (sh["abs_y"]<=15), "FNFI", "OTHER")))
sh["abs_time"] = sh["time_secs"].astype(int) + (sh["period"].astype(int)-1)*1200
sh["_shoot_home"] = sh["shooting_team_id"]==sh["home_team_id"]
sh["def_team"] = np.where(sh["_shoot_home"], sh["away_team_abbrev"], sh["home_team_abbrev"])

# Individual shooter pivot (V3 offensive)
fsh = sh[sh["shooter_pos"]=="F"]
ind_counts = fsh.groupby(["shooter_player_id","season","zone"]).size().reset_index(name="n")
ind_pivot = ind_counts.pivot_table(index=["shooter_player_id","season"], columns="zone",
                                     values="n", fill_value=0).reset_index()
for col in ["CNFI","MNFI","FNFI","OTHER"]:
    if col not in ind_pivot.columns: ind_pivot[col] = 0
ind_pivot = ind_pivot.rename(columns={"CNFI":"CNFI_n","MNFI":"MNFI_n",
                                        "FNFI":"FNFI_n","OTHER":"OTHER_n"})

# On-ice for/against counts
print("Shift-shot join (on-ice for/against) ...")
on_for_cmnfi = defaultdict(int); on_for_all = defaultdict(int)
on_ag_cmnfi  = defaultdict(int); on_ag_all  = defaultdict(int)
shots_by_game = dict(tuple(sh.groupby("game_id")))
n_games = 0; t0 = time.time()
for gid, gshots in shots_by_game.items():
    n_games += 1
    season = game_season_map.get(gid)
    if season is None or season not in POOLED_SEASONS: continue
    gshifts = shifts_by_game.get(gid)
    if gshifts is None: continue
    shifts_by_team = {}
    for tab, tsh in gshifts.groupby("team_abbrev"):
        st = tsh["abs_start_secs"].values.astype(np.int32)
        en = tsh["abs_end_secs"].values.astype(np.int32)
        pids = tsh["player_id"].values.astype(np.int64)
        order = np.argsort(st)
        shifts_by_team[tab] = (st[order], en[order], pids[order])
    for r in gshots.itertuples(index=False):
        t = int(r.abs_time); is_cm = r.zone in ("CNFI","MNFI")
        sa, da = r.shooting_team_abbrev, r.def_team
        if sa in shifts_by_team:
            st, en, pids = shifts_by_team[sa]
            idx = np.searchsorted(st, t, side="right")
            if idx > 0:
                on = en[:idx] > t
                for pid in pids[:idx][on]:
                    pid = int(pid)
                    on_for_all[(pid, season)] += 1
                    if is_cm: on_for_cmnfi[(pid, season)] += 1
        if da in shifts_by_team:
            st, en, pids = shifts_by_team[da]
            idx = np.searchsorted(st, t, side="right")
            if idx > 0:
                on = en[:idx] > t
                for pid in pids[:idx][on]:
                    pid = int(pid)
                    on_ag_all[(pid, season)] += 1
                    if is_cm: on_ag_cmnfi[(pid, season)] += 1
print(f"  done in {time.time()-t0:.1f}s")

def es_toi_min(pid, scope):
    return toi_pool_es_min.get(pid, 0.0) if scope=="pooled" else toi_curr_min.get(pid, 0.0)

def passes_age(pid):
    age = age_map.get(pid)
    if age is None: return False
    if age > AGE_MAX: return False
    if age > AGE_OLD and toi_pool_es_min.get(pid, 0.0) < AGE_OLD_TOI_GATE: return False
    return True

def get_pos_detail(pid):
    role = pos_role.get(pid, "")
    return {"C":"C","L":"LW","R":"RW","D":"D","G":"G"}.get(role, role)

def build_v3(scope, top_n=50):
    if scope=="pooled":
        sum_df = ind_pivot.groupby("shooter_player_id")[["CNFI_n","MNFI_n"]].sum().reset_index()
    else:
        sum_df = ind_pivot[ind_pivot["season"]==CURRENT_SEASON]\
                    .groupby("shooter_player_id")[["CNFI_n","MNFI_n"]].sum().reset_index()
    sum_df = sum_df.rename(columns={"shooter_player_id":"player_id"})
    rows = []
    for _, r in sum_df.iterrows():
        pid = int(r["player_id"])
        if pos_grp.get(pid) != "F": continue
        toi = es_toi_min(pid, scope)
        if scope=="pooled" and toi < MIN_POOLED_TOI: continue
        if scope=="current" and toi < MIN_CURRENT_TOI: continue
        if not passes_age(pid): continue
        n_for = int(r["CNFI_n"]) + int(r["MNFI_n"])
        if scope=="pooled":
            n_ag = sum(on_ag_cmnfi.get((pid, s), 0) for s in POOLED_SEASONS)
        else:
            n_ag = on_ag_cmnfi.get((pid, CURRENT_SEASON), 0)
        rows.append({"player_id": pid, "player_name": name_map.get(pid,""),
                      "position": get_pos_detail(pid), "age": age_map.get(pid),
                      "es_toi_min": round(toi,1),
                      "off_per60": round(n_for/toi*60.0, 4),
                      "def_per60": round(n_ag/toi*60.0, 4)})
    df = pd.DataFrame(rows)
    df["off_z"] = (df["off_per60"]-df["off_per60"].mean())/df["off_per60"].std(ddof=0)
    df["def_z"] = (df["def_per60"]-df["def_per60"].mean())/df["def_per60"].std(ddof=0)
    df["twoway_score"] = df["off_z"] - df["def_z"]
    df = df.sort_values("twoway_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df)+1)
    df["small_sample_flag"] = np.where(df["es_toi_min"] < 2000, "Y", "N")
    cols = ["rank","player_name","position","age","es_toi_min",
            "off_per60","off_z","def_per60","def_z","twoway_score",
            "small_sample_flag"]
    return df[cols].head(top_n)

def build_v5(scope, top_n=50):
    seasons_use = POOLED_SEASONS if scope=="pooled" else {CURRENT_SEASON}
    rows = []
    pids = {pid for (pid, s) in on_for_all.keys() if s in seasons_use}
    for pid in pids:
        if pos_grp.get(pid) not in ("F","D"): continue
        toi = es_toi_min(pid, scope)
        if scope=="pooled" and toi < MIN_POOLED_TOI: continue
        if scope=="current" and toi < MIN_CURRENT_TOI: continue
        if not passes_age(pid): continue
        n_for_cm  = sum(on_for_cmnfi.get((pid, s), 0) for s in seasons_use)
        n_for_all = sum(on_for_all.get((pid, s), 0)   for s in seasons_use)
        n_ag_all  = sum(on_ag_all.get((pid, s), 0)    for s in seasons_use)
        denom = n_for_all + n_ag_all
        if denom == 0: continue
        p, lo, hi = wilson(n_for_cm, denom)
        rows.append({"player_id": pid, "player_name": name_map.get(pid,""),
                      "position": get_pos_detail(pid),
                      "pos_group": pos_grp.get(pid),
                      "age": age_map.get(pid),
                      "es_toi_min": round(toi,1),
                      "onice_CMNFI_for": n_for_cm,
                      "onice_total_fenwick": denom,
                      "pct": round(p,5), "ci_low": round(lo,5), "ci_high": round(hi,5)})
    df = pd.DataFrame(rows).sort_values("pct", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df)+1)
    df["small_sample_flag"] = np.where(df["es_toi_min"] < 2000, "Y", "N")
    cols = ["rank","player_name","position","pos_group","age","es_toi_min",
            "onice_CMNFI_for","onice_total_fenwick","pct","ci_low","ci_high",
            "small_sample_flag"]
    return df[cols].head(top_n)

v3p = build_v3("pooled");  v3c = build_v3("current")
v5p = build_v5("pooled");  v5c = build_v5("current")

# Save top-50 versions alongside top-30
v3p.to_csv(f"{TOP30_DIR}/variation_3_pooled_top50.csv", index=False)
v3c.to_csv(f"{TOP30_DIR}/variation_3_current_top50.csv", index=False)
v5p.to_csv(f"{TOP30_DIR}/variation_5_pooled_top50.csv", index=False)
v5c.to_csv(f"{TOP30_DIR}/variation_5_current_top50.csv", index=False)

pd.set_option("display.max_rows", 60); pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.4f}" if not pd.isna(x) else "")

print("\n========== V3 POOLED — Top 50 two-way forward ==========")
print(v3p.to_string(index=False))
print("\n========== V3 CURRENT — Top 50 two-way forward ==========")
print(v3c.to_string(index=False))
print("\n========== V5 POOLED — Top 50 on-ice CNFI+MNFI% (F + D) ==========")
print(v5p.to_string(index=False))
print("\n========== V5 CURRENT — Top 50 on-ice CNFI+MNFI% (F + D) ==========")
print(v5c.to_string(index=False))
