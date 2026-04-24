#!/usr/bin/env python3
"""
Top 50 individual on-ice Corsi-For % (CF%) for current season 2025-26.
ES regulation only, full Corsi (SOG + missed + blocked + goal).
Min 200 ES TOI minutes for current season; age filter applied.
"""
import math, time
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
SHIFT_CSV = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
GAME_CSV = f"{ROOT}/Data/game_ids.csv"

CURRENT_SEASON = "20252026"
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

# Load shots (current season, ES, full Corsi)
print("Loading shots ...")
sh = pd.read_csv(SHOT_CSV,
                 usecols=["game_id","season","period","situation_code","event_type",
                          "shooting_team_abbrev","time_secs",
                          "home_team_id","shooting_team_id",
                          "home_team_abbrev","away_team_abbrev"],
                 dtype={"season":str,"situation_code":str})
sh = sh[sh["season"]==CURRENT_SEASON]
sh = sh[sh["period"].between(1,3)]
sh = sh[sh["situation_code"].astype(str)=="1551"]
sh = sh[sh["event_type"].isin(["shot-on-goal","missed-shot","blocked-shot","goal"])]
sh["abs_time"] = sh["time_secs"].astype(int) + (sh["period"].astype(int)-1)*1200
sh["_shoot_home"] = sh["shooting_team_id"]==sh["home_team_id"]
sh["def_team"] = np.where(sh["_shoot_home"], sh["away_team_abbrev"], sh["home_team_abbrev"])
print(f"  ES Corsi shots in 2025-26: {len(sh):,}")

# Build situation intervals per game (for ES TOI computation)
sit_shots = sh.copy()
sit_shots["state"] = "ES"  # already ES-filtered
# We also need PP/PK/EN events to define interval boundaries. Reload all-state shots quickly.
print("Loading all-state shots for interval boundaries ...")
sit_all = pd.read_csv(SHOT_CSV,
                       usecols=["game_id","season","period","situation_code",
                                "time_secs","shooting_team_id","home_team_id"],
                       dtype={"season":str,"situation_code":str})
sit_all = sit_all[sit_all["season"]==CURRENT_SEASON]
sit_all = sit_all[sit_all["period"].between(1,3)]
sc = sit_all["situation_code"].astype(str).str.zfill(4)
ag_d = sc.str[0].astype(int); ask_d = sc.str[1].astype(int)
hsk_d = sc.str[2].astype(int); hg_d = sc.str[3].astype(int)
empty = (ag_d==0) | (hg_d==0)
sh_h = sit_all["shooting_team_id"]==sit_all["home_team_id"]
sh_sk = np.where(sh_h, hsk_d, ask_d); opp_sk = np.where(sh_h, ask_d, hsk_d)
state = np.where(empty, "EN",
        np.where(sh_sk == opp_sk, "ES",
        np.where(sh_sk > opp_sk, "PP", "PK")))
sit_all["state"] = state
sit_all["abs_time"] = sit_all["time_secs"].astype(int) + (sit_all["period"].astype(int)-1)*1200
sit_by_game = dict(tuple(sit_all.sort_values(["game_id","abs_time"]).groupby("game_id")))

# Load shifts (current season)
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
    ch = ch[ch["season"]==CURRENT_SEASON]
    if len(ch): shifts_parts.append(ch)
shifts = pd.concat(shifts_parts, ignore_index=True); del shifts_parts
print(f"  shifts: {len(shifts):,}")
shifts_by_game = dict(tuple(shifts.groupby("game_id")))

# Per-(player) current-season ES TOI via state-interval intersection
print("Computing current-season ES TOI per player ...")
ts_es = defaultdict(int)
n_games = 0; t0 = time.time()
for gid, gshifts in shifts_by_game.items():
    n_games += 1
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
                if ovl > 0: ts_es[pid] += ovl
print(f"  done in {time.time()-t0:.1f}s")
toi_curr_min = {pid: sec/60.0 for pid, sec in ts_es.items()}

# Per-game shift-shot join (Corsi, on-ice for/against)
print("Shift-shot join (Corsi on-ice for/against) ...")
on_cf = defaultdict(int); on_ca = defaultdict(int)
shots_by_game = dict(tuple(sh.groupby("game_id")))
n_games = 0; t0 = time.time()
for gid, gshots in shots_by_game.items():
    n_games += 1
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
        t = int(r.abs_time); sa, da = r.shooting_team_abbrev, r.def_team
        if sa in shifts_by_team:
            st, en, pids = shifts_by_team[sa]
            idx = np.searchsorted(st, t, side="right")
            if idx > 0:
                on = en[:idx] > t
                for pid in pids[:idx][on]:
                    on_cf[int(pid)] += 1
        if da in shifts_by_team:
            st, en, pids = shifts_by_team[da]
            idx = np.searchsorted(st, t, side="right")
            if idx > 0:
                on = en[:idx] > t
                for pid in pids[:idx][on]:
                    on_ca[int(pid)] += 1
print(f"  done in {time.time()-t0:.1f}s")

def passes_age(pid):
    age = age_map.get(pid)
    if age is None: return False
    if age > AGE_MAX: return False
    if age > AGE_OLD and toi_pool_es_min.get(pid, 0.0) < AGE_OLD_TOI_GATE: return False
    return True

def get_pos_detail(pid):
    role = pos_role.get(pid, "")
    return {"C":"C","L":"LW","R":"RW","D":"D","G":"G"}.get(role, role)

# Build CF% list
rows = []
for pid in set(on_cf.keys()) | set(on_ca.keys()):
    if pos_grp.get(pid) not in ("F","D"): continue
    toi = toi_curr_min.get(pid, 0.0)
    if toi < MIN_CURRENT_TOI: continue
    if not passes_age(pid): continue
    cf = on_cf.get(pid, 0); ca = on_ca.get(pid, 0)
    denom = cf + ca
    if denom == 0: continue
    p, lo, hi = wilson(cf, denom)
    rows.append({"player_id": pid, "player_name": name_map.get(pid,""),
                  "position": get_pos_detail(pid),
                  "pos_group": pos_grp.get(pid),
                  "age": age_map.get(pid),
                  "es_toi_min": round(toi,1),
                  "CF": cf, "CA": ca, "total": denom,
                  "CF_pct": round(p,5),
                  "ci_low": round(lo,5),
                  "ci_high": round(hi,5)})
df = pd.DataFrame(rows).sort_values("CF_pct", ascending=False).reset_index(drop=True)
df["rank"] = np.arange(1, len(df)+1)

cols = ["rank","player_name","position","pos_group","age","es_toi_min",
        "CF","CA","total","CF_pct","ci_low","ci_high"]
df = df[cols]

OUT_FILE = f"{OUT}/top30_variations/corsi_cf_pct_current_top50.csv"
df.head(50).to_csv(OUT_FILE, index=False)
print(f"\nWrote {OUT_FILE}")

pd.set_option("display.max_rows", 60); pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.4f}" if not pd.isna(x) else "")

print(f"\n========== TOP 50 by individual on-ice CF% — 2025-26 ES regulation ==========")
print(f"Qualifying skaters (≥200 current ES min, age filter applied): {len(df)}")
print()
print(df.head(50).to_string(index=False))
