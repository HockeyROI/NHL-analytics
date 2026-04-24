#!/usr/bin/env python3
"""
Rerun all player-level spatial metrics with:
  (a) blocked-shot coordinate sign-flip CORRECTED:
        for event_type='blocked-shot', x_coord_norm = abs(x_coord_norm),
                                       y_coord_norm = abs(y_coord_norm)
  (b) Full Corsi event filter (SOG + missed + blocked + goal)

Overwrites:
  P1a_centrality_weighted.csv      (forwards on-ice for; wide CNFI/MNFI x-bands)
  P2_defensive_forwards.csv         (forwards on-ice against; tight zones)
  P4_defensive_D.csv                (D on-ice against; tight zones)
  P5_offensive_D.csv                (D on-ice for; tight zones)
  twoway_forward_score.csv          (z-scored two-way F)
  twoway_D_score.csv                (z-scored two-way D)
  zone_conversion_rates.csv         (Corsi columns added)

Goalie metrics NOT touched (blocked shots never reach the goalie).
"""
import os, math
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
SHOT_CSV  = f"{ROOT}/Data/nhl_shot_events.csv"
SHIFT_CSV = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
TOI_CSV   = f"{OUT}/player_toi.csv"
POS_CSV   = f"{OUT}/player_positions.csv"

SEASONS = {"20212022","20222023","20232024","20242025","20252026"}
MIN_F_TOI = 500.0
MIN_D_TOI = 500.0

# ---- Y-band weights (same as prior pipeline) ----
WEIGHTS_WIDE = {  # P1a uses wide CNFI/MNFI (no y restriction); 7 bands
    "CNFI": {"0-5":0.1579, "5-10":0.1075, "10-15":0.0781, "15-20":0.0513,
             "20-25":0.0318, "25-30":0.0209, "30+":0.0123},
    "MNFI": {"0-5":0.1109, "5-10":0.0985, "10-15":0.0887, "15-20":0.0692,
             "20-25":0.0492, "25-30":0.0278, "30+":0.0122},
}
WEIGHTS_TIGHT = {  # P2/P4/P5 use tight zones, 3 bands
    "CNFI": {"0-5":0.1579, "5-10":0.1075},
    "MNFI": {"0-5":0.1109, "5-10":0.0985, "10-15":0.0887},
    "FNFI": {"0-5":0.0367, "5-10":0.0335, "10-15":0.0289},
}
TIGHT_BANDS = ["0-5","5-10","10-15"]
WIDE_BANDS  = ["0-5","5-10","10-15","15-20","20-25","25-30","30+"]

def y_band_wide(absy):
    if absy < 5: return "0-5"
    if absy < 10: return "5-10"
    if absy < 15: return "10-15"
    if absy < 20: return "15-20"
    if absy < 25: return "20-25"
    if absy < 30: return "25-30"
    return "30+"

def y_band_tight(absy):
    if absy < 5: return "0-5"
    if absy < 10: return "5-10"
    return "10-15"

def classify_tight(x, absy):
    if 74 <= x <= 89 and absy <= 9:   return "CNFI"
    if 55 <= x <= 73 and absy <= 15:  return "MNFI"
    if 25 <= x <= 54 and absy <= 15:  return "FNFI"
    return None

def classify_wide(x):
    if 74 <= x <= 89: return "CNFI"
    if 55 <= x <= 73: return "MNFI"
    return None

def wilson(k, n, z=1.96):
    if n == 0: return (0.0,0.0,0.0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0.0,c-h), min(1.0,c+h))

def rate_ci(events, minutes, z=1.96):
    if minutes <= 0: return 0.0, 0.0, 0.0
    p, lo, hi = wilson(events, max(events, int(round(minutes))))
    return events/minutes*60.0, lo*60.0, hi*60.0

# ---- Load lookups ----
print("Loading positions / TOI ...")
pos_df = pd.read_csv(POS_CSV, dtype={"player_id":int})
pos_grp = dict(zip(pos_df["player_id"], pos_df["pos_group"]))
name_map = dict(zip(pos_df["player_id"], pos_df["player_name"]))
toi_df = pd.read_csv(TOI_CSV, dtype={"player_id":int})
toi_es_min = dict(zip(toi_df["player_id"], toi_df["toi_ES_sec"]/60.0))

# ---- Load shots, apply correction, full Corsi ----
print("Loading shots, applying blocked-shot abs() correction ...")
shot_cols = ["game_id","season","period","event_type","situation_code","time_secs",
             "home_team_id","shooting_team_id","shooting_team_abbrev",
             "home_team_abbrev","away_team_abbrev","x_coord_norm","y_coord_norm","is_goal"]
shots = pd.read_csv(SHOT_CSV, usecols=shot_cols,
                    dtype={"season":str,"situation_code":str})
shots = shots[shots["season"].isin(SEASONS)].copy()
shots = shots[shots["period"].between(1,3)].copy()
shots = shots[shots["event_type"].isin(["shot-on-goal","missed-shot","blocked-shot","goal"])].copy()
shots = shots[shots["situation_code"].astype(str)=="1551"].copy()

# THE FIX: abs() for blocked shots only
blk_mask = shots["event_type"]=="blocked-shot"
n_blk_pre = int(blk_mask.sum())
shots.loc[blk_mask, "x_coord_norm"] = shots.loc[blk_mask, "x_coord_norm"].abs()
shots.loc[blk_mask, "y_coord_norm"] = shots.loc[blk_mask, "y_coord_norm"].abs()
print(f"  blocked-shot rows corrected: {n_blk_pre:,}")
print(f"  total shots in scope: {len(shots):,} (Corsi)")

# Compute zones for both wide and tight
xs = shots["x_coord_norm"].values
absy = shots["y_coord_norm"].abs().values
shots["abs_y"] = absy
shots["zone_wide"]  = [classify_wide(x) for x in xs]
shots["zone_tight"] = [classify_tight(x, ay) for x,ay in zip(xs, absy)]
shots["band_wide"]  = [y_band_wide(ay) if zw else None
                       for ay, zw in zip(absy, shots["zone_wide"].values)]
shots["band_tight"] = [y_band_tight(ay) if zt else None
                       for ay, zt in zip(absy, shots["zone_tight"].values)]
shots["abs_time"] = shots["time_secs"].astype(int) + (shots["period"].astype(int)-1)*1200

# Filter to shots in either wide or tight zones (saves work)
in_scope = shots["zone_wide"].notna() | shots["zone_tight"].notna()
shots = shots[in_scope].copy()
print(f"  shots after zone filter: {len(shots):,}")
print(f"    wide CNFI: {(shots['zone_wide']=='CNFI').sum():,}")
print(f"    wide MNFI: {(shots['zone_wide']=='MNFI').sum():,}")
print(f"    tight CNFI: {(shots['zone_tight']=='CNFI').sum():,}")
print(f"    tight MNFI: {(shots['zone_tight']=='MNFI').sum():,}")
print(f"    tight FNFI: {(shots['zone_tight']=='FNFI').sum():,}")

shots_by_game = dict(tuple(shots.groupby("game_id")))
valid_gids = set(shots_by_game.keys())

# ---- Load shifts ----
print("Loading shifts ...")
shift_cols = ["game_id","player_id","period","team_abbrev","abs_start_secs","abs_end_secs"]
parts = []
for ch in pd.read_csv(SHIFT_CSV, usecols=shift_cols, chunksize=500_000):
    ch = ch.dropna(subset=shift_cols)
    ch["game_id"] = ch["game_id"].astype(int)
    ch["player_id"] = ch["player_id"].astype(int)
    ch["period"] = ch["period"].astype(int)
    ch["abs_start_secs"] = ch["abs_start_secs"].astype(int)
    ch["abs_end_secs"]   = ch["abs_end_secs"].astype(int)
    ch = ch[ch["game_id"].isin(valid_gids) & ch["period"].between(1,3)]
    if len(ch): parts.append(ch)
shifts = pd.concat(parts, ignore_index=True)
del parts
print(f"  shifts: {len(shifts):,}")
shifts_by_game = dict(tuple(shifts.groupby("game_id")))

# ---- Aggregators ----
# P1a: forwards on-ice FOR, wide zones
p1a_att = defaultdict(int)   # (pid, zone, band) -> count
p1a_absy_sum = defaultdict(float)
p1a_absy_n   = defaultdict(int)
# P5: D on-ice FOR, tight zones
p5_att = defaultdict(int)
# P2: F on-ice AGAINST, tight zones
p2_att = defaultdict(int)
p2_absy_sum = defaultdict(float)
p2_absy_n   = defaultdict(int)
# P4: D on-ice AGAINST, tight zones
p4_att = defaultdict(int)

print("Per-game shot-shift join ...")
n_games = 0
for gid, gshots in shots_by_game.items():
    n_games += 1
    if n_games % 500 == 0: print(f"  {n_games} games...")
    gshifts = shifts_by_game.get(gid)
    if gshifts is None: continue

    shifts_by_team = {}
    for team_ab, tsh in gshifts.groupby("team_abbrev"):
        st = tsh["abs_start_secs"].values.astype(np.int32)
        en = tsh["abs_end_secs"].values.astype(np.int32)
        pids = tsh["player_id"].values.astype(np.int64)
        order = np.argsort(st)
        shifts_by_team[team_ab] = (st[order], en[order], pids[order])

    for r in gshots.itertuples(index=False):
        t = int(r.abs_time)
        absy_v = float(r.abs_y)
        shoot_ab = r.shooting_team_abbrev
        def_ab = r.away_team_abbrev if shoot_ab == r.home_team_abbrev else r.home_team_abbrev

        # Resolve on-ice players for both sides once
        sh_pids_F = sh_pids_D = def_pids_F = def_pids_D = None
        if shoot_ab in shifts_by_team:
            st, en, pids = shifts_by_team[shoot_ab]
            idx = np.searchsorted(st, t, side="right")
            if idx > 0:
                on = en[:idx] > t
                onp = pids[:idx][on]
                sh_pids_F = [int(p) for p in onp if pos_grp.get(int(p))=="F"]
                sh_pids_D = [int(p) for p in onp if pos_grp.get(int(p))=="D"]
        if def_ab in shifts_by_team:
            st, en, pids = shifts_by_team[def_ab]
            idx = np.searchsorted(st, t, side="right")
            if idx > 0:
                on = en[:idx] > t
                onp = pids[:idx][on]
                def_pids_F = [int(p) for p in onp if pos_grp.get(int(p))=="F"]
                def_pids_D = [int(p) for p in onp if pos_grp.get(int(p))=="D"]

        # P1a — wide zones, forwards on-ice FOR
        if r.zone_wide and sh_pids_F:
            zw, bw = r.zone_wide, r.band_wide
            for pid in sh_pids_F:
                p1a_att[(pid, zw, bw)] += 1
                p1a_absy_sum[(pid, zw)] += absy_v
                p1a_absy_n[(pid, zw)]   += 1

        # tight zones for P2/P4/P5
        if r.zone_tight:
            zt, bt = r.zone_tight, r.band_tight
            if sh_pids_D:    # P5 (D on-ice FOR)
                for pid in sh_pids_D:
                    p5_att[(pid, zt, bt)] += 1
            if def_pids_F:   # P2 (F on-ice AGAINST)
                for pid in def_pids_F:
                    p2_att[(pid, zt, bt)] += 1
                    p2_absy_sum[(pid, zt)] += absy_v
                    p2_absy_n[(pid, zt)]   += 1
            if def_pids_D:   # P4 (D on-ice AGAINST)
                for pid in def_pids_D:
                    p4_att[(pid, zt, bt)] += 1

print("Aggregation complete.")

# ---- Build P1a output ----
print("Building P1a (Corsi)...")
all_f_p1a = sorted({pid for (pid,_,_) in p1a_att.keys()})
rows = []
for pid in all_f_p1a:
    toi_min = toi_es_min.get(pid, 0.0)
    if toi_min < MIN_F_TOI: continue
    rec = {"player_id": pid,
           "player_name": name_map.get(pid, ""),
           "es_toi_min": round(toi_min, 2)}
    for zone in ["CNFI","MNFI"]:
        tot_n = 0; tot_w = 0.0
        for band in WIDE_BANDS:
            n = p1a_att.get((pid, zone, band), 0)
            tot_n += n
            tot_w += n * WEIGHTS_WIDE[zone][band]
        n_c = p1a_absy_n.get((pid, zone), 0)
        cent = p1a_absy_sum.get((pid, zone), 0.0)/n_c if n_c else float("nan")
        rec[f"{zone}_attempts"]      = tot_n
        rec[f"P1a_raw_{zone}"]       = round(tot_n/toi_min*60.0, 4)
        rec[f"P1a_weighted_{zone}"]  = round(tot_w/toi_min*60.0, 4)
        rec[f"P1a_centrality_{zone}"]= round(cent, 3) if not math.isnan(cent) else np.nan
    rows.append(rec)
p1a = pd.DataFrame(rows)

def add_rank_flag(df, raw_col, w_col, prefix):
    df = df.copy()
    df[f"{prefix}_rank_raw"] = df[raw_col].rank(ascending=False, method="min").astype(int)
    df[f"{prefix}_rank_w"]   = df[w_col].rank(ascending=False, method="min").astype(int)
    df[f"{prefix}_rank_delta"] = df[f"{prefix}_rank_raw"] - df[f"{prefix}_rank_w"]
    df[f"{prefix}_centrality_flag"] = np.where(df[f"{prefix}_rank_delta"].abs() > 10,
        np.where(df[f"{prefix}_rank_delta"]>0, "BOOSTED", "PENALIZED"), "")
    return df

p1a = add_rank_flag(p1a, "P1a_raw_CNFI", "P1a_weighted_CNFI", "CNFI")
p1a = add_rank_flag(p1a, "P1a_raw_MNFI", "P1a_weighted_MNFI", "MNFI")
p1a = p1a.sort_values("P1a_weighted_CNFI", ascending=False).reset_index(drop=True)
p1a_cols = ["player_id","player_name","es_toi_min",
            "CNFI_attempts","P1a_raw_CNFI","P1a_weighted_CNFI","P1a_centrality_CNFI",
            "CNFI_rank_raw","CNFI_rank_w","CNFI_rank_delta","CNFI_centrality_flag",
            "MNFI_attempts","P1a_raw_MNFI","P1a_weighted_MNFI","P1a_centrality_MNFI",
            "MNFI_rank_raw","MNFI_rank_w","MNFI_rank_delta","MNFI_centrality_flag"]
p1a = p1a[p1a_cols]
p1a.to_csv(f"{OUT}/P1a_centrality_weighted.csv", index=False)
print(f"  P1a: {len(p1a)} forwards")

# ---- Build P2 (F defensive against; tight zones) ----
print("Building P2 (Corsi)...")
all_f_p2 = sorted({pid for (pid,_,_) in p2_att.keys()})
rows = []
for pid in all_f_p2:
    toi_min = toi_es_min.get(pid, 0.0)
    if toi_min < MIN_F_TOI: continue
    rec = {"player_id": pid, "player_name": name_map.get(pid,""), "es_toi_min": round(toi_min,2)}
    tot_n=0; tot_w=0.0
    for zone in ["CNFI","MNFI","FNFI"]:
        zn=0; zw=0.0
        for band in TIGHT_BANDS:
            n = p2_att.get((pid, zone, band), 0)
            zn += n
            zw += n * WEIGHTS_TIGHT[zone].get(band, 0.0)
        sa, lo, hi = rate_ci(zn, toi_min)
        rec[f"{zone}_SA"]=zn; rec[f"{zone}_SA_per60"]=round(sa,4)
        rec[f"{zone}_SA_lo95"]=round(lo,4); rec[f"{zone}_SA_hi95"]=round(hi,4)
        n_c = p2_absy_n.get((pid, zone), 0)
        rec[f"{zone}_centrality"] = (round(p2_absy_sum.get((pid,zone),0.0)/n_c,3)
                                     if n_c else np.nan)
        tot_n += zn; tot_w += zw
    sa, lo, hi = rate_ci(tot_n, toi_min)
    rec["TNFI_SA"]=tot_n; rec["TNFI_SA_per60"]=round(sa,4)
    rec["TNFI_SA_lo95"]=round(lo,4); rec["TNFI_SA_hi95"]=round(hi,4)
    rec["P2_weighted"]=round(tot_w/toi_min*60.0, 4)
    rows.append(rec)
p2 = pd.DataFrame(rows)
p2["TNFI_rank_raw"] = p2["TNFI_SA_per60"].rank(ascending=True, method="min").astype(int)
p2["P2_weighted_rank"] = p2["P2_weighted"].rank(ascending=True, method="min").astype(int)
p2["rank_delta"] = p2["TNFI_rank_raw"] - p2["P2_weighted_rank"]
p2["weighting_flag"] = np.where(p2["rank_delta"].abs()>10,
    np.where(p2["rank_delta"]>0, "LOW_DANGER_MIX", "HIGH_DANGER_MIX"), "")
p2 = p2.sort_values("TNFI_SA_per60").reset_index(drop=True)
p2_cols = ["player_id","player_name","es_toi_min",
           "CNFI_SA","CNFI_SA_per60","CNFI_SA_lo95","CNFI_SA_hi95","CNFI_centrality",
           "MNFI_SA","MNFI_SA_per60","MNFI_SA_lo95","MNFI_SA_hi95","MNFI_centrality",
           "FNFI_SA","FNFI_SA_per60","FNFI_SA_lo95","FNFI_SA_hi95","FNFI_centrality",
           "TNFI_SA","TNFI_SA_per60","TNFI_SA_lo95","TNFI_SA_hi95",
           "P2_weighted","TNFI_rank_raw","P2_weighted_rank","rank_delta","weighting_flag"]
p2 = p2[p2_cols]
p2.to_csv(f"{OUT}/P2_defensive_forwards.csv", index=False)
print(f"  P2: {len(p2)} forwards")

# ---- Build P4 (D against, tight) ----
def build_p4_p5(att_map, kind):
    """kind='P4' (against) or 'P5' (for)"""
    all_d = sorted({pid for (pid,_,_) in att_map.keys()})
    rows = []
    for pid in all_d:
        toi_min = toi_es_min.get(pid, 0.0)
        if toi_min < MIN_D_TOI: continue
        rec = {"player_id": pid, "player_name": name_map.get(pid,""), "es_toi_min": round(toi_min,2)}
        tot_n=0; tot_w=0.0
        for zone in ["CNFI","MNFI","FNFI"]:
            zn=0; zw=0.0
            for band in TIGHT_BANDS:
                n = att_map.get((pid, zone, band), 0)
                zn += n
                zw += n * WEIGHTS_TIGHT[zone].get(band, 0.0)
            r, lo, hi = rate_ci(zn, toi_min)
            tag = "SA" if kind=="P4" else "SF"
            rec[f"{zone}_{tag}"]=zn; rec[f"{zone}_{tag}_per60"]=round(r,4)
            rec[f"{zone}_{tag}_lo95"]=round(lo,4); rec[f"{zone}_{tag}_hi95"]=round(hi,4)
            tot_n += zn; tot_w += zw
        r, lo, hi = rate_ci(tot_n, toi_min)
        tag = "SA" if kind=="P4" else "SF"
        rec[f"TNFI_{tag}"]=tot_n; rec[f"TNFI_{tag}_per60"]=round(r,4)
        rec[f"TNFI_{tag}_lo95"]=round(lo,4); rec[f"TNFI_{tag}_hi95"]=round(hi,4)
        wkey = f"{kind}_weighted"
        rec[wkey] = round(tot_w/toi_min*60.0, 4)
        rows.append(rec)
    return pd.DataFrame(rows)

print("Building P4 (Corsi) and P5 (Corsi)...")
p4 = build_p4_p5(p4_att, "P4")
p5 = build_p4_p5(p5_att, "P5")

# P4 ranks (lower = better)
p4["TNFI_rank_raw"]    = p4["TNFI_SA_per60"].rank(ascending=True, method="min").astype(int)
p4["P4_weighted_rank"] = p4["P4_weighted"].rank(ascending=True, method="min").astype(int)
p4["rank_delta"]       = p4["TNFI_rank_raw"] - p4["P4_weighted_rank"]
p4["weighting_flag"]   = np.where(p4["rank_delta"].abs()>10,
    np.where(p4["rank_delta"]>0, "LOW_DANGER_MIX", "HIGH_DANGER_MIX"), "")
p4 = p4.sort_values("TNFI_SA_per60").reset_index(drop=True)
p4_cols = ["player_id","player_name","es_toi_min",
           "CNFI_SA","CNFI_SA_per60","CNFI_SA_lo95","CNFI_SA_hi95",
           "MNFI_SA","MNFI_SA_per60","MNFI_SA_lo95","MNFI_SA_hi95",
           "FNFI_SA","FNFI_SA_per60","FNFI_SA_lo95","FNFI_SA_hi95",
           "TNFI_SA","TNFI_SA_per60","TNFI_SA_lo95","TNFI_SA_hi95",
           "P4_weighted","TNFI_rank_raw","P4_weighted_rank","rank_delta","weighting_flag"]
p4 = p4[p4_cols]
p4.to_csv(f"{OUT}/P4_defensive_D.csv", index=False)
print(f"  P4: {len(p4)} D")

# P5 ranks (higher = better)
p5["TNFI_rank_raw"]    = p5["TNFI_SF_per60"].rank(ascending=False, method="min").astype(int)
p5["P5_weighted_rank"] = p5["P5_weighted"].rank(ascending=False, method="min").astype(int)
p5["rank_delta"]       = p5["TNFI_rank_raw"] - p5["P5_weighted_rank"]
p5["weighting_flag"]   = np.where(p5["rank_delta"].abs()>10,
    np.where(p5["rank_delta"]>0, "HIGH_DANGER_MIX", "LOW_DANGER_MIX"), "")
p5 = p5.sort_values("TNFI_SF_per60", ascending=False).reset_index(drop=True)
p5_cols = ["player_id","player_name","es_toi_min",
           "CNFI_SF","CNFI_SF_per60","CNFI_SF_lo95","CNFI_SF_hi95",
           "MNFI_SF","MNFI_SF_per60","MNFI_SF_lo95","MNFI_SF_hi95",
           "FNFI_SF","FNFI_SF_per60","FNFI_SF_lo95","FNFI_SF_hi95",
           "TNFI_SF","TNFI_SF_per60","TNFI_SF_lo95","TNFI_SF_hi95",
           "P5_weighted","TNFI_rank_raw","P5_weighted_rank","rank_delta","weighting_flag"]
p5 = p5[p5_cols]
p5.to_csv(f"{OUT}/P5_offensive_D.csv", index=False)
print(f"  P5: {len(p5)} D")

# ---- Two-way scores ----
print("Building two-way scores ...")
def z(s): return (s - s.mean()) / s.std(ddof=0)

# Forwards: P1a_weighted_total = CNFI + MNFI weighted
p1a["P1a_weighted_total"] = p1a["P1a_weighted_CNFI"] + p1a["P1a_weighted_MNFI"]
twf = p2.merge(p1a[["player_id","P1a_weighted_CNFI","P1a_weighted_MNFI",
                    "P1a_weighted_total"]], on="player_id", how="inner")
twf["z_P1a_weighted"] = z(twf["P1a_weighted_total"])
twf["z_P2_weighted"]  = z(twf["P2_weighted"])
twf["twoway_score"]   = twf["z_P1a_weighted"] - twf["z_P2_weighted"]
twf["off_rank"]       = twf["P1a_weighted_total"].rank(ascending=False, method="min").astype(int)
twf["def_rank"]       = twf["P2_weighted"].rank(ascending=True,  method="min").astype(int)
twf["twoway_rank"]    = twf["twoway_score"].rank(ascending=False, method="min").astype(int)
twf = twf.sort_values("twoway_score", ascending=False).reset_index(drop=True)
twf_cols = ["player_id","player_name","es_toi_min",
            "P1a_weighted_CNFI","P1a_weighted_MNFI","P1a_weighted_total",
            "P2_weighted","z_P1a_weighted","z_P2_weighted","twoway_score",
            "off_rank","def_rank","twoway_rank"]
twf = twf[twf_cols]
twf.to_csv(f"{OUT}/twoway_forward_score.csv", index=False)
print(f"  twoway_F: {len(twf)} F")

twd = p4[["player_id","player_name","es_toi_min","TNFI_SA_per60","P4_weighted"]].merge(
    p5[["player_id","TNFI_SF_per60","P5_weighted"]], on="player_id", how="inner")
twd["z_P5_weighted"]  = z(twd["P5_weighted"])
twd["z_P4_weighted"]  = z(twd["P4_weighted"])
twd["twoway_D_score"] = twd["z_P5_weighted"] - twd["z_P4_weighted"]
twd["off_rank"]       = twd["P5_weighted"].rank(ascending=False, method="min").astype(int)
twd["def_rank"]       = twd["P4_weighted"].rank(ascending=True,  method="min").astype(int)
twd["twoway_D_rank"]  = twd["twoway_D_score"].rank(ascending=False, method="min").astype(int)
twd = twd.sort_values("twoway_D_score", ascending=False).reset_index(drop=True)
twd_cols = ["player_id","player_name","es_toi_min",
            "TNFI_SA_per60","P4_weighted","TNFI_SF_per60","P5_weighted",
            "z_P5_weighted","z_P4_weighted","twoway_D_score",
            "off_rank","def_rank","twoway_D_rank"]
twd = twd[twd_cols]
twd.to_csv(f"{OUT}/twoway_D_score.csv", index=False)
print(f"  twoway_D: {len(twd)} D")

# ---- Update zone_conversion_rates.csv: add Corsi columns ----
print("Updating zone_conversion_rates.csv with Corsi-based rates ...")
zr = pd.read_csv(f"{OUT}/zone_conversion_rates.csv")
# Recompute Corsi (full attempts) with corrected coords for the zones reported
es = shots.copy()  # already filtered ES, regulation, situation 1551
def compute_zone_corsi(zone_name):
    if zone_name == "HD_conventional (x>69, y in ±22)":
        sub = es[(es["x_coord_norm"]>69) & (es["abs_y"]<=22)]
    elif zone_name == "TNFI (CNFI+MNFI+FNFI)":
        sub = es[es["zone_tight"].isin(["CNFI","MNFI","FNFI"])]
    elif zone_name == "ALL ES REG":
        sub = es
    elif zone_name in ["CNFI","MNFI","FNFI"]:
        sub = es[es["zone_tight"]==zone_name]
    elif zone_name == "Wide":
        # |y|>15 OR (|y| range in CNFI=>9..15)
        # Use the original definition from script 09
        # We don't have all wide-classification data; approximate from non-zone_tight rows
        sub = es[(es["abs_y"]>15) | ((es["x_coord_norm"]>=74) & (es["abs_y"]>9) & (es["abs_y"]<=15))]
    elif zone_name == "lane_other":
        # Same caveat — approximate.
        sub = es[(es["abs_y"]<=15) & ~es["zone_tight"].isin(["CNFI","MNFI","FNFI"])]
    else:
        sub = es.iloc[0:0]
    n_corsi = len(sub)
    g_corsi = int(sub["is_goal"].sum())
    p, lo, hi = wilson(g_corsi, n_corsi)
    return n_corsi, g_corsi, p, lo, hi

corsi_rows = []
for _, row in zr.iterrows():
    n_c, g_c, p, lo, hi = compute_zone_corsi(row["zone"])
    corsi_rows.append({
        "corsi_attempts": n_c, "corsi_goals": g_c,
        "corsi_conversion_pct": round(p*100,3),
        "corsi_ci_lo_pct": round(lo*100,3),
        "corsi_ci_hi_pct": round(hi*100,3),
    })
corsi_df = pd.DataFrame(corsi_rows)
zr2 = pd.concat([zr.reset_index(drop=True), corsi_df], axis=1)
zr2.to_csv(f"{OUT}/zone_conversion_rates.csv", index=False)
print("  zone_conversion_rates.csv updated with corsi_* columns")

# ---- Compare to backed-up versions ----
print("\n========== RANK CHANGE COMPARISONS ==========")
BK = f"{OUT}/_pre_corsi_backup"

def compare(new_df, file, key, score_col, asc, label):
    """Compare ranks between new and backup. score_col higher (asc=False) or lower (asc=True) = better."""
    old = pd.read_csv(f"{BK}/{file}")
    new = new_df.copy()
    old["__rank_old"] = old[score_col].rank(ascending=asc, method="min")
    new["__rank_new"] = new[score_col].rank(ascending=asc, method="min")
    m = old[["player_id",score_col,"__rank_old"]].merge(
        new[["player_id","player_name",score_col,"__rank_new"]], on="player_id",
        suffixes=("_old","_new"))
    m["rank_delta"] = m["__rank_old"] - m["__rank_new"]   # +ve = improved
    moved = m[m["rank_delta"].abs() > 10].copy()
    moved["abs_delta"] = moved["rank_delta"].abs()
    moved = moved.sort_values("abs_delta", ascending=False)
    print(f"\n--- {label} on {score_col} (n_old={len(old)}, n_new={len(new)}) ---")
    print(f"  players moved >10 ranks: {len(moved)}")
    if len(moved):
        print(f"  Biggest movers:")
        print(moved.head(15)[["player_name", f"{score_col}_old", f"{score_col}_new",
                              "__rank_old","__rank_new","rank_delta"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return m, moved

# Forwards
_, p1a_movers = compare(p1a, "P1a_centrality_weighted.csv", "player_id",
                        "P1a_weighted_CNFI", False, "P1a CNFI weighted")
_, p2_movers  = compare(p2,  "P2_defensive_forwards.csv",   "player_id",
                        "P2_weighted",        True,  "P2 weighted (lower=better)")
_, twf_movers = compare(twf, "twoway_forward_score.csv",    "player_id",
                        "twoway_score",       False, "two-way F score")

# D
_, p4_movers  = compare(p4,  "P4_defensive_D.csv",          "player_id",
                        "P4_weighted",        True,  "P4 weighted (lower=better)")
_, p5_movers  = compare(p5,  "P5_offensive_D.csv",          "player_id",
                        "P5_weighted",        False, "P5 weighted")
_, twd_movers = compare(twd, "twoway_D_score.csv",          "player_id",
                        "twoway_D_score",     False, "two-way D score")

# ---- Tier change analysis ----
def tier_change_analysis(new_df, old_file, score_col, asc, label):
    old = pd.read_csv(f"{BK}/{old_file}")
    new = new_df.copy()
    # Quartile tiers within each frame
    for d, suf in [(old, "old"), (new, "new")]:
        ranks = d[score_col].rank(ascending=asc, method="min", pct=True)
        d[f"tier_{suf}"] = pd.cut(ranks, bins=[0, 0.25, 0.5, 0.75, 1.001],
                                   labels=["Q1","Q2","Q3","Q4"], include_lowest=True)
    m = old[["player_id","tier_old"]].merge(
        new[["player_id","player_name","tier_new"]], on="player_id")
    changed = m[(m["tier_old"]=="Q1") & m["tier_new"].isin(["Q3","Q4"])]
    other_dir = m[(m["tier_old"].isin(["Q3","Q4"])) & (m["tier_new"]=="Q1")]
    print(f"\n--- {label} tier shifts ---")
    print(f"  Q1 -> middle/bottom: {len(changed)}")
    if len(changed):
        print(changed[["player_name","tier_old","tier_new"]].head(20).to_string(index=False))
    print(f"  middle/bottom -> Q1: {len(other_dir)}")
    if len(other_dir):
        print(other_dir[["player_name","tier_old","tier_new"]].head(20).to_string(index=False))

tier_change_analysis(twf, "twoway_forward_score.csv", "twoway_score", False, "two-way F")
tier_change_analysis(twd, "twoway_D_score.csv",       "twoway_D_score", False, "two-way D")

# ---- Spot-check Tanev / Faber on P4 ----
print("\n========== SPOT-CHECK: Chris Tanev & Brock Faber on P4 ==========")
p4_old = pd.read_csv(f"{BK}/P4_defensive_D.csv")

for name in ["Chris Tanev","Brock Faber"]:
    o = p4_old[p4_old["player_name"]==name]
    n = p4[p4["player_name"]==name]
    if len(o)==0 or len(n)==0:
        print(f"  {name}: not found in one of the files")
        continue
    o = o.iloc[0]; n = n.iloc[0]
    print(f"\n  {name}")
    print(f"    P4_weighted:    OLD {o['P4_weighted']:.4f}  ->  NEW {n['P4_weighted']:.4f}  "
          f"(Δ {n['P4_weighted']-o['P4_weighted']:+.4f})")
    print(f"    TNFI_SA/60:     OLD {o['TNFI_SA_per60']:.3f}  ->  NEW {n['TNFI_SA_per60']:.3f}  "
          f"(Δ {n['TNFI_SA_per60']-o['TNFI_SA_per60']:+.3f})")
    print(f"    TNFI_SA total:  OLD {o['TNFI_SA']}  ->  NEW {n['TNFI_SA']}  "
          f"(Δ {n['TNFI_SA']-o['TNFI_SA']:+d})")
    # Rank
    o_rank_w = (p4_old["P4_weighted"] <= o["P4_weighted"]).sum()
    n_rank_w = (p4["P4_weighted"]     <= n["P4_weighted"]).sum()
    o_rank_r = (p4_old["TNFI_SA_per60"] <= o["TNFI_SA_per60"]).sum()
    n_rank_r = (p4["TNFI_SA_per60"]     <= n["TNFI_SA_per60"]).sum()
    print(f"    P4_weighted rank: OLD {o_rank_w}/{len(p4_old)}  ->  NEW {n_rank_w}/{len(p4)}")
    print(f"    TNFI_SA/60 rank:  OLD {o_rank_r}/{len(p4_old)}  ->  NEW {n_rank_r}/{len(p4)}")

print("\nFiles overwritten in NFI/output/:")
for f in ["P1a_centrality_weighted.csv","P2_defensive_forwards.csv",
          "P4_defensive_D.csv","P5_offensive_D.csv",
          "twoway_forward_score.csv","twoway_D_score.csv",
          "zone_conversion_rates.csv"]:
    print(f"  {f}")
print("Backups in NFI/output/_pre_corsi_backup/")
