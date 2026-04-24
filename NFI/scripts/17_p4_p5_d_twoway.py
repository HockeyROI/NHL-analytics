#!/usr/bin/env python3
"""
P4 (defensive D) and P5 (offensive D) + two-way D score.

Zones (same as P2):
  CNFI: x in [74,89] AND |y| <= 9
  MNFI: x in [55,73] AND |y| <= 15
  FNFI: x in [25,54] AND |y| <= 15

Filters:
  - ES (situation_code 1551), regulation (period 1-3), regular season
  - 5 seasons pooled: 20212022..20252026
  - Defensemen only (pos_group == 'D')
  - Min 500 ES TOI minutes
  - Unblocked attempts (Fenwick: SOG + missed-shot + goal)

P4 (defensive): on-ice AGAINST attempts (opposing team shots) per 60.
P5 (offensive): on-ice FOR attempts (own team shots) per 60.

Y-band weights (per zone, from prior y-band analyses):
  CNFI: 0-5 .1579 / 5-10 .1075         (|y|<=9)
  MNFI: 0-5 .1109 / 5-10 .0985 / 10-15 .0887   (|y|<=15)
  FNFI: 0-5 .0367 / 5-10 .0335 / 10-15 .0289   (|y|<=15)

P4_weighted = sum(SA * band_rate) per 60 across zones (lower = better)
P5_weighted = sum(SF * band_rate) per 60 across zones (higher = better)

Two-way D score = z(P5_weighted) - z(P4_weighted)
  (across qualifying D, higher = better two-way)

Outputs:
  NFI/output/P4_defensive_D.csv
  NFI/output/P5_offensive_D.csv
  NFI/output/twoway_D_score.csv

Note: handedness is not in player_positions.csv (only L/R/C/D/G role).
We do not include a shoots-handedness column.
"""
import os, math
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
SHIFT_CSV = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
TOI_CSV = f"{ROOT}/NFI/output/player_toi.csv"
POS_CSV = f"{ROOT}/NFI/output/player_positions.csv"
OUT_P4 = f"{ROOT}/NFI/output/P4_defensive_D.csv"
OUT_P5 = f"{ROOT}/NFI/output/P5_offensive_D.csv"
OUT_TW = f"{ROOT}/NFI/output/twoway_D_score.csv"

SEASONS = {"20212022","20222023","20232024","20242025","20252026"}
MIN_ES_TOI_MIN = 500.0

WEIGHTS = {
    "CNFI": {"0-5": 0.1579, "5-10": 0.1075},
    "MNFI": {"0-5": 0.1109, "5-10": 0.0985, "10-15": 0.0887},
    "FNFI": {"0-5": 0.0367, "5-10": 0.0335, "10-15": 0.0289},
}
BAND_ORDER = ["0-5","5-10","10-15"]

def y_band(absy):
    if absy < 5:   return "0-5"
    if absy < 10:  return "5-10"
    return "10-15"

def classify(x, absy):
    if 74 <= x <= 89 and absy <= 9:   return "CNFI"
    if 55 <= x <= 73 and absy <= 15:  return "MNFI"
    if 25 <= x <= 54 and absy <= 15:  return "FNFI"
    return None

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    halfw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return p, max(0.0, center-halfw), min(1.0, center+halfw)

def rate_ci(events, minutes, z=1.96):
    if minutes <= 0: return 0.0, 0.0, 0.0
    p, lo, hi = wilson(events, max(events, int(round(minutes))))
    return events/minutes*60.0, lo*60.0, hi*60.0

# ---- Load positions, TOI ----
print("Loading positions / TOI...")
pos_df = pd.read_csv(POS_CSV, dtype={"player_id": int})
pos_map = dict(zip(pos_df["player_id"], pos_df["pos_group"]))
name_map = dict(zip(pos_df["player_id"], pos_df["player_name"]))

toi_df = pd.read_csv(TOI_CSV, dtype={"player_id": int})
toi_es_min = dict(zip(toi_df["player_id"], toi_df["toi_ES_sec"]/60.0))

# ---- Load shots ----
print("Loading shots...")
shot_cols = ["game_id","season","period","event_type","situation_code","time_secs",
             "home_team_id","shooting_team_id","shooting_team_abbrev",
             "home_team_abbrev","away_team_abbrev","x_coord_norm","y_coord_norm","is_goal"]
shots = pd.read_csv(SHOT_CSV, usecols=shot_cols,
                    dtype={"season":str,"situation_code":str})
shots = shots[shots["season"].isin(SEASONS)].copy()
shots = shots[shots["period"].between(1,3)].copy()
shots = shots[shots["event_type"].isin(["shot-on-goal","missed-shot","goal"])].copy()
shots = shots[shots["situation_code"].astype(str)=="1551"].copy()

shots["abs_y"] = shots["y_coord_norm"].abs()
shots["zone"] = [classify(x,ay) for x,ay in zip(shots["x_coord_norm"].values, shots["abs_y"].values)]
shots = shots[shots["zone"].notna()].copy()
shots["band"] = shots["abs_y"].apply(y_band)
shots["abs_time"] = shots["time_secs"].astype(int) + (shots["period"].astype(int)-1)*1200
print(f"  shots in scope: {len(shots):,}")

shots_by_game = dict(tuple(shots.groupby("game_id")))
valid_gids = set(shots_by_game.keys())

# ---- Load shifts ----
print("Loading shifts...")
shift_cols = ["game_id","player_id","period","team_abbrev","abs_start_secs","abs_end_secs"]
parts = []
for ch in pd.read_csv(SHIFT_CSV, usecols=shift_cols, chunksize=500_000):
    ch = ch.dropna(subset=shift_cols)
    ch["game_id"] = ch["game_id"].astype(int)
    ch["player_id"] = ch["player_id"].astype(int)
    ch["period"] = ch["period"].astype(int)
    ch["abs_start_secs"] = ch["abs_start_secs"].astype(int)
    ch["abs_end_secs"] = ch["abs_end_secs"].astype(int)
    ch = ch[ch["game_id"].isin(valid_gids) & ch["period"].between(1,3)]
    if len(ch): parts.append(ch)
shifts = pd.concat(parts, ignore_index=True)
del parts
print(f"  shifts: {len(shifts):,}")
shifts_by_game = dict(tuple(shifts.groupby("game_id")))

# ---- Aggregators (single pass: both for and against) ----
sf_att = defaultdict(int)   # (pid,zone,band) -> shots FOR (own team)
sf_gl  = defaultdict(int)
sa_att = defaultdict(int)   # (pid,zone,band) -> shots AGAINST (opp team)
sa_gl  = defaultdict(int)

print("Per-game shot-shift join (D, for and against)...")
n_games = 0
for gid, gshots in shots_by_game.items():
    n_games += 1
    if n_games % 500 == 0:
        print(f"  {n_games} games...")
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
        zone, band = r.zone, r.band
        is_goal = int(r.is_goal)
        shoot_ab = r.shooting_team_abbrev
        def_ab = r.away_team_abbrev if shoot_ab == r.home_team_abbrev else r.home_team_abbrev

        # FOR: shooting team D on ice
        if shoot_ab in shifts_by_team:
            st, en, pids = shifts_by_team[shoot_ab]
            idx = np.searchsorted(st, t, side="right")
            if idx > 0:
                on = en[:idx] > t
                for pid in pids[:idx][on]:
                    if pos_map.get(int(pid)) != "D": continue
                    k = (int(pid), zone, band)
                    sf_att[k] += 1
                    sf_gl[k]  += is_goal

        # AGAINST: defending team D on ice
        if def_ab in shifts_by_team:
            st, en, pids = shifts_by_team[def_ab]
            idx = np.searchsorted(st, t, side="right")
            if idx > 0:
                on = en[:idx] > t
                for pid in pids[:idx][on]:
                    if pos_map.get(int(pid)) != "D": continue
                    k = (int(pid), zone, band)
                    sa_att[k] += 1
                    sa_gl[k]  += is_goal

print("Aggregation done.")

# ---- Build per-D summaries ----
all_d = sorted({pid for (pid,_,_) in sa_att.keys()} | {pid for (pid,_,_) in sf_att.keys()})

def build_summary(att_map, weights, kind):
    """kind = 'P4' (against) or 'P5' (for)."""
    rows = []
    for pid in all_d:
        toi_min = toi_es_min.get(pid, 0.0)
        if toi_min < MIN_ES_TOI_MIN: continue
        rec = {"player_id": pid,
               "player_name": name_map.get(pid, ""),
               "es_toi_min": round(toi_min, 2)}
        total_n = 0; total_w = 0.0
        for zone in ["CNFI","MNFI","FNFI"]:
            zn, zw = 0, 0.0
            for band in BAND_ORDER:
                n = att_map.get((pid, zone, band), 0)
                zn += n
                zw += n * weights[zone].get(band, 0.0)
            r, lo, hi = rate_ci(zn, toi_min)
            tag = "SA" if kind == "P4" else "SF"
            rec[f"{zone}_{tag}"]       = zn
            rec[f"{zone}_{tag}_per60"] = round(r, 4)
            rec[f"{zone}_{tag}_lo95"]  = round(lo, 4)
            rec[f"{zone}_{tag}_hi95"]  = round(hi, 4)
            total_n += zn
            total_w += zw
        r, lo, hi = rate_ci(total_n, toi_min)
        tag = "SA" if kind == "P4" else "SF"
        rec[f"TNFI_{tag}"]       = total_n
        rec[f"TNFI_{tag}_per60"] = round(r, 4)
        rec[f"TNFI_{tag}_lo95"]  = round(lo, 4)
        rec[f"TNFI_{tag}_hi95"]  = round(hi, 4)
        wkey = f"{kind}_weighted"
        rec[wkey] = round(total_w / toi_min * 60.0, 4)
        rows.append(rec)
    return pd.DataFrame(rows)

print("Building P4 (defensive) summary...")
p4 = build_summary(sa_att, WEIGHTS, "P4")
print("Building P5 (offensive) summary...")
p5 = build_summary(sf_att, WEIGHTS, "P5")

# ---- Ranks & flags ----
# P4: lower = better
p4["TNFI_rank_raw"]    = p4["TNFI_SA_per60"].rank(ascending=True, method="min").astype(int)
p4["P4_weighted_rank"] = p4["P4_weighted"].rank(ascending=True, method="min").astype(int)
p4["rank_delta"]       = p4["TNFI_rank_raw"] - p4["P4_weighted_rank"]
p4["weighting_flag"]   = np.where(
    p4["rank_delta"].abs() > 10,
    np.where(p4["rank_delta"] > 0, "LOW_DANGER_MIX", "HIGH_DANGER_MIX"), "")
p4 = p4.sort_values("TNFI_SA_per60", ascending=True).reset_index(drop=True)

# P5: higher = better
p5["TNFI_rank_raw"]    = p5["TNFI_SF_per60"].rank(ascending=False, method="min").astype(int)
p5["P5_weighted_rank"] = p5["P5_weighted"].rank(ascending=False, method="min").astype(int)
p5["rank_delta"]       = p5["TNFI_rank_raw"] - p5["P5_weighted_rank"]
p5["weighting_flag"]   = np.where(
    p5["rank_delta"].abs() > 10,
    np.where(p5["rank_delta"] > 0, "HIGH_DANGER_MIX", "LOW_DANGER_MIX"), "")
# For P5: rank_delta>0 means weighted rank is BETTER than raw (smaller number).
# That happens when shots-for cluster CENTRAL -> high-danger feed, GOOD for offense.
# So positive delta = HIGH_DANGER_MIX (good); negative = LOW_DANGER_MIX (perimeter feed).
p5 = p5.sort_values("TNFI_SF_per60", ascending=False).reset_index(drop=True)

p4_cols = ["player_id","player_name","es_toi_min",
           "CNFI_SA","CNFI_SA_per60","CNFI_SA_lo95","CNFI_SA_hi95",
           "MNFI_SA","MNFI_SA_per60","MNFI_SA_lo95","MNFI_SA_hi95",
           "FNFI_SA","FNFI_SA_per60","FNFI_SA_lo95","FNFI_SA_hi95",
           "TNFI_SA","TNFI_SA_per60","TNFI_SA_lo95","TNFI_SA_hi95",
           "P4_weighted","TNFI_rank_raw","P4_weighted_rank","rank_delta","weighting_flag"]
p5_cols = ["player_id","player_name","es_toi_min",
           "CNFI_SF","CNFI_SF_per60","CNFI_SF_lo95","CNFI_SF_hi95",
           "MNFI_SF","MNFI_SF_per60","MNFI_SF_lo95","MNFI_SF_hi95",
           "FNFI_SF","FNFI_SF_per60","FNFI_SF_lo95","FNFI_SF_hi95",
           "TNFI_SF","TNFI_SF_per60","TNFI_SF_lo95","TNFI_SF_hi95",
           "P5_weighted","TNFI_rank_raw","P5_weighted_rank","rank_delta","weighting_flag"]
p4 = p4[p4_cols]; p5 = p5[p5_cols]
p4.to_csv(OUT_P4, index=False)
p5.to_csv(OUT_P5, index=False)
print(f"\nWrote {OUT_P4}  ({len(p4)} D)")
print(f"Wrote {OUT_P5}  ({len(p5)} D)")

# ---- Two-way D ----
tw = p4[["player_id","player_name","es_toi_min","TNFI_SA_per60","P4_weighted"]].merge(
    p5[["player_id","TNFI_SF_per60","P5_weighted"]], on="player_id", how="inner")

def z(s):
    return (s - s.mean()) / s.std(ddof=0)

tw["z_P5_weighted"]   = z(tw["P5_weighted"])
tw["z_P4_weighted"]   = z(tw["P4_weighted"])
tw["twoway_D_score"]  = tw["z_P5_weighted"] - tw["z_P4_weighted"]
tw["off_rank"]        = tw["P5_weighted"].rank(ascending=False, method="min").astype(int)
tw["def_rank"]        = tw["P4_weighted"].rank(ascending=True,  method="min").astype(int)
tw["twoway_D_rank"]   = tw["twoway_D_score"].rank(ascending=False, method="min").astype(int)
tw = tw.sort_values("twoway_D_score", ascending=False).reset_index(drop=True)
tw_cols = ["player_id","player_name","es_toi_min",
           "TNFI_SA_per60","P4_weighted","TNFI_SF_per60","P5_weighted",
           "z_P5_weighted","z_P4_weighted","twoway_D_score",
           "off_rank","def_rank","twoway_D_rank"]
tw = tw[tw_cols]
tw.to_csv(OUT_TW, index=False)
print(f"Wrote {OUT_TW}  ({len(tw)} D)\n")

# ---- Console summary ----
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 220
pd.options.display.max_columns = None

print("--- Top 20 best DEFENSIVE D (lowest TNFI SA/60) ---")
print(p4.head(20)[["player_name","es_toi_min","TNFI_SA","TNFI_SA_per60",
                   "TNFI_SA_lo95","TNFI_SA_hi95","P4_weighted",
                   "TNFI_rank_raw","P4_weighted_rank","rank_delta","weighting_flag"]].to_string(index=False))

print("\n--- Worst 10 DEFENSIVE D (highest TNFI SA/60) ---")
print(p4.tail(10)[["player_name","TNFI_SA_per60","TNFI_SA_lo95","TNFI_SA_hi95",
                   "P4_weighted","TNFI_rank_raw","P4_weighted_rank","rank_delta","weighting_flag"]].to_string(index=False))

print("\n--- Top 20 best OFFENSIVE D (highest TNFI SF/60) ---")
print(p5.head(20)[["player_name","es_toi_min","TNFI_SF","TNFI_SF_per60",
                   "TNFI_SF_lo95","TNFI_SF_hi95","P5_weighted",
                   "TNFI_rank_raw","P5_weighted_rank","rank_delta","weighting_flag"]].to_string(index=False))

print("\n--- HIGH_DANGER_MIX P4 (rank worse when weighted - allows central) ---")
hd = p4[p4["weighting_flag"]=="HIGH_DANGER_MIX"].sort_values("rank_delta")
print(hd[["player_name","TNFI_SA_per60","P4_weighted","TNFI_rank_raw","P4_weighted_rank","rank_delta"]]
      .head(15).to_string(index=False))

print("\n--- LOW_DANGER_MIX P4 (rank better when weighted - allows perimeter) ---")
ld = p4[p4["weighting_flag"]=="LOW_DANGER_MIX"].sort_values("rank_delta", ascending=False)
print(ld[["player_name","TNFI_SA_per60","P4_weighted","TNFI_rank_raw","P4_weighted_rank","rank_delta"]]
      .head(15).to_string(index=False))

print("\n--- HIGH_DANGER_MIX P5 (rank IMPROVES when weighted - feeds central) ---")
hd5 = p5[p5["weighting_flag"]=="HIGH_DANGER_MIX"].sort_values("rank_delta", ascending=False)
print(hd5[["player_name","TNFI_SF_per60","P5_weighted","TNFI_rank_raw","P5_weighted_rank","rank_delta"]]
      .head(15).to_string(index=False))

print("\n--- LOW_DANGER_MIX P5 (rank WORSENS when weighted - perimeter feeds) ---")
ld5 = p5[p5["weighting_flag"]=="LOW_DANGER_MIX"].sort_values("rank_delta")
print(ld5[["player_name","TNFI_SF_per60","P5_weighted","TNFI_rank_raw","P5_weighted_rank","rank_delta"]]
      .head(15).to_string(index=False))

print("\n--- Top 25 two-way D ---")
print(tw.head(25)[["player_name","es_toi_min","P5_weighted","P4_weighted",
                   "z_P5_weighted","z_P4_weighted","twoway_D_score",
                   "off_rank","def_rank","twoway_D_rank"]].to_string(index=False))

print("\n--- Bottom 10 two-way D ---")
print(tw.tail(10)[["player_name","P5_weighted","P4_weighted",
                   "z_P5_weighted","z_P4_weighted","twoway_D_score",
                   "off_rank","def_rank","twoway_D_rank"]].to_string(index=False))
