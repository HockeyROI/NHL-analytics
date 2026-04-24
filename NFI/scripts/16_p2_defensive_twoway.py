#!/usr/bin/env python3
"""
P2 defensive forwards + two-way score.

Defensive zone definitions (tighter than the y-band analysis pool):
  CNFI: x_coord_norm in [74,89] AND |y_coord_norm| <= 9
  MNFI: x_coord_norm in [55,73] AND |y_coord_norm| <= 15
  FNFI: x_coord_norm in [25,54] AND |y_coord_norm| <= 15

Filters:
  - ES (situation_code 1551), regulation (period 1-3), regular season
  - 5 seasons pooled: 20212022..20252026
  - Forwards only
  - Min 500 ES TOI minutes per forward
  - Unblocked attempts (Fenwick: SOG + missed-shot + goal)

For each forward we count opposing-team shots taken while the forward is on
ice (on-ice against). Per zone: SA/60 with Wilson 95% CI. TNFI = CNFI+MNFI+FNFI.

Y-band weighting (per-zone conversion rates from the y-band analyses):
  CNFI: 0-5 .1579 / 5-10 .1075 / (10-15 .0781 unused, |y|<=9)
  MNFI: 0-5 .1109 / 5-10 .0985 / 10-15 .0887     (|y|<=15)
  FNFI: 0-5 .0367 / 5-10 .0335 / 10-15 .0289     (|y|<=15)
P2_weighted = sum(shots_against * band_rate) per 60, summed across zones.

Two-way score:
  P1a_weighted_total = P1a_weighted_CNFI + P1a_weighted_MNFI from
                       P1a_centrality_weighted.csv (CNFI+MNFI offensive pillar)
  z(P1a_weighted_total) - z(P2_weighted), across qualifying forwards
  (higher = better two-way).

Outputs:
  NFI/output/P2_defensive_forwards.csv
  NFI/output/twoway_forward_score.csv
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
P1A_CSV = f"{ROOT}/NFI/output/P1a_centrality_weighted.csv"
OUT_P2  = f"{ROOT}/NFI/output/P2_defensive_forwards.csv"
OUT_TW  = f"{ROOT}/NFI/output/twoway_forward_score.csv"

SEASONS = {"20212022", "20222023", "20232024", "20242025", "20252026"}
MIN_ES_TOI_MIN = 500.0

WEIGHTS = {
    "CNFI": {"0-5": 0.1579, "5-10": 0.1075, "10-15": 0.0781},
    "MNFI": {"0-5": 0.1109, "5-10": 0.0985, "10-15": 0.0887},
    "FNFI": {"0-5": 0.0367, "5-10": 0.0335, "10-15": 0.0289},
}
BAND_ORDER = ["0-5", "5-10", "10-15"]

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
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    halfw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return p, max(0.0, center - halfw), min(1.0, center + halfw)

def rate_ci(events, minutes, z=1.96):
    """Per-60 rate with Wilson-style CI (matches existing pipeline convention)."""
    if minutes <= 0:
        return 0.0, 0.0, 0.0
    p, lo, hi = wilson(events, max(events, int(round(minutes))))
    return events / minutes * 60.0, lo * 60.0, hi * 60.0

# ---- Load positions, TOI ----
print("Loading positions / TOI...")
pos_df = pd.read_csv(POS_CSV, dtype={"player_id": int})
pos_map = dict(zip(pos_df["player_id"], pos_df["pos_group"]))
name_map = dict(zip(pos_df["player_id"], pos_df["player_name"]))

toi_df = pd.read_csv(TOI_CSV, dtype={"player_id": int})
toi_es_min = dict(zip(toi_df["player_id"], toi_df["toi_ES_sec"] / 60.0))

# ---- Load shots ----
print("Loading shots...")
shot_cols = ["game_id","season","period","event_type","situation_code","time_secs",
             "home_team_id","shooting_team_id","shooting_team_abbrev",
             "home_team_abbrev","away_team_abbrev",
             "x_coord_norm","y_coord_norm","is_goal"]
shots = pd.read_csv(SHOT_CSV, usecols=shot_cols,
                    dtype={"season": str, "situation_code": str})
shots = shots[shots["season"].isin(SEASONS)].copy()
shots = shots[shots["period"].between(1, 3)].copy()
shots = shots[shots["event_type"].isin(["shot-on-goal","missed-shot","goal"])].copy()
shots = shots[shots["situation_code"].astype(str) == "1551"].copy()

shots["abs_y"] = shots["y_coord_norm"].abs()
shots["zone"] = [classify(x, ay) for x, ay in zip(shots["x_coord_norm"].values, shots["abs_y"].values)]
shots = shots[shots["zone"].notna()].copy()
shots["band"] = shots["abs_y"].apply(y_band)
shots["abs_time"] = shots["time_secs"].astype(int) + (shots["period"].astype(int)-1) * 1200
shots["_shoot_home"] = shots["shooting_team_id"] == shots["home_team_id"]
print(f"  shots in scope: {len(shots):,}  "
      f"(C {(shots['zone']=='CNFI').sum():,}, "
      f"M {(shots['zone']=='MNFI').sum():,}, "
      f"F {(shots['zone']=='FNFI').sum():,})")

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
    ch = ch[ch["game_id"].isin(valid_gids) & ch["period"].between(1, 3)]
    if len(ch):
        parts.append(ch)
shifts = pd.concat(parts, ignore_index=True)
del parts
print(f"  shifts: {len(shifts):,}")
shifts_by_game = dict(tuple(shifts.groupby("game_id")))

# ---- Aggregator: on-ice AGAINST per (player, zone, band) ----
sa_attempts = defaultdict(int)   # (pid, zone, band) -> count
sa_goals    = defaultdict(int)
absy_sum_z  = defaultdict(float) # (pid, zone) -> sum |y|
absy_n_z    = defaultdict(int)

print("Per-game shot-shift join (on-ice AGAINST)...")
n_games = 0
for gid, gshots in shots_by_game.items():
    n_games += 1
    if n_games % 500 == 0:
        print(f"  {n_games} games...")
    gshifts = shifts_by_game.get(gid)
    if gshifts is None:
        continue

    shifts_by_team = {}
    for team_ab, tsh in gshifts.groupby("team_abbrev"):
        st = tsh["abs_start_secs"].values.astype(np.int32)
        en = tsh["abs_end_secs"].values.astype(np.int32)
        pids = tsh["player_id"].values.astype(np.int64)
        order = np.argsort(st)
        shifts_by_team[team_ab] = (st[order], en[order], pids[order])

    for r in gshots.itertuples(index=False):
        t = int(r.abs_time)
        zone = r.zone
        band = r.band
        is_goal = int(r.is_goal)
        absy = float(r.abs_y)
        shoot_ab = r.shooting_team_abbrev
        # Defending team = the OTHER team
        if shoot_ab == r.home_team_abbrev:
            def_ab = r.away_team_abbrev
        else:
            def_ab = r.home_team_abbrev
        if def_ab not in shifts_by_team:
            continue
        st, en, pids = shifts_by_team[def_ab]
        idx_max = np.searchsorted(st, t, side="right")
        if idx_max == 0:
            continue
        on_mask = en[:idx_max] > t
        on_pids = pids[:idx_max][on_mask]
        for pid in on_pids:
            if pos_map.get(int(pid)) != "F":
                continue
            key = (int(pid), zone, band)
            sa_attempts[key] += 1
            sa_goals[key] += is_goal
            absy_sum_z[(int(pid), zone)] += absy
            absy_n_z[(int(pid), zone)]   += 1

print("Aggregation done.")

# ---- Build per-forward summary ----
print("Building summary...")
all_forwards = sorted({pid for (pid, _, _) in sa_attempts.keys()})

rows = []
for pid in all_forwards:
    toi_min = toi_es_min.get(pid, 0.0)
    if toi_min < MIN_ES_TOI_MIN:
        continue
    rec = {"player_id": pid,
           "player_name": name_map.get(pid, ""),
           "es_toi_min": round(toi_min, 2)}
    total_att = 0
    total_weighted = 0.0
    for zone in ["CNFI","MNFI","FNFI"]:
        zone_att = 0
        zone_w = 0.0
        for band in BAND_ORDER:
            n = sa_attempts.get((pid, zone, band), 0)
            zone_att += n
            zone_w += n * WEIGHTS[zone].get(band, 0.0)
        sa, lo, hi = rate_ci(zone_att, toi_min)
        rec[f"{zone}_SA"]      = zone_att
        rec[f"{zone}_SA_per60"] = round(sa, 4)
        rec[f"{zone}_SA_lo95"]  = round(lo, 4)
        rec[f"{zone}_SA_hi95"]  = round(hi, 4)
        rec[f"{zone}_centrality"] = (round(absy_sum_z.get((pid, zone), 0.0)
                                           / absy_n_z.get((pid, zone), 1), 3)
                                     if absy_n_z.get((pid, zone), 0) else np.nan)
        total_att += zone_att
        total_weighted += zone_w
    sa, lo, hi = rate_ci(total_att, toi_min)
    rec["TNFI_SA"]        = total_att
    rec["TNFI_SA_per60"]   = round(sa, 4)
    rec["TNFI_SA_lo95"]    = round(lo, 4)
    rec["TNFI_SA_hi95"]    = round(hi, 4)
    rec["P2_weighted"]     = round(total_weighted / toi_min * 60.0, 4)
    rows.append(rec)

p2 = pd.DataFrame(rows)

# ---- Ranking & flag ----
p2["TNFI_rank_raw"]     = p2["TNFI_SA_per60"].rank(ascending=True, method="min").astype(int)
p2["P2_weighted_rank"]  = p2["P2_weighted"].rank(ascending=True, method="min").astype(int)
p2["rank_delta"]        = p2["TNFI_rank_raw"] - p2["P2_weighted_rank"]
p2["weighting_flag"]    = np.where(
    p2["rank_delta"].abs() > 10,
    np.where(p2["rank_delta"] > 0, "DANGER_HEAVY",  # weighted rank is BETTER than raw -> shots allowed are LOW-DANGER
             "DANGER_LOW"),                          # weighted rank WORSE than raw -> shots allowed are HIGH-DANGER
    ""
)
# Naming clarification:
#   rank_delta > 0  => P2_weighted_rank < TNFI_rank_raw (better when weighted)
#                      means the forward's allowed shots tend to be LOW-danger y-bands
#                      -> raw-volume ranks were too harsh.  We label "LOW_DANGER_MIX".
#   rank_delta < 0  => weighting made them rank worse
#                      shots allowed tend to be HIGH-danger central
#                      -> "HIGH_DANGER_MIX".
p2["weighting_flag"] = np.where(
    p2["rank_delta"].abs() > 10,
    np.where(p2["rank_delta"] > 0, "LOW_DANGER_MIX", "HIGH_DANGER_MIX"),
    ""
)

p2 = p2.sort_values("TNFI_SA_per60", ascending=True).reset_index(drop=True)

# Column order
p2_cols = ["player_id","player_name","es_toi_min",
           "CNFI_SA","CNFI_SA_per60","CNFI_SA_lo95","CNFI_SA_hi95","CNFI_centrality",
           "MNFI_SA","MNFI_SA_per60","MNFI_SA_lo95","MNFI_SA_hi95","MNFI_centrality",
           "FNFI_SA","FNFI_SA_per60","FNFI_SA_lo95","FNFI_SA_hi95","FNFI_centrality",
           "TNFI_SA","TNFI_SA_per60","TNFI_SA_lo95","TNFI_SA_hi95",
           "P2_weighted",
           "TNFI_rank_raw","P2_weighted_rank","rank_delta","weighting_flag"]
p2 = p2[p2_cols]
p2.to_csv(OUT_P2, index=False)
print(f"\nWrote {OUT_P2}  ({len(p2)} forwards)")

# ---- Two-way score ----
p1a = pd.read_csv(P1A_CSV)
p1a["P1a_weighted_total"] = p1a["P1a_weighted_CNFI"] + p1a["P1a_weighted_MNFI"]
tw = p2.merge(p1a[["player_id","P1a_weighted_CNFI","P1a_weighted_MNFI",
                   "P1a_weighted_total"]], on="player_id", how="inner")

def z(s):
    return (s - s.mean()) / s.std(ddof=0)

tw["z_P1a_weighted"] = z(tw["P1a_weighted_total"])
tw["z_P2_weighted"]  = z(tw["P2_weighted"])
tw["twoway_score"]   = tw["z_P1a_weighted"] - tw["z_P2_weighted"]

tw["off_rank"]    = tw["P1a_weighted_total"].rank(ascending=False, method="min").astype(int)
tw["def_rank"]    = tw["P2_weighted"].rank(ascending=True, method="min").astype(int)
tw["twoway_rank"] = tw["twoway_score"].rank(ascending=False, method="min").astype(int)

tw = tw.sort_values("twoway_score", ascending=False).reset_index(drop=True)

tw_cols = ["player_id","player_name","es_toi_min",
           "P1a_weighted_CNFI","P1a_weighted_MNFI","P1a_weighted_total",
           "P2_weighted",
           "z_P1a_weighted","z_P2_weighted","twoway_score",
           "off_rank","def_rank","twoway_rank"]
tw = tw[tw_cols]
tw.to_csv(OUT_TW, index=False)
print(f"Wrote {OUT_TW}  ({len(tw)} forwards)\n")

# ---- Console summary ----
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.max_columns = None
pd.options.display.width = 220

print("--- Top 25 best defenders (lowest TNFI SA/60) ---")
print(p2.head(25)[["player_name","es_toi_min","TNFI_SA","TNFI_SA_per60",
                   "TNFI_SA_lo95","TNFI_SA_hi95","P2_weighted",
                   "TNFI_rank_raw","P2_weighted_rank","rank_delta","weighting_flag"]]
      .to_string(index=False))

print("\n--- Worst 15 defenders (highest TNFI SA/60) ---")
print(p2.tail(15)[["player_name","TNFI_SA_per60","TNFI_SA_lo95","TNFI_SA_hi95",
                   "P2_weighted","TNFI_rank_raw","P2_weighted_rank",
                   "rank_delta","weighting_flag"]].to_string(index=False))

print("\n--- HIGH_DANGER_MIX (raw rank flatters them; weighted exposes high-danger allowed) ---")
hd = p2[p2["weighting_flag"]=="HIGH_DANGER_MIX"].sort_values("rank_delta")
print(hd[["player_name","TNFI_SA_per60","P2_weighted","CNFI_SA","CNFI_centrality",
          "TNFI_rank_raw","P2_weighted_rank","rank_delta"]].head(20).to_string(index=False))

print("\n--- LOW_DANGER_MIX (raw rank is harsh; weighted credits low-danger allowed) ---")
ld = p2[p2["weighting_flag"]=="LOW_DANGER_MIX"].sort_values("rank_delta", ascending=False)
print(ld[["player_name","TNFI_SA_per60","P2_weighted","CNFI_SA","CNFI_centrality",
          "TNFI_rank_raw","P2_weighted_rank","rank_delta"]].head(20).to_string(index=False))

print("\n--- Top 25 two-way forwards ---")
print(tw.head(25)[["player_name","es_toi_min","P1a_weighted_total","P2_weighted",
                   "z_P1a_weighted","z_P2_weighted","twoway_score",
                   "off_rank","def_rank","twoway_rank"]].to_string(index=False))

print("\n--- Bottom 15 two-way forwards ---")
print(tw.tail(15)[["player_name","P1a_weighted_total","P2_weighted",
                   "z_P1a_weighted","z_P2_weighted","twoway_score",
                   "off_rank","def_rank","twoway_rank"]].to_string(index=False))
