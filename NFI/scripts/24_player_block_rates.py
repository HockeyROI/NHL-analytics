#!/usr/bin/env python3
"""
player_block_rates.csv — standalone reference data for personal shot blocks.

Sits alongside spatial pillars; NOT included in two-way scores or composite NFI.
Designed for the acquisition framework and Streamlit display layer.

Method:
  - Filter blocked-shot events to ES (situation_code=1551), regulation periods
    1-3, 5 seasons pooled (2021-22 .. 2025-26), with valid blocker_player_id.
  - Apply coordinate fix: x_coord_norm = abs(x_coord_norm),
                          y_coord_norm = abs(y_coord_norm)
    (corrects upstream sign-flip bug in blocked-shot rows).
  - Classify each block by tight defensive zone of the *blocked attempt*:
      CNFI: x in [74,89] AND |y| <= 9
      MNFI: x in [55,73] AND |y| <= 15
      FNFI: x in [25,54] AND |y| <= 15
  - Aggregate blocks per (blocker_player_id, zone).
  - Rate per 60 ES TOI; Wilson 95% CI on each rate.
  - blocks_xG_prevented_per60 weights each block by zone-level CORSI
    conversion rate from zone_conversion_rates.csv (per-attempt denominator,
    appropriate for blocks since a block IS an attempt):
      CNFI 13.075%, MNFI 10.020%, FNFI 3.370%

Min 500 ES TOI minutes per player. Includes both forwards and defensemen.

Output: NFI/output/player_block_rates.csv
"""
import math
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
TOI_CSV  = f"{OUT}/player_toi.csv"
POS_CSV  = f"{OUT}/player_positions.csv"

SEASONS = {"20212022","20222023","20232024","20242025","20252026"}
MIN_ES_TOI_MIN = 500.0

# Zone-level conversion rates (per-attempt / Corsi denominator) from
# zone_conversion_rates.csv. A block IS an attempt, so the per-attempt rate
# is the dimensionally correct weight.
ZONE_RATE = {"CNFI": 0.13075, "MNFI": 0.10020, "FNFI": 0.03370}

def classify(x, absy):
    if 74 <= x <= 89 and absy <= 9:   return "CNFI"
    if 55 <= x <= 73 and absy <= 15:  return "MNFI"
    if 25 <= x <= 54 and absy <= 15:  return "FNFI"
    return None

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0.0, c-h), min(1.0, c+h))

def rate_ci(events, minutes, z=1.96):
    """Per-60 rate with Wilson-style CI (matches existing pipeline convention)."""
    if minutes <= 0:
        return 0.0, 0.0, 0.0
    p, lo, hi = wilson(events, max(events, int(round(minutes))))
    return events / minutes * 60.0, lo * 60.0, hi * 60.0

# ---- Lookups ----
print("Loading positions / TOI ...")
pos_df = pd.read_csv(POS_CSV, dtype={"player_id":int})
pos_grp  = dict(zip(pos_df["player_id"], pos_df["pos_group"]))
pos_role = dict(zip(pos_df["player_id"], pos_df["position"]))
name_map = dict(zip(pos_df["player_id"], pos_df["player_name"]))
toi_df = pd.read_csv(TOI_CSV, dtype={"player_id":int})
toi_es_min = dict(zip(toi_df["player_id"], toi_df["toi_ES_sec"]/60.0))

# ---- Load blocked-shot events ----
print("Loading blocked-shot events ...")
cols = ["game_id","season","period","event_type","situation_code",
        "blocker_player_id","x_coord_norm","y_coord_norm"]
blk = pd.read_csv(SHOT_CSV, usecols=cols, dtype={"season":str,"situation_code":str})
blk = blk[blk["season"].isin(SEASONS)].copy()
blk = blk[blk["period"].between(1,3)].copy()
blk = blk[blk["event_type"]=="blocked-shot"].copy()
blk = blk[blk["situation_code"].astype(str)=="1551"].copy()
blk = blk.dropna(subset=["blocker_player_id"])
blk["blocker_player_id"] = blk["blocker_player_id"].astype(int)
print(f"  blocked-shot events (ES, reg, w/ blocker): {len(blk):,}")

# ---- Apply coordinate fix and classify ----
blk["x_coord_norm"] = blk["x_coord_norm"].abs()
blk["y_coord_norm"] = blk["y_coord_norm"].abs()
blk["abs_y"] = blk["y_coord_norm"]
blk["zone"] = [classify(x, ay) for x, ay in zip(blk["x_coord_norm"].values,
                                                  blk["abs_y"].values)]
blk_in_zone = blk[blk["zone"].notna()].copy()
print(f"  blocks in tight CNFI/MNFI/FNFI: {len(blk_in_zone):,}")
print(f"    CNFI: {(blk_in_zone['zone']=='CNFI').sum():,}")
print(f"    MNFI: {(blk_in_zone['zone']=='MNFI').sum():,}")
print(f"    FNFI: {(blk_in_zone['zone']=='FNFI').sum():,}")

# ---- Aggregate per (blocker, zone) ----
agg = blk_in_zone.groupby(["blocker_player_id","zone"]).size().reset_index(name="blocks")
wide = agg.pivot_table(index="blocker_player_id", columns="zone",
                        values="blocks", aggfunc="sum",
                        fill_value=0).reset_index().rename(
    columns={"blocker_player_id":"player_id"})
for z in ["CNFI","MNFI","FNFI"]:
    if z not in wide.columns: wide[z] = 0
wide = wide.rename(columns={"CNFI":"CNFI_blocks","MNFI":"MNFI_blocks",
                              "FNFI":"FNFI_blocks"})
wide["TNFI_blocks"] = wide["CNFI_blocks"] + wide["MNFI_blocks"] + wide["FNFI_blocks"]

# ---- Annotate metadata, filter min TOI ----
wide["player_name"] = wide["player_id"].map(name_map)
wide["pos_group"]   = wide["player_id"].map(pos_grp)
wide["position"]    = wide["player_id"].map(pos_role)
wide["es_toi_min"]  = wide["player_id"].map(toi_es_min)
qual = wide[(wide["es_toi_min"].notna()) &
            (wide["es_toi_min"] >= MIN_ES_TOI_MIN)].copy()
print(f"\nQualifying players (≥{MIN_ES_TOI_MIN:.0f} ES min): {len(qual)}")

# ---- Compute per-60 rates with Wilson CIs ----
def add_rate_block(df, prefix, count_col):
    rs, los, his = [], [], []
    for n, t in zip(df[count_col].values, df["es_toi_min"].values):
        r, lo, hi = rate_ci(int(n), float(t))
        rs.append(round(r, 4)); los.append(round(lo, 4)); his.append(round(hi, 4))
    df[f"{prefix}_per60"]    = rs
    df[f"{prefix}_lo95"]     = los
    df[f"{prefix}_hi95"]     = his

for zone in ["CNFI","MNFI","FNFI","TNFI"]:
    add_rate_block(qual, f"{zone}_blocks", f"{zone}_blocks")

# ---- xG prevented per 60 (zone-rate weighted, point estimate) ----
qual["blocks_xG_prevented_total"] = (qual["CNFI_blocks"]*ZONE_RATE["CNFI"]
                                      + qual["MNFI_blocks"]*ZONE_RATE["MNFI"]
                                      + qual["FNFI_blocks"]*ZONE_RATE["FNFI"])
qual["blocks_xG_prevented_per60"] = (qual["blocks_xG_prevented_total"] /
                                      qual["es_toi_min"] * 60.0).round(4)

# ---- Final sort and column order ----
qual = qual.sort_values("TNFI_blocks_per60", ascending=False).reset_index(drop=True)
qual["TNFI_blocks_per60_rank"]      = qual["TNFI_blocks_per60"].rank(
    ascending=False, method="min").astype(int)
qual["xG_prevented_per60_rank"]     = qual["blocks_xG_prevented_per60"].rank(
    ascending=False, method="min").astype(int)

cols = ["player_id","player_name","position","pos_group","es_toi_min",
        "CNFI_blocks","CNFI_blocks_per60","CNFI_blocks_lo95","CNFI_blocks_hi95",
        "MNFI_blocks","MNFI_blocks_per60","MNFI_blocks_lo95","MNFI_blocks_hi95",
        "FNFI_blocks","FNFI_blocks_per60","FNFI_blocks_lo95","FNFI_blocks_hi95",
        "TNFI_blocks","TNFI_blocks_per60","TNFI_blocks_lo95","TNFI_blocks_hi95",
        "blocks_xG_prevented_total","blocks_xG_prevented_per60",
        "TNFI_blocks_per60_rank","xG_prevented_per60_rank"]
qual = qual[cols]
qual.to_csv(f"{OUT}/player_block_rates.csv", index=False)
print(f"\nWrote {OUT}/player_block_rates.csv  ({len(qual)} players)")

# ---- Reporting ----
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.width = 240
pd.options.display.max_columns = None

print("\n=== TOP 20 by TNFI blocks per 60 (any position) ===")
top_tnfi = qual.head(20)
print(top_tnfi[["player_name","position","es_toi_min","TNFI_blocks",
                "TNFI_blocks_per60","TNFI_blocks_lo95","TNFI_blocks_hi95",
                "blocks_xG_prevented_per60"]].to_string(index=False))

print("\n=== TOP 20 by blocks xG prevented per 60 ===")
top_xg = qual.sort_values("blocks_xG_prevented_per60", ascending=False).head(20)
print(top_xg[["player_name","position","es_toi_min","TNFI_blocks",
              "TNFI_blocks_per60","blocks_xG_prevented_per60",
              "CNFI_blocks_per60","MNFI_blocks_per60","FNFI_blocks_per60"]]
      .to_string(index=False))

# Position breakouts
print("\n=== TOP 10 D by xG prevented per 60 ===")
d_top = qual[qual["pos_group"]=="D"].sort_values(
    "blocks_xG_prevented_per60", ascending=False).head(10)
print(d_top[["player_name","es_toi_min","TNFI_blocks_per60",
             "blocks_xG_prevented_per60","CNFI_blocks_per60",
             "MNFI_blocks_per60","FNFI_blocks_per60"]].to_string(index=False))

print("\n=== TOP 10 F by xG prevented per 60 ===")
f_top = qual[qual["pos_group"]=="F"].sort_values(
    "blocks_xG_prevented_per60", ascending=False).head(10)
print(f_top[["player_name","es_toi_min","TNFI_blocks_per60",
             "blocks_xG_prevented_per60","CNFI_blocks_per60",
             "MNFI_blocks_per60","FNFI_blocks_per60"]].to_string(index=False))

# Position-level summary
print("\n=== Distribution by position ===")
print(qual.groupby("pos_group").agg(
    n=("player_id","count"),
    mean_TNFI_blocks_per60=("TNFI_blocks_per60","mean"),
    median_TNFI_blocks_per60=("TNFI_blocks_per60","median"),
    p90_TNFI_blocks_per60=("TNFI_blocks_per60", lambda s: s.quantile(0.90)),
    mean_xG_prevented_per60=("blocks_xG_prevented_per60","mean"),
).round(3).to_string())
