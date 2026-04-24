#!/usr/bin/env python3
"""
Build P2_net (forwards) and P4_net (defensemen) — Fenwick weighted shots-against
per 60 with a CREDIT for spatially-weighted personal blocks.

  P2_net = P2_weighted_Fenwick  -  personal_blocks_xG_per60  (forwards)
  P4_net = P4_weighted_Fenwick  -  personal_blocks_xG_per60  (D)

Both terms are in xG/60 units (expected-goals against / prevented per 60),
so the subtraction is dimensionally consistent.

Blocks are credited using corrected coordinates: for blocked-shot events,
x_coord_norm = abs(x_coord_norm) and y_coord_norm = abs(y_coord_norm) (fixes
the upstream sign-flip bug). The block's spatial weight uses the same y-band
conversion-rate weights as P2 / P4 (CNFI / MNFI / FNFI per-band rates from
the league-wide y-band analysis).

Two-way scores are rebuilt using net values:
  two-way F = z(P1a_weighted_total) - z(P2_net)
  two-way D = z(P5_weighted)        - z(P4_net)

Outputs (NFI/output):
  P2_net.csv
  P4_net.csv
  twoway_forward_score.csv  (overwritten)
  twoway_D_score.csv         (overwritten)
"""
import math
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
BK   = f"{OUT}/_pre_corsi_backup"  # holds Fenwick P2/P4 from before Corsi rerun
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
TOI_CSV  = f"{OUT}/player_toi.csv"
POS_CSV  = f"{OUT}/player_positions.csv"

SEASONS = {"20212022","20222023","20232024","20242025","20252026"}

# Same y-band weights as P2/P4
WEIGHTS_TIGHT = {
    "CNFI": {"0-5":0.1579, "5-10":0.1075},
    "MNFI": {"0-5":0.1109, "5-10":0.0985, "10-15":0.0887},
    "FNFI": {"0-5":0.0367, "5-10":0.0335, "10-15":0.0289},
}

def y_band_tight(absy):
    if absy < 5:  return "0-5"
    if absy < 10: return "5-10"
    return "10-15"

def classify_tight(x, absy):
    if 74 <= x <= 89 and absy <= 9:   return "CNFI"
    if 55 <= x <= 73 and absy <= 15:  return "MNFI"
    if 25 <= x <= 54 and absy <= 15:  return "FNFI"
    return None

# ---- Lookups ----
print("Loading positions / TOI ...")
pos_df = pd.read_csv(POS_CSV, dtype={"player_id":int})
pos_grp = dict(zip(pos_df["player_id"], pos_df["pos_group"]))
name_map = dict(zip(pos_df["player_id"], pos_df["player_name"]))
toi_df = pd.read_csv(TOI_CSV, dtype={"player_id":int})
toi_es_min = dict(zip(toi_df["player_id"], toi_df["toi_ES_sec"]/60.0))

# ---- Load Fenwick P2 and P4 from backup (these have P2_weighted / P4_weighted
#      computed under SOG+missed+goal denominator) ----
print("Loading backed-up Fenwick P2 and P4 ...")
p2_fenwick = pd.read_csv(f"{BK}/P2_defensive_forwards.csv")
p4_fenwick = pd.read_csv(f"{BK}/P4_defensive_D.csv")
print(f"  Fenwick P2: {len(p2_fenwick)} F  |  Fenwick P4: {len(p4_fenwick)} D")

# ---- Load blocked-shot events with corrected coords ----
print("Loading blocked-shot events ...")
cols = ["game_id","season","period","event_type","situation_code","time_secs",
        "shooting_team_id","shooting_team_abbrev","blocker_player_id",
        "x_coord_norm","y_coord_norm","is_goal"]
blk = pd.read_csv(SHOT_CSV, usecols=cols, dtype={"season":str,"situation_code":str})
blk = blk[blk["season"].isin(SEASONS)].copy()
blk = blk[blk["period"].between(1,3)].copy()
blk = blk[blk["event_type"]=="blocked-shot"].copy()
blk = blk[blk["situation_code"].astype(str)=="1551"].copy()
blk = blk.dropna(subset=["blocker_player_id"])
blk["blocker_player_id"] = blk["blocker_player_id"].astype(int)
print(f"  blocked-shot events (ES, regulation, w/ blocker): {len(blk):,}")

# Apply abs() correction
blk["x_coord_norm"] = blk["x_coord_norm"].abs()
blk["y_coord_norm"] = blk["y_coord_norm"].abs()
blk["abs_y"] = blk["y_coord_norm"]
# Classify into tight zones
blk["zone"] = [classify_tight(x, ay) for x, ay in zip(blk["x_coord_norm"].values,
                                                        blk["abs_y"].values)]
blk = blk[blk["zone"].notna()].copy()
blk["band"] = blk["abs_y"].apply(y_band_tight)
print(f"  blocks in tight CNFI/MNFI/FNFI zones: {len(blk):,}")
print(f"    CNFI: {(blk['zone']=='CNFI').sum():,}")
print(f"    MNFI: {(blk['zone']=='MNFI').sum():,}")
print(f"    FNFI: {(blk['zone']=='FNFI').sum():,}")

# ---- Aggregate personal blocks per (player, zone, band) ----
agg = blk.groupby(["blocker_player_id","zone","band"]).size().reset_index(name="blocks")
# Apply y-band weighting -> personal blocks xG-prevented per shift
agg["weight"] = agg.apply(lambda r: WEIGHTS_TIGHT[r["zone"]].get(r["band"], 0.0), axis=1)
agg["xg_prevented"] = agg["blocks"] * agg["weight"]

# Sum to player-level
player_xg = agg.groupby("blocker_player_id").agg(
    blocks_total=("blocks","sum"),
    xg_prevented_total=("xg_prevented","sum"),
).reset_index().rename(columns={"blocker_player_id":"player_id"})

# Per-zone breakdowns for reporting
zone_blocks = agg.pivot_table(index="blocker_player_id", columns="zone",
                               values="blocks", aggfunc="sum",
                               fill_value=0).reset_index().rename(
    columns={"blocker_player_id":"player_id"})
for z in ["CNFI","MNFI","FNFI"]:
    if z not in zone_blocks.columns: zone_blocks[z] = 0
zone_blocks = zone_blocks.rename(columns={"CNFI":"blocks_CNFI",
                                          "MNFI":"blocks_MNFI",
                                          "FNFI":"blocks_FNFI"})
player_xg = player_xg.merge(zone_blocks[["player_id","blocks_CNFI",
                                          "blocks_MNFI","blocks_FNFI"]],
                             on="player_id", how="left").fillna(0)

# Per-60 — blocks rate and xG-prevented rate
player_xg["es_toi_min"] = player_xg["player_id"].map(toi_es_min)
player_xg["blocks_per60"]      = player_xg["blocks_total"]      / player_xg["es_toi_min"] * 60.0
player_xg["blocks_xG_per60"]   = player_xg["xg_prevented_total"]/ player_xg["es_toi_min"] * 60.0
player_xg["blocks_xG_per60"]   = player_xg["blocks_xG_per60"].round(4)
player_xg["blocks_per60"]      = player_xg["blocks_per60"].round(3)

# ---- Build P2_net (forwards) ----
print("Building P2_net ...")
p2 = p2_fenwick.merge(player_xg[["player_id","blocks_total","blocks_CNFI",
                                  "blocks_MNFI","blocks_FNFI",
                                  "blocks_per60","blocks_xG_per60"]],
                       on="player_id", how="left").fillna(
    {"blocks_total":0, "blocks_CNFI":0, "blocks_MNFI":0, "blocks_FNFI":0,
     "blocks_per60":0.0, "blocks_xG_per60":0.0})
p2["P2_net"] = (p2["P2_weighted"] - p2["blocks_xG_per60"]).round(4)
p2["P2_net_rank"]    = p2["P2_net"].rank(ascending=True, method="min").astype(int)
p2["P2_weighted_rank"]= p2["P2_weighted"].rank(ascending=True, method="min").astype(int)
p2["rank_delta_net_vs_weighted"] = p2["P2_weighted_rank"] - p2["P2_net_rank"]
p2 = p2.sort_values("P2_net").reset_index(drop=True)

p2_cols = ["player_id","player_name","es_toi_min",
           "TNFI_SA","TNFI_SA_per60","P2_weighted",
           "blocks_total","blocks_CNFI","blocks_MNFI","blocks_FNFI",
           "blocks_per60","blocks_xG_per60",
           "P2_net","P2_weighted_rank","P2_net_rank","rank_delta_net_vs_weighted"]
p2_out = p2[p2_cols]
p2_out.to_csv(f"{OUT}/P2_net.csv", index=False)
print(f"  P2_net.csv: {len(p2_out)} forwards")

# ---- Build P4_net (D) ----
print("Building P4_net ...")
p4 = p4_fenwick.merge(player_xg[["player_id","blocks_total","blocks_CNFI",
                                  "blocks_MNFI","blocks_FNFI",
                                  "blocks_per60","blocks_xG_per60"]],
                       on="player_id", how="left").fillna(
    {"blocks_total":0, "blocks_CNFI":0, "blocks_MNFI":0, "blocks_FNFI":0,
     "blocks_per60":0.0, "blocks_xG_per60":0.0})
p4["P4_net"] = (p4["P4_weighted"] - p4["blocks_xG_per60"]).round(4)
p4["P4_net_rank"]      = p4["P4_net"].rank(ascending=True, method="min").astype(int)
p4["P4_weighted_rank"] = p4["P4_weighted"].rank(ascending=True, method="min").astype(int)
p4["rank_delta_net_vs_weighted"] = p4["P4_weighted_rank"] - p4["P4_net_rank"]
p4 = p4.sort_values("P4_net").reset_index(drop=True)

p4_cols = ["player_id","player_name","es_toi_min",
           "TNFI_SA","TNFI_SA_per60","P4_weighted",
           "blocks_total","blocks_CNFI","blocks_MNFI","blocks_FNFI",
           "blocks_per60","blocks_xG_per60",
           "P4_net","P4_weighted_rank","P4_net_rank","rank_delta_net_vs_weighted"]
p4_out = p4[p4_cols]
p4_out.to_csv(f"{OUT}/P4_net.csv", index=False)
print(f"  P4_net.csv: {len(p4_out)} D")

# ---- Two-way scores using P2_net / P4_net ----
print("Rebuilding two-way scores ...")
# Use current (Corsi) P1a and P5 — those use blocks legitimately as offensive attempts
p1a = pd.read_csv(f"{OUT}/P1a_centrality_weighted.csv")
p5  = pd.read_csv(f"{OUT}/P5_offensive_D.csv")

p1a["P1a_weighted_total"] = p1a["P1a_weighted_CNFI"] + p1a["P1a_weighted_MNFI"]
twf = p2_out[["player_id","player_name","es_toi_min","P2_weighted","P2_net",
              "blocks_total","blocks_xG_per60"]].merge(
    p1a[["player_id","P1a_weighted_CNFI","P1a_weighted_MNFI",
         "P1a_weighted_total"]], on="player_id", how="inner")

def z(s): return (s - s.mean()) / s.std(ddof=0)
twf["z_P1a_weighted"] = z(twf["P1a_weighted_total"])
twf["z_P2_net"]       = z(twf["P2_net"])
twf["twoway_score"]   = twf["z_P1a_weighted"] - twf["z_P2_net"]
twf["off_rank"]       = twf["P1a_weighted_total"].rank(ascending=False, method="min").astype(int)
twf["def_rank"]       = twf["P2_net"].rank(ascending=True,  method="min").astype(int)
twf["twoway_rank"]    = twf["twoway_score"].rank(ascending=False, method="min").astype(int)
twf = twf.sort_values("twoway_score", ascending=False).reset_index(drop=True)

twf_cols = ["player_id","player_name","es_toi_min",
            "P1a_weighted_CNFI","P1a_weighted_MNFI","P1a_weighted_total",
            "P2_weighted","blocks_total","blocks_xG_per60","P2_net",
            "z_P1a_weighted","z_P2_net","twoway_score",
            "off_rank","def_rank","twoway_rank"]
twf = twf[twf_cols]
twf.to_csv(f"{OUT}/twoway_forward_score.csv", index=False)
print(f"  twoway_forward_score.csv: {len(twf)} F")

twd = p4_out[["player_id","player_name","es_toi_min","P4_weighted","P4_net",
              "blocks_total","blocks_xG_per60"]].merge(
    p5[["player_id","TNFI_SF_per60","P5_weighted"]], on="player_id", how="inner")
twd["z_P5_weighted"]  = z(twd["P5_weighted"])
twd["z_P4_net"]       = z(twd["P4_net"])
twd["twoway_D_score"] = twd["z_P5_weighted"] - twd["z_P4_net"]
twd["off_rank"]       = twd["P5_weighted"].rank(ascending=False, method="min").astype(int)
twd["def_rank"]       = twd["P4_net"].rank(ascending=True,  method="min").astype(int)
twd["twoway_D_rank"]  = twd["twoway_D_score"].rank(ascending=False, method="min").astype(int)
twd = twd.sort_values("twoway_D_score", ascending=False).reset_index(drop=True)
twd_cols = ["player_id","player_name","es_toi_min",
            "TNFI_SF_per60","P5_weighted",
            "P4_weighted","blocks_total","blocks_xG_per60","P4_net",
            "z_P5_weighted","z_P4_net","twoway_D_score",
            "off_rank","def_rank","twoway_D_rank"]
twd = twd[twd_cols]
twd.to_csv(f"{OUT}/twoway_D_score.csv", index=False)
print(f"  twoway_D_score.csv: {len(twd)} D")

# ===========================================================
# Reporting
# ===========================================================
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.width = 240
pd.options.display.max_columns = None

print("\n=== TOP 20 DEFENSIVE FORWARDS by P2_net (lower = better) ===")
print(p2_out.head(20)[["player_name","es_toi_min","TNFI_SA_per60",
                       "P2_weighted","blocks_total","blocks_xG_per60",
                       "P2_net","P2_weighted_rank","P2_net_rank",
                       "rank_delta_net_vs_weighted"]].to_string(index=False))

print("\n=== TOP 20 DEFENSIVE D by P4_net (lower = better) ===")
print(p4_out.head(20)[["player_name","es_toi_min","TNFI_SA_per60",
                       "P4_weighted","blocks_total","blocks_xG_per60",
                       "P4_net","P4_weighted_rank","P4_net_rank",
                       "rank_delta_net_vs_weighted"]].to_string(index=False))

print("\n=== Biggest P2_net IMPROVERS vs P2_weighted (most credit from blocks) ===")
imp = p2_out.sort_values("rank_delta_net_vs_weighted", ascending=False).head(15)
print(imp[["player_name","P2_weighted","blocks_xG_per60","P2_net",
           "P2_weighted_rank","P2_net_rank","rank_delta_net_vs_weighted"]].to_string(index=False))

print("\n=== Biggest P4_net IMPROVERS vs P4_weighted (most credit from blocks) ===")
imp4 = p4_out.sort_values("rank_delta_net_vs_weighted", ascending=False).head(15)
print(imp4[["player_name","P4_weighted","blocks_xG_per60","P4_net",
            "P4_weighted_rank","P4_net_rank","rank_delta_net_vs_weighted"]].to_string(index=False))

print("\n=== Spot-checks: named players ===")
named_F = ["Aleksander Barkov","Artturi Lehkonen","Nathan MacKinnon",
           "Sam Bennett","Sam Reinhart","Carter Verhaeghe",
           "Anthony Cirelli","Eetu Luostarinen","Patrice Bergeron"]
named_D = ["Chris Tanev","Brock Faber","Cale Makar","Devon Toews",
           "Aaron Ekblad","Gustav Forsling","Jaccob Slavin",
           "Quinn Hughes","Brent Burns"]

print("\n--- Forwards ---")
for n in named_F:
    r = p2_out[p2_out["player_name"]==n]
    if len(r)==0:
        print(f"  {n}: not in P2_net")
        continue
    r = r.iloc[0]
    print(f"  {n}: P2_w {r['P2_weighted']:.4f}  blocks_xG/60 {r['blocks_xG_per60']:.4f}"
          f"  P2_net {r['P2_net']:.4f}  ranks {int(r['P2_weighted_rank'])} -> {int(r['P2_net_rank'])}"
          f"  (Δ {int(r['rank_delta_net_vs_weighted']):+d})")

print("\n--- D ---")
for n in named_D:
    r = p4_out[p4_out["player_name"]==n]
    if len(r)==0:
        print(f"  {n}: not in P4_net")
        continue
    r = r.iloc[0]
    print(f"  {n}: P4_w {r['P4_weighted']:.4f}  blocks_xG/60 {r['blocks_xG_per60']:.4f}"
          f"  P4_net {r['P4_net']:.4f}  ranks {int(r['P4_weighted_rank'])} -> {int(r['P4_net_rank'])}"
          f"  (Δ {int(r['rank_delta_net_vs_weighted']):+d})")

print("\n=== Updated TOP 20 TWO-WAY FORWARDS (using P2_net) ===")
print(twf.head(20)[["player_name","es_toi_min","P1a_weighted_total","P2_weighted",
                    "blocks_xG_per60","P2_net","z_P1a_weighted","z_P2_net",
                    "twoway_score","off_rank","def_rank","twoway_rank"]]
      .to_string(index=False))

print("\n=== Updated TOP 20 TWO-WAY D (using P4_net) ===")
print(twd.head(20)[["player_name","es_toi_min","P5_weighted","P4_weighted",
                    "blocks_xG_per60","P4_net","z_P5_weighted","z_P4_net",
                    "twoway_D_score","off_rank","def_rank","twoway_D_rank"]]
      .to_string(index=False))

print("\nFiles written:")
for f in ["P2_net.csv","P4_net.csv","twoway_forward_score.csv","twoway_D_score.csv"]:
    print(f"  {OUT}/{f}")
