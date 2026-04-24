#!/usr/bin/env python3
"""
FNFI downstream danger analysis:
  For ES-regulation shots in FNFI, MNFI, Wide zones (5-season pool),
  compute % producing within 2 seconds:
    (1) Tip / deflection follow-up (reb_shot_type in tip-in, deflected)
    (2) Any rebound follow-up (any shot event in rebound_sequences)
    (3) CNFI-zone follow-up (follow-up coords in CNFI box)

Rebound_sequences uses RAW coords; we classify CNFI via |x|>=74,|y|<=9.
"""
import math
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
SHOT = f"{ROOT}/Data/nhl_shot_events.csv"
REB  = f"{ROOT}/NFI/Geometry_post/Data/rebound_sequences.csv"
OUT  = f"{ROOT}/NFI/output/fnfi_downstream_analysis.csv"
SEASONS = {"20212022","20222023","20232024","20242025","20252026"}

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0.0,c-h), min(1.0,c+h))

# ---- load shots ----
print("Loading shot_events...")
cols = ["game_id","season","period","event_type","situation_code",
        "home_team_id","shooting_team_id","shooter_player_id",
        "x_coord","y_coord","x_coord_norm","y_coord_norm","is_goal"]
sh = pd.read_csv(SHOT, usecols=cols, dtype={"season":str,"situation_code":str})
sh = sh[sh["season"].isin(SEASONS) & sh["period"].between(1,3)].copy()
sh = sh[sh["event_type"].isin(["shot-on-goal","missed-shot","blocked-shot","goal"])].copy()
# drop rows missing coords/shooter
sh = sh.dropna(subset=["shooter_player_id","x_coord","y_coord","x_coord_norm","y_coord_norm"]).copy()

# ES state derivation from situation_code
sc = sh["situation_code"].astype(str).str.zfill(4)
ag = sc.str[0].astype(int); ask = sc.str[1].astype(int)
hsk = sc.str[2].astype(int); hg = sc.str[3].astype(int)
shoot_home = (sh["shooting_team_id"] == sh["home_team_id"])
import numpy as np
shk = np.where(shoot_home, hsk, ask)
opk = np.where(shoot_home, ask, hsk)
state = np.where(shk==opk, "ES", np.where(shk>opk, "PP", "PK"))
empty = (ag==0) | (hg==0)
sh = sh[(state=="ES") & ~empty].copy()
print(f"  ES regulation shots (5-season): {len(sh):,}")

# zone classification (normalized)
def classify(x, y):
    if -15 <= y <= 15:
        if 74 <= x <= 89 and -9 <= y <= 9: return "CNFI"
        if 55 <= x < 74:  return "MNFI"
        if 25 <= x < 55:  return "FNFI"
        return "lane_other"
    return "Wide"
sh["zone"] = [classify(x,y) for x,y in zip(sh["x_coord_norm"].astype(int), sh["y_coord_norm"].astype(int))]

# ---- load rebound_sequences (time_gap <= 2s, 5-season pool) ----
print("Loading rebound_sequences...")
rb = pd.read_csv(REB, dtype={"season":str})
rb = rb[rb["season"].isin(SEASONS) & rb["period"].between(1,3)].copy()
rb = rb[rb["time_gap_secs"] <= 2].copy()
print(f"  rebound pairs (time_gap<=2): {len(rb):,}")

# classify reb (follow-up) zone using RAW coords: CNFI = |x|>=74 & |x|<=89 & |y|<=9
rb["reb_abs_x"] = rb["reb_x"].abs()
rb["reb_abs_y"] = rb["reb_y"].abs()
rb["reb_in_CNFI"] = ((rb["reb_abs_x"]>=74) & (rb["reb_abs_x"]<=89) & (rb["reb_abs_y"]<=9)).astype(int)
rb["reb_is_tip"]  = rb["reb_shot_type"].isin(["tip-in","deflected"]).astype(int)

# Build flags per (game_id, orig_shooter_id, orig_x, orig_y): multiple follow-ups allowed.
# Use first-match per orig key for simplicity (any match = follow-up exists).
# For (1) tip flag, (2) any follow-up, (3) CNFI follow-up, aggregate via max.
orig_keys = rb.groupby(["game_id","orig_shooter_id","orig_x","orig_y"]).agg(
    any_followup=("reb_event_type", "size"),
    tip_followup=("reb_is_tip","max"),
    cnfi_followup=("reb_in_CNFI","max"),
).reset_index()
orig_keys["any_followup"] = (orig_keys["any_followup"]>0).astype(int)
orig_keys["tip_followup"] = orig_keys["tip_followup"].astype(int)
orig_keys["cnfi_followup"] = orig_keys["cnfi_followup"].astype(int)

# match into shots via (game_id, shooter_player_id, x_coord, y_coord)
sh["game_id"] = sh["game_id"].astype(int)
sh["shooter_player_id"] = sh["shooter_player_id"].astype(int)
sh["x_coord"] = sh["x_coord"].astype(int)
sh["y_coord"] = sh["y_coord"].astype(int)
orig_keys["game_id"] = orig_keys["game_id"].astype(int)
orig_keys["orig_shooter_id"] = orig_keys["orig_shooter_id"].astype(int)
orig_keys["orig_x"] = orig_keys["orig_x"].astype(int)
orig_keys["orig_y"] = orig_keys["orig_y"].astype(int)

m = sh.merge(
    orig_keys.rename(columns={"orig_shooter_id":"shooter_player_id","orig_x":"x_coord","orig_y":"y_coord"}),
    on=["game_id","shooter_player_id","x_coord","y_coord"], how="left"
)
m["any_followup"]  = m["any_followup"].fillna(0).astype(int)
m["tip_followup"]  = m["tip_followup"].fillna(0).astype(int)
m["cnfi_followup"] = m["cnfi_followup"].fillna(0).astype(int)

# ---- Aggregate by zone ----
rows = []
for z in ["FNFI","MNFI","Wide"]:
    sub = m[m["zone"]==z]
    n = len(sub)
    tip = int(sub["tip_followup"].sum())
    reb = int(sub["any_followup"].sum())
    cnfi = int(sub["cnfi_followup"].sum())
    for label, k in [("tip_or_deflection_within_2s", tip),
                     ("any_rebound_within_2s", reb),
                     ("CNFI_followup_within_2s", cnfi)]:
        p, lo, hi = wilson(k, n)
        rows.append({"zone": z, "metric": label, "shots": n, "events": k,
                     "pct": round(p*100, 3),
                     "ci_lo_pct": round(lo*100, 3),
                     "ci_hi_pct": round(hi*100, 3)})

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("\n=== Downstream rates by origin zone (ES regulation, 5-season pool) ===")
print(df.to_string(index=False))

# pivot for readability
piv = df.pivot(index="zone", columns="metric", values="pct").round(3)
piv = piv.reindex(["FNFI","MNFI","Wide"])
piv = piv[["tip_or_deflection_within_2s","any_rebound_within_2s","CNFI_followup_within_2s"]]
print("\n=== Pivot (%) ===")
print(piv.to_string())
