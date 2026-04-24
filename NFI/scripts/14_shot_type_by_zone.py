#!/usr/bin/env python3
"""Conversion rate (goals/attempts) by shot_type and zone.
ES regulation only, 5-season pool, min 500 attempts per cell."""
import math
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
SHOT = f"{ROOT}/Data/nhl_shot_events.csv"
OUT = f"{ROOT}/NFI/output/shot_type_by_zone.csv"
SEASONS = {"20212022","20222023","20232024","20242025","20252026"}
MIN_ATT = 500

def wilson(k, n, z=1.96):
    if n == 0: return (0.0,0.0,0.0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0.0,c-h), min(1.0,c+h))

def classify(x, y):
    if -15 <= y <= 15:
        if 74 <= x <= 89 and -9 <= y <= 9: return "CNFI"
        if 55 <= x < 74:  return "MNFI"
        if 25 <= x < 55:  return "FNFI"
        return "lane_other"
    return "Wide"

print("Loading shot_events...")
cols = ["season","period","event_type","situation_code","home_team_id",
        "shooting_team_id","shot_type","x_coord_norm","y_coord_norm","is_goal"]
df = pd.read_csv(SHOT, usecols=cols, dtype={"season":str,"situation_code":str})
df = df[df["season"].isin(SEASONS) & df["period"].between(1,3)].copy()
df = df[df["event_type"].isin(["shot-on-goal","missed-shot","blocked-shot","goal"])].copy()
df = df.dropna(subset=["x_coord_norm","y_coord_norm","shot_type"]).copy()

# ES filter
sc = df["situation_code"].astype(str).str.zfill(4)
ag = sc.str[0].astype(int); ask = sc.str[1].astype(int)
hsk = sc.str[2].astype(int); hg = sc.str[3].astype(int)
shoot_home = (df["shooting_team_id"] == df["home_team_id"])
shk = np.where(shoot_home, hsk, ask)
opk = np.where(shoot_home, ask, hsk)
es = (shk==opk) & (ag>0) & (hg>0)
df = df[es].copy()
print(f"  ES regulation rows: {len(df):,}")

df["zone"] = [classify(x,y) for x,y in zip(df["x_coord_norm"].astype(int), df["y_coord_norm"].astype(int))]
df["is_goal_i"] = df["is_goal"].astype(int)

# Per-zone per-shot-type
rows = []
for z in ["CNFI","MNFI","FNFI","Wide"]:
    sub = df[df["zone"]==z]
    for st, grp in sub.groupby("shot_type"):
        n = len(grp); g = int(grp["is_goal_i"].sum())
        if n < MIN_ATT: continue
        p, lo, hi = wilson(g, n)
        rows.append({"zone":z,"shot_type":st,"attempts":n,"goals":g,
                     "conv_pct":round(p*100,3),
                     "ci_lo_pct":round(lo*100,3),"ci_hi_pct":round(hi*100,3)})

# league-wide baseline (all zones pooled)
for st, grp in df.groupby("shot_type"):
    n = len(grp); g = int(grp["is_goal_i"].sum())
    if n < MIN_ATT: continue
    p, lo, hi = wilson(g, n)
    rows.append({"zone":"LEAGUE_ALL","shot_type":st,"attempts":n,"goals":g,
                 "conv_pct":round(p*100,3),
                 "ci_lo_pct":round(lo*100,3),"ci_hi_pct":round(hi*100,3)})

out = pd.DataFrame(rows)
# rank within each zone by conv_pct desc
out["rank_in_zone"] = out.groupby("zone")["conv_pct"].rank(method="min", ascending=False).astype(int)
out = out.sort_values(["zone","rank_in_zone"]).reset_index(drop=True)
out.to_csv(OUT, index=False)

# display
for z in ["CNFI","MNFI","FNFI","Wide","LEAGUE_ALL"]:
    sub = out[out["zone"]==z]
    if len(sub) == 0: continue
    title = "LEAGUE-WIDE BASELINE (all zones)" if z == "LEAGUE_ALL" else f"Zone: {z}"
    print(f"\n=== {title} ===")
    print(sub[["rank_in_zone","shot_type","attempts","goals","conv_pct","ci_lo_pct","ci_hi_pct"]]\
          .to_string(index=False))
