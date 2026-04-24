#!/usr/bin/env python3
"""Rerun goalie pillars 6 (FNFI/MNFI) and 7 (CNFI) with min 300 shots faced.
Rebuilds goalie save% directly from shots_tagged.csv (SOG + goal events only).
Overwrites pillar_6_goalie_FNFI_MNFI.csv, pillar_7_goalie_CNFI.csv, and
updates pillar_ci_flagging.csv."""
import math
import pandas as pd

OUT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/NFI/output"
MIN_SHOTS = 300
STATES = ["ES","PP","PK"]

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0.0,c-h), min(1.0,c+h))

sh = pd.read_csv(f"{OUT}/shots_tagged.csv")
# goalie-faced = shot-on-goal + goal (missed/blocked never reach goalie)
faced = sh[sh["event_type"].isin(["shot-on-goal","goal"])].copy()
faced = faced[faced["goalie_id"].notna()].copy()
faced["goalie_id"] = faced["goalie_id"].astype(int)
print(f"Shots-faced rows: {len(faced):,}")

# Pillar 6: FNFI + MNFI by state
rows = []
for (gid, state, zone), grp in faced.groupby(["goalie_id","state","zone"]):
    if zone not in ("FNFI","MNFI"): continue
    n = len(grp); goals = int(grp["is_goal_i"].sum())
    if n < MIN_SHOTS: continue
    p, lo, hi = wilson(n-goals, n)
    rows.append({"goalie_id":gid,"state":state,"zone":zone,
                 "faced":n,"goals":goals,
                 "save_pct":round(p,4),"sv_lo":round(lo,4),"sv_hi":round(hi,4)})
p6 = pd.DataFrame(rows)
p6.to_csv(f"{OUT}/pillar_6_goalie_FNFI_MNFI.csv", index=False)
p6_goalies = p6["goalie_id"].nunique()
print(f"Pillar 6 rows: {len(p6)}  unique goalies: {p6_goalies}")

# Pillar 7: CNFI by state
rows = []
for (gid, state), grp in faced[faced["zone"]=="CNFI"].groupby(["goalie_id","state"]):
    n = len(grp); goals = int(grp["is_goal_i"].sum())
    if n < MIN_SHOTS: continue
    p, lo, hi = wilson(n-goals, n)
    rows.append({"goalie_id":gid,"state":state,"zone":"CNFI",
                 "faced":n,"goals":goals,
                 "save_pct":round(p,4),"sv_lo":round(lo,4),"sv_hi":round(hi,4)})
p7 = pd.DataFrame(rows)
p7.to_csv(f"{OUT}/pillar_7_goalie_CNFI.csv", index=False)
p7_goalies = p7["goalie_id"].nunique()
print(f"Pillar 7 rows: {len(p7)}  unique goalies: {p7_goalies}")

# CI-clear computation: tight CI = (sv_hi - sv_lo) < 0.04
def ci_summary(df, label):
    df["informative"] = (df["sv_hi"] - df["sv_lo"]) < 0.04
    per_g = df.groupby("goalie_id")["informative"].any().reset_index()
    total = len(per_g); clears = int(per_g["informative"].sum())
    return {"pillar":label,"players":total,"ci_clear":clears,
            "ci_clear_pct":round(100*clears/total,1) if total else 0.0}

# update pillar_ci_flagging.csv
flag = pd.read_csv(f"{OUT}/pillar_ci_flagging.csv")
# drop old P6/P7 rows and re-insert
flag = flag[~flag["pillar"].isin(["P6_Goalie_FM","P7_Goalie_CNFI"])].copy()
r6 = ci_summary(p6, "P6_Goalie_FM")
r7 = ci_summary(p7, "P7_Goalie_CNFI")
r6["below_50pct"] = r6["ci_clear_pct"] < 50
r7["below_50pct"] = r7["ci_clear_pct"] < 50
flag = pd.concat([flag, pd.DataFrame([r6, r7])], ignore_index=True)
flag.to_csv(f"{OUT}/pillar_ci_flagging.csv", index=False)

print("\n=== pillar_ci_flagging.csv (updated) ===")
print(flag.to_string(index=False))
