#!/usr/bin/env python3
"""
Step 1 - Foundation: 5x5 ft save% heat map + center-lane inflection points.

Regulation periods only (1-3). 3+ seasons pooled: 20222023, 20232024, 20242025.
"""
import csv, os, math
import pandas as pd
import numpy as np

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
SHOT = f"{ROOT}/Data/nhl_shot_events.csv"
OUT_DIR = f"{ROOT}/NFI/output"
SEASONS = {"20212022", "20222023", "20232024", "20242025", "20252026"}

# ---- load shots (filtered to reduce memory) ----
print("Loading shots...")
usecols = ["game_id","season","period","period_type","event_type","situation_code",
           "shooting_team_id","goalie_id","shot_type","x_coord_norm","y_coord_norm",
           "zone_code","is_goal"]
df = pd.read_csv(SHOT, usecols=usecols, dtype={"season":str,"situation_code":str})
print(f"  raw rows: {len(df):,}")

# regulation periods only
df = df[df["period"].between(1,3)].copy()
df = df[df["season"].isin(SEASONS)].copy()
# shots on goal + missed + goals contribute to attempts; for save% we need shots faced = SOG + goals
# Actually: save% = 1 - goals / (SOG + goals) since NHL convention: shot-on-goal excludes goals.
# Here we use: shots_faced = goals + shot-on-goal (not missed/blocked) for goalie save calculation
# "shot-on-goal" in NHL data includes saves only; goals are separate event type.
print(f"  event_types: {df['event_type'].unique()}")
# heat map uses all shot ATTEMPTS (for save% uses only shots that reached net: SOG + goals)
df_heat = df[df["event_type"].isin(["shot-on-goal","goal"])].copy()

# drop empty net: any situation where either team has 0 skaters -> keep to check; actually keep everything with goalie present
df_heat = df_heat[df_heat["goalie_id"].notna()].copy()
df_heat["is_goal_i"] = df_heat["is_goal"].astype(int)
print(f"  faced-net rows (regulation, seasons pool): {len(df_heat):,}")

# ---- 5x5 grid heatmap ----
# bin by floor(x/5)*5, floor(y/5)*5
df_heat["xb"] = (df_heat["x_coord_norm"] // 5 * 5).astype(int)
df_heat["yb"] = (df_heat["y_coord_norm"] // 5 * 5).astype(int)

grid = df_heat.groupby(["xb","yb"]).agg(shots=("is_goal_i","size"), goals=("is_goal_i","sum")).reset_index()
grid["save_pct"] = 1 - grid["goals"]/grid["shots"]
grid.to_csv(f"{OUT_DIR}/heatmap_save_pct_5x5.csv", index=False)
print(f"  wrote heatmap_save_pct_5x5.csv ({len(grid)} bins)")

# ---- center lane save% by 5-ft x increments (y in [-15,15]) ----
cl = df_heat[df_heat["y_coord_norm"].between(-15,15)].copy()
# x from 0 to 89 (blue line to net), 5-ft bins
cl = cl[cl["x_coord_norm"].between(25, 89)].copy()  # up to blue line area
cl["xb5"] = (cl["x_coord_norm"] // 5 * 5).astype(int)
lane = cl.groupby("xb5").agg(shots=("is_goal_i","size"), goals=("is_goal_i","sum")).reset_index()
lane["save_pct"] = 1 - lane["goals"]/lane["shots"]
lane = lane.sort_values("xb5")
print("\nCenter-lane save% by x-bin (y in [-15,15]):")
print(lane.to_string(index=False))

# compute gradient (first difference of save%) — find where save% drops most rapidly moving toward net
lane["save_pct_smooth"] = lane["save_pct"].rolling(3, center=True, min_periods=1).mean()
lane["d_save"] = lane["save_pct_smooth"].diff()

# inflection: 2nd derivative sign changes
lane["d2"] = lane["d_save"].diff()
print("\nGradient analysis:")
print(lane[["xb5","shots","save_pct","save_pct_smooth","d_save","d2"]].to_string(index=False))

# Strategy: find x where gradient changes most negatively, then next point.
# Save% is higher far from net and drops as x -> 89. Find two most significant breakpoints
# (largest negative d_save spikes moving net-ward, i.e., lowest d_save values).
# Exclude x >= 74 (that's CNFI fixed zone).
candidates = lane[(lane["xb5"]<74) & (lane["xb5"]>=25)].copy()
candidates = candidates.sort_values("d_save")  # most negative first
brk = candidates.head(5)
print("\nTop 5 most-negative gradient points (x < 74):")
print(brk.to_string(index=False))

# The two most significant breakpoints (largest |d_save| drops)
candidates["abs_d"] = candidates["d_save"].abs()
top2 = candidates.sort_values("abs_d", ascending=False).head(2)
print("\nTop 2 breakpoints:")
print(top2.to_string(index=False))

# Save lane + breakpoints
lane.to_csv(f"{OUT_DIR}/center_lane_save_pct.csv", index=False)

with open(f"{OUT_DIR}/inflection_points.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank","x_bin","save_pct","d_save","shots"])
    for i,(_,r) in enumerate(top2.iterrows(),1):
        w.writerow([i,int(r["xb5"]),round(r["save_pct"],4),round(r["d_save"],4),int(r["shots"])])

print(f"\nWrote center_lane_save_pct.csv and inflection_points.csv")
