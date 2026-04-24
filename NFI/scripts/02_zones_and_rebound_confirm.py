#!/usr/bin/env python3
"""
Step 2 - Zone definitions (using Step 1 inflection point 1 = 55).
Step 3 - Foundation confirmation: compare zone conversion rates to subsequent tip/def/rebound.

Outputs:
  - zones_shot_counts.csv (raw zone distribution by state)
  - rebound_confirmation.csv (conversion rate to rebound/tip/deflection within 2s)
"""
import csv, math, os
import pandas as pd
import numpy as np

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
REB_CSV = f"{ROOT}/NFI/Geometry_post/Data/rebound_sequences.csv"
OUT_DIR = f"{ROOT}/NFI/output"
SEASONS = {"20212022", "20222023", "20232024", "20242025", "20252026"}

# Zones (Step 2)
INFL1 = 55
INFL2 = 60
BLUE = 25  # nominal blue line x (normalized attacking zone)
NET_X = 89

def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k/n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n))/denom
    halfw = z*math.sqrt(p*(1-p)/n + z**2/(4*n*n))/denom
    return (round(p,4), round(max(0,center-halfw),4), round(min(1,center+halfw),4))

def classify_zone(x, y):
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return "unk"
    # CNFI: 74-89, -9 to 9
    if 74 <= x <= 89 and -9 <= y <= 9:
        return "CNFI"
    # Center lane test
    if -15 <= y <= 15:
        if INFL1 <= x < 74:
            return "MNFI"
        if BLUE <= x < INFL1:
            return "FNFI"
        return "lane_other"
    # Wide: outside center lane
    return "Wide"

def situation_state(row):
    """Derive ES/PP/PK from situation_code relative to shooting team.
    situation_code is 4-digit: away_g, away_skaters, home_skaters, home_g
    (NHL api convention: position 0=away_goalie, 1=away_skaters, 2=home_skaters, 3=home_goalie)
    Drop empty net (any goalie digit == 0).
    """
    sc = str(row["situation_code"]).zfill(4)
    if len(sc) != 4 or not sc.isdigit():
        return None
    ag, ask, hsk, hg = int(sc[0]), int(sc[1]), int(sc[2]), int(sc[3])
    if ag == 0 or hg == 0:
        return None  # empty net
    # skater count for shooting team vs opponent
    if row["shooting_team_id"] == row["home_team_id"]:
        sh, op = hsk, ask
    else:
        sh, op = ask, hsk
    if sh == op:
        return "ES"
    if sh > op:
        return "PP"
    return "PK"

# ---- load shots ----
print("Loading shots...")
cols = ["game_id","season","period","event_id","event_type","situation_code",
        "home_team_id","shooting_team_id","goalie_id","x_coord_norm","y_coord_norm","is_goal","time_secs"]
shots = pd.read_csv(SHOT_CSV, usecols=cols, dtype={"season":str,"situation_code":str})
shots = shots[shots["season"].isin(SEASONS) & shots["period"].between(1,3)].copy()
# Keep shot attempts for NFI count (not blocked - blocked has recorder coords skewed)
# For zone classification we'll include all shot-type events that registered
shots = shots[shots["event_type"].isin(["shot-on-goal","missed-shot","blocked-shot","goal"])].copy()

shots["state"] = shots.apply(situation_state, axis=1)
shots = shots[shots["state"].notna()].copy()
shots["zone"] = [classify_zone(x,y) for x,y in zip(shots["x_coord_norm"], shots["y_coord_norm"])]
shots["is_goal_i"] = shots["is_goal"].astype(int)

# Save zone summary
zone_counts = shots.groupby(["state","zone"]).agg(attempts=("is_goal_i","size"), goals=("is_goal_i","sum")).reset_index()
zone_counts.to_csv(f"{OUT_DIR}/zones_shot_counts.csv", index=False)
print("Zone attempts by state:")
print(zone_counts.pivot(index="zone", columns="state", values="attempts").to_string())

# ---- Step 3: rebound confirmation ----
# rebound_sequences.csv has pairs of events within 3s. Need to join: original shot -> does it have a rebound/tip/def within 2s?
print("\nLoading rebound sequences...")
rebs = pd.read_csv(REB_CSV, dtype={"season":str})
rebs = rebs[rebs["season"].isin(SEASONS)].copy()
rebs = rebs[rebs["period"].between(1,3)].copy()
rebs = rebs[rebs["time_gap_secs"] <= 2].copy()
# tip/deflection/rebound events - all qualify since rebound_sequences already filtered
# we need: for each orig (game_id, orig_shooter_id, orig_x, orig_y) flag that it converted
# But join should be on the original shot event. rebound_sequences has orig_event info but no event_id.
# Approach: compute per-shot flag by matching (game_id, period, orig_x, orig_y, orig_shooter_id, season) to original shot.

# We can more simply: build a set of "(game_id, orig_shooter_id, orig_x, orig_y)" keys that had rebound conversion.
reb_keys = set(zip(rebs["game_id"].astype(str), rebs["orig_shooter_id"].astype(str),
                   rebs["orig_x"].astype(int), rebs["orig_y"].astype(int)))
print(f"  rebound-originating shot keys: {len(reb_keys):,}")

shots["_key"] = list(zip(shots["game_id"].astype(str), shots["shooter_player_id"].astype(str)
                         if "shooter_player_id" in shots.columns else shots["goalie_id"].astype(str),
                         shots["x_coord_norm"].astype(int), shots["y_coord_norm"].astype(int)))
# We didn't load shooter_player_id; reload with it
shots2 = pd.read_csv(SHOT_CSV,
    usecols=["game_id","season","period","event_type","situation_code","home_team_id",
             "shooting_team_id","shooter_player_id","x_coord","y_coord",
             "x_coord_norm","y_coord_norm","is_goal"],
    dtype={"season":str,"situation_code":str})
shots2 = shots2[shots2["season"].isin(SEASONS) & shots2["period"].between(1,3)].copy()
shots2 = shots2[shots2["event_type"].isin(["shot-on-goal","missed-shot","blocked-shot","goal"])].copy()
shots2["state"] = shots2.apply(situation_state, axis=1)
shots2 = shots2[shots2["state"].notna()].copy()
shots2["zone"] = [classify_zone(x,y) for x,y in zip(shots2["x_coord_norm"], shots2["y_coord_norm"])]

# rebound_sequences uses RAW x/y (not normalized), so use x_coord and y_coord for key match
# From earlier sample: "orig_x=53, orig_y=-8" while norm would be 74. So raw used.
shots2 = shots2[shots2["shooter_player_id"].notna() & shots2["x_coord"].notna() & shots2["y_coord"].notna()].copy()
shots2["_key"] = list(zip(shots2["game_id"].astype(str),
                          shots2["shooter_player_id"].astype(int).astype(str),
                          shots2["x_coord"].astype(int),
                          shots2["y_coord"].astype(int)))
shots2["had_followup"] = shots2["_key"].isin(reb_keys).astype(int)

# conversion = had_followup within 2s to tip/deflection/rebound
conv = shots2.groupby(["state","zone"]).agg(
    attempts=("had_followup","size"),
    followups=("had_followup","sum")
).reset_index()
conv[["conv_rate","conv_lo","conv_hi"]] = conv.apply(
    lambda r: pd.Series(wilson(r["followups"], r["attempts"])), axis=1)
conv = conv.sort_values(["state","zone"])
conv.to_csv(f"{OUT_DIR}/rebound_confirmation.csv", index=False)
print("\nConversion to follow-up within 2s (by zone, state):")
print(conv.to_string(index=False))
