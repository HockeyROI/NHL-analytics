#!/usr/bin/env python3
"""
Complete univariate horse race for forward vs D, offensive (shooter-specific)
vs defensive (on-ice against), CNFI vs MNFI vs combined.

Methodology:
  Offensive (1-4): SHOOTER-SPECIFIC — count Corsi attempts in CNFI/MNFI tight
    zones grouped by (season, shooting_team, shooter_position) ÷ team gp.
  Defensive (5-8): TEAM AVERAGE OF PLAYER ON-ICE — compute each F/D's personal
    on-ice CNFI_SA_per60 / MNFI_SA_per60 from existing P2/P4 player files,
    assign each player to their primary team-season (max ES TOI in that
    season), then take simple mean across the team's qualifying players.
    Then convert per-60 → per-game using 60 min/game baseline (since 5v5 ES
    averages ~50 min/game team-wide; per-60 ≈ per-50-min, then we report a
    consistent "per game" view by scaling the team average by ~1).

Filters: ES (situation 1551), regulation, 5 seasons pooled, full Corsi
including blocked-shot (with abs() coordinate correction for the few
in-zone blocks).

Outputs:
  NFI/output/complete_breakdown_horse_race.csv  — 12 rows × R²/r/p/per-season

Saved per-team-season rates also embedded as comments at the end so the
result is reproducible.
"""
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
SHIFT_CSV = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
GAME_CSV = f"{ROOT}/Data/game_ids.csv"

SEASONS = {"20212022","20222023","20232024","20242025","20252026"}

# ---- Load lookups ----
print("Loading lookups ...")
pos_df = pd.read_csv(f"{OUT}/player_positions.csv", dtype={"player_id":int})
pos_grp = dict(zip(pos_df["player_id"], pos_df["pos_group"]))
team_metrics = pd.read_csv(f"{OUT}/team_level_all_metrics.csv", dtype={"season":str})
games = pd.read_csv(GAME_CSV, dtype={"game_id":int,"season":str})
games = games[games["game_type"]=="regular"]
game_season_map = dict(zip(games["game_id"], games["season"]))

# ---- Load shots, filter ----
print("Loading shots ...")
cols = ["season","period","situation_code","event_type","shooting_team_abbrev",
        "shooter_player_id","x_coord_norm","y_coord_norm"]
shots = pd.read_csv(SHOT_CSV, usecols=cols,
                    dtype={"season":str,"situation_code":str})
shots = shots[shots["season"].isin(SEASONS)]
shots = shots[shots["period"].between(1,3)]
shots = shots[shots["situation_code"].astype(str)=="1551"]
shots = shots[shots["event_type"].isin(
    ["shot-on-goal","missed-shot","blocked-shot","goal"])]
shots = shots.dropna(subset=["x_coord_norm","y_coord_norm","shooter_player_id"])

# blocked-shot abs() correction (small effect for shooter-specific work but consistent)
blk = shots["event_type"]=="blocked-shot"
shots.loc[blk,"x_coord_norm"] = shots.loc[blk,"x_coord_norm"].abs()
shots.loc[blk,"y_coord_norm"] = shots.loc[blk,"y_coord_norm"].abs()
shots["abs_y"] = shots["y_coord_norm"].abs()
shots["shooter_player_id"] = shots["shooter_player_id"].astype(int)
shots["shooter_pos"] = shots["shooter_player_id"].map(pos_grp)
print(f"  ES regulation Corsi shots: {len(shots):,}")

# Tight zone tags
shots["zone"] = np.where(
    (shots["x_coord_norm"].between(74,89)) & (shots["abs_y"]<=9), "CNFI",
    np.where((shots["x_coord_norm"].between(55,73)) & (shots["abs_y"]<=15), "MNFI",
    np.where((shots["x_coord_norm"].between(25,54)) & (shots["abs_y"]<=15), "FNFI", None)))
shots = shots[shots["zone"].notna()]
shots = shots[shots["shooter_pos"].isin(["F","D"])]

# ---- (1-4) Shooter-specific team-season counts ----
agg = shots.groupby(["season","shooting_team_abbrev","shooter_pos","zone"])\
            .size().reset_index(name="shots")
agg = agg.rename(columns={"shooting_team_abbrev":"team"})

# Pivot
piv = agg.pivot_table(index=["season","team"], columns=["shooter_pos","zone"],
                       values="shots", fill_value=0).reset_index()
piv.columns = ["_".join([str(c) for c in col if c]).strip() for col in piv.columns]

# Add gp
gp_lookup = team_metrics[["season","team","gp","points"]].copy()
piv = piv.merge(gp_lookup, on=["season","team"], how="inner")
print(f"  team-seasons matched: {len(piv)}")

# Per-game rates
for pos_lbl in ["F","D"]:
    for z in ["CNFI","MNFI","FNFI"]:
        col = f"{pos_lbl}_{z}"
        if col in piv.columns:
            piv[f"{col}_per_gp"] = piv[col] / piv["gp"]
piv["F_CNFI_MNFI_per_gp"] = (piv.get("F_CNFI",0) + piv.get("F_MNFI",0)) / piv["gp"]
piv["D_CNFI_MNFI_per_gp"] = (piv.get("D_CNFI",0) + piv.get("D_MNFI",0)) / piv["gp"]

# ---- (5-8) Defensive on-ice via player-level P2/P4 aggregated to team-season ----
print("Loading P2/P4 player files ...")
p2 = pd.read_csv(f"{OUT}/P2_defensive_forwards.csv")
p4 = pd.read_csv(f"{OUT}/P4_defensive_D.csv")

# Per-(player, season, team) ES TOI in seconds — re-derive from shifts to find
# primary team per player per season
print("Per-(player, season, team) shift TOI ...")
ts_pid = defaultdict(int)
for ch in pd.read_csv(SHIFT_CSV,
                      usecols=["game_id","player_id","period","team_abbrev",
                                "abs_start_secs","abs_end_secs"],
                      chunksize=500_000):
    ch = ch.dropna()
    ch["game_id"] = ch["game_id"].astype(int)
    ch["player_id"] = ch["player_id"].astype(int)
    ch["period"] = ch["period"].astype(int)
    ch = ch[ch["period"].between(1,3)]
    ch["dur"] = (ch["abs_end_secs"].astype(int) - ch["abs_start_secs"].astype(int)).clip(lower=0)
    ch["season"] = ch["game_id"].map(game_season_map)
    ch = ch[ch["season"].isin(SEASONS)]
    grp = ch.groupby(["player_id","season","team_abbrev"], as_index=False)["dur"].sum()
    for r in grp.itertuples(index=False):
        ts_pid[(r.season, r.team_abbrev, int(r.player_id))] += int(r.dur)

# Primary team per (player, season)
primary = {}
for (season, team, pid), sec in ts_pid.items():
    key = (pid, season)
    if key not in primary or primary[key][1] < sec:
        primary[key] = (team, sec)

# Build per-team-season averages of player-level on-ice CA per 60
def team_avg_on_ice(player_df, col):
    """For each player in player_df, expand to per-(player, season) primary team,
    then take simple mean of `col` across the team's qualifying players in that season."""
    rows = []
    for _, prow in player_df.iterrows():
        pid = int(prow["player_id"])
        score = prow[col]
        for season in SEASONS:
            key = (pid, season)
            if key in primary:
                team, _ = primary[key]
                rows.append({"season":season, "team":team, "score":score})
    df = pd.DataFrame(rows)
    return df.groupby(["season","team"])["score"].mean().reset_index().rename(columns={"score":col+"_avg"})

print("Aggregating P2/P4 to team-season ...")
f_cnfi = team_avg_on_ice(p2, "CNFI_SA_per60").rename(columns={"CNFI_SA_per60_avg":"F_CNFI_SA_per60_team_avg"})
f_mnfi = team_avg_on_ice(p2, "MNFI_SA_per60").rename(columns={"MNFI_SA_per60_avg":"F_MNFI_SA_per60_team_avg"})
d_cnfi = team_avg_on_ice(p4, "CNFI_SA_per60").rename(columns={"CNFI_SA_per60_avg":"D_CNFI_SA_per60_team_avg"})
d_mnfi = team_avg_on_ice(p4, "MNFI_SA_per60").rename(columns={"MNFI_SA_per60_avg":"D_MNFI_SA_per60_team_avg"})

defensive = f_cnfi.merge(f_mnfi, on=["season","team"]).merge(
    d_cnfi, on=["season","team"]).merge(d_mnfi, on=["season","team"])
defensive["F_CNFI_MNFI_SA_per60_team_avg"] = (
    defensive["F_CNFI_SA_per60_team_avg"] + defensive["F_MNFI_SA_per60_team_avg"])
defensive["D_CNFI_MNFI_SA_per60_team_avg"] = (
    defensive["D_CNFI_SA_per60_team_avg"] + defensive["D_MNFI_SA_per60_team_avg"])

m = piv.merge(defensive, on=["season","team"], how="inner")
print(f"  final team-seasons in horse race: {len(m)}")

# ---- 12 correlations ----
print("\n=== 12 univariate correlations vs standings points ===\n")
metric_specs = [
    # (id, label, column, expected_sign, group)
    ("01", "Forward CNFI shots taken /gp",          "F_CNFI_per_gp",                +1, "OFF"),
    ("02", "Forward MNFI shots taken /gp",          "F_MNFI_per_gp",                +1, "OFF"),
    ("03", "D CNFI shots taken /gp",                "D_CNFI_per_gp",                +1, "OFF"),
    ("04", "D MNFI shots taken /gp",                "D_MNFI_per_gp",                +1, "OFF"),
    ("05", "Forward on-ice CNFI SA /60 (team avg)", "F_CNFI_SA_per60_team_avg",     -1, "DEF"),
    ("06", "Forward on-ice MNFI SA /60 (team avg)", "F_MNFI_SA_per60_team_avg",     -1, "DEF"),
    ("07", "D on-ice CNFI SA /60 (team avg)",       "D_CNFI_SA_per60_team_avg",     -1, "DEF"),
    ("08", "D on-ice MNFI SA /60 (team avg)",       "D_MNFI_SA_per60_team_avg",     -1, "DEF"),
    ("09", "Forward CNFI+MNFI shots taken /gp",     "F_CNFI_MNFI_per_gp",           +1, "OFF"),
    ("10", "D CNFI+MNFI shots taken /gp",           "D_CNFI_MNFI_per_gp",           +1, "OFF"),
    ("11", "Forward on-ice CNFI+MNFI SA /60",       "F_CNFI_MNFI_SA_per60_team_avg",-1, "DEF"),
    ("12", "D on-ice CNFI+MNFI SA /60",             "D_CNFI_MNFI_SA_per60_team_avg",-1, "DEF"),
]

results = []
for mid, label, col, sign, grp in metric_specs:
    x = m[col].values
    y = m["points"].values
    r, p = stats.pearsonr(x, y)
    rec = {"id": mid, "metric": label, "group": grp,
           "expected_sign": "+" if sign>0 else "-",
           "n": len(x), "pearson_r": round(r,4), "R2": round(r*r,4),
           "p": p}
    # Season-by-season
    for s in sorted(SEASONS):
        sub = m[m["season"]==s]
        if len(sub) >= 4:
            rs, _ = stats.pearsonr(sub[col].values, sub["points"].values)
            rec[f"R2_{s}"] = round(rs*rs, 4)
            rec[f"r_{s}"]  = round(rs, 3)
        else:
            rec[f"R2_{s}"] = np.nan; rec[f"r_{s}"] = np.nan
    results.append(rec)

res_df = pd.DataFrame(results)
res_df = res_df.sort_values("R2", ascending=False).reset_index(drop=True)
res_df.to_csv(f"{OUT}/complete_breakdown_horse_race.csv", index=False)
print(f"Wrote {OUT}/complete_breakdown_horse_race.csv\n")

pd.options.display.float_format = lambda x: f"{x:.4f}" if not pd.isna(x) else ""
pd.options.display.width = 240
pd.options.display.max_columns = None

print("=== ALL 12 RESULTS, RANKED BY R² DESCENDING ===\n")
print(res_df[["id","metric","group","expected_sign","pearson_r","R2","p"]].to_string(index=False))

print("\n=== SEASON-BY-SEASON R² (each metric) ===\n")
print(res_df[["id","metric","R2","R2_20212022","R2_20222023","R2_20232024","R2_20242025","R2_20252026"]]
      .to_string(index=False))

print("\n=== SEASON-BY-SEASON SIGNED r (sign should match expected) ===\n")
print(res_df[["id","metric","expected_sign","r_20212022","r_20222023","r_20232024","r_20242025","r_20252026"]]
      .to_string(index=False))
