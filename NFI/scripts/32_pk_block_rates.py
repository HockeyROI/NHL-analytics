#!/usr/bin/env python3
"""
Add PK-specific block rates to player_block_rates.csv.

Filter logic — what counts as a "PK block":
  A player credited with a PK block is a player on the SHORTHANDED side
  who personally blocks a PP attack. So we filter shot events where the
  SHOOTING team has MORE skaters than the opponent (shooter on PP). The
  blocker recorded on that event is on the PK side, and we credit the
  block per 60 of THAT BLOCKER'S PK TOI.

  Note: the user's spec wording said "shooting team has fewer skaters
  than opponent". That would credit PP-side blockers of rare SHG attempts
  against their PK TOI, which is dimensionally incoherent. We use the
  analytically meaningful filter (shooter-on-PP / blocker-on-PK / PK TOI
  denominator) so the metric matches the natural interpretation of "PK
  blocking".

Filters:
  - Regulation periods 1-3
  - Drop empty net (situation_code with any 0 in goalie digits)
  - 5 seasons pooled (2021-22 through 2025-26)
  - blocked-shot events with valid blocker_player_id
  - Apply blocked-shot coord correction: abs(x_coord_norm), abs(y_coord_norm)
  - Tight zones: CNFI x 74-89 |y|<=9 / MNFI x 55-73 |y|<=15 / FNFI x 25-54 |y|<=15
  - Blocker must have >= 50 PK TOI minutes (from player_toi.csv)

xG-prevented weights per zone (zone-level Corsi conversion rates from
zone_conversion_rates.csv): CNFI 13.04%, MNFI 10.03%, FNFI 3.36%.

Adds these columns (prefix PK_) to existing NFI/output/player_block_rates.csv:
  PK_TOI_min
  PK_CNFI_blocks PK_CNFI_blocks_per60 PK_CNFI_blocks_lo95 PK_CNFI_blocks_hi95
  PK_MNFI_blocks PK_MNFI_blocks_per60 PK_MNFI_blocks_lo95 PK_MNFI_blocks_hi95
  PK_FNFI_blocks PK_FNFI_blocks_per60 PK_FNFI_blocks_lo95 PK_FNFI_blocks_hi95
  PK_TNFI_blocks PK_TNFI_blocks_per60 PK_TNFI_blocks_lo95 PK_TNFI_blocks_hi95
  PK_blocks_xG_prevented_total
  PK_blocks_xG_prevented_per60
  PK_TNFI_blocks_per60_rank
  PK_xG_prevented_per60_rank
"""
import math
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
TOI_CSV  = f"{OUT}/player_toi.csv"
POS_CSV  = f"{OUT}/player_positions.csv"
BLOCK_CSV = f"{OUT}/player_block_rates.csv"

SEASONS = {"20212022","20222023","20232024","20242025","20252026"}
MIN_PK_TOI_MIN = 50.0
ZONE_RATE = {"CNFI": 0.13042, "MNFI": 0.10032, "FNFI": 0.03363}  # zone_conversion_rates Fenwick

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
    if minutes <= 0: return 0.0, 0.0, 0.0
    p, lo, hi = wilson(events, max(events, int(round(minutes))))
    return events/minutes*60.0, lo*60.0, hi*60.0

# ---- Lookups ----
print("Loading metadata ...")
pos_df = pd.read_csv(POS_CSV, dtype={"player_id":int})
pos_grp  = dict(zip(pos_df["player_id"], pos_df["pos_group"]))
name_map = dict(zip(pos_df["player_id"], pos_df["player_name"]))
toi_df = pd.read_csv(TOI_CSV, dtype={"player_id":int})
toi_pk_min = dict(zip(toi_df["player_id"], toi_df["toi_PK_sec"]/60.0))

# ---- Load blocked shots ----
print("Loading blocked-shot events ...")
cols = ["season","period","event_type","situation_code",
        "home_team_id","shooting_team_id",
        "blocker_player_id","x_coord_norm","y_coord_norm"]
blk = pd.read_csv(SHOT_CSV, usecols=cols,
                  dtype={"season":str,"situation_code":str})
blk = blk[blk["season"].isin(SEASONS)].copy()
blk = blk[blk["period"].between(1,3)].copy()
blk = blk[blk["event_type"]=="blocked-shot"].copy()
blk = blk.dropna(subset=["blocker_player_id"])
blk["blocker_player_id"] = blk["blocker_player_id"].astype(int)
print(f"  blocked-shot rows (reg, with blocker): {len(blk):,}")

# Parse situation, drop empty net, filter to shooter-on-PP
sc = blk["situation_code"].astype(str).str.zfill(4)
ag  = sc.str[0].astype(int); ask = sc.str[1].astype(int)
hsk = sc.str[2].astype(int); hg  = sc.str[3].astype(int)
empty_net = (ag==0) | (hg==0)
shoot_home = blk["shooting_team_id"]==blk["home_team_id"]
sh_sk  = np.where(shoot_home, hsk, ask)
opp_sk = np.where(shoot_home, ask, hsk)

# NOTE on filter direction:
# For blocked-shot events, the NHL play-by-play `shooting_team_id` field is
# inverted relative to other event types — it points to the BLOCKER's team,
# not the actual shooter's team (same data-pipeline issue as the blocked-shot
# coordinate sign-flip we corrected earlier). Verified empirically: in
# situation 1451 (home on PP), 93.6% of blocked-shot rows have
# shoot_home=False, i.e. shooting_team_id = the AWAY (PK) team = blocker's
# team. Therefore to identify "blocker on PK" rows we need sh_sk < opp_sk
# (the blocker's team has fewer skaters than the opponent / actual shooter).
blk = blk[~empty_net & (sh_sk < opp_sk)].copy()
print(f"  after filter (blocker-on-PK, no empty net): {len(blk):,}")

# Apply abs() correction
blk["x_coord_norm"] = blk["x_coord_norm"].abs()
blk["y_coord_norm"] = blk["y_coord_norm"].abs()
blk["abs_y"] = blk["y_coord_norm"]
blk["zone"] = [classify(x, ay) for x, ay in zip(blk["x_coord_norm"].values,
                                                  blk["abs_y"].values)]
blk_in = blk[blk["zone"].notna()].copy()
print(f"  in-zone blocks: {len(blk_in):,}")
print(f"    CNFI: {(blk_in['zone']=='CNFI').sum():,}")
print(f"    MNFI: {(blk_in['zone']=='MNFI').sum():,}")
print(f"    FNFI: {(blk_in['zone']=='FNFI').sum():,}")

# Aggregate per (blocker, zone)
agg = blk_in.groupby(["blocker_player_id","zone"]).size().reset_index(name="blocks")
wide = agg.pivot_table(index="blocker_player_id", columns="zone",
                        values="blocks", aggfunc="sum",
                        fill_value=0).reset_index().rename(
    columns={"blocker_player_id":"player_id"})
for z in ["CNFI","MNFI","FNFI"]:
    if z not in wide.columns: wide[z] = 0
wide = wide.rename(columns={"CNFI":"PK_CNFI_blocks","MNFI":"PK_MNFI_blocks",
                              "FNFI":"PK_FNFI_blocks"})
wide["PK_TNFI_blocks"] = (wide["PK_CNFI_blocks"]+wide["PK_MNFI_blocks"]
                           +wide["PK_FNFI_blocks"])
wide["PK_TOI_min"]  = wide["player_id"].map(toi_pk_min)
qual = wide[(wide["PK_TOI_min"].notna()) &
            (wide["PK_TOI_min"] >= MIN_PK_TOI_MIN)].copy()
print(f"\nQualifying players (>= {MIN_PK_TOI_MIN:.0f} PK min): {len(qual)}")

# Per-60 rates with Wilson CIs
def add_rate(df, prefix, count_col, toi_col):
    rs, los, his = [], [], []
    for n, t in zip(df[count_col].values, df[toi_col].values):
        r, lo, hi = rate_ci(int(n), float(t))
        rs.append(round(r, 4)); los.append(round(lo, 4)); his.append(round(hi, 4))
    df[f"{prefix}_per60"] = rs
    df[f"{prefix}_lo95"]  = los
    df[f"{prefix}_hi95"]  = his

for zone in ["CNFI","MNFI","FNFI","TNFI"]:
    add_rate(qual, f"PK_{zone}_blocks", f"PK_{zone}_blocks", "PK_TOI_min")

qual["PK_blocks_xG_prevented_total"] = (
    qual["PK_CNFI_blocks"]*ZONE_RATE["CNFI"]
    + qual["PK_MNFI_blocks"]*ZONE_RATE["MNFI"]
    + qual["PK_FNFI_blocks"]*ZONE_RATE["FNFI"])
qual["PK_blocks_xG_prevented_per60"] = (
    qual["PK_blocks_xG_prevented_total"] / qual["PK_TOI_min"] * 60.0).round(4)

qual = qual.sort_values("PK_TNFI_blocks_per60", ascending=False).reset_index(drop=True)
qual["PK_TNFI_blocks_per60_rank"]   = qual["PK_TNFI_blocks_per60"].rank(
    ascending=False, method="min").astype(int)
qual["PK_xG_prevented_per60_rank"]  = qual["PK_blocks_xG_prevented_per60"].rank(
    ascending=False, method="min").astype(int)

# ---- Merge into existing player_block_rates.csv ----
print("\nMerging PK columns into player_block_rates.csv ...")
existing = pd.read_csv(BLOCK_CSV)
# Drop any pre-existing PK_ columns to avoid suffix collisions on rerun
existing = existing.drop(columns=[c for c in existing.columns if c.startswith("PK_")])
print(f"  existing rows: {len(existing)}")

pk_cols = ["player_id","PK_TOI_min",
           "PK_CNFI_blocks","PK_CNFI_blocks_per60","PK_CNFI_blocks_lo95","PK_CNFI_blocks_hi95",
           "PK_MNFI_blocks","PK_MNFI_blocks_per60","PK_MNFI_blocks_lo95","PK_MNFI_blocks_hi95",
           "PK_FNFI_blocks","PK_FNFI_blocks_per60","PK_FNFI_blocks_lo95","PK_FNFI_blocks_hi95",
           "PK_TNFI_blocks","PK_TNFI_blocks_per60","PK_TNFI_blocks_lo95","PK_TNFI_blocks_hi95",
           "PK_blocks_xG_prevented_total","PK_blocks_xG_prevented_per60",
           "PK_TNFI_blocks_per60_rank","PK_xG_prevented_per60_rank"]
qual_for_merge = qual[pk_cols].copy()

merged = existing.merge(qual_for_merge, on="player_id", how="left")
# Fill NaNs for players with no PK data: keep them in file but PK columns blank
merged.to_csv(BLOCK_CSV, index=False)
print(f"  merged file: {len(merged)} players, {merged['PK_TOI_min'].notna().sum()} with PK data")

# ============================================================
# Reporting
# ============================================================
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.width = 240
pd.options.display.max_columns = None

# Add position to qual for sorting
qual["pos_group"] = qual["player_id"].map(pos_grp)
qual["player_name"] = qual["player_id"].map(name_map)

print("\n=== TOP 15 D by PK_blocks_xG_prevented_per60 ===")
d_top = qual[qual["pos_group"]=="D"].sort_values(
    "PK_blocks_xG_prevented_per60", ascending=False).head(15)
print(d_top[["player_name","PK_TOI_min","PK_TNFI_blocks","PK_TNFI_blocks_per60",
              "PK_blocks_xG_prevented_per60","PK_CNFI_blocks_per60",
              "PK_MNFI_blocks_per60","PK_FNFI_blocks_per60"]].to_string(index=False))

print("\n=== TOP 10 F by PK_blocks_xG_prevented_per60 ===")
f_top = qual[qual["pos_group"]=="F"].sort_values(
    "PK_blocks_xG_prevented_per60", ascending=False).head(10)
print(f_top[["player_name","PK_TOI_min","PK_TNFI_blocks","PK_TNFI_blocks_per60",
              "PK_blocks_xG_prevented_per60","PK_CNFI_blocks_per60",
              "PK_MNFI_blocks_per60","PK_FNFI_blocks_per60"]].to_string(index=False))

# Compare ES vs PK for the named players
print("\n=== ES vs PK for top ES blockers — is PK blocking a distinct skill? ===")
# Pull both ES and PK metrics from merged file
named = ["Alec Martinez","Nick Seeler","Jacob Trouba","Brayden McNabb","Chris Tanev",
         "Brandt Clarke","Jared Spurgeon","Jonas Brodin","Andy Greene","Mark Borowiecki"]
view = merged[merged["player_name"].isin(named)].copy()
view = view[["player_name","es_toi_min","TNFI_blocks_per60","blocks_xG_prevented_per60",
             "PK_TOI_min","PK_TNFI_blocks_per60","PK_blocks_xG_prevented_per60",
             "PK_xG_prevented_per60_rank"]]
print(view.sort_values("blocks_xG_prevented_per60", ascending=False).to_string(index=False))

# Correlation between ES and PK blocking among players with both
both = merged[(merged["es_toi_min"].notna()) & (merged["PK_TOI_min"].notna())].copy()
if len(both) > 10:
    rho_xg = both[["blocks_xG_prevented_per60","PK_blocks_xG_prevented_per60"]].corr().iloc[0,1]
    rho_n  = both[["TNFI_blocks_per60","PK_TNFI_blocks_per60"]].corr().iloc[0,1]
    rho_xg_d = both[both["pos_group"]=="D"][["blocks_xG_prevented_per60","PK_blocks_xG_prevented_per60"]].corr().iloc[0,1] if "pos_group" in both.columns else None

    # Compute D-only correlation manually (pos_group not in merged file; use position from existing file)
    pos_role_d = (existing["position"] == "D")
    bothD_ids = set(existing[pos_role_d]["player_id"]) & set(both["player_id"])
    bothD = both[both["player_id"].isin(bothD_ids)]
    rho_xg_d = bothD[["blocks_xG_prevented_per60","PK_blocks_xG_prevented_per60"]].corr().iloc[0,1]

    print(f"\nPearson(ES xG-prev/60, PK xG-prev/60), all positions:  {rho_xg:.3f}  (n={len(both)})")
    print(f"Pearson(ES TNFI/60,    PK TNFI/60),    all positions:  {rho_n:.3f}")
    print(f"Pearson(ES xG-prev/60, PK xG-prev/60), D only:          {rho_xg_d:.3f}  (n={len(bothD)})")

print(f"\nFile overwritten: {BLOCK_CSV}")
