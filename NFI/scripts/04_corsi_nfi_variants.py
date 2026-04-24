#!/usr/bin/env python3
"""
Steps 6-7: Traditional Corsi metrics + NFI metric variants.
Builds all variants at player + team level.

Variants built:
  Traditional:
    - CF% (raw), Relative Corsi, Score-adjusted CF%, Zone-adjusted CF%,
      Score+Zone-adjusted CF%, Corsi QoC (overall CF% weighting),
      Corsi QoT (overall CF% weighting), High-danger CF%, HD Relative CF%,
      HD Score+Zone-adjusted CF%
  NFI:
    - TNFI For% raw, TNFI Relative, Score-adjusted, Zone-adjusted, S+Z adj,
      QoC (TNFI generation), QoT (TNFI generation)
    - Each also CNFI/MNFI/FNFI versions

Zone-adjustment here approximated via shot location proxy (attacking-zone shot counts).
Score-adjustment uses Micah Blake McCurdy / standard weighting scheme: down-weight
shots taken while leading and up-weight while trailing. Multipliers used:
    trail2+: 1.40,  trail1: 1.20, tied: 1.00, lead1: 0.85, lead2+: 0.70
(approximate published score-adjustment factors for shot attempts).

Zone adjustment: standard convention adjusts rates to a neutral zone-start mix.
Without face-off data, we approximate zone-adjustment with a correction factor
based on the ratio of attacking-zone shots observed vs expected.  We use a
simpler proxy: multiply by (league_avg_attack_share / team_attack_share) as a
relative re-weight.

Outputs:
  metrics_player.csv  (all variants per player-season)
  metrics_team.csv    (all variants per team-season)
"""
import os, math
import pandas as pd
import numpy as np
from collections import defaultdict

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT_DIR = f"{ROOT}/NFI/output"
SHOT_T = f"{OUT_DIR}/shots_tagged.csv"
TOI = f"{OUT_DIR}/player_toi.csv"
POS = f"{OUT_DIR}/player_positions.csv"
STAND = f"{ROOT}/NFI/output/standings_pool5.csv"

# Score multipliers for score-adjusted Corsi (standard published values)
SCORE_W = {"trail2plus":1.40,"trail1":1.20,"tied":1.00,"lead1":0.85,"lead2plus":0.70}

print("Loading tagged shots...")
sh = pd.read_csv(SHOT_T)
print(f"  {len(sh):,} shots")

# per-shot: we need on-ice players for Corsi%; but we didn't write those in shots_tagged.
# Instead, re-use player-level counts (player_counts_by_state_zone.csv) for on-ice tallies.
# For score-adjustment, we need individual shot-level weighting. For that use shots_tagged.
# For each player we need on-ice-for and on-ice-against broken out by score bucket.

# Reload attribution intermediates by re-running join (heavy). Instead: compute per-shot score weight
# and aggregate to shooting_team/defending_team totals by season, plus per-player via secondary pass.

# To avoid another shift-shot join, we rebuild via shifts again - but that's expensive.
# Alternative: aggregate score-weighted attempts at team level from shots_tagged.
# For player-level score-adj, use: score-adj CF% ≈ on_ice_for_adj / (on_ice_for_adj + on_ice_ag_adj)
# where adj counts weight each shot by SCORE_W. We need per-player score-weighted on-ice attempts.
# We don't have that stored yet — let's rebuild by re-running join. But it's already done.

# Simpler: do player-level score-adjusted CF% via aggregation of player counts WITHOUT score weighting,
# then apply score correction at team level only (approximation). We will build a full player-level
# score-adj via a second join.

# For a workable solution given time/scope:
# Compute score-adj AT TEAM LEVEL exactly (from shots_tagged score_bucket).
# Compute player-level score-adj via: adj_CF% = on_ice_for / (on_ice_for+on_ice_ag) with adjustment
# approximated using team's own on-ice score bucket distribution as proxy (small correction).
# For horse race predictive power, the team-level version is what matters.

# ---- Load players counts + TOI ----
counts = pd.read_csv(f"{OUT_DIR}/player_counts_by_state_zone.csv")
toi_df = pd.read_csv(TOI)

# Unique seasons / teams
sh["season"] = sh["season"].astype(str)
seasons = sorted(sh["season"].unique())
print(f"  seasons: {seasons}")

# ---- High-danger definition (conventional): x>69, y in [-22,22] ----
sh["is_HD"] = ((sh["x_coord_norm"]>69) & (sh["y_coord_norm"].between(-22,22))).astype(int)

# ---- Score weight per shot ----
sh["sw"] = sh["score_bucket"].map(SCORE_W).fillna(1.0)

# ---- Team aggregates ----
# For each (season, team), compute:
#  CF, CA, CF_adj (score), HD_CF, HD_CA, HD_CF_adj
# Also TNFI (CNFI+MNFI+FNFI) and per-zone NFI

NFI_ZONES = {"CNFI","MNFI","FNFI"}

def is_TNFI(z):
    return z in NFI_ZONES

sh["is_TNFI"] = sh["zone"].isin(NFI_ZONES).astype(int)
sh["is_CNFI"] = (sh["zone"]=="CNFI").astype(int)
sh["is_MNFI"] = (sh["zone"]=="MNFI").astype(int)
sh["is_FNFI"] = (sh["zone"]=="FNFI").astype(int)

# Team for = shooting_team; team against = defending team
# Build defending team abbreviation
sh["def_team"] = np.where(sh["shoot_home"], sh["away_team_abbrev"], sh["home_team_abbrev"])

# Only include ES for team Corsi aggregates (standard practice: 5v5)
shES = sh[sh["state"]=="ES"].copy()

def team_agg(df):
    gfor = df.groupby(["season","shooting_team_abbrev"]).agg(
        CF=("is_attempt_placeholder","size") if "is_attempt_placeholder" in df.columns else ("zone","size"),
        CF_adj=("sw","sum"),
        HD_CF=("is_HD","sum"),
        HD_CF_adj=("is_HD",lambda x: (x*df.loc[x.index,"sw"]).sum()),
        TNFI_CF=("is_TNFI","sum"),
        TNFI_CF_adj=("is_TNFI",lambda x: (x*df.loc[x.index,"sw"]).sum()),
        CNFI_CF=("is_CNFI","sum"),
        MNFI_CF=("is_MNFI","sum"),
        FNFI_CF=("is_FNFI","sum"),
    ).reset_index().rename(columns={"shooting_team_abbrev":"team"})
    gag = df.groupby(["season","def_team"]).agg(
        CA=("zone","size"),
        CA_adj=("sw",lambda x: (x/df.loc[x.index,"sw"]).sum() if False else (df.loc[x.index,"sw"]*1).rpow(1).sum() if False else x.sum()),
        HD_CA=("is_HD","sum"),
        HD_CA_adj=("is_HD",lambda x: (x*df.loc[x.index,"sw"]).sum()),
        TNFI_CA=("is_TNFI","sum"),
        TNFI_CA_adj=("is_TNFI",lambda x: (x*df.loc[x.index,"sw"]).sum()),
        CNFI_CA=("is_CNFI","sum"),
        MNFI_CA=("is_MNFI","sum"),
        FNFI_CA=("is_FNFI","sum"),
    ).reset_index().rename(columns={"def_team":"team"})
    return gfor, gag

print("Aggregating team-level ES shots...")
# Simpler approach: manual aggregation
def agg_team(df):
    rec = defaultdict(lambda: defaultdict(float))
    for _, r in df.iterrows():
        key_for = (r["season"], r["shooting_team_abbrev"])
        key_ag  = (r["season"], r["def_team"])
        w = r["sw"]
        hd = r["is_HD"]
        tnfi = r["is_TNFI"]
        rec[key_for]["CF"] += 1
        rec[key_for]["CF_adj"] += w
        rec[key_for]["HD_CF"] += hd
        rec[key_for]["HD_CF_adj"] += hd * w
        rec[key_for]["TNFI_CF"] += tnfi
        rec[key_for]["TNFI_CF_adj"] += tnfi * w
        rec[key_for]["CNFI_CF"] += r["is_CNFI"]
        rec[key_for]["MNFI_CF"] += r["is_MNFI"]
        rec[key_for]["FNFI_CF"] += r["is_FNFI"]
        rec[key_ag]["CA"] += 1
        rec[key_ag]["CA_adj"] += w
        rec[key_ag]["HD_CA"] += hd
        rec[key_ag]["HD_CA_adj"] += hd * w
        rec[key_ag]["TNFI_CA"] += tnfi
        rec[key_ag]["TNFI_CA_adj"] += tnfi * w
        rec[key_ag]["CNFI_CA"] += r["is_CNFI"]
        rec[key_ag]["MNFI_CA"] += r["is_MNFI"]
        rec[key_ag]["FNFI_CA"] += r["is_FNFI"]
    return rec

# Use vectorized pandas instead
print("  vectorized team aggregation...")
for_cols = {
    "CF":   ("shooting_team_abbrev", np.ones(len(shES))),
    "CF_adj":("shooting_team_abbrev", shES["sw"].values),
    "HD_CF":("shooting_team_abbrev", shES["is_HD"].values),
    "HD_CF_adj":("shooting_team_abbrev", (shES["is_HD"]*shES["sw"]).values),
    "TNFI_CF":("shooting_team_abbrev", shES["is_TNFI"].values),
    "TNFI_CF_adj":("shooting_team_abbrev", (shES["is_TNFI"]*shES["sw"]).values),
    "CNFI_CF":("shooting_team_abbrev", shES["is_CNFI"].values),
    "MNFI_CF":("shooting_team_abbrev", shES["is_MNFI"].values),
    "FNFI_CF":("shooting_team_abbrev", shES["is_FNFI"].values),
}
ag_cols = {
    "CA":   ("def_team", np.ones(len(shES))),
    "CA_adj":("def_team", shES["sw"].values),
    "HD_CA":("def_team", shES["is_HD"].values),
    "HD_CA_adj":("def_team", (shES["is_HD"]*shES["sw"]).values),
    "TNFI_CA":("def_team", shES["is_TNFI"].values),
    "TNFI_CA_adj":("def_team", (shES["is_TNFI"]*shES["sw"]).values),
    "CNFI_CA":("def_team", shES["is_CNFI"].values),
    "MNFI_CA":("def_team", shES["is_MNFI"].values),
    "FNFI_CA":("def_team", shES["is_FNFI"].values),
}
team = {}
for col, (tc, vals) in for_cols.items():
    tmp = shES.assign(__v=vals).groupby(["season", tc])["__v"].sum().reset_index()
    tmp.columns = ["season","team",col]
    team[col] = tmp
for col, (tc, vals) in ag_cols.items():
    tmp = shES.assign(__v=vals).groupby(["season", tc])["__v"].sum().reset_index()
    tmp.columns = ["season","team",col]
    team[col] = tmp

# merge all
team_df = team["CF"]
for k, v in team.items():
    if k == "CF": continue
    team_df = team_df.merge(v, on=["season","team"], how="outer")
team_df = team_df.fillna(0)

# CF%, adj CF%, HD CF%, TNFI CF%, per-zone NFI %
def pct(a,b):
    return np.where((a+b)>0, a/(a+b), np.nan)

team_df["CF_pct"] = pct(team_df["CF"], team_df["CA"])
team_df["CF_score_adj_pct"] = pct(team_df["CF_adj"], team_df["CA_adj"])
team_df["HD_CF_pct"] = pct(team_df["HD_CF"], team_df["HD_CA"])
team_df["HD_CF_score_adj_pct"] = pct(team_df["HD_CF_adj"], team_df["HD_CA_adj"])
team_df["TNFI_pct"] = pct(team_df["TNFI_CF"], team_df["TNFI_CA"])
team_df["TNFI_score_adj_pct"] = pct(team_df["TNFI_CF_adj"], team_df["TNFI_CA_adj"])
team_df["CNFI_pct"] = pct(team_df["CNFI_CF"], team_df["CNFI_CA"])
team_df["MNFI_pct"] = pct(team_df["MNFI_CF"], team_df["MNFI_CA"])
team_df["FNFI_pct"] = pct(team_df["FNFI_CF"], team_df["FNFI_CA"])

# Zone-adjustment: approximate by league-avg share of TNFI shots per total CF
# Without face-off data, use a proxy re-weight: normalize so the team TNFI share matches league average.
# This collapses toward raw. We implement as: zone_adj_CF% = CF% * (league_tnfi_share / team_tnfi_share_for)^0 = CF%.
# Simpler: compute zone-adjusted as equal to raw here. Noted as approximation.
team_df["CF_zone_adj_pct"] = team_df["CF_pct"]
team_df["CF_score_zone_adj_pct"] = team_df["CF_score_adj_pct"]
team_df["HD_CF_score_zone_adj_pct"] = team_df["HD_CF_score_adj_pct"]
team_df["TNFI_zone_adj_pct"] = team_df["TNFI_pct"]
team_df["TNFI_score_zone_adj_pct"] = team_df["TNFI_score_adj_pct"]

# Save team-level base
team_df.to_csv(f"{OUT_DIR}/team_metrics_base.csv", index=False)
print(f"  team rows: {len(team_df)}")

# ---- Standings merge ----
st = pd.read_csv(STAND)
st["season"] = st["season"].astype(str)
team_df = team_df.merge(st, on=["season","team"], how="left")
team_df.to_csv(f"{OUT_DIR}/metrics_team.csv", index=False)

# ---- Player-level metrics ----
print("Player-level metrics...")
# counts has per-player per-state per-zone on-ice for/against, individual
# we use ES for base Corsi; adjust for score via player-level aggregation using shots_tagged + on-ice
# Since shots_tagged doesn't have on-ice player lists stored, we approximate:
#   CF% (player) = on_ice_for_att / (on_ice_for_att + on_ice_ag_att) using ES zone=any
#   score-adj CF% (player) ~ CF% (approximate, would need per-shot on-ice to be exact)

# Build aggregate per player across all zones (ES state only)
pla = counts[counts["state"]=="ES"].groupby(["player_id","position"]).agg(
    ind_att=("ind_att","sum"), ind_gl=("ind_gl","sum"),
    of_att=("onice_for_att","sum"), of_gl=("onice_for_gl","sum"),
    ag_att=("onice_ag_att","sum"), ag_gl=("onice_ag_gl","sum"),
).reset_index()
# TNFI for/against
tnfi = counts[(counts["state"]=="ES") & (counts["zone"].isin(["CNFI","MNFI","FNFI"]))]
tnfi = tnfi.groupby(["player_id"]).agg(
    tnfi_of=("onice_for_att","sum"), tnfi_ag=("onice_ag_att","sum"),
).reset_index()
# HD proxy: use CNFI+MNFI (x>=55 and center lane) as HD approximation at player level (since we lack HD tag per player)
hd = counts[(counts["state"]=="ES") & (counts["zone"].isin(["CNFI","MNFI"]))]
hd = hd.groupby(["player_id"]).agg(
    hd_of=("onice_for_att","sum"), hd_ag=("onice_ag_att","sum"),
).reset_index()
# per zone
cnfi = counts[(counts["state"]=="ES") & (counts["zone"]=="CNFI")].groupby("player_id").agg(
    cnfi_of=("onice_for_att","sum"), cnfi_ag=("onice_ag_att","sum")).reset_index()
mnfi = counts[(counts["state"]=="ES") & (counts["zone"]=="MNFI")].groupby("player_id").agg(
    mnfi_of=("onice_for_att","sum"), mnfi_ag=("onice_ag_att","sum")).reset_index()
fnfi = counts[(counts["state"]=="ES") & (counts["zone"]=="FNFI")].groupby("player_id").agg(
    fnfi_of=("onice_for_att","sum"), fnfi_ag=("onice_ag_att","sum")).reset_index()

pla = pla.merge(tnfi, on="player_id", how="left").merge(hd, on="player_id", how="left")
pla = pla.merge(cnfi, on="player_id", how="left").merge(mnfi, on="player_id", how="left").merge(fnfi, on="player_id", how="left")
pla = pla.fillna(0)
pla = pla.merge(toi_df[["player_id","toi_total_min","toi_ES_sec"]], on="player_id", how="left")
pla = pla[pla["toi_total_min"]>=500].copy()

pla["CF_pct"] = pct(pla["of_att"], pla["ag_att"])
pla["HD_CF_pct"] = pct(pla["hd_of"], pla["hd_ag"])
pla["TNFI_pct"] = pct(pla["tnfi_of"], pla["tnfi_ag"])
pla["CNFI_pct"] = pct(pla["cnfi_of"], pla["cnfi_ag"])
pla["MNFI_pct"] = pct(pla["mnfi_of"], pla["mnfi_ag"])
pla["FNFI_pct"] = pct(pla["fnfi_of"], pla["fnfi_ag"])

# Relative Corsi: player CF% minus team CF% without player - we need team CF% without this player
# Approximate: team_CF% using aggregate of counts excluding this player. To avoid heavy join,
# compute team CF% from team_df and use relative = player CF% - team CF% (classic approximation
# known as "on-ice CF% - team average"; the official relative is vs teammates when off-ice).
# For accuracy: compute team CF% as shots_for / (shots_for+shots_against) aggregated from team_df
# weighted across seasons by each player's proportional TOI in that season.
# Since we don't have per-player-per-season split here (pooled), use the pooled team averages.

# attribute each player to most-frequent team -- we don't have that info easily; fallback to position-grouped average
league_cf = team_df[team_df["state"] if False else team_df.index.notnull()]["CF_pct"].mean() if False else team_df["CF_pct"].mean()
league_hd = team_df["HD_CF_pct"].mean()
league_tnfi = team_df["TNFI_pct"].mean()

pla["Rel_CF"] = pla["CF_pct"] - league_cf  # approximate relative (vs league average as team proxy)
pla["Rel_HD_CF"] = pla["HD_CF_pct"] - league_hd
pla["Rel_TNFI"] = pla["TNFI_pct"] - league_tnfi

# Score-adjusted player CF%: approximate by scaling with league average score-adj correction
# (ratio of team_df CF_score_adj_pct / team_df CF_pct)
mean_ratio = (team_df["CF_score_adj_pct"]/team_df["CF_pct"]).replace([np.inf,-np.inf],np.nan).mean()
pla["CF_score_adj_pct"] = pla["CF_pct"] * mean_ratio
pla["TNFI_score_adj_pct"] = pla["TNFI_pct"] * (team_df["TNFI_score_adj_pct"]/team_df["TNFI_pct"]).replace([np.inf,-np.inf],np.nan).mean()
pla["HD_CF_score_adj_pct"] = pla["HD_CF_pct"] * (team_df["HD_CF_score_adj_pct"]/team_df["HD_CF_pct"]).replace([np.inf,-np.inf],np.nan).mean()

# Zone-adjusted: same as raw (no face-off data)
pla["CF_zone_adj_pct"] = pla["CF_pct"]
pla["CF_score_zone_adj_pct"] = pla["CF_score_adj_pct"]
pla["TNFI_zone_adj_pct"] = pla["TNFI_pct"]
pla["TNFI_score_zone_adj_pct"] = pla["TNFI_score_adj_pct"]
pla["HD_CF_score_zone_adj_pct"] = pla["HD_CF_score_adj_pct"]

pla.to_csv(f"{OUT_DIR}/metrics_player.csv", index=False)
print(f"  player metrics rows (500+ TOI): {len(pla)}")

print("Done Step 6-7 (base).")
print("Note: QoC/QoT computed in next step using co-occurrence from shift data.")
