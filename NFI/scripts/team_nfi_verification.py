#!/usr/bin/env python3
"""
Team NFI% verification + attack/suppression per-game pull, all seasons
2022-23..2025-26 (identical definition per season; 2025-26 unchanged).

Methodology:
  - Source: post-audit raw events file (nhl_shot_events.v2.csv, 4-season pool
    2022-23..2025-26)
  - Seasons: 20222023, 20232024, 20242025, 20252026 (one row-block each)
  - State filter: 5v5 ES (situation_code = 1551, which guarantees no empty net)
  - Period filter: regulation only (1-3)
  - Event filter: Fenwick (shot-on-goal + missed-shot + goal); blocked-shot
    excluded
  - Zone filter: CNFI tight (x 74-89, |y| <= 9) OR MNFI tight (x 55-73, |y| <= 15)
    — FNFI explicitly excluded
  - Per shot: shooting team gains 1 attack_count; defending team gains 1
    suppress_count

  team_nfi_pct = attack / (attack + suppress)
  attack_per_game / suppress_per_game = totals / GP
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/ashgarg/Documents/HockeyROI")
SHOT_CSV = ROOT / "Data" / "nhl_shot_events.v2.csv"
TLM_CSV = ROOT / "NFI" / "output" / "team_level_all_metrics.csv"
# Write to the Streamlit app's read path (NFI/output/), not the legacy
# top-level Output/ dir, so the Teams tab picks up all seasons directly.
OUT_CSV = ROOT / "NFI" / "output" / "team_nfi_verification_and_attack_suppress.csv"

PRE_AUDIT = {
    "COL":0.5690,"OTT":0.5536,"CAR":0.5521,"VGK":0.5479,"TBL":0.5466,
    "LAK":0.5314,"UTA":0.5257,"PIT":0.5235,"PHI":0.5225,"DAL":0.5221,
    "CBJ":0.5217,"STL":0.5129,"WSH":0.5093,"EDM":0.5089,"NYR":0.5039,
    "MIN":0.5030,"ANA":0.5022,"BUF":0.5016,"FLA":0.4921,"MTL":0.4907,
    "NSH":0.4889,"NJD":0.4885,"SJS":0.4781,"WPG":0.4764,"BOS":0.4725,
    "DET":0.4680,"NYI":0.4659,"TOR":0.4572,"CGY":0.4568,"SEA":0.4556,
    "VAN":0.4308,"CHI":0.4212,
}

# ----- Provenance -----
print("="*78)
print("DATA PROVENANCE")
print("="*78)
print(f"Raw events file: {SHOT_CSV}")
mtime = pd.Timestamp(os.path.getmtime(SHOT_CSV), unit="s")
print(f"  last-modified: {mtime}")
print(f"  size:          {os.path.getsize(SHOT_CSV):,} bytes")
print(f"  row count:     ", end="", flush=True)
with open(SHOT_CSV, "rb") as f:
    n_rows = sum(1 for _ in f) - 1  # minus header
print(f"{n_rows:,}")

print(f"\nTeam metadata file: {TLM_CSV}")
print(f"  last-modified: {pd.Timestamp(os.path.getmtime(TLM_CSV), unit='s')}")

# ----- Load events -----
print("\nLoading shots ...")
df = pd.read_csv(SHOT_CSV,
                 usecols=["season","period","situation_code","event_type","game_type",
                          "shooting_team_id","shooting_team_abbrev",
                          "home_team_id","home_team_abbrev","away_team_abbrev",
                          "x_coord_norm","y_coord_norm"],
                 dtype={"season":str,"situation_code":str})
print(f"  raw rows loaded: {len(df):,}")

# GP from team_level_all_metrics (loaded once, sliced per season in the loop)
tlm = pd.read_csv(TLM_CSV, dtype={"season":str})

# Process every season with the IDENTICAL definition. 2025-26 stays byte-
# identical to the prior single-season output; the prior three seasons are
# added. `m` / `fen` are left holding the last (2025-26) season for the
# verification reporting block that follows.
SEASONS = ["20222023", "20232024", "20242025", "20252026"]
all_m = []
m = None
fen = None

for season in SEASONS:
    # Filter to <season> regulation 5v5 ES.
    # game_type filter prevents playoff contamination (needed for prior seasons).
    sub = df[(df["season"]==season) & (df["game_type"]=="regular") &
             (df["period"].between(1,3)) & (df["situation_code"].astype(str)=="1551")].copy()
    print(f"\n[{season}] after season/period/state filter (5v5 ES, reg): {len(sub):,}")

    # Event-type counts BEFORE Fenwick filter
    print(f"  Event type breakdown (pre-Fenwick filter):")
    for et, n in sub["event_type"].value_counts().items():
        print(f"    {et:<14} {n:,}")

    # Fenwick filter (drop blocked-shot)
    fen = sub[sub["event_type"].isin(["shot-on-goal","missed-shot","goal"])].copy()
    fen = fen.dropna(subset=["x_coord_norm","y_coord_norm"])
    print(f"  Fenwick (SOG + missed + goal): {len(fen):,}")

    # Zone classification (NO blocks → no abs() coordinate fix needed; SOG/missed/
    # goal events have correct coords)
    fen["abs_y"] = fen["y_coord_norm"].abs()
    fen["zone"] = np.where(
        (fen["x_coord_norm"].between(74, 89)) & (fen["abs_y"] <= 9), "CNFI",
        np.where((fen["x_coord_norm"].between(55, 73)) & (fen["abs_y"] <= 15), "MNFI",
        np.where((fen["x_coord_norm"].between(25, 54)) & (fen["abs_y"] <= 15), "FNFI", "OTHER")))

    print(f"  Zone tally (post-coord-filter, Fenwick):")
    for z, n in fen["zone"].value_counts().items():
        print(f"    {z:<8} {n:,}")

    # Defending team (the team that did NOT shoot)
    fen["_shoot_home"] = fen["shooting_team_id"]==fen["home_team_id"]
    fen["defending_team_abbrev"] = np.where(fen["_shoot_home"],
                                              fen["away_team_abbrev"],
                                              fen["home_team_abbrev"])

    # Restrict to CNFI + MNFI only (FNFI explicitly excluded)
    cmnfi = fen[fen["zone"].isin(["CNFI","MNFI"])].copy()
    print(f"  CNFI + MNFI only (FNFI excluded): {len(cmnfi):,} "
          f"| FNFI events excluded: {(fen['zone']=='FNFI').sum():,}")

    # ----- Per-team aggregation -----
    attack = cmnfi.groupby("shooting_team_abbrev").size().rename("attack_count").reset_index()\
                  .rename(columns={"shooting_team_abbrev":"team"})
    suppress = cmnfi.groupby("defending_team_abbrev").size().rename("suppress_count").reset_index()\
                    .rename(columns={"defending_team_abbrev":"team"})

    gp_season = tlm[tlm["season"]==season][["team","gp"]].copy()

    m = gp_season.merge(attack, on="team", how="left").merge(suppress, on="team", how="left")
    m[["attack_count","suppress_count"]] = m[["attack_count","suppress_count"]].fillna(0).astype(int)
    m["total_events"] = m["attack_count"] + m["suppress_count"]
    m["attack_per_game"] = m["attack_count"] / m["gp"]
    m["suppress_per_game"] = m["suppress_count"] / m["gp"]
    m["team_nfi_pct_post_audit"] = m["attack_count"] / m["total_events"]

    # Per-season post-audit rank is always valid.
    m = m.sort_values("team_nfi_pct_post_audit", ascending=False).reset_index(drop=True)
    m["post_audit_rank"] = m["team_nfi_pct_post_audit"].rank(ascending=False, method="min").astype(int)

    # PRE_AUDIT comparison columns are a 2025-26 snapshot only — leave NaN for
    # prior seasons (the attack/suppress/per_game columns are what matter and are
    # correct for every season).
    if season == "20252026":
        m["team_nfi_pct_pre_audit"] = m["team"].map(PRE_AUDIT)
        m["delta_value"] = m["team_nfi_pct_post_audit"] - m["team_nfi_pct_pre_audit"]
        pre_rank = (pd.Series(PRE_AUDIT).sort_values(ascending=False)
                    .rank(ascending=False, method="min").astype(int))
        m["pre_audit_rank"] = m["team"].map(pre_rank.to_dict())
        m["delta_rank"] = m["post_audit_rank"] - m["pre_audit_rank"]
    else:
        m["team_nfi_pct_pre_audit"] = np.nan
        m["delta_value"] = np.nan
        m["pre_audit_rank"] = np.nan
        m["delta_rank"] = np.nan

    # CNFI+MNFI-only column from CSV: TLM does not have one. Document that.
    m["nfi_pct_in_csv"] = np.nan
    m["csv_match_check"] = "N/A — no CNFI+MNFI-only share column in team_level_all_metrics; recomputed from raw counts"

    # Rank columns → nullable Int64 so 2025-26 stays integer and prior-season
    # blanks render as empty (not "5.0") in the combined CSV.
    for c in ("post_audit_rank", "pre_audit_rank", "delta_rank"):
        m[c] = m[c].astype("Int64")

    # Reorder columns (season first)
    m = m.rename(columns={"gp":"games_played"})
    m["season"] = season
    cols = ["season","team","games_played","attack_count","suppress_count","total_events",
            "attack_per_game","suppress_per_game",
            "team_nfi_pct_post_audit","team_nfi_pct_pre_audit","delta_value",
            "post_audit_rank","pre_audit_rank","delta_rank",
            "nfi_pct_in_csv","csv_match_check"]
    m = m[cols]
    m = m.sort_values("post_audit_rank").reset_index(drop=True)
    all_m.append(m)

# Combine all seasons → single file at the app's read path.
combined = pd.concat(all_m, ignore_index=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(OUT_CSV, index=False)
print(f"\nWrote {len(combined)} rows ({len(SEASONS)} seasons) → {OUT_CSV}")

# ============================================================================
# Reporting
# ============================================================================
pd.set_option("display.max_rows", 50)
pd.set_option("display.width", 200)

print("\n" + "="*78)
print("1. FULL VERIFICATION TABLE (32 teams)")
print("="*78)
disp = m.copy()
disp["attack_per_game"] = disp["attack_per_game"].round(1)
disp["suppress_per_game"] = disp["suppress_per_game"].round(1)
disp["pre"] = disp["team_nfi_pct_pre_audit"].round(4)
disp["post"] = disp["team_nfi_pct_post_audit"].round(4)
disp["delta"] = disp["delta_value"].round(4)
print(disp[["team","games_played","attack_per_game","suppress_per_game",
              "pre","post","delta",
              "pre_audit_rank","post_audit_rank","delta_rank"]].to_string(index=False))

# ----- Audit impact summary -----
print("\n" + "="*78)
print("2. AUDIT IMPACT SUMMARY")
print("="*78)
abs_rank_delta = m["delta_rank"].abs()
shifts_1plus = (abs_rank_delta >= 1).sum()
shifts_3plus = (abs_rank_delta >= 3).sum()
largest_rank_idx = abs_rank_delta.idxmax()
largest_rank_team = m.loc[largest_rank_idx, "team"]
largest_rank_shift = m.loc[largest_rank_idx, "delta_rank"]
abs_value_delta = m["delta_value"].abs()
largest_value_idx = abs_value_delta.idxmax()
largest_value_team = m.loc[largest_value_idx, "team"]
largest_value_shift = m.loc[largest_value_idx, "delta_value"]
mean_abs = abs_value_delta.mean()
print(f"  Teams with |delta_rank| >= 1:          {shifts_1plus} of 32")
print(f"  Teams with |delta_rank| >= 3:          {shifts_3plus} of 32")
print(f"  Largest rank shift:                    {largest_rank_team} ({largest_rank_shift:+d} spots)")
print(f"  Largest value shift:                   {largest_value_team} ({largest_value_shift:+.4f} = {largest_value_shift*100:+.2f} pp)")
print(f"  Mean |delta_value| across 32 teams:    {mean_abs:.5f} ({mean_abs*100:.3f} pp)")

# ----- Sanity checks -----
print("\n" + "="*78)
print("3. SANITY CHECKS")
print("="*78)

n_teams = len(m)
chk1 = "PASS" if n_teams == 32 else "FAIL"
print(f"  [{chk1}] All 32 teams found data: {n_teams}/32")

bad_gp = m[m["games_played"] != 82]
chk2 = "PASS" if len(bad_gp) == 0 else "FAIL"
print(f"  [{chk2}] All teams played 82 games: {(m['games_played']==82).sum()}/32 at 82 GP")
if len(bad_gp):
    for _, r in bad_gp.iterrows():
        print(f"         flagged: {r['team']} GP={r['games_played']}")

in_range = ((m["team_nfi_pct_post_audit"] >= 0.30) &
            (m["team_nfi_pct_post_audit"] <= 0.70)).all()
chk3 = "PASS" if in_range else "FAIL"
print(f"  [{chk3}] Every team_nfi_pct in [0.30, 0.70]: range = "
      f"[{m['team_nfi_pct_post_audit'].min():.4f}, {m['team_nfi_pct_post_audit'].max():.4f}]")

col_row = m[m["team"]=="COL"].iloc[0]
col_apg = col_row["attack_per_game"]
col_spg = col_row["suppress_per_game"]
print(f"\n  COL per-game leak check:")
print(f"    attack_per_game:   {col_apg:.2f}  (pre-audit ref ~14.5; post-audit expected ~11.0+)")
print(f"    suppress_per_game: {col_spg:.2f}  (pre-audit ref ~11.0; post-audit expected ~8.4+)")
if col_apg > 16:
    print(f"    [FAIL] FNFI LEAK SUSPECTED — attack_per_game = {col_apg:.2f} > 16")
elif col_apg >= 9 and col_apg <= 16:
    print(f"    [PASS] attack_per_game in plausible range — no leak suspected")
else:
    print(f"    [WARN] attack_per_game = {col_apg:.2f} is below pre/post reference range")

sum_attack = m["attack_count"].sum()
sum_suppress = m["suppress_count"].sum()
chk5 = "PASS" if sum_attack == sum_suppress else "FAIL"
print(f"\n  [{chk5}] League-level attack sum == suppress sum: "
      f"attack {sum_attack:,} | suppress {sum_suppress:,}")

print(f"\n  Per-team total event check (attack + suppress):")
median_total = m["total_events"].median()
print(f"    median:  {median_total:.0f}")
print(f"    flagged (< 50% of median or > 200%):")
flagged = m[(m["total_events"] < 0.5*median_total) | (m["total_events"] > 2.0*median_total)]
if len(flagged) == 0:
    print(f"      none")
else:
    for _, r in flagged.iterrows():
        print(f"      {r['team']}: total {r['total_events']:,} (median {median_total:.0f})")

fnfi_count = (fen["zone"]=="FNFI").sum()
print(f"\n  FNFI events found in 2025-26 reg 5v5 Fenwick set: {fnfi_count:,}")
print(f"  FNFI events excluded from team NFI%:                yes")

# ----- Provenance -----
print("\n" + "="*78)
print("4. DATA PROVENANCE (recap)")
print("="*78)
print(f"  Raw events file:        {SHOT_CSV}")
print(f"    last-modified:        {mtime}")
print(f"    row count:            {n_rows:,}")
print(f"  team_level_all_metrics: {TLM_CSV}")
print(f"    columns referenced:   gp (only — for games-played count)")
print(f"    columns NOT used:     TNFI_pct (would include FNFI)")
print()
print(f"  Output:                 {OUT_CSV}")
