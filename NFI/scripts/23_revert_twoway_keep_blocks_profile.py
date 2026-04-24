#!/usr/bin/env python3
"""
Revert two-way scores to Fenwick-based positional suppression (P2_weighted /
P4_weighted), and create defensive *profiles* that show blocks credit alongside
positional suppression as a separate diagnostic column (no subtraction).

Why: P2_net / P4_net rewards physical shot-blocking but blind-spots elite
positional defenders (Slavin, Makar, Toews, Hughes) who prevent attempts at
the source. Keeping the two-way score on Fenwick weighted suppression
preserves the original ranking of those defenders. Blocks data is retained as
a standalone column for users who want to inspect physical-vs-positional
defensive style.

Outputs (NFI/output):
  P2_defensive_forwards.csv  (RESTORED to Fenwick from backup)
  P4_defensive_D.csv          (RESTORED to Fenwick from backup)
  P2_defensive_profiles.csv  (NEW — Fenwick P2 enriched with blocks columns)
  P4_defensive_profiles.csv  (NEW — Fenwick P4 enriched with blocks columns)
  twoway_forward_score.csv    (REBUILT — z(P1a_w_total) - z(P2_weighted))
  twoway_D_score.csv          (REBUILT — z(P5_weighted) - z(P4_weighted))
"""
import shutil
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT = f"{ROOT}/NFI/output"
BK  = f"{OUT}/_pre_corsi_backup"

# ----- 1. Restore Fenwick P2 / P4 from backup -----
print("Restoring Fenwick P2/P4 from backup ...")
shutil.copy(f"{BK}/P2_defensive_forwards.csv", f"{OUT}/P2_defensive_forwards.csv")
shutil.copy(f"{BK}/P4_defensive_D.csv",        f"{OUT}/P4_defensive_D.csv")
p2 = pd.read_csv(f"{OUT}/P2_defensive_forwards.csv")
p4 = pd.read_csv(f"{OUT}/P4_defensive_D.csv")
print(f"  P2 (Fenwick): {len(p2)} F  |  P4 (Fenwick): {len(p4)} D")

# ----- 2. Pull blocks columns from the just-built P2_net / P4_net files -----
print("Pulling blocks columns from P2_net / P4_net ...")
p2_net = pd.read_csv(f"{OUT}/P2_net.csv")
p4_net = pd.read_csv(f"{OUT}/P4_net.csv")
blocks_F = p2_net[["player_id","blocks_total","blocks_CNFI","blocks_MNFI",
                   "blocks_FNFI","blocks_per60","blocks_xG_per60","P2_net"]]\
              .rename(columns={"blocks_xG_per60":"blocks_xG_prevented_per60"})
blocks_D = p4_net[["player_id","blocks_total","blocks_CNFI","blocks_MNFI",
                   "blocks_FNFI","blocks_per60","blocks_xG_per60","P4_net"]]\
              .rename(columns={"blocks_xG_per60":"blocks_xG_prevented_per60"})

# ----- 3. Build profile files (Fenwick base + blocks side-by-side) -----
print("Building P2_defensive_profiles.csv and P4_defensive_profiles.csv ...")
p2_prof = p2.merge(blocks_F, on="player_id", how="left").fillna(
    {"blocks_total":0,"blocks_CNFI":0,"blocks_MNFI":0,"blocks_FNFI":0,
     "blocks_per60":0.0,"blocks_xG_prevented_per60":0.0,"P2_net":np.nan})
p2_prof_cols = ["player_id","player_name","es_toi_min",
                "CNFI_SA","CNFI_SA_per60","CNFI_SA_lo95","CNFI_SA_hi95","CNFI_centrality",
                "MNFI_SA","MNFI_SA_per60","MNFI_SA_lo95","MNFI_SA_hi95","MNFI_centrality",
                "FNFI_SA","FNFI_SA_per60","FNFI_SA_lo95","FNFI_SA_hi95","FNFI_centrality",
                "TNFI_SA","TNFI_SA_per60","TNFI_SA_lo95","TNFI_SA_hi95",
                "P2_weighted",                       # positional suppression (Fenwick xG/60)
                "blocks_total","blocks_CNFI","blocks_MNFI","blocks_FNFI",
                "blocks_per60","blocks_xG_prevented_per60",
                "P2_net"]                            # diagnostic only — NOT used in two-way
p2_prof = p2_prof[p2_prof_cols].sort_values("P2_weighted").reset_index(drop=True)
p2_prof.to_csv(f"{OUT}/P2_defensive_profiles.csv", index=False)

p4_prof = p4.merge(blocks_D, on="player_id", how="left").fillna(
    {"blocks_total":0,"blocks_CNFI":0,"blocks_MNFI":0,"blocks_FNFI":0,
     "blocks_per60":0.0,"blocks_xG_prevented_per60":0.0,"P4_net":np.nan})
p4_prof_cols = ["player_id","player_name","es_toi_min",
                "CNFI_SA","CNFI_SA_per60","CNFI_SA_lo95","CNFI_SA_hi95",
                "MNFI_SA","MNFI_SA_per60","MNFI_SA_lo95","MNFI_SA_hi95",
                "FNFI_SA","FNFI_SA_per60","FNFI_SA_lo95","FNFI_SA_hi95",
                "TNFI_SA","TNFI_SA_per60","TNFI_SA_lo95","TNFI_SA_hi95",
                "P4_weighted",                       # positional suppression (Fenwick xG/60)
                "blocks_total","blocks_CNFI","blocks_MNFI","blocks_FNFI",
                "blocks_per60","blocks_xG_prevented_per60",
                "P4_net"]                            # diagnostic only — NOT used in two-way
p4_prof = p4_prof[p4_prof_cols].sort_values("P4_weighted").reset_index(drop=True)
p4_prof.to_csv(f"{OUT}/P4_defensive_profiles.csv", index=False)
print(f"  P2_defensive_profiles.csv: {len(p2_prof)} F")
print(f"  P4_defensive_profiles.csv: {len(p4_prof)} D")

# ----- 4. Rebuild two-way scores using Fenwick P2_weighted / P4_weighted -----
print("Rebuilding two-way scores (Fenwick-based defensive z) ...")
p1a = pd.read_csv(f"{OUT}/P1a_centrality_weighted.csv")
p5  = pd.read_csv(f"{OUT}/P5_offensive_D.csv")

p1a["P1a_weighted_total"] = p1a["P1a_weighted_CNFI"] + p1a["P1a_weighted_MNFI"]
twf = p2[["player_id","player_name","es_toi_min","P2_weighted"]].merge(
    p1a[["player_id","P1a_weighted_CNFI","P1a_weighted_MNFI",
         "P1a_weighted_total"]], on="player_id", how="inner")

def z(s): return (s - s.mean()) / s.std(ddof=0)

twf["z_P1a_weighted"] = z(twf["P1a_weighted_total"])
twf["z_P2_weighted"]  = z(twf["P2_weighted"])
twf["twoway_score"]   = twf["z_P1a_weighted"] - twf["z_P2_weighted"]
twf["off_rank"]       = twf["P1a_weighted_total"].rank(ascending=False, method="min").astype(int)
twf["def_rank"]       = twf["P2_weighted"].rank(ascending=True,  method="min").astype(int)
twf["twoway_rank"]    = twf["twoway_score"].rank(ascending=False, method="min").astype(int)
twf = twf.sort_values("twoway_score", ascending=False).reset_index(drop=True)
twf_cols = ["player_id","player_name","es_toi_min",
            "P1a_weighted_CNFI","P1a_weighted_MNFI","P1a_weighted_total",
            "P2_weighted","z_P1a_weighted","z_P2_weighted","twoway_score",
            "off_rank","def_rank","twoway_rank"]
twf = twf[twf_cols]
twf.to_csv(f"{OUT}/twoway_forward_score.csv", index=False)
print(f"  twoway_forward_score.csv: {len(twf)} F")

twd = p4[["player_id","player_name","es_toi_min","TNFI_SA_per60","P4_weighted"]].merge(
    p5[["player_id","TNFI_SF_per60","P5_weighted"]], on="player_id", how="inner")
twd["z_P5_weighted"]  = z(twd["P5_weighted"])
twd["z_P4_weighted"]  = z(twd["P4_weighted"])
twd["twoway_D_score"] = twd["z_P5_weighted"] - twd["z_P4_weighted"]
twd["off_rank"]       = twd["P5_weighted"].rank(ascending=False, method="min").astype(int)
twd["def_rank"]       = twd["P4_weighted"].rank(ascending=True,  method="min").astype(int)
twd["twoway_D_rank"]  = twd["twoway_D_score"].rank(ascending=False, method="min").astype(int)
twd = twd.sort_values("twoway_D_score", ascending=False).reset_index(drop=True)
twd_cols = ["player_id","player_name","es_toi_min",
            "TNFI_SA_per60","P4_weighted","TNFI_SF_per60","P5_weighted",
            "z_P5_weighted","z_P4_weighted","twoway_D_score",
            "off_rank","def_rank","twoway_D_rank"]
twd = twd[twd_cols]
twd.to_csv(f"{OUT}/twoway_D_score.csv", index=False)
print(f"  twoway_D_score.csv: {len(twd)} D")

# ----- 5. Spot-checks: Florida core, elite positional D -----
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.width = 240
pd.options.display.max_columns = None

print("\n=== SPOT-CHECK Florida core / Avs core (forwards): two-way ranks ===")
named_F = ["Aleksander Barkov","Sam Bennett","Sam Reinhart","Carter Verhaeghe",
           "Anthony Cirelli","Eetu Luostarinen","Matthew Tkachuk",
           "Nathan MacKinnon","Mikko Rantanen","Artturi Lehkonen",
           "Valeri Nichushkin","Patrice Bergeron"]
for n in named_F:
    r = twf[twf["player_name"]==n]
    if len(r): r = r.iloc[0]
    else:
        print(f"  {n}: not found")
        continue
    print(f"  {n:<22}  twoway {r['twoway_score']:+.3f}  rank {int(r['twoway_rank']):>3}/{len(twf)}  "
          f"(off {int(r['off_rank']):>3}, def {int(r['def_rank']):>3})")

print("\n=== SPOT-CHECK Elite positional D: two-way ranks ===")
named_D = ["Cale Makar","Devon Toews","Jaccob Slavin","Quinn Hughes",
           "Mattias Ekholm","Aaron Ekblad","Gustav Forsling",
           "Brent Burns","Dougie Hamilton","Samuel Girard"]
for n in named_D:
    r = twd[twd["player_name"]==n]
    if len(r): r = r.iloc[0]
    else:
        print(f"  {n}: not found")
        continue
    print(f"  {n:<22}  twoway {r['twoway_D_score']:+.3f}  rank {int(r['twoway_D_rank']):>3}/{len(twd)}  "
          f"(off {int(r['off_rank']):>3}, def {int(r['def_rank']):>3})")

# ----- 6. Top 20s -----
print("\n=== TOP 20 TWO-WAY FORWARDS (Fenwick P2_weighted) ===")
print(twf.head(20)[["player_name","es_toi_min","P1a_weighted_total","P2_weighted",
                    "z_P1a_weighted","z_P2_weighted","twoway_score",
                    "off_rank","def_rank","twoway_rank"]].to_string(index=False))

print("\n=== TOP 20 TWO-WAY D (Fenwick P4_weighted) ===")
print(twd.head(20)[["player_name","es_toi_min","P5_weighted","P4_weighted",
                    "z_P5_weighted","z_P4_weighted","twoway_D_score",
                    "off_rank","def_rank","twoway_D_rank"]].to_string(index=False))

# ----- 7. Confirm match to original (pre-Corsi, pre-blocks) two-way scores -----
print("\n=== Comparison to pre-Corsi backup (sanity check that we restored properly) ===")
twf_old = pd.read_csv(f"{BK}/twoway_forward_score.csv")
twd_old = pd.read_csv(f"{BK}/twoway_D_score.csv")
mF = twf_old[["player_id","twoway_score","twoway_rank"]].rename(
        columns={"twoway_score":"old_score","twoway_rank":"old_rank"}).merge(
     twf[["player_id","player_name","twoway_score","twoway_rank"]], on="player_id")
mF["score_diff"]   = (mF["twoway_score"] - mF["old_score"]).round(4)
mF["rank_diff"]    = mF["twoway_rank"] - mF["old_rank"]
print(f"\nForwards matched: {len(mF)}/{len(twf_old)}")
print(f"  Mean abs(score_diff) vs pre-Corsi: {mF['score_diff'].abs().mean():.5f}")
print(f"  Mean abs(rank_diff)  vs pre-Corsi: {mF['rank_diff'].abs().mean():.2f}")
print(f"  Max  abs(rank_diff)  vs pre-Corsi: {mF['rank_diff'].abs().max():.0f}")
mD = twd_old[["player_id","twoway_D_score","twoway_D_rank"]].rename(
        columns={"twoway_D_score":"old_score","twoway_D_rank":"old_rank"}).merge(
     twd[["player_id","player_name","twoway_D_score","twoway_D_rank"]], on="player_id")
mD["score_diff"] = (mD["twoway_D_score"] - mD["old_score"]).round(4)
mD["rank_diff"]  = mD["twoway_D_rank"] - mD["old_rank"]
print(f"\nD matched: {len(mD)}/{len(twd_old)}")
print(f"  Mean abs(score_diff) vs pre-Corsi: {mD['score_diff'].abs().mean():.5f}")
print(f"  Mean abs(rank_diff)  vs pre-Corsi: {mD['rank_diff'].abs().mean():.2f}")
print(f"  Max  abs(rank_diff)  vs pre-Corsi: {mD['rank_diff'].abs().max():.0f}")

print("\nFiles written:")
for f in ["P2_defensive_forwards.csv (restored)",
          "P4_defensive_D.csv (restored)",
          "P2_defensive_profiles.csv (new)",
          "P4_defensive_profiles.csv (new)",
          "twoway_forward_score.csv (rebuilt — Fenwick)",
          "twoway_D_score.csv (rebuilt — Fenwick)"]:
    print(f"  {OUT}/{f}")
