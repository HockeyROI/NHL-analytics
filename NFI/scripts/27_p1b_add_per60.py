#!/usr/bin/env python3
"""
Add P1b_per60 (TOI-normalized rebound arrival rate) to P1b_rebound_arrival.csv.

P1b_per60 = (rebound attempts in doorstep corridor) / ES TOI minutes * 60
Wilson 95% CI computed using the existing pipeline's rate_ci convention
(treat per-minute as binomial-ish; matches script 03/16/17 elsewhere).

ES TOI is read from player_toi.csv (same 5-season pool 2021-22 through
2025-26 used everywhere else in this pipeline; values were aggregated from
shift_data.csv when player_toi.csv was built).

Overwrites: NFI/output/P1b_rebound_arrival.csv
"""
import math
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
P1B_CSV = f"{OUT}/P1b_rebound_arrival.csv"
TOI_CSV = f"{OUT}/player_toi.csv"

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0.0, c-h), min(1.0, c+h))

def rate_ci(events, minutes, z=1.96):
    """Per-60 rate with Wilson-style CI (matches existing pipeline convention)."""
    if minutes <= 0: return (0.0, 0.0, 0.0)
    p, lo, hi = wilson(events, max(events, int(round(minutes))))
    return events/minutes*60.0, lo*60.0, hi*60.0

# ---- Load files ----
print("Loading P1b and player_toi ...")
p1b = pd.read_csv(P1B_CSV)
toi = pd.read_csv(TOI_CSV, dtype={"player_id":int})
toi["es_toi_min"] = toi["toi_ES_sec"] / 60.0
toi_map = dict(zip(toi["player_id"], toi["es_toi_min"]))
print(f"  P1b forwards: {len(p1b)}")
print(f"  TOI rows:     {len(toi)}")

# Attach ES TOI
p1b["es_toi_min"] = p1b["player_id"].map(toi_map)
missing = p1b[p1b["es_toi_min"].isna()]
if len(missing):
    print(f"  WARNING: {len(missing)} forwards missing ES TOI:")
    print(missing[["player_id","player_name","reb_attempts"]].to_string(index=False))

# Compute P1b_per60 with Wilson CI
rs, los, his = [], [], []
for k, t in zip(p1b["reb_attempts"].values, p1b["es_toi_min"].values):
    if pd.isna(t) or t <= 0:
        rs.append(np.nan); los.append(np.nan); his.append(np.nan); continue
    r, lo, hi = rate_ci(int(k), float(t))
    rs.append(round(r, 4)); los.append(round(lo, 4)); his.append(round(hi, 4))
p1b["P1b_per60"]      = rs
p1b["P1b_per60_lo95"] = los
p1b["P1b_per60_hi95"] = his

# Rank by per-60
p1b["P1b_per60_rank"] = p1b["P1b_per60"].rank(ascending=False, method="min").astype(int)
# Existing P1b_rank already in file (share of league); compute delta
p1b["rank_delta_pct_vs_per60"] = p1b["P1b_rank"] - p1b["P1b_per60_rank"]
# +ve = per60 ranks BETTER than share-of-league (TOI penalty in old metric)
# -ve = per60 ranks WORSE  (player benefited from high TOI in share-of-league)
p1b["toi_flag"] = np.where(p1b["rank_delta_pct_vs_per60"].abs() > 20,
    np.where(p1b["rank_delta_pct_vs_per60"] > 0, "TOI_DEFLATED",
                                                  "TOI_INFLATED"), "")

# Sort by per-60
p1b = p1b.sort_values("P1b_per60", ascending=False).reset_index(drop=True)

cols = ["player_id","player_name","reb_attempts","league_denom","es_toi_min",
        "P1b","P1b_lo","P1b_hi","P1b_pct","P1b_lo_pct","P1b_hi_pct","P1b_rank",
        "P1b_per60","P1b_per60_lo95","P1b_per60_hi95","P1b_per60_rank",
        "rank_delta_pct_vs_per60","toi_flag"]
p1b = p1b[cols]
p1b.to_csv(P1B_CSV, index=False)
print(f"\nOverwrote {P1B_CSV} ({len(p1b)} forwards)")

# ---- Reporting ----
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.width = 220
pd.options.display.max_columns = None

print("\n=== TOP 15 by P1b_per60 ===")
print(p1b.head(15)[["P1b_per60_rank","player_name","reb_attempts","es_toi_min",
                    "P1b_per60","P1b_per60_lo95","P1b_per60_hi95",
                    "P1b_rank","rank_delta_pct_vs_per60","toi_flag"]]
      .to_string(index=False))

print("\n=== BOTTOM 5 by P1b_per60 ===")
print(p1b.tail(5)[["P1b_per60_rank","player_name","reb_attempts","es_toi_min",
                   "P1b_per60","P1b_per60_lo95","P1b_per60_hi95",
                   "P1b_rank","rank_delta_pct_vs_per60","toi_flag"]]
      .to_string(index=False))

print("\n=== TOI_DEFLATED (low-TOI forwards: per-60 ranks BETTER than share-of-league) ===")
defl = p1b[p1b["toi_flag"]=="TOI_DEFLATED"].sort_values("rank_delta_pct_vs_per60", ascending=False)
print(f"  count: {len(defl)}")
print(defl[["player_name","reb_attempts","es_toi_min","P1b_pct",
            "P1b_per60","P1b_rank","P1b_per60_rank","rank_delta_pct_vs_per60"]]
      .head(20).to_string(index=False))

print("\n=== TOI_INFLATED (high-TOI forwards: per-60 ranks WORSE than share-of-league) ===")
infl = p1b[p1b["toi_flag"]=="TOI_INFLATED"].sort_values("rank_delta_pct_vs_per60")
print(f"  count: {len(infl)}")
print(infl[["player_name","reb_attempts","es_toi_min","P1b_pct",
            "P1b_per60","P1b_rank","P1b_per60_rank","rank_delta_pct_vs_per60"]]
      .head(20).to_string(index=False))

print(f"\nTotal flagged (|rank delta| > 20): {len(defl)+len(infl)} of {len(p1b)}")

# Hyman / Lee context
print("\n=== Hyman / Lee context ===")
for n in ["Anders Lee","Zach Hyman","John Tavares","Auston Matthews","Connor McDavid",
          "Sam Bennett","Brady Tkachuk","Nathan MacKinnon"]:
    r = p1b[p1b["player_name"]==n]
    if len(r):
        r = r.iloc[0]
        print(f"  {n:<22} reb={int(r['reb_attempts']):>3}  TOI={r['es_toi_min']:>7.0f}  "
              f"P1b%={r['P1b_pct']:.4f}  rank {int(r['P1b_rank']):>3}  "
              f"P1b/60={r['P1b_per60']:.4f}  rank {int(r['P1b_per60_rank']):>3}  "
              f"(Δ {int(r['rank_delta_pct_vs_per60']):+d})")
