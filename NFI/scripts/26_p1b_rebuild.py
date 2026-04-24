#!/usr/bin/env python3
"""
Rebuild P1b — Rebound Arrival Rate (forwards) — using rebuilt
rebound_sequences.csv.

Definition (matching prior P1b convention):
  P1b = (player's in-corridor rebound attempts) / (league-wide in-corridor
                                                    rebound attempts)
  -> "What share of league-wide doorstep rebound chances did this forward
      personally capture?"

Filters:
  - Seasons 20212022 .. 20252026 (exclude 2020-21)
  - ES (situation_code = 1551)
  - Doorstep corridor: 74 <= reb_x <= 89 AND -9 <= reb_y <= 9
  - Time window: time_gap_secs <= 2 (drop 3-second rebounds)
  - Forwards only (pos_group == 'F')
  - Min 30 rebound attempts

Wilson 95% CIs on the proportion (k/n).

Output: NFI/output/P1b_rebound_arrival.csv
"""
import math
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
SEQ_CSV = f"{ROOT}/Data/rebound_sequences.csv"
POS_CSV = f"{ROOT}/NFI/output/player_positions.csv"
OLD_CSV = f"{ROOT}/NFI/output/_pre_corsi_backup/P1b_rebound_arrival_OLD.csv"
OUT_CSV = f"{ROOT}/NFI/output/P1b_rebound_arrival.csv"

SEASONS = [20212022, 20212022, 20222023, 20232024, 20242025, 20252026]
SEASONS = sorted(set(SEASONS))

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0.0, c-h), min(1.0, c+h))

# ---- Load rebounds ----
print("Loading rebound_sequences.csv ...")
reb = pd.read_csv(SEQ_CSV)
print(f"  {len(reb):,} pairs total")

# Apply filters
reb = reb[reb["season"].isin(SEASONS)]
reb = reb[reb["situation_code"]==1551]
reb = reb[reb["time_gap_secs"] <= 2]
# Doorstep corridor on REBOUND coords
reb = reb[(reb["reb_x"] >= 74) & (reb["reb_x"] <= 89)]
reb = reb[(reb["reb_y"] >= -9) & (reb["reb_y"] <= 9)]
reb = reb.dropna(subset=["reb_shooter_id"])
reb["reb_shooter_id"] = reb["reb_shooter_id"].astype(int)
print(f"  after season/ES/window/corridor filters: {len(reb):,}")

# League denominator
league_denom = len(reb)
print(f"  league_denom (total in-corridor rebound attempts): {league_denom:,}")

# Per-shooter counts
counts = reb.groupby("reb_shooter_id").size().reset_index(name="reb_attempts")

# Filter to forwards
pos = pd.read_csv(POS_CSV, dtype={"player_id":int})
fwd_ids = set(pos[pos["pos_group"]=="F"]["player_id"].tolist())
name_map = dict(zip(pos["player_id"], pos["player_name"]))
counts = counts[counts["reb_shooter_id"].isin(fwd_ids)]
print(f"  forwards with ≥1 in-corridor rebound: {len(counts)}")

# Min 30 rebound attempts
counts = counts[counts["reb_attempts"] >= 30].copy()
print(f"  forwards meeting ≥30 rebound attempts: {len(counts)}")

# Compute P1b + Wilson CI
counts["league_denom"] = league_denom
ps, los, his = [], [], []
for k in counts["reb_attempts"].values:
    p, lo, hi = wilson(int(k), league_denom)
    ps.append(p); los.append(lo); his.append(hi)
counts["P1b"]     = np.round(ps, 6)
counts["P1b_lo"]  = np.round(los, 6)
counts["P1b_hi"]  = np.round(his, 6)
counts["P1b_pct"]    = np.round(counts["P1b"]    * 100, 4)
counts["P1b_lo_pct"] = np.round(counts["P1b_lo"] * 100, 4)
counts["P1b_hi_pct"] = np.round(counts["P1b_hi"] * 100, 4)
counts["player_name"] = counts["reb_shooter_id"].map(name_map)
counts = counts.rename(columns={"reb_shooter_id":"player_id"})

counts = counts.sort_values("P1b", ascending=False).reset_index(drop=True)
counts["P1b_rank"] = counts["P1b"].rank(ascending=False, method="min").astype(int)

cols = ["player_id","player_name","reb_attempts","league_denom",
        "P1b","P1b_lo","P1b_hi","P1b_pct","P1b_lo_pct","P1b_hi_pct","P1b_rank"]
counts = counts[cols]
counts.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV} ({len(counts)} forwards)")

# ---- Compare to old P1b ----
print("\n=== Comparison to OLD P1b ===")
old = pd.read_csv(OLD_CSV)
print(f"  OLD n forwards: {len(old)}")
print(f"  NEW n forwards: {len(counts)}")
print(f"  Forwards in both: {len(set(old['player_id'])&set(counts['player_id']))}")
print(f"  In OLD only: {len(set(old['player_id'])-set(counts['player_id']))}")
print(f"  In NEW only: {len(set(counts['player_id'])-set(old['player_id']))}")

# Re-rank both for comparison
old = old.sort_values("P1b", ascending=False).reset_index(drop=True)
old["old_rank"] = np.arange(1, len(old)+1)
counts["new_rank"] = counts["P1b_rank"]
m = old[["player_id","player_name","P1b","old_rank","reb_attempts"]]\
    .rename(columns={"P1b":"P1b_old","reb_attempts":"reb_attempts_old"}).merge(
    counts[["player_id","P1b","new_rank","reb_attempts"]]\
    .rename(columns={"P1b":"P1b_new","reb_attempts":"reb_attempts_new"}),
    on="player_id", how="inner")
m["rank_delta"] = m["old_rank"] - m["new_rank"]   # +ve = improved
m["abs_delta"]  = m["rank_delta"].abs()

big_movers_up = m.sort_values("rank_delta", ascending=False).head(15)
big_movers_dn = m.sort_values("rank_delta").head(15)
print(f"\nForwards moved >10 ranks: {(m['abs_delta']>10).sum()}")
print(f"Forwards moved >25 ranks: {(m['abs_delta']>25).sum()}")
print(f"Forwards moved >50 ranks: {(m['abs_delta']>50).sum()}")

pd.options.display.float_format = '{:.6f}'.format
pd.options.display.width = 200
pd.options.display.max_columns = None

print("\n--- Biggest UPWARD movers (NEW rank better than OLD) ---")
print(big_movers_up[["player_name","reb_attempts_old","reb_attempts_new",
                     "P1b_old","P1b_new","old_rank","new_rank","rank_delta"]]
      .to_string(index=False))

print("\n--- Biggest DOWNWARD movers ---")
print(big_movers_dn[["player_name","reb_attempts_old","reb_attempts_new",
                     "P1b_old","P1b_new","old_rank","new_rank","rank_delta"]]
      .to_string(index=False))

# Confirm Hyman
print("\n=== Hyman check ===")
h_new = counts[counts["player_name"]=="Zach Hyman"]
h_old = old[old["player_name"]=="Zach Hyman"]
if len(h_new):
    h = h_new.iloc[0]
    print(f"  NEW: Zach Hyman rank #{int(h['P1b_rank'])} of {len(counts)}, "
          f"P1b={h['P1b']:.6f} ({h['P1b_pct']:.4f}%), reb_attempts={int(h['reb_attempts'])}")
else:
    print("  NEW: Zach Hyman NOT in qualifying list")
if len(h_old):
    h = h_old.iloc[0]
    print(f"  OLD: Zach Hyman rank #{int(h['old_rank'])} of {len(old)}, "
          f"P1b={h['P1b']:.6f}, reb_attempts={int(h['reb_attempts'])}")

print("\n=== FULL ranked list ===")
pd.options.display.float_format = '{:.4f}'.format
print(counts[["P1b_rank","player_name","reb_attempts","P1b_pct",
              "P1b_lo_pct","P1b_hi_pct"]].to_string(index=False))
