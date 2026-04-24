"""Top 200 player article dataset.

Columns per player (sort by NFI_pct_ZA_pooled desc):
  - NFI_pct_ZA_pooled    (career TOI-weighted mean, min 2000 ES min pooled)
  - NFI_pct_3A_pooled
  - NFI_pct_ZA_2526      (2025-26 single season)
  - NFI_pct_3A_2526
  - rank_change          (rank_ZA_pooled − rank_3A_pooled; positive = undervalued)
  - NFI_pct_3A_MOM_2526
  - toi_total, toi_2526
  - flags: small_current_sample, big_mover, ascending_momentum

Highlight lists:
  - Top 20 biggest risers (rank_change > 0, most undervalued)
  - Top 20 biggest fallers (rank_change < 0, most system-dependent)
  - Top 20 strongest ascending momentum

Output: NFI/output/fully_adjusted/top200_article_dataset.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "NFI/output/fully_adjusted"
CURRENT = "20252026"
MIN_POOLED_TOI = 2000
MIN_CURRENT_TOI = 500
SMALL_CURRENT_CUTOFF = 1000  # flag as small sample if < this
BIG_MOVER_CUTOFF = 20
ASCEND_MOM_CUTOFF = 0.02     # momentum flag threshold

print("[0] loading ...")
pp = pd.read_csv(OUT / "player_fully_adjusted.csv")
pp["season"] = pp["season"].astype(str)

# Career TOI-weighted means per player
def tw_mean(vals, w):
    vals = np.array(vals, dtype=float); w = np.array(w, dtype=float)
    m = (~np.isnan(vals)) & (w>0)
    return np.average(vals[m], weights=w[m]) if m.any() else np.nan

print("[1] computing pooled career means ...")
career = []
for (pid, name, pos), g in pp.groupby(["player_id","player_name","position"]):
    toi_min = g["toi_min"] if "toi_min" in g.columns else (g.get("toi_sec", pd.Series(0))/60)
    toi_total = float(toi_min.sum())
    if toi_total < MIN_POOLED_TOI: continue
    career.append({
        "player_id": int(pid), "player_name": name, "position": pos,
        "n_seasons": int(len(g)),
        "toi_total": toi_total,
        "team_recent": g.sort_values("season")["team"].iloc[-1],
        "NFI_pct_ZA_pooled": tw_mean(g["NFI_pct_ZA"], toi_min),
        "NFI_pct_3A_pooled": tw_mean(g["NFI_pct_3A"], toi_min),
    })
career = pd.DataFrame(career)
print(f"    qualifying pool (2000+ ES min total): {len(career)} players")

# Ranks (1 = best) within the qualifying pool
career["rank_ZA_pooled"] = career["NFI_pct_ZA_pooled"].rank(ascending=False, method="min").astype(int)
career["rank_3A_pooled"] = career["NFI_pct_3A_pooled"].rank(ascending=False, method="min").astype(int)
career["rank_change"]    = career["rank_ZA_pooled"] - career["rank_3A_pooled"]

# Current season 2025-26 data
curr = pp[pp["season"]==CURRENT][["player_id","toi_min","NFI_pct_ZA","NFI_pct_3A","NFI_pct_3A_MOM"]].copy()
curr = curr.rename(columns={
    "toi_min": "toi_2526",
    "NFI_pct_ZA": "NFI_pct_ZA_2526",
    "NFI_pct_3A": "NFI_pct_3A_2526",
    "NFI_pct_3A_MOM": "NFI_pct_3A_MOM_2526",
})
career = career.merge(curr, on="player_id", how="left")

# Flags
career["small_current_sample"] = (career["toi_2526"].fillna(0) < SMALL_CURRENT_CUTOFF)
career["big_mover"] = career["rank_change"].abs() > BIG_MOVER_CUTOFF
career["ascending_momentum"] = career["NFI_pct_3A_MOM_2526"].fillna(0) > ASCEND_MOM_CUTOFF

# Top 200 by NFI_pct_ZA_pooled descending
career_sorted = career.sort_values("NFI_pct_ZA_pooled", ascending=False).reset_index(drop=True)
top200 = career_sorted.head(200).copy()
top200.insert(0, "rank", np.arange(1, len(top200)+1))

# Write
top200.to_csv(OUT / "top200_article_dataset.csv", index=False)

# ---------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------
def fmt_row(i, r):
    toi26 = r["toi_2526"] if pd.notna(r["toi_2526"]) else 0
    flags = []
    if r["small_current_sample"] or pd.isna(r["toi_2526"]): flags.append("smallcurr")
    if r["big_mover"]: flags.append("bigmove")
    if r["ascending_momentum"]: flags.append("asc")
    flag_str = " ".join(flags) if flags else ""
    mom = r["NFI_pct_3A_MOM_2526"] if pd.notna(r["NFI_pct_3A_MOM_2526"]) else np.nan
    nfi_za_26 = r["NFI_pct_ZA_2526"] if pd.notna(r["NFI_pct_ZA_2526"]) else np.nan
    return (f"{int(r['rank']):>3}  {r['player_name']:<26} {r['position']:<2} {r['team_recent']:<4}  "
            f"ZA_pool={r['NFI_pct_ZA_pooled']:.4f}  3A_pool={r['NFI_pct_3A_pooled']:.4f}  "
            f"ZA_26={nfi_za_26:.4f}" + ("  " if pd.notna(nfi_za_26) else "  NaN   ") +
            f"Δrk={r['rank_change']:+d}  MOM26={mom:+.3f}" + (f"  [{flag_str}]" if flag_str else ""))

print("\n" + "="*110)
print(f"TOP 50 — sorted by NFI_pct_ZA_pooled (of {len(career_sorted)} qualifying players with ≥ 2000 ES min)")
print("="*110)
for i, r in top200.head(50).iterrows():
    print(fmt_row(i, r))

# Risers / fallers (within the 200-player pool, biggest rank changes)
print("\n" + "="*110)
print("TOP 20 RISERS (ZA → 3A) — most undervalued by ZA, rewarded by 3A")
print("="*110)
risers = career_sorted.sort_values("rank_change", ascending=False).head(20).reset_index(drop=True)
for i, r in risers.iterrows():
    print(f"{i+1:>3}  {r['player_name']:<26} {r['position']:<2} {r['team_recent']:<4}  "
          f"rank_ZA={r['rank_ZA_pooled']:>4}  rank_3A={r['rank_3A_pooled']:>4}  Δ={r['rank_change']:+d}  "
          f"ZA_pool={r['NFI_pct_ZA_pooled']:.4f}  3A_pool={r['NFI_pct_3A_pooled']:.4f}")

print("\n" + "="*110)
print("TOP 20 FALLERS (ZA → 3A) — most system-dependent, penalized by 3A")
print("="*110)
fallers = career_sorted.sort_values("rank_change", ascending=True).head(20).reset_index(drop=True)
for i, r in fallers.iterrows():
    print(f"{i+1:>3}  {r['player_name']:<26} {r['position']:<2} {r['team_recent']:<4}  "
          f"rank_ZA={r['rank_ZA_pooled']:>4}  rank_3A={r['rank_3A_pooled']:>4}  Δ={r['rank_change']:+d}  "
          f"ZA_pool={r['NFI_pct_ZA_pooled']:.4f}  3A_pool={r['NFI_pct_3A_pooled']:.4f}")

print("\n" + "="*110)
print("TOP 20 STRONGEST ASCENDING MOMENTUM — NFI%_3A_MOM in 2025-26")
print("="*110)
# Filter to players with valid momentum AND in qualifying pool AND current season TOI ≥ 500
mom_pool = career_sorted[career_sorted["NFI_pct_3A_MOM_2526"].notna() &
                          (career_sorted["toi_2526"].fillna(0) >= MIN_CURRENT_TOI)]
mom_top = mom_pool.sort_values("NFI_pct_3A_MOM_2526", ascending=False).head(20).reset_index(drop=True)
for i, r in mom_top.iterrows():
    print(f"{i+1:>3}  {r['player_name']:<26} {r['position']:<2} {r['team_recent']:<4}  "
          f"MOM={r['NFI_pct_3A_MOM_2526']:+.4f}  ZA_pool={r['NFI_pct_ZA_pooled']:.4f}  "
          f"3A_pool={r['NFI_pct_3A_pooled']:.4f}  toi_26={int(r['toi_2526']):>5}")

print(f"\n[done] top 200 CSV -> {OUT/'top200_article_dataset.csv'}")
