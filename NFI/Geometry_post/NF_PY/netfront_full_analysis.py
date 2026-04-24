#!/usr/bin/env python3
"""
HockeyROI - Net-Front Full Analysis (Expanded Definition)
Save to: NHL analysis/Goalies/netfront_full_analysis.py

Usage:
  cd "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies"
  python3 netfront_full_analysis.py

Net-front attempt = ES shot where is_rebound == True
                    OR shot_type in {"tip-in", "deflected", "bat"}

Outputs (all in Goalies/Benchmarks/):
  netfront_goalrate.csv       — league-wide breakdown by component
  netfront_shooters_full.csv  — per-shooter rankings with player names
"""

import math
import os
import time

import numpy as np
import pandas as pd
import requests

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BENCH_DIR       = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies/Benchmarks"
DATA_FILE       = os.path.join(BENCH_DIR, "all_goalie_shots_3seasons.csv")
BASE_URL        = "https://api-web.nhle.com/v1"
MIN_ATTEMPTS    = 100   # minimum ES attempts for shooter to qualify
MIN_NF_ATTEMPTS = 20    # minimum net-front attempts to report net-front goal rate
Z95             = 1.96

NF_SHOT_TYPES   = {"tip-in", "deflected", "bat"}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "HockeyROI-Analysis/2.0"


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def fetch(url, timeout=12):
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(0.5 * (attempt + 1))
    return None


def wilson_ci(k, n, z=Z95):
    """Wilson score CI for proportion k/n. Returns (p_hat, lo, hi)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def goal_rate(goals, shots):
    return goals / shots if shots > 0 else np.nan


# ─── LOAD & PREPARE ────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_FILE)

# Normalise boolean columns (stored as strings in CSV)
for col in ("is_rebound", "is_rush"):
    df[col] = df[col].astype(str).str.strip().map({"True": True, "False": False}).fillna(False)
df["is_goal"] = df["is_goal"].astype(int)

# Even Strength only
es = df[df["situation"] == "Even Strength"].copy()
print(f"  Total rows: {len(df):,}  |  Even Strength: {len(es):,}")

# Net-front flag
es["is_netfront"] = es["is_rebound"] | es["shot_type"].isin(NF_SHOT_TYPES)

# Component labels (mutually exclusive priority: rebound first, then shot type)
def component(row):
    if row["is_rebound"]:
        return "rebound"
    t = row["shot_type"]
    if t in NF_SHOT_TYPES:
        return t
    return "other"

es["component"] = es.apply(component, axis=1)

nf  = es[es["is_netfront"]]
non = es[~es["is_netfront"]]

print(f"  Net-front shots: {len(nf):,}  |  Other shots: {len(non):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — League-wide goal rate breakdown
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Analysis 1: League-wide net-front goal rate ──")

nf_goal_rate      = goal_rate(nf["is_goal"].sum(), len(nf))
baseline_goal_rate = goal_rate(non["is_goal"].sum(), len(non))
multiplier        = nf_goal_rate / baseline_goal_rate if baseline_goal_rate else np.nan

print(f"  Net-front goal rate : {nf_goal_rate:.4f}  ({len(nf):,} shots, {nf['is_goal'].sum()} goals)")
print(f"  Baseline goal rate  : {baseline_goal_rate:.4f}  ({len(non):,} shots, {non['is_goal'].sum()} goals)")
print(f"  Multiplier          : {multiplier:.2f}x")

# Component breakdown
component_rows = []

# Net-front aggregate
component_rows.append({
    "component"  : "NET-FRONT (total)",
    "shots"      : len(nf),
    "goals"      : int(nf["is_goal"].sum()),
    "goal_rate"  : round(nf_goal_rate, 4),
})

# Components
for comp in ["rebound", "tip-in", "deflected", "bat"]:
    sub   = es[es["component"] == comp]
    shots = len(sub)
    goals = int(sub["is_goal"].sum())
    gr    = goal_rate(goals, shots)
    component_rows.append({
        "component"  : comp,
        "shots"      : shots,
        "goals"      : goals,
        "goal_rate"  : round(gr, 4) if not np.isnan(gr) else np.nan,
    })
    print(f"  {comp:<12}: {shots:>6,} shots  {goals:>4} goals  {gr:.4f} goal rate")

# Baseline (non-net-front)
component_rows.append({
    "component"  : "other (baseline)",
    "shots"      : len(non),
    "goals"      : int(non["is_goal"].sum()),
    "goal_rate"  : round(baseline_goal_rate, 4),
})

# Add league multiplier vs baseline for each row
for r in component_rows:
    if not np.isnan(r["goal_rate"]) and baseline_goal_rate:
        r["multiplier_vs_baseline"] = round(r["goal_rate"] / baseline_goal_rate, 2)
    else:
        r["multiplier_vs_baseline"] = np.nan

goalrate_df = pd.DataFrame(component_rows)
out_gr = os.path.join(BENCH_DIR, "netfront_goalrate.csv")
goalrate_df.to_csv(out_gr, index=False)
print(f"\n  Saved: {out_gr}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Per-shooter net-front rankings
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Analysis 2: Per-shooter net-front rankings ──")

# League average net-front attempt rate (for CI flagging)
league_nf_rate = len(nf) / len(es)
print(f"  League avg net-front attempt rate: {league_nf_rate:.4f}")

shooter_rows = []

for sid, grp in es.groupby("shooting_player_id"):
    total = len(grp)
    if total < MIN_ATTEMPTS:
        continue

    nf_grp    = grp[grp["is_netfront"]]
    nf_att    = len(nf_grp)
    nf_goals  = int(nf_grp["is_goal"].sum())
    nf_rate   = nf_att / total

    _, nf_rate_lo, nf_rate_hi = wilson_ci(nf_att, total)

    nf_goal_rate_val = nf_goals / nf_att if nf_att >= MIN_NF_ATTEMPTS else np.nan

    shooter_rows.append({
        "shooting_player_id"   : int(sid),
        "player_name"          : "",           # filled after API lookup
        "total_es_attempts"    : total,
        "netfront_attempts"    : nf_att,
        "netfront_attempt_rate": round(nf_rate, 4),
        "nf_rate_ci_lo"        : round(nf_rate_lo, 4),
        "nf_rate_ci_hi"        : round(nf_rate_hi, 4),
        "netfront_goals"       : nf_goals,
        "netfront_goal_rate"   : round(nf_goal_rate_val, 4) if not np.isnan(nf_goal_rate_val) else np.nan,
        "confirmed_netfront"   : nf_rate_lo > league_nf_rate,
    })

shooter_df = pd.DataFrame(shooter_rows)
print(f"  Qualifying shooters (≥{MIN_ATTEMPTS} ES attempts): {len(shooter_df)}")
confirmed  = shooter_df["confirmed_netfront"].sum()
print(f"  Statistically confirmed net-front players: {confirmed}")


# ─── NHL API PLAYER NAME LOOKUP ────────────────────────────────────────────────
print(f"\n  Fetching player names from NHL API ({len(shooter_df)} players)...")

player_ids = shooter_df["shooting_player_id"].tolist()
name_map   = {}
failed     = 0

for i, pid in enumerate(player_ids):
    data = fetch(f"{BASE_URL}/player/{pid}/landing")
    if data:
        first = data.get("firstName", {}).get("default", "")
        last  = data.get("lastName",  {}).get("default", "")
        name_map[pid] = f"{first} {last}".strip()
    else:
        name_map[pid] = f"ID:{pid}"
        failed += 1

    # Progress every 50
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(player_ids)} fetched...")
    # Polite rate limit
    time.sleep(0.05)

print(f"  Done. {len(name_map) - failed} names resolved, {failed} not found (kept ID).")

shooter_df["player_name"] = shooter_df["shooting_player_id"].map(name_map)

# Sort by net-front attempt rate descending
shooter_df = shooter_df.sort_values("netfront_attempt_rate", ascending=False).reset_index(drop=True)

out_sh = os.path.join(BENCH_DIR, "netfront_shooters_full.csv")
shooter_df.to_csv(out_sh, index=False)
print(f"  Saved: {out_sh}")


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  HOCKEYROI — NET-FRONT ANALYSIS (EXPANDED DEFINITION)")
print("═" * 70)
print(f"\n  Even Strength shots: {len(es):,}  |  Seasons: 2023-24, 2024-25, 2025-26")
print(f"  Net-front definition: is_rebound=True OR shot_type in {{tip-in, deflected, bat}}")

print(f"\n  LEAGUE-WIDE GOAL RATES")
print(f"  {'Category':<22}  {'Shots':>8}  {'Goals':>6}  {'Goal Rate':>10}  {'vs Baseline':>12}")
print(f"  {'-'*64}")
for _, r in goalrate_df.iterrows():
    mult = f"{r['multiplier_vs_baseline']:.2f}x" if not np.isnan(r["multiplier_vs_baseline"]) else "  —"
    print(f"  {r['component']:<22}  {r['shots']:>8,}  {r['goals']:>6}  {r['goal_rate']:>10.4f}  {mult:>12}")

print(f"\n  Net-front goal rate is {multiplier:.2f}x the baseline")

print(f"\n  TOP 15 NET-FRONT PLAYERS BY ATTEMPT RATE")
print(f"  (confirmed = CI lower bound > league avg {league_nf_rate:.4f})")
print(f"  {'Rk':<4}  {'Player':<22}  {'Tot':>6}  {'NF Att':>7}  {'Rate':>7}  {'CI lo':>7}  {'CI hi':>7}  {'NF G':>6}  {'NF G%':>7}  {'Conf?':>6}")
print(f"  {'-'*90}")
for rank, row in shooter_df.head(15).iterrows():
    gr_str = f"{row['netfront_goal_rate']:.3f}" if not np.isnan(row["netfront_goal_rate"]) else "  n/a"
    conf   = "YES" if row["confirmed_netfront"] else "no"
    print(f"  {rank+1:<4}  {row['player_name']:<22}  {row['total_es_attempts']:>6,}"
          f"  {row['netfront_attempts']:>7}  {row['netfront_attempt_rate']:>7.4f}"
          f"  {row['nf_rate_ci_lo']:>7.4f}  {row['nf_rate_ci_hi']:>7.4f}"
          f"  {row['netfront_goals']:>6}  {gr_str:>7}  {conf:>6}")

print("\n" + "═" * 70)
print("  Output files written to Goalies/Benchmarks/")
print("  - netfront_goalrate.csv")
print("  - netfront_shooters_full.csv")
print("═" * 70 + "\n")
