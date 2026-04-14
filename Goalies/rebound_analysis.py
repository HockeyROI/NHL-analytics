#!/usr/bin/env python3
"""
HockeyROI - Rebound Vulnerability & Net-Front Shooter Analysis
Save to: NHL analysis/Goalies/rebound_analysis.py

Usage:
  cd "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies"
  python3 rebound_analysis.py

Outputs (all in Goalies/Benchmarks/):
  rebound_vulnerability_by_goalie.csv
  rebound_by_danger_zone.csv
  netfront_shooters.csv
"""

import pandas as pd
import numpy as np
import math
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BENCH_DIR  = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies/Benchmarks"
DATA_FILE  = os.path.join(BENCH_DIR, "all_goalie_shots_3seasons.csv")
MIN_GOALIE_SHOTS   = 200   # minimum ES shots faced (pooled across all seasons)
MIN_SHOOTER_SHOTS  = 50    # minimum ES shot attempts
MIN_REBOUND_SHOTS  = 10    # minimum rebound attempts to report rebound sv%
Z95 = 1.96                 # z for 95% CI

# ─── WILSON CI ─────────────────────────────────────────────────────────────────
def wilson_ci(k, n, z=Z95):
    """
    Wilson score interval for a proportion k/n.
    Returns (proportion, lower, upper). Returns (NaN, NaN, NaN) if n == 0.
    k = successes, n = trials.
    """
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (p_hat, max(0.0, center - margin), min(1.0, center + margin))


def sv_pct(goals, shots):
    """Save percentage: 1 - (goals / shots). Returns NaN if shots == 0."""
    if shots == 0:
        return np.nan
    return 1.0 - goals / shots


# ─── LOAD & FILTER ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_FILE)

# Normalise is_rebound/is_rush from string to bool
df['is_rebound'] = df['is_rebound'].astype(str).str.strip().map({'True': True, 'False': False}).fillna(False)
df['is_rush']    = df['is_rush'].astype(str).str.strip().map({'True': True, 'False': False}).fillna(False)
df['is_goal']    = df['is_goal'].astype(int)

# Filter to Even Strength only
es = df[df['situation'] == 'Even Strength'].copy()
print(f"  Total rows: {len(df):,}  |  Even Strength: {len(es):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1A — League-wide rebound vs non-rebound save%
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Analysis 1: Goalie rebound vulnerability ──")

non_reb = es[~es['is_rebound']]
reb     = es[es['is_rebound']]

league_non_reb_sv = sv_pct(non_reb['is_goal'].sum(), len(non_reb))
league_reb_sv     = sv_pct(reb['is_goal'].sum(), len(reb))
league_delta      = league_non_reb_sv - league_reb_sv

print(f"\n  League-wide (Even Strength, all goalies):")
print(f"    Non-rebound save%  : {league_non_reb_sv:.4f}  ({len(non_reb):,} shots)")
print(f"    Rebound save%      : {league_reb_sv:.4f}  ({len(reb):,} shots)")
print(f"    Delta (drop)       : {league_delta:+.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1B — Per-goalie rebound vulnerability (min 200 ES shots)
# ═══════════════════════════════════════════════════════════════════════════════
def goalie_rebound_row(grp):
    non_r = grp[~grp['is_rebound']]
    r     = grp[grp['is_rebound']]

    shots_total    = len(grp)
    shots_non_reb  = len(non_r)
    shots_reb      = len(r)

    goals_non_reb  = non_r['is_goal'].sum()
    goals_reb      = r['is_goal'].sum()

    sv_non_reb     = sv_pct(goals_non_reb, shots_non_reb)
    sv_reb         = sv_pct(goals_reb, shots_reb)
    delta          = sv_non_reb - sv_reb if not (np.isnan(sv_non_reb) or np.isnan(sv_reb)) else np.nan

    # Wilson CI — note: save% = 1 - goal%, so we compute CI on goal-allowed rate
    # and invert; k = goals (failures from goalie's perspective), n = shots
    _, gl_non_lo, gl_non_hi = wilson_ci(goals_non_reb, shots_non_reb)
    _, gl_reb_lo,  gl_reb_hi  = wilson_ci(goals_reb,  shots_reb)

    # Convert goal% CI to save% CI (invert bounds)
    sv_non_lo = 1 - gl_non_hi
    sv_non_hi = 1 - gl_non_lo
    sv_reb_lo = 1 - gl_reb_hi
    sv_reb_hi = 1 - gl_reb_lo

    # CIs don't overlap when rebound upper < non-rebound lower
    ci_no_overlap = (sv_reb_hi < sv_non_lo) if not any(
        np.isnan(x) for x in [sv_reb_hi, sv_non_lo]) else False

    return pd.Series({
        'goalie_name'       : grp['goalie_name'].iloc[0],
        'shots_total_es'    : shots_total,
        'shots_non_rebound' : shots_non_reb,
        'goals_non_rebound' : goals_non_reb,
        'sv_non_rebound'    : round(sv_non_reb, 4) if not np.isnan(sv_non_reb) else np.nan,
        'sv_non_reb_ci_lo'  : round(sv_non_lo, 4),
        'sv_non_reb_ci_hi'  : round(sv_non_hi, 4),
        'shots_rebound'     : shots_reb,
        'goals_rebound'     : goals_reb,
        'sv_rebound'        : round(sv_reb, 4) if not np.isnan(sv_reb) else np.nan,
        'sv_reb_ci_lo'      : round(sv_reb_lo, 4),
        'sv_reb_ci_hi'      : round(sv_reb_hi, 4),
        'sv_delta'          : round(delta, 4) if not np.isnan(delta) else np.nan,
        'ci_no_overlap'     : ci_no_overlap,
    })

# Pool all seasons, group by goalie_id
goalie_grps = es.groupby('goalie_id')

# Apply and filter minimum shots
goalie_rows = []
for gid, grp in goalie_grps:
    if len(grp) >= MIN_GOALIE_SHOTS:
        goalie_rows.append(goalie_rebound_row(grp))

goalie_df = pd.DataFrame(goalie_rows)
goalie_df = goalie_df.sort_values('sv_rebound', ascending=False).reset_index(drop=True)

out_goalie = os.path.join(BENCH_DIR, 'rebound_vulnerability_by_goalie.csv')
goalie_df.to_csv(out_goalie, index=False)
print(f"\n  Saved: {out_goalie}  ({len(goalie_df)} goalies)")

# Quick flagged count
flagged = goalie_df['ci_no_overlap'].sum()
print(f"  Goalies with statistically meaningful rebound drop (CI no overlap): {flagged}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1C — Rebound vulnerability by danger zone
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Analysis 1C: Rebound vulnerability by danger zone ──")

dz_rows = []
for dz in ['Low', 'Medium', 'High']:
    sub = es[es['danger_zone'] == dz]
    for is_reb_val, label in [(False, 'Non-rebound'), (True, 'Rebound')]:
        grp = sub[sub['is_rebound'] == is_reb_val]
        shots = len(grp)
        goals = grp['is_goal'].sum()
        sv    = sv_pct(goals, shots)
        _, gl_lo, gl_hi = wilson_ci(goals, shots)
        dz_rows.append({
            'danger_zone'   : dz,
            'shot_category' : label,
            'shots'         : shots,
            'goals'         : goals,
            'save_pct'      : round(sv, 4) if not np.isnan(sv) else np.nan,
            'sv_ci_lo'      : round(1 - gl_hi, 4),
            'sv_ci_hi'      : round(1 - gl_lo, 4),
        })

dz_df = pd.DataFrame(dz_rows)

# Also add delta rows per danger zone
dz_delta_rows = []
for dz in ['Low', 'Medium', 'High']:
    nr_row = dz_df[(dz_df['danger_zone'] == dz) & (dz_df['shot_category'] == 'Non-rebound')].iloc[0]
    r_row  = dz_df[(dz_df['danger_zone'] == dz) & (dz_df['shot_category'] == 'Rebound')].iloc[0]
    delta  = round(nr_row['save_pct'] - r_row['save_pct'], 4)
    dz_delta_rows.append({'danger_zone': dz, 'sv_non_rebound': nr_row['save_pct'],
                          'sv_rebound': r_row['save_pct'], 'delta': delta,
                          'shots_non_rebound': nr_row['shots'], 'shots_rebound': r_row['shots']})
    print(f"  {dz:6s}  Non-reb sv%: {nr_row['save_pct']:.4f} ({nr_row['shots']:>6,} shots)"
          f"  |  Rebound sv%: {r_row['save_pct']:.4f} ({r_row['shots']:>5,} shots)"
          f"  |  Delta: {delta:+.4f}")

out_dz = os.path.join(BENCH_DIR, 'rebound_by_danger_zone.csv')
dz_df.to_csv(out_dz, index=False)
print(f"\n  Saved: {out_dz}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Net-front shooters
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Analysis 2: Net-front shooters ──")

# League-wide rebound attempt rate (denominator: all ES shots)
league_reb_rate      = len(reb) / len(es)
_, league_reb_lo, league_reb_hi = wilson_ci(len(reb), len(es))
print(f"\n  League-wide ES rebound attempt rate: {league_reb_rate:.4f}"
      f"  95% CI [{league_reb_lo:.4f}, {league_reb_hi:.4f}]")

def shooter_row(grp):
    total_attempts  = len(grp)
    reb_grp         = grp[grp['is_rebound']]
    reb_attempts    = len(reb_grp)
    reb_goals       = reb_grp['is_goal'].sum()

    reb_rate        = reb_attempts / total_attempts if total_attempts > 0 else np.nan

    # Wilson CI on rebound attempt rate (k = rebound attempts, n = total shots)
    _, reb_rate_lo, reb_rate_hi = wilson_ci(reb_attempts, total_attempts)

    # Rebound shooting% only if >= MIN_REBOUND_SHOTS
    if reb_attempts >= MIN_REBOUND_SHOTS:
        reb_sh_pct = reb_goals / reb_attempts
    else:
        reb_sh_pct = np.nan

    # Confirmed net-front: CI lower bound exceeds league average
    confirmed_netfront = (reb_rate_lo > league_reb_rate) if not np.isnan(reb_rate_lo) else False

    # Grab shooter name from the most common goalie_name column — wait, that's
    # the goalie. We don't have a shooter name column; use shooting_player_id.
    # (The raw data has no shooter name column, only shooting_player_id.)
    return pd.Series({
        'shooting_player_id'   : grp['shooting_player_id'].iloc[0],
        'total_es_attempts'    : total_attempts,
        'rebound_attempts'     : reb_attempts,
        'rebound_attempt_rate' : round(reb_rate, 4) if not np.isnan(reb_rate) else np.nan,
        'reb_rate_ci_lo'       : round(reb_rate_lo, 4),
        'reb_rate_ci_hi'       : round(reb_rate_hi, 4),
        'rebound_goals'        : int(reb_goals),
        'rebound_sh_pct'       : round(reb_sh_pct, 4) if not np.isnan(reb_sh_pct) else np.nan,
        'confirmed_netfront'   : confirmed_netfront,
    })

shooter_grps = es.groupby('shooting_player_id')

shooter_rows = []
for sid, grp in shooter_grps:
    if len(grp) >= MIN_SHOOTER_SHOTS:
        shooter_rows.append(shooter_row(grp))

shooter_df = pd.DataFrame(shooter_rows)

# (a) Sort by rebound attempt rate
shooter_by_rate = shooter_df.sort_values('rebound_attempt_rate', ascending=False).reset_index(drop=True)
# (b) Sort by rebound goals
shooter_by_goals = shooter_df.sort_values('rebound_goals', ascending=False).reset_index(drop=True)

# Save: include both sort orders as separate columns
shooter_df['rank_by_rate']  = shooter_df['rebound_attempt_rate'].rank(ascending=False, method='min').astype(int)
shooter_df['rank_by_goals'] = shooter_df['rebound_goals'].rank(ascending=False, method='min').astype(int)
shooter_df = shooter_df.sort_values('rebound_attempt_rate', ascending=False).reset_index(drop=True)

out_shooter = os.path.join(BENCH_DIR, 'netfront_shooters.csv')
shooter_df.to_csv(out_shooter, index=False)

confirmed_count = shooter_df['confirmed_netfront'].sum()
print(f"  Qualifying shooters (≥{MIN_SHOOTER_SHOTS} ES attempts): {len(shooter_df)}")
print(f"  Statistically confirmed net-front players (CI lb > league avg): {confirmed_count}")
print(f"\n  Saved: {out_shooter}")


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("  HOCKEYROI — REBOUND ANALYSIS SUMMARY")
print("═" * 65)

print(f"\n  LEAGUE-WIDE (Even Strength, {len(es):,} shots, 3 seasons pooled)")
print(f"  {'Non-rebound save%':<26}: {league_non_reb_sv:.4f}  ({len(non_reb):,} shots)")
print(f"  {'Rebound save%':<26}: {league_reb_sv:.4f}  ({len(reb):,} shots)")
print(f"  {'Delta (goalie sv% drop)':<26}: {league_delta:+.4f}")

print(f"\n  REBOUND VULNERABILITY BY DANGER ZONE")
print(f"  {'Zone':<8}  {'Non-Reb sv%':>12}  {'Rebound sv%':>12}  {'Delta':>8}  {'Reb shots':>10}")
for row in dz_delta_rows:
    print(f"  {row['danger_zone']:<8}  {row['sv_non_rebound']:>12.4f}  {row['sv_rebound']:>12.4f}"
          f"  {row['delta']:>+8.4f}  {row['shots_rebound']:>10,}")

print(f"\n  TOP 10 NET-FRONT SHOOTERS BY REBOUND ATTEMPT RATE")
print(f"  (confirmed = CI lower bound > league avg {league_reb_rate:.4f})")
print(f"  {'Rank':<5}  {'Shooter ID':>12}  {'Reb Att':>8}  {'Rate':>7}  {'CI lo':>7}  {'CI hi':>7}  {'Reb G':>6}  {'Conf?':>6}")
top10 = shooter_by_rate.head(10)
for i, row in top10.iterrows():
    conf = "YES" if row['confirmed_netfront'] else "no"
    sh_pct = f"{row['rebound_sh_pct']:.3f}" if not np.isnan(row['rebound_sh_pct']) else "  n/a"
    print(f"  {i+1:<5}  {int(row['shooting_player_id']):>12}  {int(row['rebound_attempts']):>8}"
          f"  {row['rebound_attempt_rate']:>7.4f}  {row['reb_rate_ci_lo']:>7.4f}"
          f"  {row['reb_rate_ci_hi']:>7.4f}  {int(row['rebound_goals']):>6}  {conf:>6}")

print("\n" + "═" * 65)
print("  Output files written to Goalies/Benchmarks/")
print("  - rebound_vulnerability_by_goalie.csv")
print("  - rebound_by_danger_zone.csv")
print("  - netfront_shooters.csv")
print("═" * 65 + "\n")
