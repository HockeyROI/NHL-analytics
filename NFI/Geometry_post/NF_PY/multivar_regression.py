#!/usr/bin/env python3
"""
HockeyROI — Multivariate Regression: points_pct ~ NF_attempt_rate + save_pct + GA_per_game
Plus: full list of team-seasons with 2+ elite net-front players.

OLS implemented with numpy lstsq; SEs/p-values from scipy.stats.t.
"""

import csv, math, os
from collections import defaultdict
import numpy as np
import scipy.stats as st

DATA_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Net-front (doorstep) zone definition — same as elite ranking
NF_X_MIN = 74.0
NF_Y_MIN = -8.0
NF_Y_MAX =  8.0

REGULAR = "regular"

# ── PART 1: 2+ ELITE PLAYER TABLE ───────────────────────────────────────────
print("Loading team net-front roster analysis...")
roster_rows = []
with open(os.path.join(DATA_DIR, "team_netfront_roster_analysis.csv"), newline="") as f:
    for row in csv.DictReader(f):
        n = int(row["n_elite_nf"])
        if n >= 2:
            roster_rows.append({
                "season"       : row["season"],
                "team"         : row["team"],
                "n_elite"      : n,
                "players"      : row["elite_player_names"],
                "points_pct"   : float(row["points_pct"]),
                "points"       : int(row["points"]),
                "gp"           : int(row["gp"]),
                "playoff_depth": row["playoff_depth"],
            })

# Sort: n_elite DESC, then points_pct DESC
roster_rows.sort(key=lambda r: (-r["n_elite"], -r["points_pct"]))

SEP = "═" * 115
print(f"\n{SEP}")
print("  2+ ELITE NET-FRONT PLAYERS — ALL TEAM-SEASONS")
print(SEP)
print(f"  {'#':>2}  {'Season':<10} {'Team':<5} {'N':>2}  {'Pts%':>6}  {'Pts':>4}/{' GP':<4}  {'Result':<20}  Players")
print(f"  {'-'*111}")
for i, r in enumerate(roster_rows, 1):
    # Wrap long player strings
    players = r["players"]
    print(f"  {i:>2}.  {r['season']:<10} {r['team']:<5} {r['n_elite']:>2}  "
          f"{r['points_pct']*100:>5.1f}%  {r['points']:>4}/{r['gp']:<4}  "
          f"{r['playoff_depth']:<20}  {players}")
print(f"  Total: {len(roster_rows)} team-seasons with 2+ elite NF players\n")


# ── PART 2: COMPUTE PREDICTORS FROM SHOT DATA ───────────────────────────────
print("Building per-team-season stats from nhl_shot_events.csv...")

# Accumulators per (season, team)
es_shots       = defaultdict(int)   # total ES shots attempted BY team
nf_shots       = defaultdict(int)   # NF-zone shots attempted BY team
sog_against    = defaultdict(int)   # SOG faced by team (shot-on-goal + goal events vs team)
goals_against  = defaultdict(int)   # goals conceded by team
team_gp        = defaultdict(set)   # unique game_ids per (season, team)

SOG_TYPES = {"shot-on-goal", "goal"}

with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["game_type"] != REGULAR:
            continue
        season = row["season"]
        team   = row["shooting_team_abbrev"]
        gid    = row["game_id"]
        etype  = row["event_type"]

        try:
            x = float(row["x_coord_norm"])
            y = float(row["y_coord_norm"])
        except ValueError:
            x = y = 0.0

        is_goal = row["is_goal"] == "1"
        key = (season, team)

        # Offense stats
        es_shots[key] += 1
        if x >= NF_X_MIN and NF_Y_MIN <= y <= NF_Y_MAX:
            nf_shots[key] += 1
        team_gp[key].add(gid)

        # Defense stats: the OTHER team concedes this shot/goal
        # Identify opponent from home/away
        home_abbrev = row["home_team_abbrev"]
        away_abbrev = row["away_team_abbrev"]
        opponent = away_abbrev if team == home_abbrev else home_abbrev

        opp_key = (season, opponent)
        if etype in SOG_TYPES:
            sog_against[opp_key] += 1
        if is_goal:
            goals_against[opp_key] += 1

print("  Done processing shot events.")


# ── PART 3: LOAD STANDINGS ───────────────────────────────────────────────────
print("Loading standings...")
standings = {}  # (season, team) -> {points_pct, gp, points}
with open(os.path.join(DATA_DIR, "standings_5seasons.csv"), newline="") as f:
    for row in csv.DictReader(f):
        key = (row["season"], row["team"])
        standings[key] = {
            "points_pct": float(row["points_pct"]),
            "gp"        : int(row["gp"]),
            "points"    : int(row["points"]),
        }

print(f"  {len(standings)} team-seasons in standings.")


# ── PART 4: BUILD REGRESSION DATASET ────────────────────────────────────────
print("Building regression dataset...")

reg_data = []
missing = 0

for key in sorted(standings.keys()):
    season, team = key
    if key not in standings:
        continue

    stand = standings[key]
    gp    = stand["gp"]
    pts_pct = stand["points_pct"]

    # NF attempt rate
    total_es = es_shots.get(key, 0)
    total_nf = nf_shots.get(key, 0)
    if total_es < 100:   # skip teams with almost no data
        missing += 1
        continue
    nf_rate = total_nf / total_es

    # ES save percentage
    sog_ag = sog_against.get(key, 0)
    ga     = goals_against.get(key, 0)
    if sog_ag < 50:
        missing += 1
        continue
    save_pct = 1.0 - (ga / sog_ag)

    # Goals against per game
    ga_pg = ga / gp if gp > 0 else 0.0

    reg_data.append({
        "season"      : season,
        "team"        : team,
        "points_pct"  : pts_pct,
        "nf_rate"     : nf_rate,
        "save_pct"    : save_pct,
        "ga_pg"       : ga_pg,
        "total_es"    : total_es,
        "total_nf"    : total_nf,
        "sog_against" : sog_ag,
        "goals_against": ga,
        "gp"          : gp,
    })

print(f"  Regression dataset: {len(reg_data)} team-seasons  ({missing} skipped for low data)")


# ── PART 5: OLS REGRESSION ───────────────────────────────────────────────────
# y = points_pct
# X = [1, nf_rate, save_pct, ga_pg]

y  = np.array([r["points_pct"] for r in reg_data])
X  = np.column_stack([
        np.ones(len(reg_data)),
        [r["nf_rate"]  for r in reg_data],
        [r["save_pct"] for r in reg_data],
        [r["ga_pg"]    for r in reg_data],
     ])

n, k = X.shape   # n observations, k params (including intercept)

# OLS: beta = (X'X)^{-1} X'y
beta, residuals_sum, rank, sv = np.linalg.lstsq(X, y, rcond=None)

# Residuals
y_hat = X @ beta
resid = y - y_hat

# R-squared
ss_res = float(np.dot(resid, resid))
ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
r2     = 1.0 - ss_res / ss_tot
adj_r2 = 1.0 - (ss_res / (n - k)) / (ss_tot / (n - 1))

# Standard errors
sigma2 = ss_res / (n - k)
cov_b  = sigma2 * np.linalg.inv(X.T @ X)
se     = np.sqrt(np.diag(cov_b))

# t-stats and p-values (two-tailed)
t_stats = beta / se
p_vals  = [2.0 * (1.0 - st.t.cdf(abs(t), df=n - k)) for t in t_stats]

# ── UNIVARIATE BENCHMARKS ────────────────────────────────────────────────────
def pearson_r(x_arr, y_arr):
    xm = x_arr - x_arr.mean()
    ym = y_arr - y_arr.mean()
    r  = float(np.dot(xm, ym) / (math.sqrt(np.dot(xm,xm)) * math.sqrt(np.dot(ym,ym))))
    t  = r * math.sqrt((len(x_arr) - 2) / (1 - r**2 + 1e-15))
    p  = 2.0 * (1.0 - st.t.cdf(abs(t), df=len(x_arr) - 2))
    return r, p

r_nf,  p_nf  = pearson_r(X[:,1], y)
r_sv,  p_sv  = pearson_r(X[:,2], y)
r_ga,  p_ga  = pearson_r(X[:,3], y)

# ── PRINT REGRESSION TABLE ───────────────────────────────────────────────────
def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    if p < 0.10:  return ".  "
    return "   "

var_names = ["(Intercept)", "NF attempt rate", "ES save pct", "GA per game"]

print(f"\n{SEP}")
print("  MULTIVARIATE OLS REGRESSION")
print(f"  Dependent variable: points_pct   |   n = {n}  team-seasons")
print(f"  Predictors: NF attempt rate (ES doorstep shots/total ES shots)")
print(f"              ES save pct (1 - GA/SOG against, ES regular season)")
print(f"              GA per game (regular season)")
print(SEP)
print(f"  {'Variable':<22}  {'Coef':>10}  {'Std Err':>9}  {'t-stat':>8}  {'p-value':>9}  {'Sig':>4}")
print(f"  {'-'*72}")
for i, vn in enumerate(var_names):
    print(f"  {vn:<22}  {beta[i]:>10.4f}  {se[i]:>9.4f}  {t_stats[i]:>8.3f}  {p_vals[i]:>9.5f}  {sig(p_vals[i])}")

print(f"\n  R-squared:           {r2:.4f}")
print(f"  Adjusted R-squared:  {adj_r2:.4f}")
print(f"  Residual Std Error:  {math.sqrt(sigma2):.4f}  (df = {n-k})")

print(f"\n  UNIVARIATE CORRELATIONS (for context):")
print(f"  {'Variable':<22}  {'Pearson r':>10}  {'p-value':>9}  {'Sig':>4}")
print(f"  {'-'*50}")
print(f"  {'NF attempt rate':<22}  {r_nf:>10.4f}  {p_nf:>9.5f}  {sig(p_nf)}")
print(f"  {'ES save pct':<22}  {r_sv:>10.4f}  {p_sv:>9.5f}  {sig(p_sv)}")
print(f"  {'GA per game':<22}  {r_ga:>10.4f}  {p_ga:>9.5f}  {sig(p_ga)}")

print(f"\n  INTERPRETATION:")
nf_sig_text = "SIGNIFICANT" if p_vals[1] < 0.05 else ("MARGINAL (p<0.10)" if p_vals[1] < 0.10 else "NOT significant")
print(f"  NF attempt rate: {nf_sig_text} after controlling for goaltending and defense (p={p_vals[1]:.4f})")
print(f"  Save pct coef={beta[2]:.3f} (p={p_vals[2]:.5f}) — expected: higher save% → more wins")
print(f"  GA/G coef={beta[3]:.3f} (p={p_vals[3]:.5f}) — expected: negative (more GA → fewer wins)")
print(f"  The full model explains {r2*100:.1f}% of variance in points%; NF univariate alone = {r_nf**2*100:.1f}%")
print(SEP + "\n")

# ── DIAGNOSTIC: sample means ─────────────────────────────────────────────────
print(f"  PREDICTOR SUMMARY STATS (n={n}):")
for col, label in [("nf_rate","NF attempt rate"), ("save_pct","ES save pct"), ("ga_pg","GA per game"), ("points_pct","Points pct")]:
    vals = [r[col] for r in reg_data]
    print(f"    {label:<18}  mean={sum(vals)/len(vals):.4f}  "
          f"min={min(vals):.4f}  max={max(vals):.4f}  "
          f"std={math.sqrt(sum((v-sum(vals)/len(vals))**2 for v in vals)/len(vals)):.4f}")
print()
