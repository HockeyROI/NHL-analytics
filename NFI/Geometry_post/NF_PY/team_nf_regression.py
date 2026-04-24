#!/usr/bin/env python3
"""
HockeyROI — Team-level net-front metrics vs points_pct
Four metrics per team-season:
  1. NF attempt rate       = NF shots / total ES shots
  2. NF goal rate          = NF goals / NF shots
  3. Avg NF positioning    = mean x_coord_norm of NF shots (higher x → deeper crease)
  4. Avg rebound time gap  = mean time_gap_secs from rebound_sequences (regular season)

Univariate Pearson r + p-value for each.
Full multivariate OLS with coefficients, SEs, t-stats, p-values, R².
"""

import csv, math, os
from collections import defaultdict
import numpy as np
import scipy.stats as st

DATA_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"

NF_X_MIN = 74.0;  NF_Y_MIN = -8.0;  NF_Y_MAX = 8.0
REGULAR  = "regular"

SEP  = "═" * 100
SEP2 = "─" * 100

# ── PASS 1: nhl_shot_events.csv ──────────────────────────────────────────────
print("Reading nhl_shot_events.csv...")

t_es    = defaultdict(int)    # (season, team) total ES shots
t_nf    = defaultdict(int)    # (season, team) NF-zone shots
t_nfg   = defaultdict(int)    # (season, team) NF-zone goals
t_nf_x  = defaultdict(list)   # (season, team) list of x_coord_norm for NF shots

with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["game_type"] != REGULAR:
            continue
        season = row["season"]
        team   = row["shooting_team_abbrev"]
        key    = (season, team)

        try:
            x = float(row["x_coord_norm"])
            y = float(row["y_coord_norm"])
        except ValueError:
            continue

        is_goal = row["is_goal"] == "1"
        is_nf   = (x >= NF_X_MIN and NF_Y_MIN <= y <= NF_Y_MAX)

        t_es[key] += 1
        if is_nf:
            t_nf[key]  += 1
            t_nf_x[key].append(x)
            if is_goal:
                t_nfg[key] += 1

print(f"  {len(t_es)} team-season shooting records loaded.")

# ── PASS 2: rebound_sequences.csv ───────────────────────────────────────────
print("Reading rebound_sequences.csv (regular season only)...")

t_gap_vals = defaultdict(list)   # (season, team) list of time_gap_secs

with open(os.path.join(DATA_DIR, "rebound_sequences.csv"), newline="") as f:
    for row in csv.DictReader(f):
        gid = str(row["game_id"])
        # Regular season: game_id digit positions 4-5 == "02"
        if len(gid) < 6 or gid[4:6] != "02":
            continue
        season = row["season"]
        team   = row["orig_team"]
        try:
            gap = float(row["time_gap_secs"])
        except ValueError:
            continue
        t_gap_vals[(season, team)].append(gap)

print(f"  {sum(len(v) for v in t_gap_vals.values()):,} regular-season rebound sequences loaded.")

# ── LOAD STANDINGS ──────────────────────────────────────────────────────────
print("Loading standings...")
standings = {}
with open(os.path.join(DATA_DIR, "standings_5seasons.csv"), newline="") as f:
    for row in csv.DictReader(f):
        standings[(row["season"], row["team"])] = float(row["points_pct"])

# ── BUILD DATASET ────────────────────────────────────────────────────────────
print("Building dataset...")
rows = []
skipped = 0

for key in sorted(standings.keys()):
    season, team = key
    pts_pct = standings[key]

    es = t_es.get(key, 0)
    nf = t_nf.get(key, 0)
    if es < 100 or nf < 5:
        skipped += 1
        continue

    nf_rate  = nf / es
    nf_gr    = t_nfg.get(key, 0) / nf
    avg_x    = sum(t_nf_x[key]) / len(t_nf_x[key]) if t_nf_x[key] else 0.0

    gaps = t_gap_vals.get(key, [])
    if not gaps:
        skipped += 1
        continue
    avg_gap = sum(gaps) / len(gaps)

    rows.append({
        "season"  : season,
        "team"    : team,
        "pts_pct" : pts_pct,
        "nf_rate" : nf_rate,
        "nf_gr"   : nf_gr,
        "avg_x"   : avg_x,
        "avg_gap" : avg_gap,
    })

n = len(rows)
print(f"  Dataset: {n} team-seasons  ({skipped} skipped)\n")

# Print metric summary
print(f"  {'Metric':<28}  {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'Std':>8}")
print(f"  {'-'*66}")
for col, label in [
    ("nf_rate", "NF attempt rate"),
    ("nf_gr",   "NF goal rate"),
    ("avg_x",   "Avg NF x_coord_norm"),
    ("avg_gap", "Avg rebound time gap (s)"),
    ("pts_pct", "Points pct"),
]:
    vals = [r[col] for r in rows]
    mu   = sum(vals) / n
    sd   = math.sqrt(sum((v - mu)**2 for v in vals) / n)
    print(f"  {label:<28}  {mu:>8.4f}  {min(vals):>8.4f}  {max(vals):>8.4f}  {sd:>8.4f}")

# ── HELPERS ──────────────────────────────────────────────────────────────────
def pearson(x_list, y_list):
    n  = len(x_list)
    xa = np.array(x_list); ya = np.array(y_list)
    xm = xa - xa.mean();   ym = ya - ya.mean()
    r  = float(np.dot(xm, ym) / (math.sqrt(np.dot(xm, xm)) * math.sqrt(np.dot(ym, ym)) + 1e-15))
    t  = r * math.sqrt((n - 2) / max(1 - r**2, 1e-15))
    p  = 2.0 * (1.0 - st.t.cdf(abs(t), df=n - 2))
    return r, p

def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    if p < 0.10:  return ".  "
    return "   "

def ols(X, y):
    """Return beta, se, t, p, r2, adj_r2, rse given design matrix X and response y."""
    n, k   = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat   = X @ beta
    resid  = y - yhat
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    r2     = 1.0 - ss_res / ss_tot
    adj_r2 = 1.0 - (ss_res / (n - k)) / (ss_tot / (n - 1))
    sigma2 = ss_res / (n - k)
    cov_b  = sigma2 * np.linalg.inv(X.T @ X)
    se     = np.sqrt(np.diag(cov_b))
    t_stat = beta / se
    p_vals = [2.0 * (1.0 - st.t.cdf(abs(t), df=n - k)) for t in t_stat]
    return beta, se, t_stat, p_vals, r2, adj_r2, math.sqrt(sigma2)

# ── UNIVARIATE CORRELATIONS ──────────────────────────────────────────────────
y_vals = [r["pts_pct"] for r in rows]

metrics = [
    ("nf_rate", "NF attempt rate"),
    ("nf_gr",   "NF goal rate"),
    ("avg_x",   "Avg NF x_coord_norm"),
    ("avg_gap", "Avg rebound time gap"),
]

univ = {}
print(f"\n{SEP}")
print("  UNIVARIATE CORRELATIONS  (each metric vs points_pct, n={})".format(n))
print(SEP)
print(f"  {'Metric':<28}  {'Pearson r':>10}  {'r²':>7}  {'p-value':>10}  {'Sig':>4}  Direction")
print(f"  {SEP2}")
for col, label in metrics:
    x_vals = [r[col] for r in rows]
    r_val, p_val = pearson(x_vals, y_vals)
    direction = "↑ more NF → more wins" if r_val > 0 else "↓ less gap → more wins" if col == "avg_gap" else "↓"
    if col == "avg_gap":
        direction = "↓ faster rebound = better"
    elif col == "avg_x":
        direction = "↑ deeper position = better"
    elif col == "nf_rate":
        direction = "↑ more NF volume = better"
    elif col == "nf_gr":
        direction = "↑ higher NF conversion = better"
    print(f"  {label:<28}  {r_val:>10.4f}  {r_val**2:>7.4f}  {p_val:>10.6f}  {sig(p_val)}  {direction}")
    univ[col] = (r_val, p_val)

# ── MULTIVARIATE OLS ─────────────────────────────────────────────────────────
y  = np.array(y_vals)
Xm = np.column_stack([
        np.ones(n),
        [r["nf_rate"] for r in rows],
        [r["nf_gr"]   for r in rows],
        [r["avg_x"]   for r in rows],
        [r["avg_gap"] for r in rows],
     ])

beta, se, t_stat, p_vals, r2, adj_r2, rse = ols(Xm, y)
var_names = ["(Intercept)", "NF attempt rate", "NF goal rate", "Avg NF x_coord", "Avg time gap"]

print(f"\n{SEP}")
print(f"  MULTIVARIATE OLS  —  points_pct ~ NF_rate + NF_goal_rate + avg_x + avg_gap  (n={n})")
print(SEP)
print(f"  {'Variable':<28}  {'Coef':>10}  {'Std Err':>9}  {'t-stat':>8}  {'p-value':>10}  {'Sig':>4}")
print(f"  {'-'*78}")
for i, vn in enumerate(var_names):
    print(f"  {vn:<28}  {beta[i]:>10.4f}  {se[i]:>9.4f}  {t_stat[i]:>8.3f}  {p_vals[i]:>10.6f}  {sig(p_vals[i])}")

print(f"\n  R²:           {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"  Adj R²:       {adj_r2:.4f}")
print(f"  Residual SE:  {rse:.4f}  (df = {n - Xm.shape[1]})")

# ── PAIRWISE CORRELATIONS (VIF proxy) ────────────────────────────────────────
print(f"\n{SEP}")
print("  PAIRWISE CORRELATIONS BETWEEN PREDICTORS  (multicollinearity check)")
print(SEP)
pred_cols = ["nf_rate", "nf_gr", "avg_x", "avg_gap"]
pred_lbls = ["NF rate", "NF goal rate", "Avg x", "Avg gap"]
print(f"  {'':22}", end="")
for lb in pred_lbls:
    print(f"  {lb:>12}", end="")
print()
print(f"  {'-'*70}")
for i, (ci, li) in enumerate(zip(pred_cols, pred_lbls)):
    print(f"  {li:<22}", end="")
    for j, (cj, lj) in enumerate(zip(pred_cols, pred_lbls)):
        if i == j:
            print(f"  {'1.0000':>12}", end="")
        else:
            xi = [r[ci] for r in rows]
            xj = [r[cj] for r in rows]
            rv, _ = pearson(xi, xj)
            print(f"  {rv:>12.4f}", end="")
    print()

# ── INTERPRETATION ────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  INTERPRETATION")
print(SEP)

survivors = [(var_names[i+1], beta[i+1], p_vals[i+1]) for i in range(4) if p_vals[i+1] < 0.05]
marginal  = [(var_names[i+1], beta[i+1], p_vals[i+1]) for i in range(4) if 0.05 <= p_vals[i+1] < 0.10]
dropped   = [(var_names[i+1], beta[i+1], p_vals[i+1]) for i in range(4) if p_vals[i+1] >= 0.10]

print(f"\n  SIGNIFICANT in multivariate (p<0.05):")
if survivors:
    for nm, b, p in survivors:
        r_uni, p_uni = univ.get([c for c, l in metrics if l.replace(" ","_") in nm.replace(" ","_") or nm in l][0] if any(nm in l for c, l in metrics) else list(univ.keys())[0])
        print(f"    ✓  {nm:<28}  coef={b:+.4f}  p={p:.5f}  (univariate r={univ.get(next((c for c,l in metrics if nm in l or l in nm), 'nf_rate'),(0,0))[0]:.4f})")
else:
    print("    (none)")

print(f"\n  MARGINAL (0.05≤p<0.10):")
for nm, b, p in marginal:
    print(f"    ~  {nm:<28}  coef={b:+.4f}  p={p:.5f}")
if not marginal:
    print("    (none)")

print(f"\n  NOT significant (p≥0.10):")
for nm, b, p in dropped:
    print(f"    ✗  {nm:<28}  coef={b:+.4f}  p={p:.5f}")
if not dropped:
    print("    (none)")

# Clean one-line summary for each metric
print(f"\n  PER-METRIC SUMMARY:")
uni_labels = {"nf_rate": "NF attempt rate", "nf_gr": "NF goal rate",
              "avg_x": "Avg NF x_coord", "avg_gap": "Avg time gap"}
for col, label in metrics:
    rv, pv = univ[col]
    mv_p   = p_vals[metrics.index((col, label)) + 1]
    uni_sig = sig(pv).strip()
    mv_sig  = sig(mv_p).strip()
    verdict = "SURVIVES multivariate" if mv_p < 0.05 else ("MARGINAL" if mv_p < 0.10 else "drops out")
    print(f"    {label:<28}  univariate r={rv:+.4f} ({uni_sig:>3})  multivariate p={mv_p:.5f} ({mv_sig:>3})  → {verdict}")

print(f"\n  Model R²={r2:.4f} vs best univariate r²={max(rv**2 for rv,_ in univ.values()):.4f} "
      f"({max(univ, key=lambda k: univ[k][0]**2)})")
print(SEP + "\n")
