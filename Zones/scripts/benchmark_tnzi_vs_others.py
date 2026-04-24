"""Benchmark TNZI (raw / L / CL) against Corsi%, Fenwick%, HDCF%, xG%, PDO.

Read-only. Writes nothing. Prints comparison tables.

Note: MoneyPuck public CSV URLs return a data-license landing page instead of
data, so xG% is approximated using a league-average goal-rate-by-distance
model computed from Data/nhl_shot_events.csv. This is clearly labelled as
"xG% (approx)" throughout.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SHOTS_CSV = ROOT / "Data" / "nhl_shot_events.csv"
TNZI_CORR_CSV = HERE / "adjusted_rankings" / "tnzi_winning_correlation.csv"

SEASONS = ["20222023", "20232024", "20242025", "20252026"]
SEASON_LABEL = {"20222023": "22/23", "20232024": "23/24",
                "20242025": "24/25", "20252026": "25/26"}
SEASON_END_DATE = {
    "20222023": "2023-04-13",
    "20232024": "2024-04-18",
    "20242025": "2025-04-17",
    "20252026": "2026-04-17",
}

ABBR_MAP = {"ARI": "UTA"}
def norm_team(a): return ABBR_MAP.get(a, a)

# -----------------------------------------------------------------------------
# Correlation
# -----------------------------------------------------------------------------
def pearson(x, y):
    n = len(x)
    if n < 2: return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")

# -----------------------------------------------------------------------------
# Standings
# -----------------------------------------------------------------------------
def fetch_standings(date_str):
    d0 = datetime.strptime(date_str, "%Y-%m-%d")
    for off in range(5):
        d_try = (d0 - timedelta(days=off)).strftime("%Y-%m-%d")
        out = subprocess.run(
            ["curl", "-s", "-m", "20",
             f"https://api-web.nhle.com/v1/standings/{d_try}"],
            capture_output=True, text=True, check=True,
        )
        try:
            d = json.loads(out.stdout)
        except json.JSONDecodeError:
            continue
        rows = d.get("standings", [])
        if rows:
            return {norm_team(r["teamAbbrev"]["default"]): r["points"] for r in rows}
    return {}

# -----------------------------------------------------------------------------
# Compute team-season metrics from shot events
# -----------------------------------------------------------------------------
def load_shots():
    cols = ["season", "game_type", "situation_code", "event_type",
            "home_team_abbrev", "away_team_abbrev", "shooting_team_abbrev",
            "x_coord_norm", "y_coord_norm"]
    df = pd.read_csv(SHOTS_CSV, usecols=cols)
    df = df[(df["season"].astype(str).isin(SEASONS)) &
            (df["game_type"] == "regular") &
            (df["situation_code"] == 1551)].copy()
    df["season"] = df["season"].astype(str)
    df["shooting_team"] = df["shooting_team_abbrev"].map(norm_team)
    df["home_team"] = df["home_team_abbrev"].map(norm_team)
    df["away_team"] = df["away_team_abbrev"].map(norm_team)
    # Flip BS coords to shooter-perspective (BS coords are blocker-perspective)
    is_bs = df["event_type"] == "blocked-shot"
    df.loc[is_bs, "x_coord_norm"] = -df.loc[is_bs, "x_coord_norm"]
    df.loc[is_bs, "y_coord_norm"] = -df.loc[is_bs, "y_coord_norm"]
    # Derived flags
    df["is_corsi"] = df["event_type"].isin(
        ["shot-on-goal", "missed-shot", "blocked-shot", "goal"])
    df["is_fenwick"] = df["event_type"].isin(
        ["shot-on-goal", "missed-shot", "goal"])
    df["is_sog"] = df["event_type"].isin(["shot-on-goal", "goal"])
    df["is_goal"] = df["event_type"] == "goal"
    df["is_hd"] = (df["x_coord_norm"] > 54) & (df["y_coord_norm"].abs() <= 22)
    df["dist_to_goal"] = ((89 - df["x_coord_norm"]) ** 2 +
                          df["y_coord_norm"] ** 2) ** 0.5
    return df

def build_xg_model(df):
    """League avg goal rate per distance bucket, using unblocked shots
    (SOG+MS+goal) only. Returns dict bucket_edge -> goal_rate."""
    unblocked = df[df["is_fenwick"]]
    # bucket by 5 feet up to 60+
    edges = list(range(0, 65, 5)) + [200]
    # pandas cut returns right-open
    unblocked = unblocked.copy()
    unblocked["dist_bucket"] = pd.cut(unblocked["dist_to_goal"], bins=edges,
                                      right=False, include_lowest=True)
    rate = unblocked.groupby("dist_bucket", observed=True)["is_goal"].mean()
    return rate.to_dict()

def apply_xg(df, xg_map):
    edges = list(range(0, 65, 5)) + [200]
    df = df.copy()
    df["dist_bucket"] = pd.cut(df["dist_to_goal"], bins=edges,
                               right=False, include_lowest=True)
    df["xg"] = df["dist_bucket"].map(lambda b: xg_map.get(b, 0.0)).astype(float)
    return df

def team_season_metrics(df):
    """Return dict (team, season) -> metrics dict."""
    # CF / CA
    cf = df[df["is_corsi"]].groupby(["shooting_team", "season"]).size()
    # ca: events against team X are those where shooting_team != X but team X is in the game.
    # We need to identify the defending team per event. For each row, defending_team = home if shooting_team==away else home.
    ev = df[df["is_corsi"]].copy()
    ev["defending_team"] = ev.apply(
        lambda r: r["home_team"] if r["shooting_team"] == r["away_team"] else r["away_team"],
        axis=1)
    ca = ev.groupby(["defending_team", "season"]).size()

    ff_for = df[df["is_fenwick"]].groupby(["shooting_team", "season"]).size()
    ev_ff = df[df["is_fenwick"]].copy()
    ev_ff["defending_team"] = ev_ff.apply(
        lambda r: r["home_team"] if r["shooting_team"] == r["away_team"] else r["away_team"],
        axis=1)
    ff_against = ev_ff.groupby(["defending_team", "season"]).size()

    hdcf_for = df[df["is_corsi"] & df["is_hd"]].groupby(["shooting_team", "season"]).size()
    ev_hd = df[df["is_corsi"] & df["is_hd"]].copy()
    ev_hd["defending_team"] = ev_hd.apply(
        lambda r: r["home_team"] if r["shooting_team"] == r["away_team"] else r["away_team"],
        axis=1)
    hdcf_against = ev_hd.groupby(["defending_team", "season"]).size()

    # xG
    xg_for = df[df["is_fenwick"]].groupby(["shooting_team", "season"])["xg"].sum()
    ev_xg = df[df["is_fenwick"]].copy()
    ev_xg["defending_team"] = ev_xg.apply(
        lambda r: r["home_team"] if r["shooting_team"] == r["away_team"] else r["away_team"],
        axis=1)
    xg_against = ev_xg.groupby(["defending_team", "season"])["xg"].sum()

    # PDO: shooting% + save%
    sog_for = df[df["is_sog"]].groupby(["shooting_team", "season"]).size()
    g_for = df[df["is_goal"]].groupby(["shooting_team", "season"]).size()
    ev_s = df[df["is_sog"]].copy()
    ev_s["defending_team"] = ev_s.apply(
        lambda r: r["home_team"] if r["shooting_team"] == r["away_team"] else r["away_team"],
        axis=1)
    sog_against = ev_s.groupby(["defending_team", "season"]).size()
    ev_g = df[df["is_goal"]].copy()
    ev_g["defending_team"] = ev_g.apply(
        lambda r: r["home_team"] if r["shooting_team"] == r["away_team"] else r["away_team"],
        axis=1)
    g_against = ev_g.groupby(["defending_team", "season"]).size()

    teams = set(df["shooting_team"].dropna().unique()) | set(ev["defending_team"].dropna().unique())
    out = {}
    for season in SEASONS:
        for tm in teams:
            k = (tm, season)
            def g(s, tm=tm, season=season, s2=0):
                try: return int(s.loc[(tm, season)])
                except (KeyError, ValueError): return s2
            def gf(s, tm=tm, season=season):
                try: return float(s.loc[(tm, season)])
                except (KeyError, ValueError): return 0.0
            CF = g(cf); CA = g(ca)
            FFf = g(ff_for); FFa = g(ff_against)
            HDf = g(hdcf_for); HDa = g(hdcf_against)
            XGf = gf(xg_for); XGa = gf(xg_against)
            SOGf = g(sog_for); SOGa = g(sog_against)
            Gf = g(g_for); Ga = g(g_against)
            if CF + CA == 0:
                continue
            out[k] = {
                "CF_pct":  CF / (CF + CA) if (CF + CA) else float("nan"),
                "FF_pct":  FFf / (FFf + FFa) if (FFf + FFa) else float("nan"),
                "HDCF_pct": HDf / (HDf + HDa) if (HDf + HDa) else float("nan"),
                "xG_pct":  XGf / (XGf + XGa) if (XGf + XGa) else float("nan"),
                "PDO": (Gf / SOGf if SOGf else 0.0) +
                       (1 - Ga / SOGa if SOGa else 0.0),
                "CF": CF, "CA": CA, "SOGf": SOGf, "SOGa": SOGa,
                "Gf": Gf, "Ga": Ga,
            }
    return out

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("[1/5] fetching standings per season ...")
    standings = {s: fetch_standings(SEASON_END_DATE[s]) for s in SEASONS}
    for s in SEASONS:
        print(f"    {s}: {len(standings[s])} teams")

    print("[2/5] loading shot events (this may take 15-30s) ...")
    shots = load_shots()
    print(f"    rows: {len(shots):,}  seasons: {sorted(shots['season'].unique())}")

    print("[3/5] building league xG (approx, distance-based) ...")
    xg_map = build_xg_model(shots)
    shots = apply_xg(shots, xg_map)
    print(f"    distance buckets -> goal rates:")
    for b, r in list(xg_map.items())[:14]:
        print(f"      {b!s:<20} {r:.4f}")

    print("[4/5] computing team-season metrics ...")
    metrics = team_season_metrics(shots)
    n_ts = len(metrics)
    print(f"    team-seasons computed: {n_ts}")

    print("[5/5] correlating per season and pooled ...")
    # Per-season pearson
    metric_keys = ["CF_pct", "FF_pct", "HDCF_pct", "xG_pct", "PDO"]
    pretty = {"CF_pct": "Corsi%", "FF_pct": "Fenwick%",
              "HDCF_pct": "HDCF%", "xG_pct": "xG% (approx)", "PDO": "PDO"}

    # stacked pooled data (N=team-seasons)
    stacked_m = {mk: [] for mk in metric_keys}
    stacked_pts = []

    per_season_r = {mk: {} for mk in metric_keys}
    for season in SEASONS:
        pts_map = standings[season]
        for mk in metric_keys:
            xs, ys = [], []
            for (tm, s), m in metrics.items():
                if s != season: continue
                if tm not in pts_map: continue
                if isinstance(m[mk], float) and math.isnan(m[mk]): continue
                xs.append(m[mk]); ys.append(pts_map[tm])
            r = pearson(xs, ys) if len(xs) >= 5 else float("nan")
            per_season_r[mk][season] = r

        # Stack pooled
        for (tm, s), m in metrics.items():
            if s != season: continue
            if tm not in pts_map: continue
            stacked_pts.append(pts_map[tm])
            for mk in metric_keys:
                v = m[mk]
                if isinstance(v, float) and math.isnan(v): v = 0.0
                stacked_m[mk].append(v)

    pooled_r = {}
    for mk in metric_keys:
        pooled_r[mk] = pearson(stacked_m[mk], stacked_pts)
    N_stacked = len(stacked_pts)

    # --- TNZI — read precomputed correlations from the CSV ---
    tnzi_per_season = {"raw": {}, "L": {}, "CL": {}}
    tnzi_pooled = {}
    with open(TNZI_CORR_CSV) as f:
        for row in csv.DictReader(f):
            scen = row["scenario"]; variant = row["variant"]
            try:
                r = float(row["pearson_r"])
            except ValueError:
                r = float("nan")
            if scen == "pooled":
                tnzi_pooled[variant] = r
            elif scen in SEASONS and variant in tnzi_per_season:
                tnzi_per_season[variant][scen] = r

    # Assemble master table
    print("\n" + "=" * 104)
    print("MASTER COMPARISON — Pearson r vs team points  (n=32 per season; "
          f"pooled TNZI at n=32, benchmarks pooled at N={N_stacked} team-seasons)")
    print("=" * 104)
    hdr = (f"{'Metric':<18} {'22/23 r':>8} {'23/24 r':>8} {'24/25 r':>8} {'25/26 r':>8} "
           f"{'Pooled r':>9} {'Pooled R²':>10}")
    print(hdr)
    print("-" * 104)

    rows_out = []
    def add(label, r_per, r_pool, pool_n_note=""):
        def fmt(r):
            return f"{r:+.3f}" if isinstance(r, float) and not math.isnan(r) else "   -  "
        def fmt_pool(r):
            return f"{r:+.3f}" if isinstance(r, float) and not math.isnan(r) else "   -  "
        r2 = (r_pool ** 2) if isinstance(r_pool, float) and not math.isnan(r_pool) else float("nan")
        r2s = f"{r2:.3f}" if isinstance(r2, float) and not math.isnan(r2) else "  -  "
        r_vals = [r_per.get(s, float("nan")) for s in SEASONS]
        print(f"{label:<18} {fmt(r_vals[0]):>8} {fmt(r_vals[1]):>8} "
              f"{fmt(r_vals[2]):>8} {fmt(r_vals[3]):>8} "
              f"{fmt_pool(r_pool):>9} {r2s:>10}  {pool_n_note}")
        rows_out.append({"metric": label, "per_season": r_vals,
                         "pooled_r": r_pool, "pooled_r2": r2})

    # TNZI first
    add("TNZI raw (n=32 pool)", tnzi_per_season["raw"], tnzi_pooled.get("raw", float("nan")),
        pool_n_note="[pool n=32]")
    add("TNZI_L (n=32 pool)",  tnzi_per_season["L"],   tnzi_pooled.get("L",  float("nan")),
        pool_n_note="[pool n=32]")
    add("TNZI_CL (n=32 pool)", tnzi_per_season["CL"],  tnzi_pooled.get("CL", float("nan")),
        pool_n_note="[pool n=32]")

    # Benchmarks
    for mk in metric_keys:
        add(pretty[mk], per_season_r[mk], pooled_r[mk],
            pool_n_note=f"[pool N={N_stacked}]")

    # Consistency report
    print("\n" + "=" * 80)
    print("CONSISTENCY — std-dev of Pearson r across the 4 seasons (smaller = more stable)")
    print("=" * 80)
    consistency = []
    for row in rows_out:
        vals = [v for v in row["per_season"] if isinstance(v, float) and not math.isnan(v)]
        sd = pstdev(vals) if len(vals) >= 2 else float("nan")
        consistency.append((row["metric"], sd, vals))
    consistency.sort(key=lambda x: (float("inf") if math.isnan(x[1]) else x[1]))
    for name, sd, vals in consistency:
        best = max(vals) if vals else float("nan")
        worst = min(vals) if vals else float("nan")
        print(f"  {name:<22} stdev={sd:.4f}  best-season r={best:+.3f}  "
              f"worst-season r={worst:+.3f}")

    # Single-season leaders
    print("\n" + "=" * 80)
    print("HIGHEST SINGLE-SEASON PEARSON r")
    print("=" * 80)
    best_single = []
    for row in rows_out:
        for i, s in enumerate(SEASONS):
            v = row["per_season"][i]
            if isinstance(v, float) and not math.isnan(v):
                best_single.append((v, row["metric"], SEASON_LABEL[s]))
    best_single.sort(reverse=True)
    for v, name, s in best_single[:8]:
        print(f"  {name:<22} {s}  r = {v:+.3f}")

    # Direct TNZI_L vs benchmarks
    print("\n" + "=" * 80)
    print("TNZI_L vs KEY BENCHMARKS (pooled)")
    print("=" * 80)
    l_pool = tnzi_pooled.get("L", float("nan"))
    xg_pool = pooled_r.get("xG_pct", float("nan"))
    hd_pool = pooled_r.get("HDCF_pct", float("nan"))
    corsi_pool = pooled_r.get("CF_pct", float("nan"))
    def cmp(name, bench):
        if math.isnan(l_pool) or math.isnan(bench):
            print(f"  vs {name}: n/a"); return
        diff = l_pool - bench
        winner = "TNZI_L" if diff > 0 else name
        print(f"  TNZI_L (r={l_pool:+.3f})  vs  {name} (r={bench:+.3f})   "
              f"Δ = {diff:+.3f}   winner: {winner}")
    cmp("xG% (approx)", xg_pool)
    cmp("High Danger Corsi%", hd_pool)
    cmp("Corsi%", corsi_pool)

    print("\nNOTES")
    print("  - TNZI pooled correlation uses n=32 teams (4-season-averaged player data "
          "→ team avg).")
    print("  - Benchmark pooled correlations use the stacked team-season frame "
          f"(N={N_stacked}) — Pearson captures both within-season and across-season variance.")
    print("  - xG% here is a DISTANCE-BASED APPROXIMATION (MoneyPuck's public CSV URL "
          "redirects to a data-license landing page and cannot be scraped).")

if __name__ == "__main__":
    main()
