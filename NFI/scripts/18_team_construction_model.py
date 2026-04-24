#!/usr/bin/env python3
"""
Team Construction Model — full 6-step pipeline.

Inputs (NFI/output):
  - twoway_forward_score.csv  (5-season pooled, ≥500 ES TOI)
  - twoway_D_score.csv         (5-season pooled, ≥500 ES TOI)
  - goalie_metric_comparison.csv  (NFI_GSAx_per60 per goalie)
  - team_level_all_metrics.csv (per team-season; TNFI%, points, etc.)
Plus shifts to compute per-(team, season, player) TOI.

Steps 1-6 per spec.

Outputs (NFI/output):
  - team_construction_model.csv   (one row per team-season w/ all features)
  - team_archetypes.csv            (archetype assignment + summary)
  - goalie_impact_by_archetype.csv
  - cup_contention_profile.csv     (logistic-regression coefficients per outcome)
"""
import os, math, json
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT = f"{ROOT}/NFI/output"
SHIFT_CSV = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
GAME_CSV = f"{ROOT}/Data/game_ids.csv"
POS_CSV = f"{OUT}/player_positions.csv"

SEASONS = ["20212022","20222023","20232024","20242025","20252026"]
COMPLETED_PLAYOFF_SEASONS = ["20212022","20222023","20232024","20242025"]  # 2025-26 mid-playoffs

# ---------- Hardcoded playoff results (regular-season teams that
# advanced to each round) ----------
# Source: NHL.com / standard reference for completed seasons.
PLAYOFFS = {
    "20212022": {
        "playoffs": ["FLA","COL","CAR","TOR","CGY","NYR","TBL","MIN",
                     "STL","DAL","EDM","NSH","BOS","PIT","WSH","LAK"],
        "round1_winner": ["FLA","COL","CAR","NYR","CGY","TBL","STL","EDM"],
        "round2_winner": ["COL","NYR","TBL","EDM"],
        "conf_final_winner": ["COL","TBL"],
        "cup_winner": ["COL"],
    },
    "20222023": {
        "playoffs": ["BOS","CAR","NJD","TOR","NYR","NYI","TBL","FLA",
                     "VGK","EDM","DAL","COL","MIN","WPG","SEA","LAK"],
        "round1_winner": ["FLA","CAR","NJD","TOR","VGK","DAL","SEA","EDM"],
        "round2_winner": ["FLA","CAR","VGK","DAL"],
        "conf_final_winner": ["FLA","VGK"],
        "cup_winner": ["VGK"],
    },
    "20232024": {
        "playoffs": ["NYR","CAR","FLA","BOS","TOR","TBL","NYI","WSH",
                     "DAL","WPG","COL","VAN","EDM","NSH","VGK","LAK"],
        "round1_winner": ["NYR","CAR","FLA","BOS","DAL","COL","VAN","EDM"],
        "round2_winner": ["NYR","FLA","DAL","EDM"],
        "conf_final_winner": ["FLA","EDM"],
        "cup_winner": ["FLA"],
    },
    "20242025": {
        "playoffs": ["WSH","CAR","TOR","FLA","NJD","TBL","OTT","MTL",
                     "WPG","DAL","VGK","LAK","COL","EDM","MIN","STL"],
        "round1_winner": ["WSH","CAR","TOR","FLA","WPG","DAL","VGK","EDM"],
        "round2_winner": ["CAR","FLA","DAL","EDM"],
        "conf_final_winner": ["FLA","EDM"],
        "cup_winner": ["FLA"],
    },
}

# Conference assignments (current NHL alignment)
CONFERENCE = {
    # Eastern
    **{t:"E" for t in ["BOS","BUF","DET","FLA","MTL","OTT","TBL","TOR",
                       "CAR","CBJ","NJD","NYI","NYR","PHI","PIT","WSH"]},
    # Western
    **{t:"W" for t in ["CHI","COL","DAL","MIN","NSH","STL","UTA","WPG",
                       "ANA","CGY","EDM","LAK","SJS","SEA","VAN","VGK",
                       "ARI"]},  # ARI for pre-relocation seasons
}

# ---------------------------------------------------------------
# Step 0 — Load player-level pillar outputs
# ---------------------------------------------------------------
print("Loading player-level outputs...")
pos = pd.read_csv(POS_CSV, dtype={"player_id":int})
pos_grp = dict(zip(pos["player_id"], pos["pos_group"]))
pos_name = dict(zip(pos["player_id"], pos["player_name"]))

twf = pd.read_csv(f"{OUT}/twoway_forward_score.csv", dtype={"player_id":int})
twd = pd.read_csv(f"{OUT}/twoway_D_score.csv", dtype={"player_id":int})
gk  = pd.read_csv(f"{OUT}/goalie_metric_comparison.csv", dtype={"goalie_id":int})

f_score = dict(zip(twf["player_id"], twf["twoway_score"]))
d_score = dict(zip(twd["player_id"], twd["twoway_D_score"]))
g_score = dict(zip(gk["goalie_id"],  gk["NFI_GSAx_per60"]))
g_name  = dict(zip(gk["goalie_id"],  gk["goalie_name"]))
print(f"  F w/ two-way: {len(f_score)}, D w/ two-way: {len(d_score)}, G w/ GSAx/60: {len(g_score)}")

team_metrics = pd.read_csv(f"{OUT}/team_level_all_metrics.csv", dtype={"season":str})
team_metrics["season"] = team_metrics["season"].astype(str)

# ---------------------------------------------------------------
# Step 1a — Per-team-season player TOI from shifts
# ---------------------------------------------------------------
print("Computing per-team-season player TOI from shifts...")
games = pd.read_csv(GAME_CSV, dtype={"game_id":int,"season":str})
games = games[games["game_type"]=="regular"]
game_season = dict(zip(games["game_id"], games["season"]))

# Stream shifts
ts_toi = defaultdict(float)  # (player_id, season, team) -> sec
shift_cols = ["game_id","player_id","period","team_abbrev",
              "abs_start_secs","abs_end_secs"]
for ch in pd.read_csv(SHIFT_CSV, usecols=shift_cols, chunksize=500_000):
    ch = ch.dropna(subset=shift_cols)
    ch["game_id"] = ch["game_id"].astype(int)
    ch["player_id"] = ch["player_id"].astype(int)
    ch["period"] = ch["period"].astype(int)
    ch["abs_start_secs"] = ch["abs_start_secs"].astype(int)
    ch["abs_end_secs"] = ch["abs_end_secs"].astype(int)
    ch = ch[ch["period"].between(1,3)]
    ch["dur"] = (ch["abs_end_secs"] - ch["abs_start_secs"]).clip(lower=0)
    ch["season"] = ch["game_id"].map(game_season)
    ch = ch[ch["season"].isin(SEASONS)]
    grp = ch.groupby(["player_id","season","team_abbrev"], as_index=False)["dur"].sum()
    for r in grp.itertuples(index=False):
        ts_toi[(int(r.player_id), r.season, r.team_abbrev)] += float(r.dur)

print(f"  per (player, season, team) TOI rows: {len(ts_toi):,}")

# Convert to DataFrame
toi_rows = [{"player_id":pid, "season":s, "team":t, "toi_min": v/60.0}
            for (pid, s, t), v in ts_toi.items()]
toi_df = pd.DataFrame(toi_rows)
toi_df["pos_group"] = toi_df["player_id"].map(pos_grp)

# ---------------------------------------------------------------
# Step 1b — Per team-season profile
# ---------------------------------------------------------------
print("Building team-season profiles...")
MIN_TOI_F = 200.0   # forwards: ≥200 min for that team-season
MIN_TOI_D = 200.0
MIN_TOI_G = 200.0   # goalies often have wider gaps but use same threshold

profile_rows = []
for (season, team), g in toi_df.groupby(["season","team"]):
    # Forwards
    fs = g[(g["pos_group"]=="F") & (g["toi_min"]>=MIN_TOI_F)].copy()
    fs["score"] = fs["player_id"].map(f_score)
    fs_scored = fs.dropna(subset=["score"]).sort_values("toi_min", ascending=False)
    if len(fs_scored) >= 6:
        top6_f = fs_scored.head(6)
        bot6_f = fs_scored.iloc[6:].nsmallest(6, "toi_min") if len(fs_scored) >= 12 else fs_scored.iloc[6:]
    else:
        top6_f = fs_scored
        bot6_f = fs_scored.iloc[6:6]
    # Defensemen
    ds = g[(g["pos_group"]=="D") & (g["toi_min"]>=MIN_TOI_D)].copy()
    ds["score"] = ds["player_id"].map(d_score)
    ds_scored = ds.dropna(subset=["score"]).sort_values("toi_min", ascending=False)
    top4_d = ds_scored.head(4)
    # Goalies
    gs = g[(g["pos_group"]=="G") & (g["toi_min"]>=MIN_TOI_G)].copy()
    gs = gs.sort_values("toi_min", ascending=False)
    starter_id = int(gs.iloc[0]["player_id"]) if len(gs) else None
    starter_gsax = g_score.get(starter_id, np.nan) if starter_id else np.nan
    starter_name = g_name.get(starter_id, "") if starter_id else ""
    starter_toi  = float(gs.iloc[0]["toi_min"]) if len(gs) else np.nan

    profile_rows.append({
        "season": season,
        "team": team,
        "F_top6_n": len(top6_f),
        "F_bot6_n": len(bot6_f),
        "D_top4_n": len(top4_d),
        "F_top6_twoway": top6_f["score"].mean() if len(top6_f) else np.nan,
        "F_bot6_twoway": bot6_f["score"].mean() if len(bot6_f) else np.nan,
        "D_top4_twoway": top4_d["score"].mean() if len(top4_d) else np.nan,
        "starter_goalie_id": starter_id,
        "starter_goalie_name": starter_name,
        "starter_goalie_toi_min": starter_toi,
        "starter_NFI_GSAx_per60": starter_gsax,
    })

prof = pd.DataFrame(profile_rows)
# Merge in team-season points / standings / TNFI%
keep = ["season","team","gp","points","points_pct","wins","goal_diff",
        "TNFI_pct","TNFI_score_adj_pct","CF_pct","HD_CF_pct","NFI_composite"]
prof = prof.merge(team_metrics[keep], on=["season","team"], how="left")
prof["gf_per_game"] = (team_metrics["wins"]*0)  # placeholder
# Compute GF/GA per game from team_metrics? team_metrics doesn't have explicit GF/GA.
# We have goal_diff and gp; but not GF/GA separately. Use approximations.
# Pull from raw season totals: GF = (goal_diff + GA)/?; cannot derive both without one.
# Use proxy: NFI_composite or just goal_diff/gp.
prof["goal_diff_per_game"] = prof["goal_diff"] / prof["gp"]
prof = prof.drop(columns=["gf_per_game"])

# Conference + playoffs annotations
prof["conference"] = prof["team"].map(CONFERENCE)
prof["made_playoffs"] = False
prof["round1_win"]    = False
prof["round2_win"]    = False
prof["cf_win"]        = False
prof["cup_win"]       = False
for s, info in PLAYOFFS.items():
    mask = prof["season"]==s
    prof.loc[mask & prof["team"].isin(info["playoffs"]),         "made_playoffs"] = True
    prof.loc[mask & prof["team"].isin(info["round1_winner"]),    "round1_win"]   = True
    prof.loc[mask & prof["team"].isin(info["round2_winner"]),    "round2_win"]   = True
    prof.loc[mask & prof["team"].isin(info["conf_final_winner"]),"cf_win"]       = True
    prof.loc[mask & prof["team"].isin(info["cup_winner"]),       "cup_win"]      = True

# top-4-in-conference flag (cup contender per spec)
prof["conf_rank"] = prof.groupby(["season","conference"])["points"]\
                        .rank(ascending=False, method="min").astype(int)
prof["contender_top4_conf"] = prof["conf_rank"] <= 4

# ---------------------------------------------------------------
# Step 2 — Archetype classification
# ---------------------------------------------------------------
print("Classifying archetypes...")
fwd_med = prof["F_top6_twoway"].median()
d_med   = prof["D_top4_twoway"].median()
def archetype(row):
    f_hi = row["F_top6_twoway"] >= fwd_med
    d_hi = row["D_top4_twoway"] >= d_med
    if f_hi and d_hi:     return "A_HighF_HighD"
    if f_hi and not d_hi: return "B_HighF_LowD"
    if (not f_hi) and d_hi: return "C_LowF_HighD"
    return "D_LowF_LowD"
prof["archetype"] = prof.apply(archetype, axis=1)
print(f"  median F_top6_twoway={fwd_med:.3f}, median D_top4_twoway={d_med:.3f}")

# ---------------------------------------------------------------
# Step 3 — Archetype performance
# ---------------------------------------------------------------
print("Step 3 — Archetype vs winning...")
arche_summary = prof.groupby("archetype").agg(
    n=("team","count"),
    mean_points=("points","mean"),
    mean_wins=("wins","mean"),
    mean_goal_diff_per_game=("goal_diff_per_game","mean"),
    mean_TNFI_pct=("TNFI_pct","mean"),
    playoff_rate=("made_playoffs","mean"),
    contender_rate=("contender_top4_conf","mean"),
    cup_rate=("cup_win","mean"),
).round(3)

# ANOVA on points across archetypes
from scipy import stats
groups = [prof[prof["archetype"]==a]["points"].values
          for a in sorted(prof["archetype"].unique())]
F_stat, anova_p = stats.f_oneway(*groups)

# ---------------------------------------------------------------
# Step 4 — Goalie impact within archetype
# ---------------------------------------------------------------
print("Step 4 — Goalie impact within archetype...")
g_med = prof["starter_NFI_GSAx_per60"].median()
prof["g_high"] = prof["starter_NFI_GSAx_per60"] >= g_med
prof["arche_g"] = prof["archetype"] + "_" + np.where(prof["g_high"], "Ghi", "Glo")

g_split = prof.groupby(["archetype","g_high"]).agg(
    n=("team","count"),
    mean_points=("points","mean"),
    playoff_rate=("made_playoffs","mean"),
    contender_rate=("contender_top4_conf","mean"),
    mean_goal_diff_per_game=("goal_diff_per_game","mean"),
).round(3).reset_index()

# ---------------------------------------------------------------
# Step 5 — Hyman hypothesis: A vs B controlling for talent (TNFI%)
# ---------------------------------------------------------------
print("Step 5 — Hyman hypothesis OLS (A vs B controlling for TNFI%)...")
ab = prof[prof["archetype"].isin(["A_HighF_HighD","B_HighF_LowD"])].copy()
ab["is_A"] = (ab["archetype"]=="A_HighF_HighD").astype(int)
# OLS: points = b0 + b1*is_A + b2*TNFI_pct
from numpy.linalg import lstsq
X = np.column_stack([np.ones(len(ab)), ab["is_A"].values,
                     ab["TNFI_pct"].values])
y = ab["points"].values
beta, *_ = lstsq(X, y, rcond=None)
y_hat = X @ beta
resid = y - y_hat
n, k = X.shape
sig2 = (resid**2).sum() / (n - k)
cov = sig2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))
t_stats = beta / se
from scipy.stats import t as t_dist
p_vals = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n-k))

hyman_table = pd.DataFrame({
    "var": ["intercept","is_A_vs_B","TNFI_pct"],
    "beta": beta,
    "se": se,
    "t": t_stats,
    "p": p_vals,
}).round(4)

# ---------------------------------------------------------------
# Step 6 — Cup contention logistic regressions
# ---------------------------------------------------------------
print("Step 6 — Logistic regressions per playoff outcome...")
import warnings
warnings.filterwarnings("ignore")

def logit_fit(X, y):
    """Newton-IRLS logistic regression with L2 ridge for stability."""
    n, k = X.shape
    beta = np.zeros(k)
    ridge = 1e-4 * np.eye(k)
    ridge[0,0] = 0.0  # don't ridge intercept
    for _ in range(50):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(p, 1e-9, 1-1e-9)
        W = p * (1 - p)
        # Newton step
        H = X.T @ (W[:,None] * X) + ridge
        g = X.T @ (y - p) - ridge @ beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-7:
            break
    # SE from inverse Hessian
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    z = beta / se
    p_val = 2 * (1 - stats.norm.cdf(np.abs(z)))
    # log-loss / pseudo-R2 (McFadden)
    eta = X @ beta
    ll_full = np.sum(y*eta - np.log1p(np.exp(eta)))
    p0 = y.mean()
    ll_null = len(y) * (p0*np.log(p0+1e-12) + (1-p0)*np.log(1-p0+1e-12))
    r2 = 1.0 - ll_full / ll_null if ll_null != 0 else np.nan
    return beta, se, z, p_val, r2

# Restrict to completed playoff seasons
seg = prof[prof["season"].isin(COMPLETED_PLAYOFF_SEASONS)].copy()
# Drop rows with NaN in features
features = ["F_top6_twoway","F_bot6_twoway","D_top4_twoway",
            "starter_NFI_GSAx_per60","TNFI_pct"]
seg = seg.dropna(subset=features).copy()

# Standardize features
feat_means = seg[features].mean()
feat_stds  = seg[features].std(ddof=0)
Xz = ((seg[features] - feat_means) / feat_stds).values
X_full = np.column_stack([np.ones(len(seg)), Xz])

logit_results = []
for outcome in ["made_playoffs","round1_win","round2_win","cf_win","cup_win"]:
    y = seg[outcome].astype(int).values
    if y.sum() < 4 or y.sum() > len(y) - 4:
        # Too few positives or negatives for stable fit; skip
        for i, fname in enumerate(["intercept"]+features):
            logit_results.append({
                "outcome": outcome, "n": len(y), "n_pos": int(y.sum()),
                "var": fname, "beta": np.nan, "se": np.nan, "z": np.nan,
                "p": np.nan, "OR": np.nan, "pseudo_R2": np.nan
            })
        continue
    beta, se, z, p_val, r2 = logit_fit(X_full, y)
    for i, fname in enumerate(["intercept"]+features):
        logit_results.append({
            "outcome": outcome,
            "n": len(y), "n_pos": int(y.sum()),
            "var": fname,
            "beta": round(beta[i], 4),
            "se":   round(se[i],   4) if not np.isnan(se[i]) else np.nan,
            "z":    round(z[i],    3),
            "p":    round(p_val[i], 4),
            "OR":   round(float(np.exp(beta[i])), 3),
            "pseudo_R2": round(r2, 3),
        })

logit_df = pd.DataFrame(logit_results)

# ---------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------
print("\nWriting outputs...")

# team_construction_model.csv
prof_out_cols = ["season","team","conference","gp","points","points_pct","wins",
                 "goal_diff","goal_diff_per_game","TNFI_pct","NFI_composite",
                 "F_top6_n","F_top6_twoway","F_bot6_n","F_bot6_twoway",
                 "D_top4_n","D_top4_twoway",
                 "starter_goalie_id","starter_goalie_name","starter_goalie_toi_min",
                 "starter_NFI_GSAx_per60","g_high",
                 "archetype","arche_g",
                 "made_playoffs","round1_win","round2_win","cf_win","cup_win",
                 "conf_rank","contender_top4_conf"]
prof[prof_out_cols].to_csv(f"{OUT}/team_construction_model.csv", index=False)

# team_archetypes.csv
arche_summary.to_csv(f"{OUT}/team_archetypes.csv")

# goalie_impact_by_archetype.csv
g_split.to_csv(f"{OUT}/goalie_impact_by_archetype.csv", index=False)

# cup_contention_profile.csv
logit_df.to_csv(f"{OUT}/cup_contention_profile.csv", index=False)

# ---------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 220
pd.options.display.max_columns = None

print("\n=== STEP 2-3: Archetype summary ===")
print(arche_summary.to_string())
print(f"\nANOVA on points across archetypes:  F={F_stat:.3f}, p={anova_p:.5f}, n={len(prof)}")

print("\n=== STEP 4: Goalie split within archetype ===")
print(g_split.to_string(index=False))

print("\n=== STEP 5: Hyman hypothesis OLS — points = a + b*is_A + c*TNFI% ===")
print(f"n_A={ab['is_A'].sum()}, n_B={(ab['is_A']==0).sum()}, total={len(ab)}")
print(hyman_table.to_string(index=False))

print("\n=== STEP 6: Logistic regression per outcome ===")
for outcome in ["made_playoffs","round1_win","round2_win","cf_win","cup_win"]:
    sub = logit_df[logit_df["outcome"]==outcome]
    if sub["beta"].isna().all():
        print(f"\n{outcome}: skipped (insufficient sample)")
        continue
    print(f"\nOutcome: {outcome}  (n={sub['n'].iloc[0]}, n_pos={sub['n_pos'].iloc[0]}, "
          f"pseudo R²={sub['pseudo_R2'].iloc[0]})")
    print(sub[["var","beta","se","z","p","OR"]].to_string(index=False))

print("\nFiles written:")
for f in ["team_construction_model.csv","team_archetypes.csv",
          "goalie_impact_by_archetype.csv","cup_contention_profile.csv"]:
    print(f"  {OUT}/{f}")
