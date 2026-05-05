#!/usr/bin/env python3
"""Test whether CNFI rebound control adds incremental predictive value
for team standings beyond NFI-GSAx /60.

Outputs: console report only (no charts, no CSV writes).
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
OUT = f"{ROOT}/NFI/output"

ES_CODES = {1551, 1441, 1331}
LEAGUE_ES_SA_PER_60 = 30.0   # standard league average; used to convert
                             # per-shot GSAx to a per-60-minutes proxy

REB = pd.read_csv(f"{OUT}/rebound_sequences.csv")
SHOTS = pd.read_csv(f"{OUT}/shots_tagged.csv")
# Pooled v2 goalie file — used here only for goalie_id → goalie_name lookup.
# All NFI_GSAx values used downstream are computed locally from shots_tagged.
NFI = pd.read_csv(f"{OUT}/goalie_nfi_gsax_pooled_v2.csv")
TOI = pd.read_csv(f"{OUT}/player_toi.csv")
STDG = pd.read_csv(f"{OUT}/standings_pool5.csv")

# ============================================================
# Per-(goalie, season) rebound control
# ============================================================
st = SHOTS.dropna(subset=["goalie_id"]).copy()
st["goalie_id"] = st["goalie_id"].astype("int64")
st["season"] = st["season"].astype(int)
st["defending_team"] = np.where(
    st["shooting_team_abbrev"] == st["home_team_abbrev"],
    st["away_team_abbrev"], st["home_team_abbrev"],
)

# Goalie -> defending team per (game, period) — used to attach goalie to
# rebound sequences strictly via shots_tagged.
gp_goalie = (
    st.groupby(["game_id", "period", "defending_team"])["goalie_id"]
      .agg(lambda s: s.value_counts().idxmax())
      .reset_index()
      .rename(columns={"goalie_id": "goalie_id_lookup"})
)
team_pair = SHOTS.groupby("game_id")[["home_team_abbrev", "away_team_abbrev"]].first().reset_index()

reb = REB[
    (REB["time_gap_secs"] <= 2.0)
    & (REB["orig_event_type"] == "shot-on-goal")
    & REB["orig_x"].between(74, 89)
    & REB["orig_y"].abs().le(9)
    & REB["situation_code"].isin(ES_CODES)
].copy()
reb["season"] = reb["season"].astype(int)
reb = reb.merge(team_pair, on="game_id", how="left")
reb["defending_team"] = np.where(
    reb["orig_team"] == reb["home_team_abbrev"],
    reb["away_team_abbrev"], reb["home_team_abbrev"],
)
reb = reb.merge(gp_goalie, on=["game_id", "period", "defending_team"], how="left")
reb["goalie_id"] = reb["goalie_id_lookup"].fillna(reb["orig_goalie_id"])
reb = reb.dropna(subset=["goalie_id"]).copy()
reb["goalie_id"] = reb["goalie_id"].astype("int64")

reb_season = (
    reb.groupby(["goalie_id", "season"]).agg(
        CNFI_rebound_attempts=("reb_event_type", "size"),
        CNFI_rebound_goals=("reb_is_goal", "sum"),
    ).reset_index()
)

# Per-goalie-season CNFI saves (denominator)
cnfi_shots = st[
    (st["state"] == "ES")
    & st["period"].between(1, 3)
    & st["x_coord_norm"].between(74, 89)
    & st["y_coord_norm"].abs().le(9)
    & (st["event_type"] == "shot-on-goal")
    & (st["is_goal_i"] == 0)
].copy()
saves_season = (
    cnfi_shots.groupby(["goalie_id", "season"]).size().rename("CNFI_saves").reset_index()
)

reb_full = saves_season.merge(reb_season, on=["goalie_id", "season"], how="left").fillna(
    {"CNFI_rebound_attempts": 0, "CNFI_rebound_goals": 0}
)
reb_full["CNFI_rebound_attempts"] = reb_full["CNFI_rebound_attempts"].astype(int)
reb_full["CNFI_rebound_goals"] = reb_full["CNFI_rebound_goals"].astype(int)
reb_full["CNFI_rebound_goal_rate"] = (
    reb_full["CNFI_rebound_goals"] / reb_full["CNFI_saves"] * 60
)

# ============================================================
# Per-(goalie, season) NFI-GSAx /60 proxy
#   xG = sum over shots of league_zone_state_goal_rate (per season)
#   GSAx = xG - actual goals
#   per-60 proxy = GSAx / faced * LEAGUE_ES_SA_PER_60
# ============================================================
shot_pool = st[
    (st["state"] == "ES")
    & st["period"].between(1, 3)
    & st["zone"].isin(["CNFI", "MNFI", "FNFI"])
    & st["event_type"].isin(["shot-on-goal", "goal"])
].copy()

lg_rate = (
    shot_pool.groupby(["season", "zone"])["is_goal_i"].mean()
             .rename("league_goal_rate").reset_index()
)
shot_pool = shot_pool.merge(lg_rate, on=["season", "zone"], how="left")
shot_pool["xG"] = shot_pool["league_goal_rate"]

gsax_season = (
    shot_pool.groupby(["goalie_id", "season"]).agg(
        faced=("is_goal_i", "size"),
        goals=("is_goal_i", "sum"),
        xG=("xG", "sum"),
    ).reset_index()
)
gsax_season["NFI_GSAx"] = gsax_season["xG"] - gsax_season["goals"]
gsax_season["NFI_GSAx_per60"] = (
    gsax_season["NFI_GSAx"] / gsax_season["faced"] * LEAGUE_ES_SA_PER_60
)

# ============================================================
# Goalie -> team -> starter per (team, season)
#   "Starter" = goalie with most ES shots faced for the team in that season
# ============================================================
goalie_team_season = (
    st.groupby(["goalie_id", "season", "defending_team"]).size().rename("faced").reset_index()
)
goalie_team_season = goalie_team_season.rename(columns={"defending_team": "team"})

# A goalie may face shots for multiple teams per season (trade); keep all rows.
# Then per (team, season) take goalie with max faced.
starter = (
    goalie_team_season.sort_values(["team", "season", "faced"], ascending=[True, True, False])
                      .drop_duplicates(["team", "season"], keep="first")
                      .rename(columns={"faced": "starter_faced"})
)
starter = starter.merge(NFI[["goalie_id", "goalie_name"]], on="goalie_id", how="left")

# ============================================================
# Merge starters with metrics + standings (seasons 22-23 to 24-25)
# ============================================================
TARGET_SEASONS = [20222023, 20232024, 20242025]
team_panel = starter[starter["season"].isin(TARGET_SEASONS)].copy()
team_panel = team_panel.merge(
    reb_full[["goalie_id", "season", "CNFI_saves", "CNFI_rebound_goal_rate"]],
    on=["goalie_id", "season"], how="left",
)
team_panel = team_panel.merge(
    gsax_season[["goalie_id", "season", "faced", "NFI_GSAx", "NFI_GSAx_per60"]],
    on=["goalie_id", "season"], how="left",
)
team_panel = team_panel.merge(
    STDG[["season", "team", "points", "wins"]], on=["season", "team"], how="left"
)

# Drop teams with very thin starter samples (e.g. < 200 ES shots faced — backup
# never named starter in practice but want clean signal).
team_panel = team_panel.dropna(subset=["points", "NFI_GSAx_per60", "CNFI_rebound_goal_rate"]).copy()
team_panel = team_panel[team_panel["starter_faced"] >= 300].copy()

print("=" * 72)
print(f"Step 1: Team panel — {len(team_panel)} (team, season) rows "
      f"across {team_panel['season'].nunique()} seasons.")
print(team_panel[["season", "team", "goalie_name", "starter_faced",
                  "NFI_GSAx_per60", "CNFI_rebound_goal_rate", "points"]]
      .sort_values(["season", "team"]).head(15).to_string(index=False))


# ============================================================
# Step 2: OLS regressions
# ============================================================
def ols(X, y):
    """Return (beta_dict, R2, n)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    Xc = np.hstack([np.ones((n, 1)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    yhat = Xc @ beta
    resid = y - yhat
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    df_resid = n - k - 1
    sigma2 = ss_res / df_resid
    cov = sigma2 * np.linalg.inv(Xc.T @ Xc)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df_resid))
    return beta, se, tvals, pvals, r2, n

y = team_panel["points"].values
X1 = team_panel[["NFI_GSAx_per60"]].values
X2 = team_panel[["CNFI_rebound_goal_rate"]].values
X3 = team_panel[["NFI_GSAx_per60", "CNFI_rebound_goal_rate"]].values

print("\n" + "=" * 72)
print("Step 2: OLS regressions — team standings points predicted by goalie metrics")
print(f"  n = {len(y)}  (team-season observations)\n")

for name, X, labels in [
    ("Model 1: points ~ NFI_GSAx_per60", X1, ["intercept", "NFI_GSAx_per60"]),
    ("Model 2: points ~ CNFI_rebound_goal_rate", X2, ["intercept", "CNFI_rebound_goal_rate"]),
    ("Model 3: points ~ NFI_GSAx_per60 + CNFI_rebound_goal_rate", X3,
     ["intercept", "NFI_GSAx_per60", "CNFI_rebound_goal_rate"]),
]:
    beta, se, tvals, pvals, r2, n = ols(X, y)
    print(f"--- {name} ---")
    print(f"  R² = {r2:.4f}    n = {n}")
    for lab, b, s, p in zip(labels, beta, se, pvals):
        print(f"  {lab:30s}  β = {b:+.3f}   SE = {s:.3f}   p = {p:.4f}")
    print()

# Incremental R²
beta1, _, _, _, r2_1, _ = ols(X1, y)
_, _, _, _, r2_3, _ = ols(X3, y)
inc = r2_3 - r2_1
# F-test on incremental
n = len(y)
ss_tot = ((y - y.mean()) ** 2).sum()
ss_res_1 = ss_tot * (1 - r2_1)
ss_res_3 = ss_tot * (1 - r2_3)
F = ((ss_res_1 - ss_res_3) / 1) / (ss_res_3 / (n - 3))
p_F = 1 - stats.f.cdf(F, 1, n - 3)
print(f"Incremental R² (Model 3 - Model 1): {inc:+.4f}")
print(f"F({1},{n-3}) = {F:.3f}   p = {p_F:.4f}\n")

# ============================================================
# Step 3: Year-over-year stability of rebound control
# ============================================================
# Use per-season rebound control with min 200 CNFI saves to keep noise low
yoy_pool = reb_full[reb_full["CNFI_saves"] >= 200].copy()
yoy_pool = yoy_pool[yoy_pool["season"].isin(TARGET_SEASONS + [20212022, 20252026])]
yoy_pool["next_season"] = yoy_pool["season"] + 10001  # 20222023 -> 20232024
yoy_join = yoy_pool.merge(
    yoy_pool[["goalie_id", "season", "CNFI_rebound_goal_rate"]]
        .rename(columns={"season": "next_season",
                         "CNFI_rebound_goal_rate": "next_rate"}),
    on=["goalie_id", "next_season"], how="inner",
)

print("=" * 72)
print("Step 3: Year-over-year stability (CNFI rebound goal rate)")
print(f"  n_pairs = {len(yoy_join)}    "
      f"({yoy_join['goalie_id'].nunique()} unique goalies)")
if len(yoy_join) >= 3:
    r, p = stats.pearsonr(yoy_join["CNFI_rebound_goal_rate"], yoy_join["next_rate"])
    sig = "yes" if p < 0.05 else "no"
    print(f"  Pearson r = {r:.3f}   p = {p:.4f}   significant @ 0.05? {sig}")
else:
    print("  Too few pairs.")

# Show season-pair breakdown
print("\n  Season pairs in sample:")
print(
    yoy_join.groupby(["season", "next_season"]).size().rename("n_pairs").to_string()
)

# ============================================================
# Step 4: Flag divergent goalies (rebound control vs GSAx)
# Use POOLED metrics (>=500 CNFI saves) so z-scores are stable
# ============================================================
pool_reb = (
    reb_full.groupby("goalie_id").agg(
        CNFI_saves=("CNFI_saves", "sum"),
        CNFI_rebound_goals=("CNFI_rebound_goals", "sum"),
    ).reset_index()
)
pool_reb["CNFI_rebound_goal_rate"] = (
    pool_reb["CNFI_rebound_goals"] / pool_reb["CNFI_saves"] * 60
)
pool_gsax = (
    gsax_season.groupby("goalie_id").agg(
        faced=("faced", "sum"),
        goals=("goals", "sum"),
        xG=("xG", "sum"),
    ).reset_index()
)
pool_gsax["NFI_GSAx"] = pool_gsax["xG"] - pool_gsax["goals"]
pool_gsax["NFI_GSAx_per60"] = pool_gsax["NFI_GSAx"] / pool_gsax["faced"] * LEAGUE_ES_SA_PER_60

merged = pool_reb.merge(pool_gsax[["goalie_id", "faced", "NFI_GSAx_per60"]], on="goalie_id")
merged = merged[merged["CNFI_saves"] >= 500].copy()
merged = merged.merge(NFI[["goalie_id", "goalie_name"]], on="goalie_id", how="left")

# z-scores: GOOD direction
merged["z_rebound_GOOD"] = -(
    (merged["CNFI_rebound_goal_rate"] - merged["CNFI_rebound_goal_rate"].mean())
    / merged["CNFI_rebound_goal_rate"].std(ddof=0)
)
merged["z_GSAx_GOOD"] = (
    (merged["NFI_GSAx_per60"] - merged["NFI_GSAx_per60"].mean())
    / merged["NFI_GSAx_per60"].std(ddof=0)
)

merged["gap"] = merged["z_GSAx_GOOD"] - merged["z_rebound_GOOD"]
hidden_risk = merged[(merged["z_GSAx_GOOD"] > 0) & (merged["z_rebound_GOOD"] < 0) & (merged["gap"] > 1.5)]
undervalued = merged[(merged["z_GSAx_GOOD"] < 0) & (merged["z_rebound_GOOD"] > 0) & (merged["gap"] < -1.5)]

print("\n" + "=" * 72)
print("Step 4: Divergent goalies (|gap| > 1.5 SD, opposite directions)")
print(f"\n  HIDDEN RISK  — good GSAx /60 but bad rebound control:")
if len(hidden_risk):
    print(hidden_risk.sort_values("gap", ascending=False)[
        ["goalie_name", "CNFI_saves", "NFI_GSAx_per60", "z_GSAx_GOOD",
         "CNFI_rebound_goal_rate", "z_rebound_GOOD", "gap"]
    ].to_string(index=False))
else:
    print("    (none)")
print(f"\n  UNDERVALUED  — bad GSAx /60 but good rebound control:")
if len(undervalued):
    print(undervalued.sort_values("gap")[
        ["goalie_name", "CNFI_saves", "NFI_GSAx_per60", "z_GSAx_GOOD",
         "CNFI_rebound_goal_rate", "z_rebound_GOOD", "gap"]
    ].to_string(index=False))
else:
    print("    (none)")

print("\nNote: NFI_GSAx_per60 here uses a per-season league zone-state goal-rate")
print(f"      and a {LEAGUE_ES_SA_PER_60:.0f}-shots-per-60 proxy for ES TOI.")
print("      Rankings/coefficients vs the canonical pipeline are equivalent up to scale.")
