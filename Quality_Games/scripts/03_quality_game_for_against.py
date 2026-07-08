#!/usr/bin/env python3
"""
Quality_Games / Step 3 — Quality Game For / Against split.

Extends the Quality Game framework to separate a game's OFFENSE quality ("For")
from its DEFENSE quality ("Against"), for both xG and NFI, at the player-season
and team-season level. The existing QG metrics (02_quality_game_aggregation.py)
use an on-ice SHARE, for/(for+against), which collapses offense and defense into
one number; this step splits them by grading the For and Against per-60 rates
separately against league position-median rates.

Reads:    Quality_Games/output/per_player_game{_SUF}.csv
Produces: Quality_Games/output/per_player_qg_fa{_SUF}.csv
          Quality_Games/output/per_team_qg_fa{_SUF}.csv
          Quality_Games/output/position_medians_fa{_SUF}.csv

Set QG_SCOPE=playoff for the playoff build (reads *_playoffs input, writes
*_playoffs outputs with an `all_playoffs` pooled row), matching step 02.

Methodology (mirrors 02_quality_game_aggregation.py's conventions):
  Per-game qualifying floor:  TOI_on_sec >= 480 (8 min) AND attempts >= 5.
  Per-game per-60 rates on qualifying games:
    xG_for_60  = xG_for  / TOI_on_sec * 3600     NFI_for_60 = NFI_for / TOI_on_sec * 3600
    xG_ag_60   = xG_ag   / TOI_on_sec * 3600     NFI_ag_60  = NFI_ag  / TOI_on_sec * 3600
  Position-median thresholds (F / D, league-wide, all seasons pooled): the median
    of each of the four per-60 rates over qualifying known-position player-games.
  Quality-Game flags (a game "qualifies" on the For or Against dimension):
    For  (offense, HIGHER rate = better):
      is_xG_QG_F  = 1 if xG_for_60  >= pos-median xG_for_60   (>=, like absolute xG-QG)
      is_NFI_QG_F = 1 if NFI_for_60 >  pos-median NFI_for_60,  0.5 if ==, 0 if <
    Against (defense, LOWER rate = better):
      is_xG_QG_A  = 1 if xG_ag_60   <= pos-median xG_ag_60
      is_NFI_QG_A = 1 if NFI_ag_60  <  pos-median NFI_ag_60,   0.5 if ==, 0 if >
    NFI keeps the half-credit-tie rule from step 02 (NFI per-60 rates land on
    discrete values more often than the continuous xG weights); xG uses >= / <=.
  Player-season rate = sum(flag) / (qualifying games with a valid rate).
  Team-season = TOI-weighted mean of player-season rates (20+ GP with the team;
    playoffs drop this floor to 1), matching step 02's team aggregation.
"""
import os
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Documents/HockeyROI"
OUT_DIR = f"{ROOT}/Quality_Games/output"

MIN_TOI_SEC = 480
MIN_ATTEMPTS = 5
TEAM_GP_FLOOR = 20

SCOPE = os.environ.get("QG_SCOPE", "regular")
_IS_PLAYOFF = SCOPE == "playoff"
_SUF = "_playoffs" if _IS_PLAYOFF else ""
if _IS_PLAYOFF:
    TEAM_GP_FLOOR = 1

# The four per-60 rate metrics and their "quality" direction (True = higher is
# better / For; False = lower is better / Against). NFI uses the half-credit tie.
METRICS = [
    ("xG_QG_F",  "xG_for",  True,  False),   # (out col, base col, higher_better, half_credit_tie)
    ("xG_QG_A",  "xG_ag",   False, False),
    ("NFI_QG_F", "NFI_for", True,  True),
    ("NFI_QG_A", "NFI_ag",  False, True),
]


def log(msg=""):
    print(str(msg), flush=True)


log("=" * 78)
log("Quality_Games — 03_quality_game_for_against.py  (scope=%s)" % SCOPE)
log("=" * 78)

df = pd.read_csv(f"{OUT_DIR}/per_player_game{_SUF}.csv")
log(f"Loaded {len(df):,} player-game rows")

df["attempts_total"] = df.attempts_for + df.attempts_ag
qual = df[(df.TOI_on_sec >= MIN_TOI_SEC) & (df.attempts_total >= MIN_ATTEMPTS)].copy()
log(f"Qualifying player-games: {len(qual):,} / {len(df):,} ({len(qual)/len(df):.1%})")

# Per-60 rates for the four base counters (guard TOI>0; qualifying floor ensures it).
_sec = qual.TOI_on_sec.astype(float)
for _out, _base, _, _ in METRICS:
    qual[f"{_base}_60"] = np.where(_sec > 0, qual[_base].astype(float) / _sec * 3600.0, np.nan)

# Position medians (F / D), pooled over all qualifying known-position games.
qk = qual[qual.position.isin(["F", "D"])].copy()
medians = {}
med_rows = []
for pos in ["F", "D"]:
    sub = qk[qk.position == pos]
    for _out, _base, _hi, _ in METRICS:
        m = float(sub[f"{_base}_60"].dropna().median())
        medians[(pos, _base)] = m
        med_rows.append({"position": pos, "metric": _out, "base_col": _base,
                         "median_per60": m,
                         "n_qualifying_games": int(sub[f"{_base}_60"].notna().sum())})
pd.DataFrame(med_rows).to_csv(f"{OUT_DIR}/position_medians_fa{_SUF}.csv", index=False)
log(f"Wrote position_medians_fa{_SUF}.csv")
for r in med_rows:
    log(f"  {r['position']} {r['metric']:<9} median/60={r['median_per60']:.4f}  n={r['n_qualifying_games']:,}")

# Flags per qualifying player-game.
for _out, _base, _hi, _tie in METRICS:
    thr = qk.position.map({"F": medians[("F", _base)], "D": medians[("D", _base)]})
    rate = qk[f"{_base}_60"]
    if _hi:                       # For: higher rate is better
        if _tie:
            flag = np.where(rate.isna(), np.nan,
                    np.where(rate > thr, 1.0, np.where(rate == thr, 0.5, 0.0)))
        else:
            flag = np.where(rate.isna(), np.nan, (rate >= thr).astype(float))
    else:                         # Against: lower rate is better
        if _tie:
            flag = np.where(rate.isna(), np.nan,
                    np.where(rate < thr, 1.0, np.where(rate == thr, 0.5, 0.0)))
        else:
            flag = np.where(rate.isna(), np.nan, (rate <= thr).astype(float))
    qk[f"is_{_out}"] = flag
    log(f"  is_{_out}: flag-rate {np.nanmean(qk[f'is_{_out}']):.3f} "
        f"(valid {int(qk[f'is_{_out}'].notna().sum()):,})")

# ---- Per (player, season, team) counts ----
agg_spec = {}
for _out, _b, _h, _t in METRICS:
    agg_spec[f"{_out}_count"] = (f"is_{_out}", "sum")
    agg_spec[f"{_out}_qual_GP"] = (f"is_{_out}", lambda x: x.notna().sum())
pst = qk.groupby(["player_id", "season", "team_abbrev"]).agg(**agg_spec).reset_index()
# GP + TOI from the full df (played, qualifying or not).
gp = df.groupby(["player_id", "season", "team_abbrev"]).agg(
    GP=("game_id", "size"), TOI_total_sec=("TOI_on_sec", "sum"),
    position=("position", lambda x: x.mode().iloc[0] if not x.mode().empty else "")).reset_index()
pst = gp.merge(pst, on=["player_id", "season", "team_abbrev"], how="left")


def _pct(g, out):
    c, q = g[f"{out}_count"], g[f"{out}_qual_GP"]
    return np.where(q > 0, c / q, np.nan)


def _player_agg(keys, season_label=None):
    spec = {"GP": ("GP", "sum"), "TOI_total_sec": ("TOI_total_sec", "sum"),
            "position": ("position", lambda x: x.mode().iloc[0] if not x.mode().empty else "")}
    for _out, _b, _h, _t in METRICS:
        spec[f"{_out}_count"] = (f"{_out}_count", "sum")
        spec[f"{_out}_qual_GP"] = (f"{_out}_qual_GP", "sum")
    g = pst.groupby(keys).agg(**spec).reset_index()
    if season_label is not None:
        g["season"] = season_label
    for _out, _b, _h, _t in METRICS:
        g[f"{_out}_pct"] = _pct(g, _out)
    return g


ps = _player_agg(["player_id", "season"])
if _IS_PLAYOFF:
    ps = pd.concat([ps, _player_agg(["player_id"], "all_playoffs")], ignore_index=True)
ps = ps[ps.position.isin(["F", "D"])].copy()

_pcols = [f"{_out}_pct" for _out, _b, _h, _t in METRICS]
# Emit counts + qual_GP too so the app can pool multi-season views by ratio-of-sums
# (sum counts / sum qual_GP), matching the existing QG pooling in step 02 / the app.
_ccols = []
for _out, _b, _h, _t in METRICS:
    _ccols += [f"{_out}_count", f"{_out}_qual_GP"]
ps_out = ps[["player_id", "season", "position", "GP", "TOI_total_sec"] + _ccols + _pcols]
ps_out.to_csv(f"{OUT_DIR}/per_player_qg_fa{_SUF}.csv", index=False)
log(f"Wrote per_player_qg_fa{_SUF}.csv ({len(ps_out):,} rows)")

# ---- Per (team, season): TOI-weighted mean of player rates, 20+ GP eligibility ----
pst_rates = pst.merge(ps[["player_id", "season"] + _pcols], on=["player_id", "season"], how="inner")
elig = pst_rates[pst_rates.GP >= TEAM_GP_FLOOR].copy()


def _tw_mean(grp, col):
    v = grp.dropna(subset=[col])
    w = v["TOI_total_sec"].astype(float).values
    return float(np.average(v[col].values, weights=w)) if len(v) and w.sum() > 0 else np.nan


team_rows = []
for (team, season), grp in elig.groupby(["team_abbrev", "season"]):
    row = {"team_abbrev": team, "season": int(season),
           "n_eligible_players": int(len(grp)),
           "total_team_TOI_min": round(grp.TOI_total_sec.sum() / 60, 1)}
    for _out, _b, _h, _t in METRICS:
        row[f"team_{_out}_pct"] = _tw_mean(grp, f"{_out}_pct")
    team_rows.append(row)
if _IS_PLAYOFF:
    pool = pst.groupby(["player_id", "team_abbrev"]).agg(
        GP=("GP", "sum"), TOI_total_sec=("TOI_total_sec", "sum")).reset_index()
    pool = pool.merge(ps[ps.season == "all_playoffs"][["player_id"] + _pcols], on="player_id", how="inner")
    for team, grp in pool[pool.GP >= TEAM_GP_FLOOR].groupby("team_abbrev"):
        row = {"team_abbrev": team, "season": "all_playoffs",
               "n_eligible_players": int(len(grp)),
               "total_team_TOI_min": round(grp.TOI_total_sec.sum() / 60, 1)}
        for _out, _b, _h, _t in METRICS:
            row[f"team_{_out}_pct"] = _tw_mean(grp, f"{_out}_pct")
        team_rows.append(row)
team_df = pd.DataFrame(team_rows)
if _IS_PLAYOFF:
    team_df["season"] = team_df["season"].astype(str)
team_df = team_df.sort_values(["season", "team_abbrev"]).reset_index(drop=True)
team_df.to_csv(f"{OUT_DIR}/per_team_qg_fa{_SUF}.csv", index=False)
log(f"Wrote per_team_qg_fa{_SUF}.csv ({len(team_df):,} rows)")
log("Done.")
