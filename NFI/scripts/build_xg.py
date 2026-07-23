#!/usr/bin/env python3
"""DEPRECATED (2026-07-23) — superseded by xG/build_xg.py (v2).

The canonical HockeyROI xG now lives at xG/build_xg.py → xG/output/
shot_xg_per_event.csv: a Fenwick-based gradient-boosted model with pre-shot
features (rebound, time-since-last, score/strength state) that correlates
0.99 with MoneyPuck's player-season xGoal. All consumers (build_pdo_sog[_
playoffs].py, build_situation_onice.py) were repointed to the xG/ folder.
This v1 geometry-only logistic model and its NFI/output/shot_xg_per_event.csv
output are retained only for reference/history; do not use for new work.

--- original v1 docstring below ---
Per-shot expected goals (xG) — SOG-conditional, for the PDOxG metric.

Trains a logistic model of P(goal | shot-on-goal) from shot geometry
(distance, angle, shot_type) and scores every shot-on-goal / goal event.
The target is deliberately SOG-conditional (denominator = shots on goal),
so the resulting xG is directly comparable to the SOG-based SH%/SV% that
PDO uses: expected SH% = xGF / SOG_for, expected SV% = 1 - xGA / SOG_against.

Reusable artifact — keyed (game_id, event_id) so any on-ice attribution
(build_pdo_sog.py, future GSAx, etc.) can merge xG onto shot events.

Output: NFI/Output/shot_xg_per_event.csv  (game_id, event_id, xg)

Notes / v1 scope:
- Trained on all-strength SOG (periods 1-3, empty-net excluded) for
  stability; scored per event by its own geometry. PDO consumers filter to
  the states they care about (e.g. ES) after the merge.
- Features: distance & angle to net (net at normalized x=+89, y=0),
  their squares and interaction, plus shot_type dummies. No pre-shot
  movement / rebound / rush flags in v1 — so xG spread is a lower bound.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
OUT_CSV = f"{ROOT}/NFI/output/shot_xg_per_event.csv"

NET_X = 89.0  # normalized coords: attacking net at x=+89, y=0

print("Loading shots...")
cols = ["game_id", "event_id", "season", "period", "event_type", "situation_code",
        "shooting_team_id", "home_team_id", "is_goal", "x_coord_norm", "y_coord_norm", "shot_type"]
s = pd.read_csv(SHOT_CSV, usecols=cols, dtype={"season": str, "situation_code": str})

# SOG set (shot-on-goal + goal), periods 1-3, drop empty-net (distorts finishing).
s = s[s["event_type"].isin(["shot-on-goal", "goal"]) & s["period"].between(1, 3)].copy()
sc = s["situation_code"].astype(str).str.zfill(4)
ag = sc.str[0].astype(int); hg = sc.str[3].astype(int)
empty_net = (ag == 0) | (hg == 0)
s = s[~empty_net].copy()
s["is_goal_i"] = s["is_goal"].astype(int)
print(f"  training/scoring SOG shots: {len(s):,}")

# --- geometry features ---
x = s["x_coord_norm"].astype(float)
y = s["y_coord_norm"].astype(float)
dist = np.sqrt((x - NET_X) ** 2 + y ** 2)
angle = np.abs(np.arctan2(y, (NET_X - x).clip(lower=0.1)))
st_dummies = pd.get_dummies(s["shot_type"].fillna("unk"), prefix="st")
feat = pd.concat([
    pd.DataFrame({"dist": dist, "dist2": dist ** 2, "angle": angle,
                  "angle2": angle ** 2, "dist_angle": dist * angle}, index=s.index),
    st_dummies,
], axis=1)

valid = feat.notna().all(axis=1) & dist.notna()
print(f"  rows with usable geometry: {valid.sum():,} of {len(s):,}")

print("Fitting logistic xG...")
clf = LogisticRegression(max_iter=5000, C=1.0)
clf.fit(feat[valid].values, s.loc[valid, "is_goal_i"].values)

xg = pd.Series(np.nan, index=s.index)
xg[valid] = clf.predict_proba(feat[valid].values)[:, 1]
# fallback for missing-coord shots: overall SOG->goal rate
xg = xg.fillna(s["is_goal_i"].mean())
s["xg"] = xg.values

# --- calibration report (decile pred vs actual) ---
overall = s["is_goal_i"].mean()
print(f"  overall goal rate {overall:.4f} | mean xG {s['xg'].mean():.4f}")
q = pd.qcut(s["xg"], 10, duplicates="drop")
cal = s.groupby(q, observed=True).agg(pred=("xg", "mean"), act=("is_goal_i", "mean"), n=("xg", "size"))
print("  calibration by xG decile (pred vs actual):")
for _, r in cal.iterrows():
    print(f"    pred {r['pred']:.3f}  act {r['act']:.3f}  (n={int(r['n']):,})")

out = s[["game_id", "event_id", "xg"]].copy()
out["xg"] = out["xg"].round(5)
out.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}: {len(out):,} scored SOG events.")
