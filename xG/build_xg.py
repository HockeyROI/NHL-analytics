#!/usr/bin/env python3
"""HockeyROI expected goals (xG) — v2, MoneyPuck-comparable.

Improves on the v1 geometry-only logistic model (NFI/scripts/build_xg.py):
  - Fenwick target: P(goal | UNBLOCKED shot attempt) over {shot-on-goal,
    missed-shot, goal}, so MISSES get real xG too (v1 was SOG-conditional and
    gave misses 0). This matches MoneyPuck's basis and makes on-ice xGF/xGA
    (incl. the per-situation engine) count misses correctly.
  - Pre-shot features: rebound flag + time since last attempt, prior-shot
    distance, running score state, strength (skater differential), home/away —
    on top of distance/angle/shot_type. (No "rush" flag: needs full non-shot
    play-by-play we don't store; it's the one MoneyPuck feature omitted.)
  - Gradient boosting (HistGradientBoostingClassifier) instead of logistic.

Empty-net shots are INCLUDED, with an empty_net feature, so they carry real
xG (they are real chances, and MoneyPuck counts them too). Periods 1-3.

Output: xG/output/shot_xg_per_event.csv  (game_id, event_id, xg)
Keyed (game_id, event_id) so every consumer merges xG onto shot events.

Validation: reports test-split AUC / log-loss / calibration, and correlates
player-season ixG totals against MoneyPuck's published xGoal.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "NFI" / "scripts"))
import _data_sources as _ds
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))
SHOT_CSV = ROOT / "Data" / "nhl_shot_events.csv"
MP_DIR = ROOT / "Quality_Games" / "Data" / "Money_puck"
OUT_CSV = Path(os.environ.get("XG_OUT", ROOT / "xG" / "output" / "shot_xg_per_event.csv"))
PBP_FEAT_CSV = ROOT / "xG" / "output" / "shot_pbp_features.csv"  # xG/build_pbp_features.py

NET_X = 89.0
FENWICK = ["shot-on-goal", "missed-shot", "goal"]
REBOUND_SEC = 3.0


def main() -> int:
    print("Loading shots...")
    cols = ["game_id", "event_id", "season", "period", "time_secs", "event_type",
            "situation_code", "shooting_team_id", "home_team_id", "is_goal",
            "x_coord_norm", "y_coord_norm", "shot_type", "shooter_player_id"]
    s = _ds.load_shot_events(usecols=cols,
                             dtype={"season": str, "situation_code": str})
    s = s[s["event_type"].isin(FENWICK) & s["period"].between(1, 3)].copy()
    s["is_goal_i"] = s["is_goal"].astype(int)
    s["abs_time"] = s["time_secs"].astype(int) + (s["period"].astype(int) - 1) * 1200
    s["shoot_home"] = s["shooting_team_id"] == s["home_team_id"]
    # Empty net = the DEFENDING team's goalie digit is 0.
    # These are KEPT and carry a model feature rather than being dropped. They
    # were previously excluded from train+score, which left them with no xG at
    # all — so on-ice xGF/xGA, PDOxG and GSAx silently ignored real chances, and
    # our GSAx could never line up with MoneyPuck's (which includes them).
    # Only the shot CHART hides EN shots, and it does that at draw time.
    _sc0 = s["situation_code"].astype(str).str.zfill(4)
    s["empty_net"] = np.where(s["shoot_home"],
                              _sc0.str[0].astype(int) == 0,
                              _sc0.str[3].astype(int) == 0).astype(int)
    s = s.sort_values(["game_id", "abs_time", "event_id"]).reset_index(drop=True)
    print(f"  Fenwick shots (train/score set): {len(s):,}")

    # --- geometry ---
    x = s["x_coord_norm"].astype(float)
    y = s["y_coord_norm"].astype(float)
    dist = np.sqrt((x - NET_X) ** 2 + y ** 2)
    angle = np.abs(np.arctan2(y, (NET_X - x).clip(lower=0.1)))
    s["dist"], s["angle"] = dist, angle

    # --- pre-shot context from FULL PBP (last EVENT, not just last shot) ---
    if PBP_FEAT_CSV.exists():
        pbp = pd.read_csv(PBP_FEAT_CSV).drop_duplicates(["game_id", "event_id"])
        s = s.merge(pbp, on=["game_id", "event_id"], how="left")  # keeps left order
        matched = s["time_since_last"].notna().mean()
        print(f"  PBP last-event context matched: {matched:.1%} of shots "
              f"(uncached 2020-22 fall back to defaults)")
    else:
        print("  WARNING: no PBP features file — run xG/build_pbp_features.py")
        for c in ("time_since_last", "dist_last", "last_type", "last_zone", "rush"):
            s[c] = np.nan
    s["time_since_last"] = s["time_since_last"].fillna(999.0).clip(0, 999)
    s["dist_last"] = s["dist_last"].fillna(float(np.nanmedian(dist)))
    s["last_type"] = s["last_type"].fillna("none")
    s["last_zone"] = s["last_zone"].fillna("U")
    s["rush"] = s["rush"].fillna(0).astype(int)
    # rebound = the previous EVENT was itself a shot, within 3s
    _shot_last = s["last_type"].isin(["shot-on-goal", "missed-shot", "blocked-shot", "goal"])
    s["rebound"] = ((s["time_since_last"] <= REBOUND_SEC) & _shot_last).astype(int)
    # running pre-shot score, shooter perspective (clip ±3)
    g = s.groupby("game_id", sort=False)
    s["goal_home"] = (s["is_goal_i"] == 1) & s["shoot_home"]
    s["goal_away"] = (s["is_goal_i"] == 1) & (~s["shoot_home"])
    h_pre = g["goal_home"].cumsum() - s["goal_home"].astype(int)
    a_pre = g["goal_away"].cumsum() - s["goal_away"].astype(int)
    s["score_diff"] = np.where(s["shoot_home"], h_pre - a_pre, a_pre - h_pre).clip(-3, 3)
    # strength = shooter skater advantage (clip ±2); recompute from situation_code
    # on the (possibly merge-reindexed) frame to stay row-aligned
    sc2 = s["situation_code"].astype(str).str.zfill(4)
    ask2, hsk2 = sc2.str[1].astype(int), sc2.str[2].astype(int)
    sh_sk = np.where(s["shoot_home"], hsk2, ask2)
    op_sk = np.where(s["shoot_home"], ask2, hsk2)
    s["strength_diff"] = np.clip(sh_sk - op_sk, -2, 2)
    s["is_home"] = s["shoot_home"].astype(int)
    # geometry recomputed as row-aligned Series on the merged frame
    xx = s["x_coord_norm"].astype(float); yy = s["y_coord_norm"].astype(float)
    dd = np.sqrt((xx - NET_X) ** 2 + yy ** 2)
    aa = np.abs(np.arctan2(yy, (NET_X - xx).clip(lower=0.1)))

    st_d = pd.get_dummies(s["shot_type"].fillna("unk"), prefix="st")
    lt_d = pd.get_dummies(s["last_type"], prefix="lt")
    lz_d = pd.get_dummies(s["last_zone"], prefix="lz")
    num = pd.DataFrame({
        "dist": dd, "angle": aa, "dist2": dd ** 2, "angle2": aa ** 2,
        "dist_angle": dd * aa, "time_since_last": s["time_since_last"],
        "log_tsl": np.log1p(s["time_since_last"]), "dist_last": s["dist_last"],
        "rebound": s["rebound"], "rush": s["rush"], "score_diff": s["score_diff"],
        "strength_diff": s["strength_diff"], "is_home": s["is_home"],
        "empty_net": s["empty_net"],
    }, index=s.index)
    feat = pd.concat([num, st_d.set_index(s.index), lt_d.set_index(s.index),
                      lz_d.set_index(s.index)], axis=1)
    dist = dd  # downstream references
    valid = feat.notna().all(axis=1) & dd.notna()
    X = feat[valid].astype(float).values
    yv = s.loc[valid, "is_goal_i"].values
    print(f"  usable rows: {valid.sum():,}  |  overall goal rate {yv.mean():.4f}")

    # --- held-out eval ---
    Xtr, Xte, ytr, yte = train_test_split(X, yv, test_size=0.2, random_state=7,
                                          stratify=yv)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31,
        l2_regularization=1.0, min_samples_leaf=200, random_state=7)
    clf.fit(Xtr, ytr)
    p_te = clf.predict_proba(Xte)[:, 1]
    print(f"\n  TEST AUC   {roc_auc_score(yte, p_te):.4f}")
    print(f"  TEST logloss {log_loss(yte, p_te):.4f}  "
          f"(baseline {log_loss(yte, np.full_like(p_te, ytr.mean())):.4f})")
    q = pd.qcut(p_te, 10, duplicates="drop")
    cal = pd.DataFrame({"p": p_te, "y": yte}).groupby(q, observed=True).agg(
        pred=("p", "mean"), act=("y", "mean"), n=("y", "size"))
    print("  calibration (pred vs actual by decile):")
    for _, r in cal.iterrows():
        print(f"    pred {r['pred']:.3f}  act {r['act']:.3f}  (n={int(r['n']):,})")

    # --- refit on all data, score every Fenwick event ---
    clf_full = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, min_samples_leaf=200, random_state=7)
    clf_full.fit(X, yv)
    xg = pd.Series(np.nan, index=s.index)
    xg[valid] = clf_full.predict_proba(X)[:, 1]
    xg = xg.fillna(s["is_goal_i"].mean())
    s["xg"] = xg.values
    print(f"\n  scored mean xG {s['xg'].mean():.4f}  vs goal rate {s['is_goal_i'].mean():.4f}")

    # --- validation vs MoneyPuck published xGoal (player-season totals) ---
    try:
        mine = (s.dropna(subset=["shooter_player_id"])
                .groupby([s["shooter_player_id"].astype(int), "season"])["xg"]
                .sum().rename("my_ixg").reset_index())
        mine.columns = ["playerId", "season", "my_ixg"]
        mine["yr"] = mine["season"].str[:4].astype(int)
        mp_parts = []
        for yr in sorted(mine["yr"].unique()):
            fp = MP_DIR / f"shots_{yr}.csv"
            if not fp.exists():
                continue
            # EN kept on BOTH sides so the comparison is like-for-like.
            m = pd.read_csv(fp, usecols=["shooterPlayerId", "season", "xGoal"])
            mp_parts.append(m.groupby(["shooterPlayerId", "season"])["xGoal"]
                            .sum().rename("mp_xg").reset_index())
        if mp_parts:
            mp = pd.concat(mp_parts)
            mp.columns = ["playerId", "yr", "mp_xg"]
            j = mine.merge(mp, on=["playerId", "yr"], how="inner")
            j = j[(j["my_ixg"] > 2) & (j["mp_xg"] > 2)]
            print(f"\n  vs MoneyPuck xGoal — {len(j):,} player-seasons matched")
            print(f"    Pearson r  {j['my_ixg'].corr(j['mp_xg']):.4f}")
            print(f"    Spearman   {j['my_ixg'].corr(j['mp_xg'], method='spearman'):.4f}")
            print(f"    mean mine {j['my_ixg'].mean():.1f}  vs MP {j['mp_xg'].mean():.1f}")
    except Exception as e:  # validation is best-effort, never blocks the build
        print(f"  (MoneyPuck validation skipped: {e})")

    out = s[["game_id", "event_id", "xg"]].copy()
    out["xg"] = out["xg"].round(5)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}: {len(out):,} scored Fenwick events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
