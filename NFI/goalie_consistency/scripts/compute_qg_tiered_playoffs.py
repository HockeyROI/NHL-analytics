"""Playoff QG%s / QG%b — playoff companion to compute_qg_tiered.py.

Same save%-based Quality Start idea (per-game save% on shots-on-goal only,
scope = 5v5 or all situations), but on PLAYOFF games (isPlayoffGame == 1).

Playoff games DON'T get their own starter/backup tier split: playoff rosters
are heavily skewed toward starters (a true backup often plays 0-2 games), so
a fresh top-32/next-32 GP ranking on playoff-only samples would be unstable
and largely meaningless. Instead, each playoff game is judged against THAT
SEASON'S REGULAR-SEASON starter/backup baseline (read from
qg_tier_baselines_by_season{suffix}.csv, produced by compute_qg_tiered.py) —
the same "what does a good starter/backup save% look like this year" bar,
just applied to playoff performance instead of regular-season performance.

NO goalie-level qualifying floor (matches the rest of the playoff pipeline,
e.g. compute_qs_gsax_playoffs.py) — every goalie with a qualifying game
appears, per playoff season PLUS an `all_playoffs` pooled row (each pooled
game judged by its own season's baseline, then summed).

Output per scope (suffix "" = 5v5, "_allsit" = all situations):
  NFI/goalie_consistency/output/qg_savepct_playoffs{suffix}.csv
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _data_sources as _ds

NAMES_FILE = Path("/Users/ashgarg/Documents/HockeyROI/NFI/output/player_positions.csv")
OUT_DIR = Path("/Users/ashgarg/Documents/HockeyROI/NFI/goalie_consistency/output")

SEASONS = {20222023, 20232024, 20242025, 20252026}
MIN_SHOTS_PER_GAME = 10   # metric definition (NOT a goalie qualifying floor)
SCOPES = {"5v5": "", "all": "_allsit"}


def wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (center - margin) / denom


def run_scope(scope, suffix):
    print(f"\n{'='*70}\nSCOPE = {scope!r}\n{'='*70}")
    baseline_fp = OUT_DIR / f"qg_tier_baselines_by_season{suffix}.csv"
    if not baseline_fp.exists():
        raise FileNotFoundError(
            f"Missing {baseline_fp} — run compute_qg_tiered.py first (regular "
            f"season baselines are a prerequisite for the playoff build).")
    baselines = pd.read_csv(baseline_fp)
    b_starter = baselines[baselines["tier"] == "starter"].set_index("season")["baseline_save_pct"] / 100
    b_backup = baselines[baselines["tier"] == "backup"].set_index("season")["baseline_save_pct"] / 100

    allshots = _ds.load_shots_mp_schema()
    mask = (allshots["season"].isin(SEASONS) & (allshots["isPlayoffGame"] == 1)
            & (allshots["goalieIdForShot"].notna()) & (allshots["event"] != "MISS")
            & (allshots["period"] <= 3))
    if scope == "5v5":
        mask &= (allshots["homeSkatersOnIce"] == 5) & (allshots["awaySkatersOnIce"] == 5)
    df = allshots[mask]
    pg = (df.groupby(["game_id", "goalieIdForShot", "season"])
          .agg(shots_faced=("goal", "size"), goals=("goal", "sum")).reset_index()
          .rename(columns={"goalieIdForShot": "goalie_id"}))
    pg["saves"] = pg["shots_faced"] - pg["goals"]
    per_game = pg[pg["shots_faced"] >= MIN_SHOTS_PER_GAME].copy()
    per_game["goalie_id"] = per_game["goalie_id"].astype(int)
    per_game["game_save_pct"] = per_game["saves"] / per_game["shots_faced"]
    for s in sorted(per_game["season"].unique()):
        print(f"  season {s}: {int((per_game['season']==s).sum())} qualifying game-goalie rows")
    per_game["baseline_starter"] = per_game["season"].map(b_starter)
    per_game["baseline_backup"] = per_game["season"].map(b_backup)
    if per_game["baseline_starter"].isna().any() or per_game["baseline_backup"].isna().any():
        missing = sorted(per_game.loc[per_game["baseline_starter"].isna(), "season"].unique())
        raise ValueError(f"No regular-season baseline for playoff season(s) {missing} — "
                          f"re-run compute_qg_tiered.py to cover them.")
    per_game["is_QGs"] = (per_game["game_save_pct"] >= per_game["baseline_starter"]).astype(int)
    per_game["is_QGb"] = (per_game["game_save_pct"] >= per_game["baseline_backup"]).astype(int)

    _names = pd.read_csv(NAMES_FILE)
    _canon = dict(zip(_names["player_id"].astype(int), _names["player_name"].astype(str)))
    canon = {g: _canon.get(g, str(g)) for g in per_game["goalie_id"].unique()}

    def agg(group_cols, label=None):
        a = (per_game.groupby(group_cols)
             .agg(GP=("game_id", "size"), QGs_games=("is_QGs", "sum"),
                  QGb_games=("is_QGb", "sum")).reset_index())
        if label is not None:
            a["season"] = label
        return a

    out = pd.concat([agg(["goalie_id", "season"]), agg(["goalie_id"], "all_playoffs")],
                    ignore_index=True)
    out["goalie_name"] = out["goalie_id"].map(canon)
    out["QG_pct_s"] = out["QGs_games"] / out["GP"] * 100
    out["QG_pct_b"] = out["QGb_games"] / out["GP"] * 100
    out["QG_pct_s_lo"] = out.apply(lambda r: wilson_lower(r["QGs_games"], r["GP"]) * 100, axis=1)
    out["QG_pct_b_lo"] = out.apply(lambda r: wilson_lower(r["QGb_games"], r["GP"]) * 100, axis=1)
    out = out[["goalie_id", "goalie_name", "season", "GP", "QGs_games", "QGb_games",
               "QG_pct_s", "QG_pct_b", "QG_pct_s_lo", "QG_pct_b_lo"]]
    out = out.sort_values(["season", "QG_pct_s"], ascending=[True, False]).reset_index(drop=True)

    out_fp = OUT_DIR / f"qg_savepct_playoffs{suffix}.csv"
    out.to_csv(out_fp, index=False)
    print(f"\nWrote {out_fp} — {len(out)} rows, {out['goalie_id'].nunique()} goalies")
    ap = out[(out["season"] == "all_playoffs") & (out["GP"] >= 8)]
    print(f"\n=== all_playoffs top 10 by QG%s (GP>=8), scope={scope} ===")
    print(ap.nlargest(10, "QG_pct_s")[["goalie_name", "GP", "QG_pct_s", "QG_pct_b"]]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    for scope, suffix in SCOPES.items():
        run_scope(scope, suffix)
    print("\nDONE — both scopes written.")
