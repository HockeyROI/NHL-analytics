"""
Build QG%s / QG%b — tiered save%-based Quality Games.

Conventional "Quality Start" (Vollman, ~2009) flags a game if save% clears a
single league-average save% line. That line moves every season and, more
importantly, blends two very different populations: the ~32 goalies who carry
a starter's workload and the ~32 who play a backup's schedule. A backup who
clears the *starter* bar is genuinely playing like a 1; a backup judged
against the same bar as Hellebuyck is set up to fail. QG%s / QG%b replace the
single league-average line with two population-specific baselines, computed
fresh each season:

  - QG%s: share of a goalie's qualifying games where per-game save% clears
    that SEASON's starter-tier baseline save%.
  - QG%b: share of a goalie's qualifying games where per-game save% clears
    that SEASON's backup-tier baseline save%.

Both are computed for every goalie, regardless of which tier that goalie
themselves falls in — QG%s answers "does this goalie perform like a starter?"
and QG%b answers "does this goalie perform like a quality backup?" A goalie
can be measured against either bar; the Streamlit toggle just picks which
column to show.

Tiering (per season, independent of the game-level analysis below):
  - Rank all goalies who logged >=1 qualifying game that season by GP,
    descending (ties broken by season shots faced, then goalie_id).
  - Top 32 by GP = "starter" tier for that season.
  - Next 32 by GP = "backup" tier for that season.
  - Remainder = "depth" (not used to build either baseline).
  - Starter/backup baseline save% = TOTAL saves / TOTAL shots faced across
    all goalies in that tier that season (volume-weighted league save%,
    matching how "league average save%" is conventionally computed — NOT an
    unweighted average of goalies' individual save%s).
  - Tiering is computed independently per shot-scope run (see below): a
    goalie's GP-rank and thus tier can differ slightly between the 5v5 and
    all-situations runs since the qualifying-game floor is scope-specific.

Shot scope — this pipeline runs TWICE, once per scope, writing separate
output files (see OUTPUT FILES below). Neither is "the" QG%s/QG%b; Streamlit
carries a scope toggle alongside the existing game-type filter:
  - "5v5": 5v5 ES regulation only. Matches the shot scope used by the rest of
    the goalie consistency pipeline (QNFS%, GQG/QGx), but diverges from the
    traditional Quality Start convention.
  - "all": all situations (5v5 + PP + PK), regulation. Matches the
    conventional meaning of "league-average save%" that Quality Starts is
    built on elsewhere in hockey analytics — this is the scope most people
    mean when they say "league average save% is ~.895."
  Both scopes restrict to 5v5-or-not by the homeSkatersOnIce/awaySkatersOnIce
  columns; "all" simply skips that filter.

  - Regular season only (isPlayoffGame == 0)
  - Shots-on-goal only (event in {SHOT, GOAL}); missed shots are excluded from
    the faced denominator since a miss never reaches the goalie and save% is
    conventionally saves/SOG, not saves/Fenwick. (The rest of this pipeline's
    GSAx-based metrics correctly use the full Fenwick set instead, since the
    xG model prices in the shot-quality difference — that's not a bug there.)
  - Min 10 shots faced per game for the game to qualify (same floor as GQG)
  - Min 25 qualifying games per season for a goalie-season to be "qualified"
    for ranking (same floor as QNFS% / GQG)
  - Per-game save% = 1 - goals/shots_faced (goals from MoneyPuck xG shot log)
  - Quality game = per-game save% >= that season's tier baseline (>=, no
    half-credit ties — see tie-rate diagnostic printed below)
  - Wilson 95% lower bound for ranking

OUTPUT FILES (in NFI/goalie_consistency/output/):
  - qg_tier_baselines_by_season{suffix}.csv
  - qg_savepct_per_season_2022-2026{suffix}.csv
  - qg_savepct_2022-2026{suffix}.csv
  where suffix is "" for the 5v5 scope and "_allsit" for all-situations.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _data_sources as _ds

# ---- CONFIG ----
NAMES_FILE = Path("/Users/ashgarg/Documents/HockeyROI/NFI/output/player_positions.csv")
OUT_DIR = Path("/Users/ashgarg/Documents/HockeyROI/NFI/goalie_consistency/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = {
    "shots_2022.csv": 20222023,
    "shots_2023.csv": 20232024,
    "shots_2024.csv": 20242025,
    "shots_2025.csv": 20252026,
}

MIN_SHOTS_PER_GAME = 10
MIN_GP_PER_SEASON = 25
STARTER_TIER_SIZE = 32
BACKUP_TIER_SIZE = 32

SCOPES = {"5v5": "", "all": "_allsit"}

SIX_IDS = {
    "Jeremy Swayman": 8480280,
    "Connor Hellebuyck": 8476945,
    "Igor Shesterkin": 8478048,
    "Ilya Sorokin": 8478009,
    "John Gibson": 8476434,
    "Logan Thompson": 8480313,
}


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

    # ---- BUILD PER-GAME SAVE% (all 4 seasons, via the MP-schema shim) ----
    # Filter: regular season, valid goalie, shots-on-goal only (event != MISS),
    # + 5v5-only if scope == "5v5". Save% is saves / shots ON GOAL — a missed
    # shot never reaches the goalie, so it's excluded from the faced denominator
    # (unlike the GSAx/xG metrics, which use the full Fenwick set).
    allshots = _ds.load_shots_mp_schema()
    mask = (
        allshots["season"].isin(SEASONS.values())
        & (allshots["isPlayoffGame"] == 0)
        & (allshots["goalieIdForShot"].notna())
        & (allshots["event"] != "MISS")
        & (allshots["period"] <= 3)
    )
    if scope == "5v5":
        mask &= (allshots["homeSkatersOnIce"] == 5) & (allshots["awaySkatersOnIce"] == 5)
    df = allshots[mask]
    print(f"    {len(df):,} SOG after reg/SOG{'/5v5' if scope == '5v5' else ''} filters")

    pg = (
        df.groupby(["game_id", "goalieIdForShot", "season"])
        .agg(shots_faced=("goal", "size"), goals=("goal", "sum"))
        .reset_index()
        .rename(columns={"goalieIdForShot": "goalie_id"})
    )
    pg["saves"] = pg["shots_faced"] - pg["goals"]
    per_game = pg[pg["shots_faced"] >= MIN_SHOTS_PER_GAME].copy()
    per_game["goalie_id"] = per_game["goalie_id"].astype(int)
    per_game["game_save_pct"] = per_game["saves"] / per_game["shots_faced"]
    print(f"per-game rows (>= {MIN_SHOTS_PER_GAME} shots): {len(per_game):,}  "
          f"goalies: {per_game['goalie_id'].nunique()}")

    # ---- CANONICAL NAME MAP (same pattern as compute_qs_gsax.py) ----
    _names = pd.read_csv(NAMES_FILE)
    _canon = dict(zip(_names["player_id"].astype(int), _names["player_name"].astype(str)))
    _mp_name = {}
    canon_name = {gid: _canon.get(gid, _mp_name.get(gid)) for gid in per_game["goalie_id"].unique()}
    per_game["goalie_name_canon"] = per_game["goalie_id"].map(canon_name)

    # ---- PER-SEASON GOALIE TOTALS (for tiering) ----
    season_totals = (
        per_game.groupby(["goalie_id", "season"])
        .agg(GP=("game_id", "size"), shots_faced_season=("shots_faced", "sum"),
             saves_season=("saves", "sum"))
        .reset_index()
    )
    season_totals["goalie_name"] = season_totals["goalie_id"].map(canon_name)
    season_totals["season_save_pct"] = season_totals["saves_season"] / season_totals["shots_faced_season"]

    # ---- TIER ASSIGNMENT + BASELINES, PER SEASON ----
    tier_rows = []
    baseline_rows = []
    season_totals["tier"] = "depth"
    for season_int, grp in season_totals.groupby("season"):
        ranked = grp.sort_values(
            ["GP", "shots_faced_season", "goalie_id"], ascending=[False, False, True]
        ).reset_index(drop=True)
        ranked["gp_rank"] = ranked.index + 1

        starter_mask = ranked["gp_rank"] <= STARTER_TIER_SIZE
        backup_mask = (ranked["gp_rank"] > STARTER_TIER_SIZE) & (ranked["gp_rank"] <= STARTER_TIER_SIZE + BACKUP_TIER_SIZE)
        ranked.loc[starter_mask, "tier"] = "starter"
        ranked.loc[backup_mask, "tier"] = "backup"

        starter_shots = ranked.loc[starter_mask, "shots_faced_season"].sum()
        starter_saves = ranked.loc[starter_mask, "saves_season"].sum()
        backup_shots = ranked.loc[backup_mask, "shots_faced_season"].sum()
        backup_saves = ranked.loc[backup_mask, "saves_season"].sum()

        baseline_starter = starter_saves / starter_shots
        baseline_backup = backup_saves / backup_shots

        print(f"\nSeason {season_int}: {len(ranked)} goalies with >=1 qualifying game")
        print(f"  starter tier: n={int(starter_mask.sum())}, baseline save% = {baseline_starter*100:.3f}%")
        print(f"  backup  tier: n={int(backup_mask.sum())}, baseline save% = {baseline_backup*100:.3f}%")

        ranked["baseline_starter_savepct"] = baseline_starter
        ranked["baseline_backup_savepct"] = baseline_backup
        tier_rows.append(ranked)

        baseline_rows.append({"season": season_int, "tier": "starter", "n_goalies": int(starter_mask.sum()),
                               "total_shots_faced": int(starter_shots), "total_saves": int(starter_saves),
                               "baseline_save_pct": baseline_starter * 100})
        baseline_rows.append({"season": season_int, "tier": "backup", "n_goalies": int(backup_mask.sum()),
                               "total_shots_faced": int(backup_shots), "total_saves": int(backup_saves),
                               "baseline_save_pct": baseline_backup * 100})

    season_totals = pd.concat(tier_rows, ignore_index=True)
    baselines = pd.DataFrame(baseline_rows).sort_values(["season", "tier"]).reset_index(drop=True)

    # ---- APPLY BASELINES TO PER-GAME ROWS ----
    baseline_map_starter = season_totals.drop_duplicates("season").set_index("season")["baseline_starter_savepct"].to_dict()
    baseline_map_backup = season_totals.drop_duplicates("season").set_index("season")["baseline_backup_savepct"].to_dict()
    per_game["baseline_starter"] = per_game["season"].map(baseline_map_starter)
    per_game["baseline_backup"] = per_game["season"].map(baseline_map_backup)

    per_game["is_QGs"] = (per_game["game_save_pct"] >= per_game["baseline_starter"]).astype(int)
    per_game["is_QGb"] = (per_game["game_save_pct"] >= per_game["baseline_backup"]).astype(int)

    tie_rate_s = (per_game["game_save_pct"] == per_game["baseline_starter"]).mean() * 100
    tie_rate_b = (per_game["game_save_pct"] == per_game["baseline_backup"]).mean() * 100
    print(f"\nExact-tie rate vs starter baseline: {tie_rate_s:.2f}% of qualifying games")
    print(f"Exact-tie rate vs backup baseline:  {tie_rate_b:.2f}% of qualifying games")
    print("(>= rule, no half-credit — see script docstring)")

    # ---- PER-SEASON AGGREGATION ----
    per_season = (
        per_game.groupby(["goalie_id", "season"])
        .agg(GP=("game_id", "size"), QGs_games=("is_QGs", "sum"), QGb_games=("is_QGb", "sum"))
        .reset_index()
    )
    per_season = per_season.merge(
        season_totals[["goalie_id", "season", "goalie_name", "season_save_pct", "tier",
                        "baseline_starter_savepct", "baseline_backup_savepct"]],
        on=["goalie_id", "season"], how="left",
    )
    per_season["QG_pct_s"] = per_season["QGs_games"] / per_season["GP"] * 100
    per_season["QG_pct_b"] = per_season["QGb_games"] / per_season["GP"] * 100
    per_season["QG_pct_s_lo"] = per_season.apply(lambda r: wilson_lower(r["QGs_games"], r["GP"]) * 100, axis=1)
    per_season["QG_pct_b_lo"] = per_season.apply(lambda r: wilson_lower(r["QGb_games"], r["GP"]) * 100, axis=1)
    per_season["qualified"] = per_season["GP"] >= MIN_GP_PER_SEASON

    n_q = int(per_season["qualified"].sum())
    print(f"\nPer-season: {len(per_season)} rows emitted, {n_q} qualified (GP >= {MIN_GP_PER_SEASON})")

    # ---- POOLED 4-SEASON ----
    qualified_goalies = set(per_season.loc[per_season["qualified"], "goalie_id"])

    pooled = (
        per_game.groupby(["goalie_id"])
        .agg(GP=("game_id", "size"), QGs_games=("is_QGs", "sum"), QGb_games=("is_QGb", "sum"))
        .reset_index()
    )
    pooled["goalie_name"] = pooled["goalie_id"].map(canon_name)
    pooled["QG_pct_s"] = pooled["QGs_games"] / pooled["GP"] * 100
    pooled["QG_pct_b"] = pooled["QGb_games"] / pooled["GP"] * 100
    pooled["QG_pct_s_lo"] = pooled.apply(lambda r: wilson_lower(r["QGs_games"], r["GP"]) * 100, axis=1)
    pooled["QG_pct_b_lo"] = pooled.apply(lambda r: wilson_lower(r["QGb_games"], r["GP"]) * 100, axis=1)
    pooled["qualified"] = pooled["goalie_id"].isin(qualified_goalies)

    # Reference-only "primary_tier": the tier of the season in which this goalie logged the most GP.
    _primary = (
        per_season.sort_values("GP", ascending=False)
        .drop_duplicates("goalie_id")[["goalie_id", "tier"]]
        .rename(columns={"tier": "primary_tier"})
    )
    pooled = pooled.merge(_primary, on="goalie_id", how="left")

    pooled = pooled.sort_values(["qualified", "QG_pct_s_lo"], ascending=False).reset_index(drop=True)
    pooled["rank"] = pooled.index + 1

    print(f"\nPooled 4-season: {len(pooled)} goalies, {int(pooled['qualified'].sum())} qualified")

    # ---- WRITE OUTPUTS ----
    out_baselines = OUT_DIR / f"qg_tier_baselines_by_season{suffix}.csv"
    out_per_season = OUT_DIR / f"qg_savepct_per_season_2022-2026{suffix}.csv"
    out_pooled = OUT_DIR / f"qg_savepct_2022-2026{suffix}.csv"

    baselines.to_csv(out_baselines, index=False)
    per_season.to_csv(out_per_season, index=False)
    pooled.to_csv(out_pooled, index=False)
    print(f"\nWritten:")
    print(f"  {out_baselines}")
    print(f"  {out_per_season}")
    print(f"  {out_pooled}")

    # ---- VALIDATION ----
    print("\n=== SEASON BASELINES ===")
    print(baselines.to_string(index=False))

    print(f"\n=== TOP 20 POOLED QG%s (starter baseline, 4-season, scope={scope}) ===")
    print(pooled.head(20)[["rank", "goalie_name", "primary_tier", "GP", "QG_pct_s", "QG_pct_s_lo", "QG_pct_b"]].to_string(index=False))

    print("\n=== SIX VALIDATION ===")
    for name, gid in SIX_IDS.items():
        row = pooled[pooled["goalie_id"] == gid]
        if len(row):
            r = row.iloc[0]
            print(f"  {name}: rank {int(r['rank'])}, tier {r['primary_tier']}, GP {int(r['GP'])}, "
                  f"QG%s {r['QG_pct_s']:.2f} (Wilson {r['QG_pct_s_lo']:.2f}), QG%b {r['QG_pct_b']:.2f}")
        else:
            print(f"  {name}: NOT FOUND by ID {gid}")


if __name__ == "__main__":
    for scope, suffix in SCOPES.items():
        run_scope(scope, suffix)
    print("\nDONE — both scopes written.")
