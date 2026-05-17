#!/usr/bin/env python3
"""
NFI-Score Step 1 — Score-state lookup table (regenerated from 6-season source).

Reads NFI's master shot events file (Data/nhl_shot_events.csv, the same source
the existing two-way pipeline uses). Tags every event with the score state at
the moment of the event from the shooter's team perspective:

    Down2 (diff <= -2), Down1 (-1), Tied (0), Up1 (+1), Up2 (>= +2)

Step 0 diagnostic is also performed in-line: documents why the 5-season
shots_tagged.csv exists by extracting the SEASONS literal from the script that
builds it. Findings are appended to Output/window_gap_report.csv.

Output columns: event_id, game_id, season, shoot_diff, score_state.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/ashgarg/Documents/HockeyROI")
SHOT_CSV = ROOT / "Data" / "nhl_shot_events.csv"
SHOTS_TAGGED_BUILDER = ROOT / "NFI" / "scripts" / "03_onice_attribution_pillars.py"

OUT_DIR = ROOT / "NFI" / "NFI_Score_Adj" / "2026" / "05" / "Output"
OUT_CSV = OUT_DIR / "nfi_score_state_lookup.csv"
WINDOW_REPORT = OUT_DIR / "window_gap_report.csv"

EXPECTED_SEASONS = ["20202021", "20212022", "20222023", "20232024",
                    "20242025", "20252026"]
STATE_ORDER = ["Down2", "Down1", "Tied", "Up1", "Up2"]
TARGETS_PCT = {"Tied": 35.0, "Down1": 18.5, "Up1": 17.5,
               "Down2": 14.5, "Up2": 14.5}
TARGET_TOL_PCT = 3.0


def diff_to_state(d: int) -> str:
    if d <= -2:
        return "Down2"
    if d == -1:
        return "Down1"
    if d == 0:
        return "Tied"
    if d == 1:
        return "Up1"
    return "Up2"


def stop(msg: str, code: int = 2) -> int:
    print(f"[step1] STOP — {msg}", file=sys.stderr)
    return code


def step0_window_diagnostic() -> dict:
    """Read the shots_tagged builder and extract its SEASONS filter."""
    print(f"[step0] reading {SHOTS_TAGGED_BUILDER} (read-only)")
    if not SHOTS_TAGGED_BUILDER.exists():
        print(f"[step0]   builder not found — skipping diagnostic")
        return {"builder_found": False}
    src = SHOTS_TAGGED_BUILDER.read_text()
    m = re.search(r"SEASONS\s*=\s*\{([^}]+)\}", src)
    seasons_in_builder: list[str] = []
    if m:
        seasons_in_builder = re.findall(r'"(\d{8})"', m.group(1))
        seasons_in_builder.sort()
    print(f"[step0]   shots_tagged builder season filter: {seasons_in_builder or '(none found)'}")
    print(f"[step0]   nhl_shot_events.csv covers:         {EXPECTED_SEASONS}")
    return {
        "builder_found": True,
        "shots_tagged_seasons": ",".join(seasons_in_builder),
        "source_seasons": ",".join(EXPECTED_SEASONS),
        "missing_in_shots_tagged": ",".join(
            s for s in EXPECTED_SEASONS if s not in set(seasons_in_builder)
        ),
    }


def main() -> int:
    if not SHOT_CSV.exists():
        return stop(f"input not found: {SHOT_CSV}")

    diag = step0_window_diagnostic()

    print(f"[step1] reading {SHOT_CSV}")
    use_cols = ["game_id", "season", "game_type", "period", "time_secs",
                "event_id", "event_type", "is_goal",
                "shooting_team_id", "home_team_id"]
    shots = pd.read_csv(SHOT_CSV, usecols=use_cols, dtype={"season": str})
    print(f"[step1]   loaded {len(shots):,} events across "
          f"{shots['game_id'].nunique():,} games")

    seasons_present = sorted(shots["season"].unique())
    print(f"[step1]   seasons present: {seasons_present}")
    if set(seasons_present) != set(EXPECTED_SEASONS):
        return stop(f"expected seasons {EXPECTED_SEASONS}, got {seasons_present}")

    # abs_time and ordering
    shots["abs_time"] = shots["time_secs"].astype(int) + \
                        (shots["period"].astype(int) - 1) * 1200
    shots = shots.sort_values(["game_id", "abs_time", "event_id"]).reset_index(drop=True)

    print("[step1] re-deriving running score and tagging events...")
    n = len(shots)
    shoot_diff = np.empty(n, dtype=np.int64)

    pos = 0
    for gid, g in shots.groupby("game_id", sort=False):
        team_score: dict = {}
        sh = g["shooting_team_id"].to_numpy()
        ig = g["is_goal"].to_numpy()
        sz = len(g)
        for i in range(sz):
            shooter = int(sh[i])
            sh_score = team_score.get(shooter, 0)
            opp_score = 0
            for t, s in team_score.items():
                if t != shooter:
                    opp_score += s
            shoot_diff[pos + i] = sh_score - opp_score
            if int(ig[i]) == 1:
                team_score[shooter] = sh_score + 1
        pos += sz
    shots["shoot_diff"] = shoot_diff
    shots["score_state"] = pd.Categorical(
        [diff_to_state(int(d)) for d in shoot_diff],
        categories=STATE_ORDER, ordered=True,
    )
    print(f"[step1]   tagged {n:,} events")

    # ---------- write lookup ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = shots[["event_id", "game_id", "season", "shoot_diff", "score_state"]].copy()
    out.to_csv(OUT_CSV, index=False)
    print(f"[step1] wrote {OUT_CSV}  ({len(out):,} rows)")

    # ---------- sanity: per-season counts ----------
    print("\n[step1] events tagged by season:")
    by_season = out.groupby("season").size()
    for s in EXPECTED_SEASONS:
        print(f"   {s}: {int(by_season.get(s, 0)):>10,}")

    # ---------- sanity: state distribution (regulation only, mirrors downstream filter) ----------
    reg = shots[shots["period"].between(1, 3)]
    print(f"\n[step1] state distribution (regulation only, n={len(reg):,}):")
    pct = (reg["score_state"].value_counts(normalize=True) * 100).reindex(STATE_ORDER).fillna(0)
    cnt = reg["score_state"].value_counts().reindex(STATE_ORDER).fillna(0).astype(int)
    print(f"  {'state':<6} {'count':>10} {'pct':>7}  target±{TARGET_TOL_PCT:.0f}pp")
    state_failures: list[str] = []
    for s in STATE_ORDER:
        delta = abs(pct[s] - TARGETS_PCT[s])
        flag = "" if delta <= TARGET_TOL_PCT else "  <-- FAIL"
        print(f"  {s:<6} {cnt[s]:>10,} {pct[s]:>6.2f}%   {TARGETS_PCT[s]:>5.1f}%{flag}")
        if delta > TARGET_TOL_PCT:
            state_failures.append(s)

    # ---------- spot-check 5 random regular-season games' regulation finals ----------
    print("\n[step1] spot-check 5 random games (regulation final score, derived from is_goal):")
    rs_gids = shots.loc[
        (shots["game_type"] == "regular") & shots["period"].between(1, 3),
        "game_id"
    ].drop_duplicates()
    sample = rs_gids.sample(5, random_state=42)
    for gid in sample:
        g = shots[(shots["game_id"] == gid) & shots["period"].between(1, 3)]
        home_id = int(g["home_team_id"].iloc[0])
        home_goals = int(((g["is_goal"] == 1) & (g["shooting_team_id"] == home_id)).sum())
        away_goals = int(((g["is_goal"] == 1) & (g["shooting_team_id"] != home_id)).sum())
        print(f"   game_id={gid}  home {home_goals}  -  away {away_goals}")

    # ---------- write window gap report ----------
    rep_rows = [{
        "key": "source_file",
        "value": str(SHOT_CSV),
    }, {
        "key": "source_seasons",
        "value": ",".join(EXPECTED_SEASONS),
    }, {
        "key": "shots_tagged_seasons",
        "value": diag.get("shots_tagged_seasons", ""),
    }, {
        "key": "shots_tagged_missing_seasons",
        "value": diag.get("missing_in_shots_tagged", ""),
    }, {
        "key": "shots_tagged_builder",
        "value": str(SHOTS_TAGGED_BUILDER),
    }, {
        "key": "step1_total_events",
        "value": str(len(out)),
    }]
    pd.DataFrame(rep_rows).to_csv(WINDOW_REPORT, index=False)
    print(f"\n[step1] wrote diagnostic report {WINDOW_REPORT}")

    if state_failures:
        return stop(f"state distribution outside ±{TARGET_TOL_PCT}pp for: {state_failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
