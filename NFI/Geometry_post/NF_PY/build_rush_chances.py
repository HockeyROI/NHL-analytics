"""
build_rush_chances.py

Computes rush chances against per 60 minutes on ice for each skater (non-goalie),
per season and career.

Rush shot definition
--------------------
A shot (shot-on-goal, goal, or missed-shot) is classified as a rush chance if:
  • x_coord_norm is in [25, 70]  — taken from the neutral zone or early offensive
    zone, before the defence is set up.
  • OR is_rush == True if that column exists in the dataset.

On-ice attribution
------------------
For each rush shot against a defending team, every non-goalie skater whose shift
spans the shot time (abs_start_secs ≤ shot_abs_time ≤ abs_end_secs) on that team
receives one rush-against credit.

TOI
---
Total shift time per (player, season) — all game situations — is used as the
denominator.  Proper even-strength TOI would require full play-by-play strength
tracking, which is not available in the current dataset.  The 1000-minute
qualifying threshold is therefore applied to all-situations TOI.

Output
------
Data/forward_rush_against.csv
  player_id, player_name, seasons (JSON array), total_toi_mins,
  rush_against_per60_career, rush_against_per60_by_season (JSON dict),
  team_2526
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "Data")
SHOTS_F    = os.path.join(DATA_DIR, "nhl_shot_events.csv")
SHIFTS_F   = os.path.join(DATA_DIR, "shift_data.csv")
GAME_IDS_F = os.path.join(DATA_DIR, "game_ids.csv")
OUTPUT_F   = os.path.join(DATA_DIR, "forward_rush_against.csv")

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_TOI_SECS    = 1_000 * 60       # 1 000 minutes qualifying threshold
RUSH_TYPES      = {"shot-on-goal", "goal", "missed-shot"}
X_NORM_MIN, X_NORM_MAX = 25, 70    # neutral zone → early offensive zone
SEASON_2526 = 20252526


# ── Guards ────────────────────────────────────────────────────────────────────
for path in (SHOTS_F, SHIFTS_F, GAME_IDS_F):
    if not os.path.exists(path):
        sys.exit(f"ERROR: required file not found: {path}")


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading shot events …")
shots = pd.read_csv(
    SHOTS_F,
    dtype={"situation_code": str, "zone_code": str,
           "shooting_team_abbrev": str, "home_team_abbrev": str,
           "away_team_abbrev": str},
)

print("Loading shift data …")
shifts = pd.read_csv(SHIFTS_F, dtype={"team_abbrev": str})

print("Loading game IDs …")
game_meta = (
    pd.read_csv(GAME_IDS_F)[["game_id", "season"]]
    .drop_duplicates("game_id")
)


# ── 2. Pre-process shots ──────────────────────────────────────────────────────
shots["abs_time"] = (shots["period"] - 1) * 1200 + shots["time_secs"]


# ── 3. Identify rush shots ────────────────────────────────────────────────────
# Rush = shot from neutral zone / early offensive zone before defence sets up.
# Coordinate threshold only — the previous-event zone-transition approach was
# too restrictive and returned ~755 shots across 678 games.
rush_mask = (
    shots["event_type"].isin(RUSH_TYPES)
    & shots["x_coord_norm"].between(X_NORM_MIN, X_NORM_MAX)
)

# Honour an explicit is_rush flag if the dataset carries one
if "is_rush" in shots.columns:
    rush_mask = rush_mask | (shots["is_rush"] == True)

rush_shots = shots.loc[rush_mask, [
    "game_id", "season", "period",
    "abs_time", "shooting_team_abbrev",
    "home_team_abbrev", "away_team_abbrev",
]].copy()

print(f"Rush shots identified: {len(rush_shots):,}")

# Defending team is the non-shooting team
rush_shots["defending_team"] = np.where(
    rush_shots["shooting_team_abbrev"] == rush_shots["home_team_abbrev"],
    rush_shots["away_team_abbrev"],
    rush_shots["home_team_abbrev"],
)


# ── 4. Pre-process shifts ─────────────────────────────────────────────────────
# type_code 517 = shift record (ALL players, goalies included).
# type_code 505 = goal-event rows embedded in the shift chart (zero duration,
#                 start_secs == end_secs) — exclude them entirely.
# Goalies cannot be identified by type_code; use goalie_id from shot events.
goalie_ids = set(
    shots["goalie_id"].dropna().astype(int).unique()
)

skaters = shifts[
    (shifts["type_code"] == 517) &
    (~shifts["player_id"].isin(goalie_ids))
].copy()

print(f"Skater shift rows after goalie exclusion: {len(skaters):,}")

# Attach season (needed for TOI breakdown and rush-against attribution)
skaters = skaters.merge(game_meta, on="game_id", how="left")

# Shift duration in seconds (within-period seconds cancel the period offset)
skaters["shift_dur"] = skaters["end_secs"] - skaters["start_secs"]


# ── 5. TOI per (player_id, season) ───────────────────────────────────────────
toi_df = (
    skaters
    .groupby(["player_id", "season"], as_index=False)["shift_dur"]
    .sum()
    .rename(columns={"shift_dur": "toi_secs"})
)


# ── 6. Rush-against interval join (game-by-game, vectorised) ─────────────────
print("Computing rush chances against …")

rush_against: dict[tuple, int] = defaultdict(int)   # (player_id, season) → count

# Pre-index both DataFrames by game_id for O(1) lookup
skater_by_game = {gid: grp for gid, grp in skaters.groupby("game_id")}
rush_by_game   = {gid: grp for gid, grp in rush_shots.groupby("game_id")}

n_games       = len(rush_by_game)
total_pairs   = 0          # diagnostic: total (player × shot) attributions
sample_matches = []        # diagnostic: up to 5 sample on-ice hits

for i, (game_id, game_rushes) in enumerate(rush_by_game.items(), 1):
    if i % 500 == 0 or i == n_games:
        print(f"  {i:,}/{n_games:,} games processed")

    if game_id not in skater_by_game:
        continue

    gs = skater_by_game[game_id]

    # Convert to numpy for vectorised comparisons
    s_starts  = gs["abs_start_secs"].to_numpy()
    s_ends    = gs["abs_end_secs"].to_numpy()
    s_teams   = gs["team_abbrev"].to_numpy()
    s_pids    = gs["player_id"].to_numpy()

    # Season is constant within a game — grab once
    game_season = gs["season"].iat[0]

    shot_times     = game_rushes["abs_time"].to_numpy()
    shot_def_teams = game_rushes["defending_team"].to_numpy()

    for t, def_team in zip(shot_times, shot_def_teams):
        mask = (s_teams == def_team) & (s_starts <= t) & (s_ends >= t)
        matched_pids    = s_pids[mask]
        matched_starts  = s_starts[mask]
        matched_ends    = s_ends[mask]

        for pid in matched_pids:
            rush_against[(pid, game_season)] += 1
        total_pairs += mask.sum()

        # Collect up to 5 sample matches for the diagnostic
        if len(sample_matches) < 5 and mask.sum() > 0:
            idx = np.where(mask)[0][0]
            sample_matches.append({
                "player_id":     int(s_pids[idx]),
                "game_id":       int(game_id),
                "abs_start_secs": int(s_starts[idx]),
                "abs_end_secs":   int(s_ends[idx]),
                "shot_abs_time":  int(t),
            })

# ── Diagnostic ────────────────────────────────────────────────────────────────
print(f"\n── Interval join diagnostic ──────────────────────────────────────────")
print(f"  (player × rush shot) on-ice attributions : {total_pairs:,}")
print(f"  Unique (player, season) pairs with credit : {len(rush_against):,}")
print(f"  Sample matches (player_id | game_id | shift_start | shift_end | shot_time):")
for m in sample_matches:
    print(f"    pid={m['player_id']}  game={m['game_id']}  "
          f"shift=[{m['abs_start_secs']}, {m['abs_end_secs']}]  "
          f"shot={m['shot_abs_time']}")
print()


# ── 7. Assemble per-season DataFrame ─────────────────────────────────────────
ra_df = pd.DataFrame(
    [(pid, season, cnt) for (pid, season), cnt in rush_against.items()],
    columns=["player_id", "season", "rush_against"],
)

# Left join so players with TOI but zero rush against are included
full_df = toi_df.merge(ra_df, on=["player_id", "season"], how="left")
full_df["rush_against"] = full_df["rush_against"].fillna(0).astype(int)

full_df["rush_per60_season"] = (
    full_df["rush_against"] / full_df["toi_secs"].clip(lower=1)
) * 3600


# ── 8. Career aggregation ─────────────────────────────────────────────────────
career = (
    full_df
    .groupby("player_id", as_index=False)
    .agg(
        total_toi_secs     =("toi_secs",    "sum"),
        total_rush_against =("rush_against","sum"),
        seasons            =("season",      lambda x: sorted(x.unique().tolist())),
    )
)
career["rush_against_per60_career"] = (
    career["total_rush_against"] / career["total_toi_secs"].clip(lower=1)
) * 3600
career["total_toi_mins"] = career["total_toi_secs"] / 60

# Apply minimum TOI qualifier
career = career[career["total_toi_secs"] >= MIN_TOI_SECS].copy()
print(f"Players qualifying (≥1 000 min TOI): {len(career):,}")


# ── 9. Player metadata ────────────────────────────────────────────────────────
# Name: from the most recent shift record available
player_names = (
    skaters
    .sort_values("game_id", ascending=False)
    .drop_duplicates("player_id")
    [["player_id", "first_name", "last_name"]]
)
player_names["player_name"] = (
    player_names["first_name"].fillna("").str.strip()
    + " "
    + player_names["last_name"].fillna("").str.strip()
).str.strip()

# 2025-26 team: last team played for in season 20252526
team_2526 = (
    skaters[skaters["season"] == SEASON_2526]
    .sort_values("game_id", ascending=False)
    .drop_duplicates("player_id")
    [["player_id", "team_abbrev"]]
    .rename(columns={"team_abbrev": "team_2526"})
)


# ── 10. Build final output ────────────────────────────────────────────────────
out = (
    career
    .merge(player_names[["player_id", "player_name"]], on="player_id", how="left")
    .merge(team_2526,                                  on="player_id", how="left")
)

out["seasons"] = out["seasons"].apply(json.dumps)
out["rush_against_per60_by_season"] = ""   # initialise column


# ── 11. Per-season JSON dict ──────────────────────────────────────────────────
# Build the by-season JSON per player using a simple groupby loop —
# avoids the unreliable groupby().apply().rename() chain that produced
# column-naming failures across pandas versions.
full_sub = full_df[full_df["player_id"].isin(career["player_id"])].copy()
full_sub["season_str"] = full_sub["season"].astype(str)

for pid, season_rows in full_sub.groupby("player_id"):
    by_season = (
        season_rows.groupby("season_str")["rush_per60_season"]
        .first()
        .round(4)
        .to_dict()
    )
    out.loc[out["player_id"] == pid, "rush_against_per60_by_season"] = json.dumps(
        dict(sorted(by_season.items()))
    )


# ── 12. Final column select and save ─────────────────────────────────────────
out = out[[
    "player_id",
    "player_name",
    "seasons",
    "total_toi_mins",
    "rush_against_per60_career",
    "rush_against_per60_by_season",
    "team_2526",
]].sort_values("rush_against_per60_career", ascending=False)

out.to_csv(OUTPUT_F, index=False, float_format="%.4f")
print(f"\nSaved {len(out):,} players → {OUTPUT_F}")
