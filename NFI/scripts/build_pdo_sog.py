#!/usr/bin/env python3
"""PDO (SOG-based) — descriptive diagnostic, NOT a canonical NFI metric.

Reuses the exact validated on-ice attribution logic from
03_onice_attribution_pillars.py (same shot-loading/filtering, same
state_from_code 5v5 derivation, same per-game shift<->shot join, same
on-ice-player-list determination) — copied rather than imported, because
that script executes top-to-bottom on import (it would rewrite the
canonical pillar_*.csv / player_counts_by_state_zone_per_season.csv files
as a side effect, which we must not touch here).

The ONLY new counter tracked here is on-ice SOG-for/against
(event_type in {'shot-on-goal','goal'}) per (player_id, season) at
state=='ES' (strict 5v5) — a bucket that doesn't already exist (the
canonical file only has Corsi 'att' and Fenwick 'fen' on-ice buckets).
On-ice goals-for/against and 5v5 TOI are NOT recomputed here — they
already exist correctly in player_counts_by_state_zone_per_season.csv
(onice_for_gl / onice_ag_gl / toi_min for state=='ES') and are reused
as-is.

PDO = (goals_for/SOG_for + 1 - goals_against/SOG_against) x 100
Floor: >=200 min state=='ES' TOI.

Output: NFI/output/player_pdo_5v5_per_season.csv (new file; does not
touch any canonical NFI output).
"""
import os
from collections import defaultdict
import pandas as pd
import numpy as np

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
SHIFT_CSV = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
POS_CSV = f"{ROOT}/NFI/output/player_positions.csv"
GAME_CSV = f"{ROOT}/Data/game_ids.csv"
COUNTS_CSV = f"{ROOT}/NFI/output/player_counts_by_state_zone_per_season.csv"
OUT_CSV = f"{ROOT}/NFI/output/player_pdo_5v5_per_season.csv"
OUT_CSV_ALLSIT = f"{ROOT}/NFI/output/player_pdo_allsit_per_season.csv"

SEASONS = {"20212022", "20222023", "20232024", "20242025", "20252026"}
TOI_FLOOR_MIN = 200.0


def state_from_code(sc, shoot_home):
    """Verbatim copy of 03_onice_attribution_pillars.state_from_code."""
    if pd.isna(sc):
        return None
    s = str(int(sc)).zfill(4)
    if len(s) != 4:
        return None
    ag, ask, hsk, hg = int(s[0]), int(s[1]), int(s[2]), int(s[3])
    if ag == 0 or hg == 0:
        return None
    sh, op = (hsk, ask) if shoot_home else (ask, hsk)
    if sh == op:
        if sh == 5: return "ES"
        if sh == 4: return "4v4"
        if sh == 3: return "3v3"
        return "ES_other"
    if sh > op: return "PP"
    return "PK"


print("Loading positions...")
pos_df = pd.read_csv(POS_CSV, dtype={"player_id": int})
pos_map = dict(zip(pos_df["player_id"], pos_df["pos_group"]))

print("Loading game -> season map...")
g_df = pd.read_csv(GAME_CSV, dtype={"game_id": int, "season": str})
g_df = g_df[g_df["game_type"] == "regular"]
game_season = dict(zip(g_df["game_id"], g_df["season"]))

print("Loading shots...")
cols = ["game_id", "season", "period", "event_id", "event_type", "situation_code", "time_secs",
        "home_team_id", "shooting_team_id", "home_team_abbrev", "away_team_abbrev", "shooting_team_abbrev",
        "shooter_player_id", "goalie_id", "is_goal"]
shots = pd.read_csv(SHOT_CSV, usecols=cols, dtype={"season": str, "situation_code": str})
shots["season"] = shots["season"].astype(str)
shots = shots[shots["season"].isin(SEASONS) & shots["period"].between(1, 3)].copy()
shots = shots[shots["event_type"].isin(["shot-on-goal", "missed-shot", "blocked-shot", "goal"])].copy()

shots["abs_time"] = shots["time_secs"].astype(int) + (shots["period"].astype(int) - 1) * 1200
shots["_shoot_home"] = shots["shooting_team_id"] == shots["home_team_id"]

sc_str = shots["situation_code"].astype(str).str.zfill(4)
ag = sc_str.str[0].astype(int); ask = sc_str.str[1].astype(int)
hsk = sc_str.str[2].astype(int); hg = sc_str.str[3].astype(int)
empty_net = (ag == 0) | (hg == 0)
sh = np.where(shots["_shoot_home"], hsk, ask)
op = np.where(shots["_shoot_home"], ask, hsk)
state = np.where((sh == op) & (sh == 5), "ES",
         np.where((sh == op) & (sh == 4), "4v4",
         np.where((sh == op) & (sh == 3), "3v3",
         np.where(sh > op, "PP", "PK"))))
shots["state"] = state
shots = shots[~empty_net].copy()

# NEW: is_sog flag (the one thing this script adds beyond the original)
shots["is_sog"] = shots["event_type"].isin(["shot-on-goal", "goal"])
shots["is_goal_i"] = shots["is_goal"].astype(int)

print(f"  tagged shots: {len(shots):,}")
shots_by_game = dict(tuple(shots.groupby("game_id")))
valid_gids = set(shots_by_game.keys())
print(f"  games with shots: {len(valid_gids)}")

print("Loading shifts (streaming filter)...")
shift_cols = ["game_id", "player_id", "period", "team_abbrev", "abs_start_secs", "abs_end_secs"]
shift_iter = pd.read_csv(SHIFT_CSV, usecols=shift_cols, chunksize=500000)
shift_list = []
for ch in shift_iter:
    ch = ch.dropna(subset=["game_id", "player_id", "period", "abs_start_secs", "abs_end_secs"])
    ch["game_id"] = ch["game_id"].astype(int)
    ch["player_id"] = ch["player_id"].astype(int)
    ch["period"] = ch["period"].astype(int)
    ch["abs_start_secs"] = ch["abs_start_secs"].astype(int)
    ch["abs_end_secs"] = ch["abs_end_secs"].astype(int)
    ch = ch[ch["game_id"].isin(valid_gids) & ch["period"].between(1, 3)]
    if len(ch):
        shift_list.append(ch)
shifts = pd.concat(shift_list, ignore_index=True)
del shift_list
print(f"  shifts loaded: {len(shifts):,}")
shifts_by_game = dict(tuple(shifts.groupby("game_id")))

team_abbrevs = shots.groupby("game_id").agg(home_abbrev=("home_team_abbrev", "first"),
                                             away_abbrev=("away_team_abbrev", "first")).to_dict(orient="index")

# on-ice SOG counters — the only new state this script accumulates
plr_onice_for_sog_s = defaultdict(lambda: defaultdict(int))  # (pid, season) -> {state: count}
plr_onice_ag_sog_s = defaultdict(lambda: defaultdict(int))

print("Per-game shift-shot join (SOG on-ice only)...")
n_games = 0
for gid, gshots in shots_by_game.items():
    n_games += 1
    if n_games % 500 == 0:
        print(f"  processed {n_games} games")
    if gid not in shifts_by_game:
        continue
    gshifts = shifts_by_game[gid]
    season_str = game_season.get(gid, str(gid)[:4])
    if season_str not in SEASONS:
        continue
    home_ab = team_abbrevs[gid]["home_abbrev"]
    away_ab = team_abbrevs[gid]["away_abbrev"]

    shifts_by_team = {}
    for team_ab, tsh in gshifts.groupby("team_abbrev"):
        starts = tsh["abs_start_secs"].values.astype(int)
        ends = tsh["abs_end_secs"].values.astype(int)
        pids = tsh["player_id"].values.astype(int)
        shifts_by_team[team_ab] = (starts, ends, pids)

    for _, s in gshots.iterrows():
        if not s["is_sog"]:
            continue
        t = int(s["abs_time"])
        st = s["state"]
        shoot_ab = s["shooting_team_abbrev"]
        def_ab = away_ab if shoot_ab == home_ab else home_ab

        # Inclusive end boundary — see 03_onice_attribution_pillars.py for the
        # full explanation (shift-end timestamps land in the same second as
        # the stoppage that ends them, and a goal is exactly that stoppage).
        if shoot_ab in shifts_by_team:
            st_s, en_s, pids_s = shifts_by_team[shoot_ab]
            mask = (st_s <= t) & (t <= en_s)
            onice_shoot = pids_s[mask]
        else:
            onice_shoot = np.array([], dtype=int)
        if def_ab in shifts_by_team:
            st_d, en_d, pids_d = shifts_by_team[def_ab]
            mask = (st_d <= t) & (t <= en_d)
            onice_def = pids_d[mask]
        else:
            onice_def = np.array([], dtype=int)

        onice_shoot = [int(p) for p in onice_shoot if pos_map.get(int(p)) != "G"]
        onice_def = [int(p) for p in onice_def if pos_map.get(int(p)) != "G"]

        for p in onice_shoot:
            plr_onice_for_sog_s[(p, season_str)][st] += 1
        for p in onice_def:
            plr_onice_ag_sog_s[(p, season_str)][st] += 1

print(f"Processed {n_games} games.")

# ----------------- Reuse existing on-ice goals + TOI -----------------
# Two scopes computed together in this one pass (the shift-join above already
# tracked SOG on-ice by state, so no need to re-scan): 5v5 (state=='ES') and
# all-situations (every state summed). Same pattern as the app's existing
# sQS%/QG%b 5v5-vs-all-situations toggle.
print("Loading existing on-ice goals + TOI from player_counts_by_state_zone_per_season.csv...")
counts = pd.read_csv(COUNTS_CSV, dtype={"season": str})

es = counts[counts["state"] == "ES"].copy()
gl_es = es.groupby(["player_id", "season"]).agg(
    onice_for_gl=("onice_for_gl", "sum"),
    onice_ag_gl=("onice_ag_gl", "sum"),
    toi_min=("toi_min", "first"),   # constant across zone rows per (player,season,state)
).reset_index()
print(f"  ES player-season rows: {len(gl_es)}")

# All situations: toi_min is constant across zone rows WITHIN one state, so
# take it once per (player,season,state) before summing across states —
# summing the raw column directly would double-count each state's zone rows.
toi_by_state = counts.groupby(["player_id", "season", "state"])["toi_min"].first().reset_index()
toi_all = toi_by_state.groupby(["player_id", "season"])["toi_min"].sum().reset_index()
gl_all = counts.groupby(["player_id", "season"]).agg(
    onice_for_gl=("onice_for_gl", "sum"),
    onice_ag_gl=("onice_ag_gl", "sum"),
).reset_index().merge(toi_all, on=["player_id", "season"])
print(f"  all-situations player-season rows: {len(gl_all)}")

names = pd.read_csv(POS_CSV, dtype={"player_id": int})[["player_id", "player_name"]].drop_duplicates("player_id")


def build_rows(gl_df: pd.DataFrame, sog_scope: str, toi_label: str) -> pd.DataFrame:
    rows = []
    for _, r in gl_df.iterrows():
        pid, season = int(r["player_id"]), r["season"]
        if sog_scope == "ES":
            sog_for = plr_onice_for_sog_s.get((pid, season), {}).get("ES", 0)
            sog_ag = plr_onice_ag_sog_s.get((pid, season), {}).get("ES", 0)
        else:
            sog_for = sum(plr_onice_for_sog_s.get((pid, season), {}).values())
            sog_ag = sum(plr_onice_ag_sog_s.get((pid, season), {}).values())
        gf, ga, toi_min = r["onice_for_gl"], r["onice_ag_gl"], r["toi_min"]
        if toi_min < TOI_FLOOR_MIN or sog_for <= 0 or sog_ag <= 0:
            continue
        sh_pct = gf / sog_for
        sv_pct = 1 - ga / sog_ag
        pdo = (sh_pct + sv_pct) * 100
        rows.append({
            "player_id": pid, "season": season,
            toi_label: round(toi_min, 1),
            "sog_for": sog_for, "goals_for": int(gf),
            "sog_against": sog_ag, "goals_against": int(ga),
            "sh_pct": round(sh_pct * 100, 2),
            "sv_pct": round(sv_pct * 100, 2),
            "pdo": round(pdo, 2),
        })
    return pd.DataFrame(rows)


def finish_and_write(gl_df, sog_scope, toi_label, out_path, floor_desc):
    out = build_rows(gl_df, sog_scope, toi_label)
    out = out.merge(names, on="player_id", how="left")
    out = out[["player_id", "player_name", "season", toi_label,
               "sog_for", "goals_for", "sog_against", "goals_against",
               "sh_pct", "sv_pct", "pdo"]]
    out = out.sort_values(["season", "pdo"], ascending=[True, False])
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(out)} player-season rows ({floor_desc})")
    print(out["pdo"].describe())
    print(f"mean={out['pdo'].mean():.2f}  median={out['pdo'].median():.2f}  std={out['pdo'].std():.2f}")
    return out


out_5v5 = finish_and_write(gl_es, "ES", "toi_min_es", OUT_CSV, ">=200 min 5v5 TOI")
print("\n=== 5v5 TOP 15 (highest PDO) ===")
print(out_5v5.nlargest(15, "pdo")[["player_name", "season", "toi_min_es", "sh_pct", "sv_pct", "pdo"]].to_string(index=False))
print("\n=== 5v5 BOTTOM 15 (lowest PDO) ===")
print(out_5v5.nsmallest(15, "pdo")[["player_name", "season", "toi_min_es", "sh_pct", "sv_pct", "pdo"]].to_string(index=False))

out_allsit = finish_and_write(gl_all, "ALL", "toi_min_allsit", OUT_CSV_ALLSIT, ">=200 min all-situations TOI")
print("\n=== ALL-SITUATIONS TOP 15 (highest PDO) ===")
print(out_allsit.nlargest(15, "pdo")[["player_name", "season", "toi_min_allsit", "sh_pct", "sv_pct", "pdo"]].to_string(index=False))
print("\n=== ALL-SITUATIONS BOTTOM 15 (lowest PDO) ===")
print(out_allsit.nsmallest(15, "pdo")[["player_name", "season", "toi_min_allsit", "sh_pct", "sv_pct", "pdo"]].to_string(index=False))
