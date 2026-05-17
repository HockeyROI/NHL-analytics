#!/usr/bin/env python3
"""
NFI-Score Step 2 (DEFENSEMEN) — Per-state ES TOI for the 198-defenseman universe.

Parallel to 02_per_state_toi.py but for defensemen (position_cohort == 'D').

Same approach, same filters, same hard checkpoints. Threshold is TOI >= 1500,
GP_2025_26 >= 20 (matches forwards run; player_two_way_split.csv already enforces
this — all 198 D pass).
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/ashgarg/Documents/HockeyROI")
SHOT_CSV = ROOT / "Data" / "nhl_shot_events.csv"
SHIFT_CSV = ROOT / "NFI" / "Geometry_post" / "Data" / "shift_data.csv"
TWOWAY = ROOT / "NFI" / "Output" / "player_two_way_split.csv"
POSITIONS = ROOT / "NFI" / "Output" / "player_positions.csv"
GAMES = ROOT / "Data" / "game_ids.csv"

OUT_DIR = ROOT / "NFI" / "NFI_Score_Adj" / "2026" / "05" / "Output"
OUT_CSV = OUT_DIR / "per_state_toi_D.csv"

REQUIRED_SHIFT_COLS = ["game_id", "player_id", "period", "team_abbrev",
                       "abs_start_secs", "abs_end_secs"]
REGULATION_END = 3600
STATE_ORDER = ["Down2", "Down1", "Tied", "Up1", "Up2"]
EXPECTED_SEASONS = ["20202021", "20212022", "20222023", "20232024",
                    "20242025", "20252026"]


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
    print(f"[step2D] STOP — {msg}", file=sys.stderr)
    return code


def main() -> int:
    for p in (SHOT_CSV, SHIFT_CSV, TWOWAY, POSITIONS, GAMES):
        if not p.exists():
            return stop(f"input not found: {p}")

    print(f"[step2D] shot source: {SHOT_CSV}")
    print(f"[step2D] shift source: {SHIFT_CSV}")
    print(f"[step2D] universe source: {TWOWAY}")

    # ---------- defenseman universe ----------
    dfs = pd.read_csv(TWOWAY)
    dfs = dfs[dfs["position_cohort"] == "D"].copy()
    pos = pd.read_csv(POSITIONS)
    dfs = dfs.merge(pos[["player_id", "player_name", "position"]],
                    left_on=["player_name", "position_specific"],
                    right_on=["player_name", "position"], how="left")
    if dfs["player_id"].isna().any():
        return stop(f"{int(dfs['player_id'].isna().sum())} defensemen failed name+position join")
    dfs["player_id"] = dfs["player_id"].astype(int)
    d_ids = set(dfs["player_id"])
    print(f"[step2D] defenseman universe: {len(d_ids):,} player_ids")

    name_map = dict(zip(dfs["player_id"], dfs["player_name"]))
    team_map = dict(zip(dfs["player_id"], dfs["team_2025_26"]))
    pos_map = dict(zip(dfs["player_id"], dfs["position_specific"]))
    existing_toi = dict(zip(dfs["player_id"], dfs["total_es_toi_min"]))
    existing_gp = dict(zip(dfs["player_id"], dfs["gp_2025_26"]))

    # ---------- regular-season games ----------
    g = pd.read_csv(GAMES, dtype={"game_id": int, "season": str})
    g_reg = g[g["game_type"] == "regular"].copy()
    g_season = dict(zip(g_reg["game_id"], g_reg["season"]))
    valid_reg_gids = set(g_reg["game_id"])
    seasons_in_games = sorted(g_reg["season"].unique())
    print(f"[step2D] regular-season games: {len(valid_reg_gids):,} "
          f"(seasons: {seasons_in_games})")

    # ---------- shots ----------
    print(f"[step2D] reading {SHOT_CSV}...")
    use_shots = ["game_id", "season", "game_type", "period", "time_secs",
                 "event_id", "is_goal", "situation_code",
                 "shooting_team_id", "home_team_id",
                 "shooting_team_abbrev", "home_team_abbrev", "away_team_abbrev"]
    shots = pd.read_csv(SHOT_CSV, usecols=use_shots,
                        dtype={"season": str, "situation_code": str})
    shots = shots[(shots["game_type"] == "regular")
                  & shots["period"].between(1, 3)].copy()
    shots["abs_time"] = shots["time_secs"].astype(int) + (shots["period"].astype(int) - 1) * 1200

    sc = shots["situation_code"].astype(str).str.zfill(4)
    ag = sc.str[0].astype(int)
    ask = sc.str[1].astype(int)
    hsk = sc.str[2].astype(int)
    hg = sc.str[3].astype(int)
    shots["_shoot_home"] = shots["shooting_team_id"] == shots["home_team_id"]
    sh_skaters = np.where(shots["_shoot_home"], hsk, ask)
    op_skaters = np.where(shots["_shoot_home"], ask, hsk)
    state = np.where(sh_skaters == op_skaters, "ES",
                     np.where(sh_skaters > op_skaters, "PP", "PK"))
    shots["state"] = state
    shots["empty_net"] = (ag == 0) | (hg == 0)

    shots = shots.sort_values(["game_id", "abs_time", "event_id"]).reset_index(drop=True)
    print(f"[step2D]   {len(shots):,} regulation regular-season events across "
          f"{shots['game_id'].nunique():,} games")
    print(f"[step2D] FILTERS APPLIED: game_type=regular, period 1-3, "
          f"strength==ES (segment-level), home/away from home_team_id")

    print(f"[step2D] checking shift_data schema...")
    sample = pd.read_csv(SHIFT_CSV, nrows=5)
    missing = [c for c in REQUIRED_SHIFT_COLS if c not in sample.columns]
    if missing:
        return stop(f"shift_data.csv missing required columns: {missing}")

    print(f"[step2D] streaming shifts...")
    parts = []
    seasons_seen_in_shifts = set()
    for ch in pd.read_csv(SHIFT_CSV, usecols=REQUIRED_SHIFT_COLS,
                          chunksize=500_000):
        ch = ch.dropna(subset=REQUIRED_SHIFT_COLS)
        ch["game_id"] = ch["game_id"].astype(int)
        ch["player_id"] = ch["player_id"].astype(int)
        ch["period"] = ch["period"].astype(int)
        ch["abs_start_secs"] = ch["abs_start_secs"].astype(int)
        ch["abs_end_secs"] = ch["abs_end_secs"].astype(int)
        for gid in ch["game_id"].unique():
            s = g_season.get(gid)
            if s:
                seasons_seen_in_shifts.add(s)
        ch = ch[ch["game_id"].isin(valid_reg_gids)
                & ch["period"].between(1, 3)
                & ch["player_id"].isin(d_ids)]
        if len(ch):
            parts.append(ch)
    shifts = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=REQUIRED_SHIFT_COLS)
    del parts
    print(f"[step2D]   shifts (regulation, regular-season, target D only): "
          f"{len(shifts):,}")
    seasons_seen = sorted(seasons_seen_in_shifts)
    print(f"[step2D]   seasons covered: {seasons_seen}")
    if set(seasons_seen) != set(EXPECTED_SEASONS):
        missing_seasons = sorted(set(EXPECTED_SEASONS) - set(seasons_seen))
        return stop(f"shift_data.csv missing seasons: {missing_seasons}")

    print("[step2D] per-game segment build + shift intersection...")
    shots_by_game = dict(tuple(shots.groupby("game_id")))
    shifts_by_game = dict(tuple(shifts.groupby("game_id")))
    team_abbrevs = (shots.groupby("game_id")
                    .agg(home_ab=("home_team_abbrev", "first"),
                         away_ab=("away_team_abbrev", "first"))
                    .to_dict(orient="index"))

    pid_state_sec: dict = defaultdict(lambda: defaultdict(float))

    n_games = 0
    n_no_shifts = 0
    for gid, gshots in shots_by_game.items():
        n_games += 1
        if n_games % 1000 == 0:
            print(f"[step2D]   processed {n_games:,} games")
        if gid not in shifts_by_game:
            n_no_shifts += 1
            continue
        gshifts = shifts_by_game[gid]
        home_ab = team_abbrevs[gid]["home_ab"]
        away_ab = team_abbrevs[gid]["away_ab"]

        ev = gshots[["abs_time", "state", "is_goal", "shooting_team_abbrev"]].to_numpy()

        seg_starts: list = []
        seg_ends: list = []
        seg_strength: list = []
        seg_home_diff: list = []

        prev_t = 0
        cur_strength = "ES"
        cur_home_diff = 0
        for row in ev:
            t = int(row[0])
            if t > REGULATION_END:
                t = REGULATION_END
            if t > prev_t:
                seg_starts.append(prev_t)
                seg_ends.append(t)
                seg_strength.append(cur_strength)
                seg_home_diff.append(cur_home_diff)
            cur_strength = str(row[1])
            if int(row[2]) == 1:
                if str(row[3]) == home_ab:
                    cur_home_diff += 1
                else:
                    cur_home_diff -= 1
            prev_t = t
            if prev_t >= REGULATION_END:
                break
        if prev_t < REGULATION_END:
            seg_starts.append(prev_t)
            seg_ends.append(REGULATION_END)
            seg_strength.append(cur_strength)
            seg_home_diff.append(cur_home_diff)

        if not seg_starts:
            continue

        seg_starts_a = np.array(seg_starts, dtype=np.int64)
        seg_ends_a = np.array(seg_ends, dtype=np.int64)
        seg_strength_a = np.array(seg_strength, dtype=object)
        seg_home_diff_a = np.array(seg_home_diff, dtype=np.int64)
        es_mask = (seg_strength_a == "ES")

        sh_st = gshifts["abs_start_secs"].to_numpy(dtype=np.int64)
        sh_en = gshifts["abs_end_secs"].to_numpy(dtype=np.int64)
        sh_pid = gshifts["player_id"].to_numpy(dtype=np.int64)
        sh_team = gshifts["team_abbrev"].to_numpy()

        for i in range(len(sh_pid)):
            s = sh_st[i]
            e = sh_en[i]
            if s >= REGULATION_END:
                continue
            if e > REGULATION_END:
                e = REGULATION_END
            if e <= s:
                continue
            team = sh_team[i]
            sign = 1 if team == home_ab else (-1 if team == away_ab else 0)
            if sign == 0:
                continue
            pid = sh_pid[i]
            lo = np.searchsorted(seg_ends_a, s, side="right")
            hi = np.searchsorted(seg_starts_a, e, side="left")
            for j in range(lo, hi):
                if not es_mask[j]:
                    continue
                ovl = min(e, seg_ends_a[j]) - max(s, seg_starts_a[j])
                if ovl <= 0:
                    continue
                diff_team = sign * int(seg_home_diff_a[j])
                state_name = diff_to_state(diff_team)
                pid_state_sec[pid][state_name] += ovl

    print(f"[step2D]   processed {n_games:,} games  (skipped, no shifts: {n_no_shifts:,})")

    rows = []
    for pid in sorted(d_ids):
        sec = pid_state_sec.get(pid, {})
        toi_state_min = {f"TOI_{s}": sec.get(s, 0.0) / 60.0 for s in STATE_ORDER}
        toi_total_min = sum(toi_state_min.values())
        rows.append({
            "player_id": pid,
            "player_name": name_map.get(pid, ""),
            "team": team_map.get(pid, ""),
            "position": pos_map.get(pid, ""),
            "GP": existing_gp.get(pid, 0),
            **{k: round(v, 4) for k, v in toi_state_min.items()},
            "TOI_Total": round(toi_total_min, 4),
            "TOI_Existing_AllSit": round(float(existing_toi.get(pid, 0.0)), 4),
        })
    out = pd.DataFrame(rows)

    sum_by_state = out[[f"TOI_{s}" for s in STATE_ORDER]].sum(axis=1)
    diff = (sum_by_state - out["TOI_Total"]).abs()
    n_bad = int((diff > 0.01).sum())
    print(f"\n[step2D] sanity (state-sum == TOI_Total within 0.01 min): "
          f"{n_bad} failures of {len(out)}")
    if n_bad:
        return stop("per-state TOI does not sum to TOI_Total")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[step2D] wrote {OUT_CSV}  ({len(out):,} rows)")

    league_total = out["TOI_Total"].sum()
    print("\n[step2D] league per-state TOI share (198 D combined):")
    for s in STATE_ORDER:
        col = f"TOI_{s}"
        share = out[col].sum() / league_total * 100 if league_total > 0 else 0
        print(f"   {s:<6} {out[col].sum():>14,.1f} min   {share:6.2f}%")
    print(f"   TOTAL  {league_total:>14,.1f} min")

    print("\n[step2D] per-state TOI distribution (per-D, in minutes):")
    for s in STATE_ORDER:
        col = f"TOI_{s}"
        print(f"   {s:<6}  min={out[col].min():>7,.1f}  median={out[col].median():>7,.1f}  "
              f"max={out[col].max():>7,.1f}")

    delta = out["TOI_Total"] - out["TOI_Existing_AllSit"]
    pct = (delta.abs() / out["TOI_Existing_AllSit"].replace(0, np.nan) * 100)
    print(f"\n[step2D] cross-check vs existing total_es_toi_min:")
    print(f"   median |Δ|: {delta.abs().median():.2f} min   "
          f"mean |Δ|: {delta.abs().mean():.2f} min   "
          f"max |Δ|: {delta.abs().max():.2f} min")
    print(f"   players within 1%: {int((pct < 1).sum())} / {len(out)}")
    print(f"   players within 5%: {int((pct < 5).sum())} / {len(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
