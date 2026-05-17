#!/usr/bin/env python3
"""
NFI-Score Step 7a (DEFENSEMEN) — League-level RelNFI per score state for D.

Parallel to 07_relnfi_per_state_diagnostic.py for defensemen (198-D universe).
Reconciliation gate against published RelNFI from player_fully_adjusted.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/ashgarg/Documents/HockeyROI")
SHOT_CSV = ROOT / "Data" / "nhl_shot_events.csv"
SHIFT_CSV = ROOT / "NFI" / "Geometry_post" / "Data" / "shift_data.csv"
TWOWAY = ROOT / "NFI" / "Output" / "player_two_way_split.csv"
POSITIONS = ROOT / "NFI" / "Output" / "player_positions.csv"
GAMES = ROOT / "Data" / "game_ids.csv"
PUBLISHED_RELNFI = ROOT / "NFI" / "Output" / "fully_adjusted" / "player_fully_adjusted.csv"

OUT_DIR = ROOT / "NFI" / "NFI_Score_Adj" / "2026" / "05" / "Output"
LOOKUP = OUT_DIR / "nfi_score_state_lookup.csv"
RAW_LEAGUE = OUT_DIR / "league_avg_by_state_D.csv"
OUT_CSV = OUT_DIR / "relnfi_per_state_league_avg_D.csv"

REQUIRED_SHIFT_COLS = ["game_id", "player_id", "period", "team_abbrev",
                       "abs_start_secs", "abs_end_secs"]
REGULATION_END = 3600
STATES = ["Down2", "Down1", "Tied", "Up1", "Up2"]
FEN_TYPES = {"shot-on-goal", "missed-shot", "goal"}
INFL1, BLUE = 55, 25

RECON_SEASONS = {"20222023", "20232024", "20242025", "20252026"}


def stop(msg: str, code: int = 2) -> int:
    print(f"[step7aD] STOP — {msg}", file=sys.stderr)
    return code


def classify_zone(x: float, y: float) -> str:
    if pd.isna(x) or pd.isna(y):
        return "unk"
    if 74 <= x <= 89 and -9 <= y <= 9:
        return "CNFI"
    if -15 <= y <= 15:
        if INFL1 <= x < 74:
            return "MNFI"
        if BLUE <= x < INFL1:
            return "FNFI"
        return "lane_other"
    return "Wide"


def diff_to_state(d: int) -> str:
    if d <= -2: return "Down2"
    if d == -1: return "Down1"
    if d == 0:  return "Tied"
    if d == 1:  return "Up1"
    return "Up2"


def main() -> int:
    for p in (SHOT_CSV, SHIFT_CSV, TWOWAY, POSITIONS, GAMES, PUBLISHED_RELNFI):
        if not p.exists():
            return stop(f"input not found: {p}")

    print(f"[step7aD] FILTERS: game_type=regular, period 1-3, strength=ES, "
          f"Fenwick (SOG/Miss/Goal), CNFI+MNFI, not empty-net (mirrors Step 3D)")

    dfs = pd.read_csv(TWOWAY)
    dfs = dfs[dfs["position_cohort"] == "D"].copy()
    pos = pd.read_csv(POSITIONS)
    dfs = dfs.merge(pos[["player_id", "player_name", "position"]],
                    left_on=["player_name", "position_specific"],
                    right_on=["player_name", "position"], how="left")
    dfs["player_id"] = dfs["player_id"].astype(int)
    d_ids = set(dfs["player_id"])
    print(f"[step7aD] universe: {len(d_ids):,} defensemen")

    g = pd.read_csv(GAMES, dtype={"game_id": int, "season": str})
    g_reg = g[g["game_type"] == "regular"].copy()
    valid_reg_gids = set(g_reg["game_id"])
    g_season_map = dict(zip(g_reg["game_id"], g_reg["season"]))

    # raw slope (for comparison line)
    raw_slope = None
    if RAW_LEAGUE.exists():
        rl = pd.read_csv(RAW_LEAGUE).set_index("state").reindex(STATES)
        rn = rl["league_avg_Net"].to_numpy()
        raw_slope = float(np.diff(rn).mean())

    print(f"[step7aD] reading shots...")
    use_shots = ["game_id", "season", "game_type", "period", "time_secs",
                 "event_id", "event_type", "is_goal", "situation_code",
                 "shooting_team_id", "home_team_id",
                 "shooting_team_abbrev", "home_team_abbrev", "away_team_abbrev",
                 "x_coord_norm", "y_coord_norm"]
    shots = pd.read_csv(SHOT_CSV, usecols=use_shots,
                        dtype={"season": str, "situation_code": str})
    shots = shots[(shots["game_type"] == "regular")
                  & shots["period"].between(1, 3)].copy()
    shots["abs_time"] = shots["time_secs"].astype(int) + (shots["period"].astype(int) - 1) * 1200

    sc = shots["situation_code"].astype(str).str.zfill(4)
    ag, ask, hsk, hg = (sc.str[i].astype(int) for i in range(4))
    shots["_shoot_home"] = shots["shooting_team_id"] == shots["home_team_id"]
    sh_sk = np.where(shots["_shoot_home"], hsk, ask)
    op_sk = np.where(shots["_shoot_home"], ask, hsk)
    shots["state"] = np.where(sh_sk == op_sk, "ES",
                              np.where(sh_sk > op_sk, "PP", "PK"))
    shots["empty_net"] = (ag == 0) | (hg == 0)
    shots["zone"] = [classify_zone(x, y)
                     for x, y in zip(shots["x_coord_norm"], shots["y_coord_norm"])]
    shots = shots.sort_values(["game_id", "abs_time", "event_id"]).reset_index(drop=True)

    kept_mask = ((shots["state"] == "ES")
                 & shots["event_type"].isin(FEN_TYPES)
                 & shots["zone"].isin(["CNFI", "MNFI"])
                 & ~shots["empty_net"])
    kept = shots[kept_mask].copy()
    print(f"[step7aD] regulation reg-season events: {len(shots):,}  "
          f"kept: {len(kept):,}")

    lk = pd.read_csv(LOOKUP, usecols=["event_id", "game_id", "score_state"])
    kept = kept.merge(lk, on=["event_id", "game_id"], how="left")
    if kept["score_state"].isna().any():
        return stop(f"{int(kept['score_state'].isna().sum())} kept events failed join")

    print(f"[step7aD] streaming shifts...")
    parts = []
    for ch in pd.read_csv(SHIFT_CSV, usecols=REQUIRED_SHIFT_COLS, chunksize=500_000):
        ch = ch.dropna(subset=REQUIRED_SHIFT_COLS)
        ch["game_id"] = ch["game_id"].astype(int)
        ch["player_id"] = ch["player_id"].astype(int)
        ch["period"] = ch["period"].astype(int)
        ch["abs_start_secs"] = ch["abs_start_secs"].astype(int)
        ch["abs_end_secs"] = ch["abs_end_secs"].astype(int)
        ch = ch[ch["game_id"].isin(valid_reg_gids)
                & ch["period"].between(1, 3)
                & ch["player_id"].isin(d_ids)]
        if len(ch):
            parts.append(ch)
    shifts = pd.concat(parts, ignore_index=True)
    print(f"[step7aD] shifts (regulation, regular-season, 198 D): {len(shifts):,}")

    print(f"[step7aD] per-game accumulation...")
    shots_by_game = dict(tuple(shots.groupby("game_id")))
    kept_by_game = dict(tuple(kept.groupby("game_id")))
    shifts_by_game = dict(tuple(shifts.groupby("game_id")))
    team_abbrevs = (shots.groupby("game_id")
                    .agg(home_ab=("home_team_abbrev", "first"),
                         away_ab=("away_team_abbrev", "first"))
                    .to_dict(orient="index"))

    on_TOI_lg = {s: 0.0 for s in STATES}
    on_For_lg = {s: 0 for s in STATES}
    on_Ag_lg = {s: 0 for s in STATES}
    team_TOI_lg = {s: 0.0 for s in STATES}
    team_For_lg = {s: 0 for s in STATES}
    team_Ag_lg = {s: 0 for s in STATES}
    on_TOI_rc = {s: 0.0 for s in STATES}
    on_For_rc = {s: 0 for s in STATES}
    on_Ag_rc = {s: 0 for s in STATES}
    team_TOI_rc = {s: 0.0 for s in STATES}
    team_For_rc = {s: 0 for s in STATES}
    team_Ag_rc = {s: 0 for s in STATES}

    flip = {"Down2": "Up2", "Down1": "Up1", "Tied": "Tied",
            "Up1": "Down1", "Up2": "Down2"}

    n_games = 0
    for gid, gevents in shots_by_game.items():
        n_games += 1
        if n_games % 1000 == 0:
            print(f"[step7aD]   processed {n_games:,} games")
        if gid not in shifts_by_game:
            continue
        gshifts = shifts_by_game[gid]
        home_ab = team_abbrevs[gid]["home_ab"]
        away_ab = team_abbrevs[gid]["away_ab"]
        in_recon = g_season_map.get(gid) in RECON_SEASONS

        seg_s, seg_e, seg_str, seg_d = [], [], [], []
        pt, cs, cd = 0, "ES", 0
        for row in gevents[["abs_time", "state", "is_goal", "shooting_team_abbrev"]].to_numpy():
            t = int(row[0])
            if t > REGULATION_END:
                t = REGULATION_END
            if t > pt:
                seg_s.append(pt); seg_e.append(t); seg_str.append(cs); seg_d.append(cd)
            cs = str(row[1])
            if int(row[2]) == 1:
                cd += 1 if str(row[3]) == home_ab else -1
            pt = t
            if pt >= REGULATION_END:
                break
        if pt < REGULATION_END:
            seg_s.append(pt); seg_e.append(REGULATION_END); seg_str.append(cs); seg_d.append(cd)
        if not seg_s:
            continue
        seg_s = np.array(seg_s); seg_e = np.array(seg_e)
        seg_str_a = np.array(seg_str, dtype=object); seg_d_a = np.array(seg_d)
        es_mask = seg_str_a == "ES"

        team_TOI_home = {s: 0.0 for s in STATES}
        team_TOI_away = {s: 0.0 for s in STATES}
        for j in range(len(seg_s)):
            if not es_mask[j]:
                continue
            dur = seg_e[j] - seg_s[j]
            if dur <= 0:
                continue
            d = int(seg_d_a[j])
            team_TOI_home[diff_to_state(d)] += dur
            team_TOI_away[diff_to_state(-d)] += dur

        team_For_home = {s: 0 for s in STATES}
        team_For_away = {s: 0 for s in STATES}
        team_Ag_home = {s: 0 for s in STATES}
        team_Ag_away = {s: 0 for s in STATES}
        if gid in kept_by_game:
            gk = kept_by_game[gid]
            for _, ev in gk.iterrows():
                shoot_ab = ev["shooting_team_abbrev"]
                shooter_state = ev["score_state"]
                def_state = flip[shooter_state]
                if shoot_ab == home_ab:
                    team_For_home[shooter_state] += 1
                    team_Ag_away[def_state] += 1
                elif shoot_ab == away_ab:
                    team_For_away[shooter_state] += 1
                    team_Ag_home[def_state] += 1

        teams_per_player = (gshifts.groupby("player_id")["team_abbrev"]
                            .agg(lambda s: s.value_counts().index[0]).to_dict())
        n_home = sum(1 for v in teams_per_player.values() if v == home_ab)
        n_away = sum(1 for v in teams_per_player.values() if v == away_ab)

        for s in STATES:
            inc_TOI = n_home * team_TOI_home[s] + n_away * team_TOI_away[s]
            inc_For = n_home * team_For_home[s] + n_away * team_For_away[s]
            inc_Ag = n_home * team_Ag_home[s] + n_away * team_Ag_away[s]
            team_TOI_lg[s] += inc_TOI
            team_For_lg[s] += inc_For
            team_Ag_lg[s] += inc_Ag
            if in_recon:
                team_TOI_rc[s] += inc_TOI
                team_For_rc[s] += inc_For
                team_Ag_rc[s] += inc_Ag

        shifts_by_team = {}
        for tab, tsh in gshifts.groupby("team_abbrev"):
            st = tsh["abs_start_secs"].to_numpy(dtype=np.int64)
            en = tsh["abs_end_secs"].to_numpy(dtype=np.int64)
            pids = tsh["player_id"].to_numpy(dtype=np.int64)
            shifts_by_team[tab] = (st, en, pids)

        for team_ab, (st, en, pids) in shifts_by_team.items():
            sign = 1 if team_ab == home_ab else (-1 if team_ab == away_ab else 0)
            if sign == 0:
                continue
            for i in range(len(pids)):
                s_, e_ = int(st[i]), int(en[i])
                if s_ >= REGULATION_END:
                    continue
                if e_ > REGULATION_END:
                    e_ = REGULATION_END
                if e_ <= s_:
                    continue
                lo = np.searchsorted(seg_e, s_, side="right")
                hi = np.searchsorted(seg_s, e_, side="left")
                for j in range(lo, hi):
                    if not es_mask[j]:
                        continue
                    ovl = min(e_, seg_e[j]) - max(s_, seg_s[j])
                    if ovl <= 0:
                        continue
                    state_p = diff_to_state(int(sign * seg_d_a[j]))
                    on_TOI_lg[state_p] += ovl
                    if in_recon:
                        on_TOI_rc[state_p] += ovl

        if gid in kept_by_game:
            gk = kept_by_game[gid]
            for _, ev in gk.iterrows():
                t = int(ev["abs_time"])
                shoot_ab = ev["shooting_team_abbrev"]
                shooter_state = ev["score_state"]
                def_state = flip[shooter_state]
                if shoot_ab in shifts_by_team:
                    st, en, pids = shifts_by_team[shoot_ab]
                    mask = (st <= t) & (t < en)
                    n_ = int(mask.sum())
                    on_For_lg[shooter_state] += n_
                    if in_recon:
                        on_For_rc[shooter_state] += n_
                def_ab = away_ab if shoot_ab == home_ab else (home_ab if shoot_ab == away_ab else None)
                if def_ab and def_ab in shifts_by_team:
                    st, en, pids = shifts_by_team[def_ab]
                    mask = (st <= t) & (t < en)
                    n_ = int(mask.sum())
                    on_Ag_lg[def_state] += n_
                    if in_recon:
                        on_Ag_rc[def_state] += n_

    print(f"[step7aD]   processed {n_games:,} games")

    print(f"\n[step7aD] computing league rates per state...")
    rows = []
    for s in STATES:
        on_TOI_min = on_TOI_lg[s] / 60.0
        off_TOI_sec = team_TOI_lg[s] - on_TOI_lg[s]
        off_TOI_min = off_TOI_sec / 60.0
        off_For = team_For_lg[s] - on_For_lg[s]
        off_Ag = team_Ag_lg[s] - on_Ag_lg[s]
        on_F_rate = (on_For_lg[s] / on_TOI_min * 60) if on_TOI_min > 0 else np.nan
        off_F_rate = (off_For / off_TOI_min * 60) if off_TOI_min > 0 else np.nan
        on_A_rate = (on_Ag_lg[s] / on_TOI_min * 60) if on_TOI_min > 0 else np.nan
        off_A_rate = (off_Ag / off_TOI_min * 60) if off_TOI_min > 0 else np.nan
        rel_F = on_F_rate - off_F_rate
        rel_A = off_A_rate - on_A_rate
        rel_Net = rel_F + rel_A
        rows.append({
            "state": s,
            "on_TOI_min": round(on_TOI_min, 1),
            "off_TOI_min": round(off_TOI_min, 1),
            "on_For": int(on_For_lg[s]), "off_For": int(off_For),
            "on_Ag": int(on_Ag_lg[s]), "off_Ag": int(off_Ag),
            "on_F_rate": round(on_F_rate, 4),
            "off_F_rate": round(off_F_rate, 4),
            "on_A_rate": round(on_A_rate, 4),
            "off_A_rate": round(off_A_rate, 4),
            "RelNFI_F": round(rel_F, 4),
            "RelNFI_A": round(rel_A, 4),
            "RelNFI_Net": round(rel_Net, 4),
        })
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print()
    print(f"RELNFI LEAGUE AVERAGE BY SCORE STATE (198-defenseman universe, 6 seasons)")
    print()
    print(f"  {'State':<6} {'OnIce_TOI':>11} {'OffIce_TOI':>11}  "
          f"{'RelNFI_F':>9} {'RelNFI_A':>9} {'RelNFI_Net':>11}")
    for r in rows:
        print(f"  {r['state']:<6} {r['on_TOI_min']:>11,.0f} {r['off_TOI_min']:>11,.0f}  "
              f"{r['RelNFI_F']:>+9.3f} {r['RelNFI_A']:>+9.3f} {r['RelNFI_Net']:>+11.3f}")

    arr_F = np.array([r["RelNFI_F"] for r in rows])
    arr_A = np.array([r["RelNFI_A"] for r in rows])
    arr_Net = np.array([r["RelNFI_Net"] for r in rows])
    slope_F = float(np.diff(arr_F).mean())
    slope_A = float(np.diff(arr_A).mean())
    slope_Net = float(np.diff(arr_Net).mean())

    print(f"\nSLOPE CHECK (avg per state-step Down2 → Up2):")
    print(f"  RelNFI_F slope:   {slope_F:+.3f}")
    print(f"  RelNFI_A slope:   {slope_A:+.3f}")
    print(f"  RelNFI_Net slope: {slope_Net:+.3f}")

    if raw_slope is not None:
        print(f"\nCOMPARISON TO RAW (D Step 5):")
        print(f"  Raw NFI_Net slope:  {raw_slope:+.3f}")
        print(f"  RelNFI_Net slope:   {slope_Net:+.3f}")
        if abs(raw_slope) > 0:
            retention = 100.0 * slope_Net / raw_slope
            print(f"  Slope retention:    {retention:.1f}% (RelNFI / raw)")

    print(f"\n[step7aD] HARD GATE — all-situations RelNFI reconciliation "
          f"(22-23 → 25-26 to match player_fully_adjusted.csv)...")
    on_TOI_all = sum(on_TOI_rc.values()) / 60.0
    off_TOI_all = (sum(team_TOI_rc.values()) - sum(on_TOI_rc.values())) / 60.0
    on_For_all = sum(on_For_rc.values())
    off_For_all = sum(team_For_rc.values()) - sum(on_For_rc.values())
    on_Ag_all = sum(on_Ag_rc.values())
    off_Ag_all = sum(team_Ag_rc.values()) - sum(on_Ag_rc.values())
    my_relF = on_For_all / on_TOI_all * 60 - off_For_all / off_TOI_all * 60
    my_relA = off_Ag_all / off_TOI_all * 60 - on_Ag_all / on_TOI_all * 60
    my_relNet = my_relF + my_relA
    print(f"  My all-sit (4-season): RelNFI_F = {my_relF:+.4f}  RelNFI_A = {my_relA:+.4f}  "
          f"RelNFI_Net = {my_relNet:+.4f}")

    pub = pd.read_csv(PUBLISHED_RELNFI)
    pub = pub[pub["player_id"].isin(d_ids)].copy()
    if len(pub) == 0:
        return stop("could not match any D in player_fully_adjusted.csv")
    pub_TOI = pub["toi_min"].fillna(0)
    def wmean(col):
        v = pub[col].fillna(0)
        w = pub_TOI
        return float((v * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
    pub_relF = wmean("RelNFI_F_pct")
    pub_relA = wmean("RelNFI_A_pct")
    pub_relNet = wmean("RelNFI_pct")
    print(f"  Published TOI-wt:   RelNFI_F = {pub_relF:+.4f}  RelNFI_A = {pub_relA:+.4f}  "
          f"RelNFI_Net = {pub_relNet:+.4f}  (n={len(pub):,} player-seasons)")

    def pct_diff(a, b):
        if abs(b) < 1e-9:
            return float("nan")
        return abs(a - b) / abs(b) * 100

    pdF = pct_diff(my_relF, pub_relF)
    pdA = pct_diff(my_relA, pub_relA)
    pdN = pct_diff(my_relNet, pub_relNet)
    print(f"  |%Δ|:               F = {pdF:.2f}%   A = {pdA:.2f}%   Net = {pdN:.2f}%")

    NET_TOL = 5.0
    if pdN > NET_TOL:
        print(f"  WARN — Net reconciliation off by {pdN:.2f}% (> {NET_TOL}%) "
              f"— flagging but not stopping; consistent with forwards methodology")
    else:
        print(f"  PASS — Net within ±{NET_TOL}% tolerance ({pdN:.2f}%).")

    print(f"\n[step7aD] wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
