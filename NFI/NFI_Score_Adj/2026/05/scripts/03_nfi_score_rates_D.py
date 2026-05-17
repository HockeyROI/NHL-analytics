#!/usr/bin/env python3
"""
NFI-Score Step 3 (DEFENSEMEN) — per-state rates, RawAllSit, A, deltas for D.

Parallel to 03_nfi_score_rates.py but for defensemen (position_cohort == 'D').
Same filters, same identity gate.
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
LOOKUP = OUT_DIR / "nfi_score_state_lookup.csv"
TOI_CSV = OUT_DIR / "per_state_toi_D.csv"
WIDE_CSV = OUT_DIR / "nfi_score_player_rates_D.csv"
SUMMARY_CSV = OUT_DIR / "nfi_score_summary_ranking_D.csv"
WINDOW_REPORT = OUT_DIR / "window_gap_report_D.csv"

REQUIRED_SHIFT_COLS = ["game_id", "player_id", "period", "team_abbrev",
                       "abs_start_secs", "abs_end_secs"]
STATE_ORDER = ["Down2", "Down1", "Tied", "Up1", "Up2"]
TYPES = ["Off", "Def", "Net"]
FEN_TYPES = {"shot-on-goal", "missed-shot", "goal"}
INFL1, BLUE = 55, 25

RECON_TOL_PCT = 0.5
RECON_FAIL_PCT = 1.0


def stop(msg: str, code: int = 2) -> int:
    print(f"[step3D] STOP — {msg}", file=sys.stderr)
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


def main() -> int:
    for p in (SHOT_CSV, SHIFT_CSV, TWOWAY, POSITIONS, GAMES, LOOKUP, TOI_CSV):
        if not p.exists():
            return stop(f"input not found: {p}")

    print(f"[step3D] FILTERS APPLIED: game_type=regular, period 1-3, strength=ES, "
          f"Fenwick (SOG/Miss/Goal), zone in CNFI+MNFI, not empty-net")

    dfs = pd.read_csv(TWOWAY)
    dfs = dfs[dfs["position_cohort"] == "D"].copy()
    pos = pd.read_csv(POSITIONS)
    dfs = dfs.merge(pos[["player_id", "player_name", "position"]],
                    left_on=["player_name", "position_specific"],
                    right_on=["player_name", "position"], how="left")
    dfs["player_id"] = dfs["player_id"].astype(int)
    d_ids = set(dfs["player_id"])
    print(f"[step3D] universe: {len(d_ids):,} defensemen")

    name_map = dict(zip(dfs["player_id"], dfs["player_name"]))
    team_map = dict(zip(dfs["player_id"], dfs["team_2025_26"]))
    pos_map = dict(zip(dfs["player_id"], dfs["position_specific"]))
    existing_off = dict(zip(dfs["player_id"], dfs["offensive_NFI_60"]))
    existing_def = dict(zip(dfs["player_id"], dfs["defensive_NFI_60"]))
    existing_seasons_pooled = dict(zip(dfs["player_id"], dfs["seasons_pooled"]))

    toi = pd.read_csv(TOI_CSV)
    toi = toi.set_index("player_id")
    print(f"[step3D] TOI rows: {len(toi)}")

    g = pd.read_csv(GAMES, dtype={"game_id": int, "season": str})
    valid_reg_gids = set(g.loc[g["game_type"] == "regular", "game_id"])

    print(f"[step3D] reading {SHOT_CSV} ...")
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
    ag = sc.str[0].astype(int)
    ask = sc.str[1].astype(int)
    hsk = sc.str[2].astype(int)
    hg = sc.str[3].astype(int)
    shots["_shoot_home"] = shots["shooting_team_id"] == shots["home_team_id"]
    sh_skaters = np.where(shots["_shoot_home"], hsk, ask)
    op_skaters = np.where(shots["_shoot_home"], ask, hsk)
    shots["state"] = np.where(sh_skaters == op_skaters, "ES",
                              np.where(sh_skaters > op_skaters, "PP", "PK"))
    shots["empty_net"] = (ag == 0) | (hg == 0)
    shots["zone"] = [classify_zone(x, y)
                     for x, y in zip(shots["x_coord_norm"].values,
                                     shots["y_coord_norm"].values)]

    kept_mask = ((shots["state"] == "ES")
                 & shots["event_type"].isin(FEN_TYPES)
                 & shots["zone"].isin(["CNFI", "MNFI"])
                 & ~shots["empty_net"])
    kept = shots[kept_mask].copy()
    print(f"[step3D] kept events: {len(kept):,}")

    lk = pd.read_csv(LOOKUP, usecols=["event_id", "game_id", "score_state"])
    kept = kept.merge(lk, on=["event_id", "game_id"], how="left")
    if kept["score_state"].isna().any():
        return stop(f"{int(kept['score_state'].isna().sum())} kept events failed score_state lookup join")
    print(f"[step3D]   all {len(kept):,} events tagged with score_state")

    print("[step3D] kept-event distribution by score_state:")
    for s in STATE_ORDER:
        n = int((kept["score_state"] == s).sum())
        print(f"   {s:<6}  {n:>10,}  ({n/len(kept)*100:5.2f}%)")

    print(f"[step3D] streaming shifts...")
    parts = []
    for ch in pd.read_csv(SHIFT_CSV, usecols=REQUIRED_SHIFT_COLS,
                          chunksize=500_000):
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
    del parts
    print(f"[step3D]   shifts loaded (regulation, regular-season, 198 D): "
          f"{len(shifts):,}")

    print("[step3D] per-game on-ice attribution...")
    kept = kept.sort_values(["game_id", "abs_time", "event_id"]).reset_index(drop=True)
    kept_by_game = dict(tuple(kept.groupby("game_id")))
    shifts_by_game = dict(tuple(shifts.groupby("game_id")))

    team_abbrevs = (kept.groupby("game_id")
                    .agg(home_ab=("home_team_abbrev", "first"),
                         away_ab=("away_team_abbrev", "first"))
                    .to_dict(orient="index"))

    ev_for: dict = defaultdict(lambda: defaultdict(int))
    ev_ag: dict = defaultdict(lambda: defaultdict(int))

    n_games = 0
    n_no_shifts = 0
    for gid, gevents in kept_by_game.items():
        n_games += 1
        if n_games % 1000 == 0:
            print(f"[step3D]   processed {n_games:,} games")
        if gid not in shifts_by_game:
            n_no_shifts += 1
            continue
        gshifts = shifts_by_game[gid]
        home_ab = team_abbrevs[gid]["home_ab"]
        away_ab = team_abbrevs[gid]["away_ab"]

        shifts_by_team: dict = {}
        for tab, tsh in gshifts.groupby("team_abbrev"):
            st = tsh["abs_start_secs"].to_numpy(dtype=np.int64)
            en = tsh["abs_end_secs"].to_numpy(dtype=np.int64)
            pids = tsh["player_id"].to_numpy(dtype=np.int64)
            shifts_by_team[tab] = (st, en, pids)

        ev_t = gevents["abs_time"].to_numpy(dtype=np.int64)
        ev_shoot = gevents["shooting_team_abbrev"].to_numpy()
        ev_state = gevents["score_state"].to_numpy()
        for i in range(len(gevents)):
            t = int(ev_t[i])
            shoot_ab = ev_shoot[i]
            def_ab = away_ab if shoot_ab == home_ab else home_ab
            state = ev_state[i]

            if shoot_ab in shifts_by_team:
                st, en, pids = shifts_by_team[shoot_ab]
                mask = (st <= t) & (t < en)
                for p in pids[mask]:
                    ev_for[int(p)][state] += 1
            if def_ab in shifts_by_team:
                st, en, pids = shifts_by_team[def_ab]
                mask = (st <= t) & (t < en)
                for p in pids[mask]:
                    ev_ag[int(p)][state] += 1

    print(f"[step3D]   processed {n_games:,} games  (skipped no shifts: {n_no_shifts:,})")

    print("[step3D] computing rates, RawAllSit, A, deltas...")
    rows = []
    for pid in sorted(d_ids):
        if pid not in toi.index:
            continue
        trow = toi.loc[pid]
        toi_state_min = {s: float(trow[f"TOI_{s}"]) for s in STATE_ORDER}
        toi_total = float(trow["TOI_Total"])

        rec = {
            "player_id": pid,
            "player_name": name_map.get(pid, ""),
            "team": team_map.get(pid, ""),
            "position": pos_map.get(pid, ""),
            "GP": int(trow["GP"]),
            "TOI_Total": round(toi_total, 4),
        }
        for s in STATE_ORDER:
            rec[f"TOI_{s}"] = round(toi_state_min[s], 4)

        per_state_rates = {}
        for s in STATE_ORDER:
            t_min = toi_state_min[s]
            n_for = ev_for.get(pid, {}).get(s, 0)
            n_ag = ev_ag.get(pid, {}).get(s, 0)
            off = (n_for / t_min * 60.0) if t_min > 0 else np.nan
            dfn = (n_ag / t_min * 60.0) if t_min > 0 else np.nan
            net = off - dfn if (np.isfinite(off) and np.isfinite(dfn)) else np.nan
            per_state_rates[s] = (off, dfn, net)
            rec[f"NFI-Score-{s}-Off"] = round(off, 4) if np.isfinite(off) else np.nan
            rec[f"NFI-Score-{s}-Def"] = round(dfn, 4) if np.isfinite(dfn) else np.nan
            rec[f"NFI-Score-{s}-Net"] = round(net, 4) if np.isfinite(net) else np.nan

        if toi_total > 0:
            w = {s: toi_state_min[s] / toi_total for s in STATE_ORDER}
        else:
            w = {s: 0 for s in STATE_ORDER}
        raw_off = sum((per_state_rates[s][0] if np.isfinite(per_state_rates[s][0]) else 0)
                      * w[s] for s in STATE_ORDER)
        raw_def = sum((per_state_rates[s][1] if np.isfinite(per_state_rates[s][1]) else 0)
                      * w[s] for s in STATE_ORDER)
        raw_net = raw_off - raw_def
        rec["NFI-Score-RawAllSit-Off"] = round(raw_off, 4)
        rec["NFI-Score-RawAllSit-Def"] = round(raw_def, 4)
        rec["NFI-Score-RawAllSit-Net"] = round(raw_net, 4)

        rec["_for_total"] = sum(ev_for.get(pid, {}).get(s, 0) for s in STATE_ORDER)
        rec["_ag_total"] = sum(ev_ag.get(pid, {}).get(s, 0) for s in STATE_ORDER)
        rec["_pooled_off"] = (rec["_for_total"] / toi_total * 60.0) if toi_total > 0 else np.nan
        rec["_pooled_def"] = (rec["_ag_total"] / toi_total * 60.0) if toi_total > 0 else np.nan

        rows.append(rec)
    df = pd.DataFrame(rows)

    print("\n[step3D] HARD GATE — RawAllSit vs same-window pooled rate (math identity)...")
    pct_off = (df["NFI-Score-RawAllSit-Off"] - df["_pooled_off"]).abs() \
              / df["_pooled_off"].replace(0, np.nan) * 100
    pct_def = (df["NFI-Score-RawAllSit-Def"] - df["_pooled_def"]).abs() \
              / df["_pooled_def"].replace(0, np.nan) * 100
    fail_off = pct_off > RECON_TOL_PCT
    fail_def = pct_def > RECON_TOL_PCT
    n_fail = int((fail_off | fail_def).sum())
    fail_pct = 100.0 * n_fail / len(df)
    print(f"  Off: max %Δ = {pct_off.max():.5f}%   median %Δ = {pct_off.median():.5f}%")
    print(f"  Def: max %Δ = {pct_def.max():.5f}%   median %Δ = {pct_def.median():.5f}%")
    print(f"  failures (>{RECON_TOL_PCT}%): {n_fail} / {len(df)} = {fail_pct:.3f}%")
    if fail_pct > RECON_FAIL_PCT:
        return stop(f"RawAllSit ≠ pooled rate for {fail_pct:.2f}% of players")
    print("  PASS")

    league_toi = {s: float(toi[f"TOI_{s}"].sum()) for s in STATE_ORDER}
    league_total_toi = sum(league_toi.values())
    league_w = {s: (league_toi[s] / league_total_toi if league_total_toi > 0 else 0)
                for s in STATE_ORDER}
    print(f"\n[step3D] league TOI distribution (used for NFI-Score-A):")
    for s in STATE_ORDER:
        print(f"   {s:<6}  {league_toi[s]:>14,.1f} min   weight = {league_w[s]:.5f}")

    for typ in TYPES:
        a_col = f"NFI-Score-A-{typ}"
        df[a_col] = sum(df[f"NFI-Score-{s}-{typ}"].fillna(0) * league_w[s]
                        for s in STATE_ORDER).round(4)

    for s in STATE_ORDER:
        for typ in TYPES:
            df[f"NFI-Score-D-{s}-{typ}"] = (df[f"NFI-Score-{s}-{typ}"]
                                            - df[f"NFI-Score-RawAllSit-{typ}"]).round(4)

    print("\n[step3D] delta sanity (Σ delta × player_TOI%) should ≈ 0...")
    for typ in TYPES:
        accum = np.zeros(len(df))
        for s in STATE_ORDER:
            t_share = (df[f"TOI_{s}"] / df["TOI_Total"].replace(0, np.nan)).fillna(0)
            accum += df[f"NFI-Score-D-{s}-{typ}"].fillna(0).to_numpy() * t_share.to_numpy()
        n_bad = int((np.abs(accum) > 0.05).sum())
        print(f"   {typ}: max |Σ Δ × TOI%| = {np.max(np.abs(accum)):.6f}, "
              f"failures (>0.05): {n_bad}")

    print("\n[step3D] cross-check vs published offensive_NFI_60 / defensive_NFI_60 (D):")
    df["_pub_off"] = df["player_id"].map(existing_off)
    df["_pub_def"] = df["player_id"].map(existing_def)
    ratio_off = df["NFI-Score-RawAllSit-Off"] / df["_pub_off"]
    ratio_def = df["NFI-Score-RawAllSit-Def"] / df["_pub_def"]
    print(f"   median ratio mine/published — Off: {ratio_off.median():.4f}   "
          f"Def: {ratio_def.median():.4f}")

    rep_lines = [
        {"key": "step3D_kept_events", "value": str(len(kept))},
        {"key": "step3D_recon_max_pct_off", "value": f"{pct_off.max():.6f}"},
        {"key": "step3D_recon_max_pct_def", "value": f"{pct_def.max():.6f}"},
        {"key": "step3D_recon_failures", "value": str(n_fail)},
        {"key": "crosscheck_median_ratio_off_D", "value": f"{ratio_off.median():.6f}"},
        {"key": "crosscheck_median_ratio_def_D", "value": f"{ratio_def.median():.6f}"},
    ]
    pd.DataFrame(rep_lines).to_csv(WINDOW_REPORT, index=False)
    print(f"[step3D] wrote {WINDOW_REPORT}")

    drop_cols = [c for c in df.columns if c.startswith("_")]
    df_out = df.drop(columns=drop_cols).copy()

    head = ["player_id", "player_name", "team", "position", "GP", "TOI_Total"]
    head += [f"TOI_{s}" for s in STATE_ORDER]
    raw_cols = [f"NFI-Score-{s}-{t}" for s in STATE_ORDER for t in TYPES]
    raw_all = [f"NFI-Score-RawAllSit-{t}" for t in TYPES]
    a_cols = [f"NFI-Score-A-{t}" for t in TYPES]
    d_cols = [f"NFI-Score-D-{s}-{t}" for s in STATE_ORDER for t in TYPES]
    df_out = df_out[head + raw_cols + raw_all + a_cols + d_cols]
    df_out.to_csv(WIDE_CSV, index=False)
    print(f"\n[step3D] wrote {WIDE_CSV}  ({len(df_out)} rows, {len(df_out.columns)} cols)")

    df_out["A_minus_RawAllSit_Net"] = (df_out["NFI-Score-A-Net"]
                                       - df_out["NFI-Score-RawAllSit-Net"]).round(4)
    summary = df_out.sort_values("A_minus_RawAllSit_Net", ascending=False)[[
        "player_id", "player_name", "team", "position", "GP", "TOI_Total",
        "NFI-Score-RawAllSit-Off", "NFI-Score-RawAllSit-Def", "NFI-Score-RawAllSit-Net",
        "NFI-Score-A-Off", "NFI-Score-A-Def", "NFI-Score-A-Net",
        "A_minus_RawAllSit_Net",
    ]]
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[step3D] wrote {SUMMARY_CSV}  ({len(summary)} rows)")

    print(f"\n[step3D] League TOI mix (D): " +
          "  ".join(f"{s}={league_w[s]*100:.1f}%" for s in STATE_ORDER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
