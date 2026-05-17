#!/usr/bin/env python3
"""
NFI-Score Step 3 — per-state rates, RawAllSit, score-adjusted A, deltas.

Reads:
  - NFI/NFI_Score_Adj/2026/05/Output/nfi_score_state_lookup.csv  (Step 1)
  - NFI/NFI_Score_Adj/2026/05/Output/per_state_toi.csv          (Step 2)
  - NFI/Output/player_two_way_split.csv                          (universe)
  - NFI/Output/player_positions.csv                              (id ↔ name)
  - Data/nhl_shot_events.csv                                     (events)
  - NFI/Geometry_post/Data/shift_data.csv                        (shifts)

Mirrors existing two-way pipeline filters EXACTLY:
  - game_type == 'regular'
  - period in {1, 2, 3}
  - strength == ES (situation_code: shooting-side skaters == defending-side skaters)
  - Fenwick event_types: shot-on-goal, missed-shot, goal
  - zone in {CNFI, MNFI}  (FNFI excluded; Wide / lane_other excluded)
  - empty_net excluded

Writes:
  - Output/nfi_score_player_rates.csv     (wide: player x all metrics)
  - Output/nfi_score_summary_ranking.csv  (headline ranking)
  - Output/window_gap_report.csv          (appended cross-check vs published)

Hard reconciliation gate: NFI-Score-RawAllSit-Off and -Def must equal each
player's same-window pooled per-60 (a math identity) within ±0.5%. If >1% of
players fail, STOP.
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
TOI_CSV = OUT_DIR / "per_state_toi.csv"
WIDE_CSV = OUT_DIR / "nfi_score_player_rates.csv"
SUMMARY_CSV = OUT_DIR / "nfi_score_summary_ranking.csv"
WINDOW_REPORT = OUT_DIR / "window_gap_report.csv"

REQUIRED_SHIFT_COLS = ["game_id", "player_id", "period", "team_abbrev",
                       "abs_start_secs", "abs_end_secs"]
STATE_ORDER = ["Down2", "Down1", "Tied", "Up1", "Up2"]
TYPES = ["Off", "Def", "Net"]
FEN_TYPES = {"shot-on-goal", "missed-shot", "goal"}
INFL1, BLUE = 55, 25  # zone classification thresholds

RECON_TOL_PCT = 0.5      # per-player |Δ| as % of pooled rate
RECON_FAIL_PCT = 1.0     # if more than X% of players fail, STOP


def stop(msg: str, code: int = 2) -> int:
    print(f"[step3] STOP — {msg}", file=sys.stderr)
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

    print(f"[step3] FILTERS APPLIED: game_type=regular, period 1-3, strength=ES, "
          f"Fenwick (SOG/Miss/Goal), zone in CNFI+MNFI, not empty-net")

    # ---------- universe & TOI ----------
    fwd = pd.read_csv(TWOWAY)
    fwd = fwd[fwd["position_cohort"] == "F"].copy()
    pos = pd.read_csv(POSITIONS)
    fwd = fwd.merge(pos[["player_id", "player_name", "position"]],
                    left_on=["player_name", "position_specific"],
                    right_on=["player_name", "position"], how="left")
    fwd["player_id"] = fwd["player_id"].astype(int)
    fwd_ids = set(fwd["player_id"])
    print(f"[step3] universe: {len(fwd_ids):,} forwards")

    name_map = dict(zip(fwd["player_id"], fwd["player_name"]))
    team_map = dict(zip(fwd["player_id"], fwd["team_2025_26"]))
    pos_map = dict(zip(fwd["player_id"], fwd["position_specific"]))
    existing_off = dict(zip(fwd["player_id"], fwd["offensive_NFI_60"]))
    existing_def = dict(zip(fwd["player_id"], fwd["defensive_NFI_60"]))
    existing_seasons_pooled = dict(zip(fwd["player_id"], fwd["seasons_pooled"]))

    toi = pd.read_csv(TOI_CSV)
    toi = toi.set_index("player_id")
    print(f"[step3] TOI rows: {len(toi)}")

    # ---------- regular-season game ids ----------
    g = pd.read_csv(GAMES, dtype={"game_id": int, "season": str})
    valid_reg_gids = set(g.loc[g["game_type"] == "regular", "game_id"])

    # ---------- shots ----------
    print(f"[step3] reading {SHOT_CSV} ...")
    use_shots = ["game_id", "season", "game_type", "period", "time_secs",
                 "event_id", "event_type", "is_goal", "situation_code",
                 "shooting_team_id", "home_team_id",
                 "shooting_team_abbrev", "home_team_abbrev", "away_team_abbrev",
                 "x_coord_norm", "y_coord_norm"]
    shots = pd.read_csv(SHOT_CSV, usecols=use_shots,
                        dtype={"season": str, "situation_code": str})

    # base filters: regular, regulation
    shots = shots[(shots["game_type"] == "regular")
                  & shots["period"].between(1, 3)].copy()
    shots["abs_time"] = shots["time_secs"].astype(int) + (shots["period"].astype(int) - 1) * 1200

    # strength derivation
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

    # zones
    shots["zone"] = [classify_zone(x, y)
                     for x, y in zip(shots["x_coord_norm"].values,
                                     shots["y_coord_norm"].values)]

    # the kept set: ES, Fenwick, CNFI+MNFI, not empty-net
    kept_mask = ((shots["state"] == "ES")
                 & shots["event_type"].isin(FEN_TYPES)
                 & shots["zone"].isin(["CNFI", "MNFI"])
                 & ~shots["empty_net"])
    kept = shots[kept_mask].copy()
    print(f"[step3] events after each filter:")
    print(f"   reg + regulation:                    {len(shots):,}")
    print(f"   + ES:                                {int((shots['state']=='ES').sum()):,}")
    print(f"   + Fenwick:                           "
          f"{int(((shots['state']=='ES') & shots['event_type'].isin(FEN_TYPES)).sum()):,}")
    print(f"   + CNFI/MNFI:                         "
          f"{int(((shots['state']=='ES') & shots['event_type'].isin(FEN_TYPES) & shots['zone'].isin(['CNFI','MNFI'])).sum()):,}")
    print(f"   + not empty-net (final kept set):    {len(kept):,}")

    # join score_state from Step 1 lookup
    print(f"[step3] joining score_state from {LOOKUP} ...")
    lk = pd.read_csv(LOOKUP, usecols=["event_id", "game_id", "score_state"])
    kept = kept.merge(lk, on=["event_id", "game_id"], how="left")
    n_unmatched = int(kept["score_state"].isna().sum())
    if n_unmatched:
        return stop(f"{n_unmatched:,} kept events failed score_state lookup join")
    print(f"[step3]   all {len(kept):,} events tagged with score_state")

    # by-state breakdown of kept events
    print("[step3] kept-event distribution by score_state:")
    for s in STATE_ORDER:
        n = int((kept["score_state"] == s).sum())
        print(f"   {s:<6}  {n:>10,}  ({n/len(kept)*100:5.2f}%)")

    # ---------- shifts ----------
    print(f"[step3] streaming shifts...")
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
                & ch["player_id"].isin(fwd_ids)]
        if len(ch):
            parts.append(ch)
    shifts = pd.concat(parts, ignore_index=True)
    del parts
    print(f"[step3]   shifts loaded (regulation, regular-season, 379 forwards): "
          f"{len(shifts):,}")

    # ---------- per-game iteration: on-ice attribution ----------
    print("[step3] per-game on-ice attribution...")
    kept = kept.sort_values(["game_id", "abs_time", "event_id"]).reset_index(drop=True)
    kept_by_game = dict(tuple(kept.groupby("game_id")))
    shifts_by_game = dict(tuple(shifts.groupby("game_id")))

    # team abbrevs per game
    team_abbrevs = (kept.groupby("game_id")
                    .agg(home_ab=("home_team_abbrev", "first"),
                         away_ab=("away_team_abbrev", "first"))
                    .to_dict(orient="index"))

    # accumulators
    ev_for: dict = defaultdict(lambda: defaultdict(int))   # pid -> state -> count
    ev_ag: dict = defaultdict(lambda: defaultdict(int))

    n_games = 0
    n_no_shifts = 0
    for gid, gevents in kept_by_game.items():
        n_games += 1
        if n_games % 1000 == 0:
            print(f"[step3]   processed {n_games:,} games")
        if gid not in shifts_by_game:
            n_no_shifts += 1
            continue
        gshifts = shifts_by_game[gid]
        home_ab = team_abbrevs[gid]["home_ab"]
        away_ab = team_abbrevs[gid]["away_ab"]

        # group shifts by team
        shifts_by_team: dict = {}
        for tab, tsh in gshifts.groupby("team_abbrev"):
            st = tsh["abs_start_secs"].to_numpy(dtype=np.int64)
            en = tsh["abs_end_secs"].to_numpy(dtype=np.int64)
            pids = tsh["player_id"].to_numpy(dtype=np.int64)
            shifts_by_team[tab] = (st, en, pids)

        # iterate kept events in this game
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

    print(f"[step3]   processed {n_games:,} games  (skipped no shifts: {n_no_shifts:,})")

    # ---------- assemble per-player per-state rates ----------
    print("[step3] computing rates, RawAllSit, A, deltas...")
    rows = []
    league_for_total = 0
    league_ag_total = 0
    for pid in sorted(fwd_ids):
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

        # per-state rates
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
            league_for_total += n_for
            league_ag_total += n_ag

        # RawAllSit (player TOI weighted)
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

    # ---------- HARD GATE: RawAllSit == pooled rate (math identity) ----------
    print("\n[step3] HARD GATE — RawAllSit vs same-window pooled rate (math identity)...")
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
        print("  first 5 failures:")
        sample = df[fail_off | fail_def].head(5)
        print(sample[["player_id", "player_name",
                      "NFI-Score-RawAllSit-Off", "_pooled_off",
                      "NFI-Score-RawAllSit-Def", "_pooled_def"]].to_string())
        return stop(f"RawAllSit ≠ pooled rate for {fail_pct:.2f}% of players "
                    f"(threshold {RECON_FAIL_PCT}%)")
    print("  PASS")

    # ---------- league TOI distribution & A ----------
    league_toi = {s: float(toi[f"TOI_{s}"].sum()) for s in STATE_ORDER}
    league_total_toi = sum(league_toi.values())
    league_w = {s: (league_toi[s] / league_total_toi if league_total_toi > 0 else 0)
                for s in STATE_ORDER}
    print(f"\n[step3] league TOI distribution (used for NFI-Score-A):")
    for s in STATE_ORDER:
        print(f"   {s:<6}  {league_toi[s]:>14,.1f} min   weight = {league_w[s]:.5f}")

    for typ in TYPES:
        a_col = f"NFI-Score-A-{typ}"
        df[a_col] = sum(df[f"NFI-Score-{s}-{typ}"].fillna(0) * league_w[s]
                        for s in STATE_ORDER).round(4)

    # ---------- deltas ----------
    for s in STATE_ORDER:
        for typ in TYPES:
            df[f"NFI-Score-D-{s}-{typ}"] = (df[f"NFI-Score-{s}-{typ}"]
                                            - df[f"NFI-Score-RawAllSit-{typ}"]).round(4)

    # ---------- delta sanity: weighted-by-player-TOI sum = 0 ----------
    print("\n[step3] delta sanity (Σ delta × player_TOI%) should ≈ 0...")
    failures = []
    for typ in TYPES:
        accum = np.zeros(len(df))
        for s in STATE_ORDER:
            t_share = (df[f"TOI_{s}"] / df["TOI_Total"].replace(0, np.nan)).fillna(0)
            accum += df[f"NFI-Score-D-{s}-{typ}"].fillna(0).to_numpy() * t_share.to_numpy()
        n_bad = int((np.abs(accum) > 0.05).sum())
        print(f"   {typ}: max |Σ Δ × TOI%| = {np.max(np.abs(accum)):.6f}, "
              f"failures (>0.05): {n_bad}")
        if n_bad:
            failures.append((typ, n_bad))
    # informational only — math identity should hold to numerical precision

    # ---------- cross-check vs published two-way ----------
    print("\n[step3] cross-check vs published offensive_NFI_60 / defensive_NFI_60 "
          "(informational, both on 6-season window now):")
    df["_pub_off"] = df["player_id"].map(existing_off)
    df["_pub_def"] = df["player_id"].map(existing_def)
    ratio_off = df["NFI-Score-RawAllSit-Off"] / df["_pub_off"]
    ratio_def = df["NFI-Score-RawAllSit-Def"] / df["_pub_def"]
    print(f"   median ratio mine/published — Off: {ratio_off.median():.4f}   "
          f"Def: {ratio_def.median():.4f}")
    print(f"   IQR (Q25, Q75) — Off: ({ratio_off.quantile(0.25):.4f}, {ratio_off.quantile(0.75):.4f})  "
          f"Def: ({ratio_def.quantile(0.25):.4f}, {ratio_def.quantile(0.75):.4f})")
    by_seasons = df.copy()
    by_seasons["seasons_pooled"] = by_seasons["player_id"].map(existing_seasons_pooled)
    print("   median ratio by seasons_pooled:")
    grp = by_seasons.groupby("seasons_pooled").agg(
        n=("player_id", "count"),
        med_off=("NFI-Score-RawAllSit-Off", "median"),
        med_def=("NFI-Score-RawAllSit-Def", "median"),
    )
    grp["ratio_off"] = (by_seasons.groupby("seasons_pooled")
                        .apply(lambda g: (g["NFI-Score-RawAllSit-Off"] / g["_pub_off"]).median()))
    grp["ratio_def"] = (by_seasons.groupby("seasons_pooled")
                        .apply(lambda g: (g["NFI-Score-RawAllSit-Def"] / g["_pub_def"]).median()))
    print(grp[["n", "ratio_off", "ratio_def"]].round(4).to_string())

    # append to window report
    rep_lines = [
        {"key": "step3_kept_events", "value": str(len(kept))},
        {"key": "step3_recon_max_pct_off", "value": f"{pct_off.max():.6f}"},
        {"key": "step3_recon_max_pct_def", "value": f"{pct_def.max():.6f}"},
        {"key": "step3_recon_failures", "value": str(n_fail)},
        {"key": "crosscheck_median_ratio_off", "value": f"{ratio_off.median():.6f}"},
        {"key": "crosscheck_median_ratio_def", "value": f"{ratio_def.median():.6f}"},
    ]
    if WINDOW_REPORT.exists():
        prior = pd.read_csv(WINDOW_REPORT)
        rep_df = pd.concat([prior, pd.DataFrame(rep_lines)], ignore_index=True)
    else:
        rep_df = pd.DataFrame(rep_lines)
    rep_df.to_csv(WINDOW_REPORT, index=False)
    print(f"[step3] appended cross-check to {WINDOW_REPORT}")

    # ---------- write wide CSV ----------
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df_out = df.drop(columns=drop_cols).copy()

    # column order
    head = ["player_id", "player_name", "team", "position", "GP", "TOI_Total"]
    head += [f"TOI_{s}" for s in STATE_ORDER]
    raw_cols = [f"NFI-Score-{s}-{t}" for s in STATE_ORDER for t in TYPES]
    raw_all = [f"NFI-Score-RawAllSit-{t}" for t in TYPES]
    a_cols = [f"NFI-Score-A-{t}" for t in TYPES]
    d_cols = [f"NFI-Score-D-{s}-{t}" for s in STATE_ORDER for t in TYPES]
    df_out = df_out[head + raw_cols + raw_all + a_cols + d_cols]
    df_out.to_csv(WIDE_CSV, index=False)
    print(f"\n[step3] wrote {WIDE_CSV}  ({len(df_out)} rows, {len(df_out.columns)} cols)")

    # ---------- headline ranking ----------
    df_out["A_minus_RawAllSit_Net"] = (df_out["NFI-Score-A-Net"]
                                       - df_out["NFI-Score-RawAllSit-Net"]).round(4)
    summary = df_out.sort_values("A_minus_RawAllSit_Net", ascending=False)[[
        "player_id", "player_name", "team", "position", "GP", "TOI_Total",
        "NFI-Score-RawAllSit-Off", "NFI-Score-RawAllSit-Def", "NFI-Score-RawAllSit-Net",
        "NFI-Score-A-Off", "NFI-Score-A-Def", "NFI-Score-A-Net",
        "A_minus_RawAllSit_Net",
    ]]
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[step3] wrote {SUMMARY_CSV}  ({len(summary)} rows)")

    # ---------- direction-of-effect verification ----------
    print(f"\n[step3] DIRECTION-OF-EFFECT EVIDENCE")
    print(f"  League TOI mix: " +
          "  ".join(f"{s}={league_w[s]*100:.1f}%" for s in STATE_ORDER))
    print(f"\n  TOP 10 by (NFI-Score-A-Net − NFI-Score-RawAllSit-Net):")
    print("  " + summary.head(10)[[
        "player_name", "team", "NFI-Score-RawAllSit-Net",
        "NFI-Score-A-Net", "A_minus_RawAllSit_Net"
    ]].to_string(index=False))
    print(f"\n  BOTTOM 10:")
    print("  " + summary.tail(10)[[
        "player_name", "team", "NFI-Score-RawAllSit-Net",
        "NFI-Score-A-Net", "A_minus_RawAllSit_Net"
    ]].to_string(index=False))
    print(f"\n  Hypothesis: positive Δ = score adjustment HELPS (player's own TOI mix")
    print(f"  was unfavorable). Verify by checking if top players over-played in")
    print(f"  states where rates are intrinsically lower than league average.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
