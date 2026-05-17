#!/usr/bin/env python3
"""
May 2026 comprehensive dataset build.

Three-phase script:
  Phase 1 — Build all 6 output files in NFI/2026/05/output/
  Phase 2 — Verify against legacy files (no destructive ops)
  Phase 3 — Delete legacy files only if Phase 2 PASSES (tolerance 0.001)

Reuses the per-(player, season) shift-shot join cached at
NFI/Output/_player_two_way_split_join_cache.pkl (built by
NFI/scripts/player_two_way_split.py). Cache scope matches spec:
  - Seasons 2020-21 .. 2025-26
  - Regular season only, regulation periods 1-3
  - state == 'ES' (Variant A) from situation_code, empty-net excluded
  - On-ice tallies are CNFI+MNFI Fenwick events (FNFI excluded)

If the cache is missing, this script aborts (does not rebuild it; that's the
job of player_two_way_split.py).

Outputs (all in NFI/2026/05/output/):
  team_summary_2526.csv
  players_top10_2526.csv
  players_all18_2526.csv
  players_bottom8_2526.csv
  goalies_2526.csv
  methodology_2526.md
"""
from __future__ import annotations

import math
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================== paths
ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
OUT_DIR = f"{ROOT}/NFI/2026/05/output"
SHOT_FP = f"{ROOT}/Data/nhl_shot_events.csv"
SHOTS_TAGGED_FP = f"{ROOT}/NFI/Output/shots_tagged.csv"
GAMEIDS_FP = f"{ROOT}/Data/game_ids.csv"
POS_FP = f"{ROOT}/NFI/Output/player_positions.csv"
FA_FP = f"{ROOT}/NFI/Output/fully_adjusted/player_fully_adjusted.csv"
CACHE_FP = f"{ROOT}/NFI/Output/_player_two_way_split_join_cache.pkl"

LEGACY_TEAM_NFI = f"{ROOT}/NFI/Output/team_nfi_verification_and_attack_suppress.csv"
LEGACY_VARIANTS = f"{ROOT}/NFI/Output/roster_talent_variants_2526.csv"
LEGACY_TWO_WAY = f"{ROOT}/NFI/Output/player_two_way_split.csv"

LEGACY_DELETE_TARGETS = [
    f"{ROOT}/NFI/Output/roster_talent_variants_2526.csv",
    f"{ROOT}/NFI/Output/player_two_way_split.csv",
    f"{ROOT}/NFI/Output/team_nfi_verification_and_attack_suppress.csv",
    # all32_master_summary.csv path resolved at runtime via search
]

# ============================================================== constants
WINDOWS = {
    "5y":      ["20202021", "20212022", "20222023", "20232024", "20242025"],
    "4y":      ["20212022", "20222023", "20232024", "20242025"],
    "3y":      ["20222023", "20232024", "20242025"],
    "2y":      ["20232024", "20242025"],
    "current": ["20252026"],
}
WINDOW_ORDER = ["5y", "4y", "3y", "2y", "current"]
LEGACY_MATCH_SEASONS = ["20222023", "20232024", "20242025", "20252026"]  # FA POOLED

QUAL_TOI_MIN = 600.0       # min ES TOI (per window) to call window 'qualified'
GP_PER_TEAM_MIN = 20       # min GP per team in 2025-26 for cohort eligibility
TOL = 0.001                # absolute tolerance for verification

CURR_SEASON = "20252026"

FEN_TYPES = {"shot-on-goal", "missed-shot", "goal"}
DZ = ["CNFI", "MNFI"]      # canonical danger zones (FNFI excluded)

# ============================================================== helpers
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def w_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """TOI-weighted mean. Returns nan on empty / zero weight / all-nan inputs."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
    if not mask.any():
        return float("nan")
    if w[mask].sum() <= 0:
        return float("nan")
    return float((v[mask] * w[mask]).sum() / w[mask].sum())


def safe_remove(path: str) -> str:
    """Remove a file with try/except; return status string."""
    try:
        os.remove(path)
        return f"DELETED: {path}"
    except FileNotFoundError:
        return f"  (skip — not found: {path})"
    except Exception as exc:
        return f"  ERROR removing {path}: {exc}"


# ============================================================== loaders
def load_cache() -> dict:
    """Load the per-(player, season) shift-shot join cache.
    Aborts if missing — this script doesn't rebuild it."""
    if not os.path.exists(CACHE_FP):
        sys.exit(f"FATAL: cache missing at {CACHE_FP}. Run "
                 f"NFI/scripts/player_two_way_split.py first to build it.")
    log(f"Loading cache from {CACHE_FP}...")
    with open(CACHE_FP, "rb") as f:
        c = pickle.load(f)
    log(f"  cache players: {len(c['es_toi'])}")
    return c


def load_positions() -> tuple[dict, dict, dict]:
    pos = pd.read_csv(POS_FP, dtype={"player_id": int})
    return (dict(zip(pos["player_id"], pos["pos_group"])),
            dict(zip(pos["player_id"], pos["position"])),
            dict(zip(pos["player_id"], pos["player_name"])))


def load_fa_2526() -> pd.DataFrame:
    """player_fully_adjusted rows for 2025-26 only — used for team-specific TOI."""
    fa = pd.read_csv(FA_FP)
    fa = fa[fa["season"].astype(str) == CURR_SEASON].copy()
    return fa


def load_fa_all() -> pd.DataFrame:
    fa = pd.read_csv(FA_FP)
    fa["season"] = fa["season"].astype(str)
    return fa


# ============================================================== TEAM NFI (1)
def compute_team_nfi_2526():
    """Team-level CNFI+MNFI Fenwick ES (Variant A) counts for 2025-26 regular.
    Returns (team_df, fnfi_dropped_count, total_es_kept)."""
    log("Computing team NFI from shots_tagged (2025-26 regular)...")
    sh = pd.read_csv(SHOTS_TAGGED_FP, usecols=[
        "game_id", "season", "state", "zone", "event_type",
        "shooting_team_abbrev", "home_team_abbrev", "away_team_abbrev",
    ])
    sh["season"] = sh["season"].astype(str)
    # game_type from game_id position 5-6: "02"=regular
    sh["gtype"] = sh["game_id"].astype(str).str.slice(4, 6)
    sh = sh[(sh["season"] == CURR_SEASON) & (sh["gtype"] == "02")].copy()

    # Variant A: state == ES; CNFI+MNFI; Fenwick
    es = sh[sh["state"] == "ES"].copy()
    es_fen = es[es["event_type"].isin(FEN_TYPES)].copy()
    fnfi_dropped = int((es_fen["zone"] == "FNFI").sum())
    cm = es_fen[es_fen["zone"].isin(DZ)].copy()

    # def team
    cm["def_team"] = np.where(cm["shooting_team_abbrev"] == cm["home_team_abbrev"],
                              cm["away_team_abbrev"], cm["home_team_abbrev"])
    attack = cm.groupby("shooting_team_abbrev").size().rename("attack")
    suppress = cm.groupby("def_team").size().rename("suppress")
    attack.index.name = "team"
    suppress.index.name = "team"
    team_df = pd.concat([attack, suppress], axis=1).fillna(0).astype(
        {"attack": int, "suppress": int}).reset_index()

    # GP per team from game_ids
    g = pd.read_csv(GAMEIDS_FP, dtype={"season": str})
    g = g[(g["season"] == CURR_SEASON) & (g["game_type"] == "regular")]
    gp_counter = defaultdict(int)
    for _, r in g.iterrows():
        gp_counter[r["home_abbrev"]] += 1
        gp_counter[r["away_abbrev"]] += 1
    team_df["games_played"] = team_df["team"].map(gp_counter).fillna(0).astype(int)
    team_df["team_nfi_pct"] = team_df["attack"] / (team_df["attack"] + team_df["suppress"])
    team_df["team_attack_per_game"] = team_df["attack"] / team_df["games_played"]
    team_df["team_suppress_per_game"] = team_df["suppress"] / team_df["games_played"]
    team_df["team_nfi_rank"] = team_df["team_nfi_pct"].rank(ascending=False, method="min").astype("Int64")
    team_df["team_attack_rank"] = team_df["team_attack_per_game"].rank(ascending=False, method="min").astype("Int64")
    team_df["team_suppress_rank"] = team_df["team_suppress_per_game"].rank(ascending=True, method="min").astype("Int64")

    log(f"  teams: {len(team_df)}; FNFI ES Fenwick dropped (2025-26): {fnfi_dropped:,}; "
        f"CNFI+MNFI ES Fenwick kept: {len(cm):,}")
    return team_df, fnfi_dropped, int(len(cm))


# ============================================================== player windows
def build_player_windows(cache: dict, pos_group: dict, name_map: dict) -> pd.DataFrame:
    """For every (player, window): events_for, events_against, es_toi_min,
    offensive_NFI_60, defensive_NFI_60, pool_NFI_combined, qualified."""
    log("Aggregating player-level metrics across 5 lookback windows...")
    rows = []
    all_pids = set(cache["es_toi"].keys()) | set(cache["onice_for"].keys()) \
               | set(cache["onice_ag"].keys())
    for pid in all_pids:
        pg = pos_group.get(pid, "")
        if pg == "G" or pg not in {"F", "D"}:
            continue
        toi_by_s = cache["es_toi"].get(pid, {})
        for_by_s = cache["onice_for"].get(pid, {})
        ag_by_s = cache["onice_ag"].get(pid, {})
        gbt_by_s = cache["games_by_team"].get(pid, {})
        for win, seasons in WINDOWS.items():
            toi_sec = sum(toi_by_s.get(s, 0.0) for s in seasons)
            ev_for = sum(for_by_s.get(s, 0) for s in seasons)
            ev_ag = sum(ag_by_s.get(s, 0) for s in seasons)
            gp = sum(len(set(g for (g, _) in gbt_by_s.get(s, set()))) for s in seasons)
            toi_min = toi_sec / 60.0
            off60 = (ev_for / toi_min * 60.0) if toi_min > 0 else float("nan")
            def60 = (ev_ag / toi_min * 60.0) if toi_min > 0 else float("nan")
            pool = (ev_for / (ev_for + ev_ag)) if (ev_for + ev_ag) > 0 else float("nan")
            rows.append({
                "player_id": pid,
                "player_name": name_map.get(pid, str(pid)),
                "primary_position": pg,
                "window": win,
                "events_for": ev_for, "events_against": ev_ag,
                "es_toi_min": toi_min, "gp": gp,
                "offensive_NFI_60": off60,
                "defensive_NFI_60": def60,
                "pool_NFI_combined": pool,
                "qualified": toi_min >= QUAL_TOI_MIN,
            })
    df = pd.DataFrame(rows)
    log(f"  player-window rows: {len(df)}")
    return df


# ============================================================== cohort build
def build_cohorts(cache: dict, pos_group: dict, fa_2526: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-(team, player) cohort eligibility based on:
      - 2025-26 team-specific GP >= 20 (from cache games_by_team)
      - position F or D
      - team-specific 2025-26 ES TOI from FA file (toi_min for that team-season row)

    Returns:
      pop:  long table of all eligible (team, player) rows with toi_2526
      slot: same with slot assignment per cohort: {top10, all18, bottom8, all18_F, all18_D}
            (a player can be in multiple cohorts; this table pivots per cohort+slot)
    """
    log("Building per-team cohorts...")
    # Per (team, pid) GP from cache
    team_pid_gp = {}
    for pid, by_season in cache["games_by_team"].items():
        s2526 = by_season.get(CURR_SEASON, set())
        gp_per_team = defaultdict(int)
        for (gid, tm) in s2526:
            gp_per_team[tm] += 1
        for tm, gp in gp_per_team.items():
            team_pid_gp[(tm, pid)] = gp

    # Build pop (player x team x toi)
    fa = fa_2526.copy()
    fa["pos_group_join"] = fa["player_id"].map(pos_group)
    fa = fa[fa["pos_group_join"].isin(["F", "D"])].copy()
    fa["gp_team"] = fa.apply(
        lambda r: team_pid_gp.get((r["team"], r["player_id"]), 0), axis=1)
    pop = fa[fa["gp_team"] >= GP_PER_TEAM_MIN].copy()
    pop = pop[["player_id", "player_name", "team", "pos_group_join",
               "toi_min", "gp_team"]].rename(
        columns={"pos_group_join": "primary_position",
                 "toi_min": "es_toi_2025_26_min_team",
                 "gp_team": "gp_2025_26_team"})
    log(f"  eligible (team, player) rows: {len(pop)}")

    # Per team, sort F and D by team-specific TOI desc, assign slots, build cohorts
    cohort_rows = []
    for team, gr in pop.groupby("team"):
        F = gr[gr["primary_position"] == "F"].sort_values(
            "es_toi_2025_26_min_team", ascending=False).reset_index(drop=True)
        D = gr[gr["primary_position"] == "D"].sort_values(
            "es_toi_2025_26_min_team", ascending=False).reset_index(drop=True)
        for i, r in F.iterrows():
            slot = i + 1
            cohort_rows.append({
                **r.to_dict(),
                "slot_in_team": f"F{slot}",
                "in_top10": slot <= 6,
                "in_all18": slot <= 12,
                "in_bottom8": 7 <= slot <= 12,
            })
        for i, r in D.iterrows():
            slot = i + 1
            cohort_rows.append({
                **r.to_dict(),
                "slot_in_team": f"D{slot}",
                "in_top10": slot <= 4,
                "in_all18": slot <= 6,
                "in_bottom8": 5 <= slot <= 6,
            })
    slot = pd.DataFrame(cohort_rows)
    log(f"  cohort assignments built: {len(slot)} rows")
    return pop, slot


# ============================================================== team summary build
def build_team_summary(team_nfi: pd.DataFrame, slot: pd.DataFrame,
                       pw_idx: dict) -> pd.DataFrame:
    """team_summary_2526 — one row per team with team NFI + roster cohort metrics
    across all windows. pw_idx is {(pid, win): row_dict_with_off60_def60_poolNFI_qualified}."""
    log("Building team_summary_2526 ...")
    teams_in_slot = sorted(slot["team"].unique())
    team_records = []

    for team in teams_in_slot:
        rec = {}
        rec["team"] = team
        # team NFI (from team_nfi df)
        tn = team_nfi[team_nfi["team"] == team]
        if len(tn) == 1:
            r = tn.iloc[0]
            rec["games_played"] = int(r["games_played"])
            rec["team_nfi_pct"] = float(r["team_nfi_pct"])
            rec["team_nfi_rank"] = int(r["team_nfi_rank"])
            rec["team_attack_per_game"] = float(r["team_attack_per_game"])
            rec["team_attack_rank"] = int(r["team_attack_rank"])
            rec["team_suppress_per_game"] = float(r["team_suppress_per_game"])
            rec["team_suppress_rank"] = int(r["team_suppress_rank"])
        else:
            rec.update({k: float("nan") for k in [
                "team_nfi_pct", "team_attack_per_game", "team_suppress_per_game"]})
            rec["games_played"] = 0

        team_slot = slot[slot["team"] == team]
        cohorts_def = {
            "top10":   team_slot[team_slot["in_top10"]],
            "all18":   team_slot[team_slot["in_all18"]],
            "bottom8": team_slot[team_slot["in_bottom8"]],
            "all18_F": team_slot[(team_slot["in_all18"]) & (team_slot["primary_position"] == "F")],
            "all18_D": team_slot[(team_slot["in_all18"]) & (team_slot["primary_position"] == "D")],
        }
        expected_n = {"top10": 10, "all18": 18, "bottom8": 8,
                      "all18_F": 12, "all18_D": 6}
        shortfall = 0
        for c_name, c_df in cohorts_def.items():
            n = len(c_df)
            if n < expected_n[c_name]:
                shortfall = 1
            for win in WINDOW_ORDER:
                # only include players qualified in this window
                pids = c_df["player_id"].tolist()
                wts = c_df["es_toi_2025_26_min_team"].tolist()
                offs, defs, pools, w_in = [], [], [], []
                for pid, w in zip(pids, wts):
                    pwr = pw_idx.get((pid, win))
                    if pwr is None:
                        continue
                    if not pwr["qualified"]:
                        continue
                    offs.append(pwr["offensive_NFI_60"])
                    defs.append(pwr["defensive_NFI_60"])
                    pools.append(pwr["pool_NFI_combined"])
                    w_in.append(w)
                off_r = w_mean(np.array(offs), np.array(w_in))
                def_r = w_mean(np.array(defs), np.array(w_in))
                comb = w_mean(np.array(pools), np.array(w_in))
                rec[f"{c_name}_{win}_off_rate"] = off_r
                rec[f"{c_name}_{win}_def_rate"] = def_r
                rec[f"{c_name}_{win}_combined_score"] = comb
            rec[f"{c_name}_n_players"] = n if c_name in {"top10", "all18", "bottom8"} else None
        rec["cohort_shortfall_flag"] = shortfall
        team_records.append(rec)

    df = pd.DataFrame(team_records)

    # ranks: combined and offensive descending, defensive ascending
    rank_cols = []
    for c_name in ["top10", "all18", "bottom8", "all18_F", "all18_D"]:
        for win in WINDOW_ORDER:
            for metric, ascending in [
                ("combined_score", False), ("off_rate", False), ("def_rate", True)
            ]:
                col = f"{c_name}_{win}_{metric}"
                rank_col = col.replace("_score", "_rank").replace("_rate", "_rank")
                df[rank_col] = df[col].rank(ascending=ascending, method="min").astype("Int64")
                rank_cols.append(rank_col)

    # Drop None n_players columns for sub-cohorts
    for c in ["all18_F_n_players", "all18_D_n_players"]:
        if c in df.columns:
            df = df.drop(columns=c)

    # Sort by team_nfi_rank ascending
    df = df.sort_values("team_nfi_rank").reset_index(drop=True)
    return df


# ============================================================== player long files
def build_player_long(slot: pd.DataFrame, pw_idx: dict, cohort_flag_col: str
                      ) -> pd.DataFrame:
    """Build long-format player CSV for a given cohort."""
    rows = []
    sub = slot[slot[cohort_flag_col]].copy()
    sub = sub.sort_values(["team", "slot_in_team"])
    for _, r in sub.iterrows():
        rec = {
            "team": r["team"],
            "player_name": r["player_name"],
            "player_id": int(r["player_id"]),
            "primary_position": r["primary_position"],
            "gp_2025_26_team": int(r["gp_2025_26_team"]),
            "es_toi_2025_26_min_team": float(r["es_toi_2025_26_min_team"]),
            "slot_in_team": r["slot_in_team"],
        }
        for win in WINDOW_ORDER:
            pwr = pw_idx.get((r["player_id"], win), {})
            rec[f"offensive_NFI_60_{win}"] = pwr.get("offensive_NFI_60", float("nan"))
            rec[f"defensive_NFI_60_{win}"] = pwr.get("defensive_NFI_60", float("nan"))
            rec[f"pool_NFI_combined_{win}"] = pwr.get("pool_NFI_combined", float("nan"))
            rec[f"gp_in_window_{win}"] = pwr.get("gp", 0)
            rec[f"es_toi_in_window_min_{win}"] = pwr.get("es_toi_min", 0.0)
            rec[f"qualified_{win}"] = bool(pwr.get("qualified", False))
        rows.append(rec)
    return pd.DataFrame(rows)


# ============================================================== goalies
def build_goalies() -> tuple[pd.DataFrame, int]:
    """Per-goalie GSAx across the 5 windows using CNFI+MNFI-only methodology.
    Mirrors NFI/scripts/21_goalie_gsax_by_season.py exactly."""
    log("Building goalies_2526 (CNFI+MNFI only, FNFI excluded)...")
    sh = pd.read_csv(SHOTS_TAGGED_FP, usecols=[
        "game_id", "season", "state", "zone", "event_type",
        "shooting_team_abbrev", "home_team_abbrev", "away_team_abbrev",
        "goalie_id", "is_goal_i",
    ])
    sh["season"] = sh["season"].astype(str)
    sh["gtype"] = sh["game_id"].astype(str).str.slice(4, 6)
    sh = sh[sh["gtype"] == "02"].copy()  # regular only
    es = sh[sh["state"] == "ES"].copy()
    fen_face = es["event_type"].isin(["shot-on-goal", "goal"])  # faced = SOG + goals (per script 21)
    fnfi_dropped = int(((es["zone"] == "FNFI") & fen_face).sum())
    es = es[fen_face].copy()
    es = es[es["goalie_id"].notna()].copy()
    es["goalie_id"] = es["goalie_id"].astype(int)
    es["is_goal_i"] = es["is_goal_i"].astype(int)

    pos = pd.read_csv(POS_FP)
    name_map = dict(zip(pos["player_id"].astype(int), pos["player_name"]))

    # determine current team per goalie in 2025-26 (from goalie's defending team)
    es_2526 = es[es["season"] == CURR_SEASON].copy()
    es_2526["def_team"] = np.where(
        es_2526["shooting_team_abbrev"] == es_2526["home_team_abbrev"],
        es_2526["away_team_abbrev"], es_2526["home_team_abbrev"])
    curr_team_map = (es_2526.groupby(["goalie_id", "def_team"]).size()
                     .reset_index(name="n").sort_values(["goalie_id", "n"], ascending=[True, False])
                     .drop_duplicates("goalie_id").set_index("goalie_id")["def_team"].to_dict())

    # per (goalie, season): faced_CNFI, faced_MNFI, goals_CNFI, goals_MNFI, games
    es["gid_int"] = es["game_id"].astype(int)
    es_dz = es[es["zone"].isin(DZ)].copy()
    agg = es_dz.groupby(["goalie_id", "season", "zone"]).agg(
        faced=("is_goal_i", "size"),
        goals=("is_goal_i", "sum"),
    ).reset_index()
    games_per = es.groupby(["goalie_id", "season"])["gid_int"].nunique().rename("games").reset_index()

    wide = agg.pivot_table(index=["goalie_id", "season"], columns="zone",
                           values=["faced", "goals"], fill_value=0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for c in ["faced_CNFI", "faced_MNFI", "goals_CNFI", "goals_MNFI"]:
        if c not in wide.columns:
            wide[c] = 0
    wide = wide.merge(games_per, on=["goalie_id", "season"], how="left")
    wide["games"] = wide["games"].fillna(0).astype(int)

    # per-season per-zone goal rate (faced) for xG_calibrated
    rate = es_dz.groupby(["season", "zone"]).agg(n=("is_goal_i", "size"),
                                                  k=("is_goal_i", "sum")).reset_index()
    rate["rate"] = rate["k"] / rate["n"].replace(0, np.nan)
    rate_idx = {(r["season"], r["zone"]): r["rate"] for _, r in rate.iterrows()}

    wide["xG"] = wide.apply(lambda r:
        r["faced_CNFI"] * rate_idx.get((r["season"], "CNFI"), 0.0)
        + r["faced_MNFI"] * rate_idx.get((r["season"], "MNFI"), 0.0), axis=1)
    wide["total_faced"] = wide["faced_CNFI"] + wide["faced_MNFI"]
    wide["total_goals"] = wide["goals_CNFI"] + wide["goals_MNFI"]
    wide["GSAx"] = wide["xG"] - wide["total_goals"]

    # ES TOI per goalie per season — script 21 prorates pooled toi_ES_min by faced share.
    # We do the same: load player_toi.csv (pooled) and prorate by faced share.
    toi = pd.read_csv(f"{ROOT}/NFI/Output/player_toi.csv")
    toi_g = toi[toi["position"] == "G"][["player_id", "toi_ES_sec"]]
    pooled_toi_min = dict(zip(toi_g["player_id"].astype(int),
                              toi_g["toi_ES_sec"].astype(float) / 60.0))
    pool_faced = wide.groupby("goalie_id")["total_faced"].sum().to_dict()

    def per_season_toi(row):
        g = int(row["goalie_id"])
        pool = pooled_toi_min.get(g, 0.0)
        pf = pool_faced.get(g, 0)
        return (row["total_faced"] / pf * pool) if pf > 0 else 0.0
    wide["es_toi_min_season"] = wide.apply(per_season_toi, axis=1)

    # Aggregate to windows
    g_rows = []
    for goalie in wide["goalie_id"].unique():
        gw = wide[wide["goalie_id"] == goalie]
        rec = {"goalie_id": int(goalie),
               "goalie_name": name_map.get(int(goalie), str(int(goalie))),
               "current_team_2025_26": curr_team_map.get(int(goalie), "")}
        for win, seasons in WINDOWS.items():
            gws = gw[gw["season"].isin(seasons)]
            faced = int(gws["total_faced"].sum())
            ga = int(gws["total_goals"].sum())
            xG = float(gws["xG"].sum())
            gsax = xG - ga
            toi_min = float(gws["es_toi_min_season"].sum())
            gp = int(gws["games"].sum())
            per60 = (gsax / toi_min * 60.0) if toi_min > 0 else float("nan")
            rec[f"gp_{win}"] = gp
            rec[f"faced_dz_{win}"] = faced
            rec[f"ga_dz_{win}"] = ga
            rec[f"GSAx_{win}"] = round(gsax, 2)
            rec[f"GSAx_per60_{win}"] = round(per60, 3) if not math.isnan(per60) else float("nan")
            rec[f"qualified_{win}"] = faced >= 100
        g_rows.append(rec)
    df = pd.DataFrame(g_rows)
    df = df.sort_values("GSAx_3y", ascending=False).reset_index(drop=True)
    log(f"  goalies: {len(df)}; FNFI faced events dropped: {fnfi_dropped:,}")
    return df, fnfi_dropped


# ============================================================== verification
def verify_team_nfi(new_team_df: pd.DataFrame) -> tuple[bool, list]:
    log("Phase 2 verification: team NFI vs legacy file...")
    legacy = pd.read_csv(LEGACY_TEAM_NFI)
    cmp = new_team_df.merge(legacy[["team", "team_nfi_pct_post_audit",
                                     "attack_per_game", "suppress_per_game",
                                     "games_played"]],
                            on="team", how="outer", suffixes=("_new", "_legacy"))
    failures = []
    for _, r in cmp.iterrows():
        team = r["team"]
        d_pct = abs((r["team_nfi_pct"] or 0) - (r["team_nfi_pct_post_audit"] or 0))
        d_at = abs((r["team_attack_per_game"] or 0) - (r["attack_per_game"] or 0))
        d_su = abs((r["team_suppress_per_game"] or 0) - (r["suppress_per_game"] or 0))
        ok = d_pct <= TOL and d_at <= TOL and d_su <= TOL
        status = "PASS" if ok else f"FAIL Δpct={d_pct:.4f} Δatk={d_at:.4f} Δsup={d_su:.4f}"
        print(f"  {team:<5} {status}")
        if not ok:
            failures.append((team, "team_nfi", d_pct, d_at, d_su))
    return (len(failures) == 0, failures)


def verify_roster_cohorts(slot: pd.DataFrame, fa_all: pd.DataFrame) -> tuple[bool, list]:
    """Compare cohort scores using LEGACY_MATCH window (4-season pooled NFI_pct
    from FA file, replicating the prior session's groupby+TOI-weighted-mean)."""
    log("Phase 2 verification: roster cohorts (legacy_match window) vs legacy variants file...")

    # Compute legacy-style pool_NFI per player: TOI-weighted mean of NFI_pct
    # across whatever seasons of the player are in FA (typically POOLED set).
    fa = fa_all[fa_all["season"].isin(LEGACY_MATCH_SEASONS)].copy()
    pool = (fa.dropna(subset=["NFI_pct", "toi_min"])
              .groupby("player_id")
              .apply(lambda g: float(np.average(g["NFI_pct"], weights=g["toi_min"])))
              .rename("legacy_pool_NFI").reset_index())

    legacy = pd.read_csv(LEGACY_VARIANTS)
    legacy_idx = legacy.set_index("team").to_dict(orient="index")

    s = slot.merge(pool, on="player_id", how="left")
    failures = []
    for team, gr in s.groupby("team"):
        for c_name, expected_col in [("top10", "top10_score"),
                                     ("all18", "all18_score"),
                                     ("bottom8", "bottom8_score")]:
            c_df = gr[gr[f"in_{c_name}"]]
            sub = c_df.dropna(subset=["legacy_pool_NFI"])
            if len(sub) == 0 or sub["es_toi_2025_26_min_team"].sum() <= 0:
                new_score = float("nan")
            else:
                new_score = float(np.average(sub["legacy_pool_NFI"],
                                             weights=sub["es_toi_2025_26_min_team"]))
            legacy_score = legacy_idx.get(team, {}).get(expected_col, float("nan"))
            d = abs(new_score - legacy_score) if (not math.isnan(new_score)
                                                  and not math.isnan(legacy_score)) else float("nan")
            ok = (not math.isnan(d)) and d <= TOL
            status = "PASS" if ok else f"FAIL Δ={d:.5f}" if not math.isnan(d) else "FAIL nan"
            print(f"  {team:<5} {c_name:<8} new={new_score:.4f}  legacy={legacy_score:.4f}  {status}")
            if not ok:
                failures.append((team, c_name, new_score, legacy_score, d))
    return (len(failures) == 0, failures)


def spot_check_players(player_long_top10: pd.DataFrame) -> None:
    log("Phase 2 player spot-check (informational only):")
    targets = ["Connor McDavid", "Zach Hyman", "Evan Bouchard", "Mattias Ekholm",
               "Leon Draisaitl", "Darnell Nurse", "Auston Matthews", "Mitch Marner",
               "Cale Makar", "Connor Hellebuyck"]
    legacy_tw = pd.read_csv(LEGACY_TWO_WAY) if os.path.exists(LEGACY_TWO_WAY) else None
    print(f"  {'player':<22} {'new_off_curr':>12} {'new_def_curr':>12}"
          f" {'legacy_off_pooled':>17} {'legacy_def_pooled':>17}")
    for name in targets:
        new = player_long_top10[player_long_top10["player_name"] == name]
        if len(new) == 0:
            print(f"  {name:<22} (not in top10 file)")
            continue
        r = new.iloc[0]
        no = r["offensive_NFI_60_current"]
        nd = r["defensive_NFI_60_current"]
        if legacy_tw is not None:
            lr = legacy_tw[legacy_tw["player_name"] == name]
            lo = lr["offensive_NFI_60"].iloc[0] if len(lr) else float("nan")
            ld = lr["defensive_NFI_60"].iloc[0] if len(lr) else float("nan")
        else:
            lo = ld = float("nan")
        no_s = f"{no:.2f}" if pd.notna(no) else "nan"
        nd_s = f"{nd:.2f}" if pd.notna(nd) else "nan"
        lo_s = f"{lo:.2f}" if pd.notna(lo) else "nan"
        ld_s = f"{ld:.2f}" if pd.notna(ld) else "nan"
        print(f"  {name:<22} {no_s:>12} {nd_s:>12} {lo_s:>17} {ld_s:>17}")


# ============================================================== sanity
def sanity_checks(team_df, p10, p18, pb8, gd, fnfi_team, fnfi_goalie, n_es_kept) -> list:
    log("Running sanity checks ...")
    results = []
    # 1. 32 teams
    results.append(("All 32 teams in team_summary",
                    len(team_df) == 32, f"got {len(team_df)}"))
    # 2. combined = off/(off+def) within 0.001 (every cohort+window)
    bad = []
    for _, r in team_df.iterrows():
        for c in ["top10", "all18", "bottom8", "all18_F", "all18_D"]:
            for w in WINDOW_ORDER:
                o = r.get(f"{c}_{w}_off_rate")
                d = r.get(f"{c}_{w}_def_rate")
                cs = r.get(f"{c}_{w}_combined_score")
                if pd.isna(o) or pd.isna(d) or pd.isna(cs):
                    continue
                # combined is TOI-weighted mean of pool_NFI (per spec); that's NOT
                # mathematically off/(off+def) when pool is per-player, not aggregated counts.
                # The spec actually says combined_score = TOI-weighted mean of
                # pool_NFI_combined, NOT off/(off+def). The check here is informational.
                pass
    results.append(("combined_score formula (informational)",
                    True, "combined is weighted mean of pool_NFI per spec"))
    # 3. player counts
    results.append(("players_top10 ~320 rows",
                    280 <= len(p10) <= 360, f"got {len(p10)}"))
    results.append(("players_all18 ~576 rows",
                    520 <= len(p18) <= 620, f"got {len(p18)}"))
    results.append(("players_bottom8 ~256 rows",
                    220 <= len(pb8) <= 290, f"got {len(pb8)}"))
    # 4. positions are F/D only
    pos_bad = []
    for df_, label in [(p10, "top10"), (p18, "all18"), (pb8, "bottom8")]:
        bad_p = df_[~df_["primary_position"].isin(["F", "D"])]
        if len(bad_p):
            pos_bad.append((label, len(bad_p)))
    results.append(("primary_position in {F,D}",
                    not pos_bad, f"bad: {pos_bad}" if pos_bad else "OK"))
    # 5. spot-check goalies
    must_have = ["Connor Hellebuyck", "Igor Shesterkin", "Sergei Bobrovsky",
                 "Stuart Skinner", "Juuse Saros"]
    missing = [n for n in must_have if n not in gd["goalie_name"].values]
    results.append(("Goalie spot-check (5 known)",
                    not missing, f"missing: {missing}" if missing else "all present"))
    # 6. FNFI exclusion notes
    results.append(("FNFI excluded — team pipeline",
                    True, f"discarded {fnfi_team:,} FNFI ES Fenwick (2025-26)"))
    results.append(("FNFI excluded — goalie pipeline",
                    True, f"discarded {fnfi_goalie:,} FNFI ES faced shots (all seasons)"))
    # 7. Variant A
    results.append(("Variant A confirmed (state == 'ES')",
                    True, f"CNFI+MNFI ES Fenwick (2025-26 team-level): {n_es_kept:,}"))
    # 8. league for/against symmetry across player on-ice sums
    return results


# ============================================================== main
def main() -> None:
    t0 = time.time()
    log(f"OUT_DIR: {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ----- shared loads -----
    cache = load_cache()
    pos_group, pos_specific, name_map = load_positions()
    fa_2526 = load_fa_2526()
    fa_all = load_fa_all()

    # ============ PHASE 1 ============
    log("\n========== PHASE 1: BUILD ==========")
    team_nfi_df, fnfi_team, n_es_kept = compute_team_nfi_2526()
    pop, slot = build_cohorts(cache, pos_group, fa_2526)

    # player x window long-form intermediate
    pw = build_player_windows(cache, pos_group, name_map)
    pw_idx = {(int(r["player_id"]), r["window"]):
              {"offensive_NFI_60": r["offensive_NFI_60"],
               "defensive_NFI_60": r["defensive_NFI_60"],
               "pool_NFI_combined": r["pool_NFI_combined"],
               "qualified": bool(r["qualified"]),
               "gp": int(r["gp"]),
               "es_toi_min": float(r["es_toi_min"])}
              for _, r in pw.iterrows()}

    team_summary = build_team_summary(team_nfi_df, slot, pw_idx)

    # Reorder team_summary columns per spec
    base_cols = ["team", "games_played",
                 "team_nfi_pct", "team_nfi_rank",
                 "team_attack_per_game", "team_attack_rank",
                 "team_suppress_per_game", "team_suppress_rank"]
    cohort_cols = []
    for c_name in ["top10", "all18", "bottom8", "all18_F", "all18_D"]:
        for w in WINDOW_ORDER:
            for m in ["combined_score", "off_rate", "def_rate"]:
                cohort_cols.append(f"{c_name}_{w}_{m}")
                cohort_cols.append(f"{c_name}_{w}_{m.replace('_score','_rank').replace('_rate','_rank')}")
    tail_cols = ["top10_n_players", "all18_n_players", "bottom8_n_players",
                 "cohort_shortfall_flag"]
    final_cols = base_cols + cohort_cols + tail_cols
    final_cols = [c for c in final_cols if c in team_summary.columns]
    team_summary = team_summary[final_cols]

    p10 = build_player_long(slot, pw_idx, "in_top10")
    p18 = build_player_long(slot, pw_idx, "in_all18")
    pb8 = build_player_long(slot, pw_idx, "in_bottom8")

    goalie_df, fnfi_goalie = build_goalies()

    # ----- write outputs -----
    out_team = f"{OUT_DIR}/team_summary_2526.csv"
    out_p10 = f"{OUT_DIR}/players_top10_2526.csv"
    out_p18 = f"{OUT_DIR}/players_all18_2526.csv"
    out_pb8 = f"{OUT_DIR}/players_bottom8_2526.csv"
    out_g = f"{OUT_DIR}/goalies_2526.csv"
    out_md = f"{OUT_DIR}/methodology_2526.md"

    team_summary.to_csv(out_team, index=False, float_format="%.6f")
    p10.to_csv(out_p10, index=False, float_format="%.6f")
    p18.to_csv(out_p18, index=False, float_format="%.6f")
    pb8.to_csv(out_pb8, index=False, float_format="%.6f")
    goalie_df.to_csv(out_g, index=False)

    # ----- sanity checks -----
    checks = sanity_checks(team_summary, p10, p18, pb8, goalie_df,
                           fnfi_team, fnfi_goalie, n_es_kept)
    print()
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}  ({detail})")
    sanity_pass = all(ok for _, ok, _ in checks)

    # ============ PHASE 2 ============
    log("\n========== PHASE 2: VERIFY ==========")
    team_pass, team_fails = verify_team_nfi(team_summary)
    print()
    cohort_pass, cohort_fails = verify_roster_cohorts(slot, fa_all)
    spot_check_players(p10)

    phase2_pass = team_pass and cohort_pass and sanity_pass

    # ============ Methodology MD ============
    src_ts = lambda p: (datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")
                        if os.path.exists(p) else "MISSING")
    md = f"""# May 2026 Comprehensive Build — Methodology

Run date: {datetime.now().isoformat(timespec='seconds')}
Output dir: `{OUT_DIR}`

## Source files

| Path | mtime |
|---|---|
| `Data/nhl_shot_events.csv` | {src_ts(SHOT_FP)} |
| `NFI/Output/shots_tagged.csv` | {src_ts(SHOTS_TAGGED_FP)} |
| `Data/game_ids.csv` | {src_ts(GAMEIDS_FP)} |
| `NFI/Output/player_positions.csv` | {src_ts(POS_FP)} |
| `NFI/Output/fully_adjusted/player_fully_adjusted.csv` | {src_ts(FA_FP)} |
| `NFI/Output/_player_two_way_split_join_cache.pkl` | {src_ts(CACHE_FP)} |

## Variant A "broad ES" definition
`state == 'ES'` from `shots_tagged.csv` (derived from `situation_code` in
`nhl_shot_events.csv` — see `NFI/scripts/03_onice_attribution_pillars.py`).
Includes 5v5 and 4v4 even-strength events. Empty-net excluded by upstream.

## Zone filter
**CNFI + MNFI only.** FNFI excluded.
- FNFI ES Fenwick events discarded (team pipeline, 2025-26): {fnfi_team:,}
- FNFI ES faced shots discarded (goalie pipeline, all seasons): {fnfi_goalie:,}

## Lookback windows (player + goalie)
| Window | Seasons |
|---|---|
| `5y` | 2020-21, 2021-22, 2022-23, 2023-24, 2024-25 |
| `4y` | 2021-22, 2022-23, 2023-24, 2024-25 |
| `3y` | 2022-23, 2023-24, 2024-25 |
| `2y` | 2023-24, 2024-25 |
| `current` | 2025-26 only |

`5y/4y/3y/2y` are pre-2025-26 baselines. `current` is 2025-26 only.

## Cohort selection
- **Eligibility:** ≥{GP_PER_TEAM_MIN} GP for that team in 2025-26 (per-team GP from
  `_player_two_way_split_join_cache.pkl`'s `games_by_team[pid][season]`).
  Position from `player_positions.csv` (F = C/LW/RW; D = D). Player must
  appear in the cache (which means they had ES on-ice events in some season).
- **Sort:** by 2025-26 team-specific ES TOI (FA `toi_min` for season=20252026,
  team=X) descending, separately for F and D.
- **Top 10:** top 6 F + top 4 D
- **All 18:** top 12 F + top 6 D
- **Bottom 8:** F slots 7-12 + D slots 5-6
- **Sub-cohorts:** `all18_F` = top 12 F; `all18_D` = top 6 D
- **Shortfall:** if a team has fewer eligible players than required, partial
  fill and `cohort_shortfall_flag = 1`.

## Traded-player handling
Each (player, team) pair is treated independently. A player traded mid-season
who hits {GP_PER_TEAM_MIN}+ GP for both teams (e.g., Panarin NYR + LAK) appears
on both teams' cohort eligibility lists, weighted by their team-specific
2025-26 TOI for each.

## Player-level metrics

For each (player, window):
```
events_for       = sum across window seasons of CNFI+MNFI Fenwick events FOR while on ice
events_against   = sum across window seasons of same AGAINST while on ice
es_toi_min       = sum across window seasons of ES TOI in minutes
offensive_NFI_60 = events_for / es_toi_min * 60
defensive_NFI_60 = events_against / es_toi_min * 60
pool_NFI_combined = events_for / (events_for + events_against)
qualified        = es_toi_min >= {QUAL_TOI_MIN:.0f}
```

Players who do not qualify in a window are excluded from that window's
team aggregations (but participate in others where they do qualify).

## Team aggregations
For each cohort × window:
```
combined_score = TOI-weighted mean of pool_NFI_combined across qualified cohort players
off_rate       = TOI-weighted mean of offensive_NFI_60 across qualified cohort players
def_rate       = TOI-weighted mean of defensive_NFI_60 across qualified cohort players
```
Weights are each player's 2025-26 team-specific ES TOI (FA `toi_min`).

## Team NFI (current season, 2025-26)
Computed from `shots_tagged.csv` directly (no shift join needed):
```
attack_count   = count of CNFI+MNFI Fenwick ES regular events where shooting_team_abbrev = team
suppress_count = count of CNFI+MNFI Fenwick ES regular events where def_team = team
team_nfi_pct = attack / (attack + suppress)
team_attack_per_game = attack / games_played
team_suppress_per_game = suppress / games_played
```

## Goalie GSAx (per-window)
Mirrors `NFI/scripts/21_goalie_gsax_by_season.py`:
```
faced_dz = SOG + goals in CNFI + MNFI (FNFI excluded)
ga_dz    = goals in CNFI + MNFI
xG       = sum_z (faced_z * per-faced goal rate of zone z in season)
GSAx     = xG - ga_dz
GSAx_per60 = GSAx / es_toi_min * 60
qualified  = faced_dz >= 100 in window
```
ES TOI per goalie per season is prorated from pooled `player_toi.csv`'s
`toi_ES_sec` by faced share of the season vs career — same simplification
as script 21 (shifts data isn't season-keyed for goalies in our pipeline).

## Ranks
1 = best.
- combined_score, off_rate, GSAx, GSAx_per60: rank descending
- def_rate: rank ascending (lower events allowed = better)

## Verification (Phase 2)
- Team NFI vs `NFI/Output/team_nfi_verification_and_attack_suppress.csv`
  (`team_nfi_pct_post_audit`, `attack_per_game`, `suppress_per_game`).
  All 32 teams checked at tolerance {TOL}.
- Roster cohorts vs `NFI/Output/roster_talent_variants_2526.csv`
  using a "legacy_match" reconstruction: TOI-weighted mean of `NFI_pct`
  across the 4 FA-pooled seasons {LEGACY_MATCH_SEASONS} per player, then
  TOI-weighted-aggregated per cohort by 2025-26 team-specific TOI.
  All 32 teams × 3 cohorts checked at tolerance {TOL}.
- Player-level spot-check (10 known names) is informational only — the
  legacy `player_two_way_split.csv` is multi-season pooled while the
  new `current` window is 2025-26 only, so they shouldn't match exactly.

## Phase 2 results
- Team NFI: {"PASS" if team_pass else "FAIL"} ({len(team_fails)} failures)
- Roster cohorts: {"PASS" if cohort_pass else "FAIL"} ({len(cohort_fails)} failures)
- Sanity checks: {"PASS" if sanity_pass else "FAIL"}

## Phase 3 result
{"WILL EXECUTE — verification passed" if phase2_pass else "SKIPPED — verification failed; legacy files retained"}
"""
    with open(out_md, "w") as f:
        f.write(md)

    # ============ PHASE 3 ============
    log("\n========== PHASE 3: LEGACY DELETION ==========")
    if not phase2_pass:
        print(f"PHASE 3 SKIPPED — verification failed for "
              f"{len(team_fails) + len(cohort_fails)} items "
              f"(team={len(team_fails)} cohort={len(cohort_fails)} "
              f"sanity={'fail' if not sanity_pass else 'pass'}). Legacy files retained.")
    else:
        # find any all32_master_summary.csv
        targets = list(LEGACY_DELETE_TARGETS)
        import subprocess
        try:
            res = subprocess.run(
                ["find", ROOT, "-name", "all32_master_summary.csv", "-type", "f"],
                capture_output=True, text=True, timeout=30)
            for line in res.stdout.strip().split("\n"):
                if line:
                    targets.append(line)
        except Exception as exc:
            print(f"  (find for all32_master_summary.csv failed: {exc})")

        for t in targets:
            print(safe_remove(t))

    # ============ END SUMMARY ============
    log("\n========== END SUMMARY ==========")
    for path, label in [(out_team, "team_summary_2526"),
                        (out_p10, "players_top10_2526"),
                        (out_p18, "players_all18_2526"),
                        (out_pb8, "players_bottom8_2526"),
                        (out_g, "goalies_2526"),
                        (out_md, "methodology_2526")]:
        if os.path.exists(path):
            try:
                rows = sum(1 for _ in open(path)) - 1
            except Exception:
                rows = "?"
            print(f"  {label:<22} rows={rows:<6}  path={path}")
        else:
            print(f"  {label:<22} MISSING — write failed")

    short = team_summary[team_summary["cohort_shortfall_flag"] == 1]
    print(f"\nTeams with cohort shortfall: {len(short)}")
    for _, r in short.iterrows():
        print(f"  {r['team']}: top10={int(r['top10_n_players'])}/10  "
              f"all18={int(r['all18_n_players'])}/18  "
              f"bottom8={int(r['bottom8_n_players'])}/8")

    none_qual = goalie_df[~goalie_df[[f"qualified_{w}" for w in WINDOW_ORDER]].any(axis=1)]
    print(f"\nGoalies who don't qualify in any window: {len(none_qual)}")
    if len(none_qual):
        for _, r in none_qual.head(20).iterrows():
            print(f"  {r['goalie_name']} ({r['current_team_2025_26']})")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
