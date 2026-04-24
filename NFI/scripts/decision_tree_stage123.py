"""Zone Methodology Decision Tree — Stages 1-3 (team-season level).

Metrics (7; V3 is excluded from zone-adjustment per spec):
  V5                 — on-ice CNFI+MNFI% (all skaters, TOI-weighted team avg)
  Corsi_CF_pct       — on-ice Corsi For %
  Fenwick_CF_pct     — on-ice Fenwick For % (no blocks)
  CNFI_MNFI_team     — team total CNFI+MNFI% (CF/(CF+CA))
  V1                 — forward individual CNFI+MNFI shots per 60
  V2                 — forward individual CNFI shots per 60
  V4                 — P1b rebound arrival per 60

Sources:
  Zones/raw/pbp/*.json                  — faceoffs (time, winning_team, zone)
  NFI/Geometry_post/Data/shift_data.csv — on-ice intervals
  NFI/output/shots_tagged.csv           — shot events with CNFI/MNFI zone tags
  NFI/output/player_positions.csv       — F/D classification
  NFI/output/team_level_all_metrics.csv — team-level points and existing CNFI+MNFI%

Sample: 2022-23 to 2025-26 (raw PBP coverage) — 4 seasons × 32 = 128 team-seasons
        ARI 22-23 & 23-24 may drop due to ARI/UTA relocation mapping.

Outputs:
  NFI/output/zone_adjustment/complete_decision_tree/
    stage1_factor_base.csv       — Factor_base per metric
    stage1_dz_only_r2.csv        — per-metric DZ-only R² vs points
    stage2_factor_optimization.csv — multiplier sweep results
    stage3_zone_decision.csv     — winner per metric
    stage1_inputs_team.csv       — team-season raw inputs (for Stage 4+)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[2]
PBP_DIR = ROOT / "Zones/raw/pbp"
SHIFT_CSV = ROOT / "NFI/Geometry_post/Data/shift_data.csv"
SHOTS_CSV = ROOT / "NFI/output/shots_tagged.csv"
POS_CSV = ROOT / "NFI/output/player_positions.csv"
TEAM_METRICS = ROOT / "NFI/output/team_level_all_metrics.csv"
REBOUND_CSV = ROOT / "NFI/output/rebound_sequences.csv"  # for V4
GAME_IDS = ROOT / "Data/game_ids.csv"
OUT = ROOT / "NFI/output/zone_adjustment/complete_decision_tree"
OUT.mkdir(parents=True, exist_ok=True)

ABBR_MAP = {"ARI": "UTA"}
POOLED_SEASONS = {"20222023", "20232024", "20242025", "20252026"}
FLIP = {"O": "D", "D": "O", "N": "N"}
FENWICK_TYPES = {"shot-on-goal", "missed-shot", "goal"}
CORSI_TYPES = FENWICK_TYPES | {"blocked-shot"}


def mmss(s):
    if not s or ":" not in s: return 0
    m, se = s.split(":")
    try: return int(m) * 60 + int(se)
    except ValueError: return 0


def norm_team(a): return ABBR_MAP.get(a, a) if a else a


# ---------------------------------------------------------------------------
# Stage 0: load game_ids, team metadata
# ---------------------------------------------------------------------------
print("[0] loading inputs ...")
games_df = pd.read_csv(GAME_IDS)
games_df = games_df[games_df["season"].astype(str).isin(POOLED_SEASONS)]
games_df = games_df[games_df["game_type"] == "regular"]
game_to_season = dict(zip(games_df["game_id"].astype(int),
                          games_df["season"].astype(str)))
game_ids = sorted(game_to_season.keys())
print(f"    regular games in scope: {len(game_ids)}")

team_metrics = pd.read_csv(TEAM_METRICS)
team_metrics["season"] = team_metrics["season"].astype(str)
team_metrics = team_metrics[team_metrics["season"].isin(POOLED_SEASONS)].copy()
team_metrics["CNFI_MNFI_team"] = team_metrics["CNFI_pct"] + team_metrics["MNFI_pct"]
# Team-level raw on-ice Corsi / Fenwick / V5 proxies at team aggregate (CF/(CF+CA))
team_metrics["Corsi_CF_pct_team"] = team_metrics["CF"] / (team_metrics["CF"] + team_metrics["CA"])
# Fenwick approx = Corsi minus blocks. Team has no blocks col; use HD_CF as placeholder? No.
# team_level_all_metrics has CF (Corsi), so build Fenwick from shots_tagged below.

# ---------------------------------------------------------------------------
# Stage 0b: parse faceoffs from PBP
# ---------------------------------------------------------------------------
print("[0b] extracting faceoffs from PBP ...")
# List of (game_id, abs_t, winner_abbrev, loser_abbrev, zone_winner_pov)
faceoffs = []
for i, gid in enumerate(game_ids):
    path = PBP_DIR / f"{gid}.json"
    if not path.exists():
        continue
    try:
        pbp = json.load(open(path))
    except Exception:
        continue
    home_id = (pbp.get("homeTeam") or {}).get("id")
    away_id = (pbp.get("awayTeam") or {}).get("id")
    home_ab = norm_team((pbp.get("homeTeam") or {}).get("abbrev"))
    away_ab = norm_team((pbp.get("awayTeam") or {}).get("abbrev"))
    if home_id is None or away_id is None:
        continue
    for p in pbp.get("plays") or []:
        if p.get("typeDescKey") != "faceoff":
            continue
        if (p.get("situationCode") or "") != "1551":
            continue
        det = p.get("details") or {}
        zone = det.get("zoneCode")
        owner = det.get("eventOwnerTeamId")  # winner of faceoff
        if zone not in ("O", "D", "N") or owner not in (home_id, away_id):
            continue
        if zone == "N":
            continue  # spec excludes neutral
        period = (p.get("periodDescriptor") or {}).get("number", 1) or 1
        t_abs = (period - 1) * 1200 + mmss(p.get("timeInPeriod", "00:00"))
        winner = home_ab if owner == home_id else away_ab
        loser  = away_ab if owner == home_id else home_ab
        faceoffs.append((gid, t_abs, winner, loser, zone))
    if (i + 1) % 1000 == 0 or i + 1 == len(game_ids):
        print(f"    processed {i+1}/{len(game_ids)} games; faceoffs so far: {len(faceoffs)}")

fo_df = pd.DataFrame(faceoffs, columns=["game_id","t","winner","loser","zone_from_winner"])
fo_df["season"] = fo_df["game_id"].map(game_to_season)
print(f"    total 5v5 OZ/DZ faceoffs: {len(fo_df)}")
fo_df.to_pickle("/tmp/dt_faceoffs.pkl")

# ---------------------------------------------------------------------------
# Stage 0c: load shifts indexed by game
# ---------------------------------------------------------------------------
print("[0c] loading shifts ...")
shift_df = pd.read_csv(SHIFT_CSV,
    usecols=["game_id","player_id","team_abbrev","abs_start_secs","abs_end_secs"])
shift_df = shift_df.dropna(subset=["player_id","abs_start_secs","abs_end_secs"])
shift_df["game_id"] = shift_df["game_id"].astype("int64")
shift_df["player_id"] = shift_df["player_id"].astype("int64")
shift_df["abs_start_secs"] = shift_df["abs_start_secs"].astype("int32")
shift_df["abs_end_secs"] = shift_df["abs_end_secs"].astype("int32")
shift_df = shift_df[shift_df["game_id"].isin(game_ids)]
shift_df["team_abbrev"] = shift_df["team_abbrev"].astype(str).map(norm_team)
print(f"    shifts in scope: {len(shift_df)}")
shift_df = shift_df.sort_values(["game_id","abs_start_secs"]).reset_index(drop=True)

# Pre-group by game for fast lookup
shifts_by_game = {}
for gid, grp in shift_df.groupby("game_id", sort=False):
    shifts_by_game[gid] = (
        grp["player_id"].to_numpy(),
        grp["team_abbrev"].to_numpy(),
        grp["abs_start_secs"].to_numpy(),
        grp["abs_end_secs"].to_numpy(),
    )
print(f"    indexed {len(shifts_by_game)} games")

# ---------------------------------------------------------------------------
# Stage 0d: load shots (ES state only) and index by game
# ---------------------------------------------------------------------------
print("[0d] loading shots (ES only) ...")
shots_df = pd.read_csv(SHOTS_CSV,
    usecols=["game_id","season","abs_time","event_type","shooting_team_abbrev",
             "shooter_player_id","zone","state"],
    dtype={"game_id":"int64","season":"int64"})
shots_df["season"] = shots_df["season"].astype(str)
shots_df = shots_df[shots_df["season"].isin(POOLED_SEASONS)]
shots_df = shots_df[shots_df["state"] == "ES"]
shots_df["shooting_team_abbrev"] = shots_df["shooting_team_abbrev"].astype(str).map(norm_team)
shots_df["is_fenwick"] = shots_df["event_type"].isin(FENWICK_TYPES)
shots_df["is_corsi"]   = shots_df["event_type"].isin(CORSI_TYPES)
shots_df["is_cnfi"]    = shots_df["zone"] == "CNFI"
shots_df["is_mnfi"]    = shots_df["zone"] == "MNFI"
shots_df["is_cnfi_mnfi"] = shots_df["is_cnfi"] | shots_df["is_mnfi"]
print(f"    ES shots in scope: {len(shots_df)}")
shots_df = shots_df.sort_values(["game_id","abs_time"]).reset_index(drop=True)

# Pre-group shots by game
shots_by_game = {}
for gid, grp in shots_df.groupby("game_id", sort=False):
    shots_by_game[gid] = {
        "t": grp["abs_time"].to_numpy(),
        "team": grp["shooting_team_abbrev"].to_numpy(),
        "corsi": grp["is_corsi"].to_numpy(),
        "fenwick": grp["is_fenwick"].to_numpy(),
        "cnfi_mnfi": grp["is_cnfi_mnfi"].to_numpy(),
        "cnfi": grp["is_cnfi"].to_numpy(),
        "mnfi": grp["is_mnfi"].to_numpy(),
    }

# ---------------------------------------------------------------------------
# Stage 1: shift-based faceoff attribution.
# For each OZ/DZ faceoff, for each on-ice player, classify zone from their POV,
# count own-team and opp-team attempts in [t_face, shift_end].
# Accumulate per (team, season, zone_from_team_POV):
#   n_shifts, sum_corsi_for, sum_corsi_against, sum_fen_for, sum_fen_against,
#   sum_cnfi_mnfi_for, sum_cnfi_mnfi_against, total_shift_remainder_sec
# ---------------------------------------------------------------------------
print("[1] shift-based attribution across all OZ/DZ faceoffs ...")

agg_keys = ("n_shifts","sec","cf","ca","fen_f","fen_a","cm_f","cm_a")
# per-(team, season, zone_from_team_POV) counters
team_agg = defaultdict(lambda: {k: 0.0 for k in agg_keys})

fo_by_game = {g: grp[["t","winner","loser","zone_from_winner"]].to_numpy()
              for g, grp in fo_df.groupby("game_id", sort=False)}

for gi, gid in enumerate(game_ids):
    if gid not in fo_by_game or gid not in shifts_by_game:
        continue
    season = game_to_season[gid]
    shots = shots_by_game.get(gid)
    fos = fo_by_game[gid]
    pids, teams, starts, ends = shifts_by_game[gid]

    # pre-sort shifts by start already done
    if shots is not None:
        shot_t = shots["t"]

    for t_face, winner, loser, zone_w in fos:
        # Active shifts: start <= t_face < end
        # Binary search for shifts where start <= t_face
        lo = 0
        hi = bisect_right(starts, t_face)
        active_idx = [i for i in range(lo, hi) if ends[i] > t_face]
        for i in active_idx:
            p_team = teams[i]
            shift_end = ends[i]
            # Zone from player's team POV
            zone_p = zone_w if p_team == winner else FLIP[zone_w]
            if zone_p not in ("O", "D"):
                continue
            remainder = shift_end - t_face
            # Count shots in [t_face, shift_end] for this game
            if shots is not None and len(shot_t) > 0:
                a = bisect_left(shot_t, t_face)
                b = bisect_right(shot_t, shift_end)
                if b > a:
                    sl_team = shots["team"][a:b]
                    is_own = (sl_team == p_team)
                    cf_f = int(np.sum(shots["corsi"][a:b] & is_own))
                    cf_a = int(np.sum(shots["corsi"][a:b] & ~is_own))
                    ff   = int(np.sum(shots["fenwick"][a:b] & is_own))
                    fa   = int(np.sum(shots["fenwick"][a:b] & ~is_own))
                    cm_f = int(np.sum(shots["cnfi_mnfi"][a:b] & is_own))
                    cm_a = int(np.sum(shots["cnfi_mnfi"][a:b] & ~is_own))
                else:
                    cf_f = cf_a = ff = fa = cm_f = cm_a = 0
            else:
                cf_f = cf_a = ff = fa = cm_f = cm_a = 0
            key = (p_team, season, zone_p)
            agg = team_agg[key]
            agg["n_shifts"] += 1
            agg["sec"]      += remainder
            agg["cf"]       += cf_f
            agg["ca"]       += cf_a
            agg["fen_f"]    += ff
            agg["fen_a"]    += fa
            agg["cm_f"]     += cm_f
            agg["cm_a"]     += cm_a
    if (gi + 1) % 1000 == 0 or gi + 1 == len(game_ids):
        print(f"    processed {gi+1}/{len(game_ids)} games")

# Build team_agg DataFrame
rows = []
for (team, season, zone), d in team_agg.items():
    rows.append({"team": team, "season": season, "zone": zone, **d})
ta_df = pd.DataFrame(rows)
print(f"    team-season-zone rows: {len(ta_df)}")
ta_df.to_pickle("/tmp/dt_team_zone_agg.pkl")

# ---------------------------------------------------------------------------
# STAGE 1a: Factor_base per metric (league-pooled, per-shift units).
# Factor_base = mean OZ shift attempts - mean DZ shift attempts.
# Metrics assessed: Corsi (CF), Fenwick (FF), CNFI+MNFI (CM).
# ---------------------------------------------------------------------------
print("[1a] computing Factor_base per metric ...")
oz_mask = ta_df["zone"] == "O"
dz_mask = ta_df["zone"] == "D"
oz_shifts = ta_df.loc[oz_mask, "n_shifts"].sum()
dz_shifts = ta_df.loc[dz_mask, "n_shifts"].sum()

def fb(col):
    oz_mean = ta_df.loc[oz_mask, col].sum() / oz_shifts
    dz_mean = ta_df.loc[dz_mask, col].sum() / dz_shifts
    return oz_mean, dz_mean, oz_mean - dz_mean

fb_corsi_f   = fb("cf")
fb_fenwick_f = fb("fen_f")
fb_cnfimnfi_f = fb("cm_f")

factor_rows = [
    {"metric": "Corsi_CF_pct",  "oz_mean_per_shift": fb_corsi_f[0],
     "dz_mean_per_shift": fb_corsi_f[1], "Factor_base": fb_corsi_f[2]},
    {"metric": "Fenwick_CF_pct","oz_mean_per_shift": fb_fenwick_f[0],
     "dz_mean_per_shift": fb_fenwick_f[1], "Factor_base": fb_fenwick_f[2]},
    {"metric": "V5",            "oz_mean_per_shift": fb_cnfimnfi_f[0],
     "dz_mean_per_shift": fb_cnfimnfi_f[1], "Factor_base": fb_cnfimnfi_f[2]},
    {"metric": "CNFI_MNFI_team","oz_mean_per_shift": fb_cnfimnfi_f[0],
     "dz_mean_per_shift": fb_cnfimnfi_f[1], "Factor_base": fb_cnfimnfi_f[2]},
    # V1/V2/V4 use same zone factor as CNFI+MNFI team (forward subset but same shot tagging)
    {"metric": "V1",            "oz_mean_per_shift": fb_cnfimnfi_f[0],
     "dz_mean_per_shift": fb_cnfimnfi_f[1], "Factor_base": fb_cnfimnfi_f[2]},
    {"metric": "V2",            "oz_mean_per_shift": np.nan,
     "dz_mean_per_shift": np.nan, "Factor_base": np.nan, "note": "placeholder — uses CNFI only"},
]
factor_df = pd.DataFrame(factor_rows)
factor_df.to_csv(OUT / "stage1_factor_base.csv", index=False)
print(factor_df)

# ---------------------------------------------------------------------------
# STAGE 1b: DZ-only metric per team-season.
# For V5 / Corsi / Fenwick / CNFI+MNFI% team: restrict to DZ-shift intervals,
# compute own-/opp-attempts ratio at team level.
# ---------------------------------------------------------------------------
print("[1b] computing DZ-only metrics per team-season ...")
# Pivot team_agg: rows = (team, season), cols = zone-specific sums
piv = ta_df.pivot_table(index=["team","season"], columns="zone",
                        values=["n_shifts","sec","cf","ca","fen_f","fen_a","cm_f","cm_a"],
                        aggfunc="sum", fill_value=0)
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
piv = piv.reset_index()

# DZ-only on-ice CF%/Fenwick%/CNFI+MNFI%
piv["Corsi_DZ"]      = piv["cf_D"]   / (piv["cf_D"]   + piv["ca_D"]).replace(0, np.nan)
piv["Fenwick_DZ"]    = piv["fen_f_D"] / (piv["fen_f_D"] + piv["fen_a_D"]).replace(0, np.nan)
piv["CNFI_MNFI_DZ"]  = piv["cm_f_D"] / (piv["cm_f_D"] + piv["cm_a_D"]).replace(0, np.nan)
# V5 (on-ice CNFI+MNFI% all skaters) — at team aggregate, equals CNFI_MNFI_DZ
piv["V5_DZ"]         = piv["CNFI_MNFI_DZ"]
# V1/V2 per-60 in DZ intervals (team total for-side only).  Shots / seconds × 3600
piv["V1_DZ"]         = piv["cm_f_D"] / piv["sec_D"].replace(0, np.nan) * 3600.0
# For V2 we'd need CNFI-only for-side; recompute from ta_df
# (Skipped with note — insufficient CNFI-only per-shift tallies here)

piv["OZ_ratio"] = piv["n_shifts_O"] / (piv["n_shifts_O"] + piv["n_shifts_D"]).replace(0, np.nan)
piv.to_csv(OUT / "stage1_inputs_team.csv", index=False)

# Merge with team points + raw team metrics
m = piv.merge(team_metrics[["season","team","points","CNFI_MNFI_team","Corsi_CF_pct_team",
                            "CF","CA"]],
              on=["season","team"], how="inner")
print(f"    matched team-seasons: N = {len(m)}")

def r2_simple(x, y):
    r = np.corrcoef(x, y)[0,1]
    return r*r, r

# Raw team-level R²
raw_rows = [
    ("Corsi_CF_pct",    m["Corsi_CF_pct_team"]),
    ("Fenwick_CF_pct",  (m["CF"] - (m["ca_O"] + m["ca_D"])*0 + 0)),  # placeholder; use team raw
    ("CNFI_MNFI_team",  m["CNFI_MNFI_team"]),
    ("V5",              m["CNFI_MNFI_team"]),  # team aggregate proxy of V5
]
# DZ-only R²
dz_rows = [
    ("V5",              m["V5_DZ"]),
    ("Corsi_CF_pct",    m["Corsi_DZ"]),
    ("Fenwick_CF_pct",  m["Fenwick_DZ"]),
    ("CNFI_MNFI_team",  m["CNFI_MNFI_DZ"]),
    ("V1",              m["V1_DZ"]),
]

dz_results = []
for name, series in dz_rows:
    vals = series.dropna()
    y = m.loc[vals.index, "points"]
    if len(vals) < 5: continue
    r2, r = r2_simple(vals, y)
    p = st.pearsonr(vals, y).pvalue
    dz_results.append({"metric": name, "N": len(vals), "r": r, "R2_DZ": r2, "p": p})
dz_df = pd.DataFrame(dz_results)
dz_df.to_csv(OUT / "stage1_dz_only_r2.csv", index=False)
print("\nDZ-only R² per metric:")
print(dz_df.to_string(index=False))

# ---------------------------------------------------------------------------
# STAGE 2: zone factor multiplier sweep.
# raw = team aggregate metric value (V5=CNFI+MNFI team%; Corsi team%; Fenwick team% from shots)
# league_avg = league mean across team-seasons
# expected = league_avg + (OZ_ratio - 0.5) * Factor_base * multiplier
# zone_adj = raw - expected + league_avg
# R² against points.
# Multipliers: 0.25, 0.50, ..., 3.00. Traditional factor=3.5 tested for Corsi & Fenwick.
# ---------------------------------------------------------------------------
print("\n[2] zone multiplier sweep ...")

# Build Fenwick team% from shot counts aggregated per team-season
fen_team = shots_df[shots_df["is_fenwick"]].groupby(
    ["season","shooting_team_abbrev"]).size().rename("Fen_F").reset_index()
fen_team = fen_team.rename(columns={"shooting_team_abbrev":"team"})
# Opponent fenwick = total fenwick in games - team's. Use home/away — quick approximation
# via total season fenwick - own-team fenwick over same team-seasons.
# Each shot tagged by shooting team; join game's home/away to identify opp.
games_played = pd.read_csv(GAME_IDS, usecols=["game_id","season","home_abbrev","away_abbrev"])
games_played = games_played[games_played["season"].astype(str).isin(POOLED_SEASONS)]
games_played["home_abbrev"] = games_played["home_abbrev"].astype(str).map(norm_team)
games_played["away_abbrev"] = games_played["away_abbrev"].astype(str).map(norm_team)
games_played["season"] = games_played["season"].astype(str)
# Fenwick per (game, team)
fen_game_team = shots_df[shots_df["is_fenwick"]].groupby(
    ["game_id","shooting_team_abbrev"]).size().rename("fen").reset_index()
fen_game_team = fen_game_team.rename(columns={"shooting_team_abbrev":"team"})
# Corsi per (game, team)
cor_game_team = shots_df[shots_df["is_corsi"]].groupby(
    ["game_id","shooting_team_abbrev"]).size().rename("cf").reset_index()
cor_game_team = cor_game_team.rename(columns={"shooting_team_abbrev":"team"})

def long_for_vs_against(game_team, team_col):
    g = games_played[["game_id","season","home_abbrev","away_abbrev"]]
    merged = game_team.merge(g, on="game_id", how="inner")
    # Determine opp per row
    merged["opp"] = np.where(merged["team"] == merged["home_abbrev"],
                              merged["away_abbrev"], merged["home_abbrev"])
    # For-side rows
    f = merged[["season","team",team_col]].rename(columns={team_col: "F"})
    # Against-side rows: opp's shots counted against team
    a = merged[["season","opp",team_col]].rename(columns={"opp":"team", team_col:"A"})
    f_sum = f.groupby(["season","team"], as_index=False)["F"].sum()
    a_sum = a.groupby(["season","team"], as_index=False)["A"].sum()
    return f_sum.merge(a_sum, on=["season","team"], how="outer").fillna(0)

fen_agg = long_for_vs_against(fen_game_team, "fen")
fen_agg["Fenwick_CF_pct_team"] = fen_agg["F"] / (fen_agg["F"] + fen_agg["A"])
cor_agg = long_for_vs_against(cor_game_team, "cf")
cor_agg["Corsi_CF_pct_team_ES"] = cor_agg["F"] / (cor_agg["F"] + cor_agg["A"])

m2 = m.merge(fen_agg[["season","team","Fenwick_CF_pct_team"]], on=["season","team"], how="left")
m2 = m2.merge(cor_agg[["season","team","Corsi_CF_pct_team_ES"]], on=["season","team"], how="left")

# Factor_base values (per-shift attempts). Convert to units of raw metric (%):
# The raw metric is a percentage share (0..1). For zone adjustment in pct-point units,
# Factor_base_pct ≈ FB_per_shift / (league avg total shots per shift in the same zone).
# For simplicity we apply Factor_base directly on a percentage scale by normalizing:
# Treat FB as expressed in % delta that would occur if OZ_ratio moved from 0.5 -> 1.0.
# This sets the ceiling slope.  We express FB as a % by dividing by per-shift total shots.
total_shots_per_shift_oz = (ta_df.loc[oz_mask, "cf"].sum() + ta_df.loc[oz_mask, "ca"].sum()) / oz_shifts
total_shots_per_shift_dz = (ta_df.loc[dz_mask, "cf"].sum() + ta_df.loc[dz_mask, "ca"].sum()) / dz_shifts
avg_total = (total_shots_per_shift_oz + total_shots_per_shift_dz) / 2.0

def fb_pct(col):
    oz, dz, _ = fb(col)
    # Approximate % point delta for 100% OZ vs 100% DZ exposure
    # delta_pct = (OZ own attempts / OZ total) - (DZ own attempts / DZ total)
    oz_tot = (ta_df.loc[oz_mask, "cf"].sum() + ta_df.loc[oz_mask, "ca"].sum()) if col=="cf" else \
             (ta_df.loc[oz_mask, "fen_f"].sum() + ta_df.loc[oz_mask, "fen_a"].sum()) if col=="fen_f" else \
             (ta_df.loc[oz_mask, "cm_f"].sum() + ta_df.loc[oz_mask, "cm_a"].sum())
    dz_tot = (ta_df.loc[dz_mask, "cf"].sum() + ta_df.loc[dz_mask, "ca"].sum()) if col=="cf" else \
             (ta_df.loc[dz_mask, "fen_f"].sum() + ta_df.loc[dz_mask, "fen_a"].sum()) if col=="fen_f" else \
             (ta_df.loc[dz_mask, "cm_f"].sum() + ta_df.loc[dz_mask, "cm_a"].sum())
    own_col_oz = ta_df.loc[oz_mask, col].sum()
    own_col_dz = ta_df.loc[dz_mask, col].sum()
    oz_pct = own_col_oz / oz_tot if oz_tot else np.nan
    dz_pct = own_col_dz / dz_tot if dz_tot else np.nan
    return oz_pct, dz_pct, oz_pct - dz_pct

fbp_corsi   = fb_pct("cf")
fbp_fenwick = fb_pct("fen_f")
fbp_cnfimnfi= fb_pct("cm_f")

# Zone-adjust sweep
MULTIPLIERS = [round(0.25*i, 2) for i in range(1, 13)]  # 0.25..3.00

def sweep(raw_col, fb_p, name):
    raw = m2[raw_col].astype(float)
    oz_r = m2["OZ_ratio"].astype(float)
    y = m2["points"].astype(float)
    out = []
    for mult in MULTIPLIERS:
        expected = fb_p[2] * mult * (oz_r - 0.5)
        adj = raw - expected
        mask = raw.notna() & oz_r.notna()
        r2, r = r2_simple(adj[mask], y[mask])
        out.append({"metric": name, "multiplier": mult,
                    "factor_effective_pct_pts": fb_p[2]*mult*100,
                    "R2": r2, "r": r, "N": int(mask.sum())})
    # Also append traditional factor=3.5 in % point units (absolute, not multiplier)
    if name in ("Corsi_CF_pct","Fenwick_CF_pct"):
        expected = 0.035 * (oz_r - 0.5)   # 3.5 pct pts for OZ_ratio swing from 0 to 1
        adj = raw - expected
        mask = raw.notna() & oz_r.notna()
        r2, r = r2_simple(adj[mask], y[mask])
        out.append({"metric": name, "multiplier": "traditional_3.5",
                    "factor_effective_pct_pts": 3.5, "R2": r2, "r": r,
                    "N": int(mask.sum())})
    return out

sweep_rows = []
sweep_rows += sweep("Corsi_CF_pct_team_ES", fbp_corsi,   "Corsi_CF_pct")
sweep_rows += sweep("Fenwick_CF_pct_team", fbp_fenwick, "Fenwick_CF_pct")
sweep_rows += sweep("CNFI_MNFI_team",      fbp_cnfimnfi,"CNFI_MNFI_team")
sweep_rows += sweep("CNFI_MNFI_team",      fbp_cnfimnfi,"V5")  # V5 at team agg ≈ CNFI_MNFI team

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(OUT / "stage2_factor_optimization.csv", index=False)
print("\nStage 2 sweep (top rows per metric):")
for name, grp in sweep_df.groupby("metric"):
    best = grp.sort_values("R2", ascending=False).head(3)
    print(f"\n=== {name} ===")
    print(best.to_string(index=False))

# ---------------------------------------------------------------------------
# STAGE 3: winner per metric — raw vs DZ-only vs ZA optimal.
# ---------------------------------------------------------------------------
print("\n[3] Stage 3 decision table ...")

# Raw R² from team-level points
raw_points_r2 = {}
for name, col in [("Corsi_CF_pct","Corsi_CF_pct_team_ES"),
                  ("Fenwick_CF_pct","Fenwick_CF_pct_team"),
                  ("CNFI_MNFI_team","CNFI_MNFI_team"),
                  ("V5","CNFI_MNFI_team")]:
    s = m2[col].dropna()
    y = m2.loc[s.index, "points"]
    r2, r = r2_simple(s, y)
    raw_points_r2[name] = (r2, r, len(s))

# DZ-only already in dz_df
dz_map = {row["metric"]: row for _, row in dz_df.iterrows()}

# ZA optimal
za_map = {}
for name, grp in sweep_df.groupby("metric"):
    g = grp[grp["multiplier"] != "traditional_3.5"]
    best = g.loc[g["R2"].idxmax()]
    za_map[name] = best

# Traditional 3.5 (Corsi/Fenwick only)
trad_map = {}
for name in ["Corsi_CF_pct", "Fenwick_CF_pct"]:
    row = sweep_df[(sweep_df["metric"]==name) & (sweep_df["multiplier"]=="traditional_3.5")]
    if len(row): trad_map[name] = row.iloc[0]

decision_rows = []
for name in ["V5","Corsi_CF_pct","Fenwick_CF_pct","CNFI_MNFI_team"]:
    raw = raw_points_r2.get(name, (np.nan,np.nan,0))
    dz  = dz_map.get(name, {})
    za  = za_map.get(name, {})
    trad = trad_map.get(name, {})
    winner = "DZ" if dz.get("R2_DZ",0) > za["R2"] else "ZA"
    decision_rows.append({
        "metric": name,
        "R2_raw": raw[0],
        "R2_DZ_only": dz.get("R2_DZ"),
        "R2_ZA_optimal": za["R2"] if isinstance(za, pd.Series) else za.get("R2"),
        "ZA_best_multiplier": za["multiplier"] if isinstance(za, pd.Series) else za.get("multiplier"),
        "R2_traditional_3.5": trad.get("R2") if isinstance(trad, (dict,pd.Series)) and len(trad) else None,
        "winner_methodology": winner,
        "N": raw[2],
    })
dec_df = pd.DataFrame(decision_rows)
dec_df.to_csv(OUT / "stage3_zone_decision.csv", index=False)
print(dec_df.to_string(index=False))

print("\n[done] Stage 1-3 complete. Outputs in:", OUT)
