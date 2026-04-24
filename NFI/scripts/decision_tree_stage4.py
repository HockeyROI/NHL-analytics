"""Stage 4 — per-player IOC/IOL iterative convergence.

Metrics implemented in this run: V5, V1b, Fenwick_CF%, Corsi_CF%.
xG% per-player requires MoneyPuck skater data fetch — team-level proxy used here.
V3 uses existing twoway_forward_score.csv (career aggregate).

Methodology (per confirmed spec):
  1. Per-player per-season on-ice attribution (Fenwick + CNFI+MNFI, Corsi for Corsi metric).
  2. Zone adjust each player via (OZ_ratio - 0.5) * effective_factor,
     effective factors from Stage 2 optimal multipliers.
  3. Convergent: z-score ratings, iterate IOC/IOL regression, residuals → new rating,
     until max |delta z| < 0.001.
  4. Traditional: factor=3.5 pp zone adj + single-pass QoC/QoT regression.
  5. Team aggregate via TOI-weighted mean of player z-scores.
  6. R² vs points at Raw / ZA / Traditional / Converged.
  7. Top-30 lists Raw / ZA / Converged.

Outputs:
  NFI/output/zone_adjustment/complete_decision_tree/
    stage4_convergence_history.csv
    stage4_player_ratings_{metric}.csv
    stage5_r2_summary.csv
    stage6_player_lists_V5.csv
    stage6_player_lists_Fenwick.csv
    stage6_player_lists_V1b.csv
    stage6_player_lists_Corsi.csv
    stage6_cross_metric_elite.csv
    stage7_annual_ratings.csv
"""
from __future__ import annotations

import json
from pathlib import Path
from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "NFI/output/zone_adjustment/complete_decision_tree"
PBP_DIR = ROOT / "Zones/raw/pbp"
SHIFT_CSV = ROOT / "NFI/Geometry_post/Data/shift_data.csv"
SHOTS_CSV = ROOT / "NFI/output/shots_tagged.csv"
POS_CSV = ROOT / "NFI/output/player_positions.csv"
TEAM_METRICS = ROOT / "NFI/output/team_level_all_metrics.csv"
GAME_IDS = ROOT / "Data/game_ids.csv"

ABBR_MAP = {"ARI": "UTA"}
POOLED = {"20222023", "20232024", "20242025", "20252026"}
FLIP = {"O":"D","D":"O","N":"N"}
FEN  = {"shot-on-goal","missed-shot","goal"}
COR  = FEN | {"blocked-shot"}

# Stage 2 optimal effective factors (pct-point delta in own-share per unit OZ_ratio swing)
# V5 / CNFI+MNFI: 2.50× × 10.71pp = 26.77 pp
# Fenwick CF%   : 3.00× × 11.91pp = 35.73 pp
# Corsi CF%     : 1.50× ×  3.87pp =  5.80 pp
# V1b (F-only)  : 0.75× × 10.71pp =  8.03 pp   (approximate; share filter)
FACTOR_EFFECTIVE = {
    "V5":        0.2677,
    "V1b":       0.0803,
    "Fenwick":   0.3573,
    "Corsi":     0.0580,
}
TRADITIONAL_FACTOR = 0.035  # 3.5 pct-pts, applied to all metrics in traditional variant
MIN_TOI_MIN = 200           # minimum ES TOI to include a player in a season

def norm_team(a): return ABBR_MAP.get(a, a) if a else a
def mmss(s):
    if not s or ":" not in s: return 0
    m, ss = s.split(":")
    try: return int(m)*60 + int(ss)
    except ValueError: return 0

# ---------------------------------------------------------------------------
# Load basic inputs
# ---------------------------------------------------------------------------
print("[0] loading inputs ...")
games_df = pd.read_csv(GAME_IDS)
games_df = games_df[(games_df["season"].astype(str).isin(POOLED)) &
                    (games_df["game_type"] == "regular")]
g2s = dict(zip(games_df["game_id"].astype(int), games_df["season"].astype(str)))
gids = sorted(g2s.keys())

pos_df = pd.read_csv(POS_CSV)
pos_map = dict(zip(pos_df["player_id"].astype(int), pos_df["pos_group"].astype(str)))
name_map = dict(zip(pos_df["player_id"].astype(int), pos_df["player_name"].astype(str)))

tm = pd.read_csv(TEAM_METRICS)
tm["season"] = tm["season"].astype(str)
tm = tm[tm["season"].isin(POOLED)][["season","team","points"]]

# ---------------------------------------------------------------------------
# Load shifts, index by game
# ---------------------------------------------------------------------------
print("[1] loading shifts ...")
sd = pd.read_csv(SHIFT_CSV,
    usecols=["game_id","player_id","team_abbrev","abs_start_secs","abs_end_secs"])
sd = sd.dropna(subset=["player_id","abs_start_secs","abs_end_secs"])
sd["game_id"] = sd["game_id"].astype("int64")
sd = sd[sd["game_id"].isin(gids)]
sd["player_id"] = sd["player_id"].astype("int64")
sd["abs_start_secs"] = sd["abs_start_secs"].astype("int32")
sd["abs_end_secs"]   = sd["abs_end_secs"].astype("int32")
sd["team_abbrev"] = sd["team_abbrev"].astype(str).map(norm_team)
sd = sd.sort_values(["game_id","abs_start_secs"]).reset_index(drop=True)
sd["season"] = sd["game_id"].map(g2s)

# Per-game shift arrays
print("    indexing shifts by game ...")
shifts_by_game = {}
for gid, g in sd.groupby("game_id", sort=False):
    shifts_by_game[gid] = {
        "pid": g["player_id"].to_numpy(),
        "team": g["team_abbrev"].to_numpy(),
        "s": g["abs_start_secs"].to_numpy(),
        "e": g["abs_end_secs"].to_numpy(),
    }

# ---------------------------------------------------------------------------
# Load shots, index by game, ES only
# ---------------------------------------------------------------------------
print("[2] loading shots (ES only) ...")
xd = pd.read_csv(SHOTS_CSV,
    usecols=["game_id","season","abs_time","event_type","shooting_team_abbrev",
             "zone","state"])
xd["season"] = xd["season"].astype(str)
xd = xd[(xd["season"].isin(POOLED)) & (xd["state"]=="ES")]
xd["shooting_team_abbrev"] = xd["shooting_team_abbrev"].astype(str).map(norm_team)
xd["is_fen"] = xd["event_type"].isin(FEN)
xd["is_cor"] = xd["event_type"].isin(COR)
xd["is_cm"]  = xd["zone"].isin(["CNFI","MNFI"])
xd = xd.sort_values(["game_id","abs_time"]).reset_index(drop=True)
shots_by_game = {}
for gid, g in xd.groupby("game_id", sort=False):
    shots_by_game[gid] = {
        "t":    g["abs_time"].to_numpy(),
        "team": g["shooting_team_abbrev"].to_numpy(),
        "fen":  g["is_fen"].to_numpy(),
        "cor":  g["is_cor"].to_numpy(),
        "cm":   g["is_cm"].to_numpy(),
    }

# ---------------------------------------------------------------------------
# PHASE A: per-player per-season on-ice attribution + individual TOI.
# For each shot: find all on-ice players at shot time, attribute CF/CA.
# ---------------------------------------------------------------------------
print("[3] per-player on-ice attribution ...")
# Counters: (pid, season) -> dict of counts
pp = defaultdict(lambda: {"TOI":0.0,
                          "cf_fen":0,"ca_fen":0,
                          "cf_cor":0,"ca_cor":0,
                          "cf_cm":0, "ca_cm":0})

for gi, gid in enumerate(gids):
    if gid not in shifts_by_game:
        continue
    season = g2s[gid]
    s_arr = shifts_by_game[gid]
    pids, teams, starts, ends = s_arr["pid"], s_arr["team"], s_arr["s"], s_arr["e"]
    # Per-player TOI per game = sum of shift lengths
    for i in range(len(pids)):
        pp[(int(pids[i]), season)]["TOI"] += (ends[i] - starts[i])

    # Attribute shots: iterate per-shift; find shots within [start, end)
    shots = shots_by_game.get(gid)
    if shots is None or len(shots["t"]) == 0:
        continue
    shot_t = shots["t"]; shot_team = shots["team"]
    shot_fen = shots["fen"]; shot_cor = shots["cor"]; shot_cm = shots["cm"]
    # For each shift, bisect [start, end) in shot_t, vectorize tallies
    for i in range(len(pids)):
        s_i, e_i = starts[i], ends[i]
        a = bisect_left(shot_t, s_i)
        b = bisect_left(shot_t, e_i)
        if b <= a:
            continue
        sl_team = shot_team[a:b]
        pid_i = int(pids[i]); team_i = teams[i]
        is_own = (sl_team == team_i)
        f_arr = shot_fen[a:b]; c_arr = shot_cor[a:b]; m_arr = shot_cm[a:b]
        rec = pp[(pid_i, season)]
        rec["cf_fen"] += int((f_arr & is_own).sum())
        rec["ca_fen"] += int((f_arr & ~is_own).sum())
        rec["cf_cor"] += int((c_arr & is_own).sum())
        rec["ca_cor"] += int((c_arr & ~is_own).sum())
        rec["cf_cm"]  += int((m_arr & is_own).sum())
        rec["ca_cm"]  += int((m_arr & ~is_own).sum())
    if (gi+1) % 1000 == 0 or gi+1 == len(gids):
        print(f"    attribution: {gi+1}/{len(gids)}")

# Build player-season DataFrame
rows = []
for (pid, season), d in pp.items():
    rows.append({"player_id":pid,"season":season,
                 "toi_sec":d["TOI"],
                 "cf_fen":d["cf_fen"],"ca_fen":d["ca_fen"],
                 "cf_cor":d["cf_cor"],"ca_cor":d["ca_cor"],
                 "cf_cm":d["cf_cm"], "ca_cm":d["ca_cm"]})
ppdf = pd.DataFrame(rows)
ppdf["pos"] = ppdf["player_id"].map(lambda p: pos_map.get(int(p), ""))
ppdf["name"] = ppdf["player_id"].map(lambda p: name_map.get(int(p), ""))
ppdf = ppdf[(ppdf["pos"].isin(["F","D"])) & (ppdf["toi_sec"]/60 >= MIN_TOI_MIN)]
ppdf["fen_pct"] = ppdf["cf_fen"] / (ppdf["cf_fen"] + ppdf["ca_fen"]).replace(0,np.nan)
ppdf["cor_pct"] = ppdf["cf_cor"] / (ppdf["cf_cor"] + ppdf["ca_cor"]).replace(0,np.nan)
ppdf["cm_pct"]  = ppdf["cf_cm"]  / (ppdf["cf_cm"]  + ppdf["ca_cm"]).replace(0,np.nan)
# Identify primary team: team with most TOI in season. We don't have per-shift team directly,
# but every shift record has team. Aggregate per (player, season, team) TOI:
print(f"    {len(ppdf)} qualifying player-seasons")
pt = sd.groupby(["player_id","season","team_abbrev"], as_index=False)["abs_end_secs"].count()
pt.columns = ["player_id","season","team","n_shifts"]
pt_toi = sd.copy()
pt_toi["toi"] = pt_toi["abs_end_secs"] - pt_toi["abs_start_secs"]
pt_agg = pt_toi.groupby(["player_id","season","team_abbrev"], as_index=False)["toi"].sum()
pt_agg = pt_agg.rename(columns={"team_abbrev":"team"})
primary = pt_agg.sort_values("toi", ascending=False).groupby(["player_id","season"]).head(1)
ppdf = ppdf.merge(primary[["player_id","season","team"]], on=["player_id","season"], how="left")
ppdf.to_pickle("/tmp/s4_ppdf.pkl")
print(f"    saved player-season attributions")

# ---------------------------------------------------------------------------
# PHASE B: per-player OZ / DZ faceoff exposure
# Reuse cached faceoffs; attribute per player.
# ---------------------------------------------------------------------------
print("[4] per-player OZ/DZ faceoff exposure ...")
fo_df = pd.read_pickle("/tmp/dt_faceoffs.pkl")
fo_by_g = {g: grp[["t","winner","loser","zone_from_winner"]].to_numpy()
           for g, grp in fo_df.groupby("game_id", sort=False)}
pfo = defaultdict(lambda: {"oz":0,"dz":0})
for gi, gid in enumerate(gids):
    if gid not in fo_by_g or gid not in shifts_by_game: continue
    season = g2s[gid]
    shifts = shifts_by_game[gid]
    pids, teams, starts, ends = shifts["pid"], shifts["team"], shifts["s"], shifts["e"]
    for t_face, winner, loser, zone_w in fo_by_g[gid]:
        hi = bisect_right(starts, t_face)
        for i in range(hi):
            if ends[i] <= t_face: continue
            p_team = teams[i]
            zp = zone_w if p_team == winner else FLIP[zone_w]
            if zp == "O":   pfo[(int(pids[i]), season)]["oz"] += 1
            elif zp == "D": pfo[(int(pids[i]), season)]["dz"] += 1
    if (gi+1) % 2000 == 0: print(f"    {gi+1}/{len(gids)}")
pfo_df = pd.DataFrame([{"player_id":k[0],"season":k[1],**v} for k,v in pfo.items()])
pfo_df["oz_ratio"] = pfo_df["oz"] / (pfo_df["oz"] + pfo_df["dz"]).replace(0,np.nan)
ppdf = ppdf.merge(pfo_df[["player_id","season","oz_ratio"]], on=["player_id","season"], how="left")
ppdf["oz_ratio"] = ppdf["oz_ratio"].fillna(0.5)

# ---------------------------------------------------------------------------
# PHASE C: per-player zone adjustment (raw → ZA) per metric.
# ---------------------------------------------------------------------------
print("[5] applying per-player zone adjustment ...")
for metric, raw_col, fe in [("V5","cm_pct",FACTOR_EFFECTIVE["V5"]),
                             ("Fenwick","fen_pct",FACTOR_EFFECTIVE["Fenwick"]),
                             ("Corsi","cor_pct",FACTOR_EFFECTIVE["Corsi"]),
                             ("V1b","cm_pct",FACTOR_EFFECTIVE["V1b"])]:
    ppdf[f"ZA_{metric}"] = ppdf[raw_col] - fe * (ppdf["oz_ratio"] - 0.5)
    # Traditional: factor = 3.5 pp for all
    ppdf[f"TRAD_{metric}"] = ppdf[raw_col] - TRADITIONAL_FACTOR * (ppdf["oz_ratio"] - 0.5)

# ---------------------------------------------------------------------------
# PHASE D: shared-minutes matrix per season (linemates + opponents).
# For each season: all qualifying skaters. Per-game, per-team, compute pairwise
# shift-interval intersections. Aggregate across games.
# ---------------------------------------------------------------------------
print("[6] shared-minutes matrix ...")
# Build dict: season -> {pid -> idx}, and sparse TOI matrices (n x n)
from scipy.sparse import lil_matrix, csr_matrix

qualifying = set((int(r.player_id), r.season) for r in ppdf.itertuples())

shared = {}   # season -> {"idx": {pid:i}, "LM": csr(n,n), "OPP": csr(n,n)}
for season in sorted(POOLED):
    pids_s = sorted({pid for (pid, s) in qualifying if s == season})
    idx = {pid:i for i,pid in enumerate(pids_s)}
    n = len(pids_s)
    LM  = np.zeros((n,n), dtype=np.float32)
    OPP = np.zeros((n,n), dtype=np.float32)
    # Per-game accumulation
    season_games = [g for g in gids if g2s[g] == season]
    for gi, gid in enumerate(season_games):
        shifts = shifts_by_game.get(gid)
        if shifts is None: continue
        # Filter to qualifying players
        pids = shifts["pid"]; teams = shifts["team"]
        starts = shifts["s"]; ends = shifts["e"]
        # Build per-game player shifts arrays
        per_player_shifts = defaultdict(list)
        per_player_team = {}
        for i in range(len(pids)):
            pid_i = int(pids[i])
            if pid_i not in idx: continue
            per_player_shifts[pid_i].append((starts[i], ends[i]))
            per_player_team[pid_i] = teams[i]
        plist = list(per_player_shifts.keys())
        arrs = {p: np.array(per_player_shifts[p], dtype=np.int32) for p in plist}
        for a_i, pa in enumerate(plist):
            a_s, a_e = arrs[pa][:,0], arrs[pa][:,1]
            for b_i in range(a_i+1, len(plist)):
                pb = plist[b_i]
                b_s, b_e = arrs[pb][:,0], arrs[pb][:,1]
                ov = np.maximum(0,
                                np.minimum(a_e[:,None], b_e[None,:])
                                - np.maximum(a_s[:,None], b_s[None,:]))
                tot = ov.sum()
                if tot == 0: continue
                ia, ib = idx[pa], idx[pb]
                if per_player_team[pa] == per_player_team[pb]:
                    LM[ia, ib] += tot; LM[ib, ia] += tot
                else:
                    OPP[ia, ib] += tot; OPP[ib, ia] += tot
        if (gi+1) % 300 == 0 or gi+1 == len(season_games):
            print(f"    {season}: {gi+1}/{len(season_games)}")
    shared[season] = {"idx":idx, "LM":LM, "OPP":OPP,
                      "pids": pids_s}
    print(f"  season {season}: n_players={n}, LM nonzero={(LM>0).sum()}, OPP nonzero={(OPP>0).sum()}")

# ---------------------------------------------------------------------------
# PHASE E: IOC / IOL iterative convergence, per metric per season.
# Round 0 = zone-adjusted z-scored ratings (or raw z-scored for traditional).
# ---------------------------------------------------------------------------
print("[7] IOC/IOL convergence ...")
CONV_TOL = 0.001
MAX_ITERS = 50

def z(arr):
    arr = np.array(arr, dtype=float)
    mu, sd_ = np.nanmean(arr), np.nanstd(arr)
    return (arr - mu)/sd_ if sd_>0 else arr - mu

def ioc_iol(rating_vec, LM, OPP):
    """Return IOC, IOL vectors. Rows of LM/OPP are shared-TOI weights."""
    denom_lm  = LM.sum(axis=1); denom_opp = OPP.sum(axis=1)
    # Use np.where to avoid divide-by-zero
    IOL = np.where(denom_lm>0,  (LM @ rating_vec) / np.where(denom_lm>0, denom_lm, 1), 0.0)
    IOC = np.where(denom_opp>0, (OPP @ rating_vec) / np.where(denom_opp>0, denom_opp, 1), 0.0)
    return IOC, IOL

def regress_residuals(r, IOC, IOL):
    X = np.column_stack([np.ones(len(r)), IOC, IOL])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    resid = r - X @ beta
    return resid, beta

conv_history_rows = []
converged_ratings = {}   # (metric, season) -> dict pid→rating
traditional_ratings = {} # (metric, season) -> dict

# For forward-only metrics (V1b, V3) — filter IOC/IOL to forwards
FWD_ONLY = {"V1b", "V3"}

def forward_mask_for_season(season, pids_s):
    return np.array([pos_map.get(p) == "F" for p in pids_s])

for metric in ["V5","Fenwick","Corsi","V1b"]:
    for season in sorted(POOLED):
        info = shared[season]; idx = info["idx"]; pids_s = info["pids"]
        LM  = info["LM"].copy(); OPP = info["OPP"].copy()
        if metric in FWD_ONLY:
            mask = forward_mask_for_season(season, pids_s)
            LM[~mask, :]  = 0; LM[:, ~mask]  = 0
            OPP[~mask, :] = 0; OPP[:, ~mask] = 0
        # Build starting vector (ZA rating z-scored) for this metric, season
        sub = ppdf[(ppdf["season"]==season)]
        rating = np.full(len(pids_s), np.nan)
        for _, row in sub.iterrows():
            if row["player_id"] in idx:
                rating[idx[row["player_id"]]] = row[f"ZA_{metric}"]
        mask_valid = ~np.isnan(rating)
        # For forward-only metric: also require F position
        if metric in FWD_ONLY:
            mask_valid = mask_valid & mask
        rating_valid = rating[mask_valid]
        if rating_valid.size < 10:
            continue
        # z-score
        r_z = z(rating_valid)
        r_full = np.zeros(len(pids_s)); r_full[mask_valid] = r_z
        # Convergent iterations
        prev = r_full.copy()
        for it in range(1, MAX_ITERS+1):
            IOC, IOL = ioc_iol(prev, LM, OPP)
            resid, beta = regress_residuals(prev, IOC, IOL)
            # Re-z-score ONLY on valid subset
            if mask_valid.sum() > 1:
                sub_vals = resid[mask_valid]
                mu, ssd = sub_vals.mean(), sub_vals.std()
                if ssd > 0:
                    new_valid = (sub_vals - mu) / ssd
                else:
                    new_valid = sub_vals - mu
                new = np.zeros(len(pids_s))
                new[mask_valid] = new_valid
            else:
                new = resid
            max_delta = float(np.max(np.abs(new - prev))) if len(new) else 0.0
            conv_history_rows.append({"metric":metric,"season":season,"iter":it,
                                      "max_delta":max_delta,"beta_IOC":beta[1],"beta_IOL":beta[2]})
            prev = new
            if max_delta < CONV_TOL:
                break
        converged = {pids_s[i]: prev[i] for i in range(len(pids_s)) if mask_valid[i]}
        converged_ratings[(metric, season)] = converged
        # --- Traditional: single-pass QoC/QoT using RAW (un-z-scored) ratings as weights ---
        raw_col_map = {"V5":"cm_pct","Fenwick":"fen_pct","Corsi":"cor_pct","V1b":"cm_pct"}
        raw_vec = np.full(len(pids_s), np.nan)
        for _, row in sub.iterrows():
            if row["player_id"] in idx and pd.notna(row[raw_col_map[metric]]):
                raw_vec[idx[row["player_id"]]] = row[raw_col_map[metric]]
        trad_valid = ~np.isnan(raw_vec)
        if metric in FWD_ONLY:
            trad_valid = trad_valid & mask
        r_valid_raw = raw_vec.copy()
        r_valid_raw[~trad_valid] = 0
        QoC, QoT = ioc_iol(r_valid_raw, LM, OPP)
        # Starting from TRAD-ZA rating
        trad_rating = np.full(len(pids_s), np.nan)
        for _, row in sub.iterrows():
            if row["player_id"] in idx:
                trad_rating[idx[row["player_id"]]] = row[f"TRAD_{metric}"]
        # Regress trad_rating ~ QoC + QoT on valid subset; residuals = trad adjusted
        valid_idx = np.where(~np.isnan(trad_rating) & trad_valid)[0]
        X = np.column_stack([np.ones(len(valid_idx)), QoC[valid_idx], QoT[valid_idx]])
        beta_t, *_ = np.linalg.lstsq(X, trad_rating[valid_idx], rcond=None)
        resid_t = trad_rating[valid_idx] - X @ beta_t
        trad_map = {pids_s[vi]: resid_t[k] for k, vi in enumerate(valid_idx)}
        traditional_ratings[(metric, season)] = trad_map
        print(f"  {metric} {season}: converged in {it} iters, "
              f"max_delta={max_delta:.5f}")

# ---------------------------------------------------------------------------
# PHASE F: team aggregation + R² vs points for each stage.
# Team rating = TOI-weighted mean of player z-scores (or raw values).
# ---------------------------------------------------------------------------
print("[8] team aggregation + regressions ...")
def team_agg(ppdf_sub, rating_col, is_zscore=False, rating_map=None):
    """Aggregate player ratings to team using TOI weights.
    rating_map: dict (pid, season) -> value, overrides rating_col if provided."""
    df = ppdf_sub.copy()
    if rating_map is not None:
        df["r_"] = df.apply(lambda r: rating_map.get((r["player_id"],r["season"]), np.nan), axis=1)
    else:
        df["r_"] = df[rating_col]
    df = df.dropna(subset=["r_","team"])
    df["w"] = df["toi_sec"]
    g = df.groupby(["season","team"]).apply(
        lambda x: np.average(x["r_"], weights=x["w"])).rename("team_val").reset_index()
    return g

def r2(x, y):
    x = np.array(x,dtype=float); y = np.array(y,dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 5: return np.nan, np.nan, 0
    r = np.corrcoef(x[mask], y[mask])[0,1]
    return r*r, r, int(mask.sum())

summary_rows = []
for metric, raw_col in [("V5","cm_pct"),("Fenwick","fen_pct"),
                         ("Corsi","cor_pct"),("V1b","cm_pct")]:
    # Raw
    tb = team_agg(ppdf, raw_col).merge(tm, on=["season","team"])
    r_raw = r2(tb["team_val"], tb["points"])
    # ZA
    tb = team_agg(ppdf, f"ZA_{metric}").merge(tm, on=["season","team"])
    r_za = r2(tb["team_val"], tb["points"])
    # Traditional
    trad_map = {k: v for (met, _), d in traditional_ratings.items() if met==metric for k,v in d.items()}
    trad_rows = []
    for (met, season), d in traditional_ratings.items():
        if met != metric: continue
        for pid, v in d.items():
            trad_rows.append({"player_id":pid,"season":season,"trad":v})
    tr_df = pd.DataFrame(trad_rows)
    ppdf_tr = ppdf.merge(tr_df, on=["player_id","season"], how="left")
    tb = team_agg(ppdf_tr, "trad").merge(tm, on=["season","team"])
    r_trad = r2(tb["team_val"], tb["points"])
    # Converged
    conv_rows = []
    for (met, season), d in converged_ratings.items():
        if met != metric: continue
        for pid, v in d.items():
            conv_rows.append({"player_id":pid,"season":season,"conv":v})
    cv_df = pd.DataFrame(conv_rows)
    ppdf_cv = ppdf.merge(cv_df, on=["player_id","season"], how="left")
    tb = team_agg(ppdf_cv, "conv").merge(tm, on=["season","team"])
    r_conv = r2(tb["team_val"], tb["points"])
    summary_rows.append({
        "metric":metric,
        "R2_raw":r_raw[0],  "r_raw":r_raw[1],
        "R2_ZA": r_za[0],   "r_ZA": r_za[1],
        "R2_TRAD":r_trad[0],"r_TRAD":r_trad[1],
        "R2_CONV":r_conv[0],"r_CONV":r_conv[1],
        "N": r_conv[2],
    })

summary_df = pd.DataFrame(summary_rows)
print("\n=== STAGE 5 R² SUMMARY ===")
print(summary_df.to_string(index=False))

# ---------------------------------------------------------------------------
# PHASE G: write outputs
# ---------------------------------------------------------------------------
print("[9] writing outputs ...")
pd.DataFrame(conv_history_rows).to_csv(OUT / "stage4_convergence_history.csv", index=False)

# Stage 5 summary (merged with previous Stage 3 team-level benchmarks)
# Include previous table rows for xG, HD, PDO, CNFI+MNFI_team
prior = pd.DataFrame([
    {"metric":"V5_team","R2_raw":0.5955,"R2_ZA":0.6000,"R2_TRAD":np.nan,"R2_CONV":np.nan,"note":"team-agg Stage 1-3"},
    {"metric":"CNFI+MNFI_team","R2_raw":0.5955,"R2_ZA":0.6000,"R2_TRAD":np.nan,"R2_CONV":np.nan,"note":"team-agg"},
    {"metric":"xG%_team","R2_raw":0.5377,"R2_ZA":0.5484,"R2_TRAD":np.nan,"R2_CONV":np.nan,"note":"MoneyPuck team"},
    {"metric":"HD_CF%_team","R2_raw":0.4803,"R2_ZA":0.4818,"R2_TRAD":np.nan,"R2_CONV":np.nan,"note":"team-agg"},
    {"metric":"PDO_team","R2_raw":0.3360,"R2_ZA":np.nan,"R2_TRAD":np.nan,"R2_CONV":np.nan,"note":"neg control"},
])
stage5 = pd.concat([summary_df, prior], ignore_index=True)
stage5.to_csv(OUT / "stage5_r2_summary.csv", index=False)

# Player ratings per metric (Raw / ZA / Trad / Converged) + top-30 lists
for metric, raw_col in [("V5","cm_pct"),("Fenwick","fen_pct"),
                         ("Corsi","cor_pct"),("V1b","cm_pct")]:
    all_rows = []
    for _, r_ in ppdf.iterrows():
        pid = int(r_["player_id"]); season = r_["season"]
        trad_v = traditional_ratings.get((metric, season), {}).get(pid, np.nan)
        conv_v = converged_ratings.get((metric, season), {}).get(pid, np.nan)
        all_rows.append({
            "player_id":pid, "name":r_["name"], "team":r_["team"], "pos":r_["pos"],
            "season":season, "toi_min":r_["toi_sec"]/60, "oz_ratio":r_["oz_ratio"],
            "raw":r_[raw_col], "ZA":r_[f"ZA_{metric}"],
            "TRAD":trad_v, "CONV":conv_v,
        })
    mdf = pd.DataFrame(all_rows)
    # Filter to forwards for V1b
    if metric == "V1b":
        mdf = mdf[mdf["pos"]=="F"]
    mdf.to_csv(OUT / f"stage4_player_ratings_{metric}.csv", index=False)
    # Top-30 lists (career average of converged rating across seasons, weighted by TOI)
    # For ranking, use mean converged across seasons (players must appear in ≥2 seasons for stability)
    grp = mdf.groupby(["player_id","name","pos"]).agg(
        n_seasons=("season","count"),
        toi_total=("toi_min","sum"),
        raw_mean=("raw", lambda s: np.nanmean(s)),
        ZA_mean=("ZA",  lambda s: np.nanmean(s)),
        CONV_mean=("CONV", lambda s: np.nanmean(s)),
        team_recent=("team","last"),
    ).reset_index()
    grp = grp[grp["toi_total"]>=500]
    grp["rank_raw"]  = grp["raw_mean"].rank(ascending=False, method="min").astype(int)
    grp["rank_ZA"]   = grp["ZA_mean"].rank(ascending=False, method="min").astype(int)
    grp["rank_CONV"] = grp["CONV_mean"].rank(ascending=False, method="min").astype(int)
    grp["delta_ZA_to_CONV"] = grp["rank_ZA"] - grp["rank_CONV"]
    grp["delta_raw_to_CONV"] = grp["rank_raw"] - grp["rank_CONV"]
    grp = grp.sort_values("rank_CONV")
    grp.to_csv(OUT / f"stage6_player_lists_{metric}.csv", index=False)

# Cross-metric consistency
top30_cv = {}
for metric in ["V5","Fenwick","V1b"]:
    g = pd.read_csv(OUT / f"stage6_player_lists_{metric}.csv")
    top30_cv[metric] = set(g.head(30)["player_id"].astype(int))
elite = top30_cv["V5"] & top30_cv["Fenwick"] & top30_cv["V1b"]
elite_df = pd.DataFrame([{"player_id":pid, "name":name_map.get(pid,""), "pos":pos_map.get(pid,"")}
                          for pid in elite])
elite_df.to_csv(OUT / "stage6_cross_metric_elite.csv", index=False)
print(f"\nCross-metric elite (top 30 in V5 AND Fenwick AND V1b converged): {len(elite)} players")
print(elite_df.to_string(index=False))

# Stage 7 annual ratings export (per player per metric per season)
s7_rows = []
for metric in ["V5","Fenwick","Corsi","V1b"]:
    for _, r_ in ppdf.iterrows():
        pid = int(r_["player_id"]); season = r_["season"]
        s7_rows.append({
            "player_id":pid, "name":r_["name"], "team":r_["team"], "pos":r_["pos"],
            "season":season, "metric":metric, "toi_min":r_["toi_sec"]/60,
            "raw":r_[{"V5":"cm_pct","Fenwick":"fen_pct","Corsi":"cor_pct","V1b":"cm_pct"}[metric]],
            "ZA":r_[f"ZA_{metric}"],
            "TRAD":traditional_ratings.get((metric,season),{}).get(pid, np.nan),
            "CONV":converged_ratings.get((metric,season),{}).get(pid, np.nan),
        })
s7 = pd.DataFrame(s7_rows)
s7.to_csv(OUT / "stage7_annual_ratings.csv", index=False)

print(f"\n[done] all outputs in {OUT}")
