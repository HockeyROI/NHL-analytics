"""Stage 1-3 extension for V1a, V1b, V2 (forward-scoped metrics).

V1a — individual forward CNFI+MNFI shots per 60 of forward TOI
V1b — on-ice CNFI+MNFI% attributed to forwards only (team agg = sum forward
      on-ice CNFI+MNFI CF / sum forward on-ice (CF+CA) in CNFI+MNFI zones)
V2  — individual forward CNFI-only shots per 60

Reuses faceoff data from /tmp/dt_faceoffs.pkl. Re-runs shift attribution
with forward-specific counters.

Outputs (appends to existing decision tree dir):
  stage1_factor_base_v1v2.csv
  stage1_dz_only_r2_v1v2.csv
  stage2_factor_optimization_v1v2.csv
  stage3_zone_decision_v1v2.csv
"""
from __future__ import annotations

import json
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
GAME_IDS = ROOT / "Data/game_ids.csv"
OUT = ROOT / "NFI/output/zone_adjustment/complete_decision_tree"

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


print("[0] loading inputs ...")
games_df = pd.read_csv(GAME_IDS)
games_df = games_df[games_df["season"].astype(str).isin(POOLED_SEASONS)]
games_df = games_df[games_df["game_type"] == "regular"]
game_to_season = dict(zip(games_df["game_id"].astype(int),
                          games_df["season"].astype(str)))
game_ids = sorted(game_to_season.keys())

# Position lookup
pos_df = pd.read_csv(POS_CSV)
pos_map = dict(zip(pos_df["player_id"].astype(int), pos_df["pos_group"].astype(str)))
is_forward = lambda pid: pos_map.get(int(pid)) == "F"

# Team metrics
team_metrics = pd.read_csv(TEAM_METRICS)
team_metrics["season"] = team_metrics["season"].astype(str)
team_metrics = team_metrics[team_metrics["season"].isin(POOLED_SEASONS)].copy()
team_metrics["CNFI_MNFI_team"] = team_metrics["CNFI_pct"] + team_metrics["MNFI_pct"]

# ---------------------------------------------------------------------------
# Load faceoffs (cached from first run)
# ---------------------------------------------------------------------------
print("[1] loading cached faceoffs ...")
fo_df = pd.read_pickle("/tmp/dt_faceoffs.pkl")
fo_by_game = {g: grp[["t","winner","loser","zone_from_winner"]].to_numpy()
              for g, grp in fo_df.groupby("game_id", sort=False)}

# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------
print("[2] loading shifts ...")
shift_df = pd.read_csv(SHIFT_CSV,
    usecols=["game_id","player_id","team_abbrev","abs_start_secs","abs_end_secs"])
shift_df = shift_df.dropna(subset=["player_id","abs_start_secs","abs_end_secs"])
shift_df["game_id"] = shift_df["game_id"].astype("int64")
shift_df["player_id"] = shift_df["player_id"].astype("int64")
shift_df["abs_start_secs"] = shift_df["abs_start_secs"].astype("int32")
shift_df["abs_end_secs"] = shift_df["abs_end_secs"].astype("int32")
shift_df = shift_df[shift_df["game_id"].isin(game_ids)]
shift_df["team_abbrev"] = shift_df["team_abbrev"].astype(str).map(norm_team)
shift_df["is_fwd"] = shift_df["player_id"].map(lambda p: pos_map.get(p) == "F")
shift_df = shift_df.sort_values(["game_id","abs_start_secs"]).reset_index(drop=True)

shifts_by_game = {}
for gid, grp in shift_df.groupby("game_id", sort=False):
    shifts_by_game[gid] = (
        grp["player_id"].to_numpy(),
        grp["team_abbrev"].to_numpy(),
        grp["abs_start_secs"].to_numpy(),
        grp["abs_end_secs"].to_numpy(),
        grp["is_fwd"].to_numpy(),
    )

# ---------------------------------------------------------------------------
# Shots (ES only) with shooter position + CNFI flag
# ---------------------------------------------------------------------------
print("[3] loading shots ...")
shots_df = pd.read_csv(SHOTS_CSV,
    usecols=["game_id","season","abs_time","event_type","shooting_team_abbrev",
             "shooter_player_id","zone","state"])
shots_df["season"] = shots_df["season"].astype(str)
shots_df = shots_df[shots_df["season"].isin(POOLED_SEASONS)]
shots_df = shots_df[shots_df["state"] == "ES"]
shots_df["shooting_team_abbrev"] = shots_df["shooting_team_abbrev"].astype(str).map(norm_team)
shots_df["shooter_is_fwd"] = shots_df["shooter_player_id"].map(
    lambda p: pos_map.get(int(p)) == "F" if pd.notna(p) else False)
shots_df["is_fenwick"] = shots_df["event_type"].isin(FENWICK_TYPES)
shots_df["is_corsi"]   = shots_df["event_type"].isin(CORSI_TYPES)
shots_df["is_cnfi"]    = shots_df["zone"] == "CNFI"
shots_df["is_mnfi"]    = shots_df["zone"] == "MNFI"
shots_df["is_cnfi_mnfi"] = shots_df["is_cnfi"] | shots_df["is_mnfi"]
shots_df = shots_df.sort_values(["game_id","abs_time"]).reset_index(drop=True)

shots_by_game = {}
for gid, grp in shots_df.groupby("game_id", sort=False):
    shots_by_game[gid] = {
        "t": grp["abs_time"].to_numpy(),
        "team": grp["shooting_team_abbrev"].to_numpy(),
        "fwd_shooter": grp["shooter_is_fwd"].to_numpy(),
        "is_corsi": grp["is_corsi"].to_numpy(),
        "cnfi_mnfi": grp["is_cnfi_mnfi"].to_numpy(),
        "cnfi": grp["is_cnfi"].to_numpy(),
    }

# ---------------------------------------------------------------------------
# STAGE 1 attribution — forward-scoped counters.
# For every OZ/DZ faceoff, for every on-ice FORWARD player, accumulate per
# (team, season, zone_from_player_POV):
#   n_fwd_shifts, fwd_sec
#   fwd_cm_for, fwd_cm_against  (CNFI+MNFI on-ice for V1b)
#   fwd_ind_cm                   (individual CNFI+MNFI shots by any team forward during shift remainder)
#   fwd_ind_cnfi                 (individual CNFI shots by any team forward during shift remainder)
# Using forward-only on-ice shifts as the attribution basis.
# ---------------------------------------------------------------------------
print("[4] shift-based attribution (forward-scoped) ...")
agg_keys = ("n","sec","cm_f","cm_a","ind_cm","ind_cnfi")
team_agg = defaultdict(lambda: {k: 0.0 for k in agg_keys})

for gi, gid in enumerate(game_ids):
    if gid not in fo_by_game or gid not in shifts_by_game:
        continue
    season = game_to_season[gid]
    shots = shots_by_game.get(gid)
    fos = fo_by_game[gid]
    pids, teams, starts, ends, is_fwds = shifts_by_game[gid]
    if shots is not None:
        shot_t = shots["t"]

    for t_face, winner, loser, zone_w in fos:
        hi = bisect_right(starts, t_face)
        for i in range(hi):
            if not is_fwds[i]:   # forward-only
                continue
            if ends[i] <= t_face:
                continue
            p_team = teams[i]
            shift_end = ends[i]
            zone_p = zone_w if p_team == winner else FLIP[zone_w]
            if zone_p not in ("O", "D"):
                continue
            remainder = shift_end - t_face
            if shots is not None and len(shot_t) > 0:
                a = bisect_left(shot_t, t_face)
                b = bisect_right(shot_t, shift_end)
                if b > a:
                    sl_team = shots["team"][a:b]
                    is_own = (sl_team == p_team)
                    cm_arr = shots["cnfi_mnfi"][a:b]
                    cm_f = int(np.sum(cm_arr & is_own))
                    cm_a = int(np.sum(cm_arr & ~is_own))
                    fwd_sh = shots["fwd_shooter"][a:b]
                    cm_ind = int(np.sum(cm_arr & is_own & fwd_sh))
                    cnfi_ind = int(np.sum(shots["cnfi"][a:b] & is_own & fwd_sh))
                else:
                    cm_f = cm_a = cm_ind = cnfi_ind = 0
            else:
                cm_f = cm_a = cm_ind = cnfi_ind = 0
            key = (p_team, season, zone_p)
            d = team_agg[key]
            d["n"]        += 1
            d["sec"]      += remainder
            d["cm_f"]     += cm_f
            d["cm_a"]     += cm_a
            d["ind_cm"]   += cm_ind
            d["ind_cnfi"] += cnfi_ind
    if (gi + 1) % 1000 == 0 or gi + 1 == len(game_ids):
        print(f"    {gi+1}/{len(game_ids)}")

# ---------------------------------------------------------------------------
# STAGE 1a — Factor_base per metric (league pooled).
# V1a: individual forward CNFI+MNFI shots per forward-shift
# V1b: forward on-ice CNFI+MNFI (for) per forward-shift
# V2:  individual forward CNFI shots per forward-shift
# ---------------------------------------------------------------------------
rows = [{"team": k[0], "season": k[1], "zone": k[2], **v}
        for k, v in team_agg.items()]
ta = pd.DataFrame(rows)

oz = ta[ta["zone"]=="O"]; dz = ta[ta["zone"]=="D"]
nO, nD = oz["n"].sum(), dz["n"].sum()

def pool_fb(col):
    oz_m = oz[col].sum() / nO
    dz_m = dz[col].sum() / nD
    return oz_m, dz_m, oz_m - dz_m

fb_v1a  = pool_fb("ind_cm")     # individual F CNFI+MNFI shots per forward-shift
fb_v1b  = pool_fb("cm_f")       # F on-ice CNFI+MNFI (for) per forward-shift
fb_v2   = pool_fb("ind_cnfi")   # individual F CNFI shots per forward-shift

factor_df = pd.DataFrame([
    {"metric":"V1a","oz_mean":fb_v1a[0],"dz_mean":fb_v1a[1],"Factor_base":fb_v1a[2]},
    {"metric":"V1b","oz_mean":fb_v1b[0],"dz_mean":fb_v1b[1],"Factor_base":fb_v1b[2]},
    {"metric":"V2" ,"oz_mean":fb_v2[0] ,"dz_mean":fb_v2[1] ,"Factor_base":fb_v2[2] },
])
factor_df.to_csv(OUT / "stage1_factor_base_v1v2.csv", index=False)
print("\n[1a] Factor_base:")
print(factor_df)

# ---------------------------------------------------------------------------
# STAGE 1b — DZ-only team-season metrics + R² vs points.
# V1a = ind_cm_DZ / sec_DZ × 3600
# V1b = cm_f_DZ / (cm_f_DZ + cm_a_DZ)
# V2  = ind_cnfi_DZ / sec_DZ × 3600
# ---------------------------------------------------------------------------
piv = ta.pivot_table(index=["team","season"], columns="zone",
                     values=list(agg_keys), aggfunc="sum", fill_value=0)
piv.columns = [f"{a}_{b}" for a,b in piv.columns]
piv = piv.reset_index()

piv["V1a_DZ"] = piv["ind_cm_D"] / piv["sec_D"].replace(0,np.nan) * 3600.0
piv["V1b_DZ"] = piv["cm_f_D"]   / (piv["cm_f_D"] + piv["cm_a_D"]).replace(0,np.nan)
piv["V2_DZ"]  = piv["ind_cnfi_D"] / piv["sec_D"].replace(0,np.nan) * 3600.0
piv["OZ_ratio"] = piv["n_O"] / (piv["n_O"] + piv["n_D"]).replace(0,np.nan)

# Full-sample raw team metrics (not DZ-only).
# V1a_raw = total individual forward CNFI+MNFI shots at 5v5 ES / forward shift-seconds × 3600
# V1b_raw = (forward on-ice CNFI+MNFI CF across all shift-remainders)
#          / (forward on-ice CNFI+MNFI CF+CA across all shift-remainders)
# V2_raw  = total individual forward CNFI shots at 5v5 ES / forward shift-seconds × 3600
# Computed by summing across OZ+DZ shifts (we drop NZ because attribution
# only captured O/D faceoff-originated shifts — same caveat as other metrics).
piv["ind_cm_all"]   = piv["ind_cm_O"]   + piv["ind_cm_D"]
piv["ind_cnfi_all"] = piv["ind_cnfi_O"] + piv["ind_cnfi_D"]
piv["cm_f_all"]     = piv["cm_f_O"]     + piv["cm_f_D"]
piv["cm_a_all"]     = piv["cm_a_O"]     + piv["cm_a_D"]
piv["sec_all"]      = piv["sec_O"]      + piv["sec_D"]
piv["V1a_raw"] = piv["ind_cm_all"]   / piv["sec_all"].replace(0,np.nan) * 3600.0
piv["V1b_raw"] = piv["cm_f_all"]     / (piv["cm_f_all"] + piv["cm_a_all"]).replace(0,np.nan)
piv["V2_raw"]  = piv["ind_cnfi_all"] / piv["sec_all"].replace(0,np.nan) * 3600.0

# Merge points
m = piv.merge(team_metrics[["season","team","points"]], on=["season","team"], how="inner")
print(f"\n[1b] matched team-seasons: N = {len(m)}")

def r2_simple(x, y):
    vx, vy = x.astype(float), y.astype(float)
    mask = vx.notna() & vy.notna()
    vx, vy = vx[mask], vy[mask]
    r = np.corrcoef(vx, vy)[0,1]
    return r*r, r, len(vx)

dz_rows = []
for name, col in [("V1a","V1a_DZ"), ("V1b","V1b_DZ"), ("V2","V2_DZ")]:
    r2, r, n = r2_simple(m[col], m["points"])
    p = st.pearsonr(m[col].dropna(), m.loc[m[col].dropna().index, "points"]).pvalue
    dz_rows.append({"metric":name,"N":n,"r":r,"R2_DZ":r2,"p":p})
dz_df = pd.DataFrame(dz_rows)
dz_df.to_csv(OUT / "stage1_dz_only_r2_v1v2.csv", index=False)
print("\nDZ-only R² per metric:")
print(dz_df.to_string(index=False))

# Raw R²
raw_rows = []
for name, col in [("V1a","V1a_raw"), ("V1b","V1b_raw"), ("V2","V2_raw")]:
    r2, r, n = r2_simple(m[col], m["points"])
    raw_rows.append({"metric":name,"N":n,"r":r,"R2_raw":r2})
raw_df = pd.DataFrame(raw_rows)
print("\nRaw R² per metric:")
print(raw_df.to_string(index=False))

# ---------------------------------------------------------------------------
# STAGE 2 — zone adjustment sweep.
# raw_adj = raw - (OZ_ratio - 0.5) * Factor_base * multiplier
# Factor_base and raw are in same units (per forward-shift attempts
# converted to per-60 via sec, but FB is in attempts-per-shift and raw is
# per-60. Normalize FB to per-60 using avg remainder duration).
# ---------------------------------------------------------------------------
# Avg shift remainder (sec) league-wide
avg_remainder = ta["sec"].sum() / ta["n"].sum()
scale_per60 = 3600.0 / avg_remainder

# Per-60 Factor_base (same units as V1a/V2 raw)
fb_v1a_per60 = fb_v1a[2] * scale_per60
fb_v2_per60  = fb_v2[2]  * scale_per60
# For V1b (pct ratio), convert FB to pct-pt delta analogous to Corsi:
oz_cm_for = oz["cm_f"].sum(); oz_cm_ag = oz["cm_a"].sum()
dz_cm_for = dz["cm_f"].sum(); dz_cm_ag = dz["cm_a"].sum()
oz_pct = oz_cm_for / (oz_cm_for + oz_cm_ag)
dz_pct = dz_cm_for / (dz_cm_for + dz_cm_ag)
fb_v1b_pct = oz_pct - dz_pct
print(f"\n[2] FB per-60 V1a={fb_v1a_per60:.4f}  V2={fb_v2_per60:.4f}  "
      f"V1b_pct_delta={fb_v1b_pct:.4f}")

MULTIPLIERS = [round(0.25*i,2) for i in range(1,13)]

def sweep(raw_col, fb_val, name):
    raw = m[raw_col].astype(float)
    oz_r = m["OZ_ratio"].astype(float)
    y = m["points"].astype(float)
    out = []
    for mult in MULTIPLIERS:
        adj = raw - fb_val * mult * (oz_r - 0.5)
        r2, r, n = r2_simple(adj, y)
        out.append({"metric":name,"multiplier":mult,
                    "effective_factor":fb_val*mult,
                    "R2":r2,"r":r,"N":n})
    return out

sweep_rows = []
sweep_rows += sweep("V1a_raw", fb_v1a_per60, "V1a")
sweep_rows += sweep("V1b_raw", fb_v1b_pct,   "V1b")
sweep_rows += sweep("V2_raw",  fb_v2_per60,  "V2")
sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(OUT / "stage2_factor_optimization_v1v2.csv", index=False)

print("\n[2] Stage 2 sweep — top-3 per metric:")
for name, grp in sweep_df.groupby("metric"):
    best = grp.sort_values("R2", ascending=False).head(3)
    print(f"\n=== {name} ===")
    print(best.to_string(index=False))

# ---------------------------------------------------------------------------
# STAGE 3 — decision table.
# ---------------------------------------------------------------------------
print("\n[3] Stage 3 decision:")
decision = []
for name in ["V1a","V1b","V2"]:
    raw_r2 = raw_df[raw_df["metric"]==name]["R2_raw"].iloc[0]
    dz_r2  = dz_df[dz_df["metric"]==name]["R2_DZ"].iloc[0]
    za_row = sweep_df[sweep_df["metric"]==name].sort_values("R2",ascending=False).iloc[0]
    winner = "DZ" if dz_r2 > za_row["R2"] else "ZA"
    decision.append({
        "metric": name,
        "R2_raw": raw_r2,
        "R2_DZ_only": dz_r2,
        "R2_ZA_optimal": za_row["R2"],
        "ZA_best_multiplier": za_row["multiplier"],
        "winner_methodology": winner,
        "N": int(za_row["N"]),
    })
dec_df = pd.DataFrame(decision)
dec_df.to_csv(OUT / "stage3_zone_decision_v1v2.csv", index=False)
print(dec_df.to_string(index=False))

print("\n[done]")
