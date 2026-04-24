"""Definitive 5-metric zone-adjustment factor comparison.

Metrics:
  1. Regular Corsi CF%     — rink-wide, all events including blocked shots
  2. HD Corsi CF%           — HD filter (x>69, |y|<=22) WITH abs() correction for blocks
  3. Regular Fenwick FF%    — rink-wide, Fenwick events only
  4. HD Fenwick FF%         — HD filter, Fenwick only
  5. NFI% (CNFI+MNFI% Fenwick)

For each:
  - Empirical zone factor: league-pooled OZ%−DZ% pct-pt gap from shift-based faceoff attribution
  - Raw, ZA_traditional (3.5pp), ZA_empirical (correct factor) R² vs standings points (N=126)
"""
from __future__ import annotations

import json, pickle
from pathlib import Path
from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "NFI/output/zone_adjustment"
OUT.mkdir(parents=True, exist_ok=True)
SHOTS_CSV = ROOT / "NFI/output/shots_tagged.csv"
SHIFT_CSV = ROOT / "NFI/Geometry_post/Data/shift_data.csv"
TEAM_METRICS = ROOT / "NFI/output/team_level_all_metrics.csv"
GAME_IDS = ROOT / "Data/game_ids.csv"

ABBR = {"ARI":"UTA"}
POOLED = {"20222023","20232024","20242025","20252026"}
FLIP = {"O":"D","D":"O","N":"N"}
FEN = {"shot-on-goal","missed-shot","goal"}
COR = FEN | {"blocked-shot"}
def norm(a): return ABBR.get(a,a) if a else a

print("[0] loading ...")
games = pd.read_csv(GAME_IDS)
games = games[(games["season"].astype(str).isin(POOLED)) & (games["game_type"]=="regular")]
games["home_abbrev"] = games["home_abbrev"].astype(str).map(norm)
games["away_abbrev"] = games["away_abbrev"].astype(str).map(norm)
g2s = dict(zip(games["game_id"].astype(int), games["season"].astype(str)))
gids = sorted(g2s.keys())

# Cached faceoffs
fo_df = pd.read_pickle("/tmp/dt_faceoffs.pkl")
fo_by_g = {g: grp[["t","winner","loser","zone_from_winner"]].to_numpy()
           for g, grp in fo_df.groupby("game_id", sort=False)}

# Shifts
print("[1] loading shifts ...")
sd = pd.read_csv(SHIFT_CSV,
    usecols=["game_id","player_id","team_abbrev","abs_start_secs","abs_end_secs"])
sd = sd.dropna(subset=["player_id","abs_start_secs","abs_end_secs"])
sd["game_id"] = sd["game_id"].astype("int64")
sd = sd[sd["game_id"].isin(gids)]
sd["abs_start_secs"] = sd["abs_start_secs"].astype("int32")
sd["abs_end_secs"]   = sd["abs_end_secs"].astype("int32")
sd["team_abbrev"] = sd["team_abbrev"].astype(str).map(norm)
sd = sd.sort_values(["game_id","abs_start_secs"]).reset_index(drop=True)
sbg = {gid: (g["team_abbrev"].to_numpy(),
             g["abs_start_secs"].to_numpy(), g["abs_end_secs"].to_numpy())
       for gid, g in sd.groupby("game_id", sort=False)}

# Shots with HD-shooter-POV coordinate correction
print("[2] loading shots + computing HD flag with block coord fix ...")
xd = pd.read_csv(SHOTS_CSV,
    usecols=["game_id","season","abs_time","event_type","shooting_team_abbrev",
             "x_coord_norm","y_coord_norm","zone","state"])
xd["season"] = xd["season"].astype(str)
xd = xd[(xd["season"].isin(POOLED)) & (xd["state"]=="ES")]
xd["shooting_team_abbrev"] = xd["shooting_team_abbrev"].astype(str).map(norm)
xd["x_shooter"] = np.where(xd["event_type"]=="blocked-shot", -xd["x_coord_norm"], xd["x_coord_norm"])
xd["is_fen"] = xd["event_type"].isin(FEN)
xd["is_cor"] = xd["event_type"].isin(COR)
xd["is_hd"]  = (xd["x_shooter"] > 69) & (xd["y_coord_norm"].abs() <= 22)
xd["is_cm"]  = xd["zone"].isin(["CNFI","MNFI"])
# Derived flags
xd["is_hd_cor"] = xd["is_hd"] & xd["is_cor"]
xd["is_hd_fen"] = xd["is_hd"] & xd["is_fen"]
xd = xd.sort_values(["game_id","abs_time"]).reset_index(drop=True)
xbg = {gid: {"t": g["abs_time"].to_numpy(),
             "team": g["shooting_team_abbrev"].to_numpy(),
             "cor": g["is_cor"].to_numpy(),
             "fen": g["is_fen"].to_numpy(),
             "hd_cor": g["is_hd_cor"].to_numpy(),
             "hd_fen": g["is_hd_fen"].to_numpy(),
             "cm":  g["is_cm"].to_numpy()}
       for gid, g in xd.groupby("game_id", sort=False)}

# Shift-based attribution — track all 5 metric counters per zone
print("[3] shift-based OZ/DZ attribution ...")
ag = defaultdict(lambda: {
    "n":0, "sec":0.0,
    "cf":0,"ca":0, "ff":0,"fa":0,
    "hd_cf_f":0,"hd_cf_a":0,
    "hd_ff_f":0,"hd_ff_a":0,
    "cm_f":0,"cm_a":0})

for gi, gid in enumerate(gids):
    if gid not in fo_by_g or gid not in sbg: continue
    season = g2s[gid]
    shots = xbg.get(gid)
    teams, starts, ends = sbg[gid]
    st_arr = shots["t"] if shots else None
    for t_face, winner, loser, zone_w in fo_by_g[gid]:
        hi = bisect_right(starts, t_face)
        for i in range(hi):
            if ends[i] <= t_face: continue
            p_team = teams[i]; se = ends[i]
            zp = zone_w if p_team == winner else FLIP[zone_w]
            if zp not in ("O","D"): continue
            rem = se - t_face
            cf=ca=ff=fa=hdc_f=hdc_a=hdf_f=hdf_a=cm_f=cm_a = 0
            if st_arr is not None and len(st_arr) > 0:
                a = bisect_left(st_arr, t_face); b = bisect_right(st_arr, se)
                if b > a:
                    sl_team = shots["team"][a:b]
                    own = (sl_team == p_team)
                    cor_arr = shots["cor"][a:b]; fen_arr = shots["fen"][a:b]
                    hdc_arr = shots["hd_cor"][a:b]; hdf_arr = shots["hd_fen"][a:b]
                    cm_arr  = shots["cm"][a:b]
                    cf = int(np.sum(cor_arr & own));  ca = int(np.sum(cor_arr & ~own))
                    ff = int(np.sum(fen_arr & own));  fa = int(np.sum(fen_arr & ~own))
                    hdc_f = int(np.sum(hdc_arr & own));  hdc_a = int(np.sum(hdc_arr & ~own))
                    hdf_f = int(np.sum(hdf_arr & own));  hdf_a = int(np.sum(hdf_arr & ~own))
                    cm_f  = int(np.sum(cm_arr & own));   cm_a  = int(np.sum(cm_arr & ~own))
            d = ag[(p_team, season, zp)]
            d["n"] += 1; d["sec"] += rem
            d["cf"] += cf; d["ca"] += ca
            d["ff"] += ff; d["fa"] += fa
            d["hd_cf_f"] += hdc_f; d["hd_cf_a"] += hdc_a
            d["hd_ff_f"] += hdf_f; d["hd_ff_a"] += hdf_a
            d["cm_f"] += cm_f; d["cm_a"] += cm_a
    if (gi+1) % 1000 == 0 or gi+1 == len(gids): print(f"    {gi+1}/{len(gids)}")

rows = [{"team":k[0],"season":k[1],"zone":k[2],**v} for k,v in ag.items()]
ta = pd.DataFrame(rows)
print(f"    team-season-zone rows: {len(ta)}")

# --- Empirical zone factors (league pooled pct-pt OZ-DZ) ---
print("\n[4] empirical factors (league-pooled OZ%−DZ% gap) ...")
oz = ta[ta["zone"]=="O"]; dz = ta[ta["zone"]=="D"]
def gap(col_f, col_a):
    o = oz[col_f].sum()/(oz[col_f].sum()+oz[col_a].sum())
    d = dz[col_f].sum()/(dz[col_f].sum()+dz[col_a].sum())
    return o, d, o-d

reg_cor = gap("cf","ca")
hd_cor  = gap("hd_cf_f","hd_cf_a")
reg_fen = gap("ff","fa")
hd_fen  = gap("hd_ff_f","hd_ff_a")
nfi     = gap("cm_f","cm_a")

factors = [
    ("Regular Corsi CF%", "cf","ca",  reg_cor),
    ("HD Corsi CF%",      "hd_cf_f","hd_cf_a", hd_cor),
    ("Regular Fenwick FF%","ff","fa", reg_fen),
    ("HD Fenwick FF%",    "hd_ff_f","hd_ff_a", hd_fen),
    ("NFI% (CNFI+MNFI% Fenwick)", "cm_f","cm_a", nfi),
]
print(f"\n{'metric':<30} {'OZ%':>8} {'DZ%':>8} {'factor (pp)':>14}")
for name, _, _, (o, d, g) in factors:
    print(f"{name:<30} {o*100:>8.2f} {d*100:>8.2f} {g*100:>+14.3f}")

# --- Team-level raw metric per team-season ---
print("\n[5] team-level metric variants + regressions ...")
piv = ta.pivot_table(index=["team","season"], columns="zone",
                     values=["n","cf","ca","ff","fa","hd_cf_f","hd_cf_a",
                             "hd_ff_f","hd_ff_a","cm_f","cm_a"],
                     aggfunc="sum", fill_value=0)
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
piv = piv.reset_index()

def team_pct(f_o, a_o, f_d, a_d):
    F = f_o + f_d; A = a_o + a_d
    return F / (F + A).replace(0, np.nan)

piv["Corsi_raw"]   = team_pct(piv["cf_O"], piv["ca_O"], piv["cf_D"], piv["ca_D"])
piv["HDCorsi_raw"] = team_pct(piv["hd_cf_f_O"], piv["hd_cf_a_O"], piv["hd_cf_f_D"], piv["hd_cf_a_D"])
piv["Fen_raw"]     = team_pct(piv["ff_O"], piv["fa_O"], piv["ff_D"], piv["fa_D"])
piv["HDFen_raw"]   = team_pct(piv["hd_ff_f_O"], piv["hd_ff_a_O"], piv["hd_ff_f_D"], piv["hd_ff_a_D"])
piv["NFI_raw"]     = team_pct(piv["cm_f_O"], piv["cm_a_O"], piv["cm_f_D"], piv["cm_a_D"])
piv["OZ_ratio"]    = piv["n_O"] / (piv["n_O"] + piv["n_D"])

# Merge team points
tm = pd.read_csv(TEAM_METRICS)
tm["season"] = tm["season"].astype(str)
tm = tm[tm["season"].isin(POOLED)][["season","team","points"]]
m = piv.merge(tm, on=["season","team"], how="inner")
print(f"    matched team-seasons: N = {len(m)}")

def r2(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 5: return (np.nan, np.nan)
    r = np.corrcoef(x[mask], y[mask])[0,1]
    return r*r, r

def apply_za(raw_col, factor):
    return m[raw_col] - factor * (m["OZ_ratio"] - 0.5)

TRAD = 0.035
report_rows = []
for (name, _, _, (o, d, g)), raw_col in zip(factors, ["Corsi_raw","HDCorsi_raw","Fen_raw","HDFen_raw","NFI_raw"]):
    r_raw    = r2(m[raw_col], m["points"])
    r_trad   = r2(apply_za(raw_col, TRAD), m["points"])
    r_emp    = r2(apply_za(raw_col, g),    m["points"])
    under_over = TRAD - g   # if positive → traditional is higher → OVER-correcting
    pct_err    = 100.0 * (TRAD - g) / g if g != 0 else np.nan
    report_rows.append({
        "metric": name,
        "OZ_pct": o*100, "DZ_pct": d*100, "empirical_factor_pp": g*100,
        "raw_R2": r_raw[0], "raw_r": r_raw[1],
        "ZA_trad_R2": r_trad[0], "ZA_trad_r": r_trad[1],
        "ZA_emp_R2":  r_emp[0],  "ZA_emp_r":  r_emp[1],
        "trad_error_pp": (TRAD - g)*100,
        "trad_error_pct": pct_err,
    })

rep = pd.DataFrame(report_rows).sort_values("ZA_emp_R2", ascending=False)
rep.to_csv(OUT / "complete_factor_comparison.csv", index=False)

print("\n" + "="*100)
print("DEFINITIVE 5-METRIC COMPARISON (N=126 pooled)")
print("="*100)
with pd.option_context("display.max_columns", None, "display.width", 160,
                       "display.float_format", "{:.4f}".format):
    print(rep.to_string(index=False))

# --- Analysis: why is NFI% factor lower than all-zone Fenwick? ---
print("\n" + "="*100)
print("NFI% vs all-zone Fenwick factor analysis")
print("="*100)
# Show per-side block rates in CNFI+MNFI zones vs overall
oz_cm_f = oz["cm_f"].sum(); oz_cm_a = oz["cm_a"].sum()
dz_cm_f = dz["cm_f"].sum(); dz_cm_a = dz["cm_a"].sum()
print(f"\nNFI% is FENWICK (blocks already excluded by zone-tag behavior in shots_tagged)")
print(f"\n{'zone':<6} {'NFI for/sh':>14} {'NFI ag/sh':>14} {'For-share':>12}")
print(f"{'OZ':<6} {oz_cm_f/oz['n'].sum():>14.4f} {oz_cm_a/oz['n'].sum():>14.4f} {oz_cm_f/(oz_cm_f+oz_cm_a)*100:>11.2f}%")
print(f"{'DZ':<6} {dz_cm_f/dz['n'].sum():>14.4f} {dz_cm_a/dz['n'].sum():>14.4f} {dz_cm_f/(dz_cm_f+dz_cm_a)*100:>11.2f}%")

# Compare gaps
print(f"\nPer-shift rates:")
print(f"  Regular Fenwick OZ: {oz['ff'].sum()/oz['n'].sum():.4f}   DZ: {dz['ff'].sum()/dz['n'].sum():.4f}")
print(f"  NFI             OZ: {oz['cm_f'].sum()/oz['n'].sum():.4f}   DZ: {dz['cm_f'].sum()/dz['n'].sum():.4f}")
print(f"  HD Fenwick      OZ: {oz['hd_ff_f'].sum()/oz['n'].sum():.4f}   DZ: {dz['hd_ff_f'].sum()/dz['n'].sum():.4f}")

# NFI shots as fraction of all Fenwick shots in each zone
nfi_oz_share = oz_cm_f / oz["ff"].sum()
nfi_dz_share = dz_cm_f / dz["ff"].sum()
hdf_oz_share = oz["hd_ff_f"].sum() / oz["ff"].sum()
hdf_dz_share = dz["hd_ff_f"].sum() / dz["ff"].sum()
print(f"\nCNFI+MNFI share of total Fenwick by zone:")
print(f"  OZ: NFI {oz_cm_f:,.0f} / Fenwick {oz['ff'].sum():,.0f} = {nfi_oz_share*100:.2f}%")
print(f"  DZ: NFI {dz_cm_f:,.0f} / Fenwick {dz['ff'].sum():,.0f} = {nfi_dz_share*100:.2f}%")
print(f"\nHD share of total Fenwick by zone:")
print(f"  OZ: HD  {oz['hd_ff_f'].sum():,.0f} / Fenwick {oz['ff'].sum():,.0f} = {hdf_oz_share*100:.2f}%")
print(f"  DZ: HD  {dz['hd_ff_f'].sum():,.0f} / Fenwick {dz['ff'].sum():,.0f} = {hdf_dz_share*100:.2f}%")

# Traditional 3.5 pp error per metric
print("\n-- Traditional 3.5 pp error per metric --")
print(f"{'metric':<30} {'empirical':>11} {'trad':>8} {'error (pp)':>12} {'error (%)':>11}")
for row in rep.itertuples():
    print(f"{row.metric:<30} {row.empirical_factor_pp:>+10.2f}pp {TRAD*100:>+6.2f}pp "
          f"{row.trad_error_pp:>+11.2f}pp {row.trad_error_pct:>+10.2f}%")
