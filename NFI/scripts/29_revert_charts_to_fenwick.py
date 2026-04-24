#!/usr/bin/env python3
"""
Revert chart-data files to Fenwick (SOG + missed + goal; blocked shots
excluded entirely). Matches the methodology used for P1a, P2, P4, P5
player metrics.

Overwrites:
  NFI/output/heatmap_publication.csv
  NFI/output/zone_conversion_bars.csv
  NFI/output/yband_dropoff.csv

Pre-Fenwick versions backed up to NFI/output/_pre_fenwick_backup/.

Save_pct in the heatmap continues to use (SOG + goal) as denominator
(standard goalie-save-percentage convention; blocked shots never reach the
goalie regardless of Corsi/Fenwick choice). The Fenwick filter only changes
conversion_rate.
"""
import math
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
BK   = f"{OUT}/_pre_fenwick_backup"

SEASONS = {"20212022","20222023","20232024","20242025","20252026"}

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0.0, c-h), min(1.0, c+h))

# Load Fenwick shots (drop blocked entirely; ES regulation only)
print("Loading shots (Fenwick: SOG + missed + goal) ...")
cols = ["season","period","situation_code","event_type",
        "x_coord_norm","y_coord_norm","is_goal"]
shots = pd.read_csv(SHOT_CSV, usecols=cols,
                    dtype={"season":str,"situation_code":str})
shots = shots[shots["season"].isin(SEASONS)].copy()
shots = shots[shots["period"].between(1,3)].copy()
shots = shots[shots["situation_code"].astype(str)=="1551"].copy()
shots = shots[shots["event_type"].isin(
    ["shot-on-goal","missed-shot","goal"])].copy()  # NO blocked-shot
shots = shots.dropna(subset=["x_coord_norm","y_coord_norm"])
print(f"  Fenwick ES regulation shots: {len(shots):,}")

xs = shots["x_coord_norm"].values
ys = shots["y_coord_norm"].values
goals = shots["is_goal"].values
ets = shots["event_type"].values

def zone_label(x, y):
    absy = abs(y)
    if 74 <= x <= 89 and absy <= 9:   return "CNFI"
    if 55 <= x <= 73 and absy <= 15:  return "MNFI"
    if 25 <= x <= 54 and absy <= 15:  return "FNFI"
    return "Wide"

# ---- File 1 — heatmap_publication.csv (Fenwick) ----
print("Rebuilding heatmap_publication.csv ...")
x_edges = np.arange(25, 95, 5)
y_edges = np.arange(-40, 45, 5)
heatmap_rows = []
for ix in range(len(x_edges)-1):
    for iy in range(len(y_edges)-1):
        xlo, xhi = x_edges[ix], x_edges[ix+1]
        ylo, yhi = y_edges[iy], y_edges[iy+1]
        m = (xs >= xlo) & (xs < xhi) & (ys >= ylo) & (ys < yhi)
        n = int(m.sum())
        if n < 50: continue
        cell_goals = int(goals[m].sum())
        conv = cell_goals / n
        cell_ets = ets[m]
        # save_pct still uses SOG+goal denom (standard) — Fenwick excludes
        # blocked, but missed-shot is also not "faced". So shots_faced = SOG+goal.
        faced = int(((cell_ets=="shot-on-goal") | (cell_ets=="goal")).sum())
        save_pct = (1.0 - cell_goals/faced) if faced > 0 else np.nan
        heatmap_rows.append({
            "x_center": (xlo+xhi)/2.0, "y_center": (ylo+yhi)/2.0,
            "x_min": xlo, "x_max": xhi, "y_min": ylo, "y_max": yhi,
            "attempts": n, "goals": cell_goals,
            "conversion_rate": round(conv, 5),
            "shots_faced": faced,
            "save_pct": round(save_pct, 5) if not np.isnan(save_pct) else np.nan,
            "zone_label": zone_label((xlo+xhi)/2.0, (ylo+yhi)/2.0),
        })
heatmap = pd.DataFrame(heatmap_rows)
heatmap.to_csv(f"{OUT}/heatmap_publication.csv", index=False)
print(f"  heatmap_publication.csv: {len(heatmap)} cells (Fenwick)")

# ---- File 2 — zone_conversion_bars.csv (Fenwick) ----
print("Rebuilding zone_conversion_bars.csv ...")
ZONE_DEFS = [
    {"zone":"CNFI",            "x_min":74, "x_max":89, "y_min":-9,  "y_max":9,  "color_hex":"#FF6B35"},
    {"zone":"MNFI",            "x_min":55, "x_max":73, "y_min":-15, "y_max":15, "color_hex":"#2E7DC4"},
    {"zone":"FNFI",            "x_min":25, "x_max":54, "y_min":-15, "y_max":15, "color_hex":"#4AB3E8"},
    {"zone":"Wide",            "x_min":25, "x_max":89, "y_min":15,  "y_max":42, "color_hex":"#888888"},
    {"zone":"HD_conventional", "x_min":69, "x_max":89, "y_min":-22, "y_max":22, "color_hex":"#CC3333"},
    {"zone":"inner_slot",      "x_min":69, "x_max":89, "y_min":-14, "y_max":14, "color_hex":"#FFB700"},
    {"zone":"slot",            "x_min":25, "x_max":89, "y_min":-22, "y_max":22, "color_hex":"#44AA66"},
]

def conv_for(xmin, xmax, ymin, ymax, allow_y_either_side=False):
    if allow_y_either_side:
        m = (xs >= xmin) & (xs <= xmax) & (np.abs(ys) >= ymin) & (np.abs(ys) <= ymax)
    else:
        m = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
    n = int(m.sum())
    g = int(goals[m].sum())
    p, lo, hi = wilson(g, n)
    return n, g, p, lo, hi

bar_rows = []
for z in ZONE_DEFS:
    if z["zone"] == "Wide":
        n, gc, p, lo, hi = conv_for(z["x_min"], z["x_max"], 15, 42, allow_y_either_side=True)
    else:
        n, gc, p, lo, hi = conv_for(z["x_min"], z["x_max"], z["y_min"], z["y_max"])
    bar_rows.append({"zone_name": z["zone"],
                     "conversion_rate": round(p, 5),
                     "ci_low": round(lo, 5),
                     "ci_high": round(hi, 5),
                     "color_hex": z["color_hex"],
                     "attempts": n, "goals": gc})
bars = pd.DataFrame(bar_rows).sort_values("conversion_rate", ascending=False).reset_index(drop=True)
bars.to_csv(f"{OUT}/zone_conversion_bars.csv", index=False)
print(f"  zone_conversion_bars.csv: {len(bars)} rows")

# ---- File 3 — yband_dropoff.csv (Fenwick) ----
print("Rebuilding yband_dropoff.csv ...")
def y_band(absy):
    if absy < 5: return "0-5"
    if absy < 10: return "5-10"
    if absy < 15: return "10-15"
    if absy < 20: return "15-20"
    if absy < 25: return "20-25"
    if absy < 30: return "25-30"
    return "30+"
band_mid = {"0-5":2.5,"5-10":7.5,"10-15":12.5,"15-20":17.5,
            "20-25":22.5,"25-30":27.5,"30+":35.0}
band_order = ["0-5","5-10","10-15","15-20","20-25","25-30","30+"]
X_RANGES = {"CNFI":(74,89), "MNFI":(55,73), "FNFI":(25,54)}

yband_rows = []
for zone_name, (xlo, xhi) in X_RANGES.items():
    sub_mask = (xs >= xlo) & (xs <= xhi)
    sub_y = np.abs(ys[sub_mask])
    sub_g = goals[sub_mask]
    bands = np.array([y_band(v) for v in sub_y])
    for b in band_order:
        m = bands == b
        n = int(m.sum())
        if n == 0: continue
        gc = int(sub_g[m].sum())
        p, lo, hi = wilson(gc, n)
        yband_rows.append({
            "zone": zone_name,
            "y_band_label": b,
            "y_band_mid": band_mid[b],
            "attempts": n, "goals": gc,
            "conversion_rate": round(p, 5),
            "ci_low": round(lo, 5),
            "ci_high": round(hi, 5),
        })
yband = pd.DataFrame(yband_rows)
yband.to_csv(f"{OUT}/yband_dropoff.csv", index=False)
print(f"  yband_dropoff.csv: {len(yband)} rows")

# ===========================================================================
# Before / after reporting
# ===========================================================================
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.width = 220
pd.options.display.max_columns = None

# Load OLD (corrected-Corsi) versions from backup
old_bars  = pd.read_csv(f"{BK}/zone_conversion_bars.csv")
old_yband = pd.read_csv(f"{BK}/yband_dropoff.csv")
old_heat  = pd.read_csv(f"{BK}/heatmap_publication.csv")

# Confirmation: CNFI zone & CNFI 0-5
print("\n=== CONFIRMATION — CNFI zone bar ===")
ob = old_bars[old_bars["zone_name"]=="CNFI"].iloc[0]
nb = bars[bars["zone_name"]=="CNFI"].iloc[0]
print(f"  Corrected-Corsi (OLD): {ob['conversion_rate']*100:.4f}%  (n={int(ob['attempts']):,}, g={int(ob['goals']):,})")
print(f"  Fenwick (NEW):         {nb['conversion_rate']*100:.4f}%  (n={int(nb['attempts']):,}, g={int(nb['goals']):,})")

print("\n=== CONFIRMATION — CNFI y-band 0-5 ===")
oy = old_yband[(old_yband["zone"]=="CNFI") & (old_yband["y_band_label"]=="0-5")].iloc[0]
ny = yband[(yband["zone"]=="CNFI") & (yband["y_band_label"]=="0-5")].iloc[0]
print(f"  Corrected-Corsi (OLD): {oy['conversion_rate']*100:.4f}%  (n={int(oy['attempts']):,}, g={int(oy['goals']):,})")
print(f"  Fenwick (NEW):         {ny['conversion_rate']*100:.4f}%  (n={int(ny['attempts']):,}, g={int(ny['goals']):,})")

# Show full bars before/after side-by-side
print("\n=== zone_conversion_bars: BEFORE (Corsi) vs AFTER (Fenwick) ===")
cmp = old_bars.merge(bars, on="zone_name", suffixes=("_corsi","_fenwick"))
cmp["delta_pp"] = (cmp["conversion_rate_fenwick"] - cmp["conversion_rate_corsi"]) * 100
cmp = cmp.sort_values("conversion_rate_fenwick", ascending=False)
print(cmp[["zone_name","attempts_corsi","conversion_rate_corsi",
           "attempts_fenwick","conversion_rate_fenwick","delta_pp"]]
      .to_string(index=False))

print("\n=== yband_dropoff: BEFORE (Corsi) vs AFTER (Fenwick) ===")
cmp_y = old_yband[["zone","y_band_label","attempts","conversion_rate"]]\
        .rename(columns={"attempts":"n_corsi","conversion_rate":"rate_corsi"})\
        .merge(yband[["zone","y_band_label","attempts","conversion_rate"]]\
               .rename(columns={"attempts":"n_fenwick","conversion_rate":"rate_fenwick"}),
               on=["zone","y_band_label"])
cmp_y["delta_pp"] = (cmp_y["rate_fenwick"] - cmp_y["rate_corsi"]) * 100
print(cmp_y.to_string(index=False))

# Match check vs original P1a weights
P1A_WEIGHTS = {
    "CNFI":{"0-5":0.1579,"5-10":0.1075,"10-15":0.0781,"15-20":0.0513,
            "20-25":0.0318,"25-30":0.0209,"30+":0.0123},
    "MNFI":{"0-5":0.1109,"5-10":0.0985,"10-15":0.0887,"15-20":0.0692,
            "20-25":0.0492,"25-30":0.0278,"30+":0.0122},
    "FNFI":{"0-5":0.0367,"5-10":0.0335,"10-15":0.0289,"15-20":0.0235,
            "20-25":0.0199,"25-30":0.0141,"30+":0.0099},
}
print("\n=== Match check: yband Fenwick vs P1a weights (used in player metrics) ===")
match_rows = []
for _, r in yband.iterrows():
    expected = P1A_WEIGHTS.get(r["zone"], {}).get(r["y_band_label"])
    if expected is None: continue
    diff_pp = (r["conversion_rate"] - expected) * 100
    match_rows.append({"zone":r["zone"], "band":r["y_band_label"],
                        "fenwick_rate": round(r["conversion_rate"]*100,4),
                        "p1a_weight":   round(expected*100,4),
                        "diff_pp": round(diff_pp,4)})
mc = pd.DataFrame(match_rows)
print(mc.to_string(index=False))

# Spot-check: CNFI heatmap cells (a few central cells)
print("\n=== Heatmap spot-check: CNFI central cells (x_center 75-87, y_center -7..7) ===")
hc = heatmap[(heatmap["x_center"].between(75,87)) & (heatmap["y_center"].between(-7,7))]
print(hc[["x_center","y_center","attempts","goals","conversion_rate","save_pct","zone_label"]].to_string(index=False))

print("\nFiles overwritten:")
for f in ["heatmap_publication.csv","zone_conversion_bars.csv","yband_dropoff.csv"]:
    print(f"  {OUT}/{f}")
print(f"Backups: {BK}/")
