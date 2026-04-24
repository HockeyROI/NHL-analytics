#!/usr/bin/env python3
"""
Publication outputs — Part 1 (player rankings) + Part 2 (chart data).

Outputs to NFI/output/:
  publication_forwards_top100.csv
  publication_D_top100.csv
  publication_goalies_top60.csv
  heatmap_publication.csv
  zone_comparison.csv
  yband_dropoff.csv
  zone_conversion_bars.csv
"""
import math
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"

SEASONS = {"20212022","20222023","20232024","20242025","20252026"}

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0.0, c-h), min(1.0, c+h))

# =========================================================================
# PART 1 — Player ranking CSVs
# =========================================================================
print("=== PART 1 — Player ranking CSVs ===")

# Load source files
twf = pd.read_csv(f"{OUT}/twoway_forward_score.csv")
twd = pd.read_csv(f"{OUT}/twoway_D_score.csv")
p1a = pd.read_csv(f"{OUT}/P1a_centrality_weighted.csv")
p2  = pd.read_csv(f"{OUT}/P2_defensive_forwards.csv")
p1b = pd.read_csv(f"{OUT}/P1b_rebound_arrival.csv")
p4  = pd.read_csv(f"{OUT}/P4_defensive_D.csv")
p5  = pd.read_csv(f"{OUT}/P5_offensive_D.csv")
blk = pd.read_csv(f"{OUT}/player_block_rates.csv")
gm  = pd.read_csv(f"{OUT}/goalie_metric_comparison.csv")
pos = pd.read_csv(f"{OUT}/player_positions.csv", dtype={"player_id":int})
pos["pos_detail"] = pos["position"].map({"C":"C","L":"LW","R":"RW","D":"D","G":"G"})

# ---- Forwards top 100 ----
fwd = twf.head(100).copy()
# Rename for publication clarity
fwd = fwd.rename(columns={"twoway_rank":"rank"})
# Position
fwd = fwd.merge(pos[["player_id","pos_detail"]], on="player_id", how="left")
# P1b columns (P1b_pct + P1b CIs and P1b_per60)
p1b_keep = ["player_id","P1b_pct","P1b_lo_pct","P1b_hi_pct",
            "P1b_per60","P1b_per60_lo95","P1b_per60_hi95"]
fwd = fwd.merge(p1b[p1b_keep], on="player_id", how="left")
# P2_weighted already in twf via earlier join. Rename for clarity.
# P1a CIs not available (weighted aggregate), so leave None.
fwd["small_sample_flag"] = np.where(fwd["es_toi_min"] < 2000.0, "<2000_ES_min", "")

fwd_cols = ["rank","player_name","pos_detail","es_toi_min",
            "P1a_weighted_total","P1b_pct","P1b_lo_pct","P1b_hi_pct",
            "P1b_per60","P1b_per60_lo95","P1b_per60_hi95",
            "P2_weighted","z_P1a_weighted","z_P2_weighted","twoway_score",
            "off_rank","def_rank","small_sample_flag"]
fwd = fwd[fwd_cols].rename(columns={"pos_detail":"position",
                                      "P1a_weighted_total":"P1a_weighted",
                                      "es_toi_min":"ES_TOI_min"})
fwd.to_csv(f"{OUT}/publication_forwards_top100.csv", index=False)
fwd_flagged = (fwd["small_sample_flag"]!="").sum()
print(f"  publication_forwards_top100.csv: 100 rows, {fwd_flagged} small-sample flagged")

# ---- D top 100 ----
d = twd.head(100).copy()
d = d.rename(columns={"twoway_D_rank":"rank"})
# blocks_xG_prevented_per60
d = d.merge(blk[["player_id","blocks_xG_prevented_per60"]], on="player_id", how="left")
d["blocks_xG_prevented_per60"] = d["blocks_xG_prevented_per60"].fillna(0.0)
d["small_sample_flag"] = np.where(d["es_toi_min"] < 2000.0, "<2000_ES_min", "")
d_cols = ["rank","player_name","es_toi_min",
          "TNFI_SA_per60","P4_weighted","TNFI_SF_per60","P5_weighted",
          "blocks_xG_prevented_per60",
          "z_P5_weighted","z_P4_weighted","twoway_D_score",
          "off_rank","def_rank","small_sample_flag"]
d = d[d_cols].rename(columns={"es_toi_min":"ES_TOI_min"})
d.to_csv(f"{OUT}/publication_D_top100.csv", index=False)
d_flagged = (d["small_sample_flag"]!="").sum()
print(f"  publication_D_top100.csv: 100 rows, {d_flagged} small-sample flagged")

# ---- Goalies top 60 ----
g = gm.copy()
# Derive games from ES TOI ÷ ~35 ES min/game
g["games_est"] = (g["toi_ES_min"] / 35.0).round(0).astype(int)
g = g[g["games_est"] >= 20].copy()
# Sort by NFI_GSAx_cumulative
g = g.sort_values("NFI_GSAx_cumulative", ascending=False).reset_index(drop=True)
g["rank"] = np.arange(1, len(g)+1)
# Keep top 60 overall but also report tier breakdown
g_top60 = g.head(60).copy()

g_cols = ["rank","goalie_name","tier_label","games_est","toi_ES_min",
          "total_faced","spatial_save_pct","sv_lo","sv_hi",
          "NFI_GSAx_cumulative","NFI_GSAx_per60",
          "rank_cumulative","rank_per60","tier"]
g_top60 = g_top60[g_cols].rename(columns={"toi_ES_min":"ES_TOI_min",
                                            "tier_label":"tier_label_text"})
g_top60.to_csv(f"{OUT}/publication_goalies_top60.csv", index=False)
print(f"  publication_goalies_top60.csv: {len(g_top60)} rows (≥20 GP est)")
print(f"    by tier: {g_top60['tier_label_text'].value_counts().to_dict()}")
print(f"    total goalies w/ ≥20 GP est: {len(g)} (60 selected)")

# =========================================================================
# PART 2 — Chart data files
# =========================================================================
print("\n=== PART 2 — Chart data files ===")

# ----- Load shots once for grid + zone aggregation -----
print("Loading shots ...")
shot_cols = ["season","period","situation_code","event_type",
             "x_coord_norm","y_coord_norm","is_goal"]
shots = pd.read_csv(SHOT_CSV, usecols=shot_cols,
                    dtype={"season":str,"situation_code":str})
shots = shots[shots["season"].isin(SEASONS)].copy()
shots = shots[shots["period"].between(1,3)].copy()
shots = shots[shots["situation_code"].astype(str)=="1551"].copy()
shots = shots[shots["event_type"].isin(
    ["shot-on-goal","missed-shot","blocked-shot","goal"])].copy()

# Apply blocked-shot coord fix
blk_mask = shots["event_type"]=="blocked-shot"
shots.loc[blk_mask,"x_coord_norm"] = shots.loc[blk_mask,"x_coord_norm"].abs()
shots.loc[blk_mask,"y_coord_norm"] = shots.loc[blk_mask,"y_coord_norm"].abs()
shots["abs_y"] = shots["y_coord_norm"].abs()
# Dropping events with NaN coords just in case
shots = shots.dropna(subset=["x_coord_norm","y_coord_norm"])
print(f"  ES regulation shots in scope: {len(shots):,}")

def zone_label(x, y):
    absy = abs(y)
    if 74 <= x <= 89 and absy <= 9:   return "CNFI"
    if 55 <= x <= 73 and absy <= 15:  return "MNFI"
    if 25 <= x <= 54 and absy <= 15:  return "FNFI"
    return "Wide"

# ---- File 1 — heatmap_publication.csv ----
# 5x5 ft grid: x 25-89 (13 bins), y -40..40 (16 bins)
print("Building heatmap_publication.csv ...")
xs = shots["x_coord_norm"].values
ys = shots["y_coord_norm"].values
goals = shots["is_goal"].values
ets = shots["event_type"].values

x_edges = np.arange(25, 95, 5)   # 25,30,...,90 (14 edges -> 13 cells)
y_edges = np.arange(-40, 45, 5)  # -40,-35,...,40 (17 edges -> 16 cells)

heatmap_rows = []
for ix in range(len(x_edges)-1):
    for iy in range(len(y_edges)-1):
        xlo, xhi = x_edges[ix], x_edges[ix+1]
        ylo, yhi = y_edges[iy], y_edges[iy+1]
        mask = (xs >= xlo) & (xs < xhi) & (ys >= ylo) & (ys < yhi)
        n = int(mask.sum())
        if n < 50: continue
        cell_goals = int(goals[mask].sum())
        # Conversion rate (Corsi denominator: all attempts in cell)
        conv = cell_goals / n
        # Save% needs SOG-faced count: SOG + goal events
        cell_ets = ets[mask]
        faced = int(((cell_ets=="shot-on-goal") | (cell_ets=="goal")).sum())
        save_pct = (1.0 - cell_goals/faced) if faced > 0 else np.nan
        center_x = (xlo + xhi) / 2.0
        center_y = (ylo + yhi) / 2.0
        zlabel = zone_label(center_x, center_y)
        heatmap_rows.append({
            "x_center": center_x, "y_center": center_y,
            "x_min": xlo, "x_max": xhi, "y_min": ylo, "y_max": yhi,
            "attempts": n, "goals": cell_goals,
            "conversion_rate": round(conv, 5),
            "shots_faced": faced,
            "save_pct": round(save_pct, 5) if not np.isnan(save_pct) else np.nan,
            "zone_label": zlabel,
        })
heatmap = pd.DataFrame(heatmap_rows)
heatmap.to_csv(f"{OUT}/heatmap_publication.csv", index=False)
total_cells = (len(x_edges)-1)*(len(y_edges)-1)
qual_cells  = len(heatmap)
print(f"  heatmap_publication.csv: {qual_cells} cells (of {total_cells} possible) meeting ≥50 attempts")
print(f"    cell zone counts: {heatmap['zone_label'].value_counts().to_dict()}")

# ---- File 2 — zone_comparison.csv ----
print("Building zone_comparison.csv ...")
ZONE_DEFS = [
    {"zone":"CNFI",            "x_min":74, "x_max":89, "y_min":-9,  "y_max":9,  "color_hex":"#FF6B35"},
    {"zone":"MNFI",            "x_min":55, "x_max":73, "y_min":-15, "y_max":15, "color_hex":"#2E7DC4"},
    {"zone":"FNFI",            "x_min":25, "x_max":54, "y_min":-15, "y_max":15, "color_hex":"#4AB3E8"},
    {"zone":"Wide",            "x_min":25, "x_max":89, "y_min":15,  "y_max":42, "color_hex":"#888888"},  # |y|>15 in OZ
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

zone_rows = []
for z in ZONE_DEFS:
    if z["zone"] == "Wide":
        # |y|>15 in OZ
        n, gcount, p, lo, hi = conv_for(z["x_min"], z["x_max"], 15, 42, allow_y_either_side=True)
    else:
        n, gcount, p, lo, hi = conv_for(z["x_min"], z["x_max"], z["y_min"], z["y_max"])
    z2 = dict(z)
    z2.update({"attempts":n, "goals":gcount,
               "conversion_rate":round(p,5),
               "ci_low":round(lo,5),"ci_high":round(hi,5)})
    zone_rows.append(z2)
zc = pd.DataFrame(zone_rows)
zc.to_csv(f"{OUT}/zone_comparison.csv", index=False)
print(f"  zone_comparison.csv: {len(zc)} zones with boundaries + colors + conv rates")

# ---- File 3 — yband_dropoff.csv ----
print("Building yband_dropoff.csv ...")
def y_band(absy):
    if absy < 5:  return ("0-5", 2.5)
    if absy < 10: return ("5-10", 7.5)
    if absy < 15: return ("10-15", 12.5)
    if absy < 20: return ("15-20", 17.5)
    if absy < 25: return ("20-25", 22.5)
    if absy < 30: return ("25-30", 27.5)
    return ("30+", 35.0)

# x ranges: CNFI 74-89, MNFI 55-73, FNFI 25-54 (no y restriction so we get all bands)
X_RANGES = {"CNFI":(74,89), "MNFI":(55,73), "FNFI":(25,54)}
band_order = ["0-5","5-10","10-15","15-20","20-25","25-30","30+"]

yband_rows = []
for zone_name, (xlo, xhi) in X_RANGES.items():
    sub_mask = (xs >= xlo) & (xs <= xhi)
    sub_y = np.abs(ys[sub_mask])
    sub_g = goals[sub_mask]
    bands = np.array([y_band(v)[0] for v in sub_y])
    band_mid_lookup = {b: y_band({"0-5":2.5,"5-10":7.5,"10-15":12.5,
                                    "15-20":17.5,"20-25":22.5,"25-30":27.5,
                                    "30+":35.0}[b])[1] for b in band_order}
    band_mid_lookup = {"0-5":2.5,"5-10":7.5,"10-15":12.5,"15-20":17.5,
                        "20-25":22.5,"25-30":27.5,"30+":35.0}
    for b in band_order:
        m = bands == b
        n = int(m.sum())
        if n == 0: continue
        gcnt = int(sub_g[m].sum())
        p, lo, hi = wilson(gcnt, n)
        yband_rows.append({
            "zone": zone_name,
            "y_band_label": b,
            "y_band_mid": band_mid_lookup[b],
            "attempts": n,
            "goals": gcnt,
            "conversion_rate": round(p, 5),
            "ci_low": round(lo, 5),
            "ci_high": round(hi, 5),
        })
yband = pd.DataFrame(yband_rows)
yband.to_csv(f"{OUT}/yband_dropoff.csv", index=False)
print(f"  yband_dropoff.csv: {len(yband)} rows ({yband['zone'].nunique()} zones × bands)")

# ---- File 4 — zone_conversion_bars.csv ----
print("Building zone_conversion_bars.csv ...")
# Reuse the zone_comparison values but pivot to bar-chart-friendly schema
bar_rows = []
for r in zone_rows:
    bar_rows.append({
        "zone_name": r["zone"],
        "conversion_rate": r["conversion_rate"],
        "ci_low": r["ci_low"],
        "ci_high": r["ci_high"],
        "color_hex": r["color_hex"],
        "attempts": r["attempts"],
        "goals": r["goals"],
    })
bars = pd.DataFrame(bar_rows).sort_values("conversion_rate", ascending=False).reset_index(drop=True)
bars.to_csv(f"{OUT}/zone_conversion_bars.csv", index=False)
print(f"  zone_conversion_bars.csv: {len(bars)} rows, sorted by conversion rate desc")

# =========================================================================
# Reporting
# =========================================================================
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.width = 220
pd.options.display.max_columns = None

print("\n=== File preview: publication_forwards_top100 (top 10) ===")
print(fwd.head(10)[["rank","player_name","position","ES_TOI_min",
                     "P1a_weighted","P1b_pct","P1b_per60","P2_weighted",
                     "twoway_score","small_sample_flag"]].to_string(index=False))

print("\n=== File preview: publication_D_top100 (top 10) ===")
print(d.head(10)[["rank","player_name","ES_TOI_min","P4_weighted","P5_weighted",
                   "blocks_xG_prevented_per60","twoway_D_score",
                   "small_sample_flag"]].to_string(index=False))

print("\n=== File preview: publication_goalies_top60 (top 10 + tier counts) ===")
print(g_top60.head(10)[["rank","goalie_name","tier_label_text","games_est",
                         "ES_TOI_min","total_faced","spatial_save_pct",
                         "NFI_GSAx_cumulative","NFI_GSAx_per60"]].to_string(index=False))

print("\n=== File preview: zone_conversion_bars (sorted) ===")
print(bars.to_string(index=False))

print("\n=== File preview: yband_dropoff ===")
print(yband.to_string(index=False))

print("\n=== File preview: heatmap_publication (first 10 + zone counts) ===")
print(heatmap.head(10).to_string(index=False))

print("\n--- Files written ---")
for f in ["publication_forwards_top100.csv","publication_D_top100.csv",
          "publication_goalies_top60.csv","heatmap_publication.csv",
          "zone_comparison.csv","yband_dropoff.csv","zone_conversion_bars.csv"]:
    print(f"  {OUT}/{f}")
