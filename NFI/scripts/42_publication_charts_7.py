#!/usr/bin/env python3
"""
HockeyROI publication charts — 7 figures.
White background, brand colors, saved to NFI/output/charts/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
CHART_DIR = f"{OUT}/charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ---- Brand palette (white theme) ----
BG_DARK  = "#1B3A5C"   # dark text on white
SURFACE  = "#0B1D2E"
BLUE     = "#2E7DC4"
LIGHT_B  = "#4AB3E8"
ORANGE   = "#FF6B35"
TEXT     = "#1B3A5C"
GREEN    = "#44AA66"
YELLOW   = "#FFB700"
RED      = "#CC3333"
GREY     = "#888888"
GRID     = "#EEEEEE"
WHITE    = "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor":  WHITE,
    "axes.facecolor":    WHITE,
    "savefig.facecolor": WHITE,
    "axes.edgecolor":    TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "font.family":       "Arial",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
HEADLINE_FONT = {"family": "Impact", "size": 18}
results = {}

# ============================================================
# CHART 1 — Shot type CNFI conversion rates
# ============================================================
print("Building Chart 1: shot type CNFI ...")
df1 = pd.read_csv(f"{OUT}/shot_type_by_zone.csv")
cnfi = df1[df1["zone"]=="CNFI"].sort_values("conv_pct", ascending=True).reset_index(drop=True)
LEAGUE_AVG_CNFI = 13.04

def color_for(stype):
    s = stype.lower()
    if s == "snap":   return ORANGE
    if s == "wrist":  return BLUE
    if s == "tip-in": return RED
    return GREY

bar_colors = [color_for(t) for t in cnfi["shot_type"]]

fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
y_pos = np.arange(len(cnfi))
bars = ax.barh(y_pos, cnfi["conv_pct"].values, color=bar_colors,
               edgecolor=WHITE, linewidth=0.8, height=0.72)
# Wilson 95% CI error bars
errs_lo = (cnfi["conv_pct"] - cnfi["ci_lo_pct"]).values
errs_hi = (cnfi["ci_hi_pct"] - cnfi["conv_pct"]).values
ax.errorbar(cnfi["conv_pct"].values, y_pos, xerr=[errs_lo, errs_hi],
            fmt="none", ecolor=TEXT, capsize=4, capthick=1.2, elinewidth=1.2)

ax.set_yticks(y_pos)
ax.set_yticklabels([t.title() for t in cnfi["shot_type"]],
                    fontsize=11, color=TEXT)
ax.set_xlabel("Conversion rate (%)", fontsize=11, color=TEXT)

# League avg line
ax.axvline(LEAGUE_AVG_CNFI, color=GREY, linestyle="--", linewidth=1.5, alpha=0.85)
ax.text(LEAGUE_AVG_CNFI + 0.2, len(cnfi)-0.5,
        f"CNFI league avg {LEAGUE_AVG_CNFI:.2f}%",
        color=GREY, fontsize=9, va="bottom", fontweight="bold")

# Label bars
for i, (rate, err_h) in enumerate(zip(cnfi["conv_pct"].values, errs_hi)):
    ax.text(rate + err_h + 0.3, i, f"{rate:.2f}%",
            va="center", color=TEXT, fontsize=10, fontweight="bold")

ax.set_xlim(0, max(cnfi["ci_hi_pct"]) + 3)
ax.set_title("CNFI Zone Conversion Rate by Shot Type",
             color=TEXT, fontsize=10, alpha=0.85, fontweight="normal", pad=10)
fig.suptitle("Not All Net-Front Shots Are Equal",
             fontfamily="Impact", fontsize=18, fontweight="bold",
             color=TEXT, y=0.97)
ax.grid(True, axis="x", color=GRID, alpha=1.0, linestyle="-", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["bottom"].set_color(TEXT); ax.spines["left"].set_color(TEXT)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = f"{CHART_DIR}/chart1_shot_type_cnfi.png"
plt.savefig(out_path, dpi=300, facecolor=WHITE)
plt.close()
results["chart1_shot_type_cnfi.png"] = ("OK", os.path.getsize(out_path))

# ============================================================
# CHART 2 — Horse race R²
# ============================================================
print("Building Chart 2: horse race R² ...")
horse = pd.DataFrame({
    "metric": ["NFI%_ZA","xG% (MoneyPuck)","HD Fenwick%",
               "All-zone Fenwick%","All-zone Corsi%","PDO"],
    "R2":     [0.583, 0.538, 0.482, 0.429, 0.397, 0.336],
})
horse = horse.sort_values("R2", ascending=True).reset_index(drop=True)
CORSI_BASE = 0.397
bar_colors_h = [ORANGE if m=="NFI%_ZA" else BLUE for m in horse["metric"]]

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
y_pos = np.arange(len(horse))
ax.barh(y_pos, horse["R2"].values, color=bar_colors_h,
        edgecolor=WHITE, linewidth=0.8, height=0.72)
ax.set_yticks(y_pos); ax.set_yticklabels(horse["metric"].values, fontsize=11, color=TEXT)
ax.set_xlabel("R² vs standings points", fontsize=11, color=TEXT)
ax.axvline(CORSI_BASE, color=GREY, linestyle="--", linewidth=1.5, alpha=0.85)
ax.text(CORSI_BASE - 0.005, len(horse)-0.5, "Corsi baseline",
        color=GREY, fontsize=9, va="bottom", ha="right", fontweight="bold")
for i, r2 in enumerate(horse["R2"].values):
    ax.text(r2 + 0.006, i, f"{r2:.3f}", va="center", color=TEXT, fontsize=10, fontweight="bold")
ax.set_xlim(0, max(horse["R2"]) + 0.07)
ax.set_title("R² vs Standings Points  ·  126 Team-Seasons Pooled",
             color=TEXT, fontsize=10, alpha=0.85, fontweight="normal", pad=10)
fig.suptitle("NFI% Beats Every Conventional Metric",
             fontfamily="Impact", fontsize=18, fontweight="bold",
             color=TEXT, y=0.97)
ax.grid(True, axis="x", color=GRID, alpha=1.0, linestyle="-", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["bottom"].set_color(TEXT); ax.spines["left"].set_color(TEXT)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = f"{CHART_DIR}/chart2_horse_race_r2.png"
plt.savefig(out_path, dpi=300, facecolor=WHITE)
plt.close()
results["chart2_horse_race_r2.png"] = ("OK", os.path.getsize(out_path))

# ============================================================
# CHART 3 — Top 20 forward RelNFI%
# ============================================================
print("Building Chart 3: top 20 forward RelNFI% ...")
relnfi = pd.read_csv(f"{OUT}/fully_adjusted/top30_RelNFI_pooled.csv")
relnfi = relnfi.sort_values("metric_mean", ascending=False).head(20).reset_index(drop=True)
relnfi = relnfi.iloc[::-1].reset_index(drop=True)  # ascending order for horizontal display

def tier_color(rank):
    # rank 1-5 green, 6-15 blue, 16-20 grey
    if rank <= 5:  return GREEN
    if rank <= 15: return BLUE
    return GREY

# Compute original (descending) ranks
relnfi["orig_rank"] = (len(relnfi) - relnfi.index)  # 20 at top, 1 at bottom
bar_colors_3 = [tier_color(r) for r in relnfi["orig_rank"]]
labels = [f"{n} ({t})" for n, t in zip(relnfi["player_name"], relnfi["team_recent"])]

fig, ax = plt.subplots(figsize=(13, 9), dpi=300)
y_pos = np.arange(len(relnfi))
ax.barh(y_pos, relnfi["metric_mean"].values, color=bar_colors_3,
        edgecolor=WHITE, linewidth=0.8, height=0.74)
ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=10, color=TEXT)
ax.set_xlabel("RelNFI% (pooled)", fontsize=11, color=TEXT)
for i, v in enumerate(relnfi["metric_mean"].values):
    ax.text(v + 0.05, i, f"{v:+.3f}", va="center", color=TEXT, fontsize=9, fontweight="bold")
ax.set_xlim(0, max(relnfi["metric_mean"]) + 0.5)
ax.set_title("RelNFI% Pooled 2022-23 → 2025-26  ·  Min 2,000 ES Min",
             color=TEXT, fontsize=10, alpha=0.85, fontweight="normal", pad=10)
fig.suptitle("Two-Way Net Front Impact",
             fontfamily="Impact", fontsize=18, fontweight="bold",
             color=TEXT, y=0.97)
ax.grid(True, axis="x", color=GRID, alpha=1.0, linestyle="-", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["bottom"].set_color(TEXT); ax.spines["left"].set_color(TEXT)

# Tier legend
leg_handles = [mpatches.Patch(facecolor=GREEN, edgecolor=WHITE, label="Top 5"),
               mpatches.Patch(facecolor=BLUE,  edgecolor=WHITE, label="6-15"),
               mpatches.Patch(facecolor=GREY,  edgecolor=WHITE, label="16-20")]
ax.legend(handles=leg_handles, loc="lower right", facecolor=WHITE,
          edgecolor=TEXT, labelcolor=TEXT, fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = f"{CHART_DIR}/chart3_forward_twoway_relnfi.png"
plt.savefig(out_path, dpi=300, facecolor=WHITE)
plt.close()
results["chart3_forward_twoway_relnfi.png"] = ("OK", os.path.getsize(out_path))

# ============================================================
# CHART 4 — Top 10 defensive forwards by RelNFI_A%
# ============================================================
print("Building Chart 4: top 10 defensive forwards ...")
da = pd.read_csv(f"{OUT}/fully_adjusted/top30_RelNFI_A_pooled.csv")
# Forwards only
da = da[da["position"]=="F"].sort_values("metric_mean", ascending=False).head(10).reset_index(drop=True)
da_show = da.iloc[::-1].reset_index(drop=True)
labels4 = [f"{n} ({t})" for n, t in zip(da_show["player_name"], da_show["team_recent"])]

fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
y_pos = np.arange(len(da_show))
ax.barh(y_pos, da_show["metric_mean"].values, color=BLUE,
        edgecolor=WHITE, linewidth=0.8, height=0.72)
ax.set_yticks(y_pos); ax.set_yticklabels(labels4, fontsize=11, color=TEXT)
ax.set_xlabel("RelNFI_A% (pooled, dangerous-zone suppression impact)",
              fontsize=11, color=TEXT)
for i, v in enumerate(da_show["metric_mean"].values):
    ax.text(v + 0.05, i, f"{v:+.3f}", va="center", color=TEXT, fontsize=10, fontweight="bold")
ax.set_xlim(0, max(da_show["metric_mean"]) + 0.5)
ax.set_title("RelNFI_A%  ·  Dangerous Zone Suppression Impact (Pooled)",
             color=TEXT, fontsize=10, alpha=0.85, fontweight="normal", pad=10)
fig.suptitle("Elite Defensive Forwards",
             fontfamily="Impact", fontsize=18, fontweight="bold",
             color=TEXT, y=0.97)
ax.grid(True, axis="x", color=GRID, alpha=1.0, linestyle="-", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["bottom"].set_color(TEXT); ax.spines["left"].set_color(TEXT)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = f"{CHART_DIR}/chart4_defensive_forwards_relnfia.png"
plt.savefig(out_path, dpi=300, facecolor=WHITE)
plt.close()
results["chart4_defensive_forwards_relnfia.png"] = ("OK", os.path.getsize(out_path))

# ============================================================
# CHART 5 — Goalie tier comparison (top 20)
# ============================================================
print("Building Chart 5: goalie tier ...")
gm = pd.read_csv(f"{OUT}/goalie_metric_comparison.csv")
gm = gm.sort_values("NFI_GSAx_cumulative", ascending=False).head(20).copy()

def tier_label(toi):
    if pd.isna(toi): return "Insufficient"
    if toi >= 4000: return "Qualified"
    if toi >= 2000: return "Small Sample"
    return "Insufficient"

gm["tier_label"] = gm["toi_ES_min"].apply(tier_label)
tier_colors = {"Qualified": GREEN, "Small Sample": YELLOW, "Insufficient": GREY}
gm = gm.iloc[::-1].reset_index(drop=True)  # ascending for horizontal display
bar_colors_5 = [tier_colors[t] for t in gm["tier_label"]]

fig, ax = plt.subplots(figsize=(13, 9), dpi=300)
y_pos = np.arange(len(gm))
ax.barh(y_pos, gm["NFI_GSAx_cumulative"].values, color=bar_colors_5,
        edgecolor=WHITE, linewidth=0.8, height=0.74)
ax.set_yticks(y_pos); ax.set_yticklabels(gm["goalie_name"].values, fontsize=10, color=TEXT)
ax.set_xlabel("NFI-GSAx Cumulative (ES regulation, 5-season pool)",
              fontsize=11, color=TEXT)
ax.axvline(0, color=TEXT, linewidth=1.0, alpha=0.6)
for i, v in enumerate(gm["NFI_GSAx_cumulative"].values):
    pad = 1.5 if v >= 0 else -1.5; ha = "left" if v >= 0 else "right"
    ax.text(v + pad, i, f"{v:+.1f}", va="center", ha=ha,
            color=TEXT, fontsize=9, fontweight="bold")
xmax = max(gm["NFI_GSAx_cumulative"].max() + 12, 10)
xmin = min(gm["NFI_GSAx_cumulative"].min() - 5, -5)
ax.set_xlim(xmin, xmax)
ax.set_title("NFI-GSAx Cumulative ES Regulation 2022-23 → 2025-26",
             color=TEXT, fontsize=10, alpha=0.85, fontweight="normal", pad=10)
fig.suptitle("Goalie Net Front Impact — Goals Saved Above Expected",
             fontfamily="Impact", fontsize=18, fontweight="bold",
             color=TEXT, y=0.97)
ax.grid(True, axis="x", color=GRID, alpha=1.0, linestyle="-", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["bottom"].set_color(TEXT); ax.spines["left"].set_color(TEXT)
leg_handles = [mpatches.Patch(facecolor=GREEN,  edgecolor=WHITE, label="Qualified (≥4,000 min)"),
               mpatches.Patch(facecolor=YELLOW, edgecolor=WHITE, label="Small Sample (2,000–3,999)"),
               mpatches.Patch(facecolor=GREY,   edgecolor=WHITE, label="Insufficient (<2,000)")]
ax.legend(handles=leg_handles, loc="lower right", facecolor=WHITE,
          edgecolor=TEXT, labelcolor=TEXT, fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = f"{CHART_DIR}/chart5_goalie_tier.png"
plt.savefig(out_path, dpi=300, facecolor=WHITE)
plt.close()
results["chart5_goalie_tier.png"] = ("OK", os.path.getsize(out_path))

# ============================================================
# CHART 6 — Jarry profile heatmap (zone × shot_type)
# ============================================================
print("Building Chart 6: Jarry profile ...")
jp = pd.read_csv(f"{OUT}/jarry_profile.csv")
zs = jp[jp["section"]=="ZONE_SHOTTYPE"].copy()
# Build wide pivot: rows = zones, cols = shot_type, values = delta_sv_pct
piv = zs.pivot_table(index="zone", columns="shot_type",
                      values="delta_sv_pct", aggfunc="first")
# Fill missing with NaN; use diverging colormap
mat = piv.values
# Custom diverging cmap red→white→green
cmap = LinearSegmentedColormap.from_list("rwg", ["#CC3333","#FFFFFF","#44AA66"])
vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat))) + 0.5

fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels([s.title() for s in piv.columns], fontsize=11, color=TEXT)
ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=12, color=TEXT, fontweight="bold")
# Cell labels
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = mat[i, j]
        if pd.isna(v):
            ax.text(j, i, "—", ha="center", va="center", color=GREY, fontsize=12)
            continue
        # Find signif
        z, st = piv.index[i], piv.columns[j]
        sig = zs[(zs["zone"]==z) & (zs["shot_type"]==st)]["signif"]
        sig = bool(sig.iloc[0]) if len(sig) else False
        text_color = WHITE if abs(v) > vmax*0.45 else TEXT
        label = f"{v:+.2f}pp"
        if sig:
            label += "\n★"
        ax.text(j, i, label, ha="center", va="center",
                color=text_color, fontsize=11, fontweight="bold")

cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("Δ Save% vs league (pp)", color=TEXT, fontsize=10)
cbar.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT)
cbar.outline.set_edgecolor(TEXT)

ax.set_title("By Zone and Shot Type  ·  ES Regulation 5 Seasons  ·  ★ = significant",
             color=TEXT, fontsize=10, alpha=0.85, fontweight="normal", pad=10)
fig.suptitle("Tristan Jarry — NFI Save% vs League Average",
             fontfamily="Impact", fontsize=18, fontweight="bold",
             color=TEXT, y=0.97)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = f"{CHART_DIR}/chart6_jarry_profile.png"
plt.savefig(out_path, dpi=300, facecolor=WHITE)
plt.close()
results["chart6_jarry_profile.png"] = ("OK", os.path.getsize(out_path))

# ============================================================
# CHART 7 — D'Astous profile
# ============================================================
print("Building Chart 7: D'Astous profile ...")
dp = pd.read_csv(f"{OUT}/dastous_profile.csv")
sa = dp[dp["metric"]=="CNFI_SA_per60"].iloc[0]
DASTOUS_VAL = float(sa["value"])
LO = float(sa["lo95"]); HI = float(sa["hi95"])
LEAGUE_MED = 8.483

fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
labels7 = ["Charles-Edouard\nD'Astous", "League D\nmedian"]
vals = [DASTOUS_VAL, LEAGUE_MED]
y_pos = np.arange(len(labels7))
bar_colors_7 = [GREEN, GREY]
ax.barh(y_pos, vals, color=bar_colors_7, edgecolor=WHITE, linewidth=0.8, height=0.55)
# Error bar on D'Astous bar
ax.errorbar([DASTOUS_VAL], [0],
            xerr=[[DASTOUS_VAL - LO],[HI - DASTOUS_VAL]],
            fmt="none", ecolor=TEXT, capsize=8, capthick=1.6, elinewidth=1.6)
ax.set_yticks(y_pos); ax.set_yticklabels(labels7, fontsize=12, color=TEXT, fontweight="bold")
ax.set_xlabel("CNFI shots against per 60 (lower = better defense)",
              fontsize=11, color=TEXT)
ax.text(DASTOUS_VAL + 0.5, 0, f"{DASTOUS_VAL:.3f}\nCI [{LO:.2f}, {HI:.2f}]",
        va="center", color=TEXT, fontsize=10, fontweight="bold")
ax.text(LEAGUE_MED + 0.2, 1, f"{LEAGUE_MED:.3f}",
        va="center", color=TEXT, fontsize=10, fontweight="bold")

# Annotation
ax.annotate("CI entirely below league median\nstatistically confirmed elite",
            xy=(HI, 0), xytext=(HI + 1.5, -0.55),
            fontsize=10, color=TEXT, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.4))

ax.set_xlim(0, max(HI, LEAGUE_MED) + 3)
ax.set_title("CNFI Shots Against Per 60  ·  vs League Median  ·  ES Regulation",
             color=TEXT, fontsize=10, alpha=0.85, fontweight="normal", pad=10)
fig.suptitle("Charles-Edouard D'Astous — CNFI Defensive Suppression",
             fontfamily="Impact", fontsize=17, fontweight="bold",
             color=TEXT, y=0.97)
ax.grid(True, axis="x", color=GRID, alpha=1.0, linestyle="-", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["bottom"].set_color(TEXT); ax.spines["left"].set_color(TEXT)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = f"{CHART_DIR}/chart7_dastous_profile.png"
plt.savefig(out_path, dpi=300, facecolor=WHITE)
plt.close()
results["chart7_dastous_profile.png"] = ("OK", os.path.getsize(out_path))

print("\n=== Build summary ===")
for fn, (status, sz) in results.items():
    print(f"  {fn:<45} {status}  ({sz:,} bytes)")
