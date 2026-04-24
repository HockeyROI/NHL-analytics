#!/usr/bin/env python3
"""
Publication-ready charts (HockeyROI brand).

Outputs to NFI/output/charts/:
  zone_comparison.png
  heatmap_conversion.png
  zone_conversion_bars.png
  yband_dropoff.png
  forward_twoway.png
  goalie_gsax.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
CHART_DIR = f"{OUT}/charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ---- Brand palette (light theme — white bg, dark text) ----
BG       = "#FFFFFF"
SURFACE  = "#F4F6F9"   # light card surface for legends
GRID     = "#EEEEEE"   # light gridlines
BLUE     = "#2E7DC4"
LIGHT_B  = "#4AB3E8"
ORANGE   = "#FF6B35"
TEXT     = "#1B3A5C"   # dark text on white
GREEN    = "#44AA66"
YELLOW   = "#FFB700"
RED      = "#CC3333"
GREY     = "#888888"
ICE      = "#E8F0F8"

# Global rcParams
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "savefig.facecolor": BG,
    "axes.edgecolor":    TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "font.family":       "Arial",
    "axes.titleweight":  "bold",
    "axes.titlecolor":   TEXT,
    "axes.titlepad":     14,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
HEADLINE_FONT = {"family": "Impact", "size": 18}

# =========================================================================
# CHART 1 — Ice rink zone diagram
# =========================================================================
print("Building Chart 1 — ice rink zone diagram ...")
fig, ax = plt.subplots(figsize=(14, 7.5), dpi=300)
ax.set_facecolor(BG)

# Ice surface (offensive zone area: x=25 (blue line) to x=100 (end boards))
# Half-rink width ±42.5
# Rounded corners: NHL boards have 28-ft radius corners
def draw_oz_rink(ax):
    # Ice fill (rounded rect)
    ice = patches.FancyBboxPatch(
        (25, -42.5), 75, 85, boxstyle="round,pad=0,rounding_size=0",
        linewidth=0, facecolor=ICE, zorder=0)
    # Use a polygon approximating the right side rounded corners
    # Approximation: rectangle with rounded right corners (radius 11ft)
    R = 11
    verts = []
    # Start at top-left, go right along top, round corner, down, round corner, left
    verts.append((25, 42.5))
    verts.append((100-R, 42.5))
    # Top-right rounded corner (centered at (100-R, 42.5-R))
    theta = np.linspace(np.pi/2, 0, 30)
    cx, cy = 100-R, 42.5-R
    for t in theta:
        verts.append((cx + R*np.cos(t), cy + R*np.sin(t)))
    verts.append((100, -42.5+R))
    # Bottom-right rounded corner
    theta = np.linspace(0, -np.pi/2, 30)
    cx, cy = 100-R, -42.5+R
    for t in theta:
        verts.append((cx + R*np.cos(t), cy + R*np.sin(t)))
    verts.append((25, -42.5))
    rink_poly = patches.Polygon(verts, closed=True, facecolor=ICE,
                                 edgecolor=TEXT, linewidth=1.5, zorder=0)
    ax.add_patch(rink_poly)

    # Blue line at x=25
    ax.plot([25, 25], [-42.5, 42.5], color="#2E7DC4", linewidth=4, zorder=1)
    # Goal line at x=89
    ax.plot([89, 89], [-37, 37], color=RED, linewidth=1.5, zorder=1)
    # Goal posts at x=89, y=-3..3
    goal = patches.Rectangle((89, -3), 4, 6, linewidth=2,
                              edgecolor=RED, facecolor="none", zorder=2)
    ax.add_patch(goal)
    # Goal crease (semicircle radius 6 from goal mouth)
    crease = patches.Wedge((89, 0), 6, 90, 270, facecolor="#A8D5F2",
                            edgecolor=RED, linewidth=1.2, zorder=1)
    # Wedge above wedges towards left side; we want crease in front of net (toward x<89)
    crease = patches.Wedge((89, 0), 6, 90, 270, facecolor="#A8D5F2",
                            edgecolor=RED, linewidth=1.2, zorder=1)
    ax.add_patch(crease)
    # Faceoff circles at (69, ±22), radius 15
    for fy in [-22, 22]:
        circ = patches.Circle((69, fy), 15, facecolor="none",
                               edgecolor=RED, linewidth=1.3, zorder=1)
        ax.add_patch(circ)
        # Faceoff dot
        dot = patches.Circle((69, fy), 0.8, facecolor=RED, zorder=2)
        ax.add_patch(dot)
    # Center red dot is at x=0, not shown here.

draw_oz_rink(ax)

# Zone overlays (filled rectangles, semi-transparent)
def add_zone(ax, xmin, xmax, ymin, ymax, color, label, alpha=0.55, dashed=False):
    if dashed:
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                  linewidth=2.5, edgecolor=color,
                                  facecolor="none", linestyle="--", zorder=3)
    else:
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                  linewidth=1.5, edgecolor=color,
                                  facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(rect)

add_zone(ax, 74, 89, -9, 9,   ORANGE, "CNFI 13.04%")
add_zone(ax, 55, 73, -15, 15, BLUE,   "MNFI 10.03%")
add_zone(ax, 25, 54, -15, 15, LIGHT_B,"FNFI 3.36%")
add_zone(ax, 69, 89, -22, 22, RED,    "HD Conventional 10.91%", dashed=True)
add_zone(ax, 69, 89, -14, 14, YELLOW, "Inner Slot 12.02%",      dashed=True)

# Labels
ax.text((74+89)/2, 0, "CNFI\n13.04%", ha="center", va="center",
        fontsize=11, fontweight="bold", color="white", zorder=4)
ax.text((55+73)/2, 0, "MNFI\n10.03%", ha="center", va="center",
        fontsize=11, fontweight="bold", color="white", zorder=4)
ax.text((25+54)/2, 0, "FNFI\n3.36%", ha="center", va="center",
        fontsize=11, fontweight="bold", color="white", zorder=4)
# Legend for the dashed conventional outlines (placed in upper area)
ax.text(94, 36, "HD Conventional\n10.91%", ha="left", va="top",
        fontsize=9, fontweight="bold", color=RED, zorder=4)
ax.text(94, 26, "Inner Slot\n12.02%", ha="left", va="top",
        fontsize=9, fontweight="bold", color=YELLOW, zorder=4)
# Annotate blue line and goal
ax.text(24.5, -41, "Blue Line", color=BLUE, fontsize=9, ha="left", va="bottom", fontweight="bold")
ax.text(89, -41, "Goal Line", color=RED, fontsize=9, ha="center", va="bottom", fontweight="bold")

# Title
ax.set_title("Net Front Impact Zones vs Conventional Definitions",
             fontdict=HEADLINE_FONT, color=TEXT, pad=18)
sub = "ES Regulation, 5 Seasons Pooled (2021-22 → 2025-26) · Fenwick"
ax.text(62.5, 49.5, sub, ha="center", va="center", fontsize=10,
        color=TEXT, alpha=0.85)

ax.set_xlim(20, 110)
ax.set_ylim(-50, 52)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/zone_comparison.png", dpi=300, facecolor=BG)
plt.close()
print(f"  saved zone_comparison.png")

# =========================================================================
# CHART 2 — Heatmap
# =========================================================================
print("Building Chart 2 — heatmap ...")
hm = pd.read_csv(f"{OUT}/heatmap_publication.csv")

# Build a 2D grid for imshow
x_centers = sorted(hm["x_center"].unique())
y_centers = sorted(hm["y_center"].unique())
xc_idx = {x:i for i,x in enumerate(x_centers)}
yc_idx = {y:i for i,y in enumerate(y_centers)}
grid = np.full((len(y_centers), len(x_centers)), np.nan)
for _, r in hm.iterrows():
    grid[yc_idx[r["y_center"]], xc_idx[r["x_center"]]] = r["conversion_rate"] * 100

# Custom dark-blue → orange colormap
cmap = LinearSegmentedColormap.from_list("nfi", [
    (0.0, "#0B1D2E"),
    (0.25, BLUE),
    (0.55, LIGHT_B),
    (0.80, YELLOW),
    (1.0, ORANGE),
])
vmax = 18.0

fig, ax = plt.subplots(figsize=(14, 7.5), dpi=300)
ax.set_facecolor(BG)
extent = [min(x_centers)-2.5, max(x_centers)+2.5,
          min(y_centers)-2.5, max(y_centers)+2.5]
im = ax.imshow(grid, origin="lower", extent=extent, aspect="equal",
               cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")

# Zone outlines
def outline(xmin, xmax, ymin, ymax, color, label, lw=2.2, ls="-"):
    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                              linewidth=lw, edgecolor=color, facecolor="none",
                              linestyle=ls, zorder=5)
    ax.add_patch(rect)

outline(74, 89, -9, 9,   ORANGE, "CNFI", lw=2.5)
outline(55, 73, -15, 15, BLUE,   "MNFI", lw=2.0)
outline(25, 54, -15, 15, LIGHT_B,"FNFI", lw=1.8)

ax.text(82, 0, "CNFI", color=TEXT, fontsize=11, fontweight="bold",
        ha="center", va="center", zorder=6,
        bbox=dict(facecolor=BG, alpha=0.5, edgecolor=ORANGE, lw=1.5))
ax.text(64, 0, "MNFI", color=TEXT, fontsize=11, fontweight="bold",
        ha="center", va="center", zorder=6,
        bbox=dict(facecolor=BG, alpha=0.5, edgecolor=BLUE, lw=1.5))
ax.text(40, 0, "FNFI", color=TEXT, fontsize=11, fontweight="bold",
        ha="center", va="center", zorder=6,
        bbox=dict(facecolor=BG, alpha=0.5, edgecolor=LIGHT_B, lw=1.5))

# Net mark
ax.plot([89, 89], [-3, 3], color=RED, linewidth=4, zorder=5)

ax.set_title("Shot Conversion Rate by Location  ·  ES Regulation, 5 Seasons",
             fontdict=HEADLINE_FONT, color=TEXT, pad=15)
ax.set_xlabel("X coordinate (ft from center, attacking →)",
              fontsize=10, color=TEXT)
ax.set_ylabel("Y coordinate (ft from center)", fontsize=10, color=TEXT)
ax.set_xticks([25, 40, 55, 70, 89])
ax.set_yticks([-30, -15, 0, 15, 30])

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("Conversion rate (%)", color=TEXT, fontsize=10)
cbar.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT)
cbar.outline.set_edgecolor(TEXT)

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/heatmap_conversion.png", dpi=300, facecolor=BG)
plt.close()
print(f"  saved heatmap_conversion.png")

# =========================================================================
# CHART 3 — Zone conversion bar chart
# =========================================================================
print("Building Chart 3 — zone bars ...")
zb = pd.read_csv(f"{OUT}/zone_conversion_bars.csv")
zb = zb.sort_values("conversion_rate")     # ascending so largest is at top

LEAGUE_AVG = 0.0440  # 4.40% from zone_conversion_rates.csv ALL ES REG

fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
ax.set_facecolor(BG)
y_pos = np.arange(len(zb))
rates_pct = zb["conversion_rate"].values * 100
err_lo = (zb["conversion_rate"] - zb["ci_low"]).values * 100
err_hi = (zb["ci_high"] - zb["conversion_rate"]).values * 100
colors = zb["color_hex"].values
bars = ax.barh(y_pos, rates_pct, color=colors, edgecolor="white",
               linewidth=0.8, height=0.7)
ax.errorbar(rates_pct, y_pos,
            xerr=[err_lo, err_hi], fmt="none",
            ecolor=TEXT, capsize=4, capthick=1.4, elinewidth=1.4)
ax.set_yticks(y_pos)
ax.set_yticklabels(zb["zone_name"].values, fontsize=11, color=TEXT)
ax.set_xlabel("Conversion rate (%)", fontsize=11, color=TEXT)

# League average line
ax.axvline(LEAGUE_AVG*100, color=GREY, linestyle="--", linewidth=1.5, alpha=0.85)
ax.text(LEAGUE_AVG*100 + 0.15, len(zb)-0.5,
        f"League avg {LEAGUE_AVG*100:.2f}%",
        color=GREY, fontsize=9, va="bottom", fontweight="bold")

# Bar labels
for i, (rate, err_h) in enumerate(zip(rates_pct, err_hi)):
    ax.text(rate + err_h + 0.25, i, f"{rate:.2f}%",
            va="center", color=TEXT, fontsize=10, fontweight="bold")

ax.set_xlim(0, max(rates_pct) + 2.5)
sub = f"ES Regulation, Fenwick, 5 Seasons Pooled · n=467,128 attempts league-wide"
ax.set_title(sub, color=TEXT, fontsize=10, alpha=0.85,
             fontweight="normal", pad=10)
fig.suptitle("Goal Conversion Rate by Zone — NFI vs Conventional Definitions",
             fontfamily="Impact", fontsize=18, fontweight="bold",
             color=TEXT, y=0.97)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(TEXT)
ax.spines["left"].set_color(TEXT)
ax.tick_params(colors=TEXT)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{CHART_DIR}/zone_conversion_bars.png", dpi=300, facecolor=BG)
plt.close()
print(f"  saved zone_conversion_bars.png")

# =========================================================================
# CHART 4 — Y-band dropoff
# =========================================================================
print("Building Chart 4 — y-band dropoff ...")
yb = pd.read_csv(f"{OUT}/yband_dropoff.csv")
fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
ax.set_facecolor(BG)
zone_color = {"CNFI": ORANGE, "MNFI": BLUE, "FNFI": LIGHT_B}

for zone in ["CNFI","MNFI","FNFI"]:
    sub = yb[yb["zone"]==zone].sort_values("y_band_mid")
    color = zone_color[zone]
    ax.plot(sub["y_band_mid"], sub["conversion_rate"]*100,
            "-o", color=color, linewidth=2.8, markersize=9,
            label=zone, zorder=4)
    ax.fill_between(sub["y_band_mid"],
                     sub["ci_low"]*100, sub["ci_high"]*100,
                     color=color, alpha=0.18, zorder=2)

# Annotate the cliff
ax.annotate("CNFI cliff:\n5pp drop in first 5 ft\noff the midline",
            xy=(2.5, 15.32), xytext=(11, 17),
            color=TEXT, fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.4))

ax.set_xticks([2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 35])
ax.set_xticklabels(["0-5","5-10","10-15","15-20","20-25","25-30","30+"],
                    color=TEXT, fontsize=10)
ax.set_xlabel("Lateral distance from center (ft)", fontsize=11, color=TEXT)
ax.set_ylabel("Conversion rate (%)", fontsize=11, color=TEXT)
ax.set_ylim(0, 19)
ax.grid(True, color=GRID, alpha=1.0, linestyle="-", linewidth=0.8)
ax.set_axisbelow(True)

leg = ax.legend(loc="upper right", facecolor=SURFACE, edgecolor=TEXT,
                labelcolor=TEXT, fontsize=11, framealpha=0.85)
for t in leg.get_texts(): t.set_color(TEXT)

sub = "Each line is one x-band (CNFI 74-89, MNFI 55-73, FNFI 25-54). Shaded = Wilson 95% CI"
ax.set_title(sub, color=TEXT, fontsize=10, alpha=0.85,
             fontweight="normal", pad=10)
fig.suptitle("Conversion Rate by Lateral Distance from Center  ·  The Center Lane Effect",
             fontfamily="Impact", fontsize=18, fontweight="bold",
             color=TEXT, y=0.97)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{CHART_DIR}/yband_dropoff.png", dpi=300, facecolor=BG)
plt.close()
print(f"  saved yband_dropoff.png")

# =========================================================================
# CHART 5 — Forward two-way dot plot
# =========================================================================
print("Building Chart 5 — forward two-way ...")
f = pd.read_csv(f"{OUT}/publication_forwards_top100.csv")
top30 = f.head(30).copy()
# x = -z_P2 (so right = better defense); y = z_P1a (up = better offense)
top30["def_axis"] = -top30["z_P2_weighted"]
top30["off_axis"] =  top30["z_P1a_weighted"]
pos_color = {"C": BLUE, "LW": GREEN, "RW": ORANGE}
top30["color"] = top30["position"].map(pos_color).fillna(GREY)
top30["marker_size"] = (top30["ES_TOI_min"]/40.0).clip(lower=60, upper=520)

fig, ax = plt.subplots(figsize=(14, 8.5), dpi=300)
ax.set_facecolor(BG)
# Quadrant lines
ax.axhline(0, color=GREY, linewidth=1, alpha=0.6)
ax.axvline(0, color=GREY, linewidth=1, alpha=0.6)

# Scatter
ax.scatter(top30["def_axis"], top30["off_axis"],
           s=top30["marker_size"], c=top30["color"],
           alpha=0.78, edgecolors="white", linewidths=1.2, zorder=4)

# Player labels (alternate offsets to reduce overlap)
for i, r in top30.iterrows():
    offset_x = 0.03; offset_y = 0.06
    ax.annotate(r["player_name"],
                xy=(r["def_axis"], r["off_axis"]),
                xytext=(offset_x, offset_y), textcoords="offset points",
                fontsize=8, color=TEXT, alpha=0.95, fontweight="bold",
                ha="left", va="bottom")

# Quadrant labels
def quad_label(ax, x, y, text, color):
    ax.text(x, y, text, color=color, fontsize=11, fontweight="bold",
            ha="center", va="center", alpha=0.85,
            bbox=dict(facecolor=SURFACE, alpha=0.65, edgecolor=color, lw=1))

xlo, xhi = top30["def_axis"].min()-0.5, top30["def_axis"].max()+0.7
# Extend ylim upward to give the legend a dedicated band above quadrant labels
ylo, yhi = top30["off_axis"].min()-0.5, top30["off_axis"].max()+1.4
# Quadrant labels positioned a bit lower to leave room for the upper-right legend
quad_label(ax, xhi-0.6, yhi-1.1, "ELITE TWO-WAY", GREEN)
quad_label(ax, xlo+0.6, yhi-1.1, "OFFENSIVE SPECIALIST", LIGHT_B)
quad_label(ax, xhi-0.6, ylo+0.4, "DEFENSIVE SPECIALIST", BLUE)
quad_label(ax, xlo+0.6, ylo+0.4, "BELOW AVERAGE", RED)

# Position legend — upper-right corner of the plot, above the ELITE TWO-WAY label
from matplotlib.patches import Patch
leg_handles = [Patch(facecolor=BLUE,  edgecolor="white", label="C"),
               Patch(facecolor=GREEN, edgecolor="white", label="LW"),
               Patch(facecolor=ORANGE,edgecolor="white", label="RW")]
leg = ax.legend(handles=leg_handles, loc="upper right",
                facecolor=SURFACE, edgecolor=TEXT, fontsize=10,
                framealpha=0.95, labelcolor=TEXT, title="Position",
                title_fontsize=10)
leg.get_title().set_color(TEXT)

ax.set_xlim(xlo, xhi)
ax.set_ylim(ylo, yhi)
ax.set_xlabel("Defensive value (z-score, higher = better)",
              fontsize=11, color=TEXT)
ax.set_ylabel("Offensive value (P1a z-score)", fontsize=11, color=TEXT)
sub = "Bubble size scales with ES TOI · ES regulation 5-season pool · Top 30 by two-way score"
ax.set_title(sub, color=TEXT, fontsize=10, alpha=0.85,
             fontweight="normal", pad=10)
fig.suptitle("Forward Two-Way NFI Score — Offensive vs Defensive (Top 30)",
             fontfamily="Impact", fontsize=18, fontweight="bold",
             color=TEXT, y=0.97)

ax.grid(True, color=GRID, alpha=1.0, linestyle="-", linewidth=0.8)
ax.set_axisbelow(True)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{CHART_DIR}/forward_twoway.png", dpi=300, facecolor=BG)
plt.close()
print(f"  saved forward_twoway.png")

# =========================================================================
# CHART 6 — Goalie GSAx bars
# =========================================================================
print("Building Chart 6 — goalie GSAx ...")
gs = pd.read_csv(f"{OUT}/publication_goalies_top60.csv")
gs_q = gs[gs["tier_label_text"]=="Qualified"].head(20).copy()
gs_q = gs_q.sort_values("NFI_GSAx_cumulative")  # asc so largest at top

fig, ax = plt.subplots(figsize=(13, 9.5), dpi=300)
ax.set_facecolor(BG)
y_pos = np.arange(len(gs_q))
gsax_vals = gs_q["NFI_GSAx_cumulative"].values
bar_colors = [GREEN if v>=0 else RED for v in gsax_vals]
bars = ax.barh(y_pos, gsax_vals, color=bar_colors, edgecolor="white",
               linewidth=0.8, height=0.72)
ax.axvline(0, color=GREY, linewidth=1.2, alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(gs_q["goalie_name"].values, fontsize=11, color=TEXT)

# Bar value labels
for i, v in enumerate(gsax_vals):
    pad = 1.5 if v>=0 else -1.5
    ha = "left" if v>=0 else "right"
    ax.text(v + pad, i, f"{v:+.1f}", va="center", ha=ha,
            color=TEXT, fontsize=10, fontweight="bold")

xmax = max(gsax_vals.max() + 8, 5)
xmin = min(gsax_vals.min() - 4, -5)
ax.set_xlim(xmin, xmax)
ax.set_xlabel("NFI Goals Saved Above Expected (cumulative, ES regulation)",
              fontsize=11, color=TEXT)
sub = "Per-faced rates by zone (CNFI/MNFI/FNFI) · Calibrated against league shots-faced denominator"
ax.set_title(sub, color=TEXT, fontsize=10, alpha=0.85,
             fontweight="normal", pad=10)
fig.suptitle("Goalie NFI-GSAx — 5 Season Cumulative (Tier 1 Qualified, Top 20)",
             fontfamily="Impact", fontsize=18, fontweight="bold",
             color=TEXT, y=0.97)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(TEXT)
ax.spines["left"].set_color(TEXT)
ax.tick_params(colors=TEXT)
ax.grid(True, axis="x", color=GRID, alpha=1.0, linestyle="-", linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{CHART_DIR}/goalie_gsax.png", dpi=300, facecolor=BG)
plt.close()
print(f"  saved goalie_gsax.png")

print("\nAll 6 charts built.")
