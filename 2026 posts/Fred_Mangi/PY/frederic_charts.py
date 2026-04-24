#!/usr/bin/env python3
"""
HockeyROI — Trent Frederic teaser charts
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Charts/frederic_teaser"

BG       = "#0d1117"
BLUE     = "#4A9EFF"
ORANGE   = "#FF6B35"
RED      = "#FF4444"
GREEN    = "#44FF88"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"
GRID     = "#21262D"

# ── Season data (from frederic_profile.py output) ─────────────────────────────
seasons    = ["20-21", "21-22", "22-23", "23-24", "24-25", "25-26"]
nf_rates   = [0.1111,  0.1859,  0.2487,  0.2479,  0.2245,  0.3158]
avg_gaps   = [1.00,    1.38,    1.71,    1.48,    1.63,    1.31]
reb_shots  = [2,       16,      14,      29,      19,      16]
MIN_REB    = 5   # minimum rebound shots required for a valid timing score

# Invert & normalise gaps → timing score (lower gap = higher bar)
# Only include seasons that meet the minimum sample threshold in normalization
valid_gaps = [g for g, n in zip(avg_gaps, reb_shots) if n >= MIN_REB]
g_min, g_max = min(valid_gaps), max(valid_gaps)
timing_scores = [
    (g_max - g) / (g_max - g_min) if n >= MIN_REB else None
    for g, n in zip(avg_gaps, reb_shots)
]

career_nf_rate     = 0.2329
career_avg_gap     = 1.49
career_timing_score = (g_max - career_avg_gap) / (g_max - g_min)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — frederic_seasons.png
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

x      = np.arange(len(seasons))
width  = 0.35

bars_nf = ax.bar(x - width/2, nf_rates, width, color=BLUE, alpha=0.88, zorder=3, label="NF attempt rate")

# Timing score bars — skip None values; draw hatched grey stub for insufficient-sample seasons
valid_ts   = [ts if ts is not None else 0.0 for ts in timing_scores]
bar_colors = [ORANGE if ts is not None else "#3A3F47" for ts in timing_scores]
bar_alpha  = [0.88   if ts is not None else 0.55      for ts in timing_scores]
bars_timing = []
for i, (ts_val, bc, ba) in enumerate(zip(valid_ts, bar_colors, bar_alpha)):
    hatch = "//" if timing_scores[i] is None else None
    b = ax.bar(x[i] + width/2, ts_val if timing_scores[i] is not None else 0.12,
               width, color=bc, alpha=ba, hatch=hatch,
               edgecolor="#5A6070" if hatch else bc, linewidth=0.8,
               zorder=3)
    bars_timing.append(b)
    # Label insufficient-sample bars
    if timing_scores[i] is None:
        ax.text(x[i] + width/2, 0.13, "n<5", ha="center", va="bottom",
                fontsize=7.5, color="#8B949E", style="italic", zorder=5)

# Career avg dashed lines
ax.axhline(career_nf_rate,      color=BLUE,   linestyle="--", linewidth=1.3, alpha=0.65, zorder=2)
ax.axhline(career_timing_score, color=ORANGE, linestyle="--", linewidth=1.3, alpha=0.65, zorder=2)

# Label career avg lines on the right
ax.text(len(seasons) - 0.45, career_nf_rate + 0.013,
        f"career avg {career_nf_rate:.3f}", color=BLUE,   fontsize=8, alpha=0.85, ha="right")
ax.text(len(seasons) - 0.45, career_timing_score - 0.022,
        f"career avg {career_timing_score:.2f}", color=ORANGE, fontsize=8, alpha=0.85, ha="right")

# "Career best" annotations on 25-26 NF rate bar and timing score bar
last_idx      = len(seasons) - 1
nf_arrow_x    = last_idx - width/2
timing_arrow_x = last_idx + width/2

ax.annotate(
    "career best",
    xy=(nf_arrow_x, nf_rates[-1]),
    xytext=(nf_arrow_x - 0.75, nf_rates[-1] + 0.08),
    fontsize=9, color=TEXT, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.4),
    zorder=5,
)

ax.annotate(
    "career best",
    xy=(timing_arrow_x, timing_scores[-1]),
    xytext=(timing_arrow_x + 0.55, timing_scores[-1] + 0.08),
    fontsize=9, color=TEXT, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.4),
    zorder=5,
)

# Axes styling
ax.set_xticks(x)
ax.set_xticklabels(seasons, color=TEXT, fontsize=11)
ax.tick_params(colors=TEXT, length=0)
ax.yaxis.set_tick_params(labelcolor=SUBTEXT)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Rate / Score (0–1)", color=SUBTEXT, fontsize=10, labelpad=10)

# Legend
legend = ax.legend(
    handles=[
        mpatches.Patch(color=BLUE,   label="NF attempt rate"),
        mpatches.Patch(color=ORANGE, label="Timing score"),
    ],
    loc="upper left", framealpha=0, labelcolor=TEXT, fontsize=10,
)

# Title block
fig.text(0.5, 0.96, "Trent Frederic — Net-Front Profile by Season",
         ha="center", fontsize=15, fontweight="bold", color=TEXT)
fig.text(0.5, 0.91, "Attempt rate and timing score  |  Even strength 5v5",
         ha="center", fontsize=10, color=SUBTEXT)

# Watermark
fig.text(0.97, 0.02, "HockeyROI", ha="right", fontsize=9,
         color=SUBTEXT, alpha=0.7, style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
out1 = f"{OUT_DIR}/frederic_seasons.png"
plt.savefig(out1, dpi=160, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — frederic_split.png
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(9, 7))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)

split_labels = ["Games 1–60", "Games 61–81"]
goal_rates   = [0.00, 0.20]
colors       = [RED, GREEN]

bars2 = ax2.bar(split_labels, goal_rates, width=0.45, color=colors, alpha=0.88, zorder=3)

# Context annotations on each bar
annotations = [
    "32 net-front attempts\n0 goals",
    "10 net-front attempts\n2 goals",
]
for bar, ann, rate in zip(bars2, annotations, goal_rates):
    bx = bar.get_x() + bar.get_width() / 2
    by = bar.get_height()
    # Put annotation inside bar if tall enough, otherwise just above
    if rate > 0.04:
        ax2.text(bx, by / 2, ann, ha="center", va="center",
                 fontsize=11, color=BG, fontweight="bold", zorder=5,
                 linespacing=1.6)
    else:
        # 0% bar — annotate above with a small offset
        ax2.text(bx, 0.025, ann, ha="center", va="bottom",
                 fontsize=11, color=TEXT, fontweight="bold", zorder=5,
                 linespacing=1.6)

# Percentage labels at top of bars
for bar, rate in zip(bars2, goal_rates):
    bx = bar.get_x() + bar.get_width() / 2
    by = bar.get_height()
    label = f"{rate*100:.0f}%"
    ax2.text(bx, by + 0.007, label, ha="center", va="bottom",
             fontsize=18, color=TEXT, fontweight="bold", zorder=5)

# Axes styling
ax2.set_ylim(0, 0.30)
ax2.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
ax2.set_yticklabels(["0%", "5%", "10%", "15%", "20%", "25%"], color=SUBTEXT, fontsize=10)
ax2.tick_params(colors=TEXT, length=0)
ax2.set_xticklabels(split_labels, color=TEXT, fontsize=13, fontweight="bold")
for spine in ax2.spines.values():
    spine.set_visible(False)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax2.set_ylabel("NF goal rate", color=SUBTEXT, fontsize=11, labelpad=10)

# Title block
fig2.text(0.5, 0.96, "Frederic 2025-26: Before and After",
          ha="center", fontsize=15, fontweight="bold", color=TEXT)
fig2.text(0.5, 0.91, "Net-front goal rate  |  Even strength 5v5",
          ha="center", fontsize=10, color=SUBTEXT)

# Watermark
fig2.text(0.97, 0.02, "HockeyROI", ha="right", fontsize=9,
          color=SUBTEXT, alpha=0.7, style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
out2 = f"{OUT_DIR}/frederic_split.png"
plt.savefig(out2, dpi=160, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")
