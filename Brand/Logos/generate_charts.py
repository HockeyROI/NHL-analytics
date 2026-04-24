import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Brand colors ──────────────────────────────────────────────────────────────
BG_COLOR      = "#0D1117"
PANEL_COLOR   = "#161B22"
TEXT_COLOR    = "#E6EDF3"
SUBTLE_COLOR  = "#8B949E"
GREEN         = "#44AA66"
RED           = "#CC3333"
BRAND_ACCENT  = "#4AB3E8"

def apply_brand(fig, ax):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)

def add_logo_text(fig):
    fig.text(0.98, 0.01, "HockeyROI", fontsize=9, color=BRAND_ACCENT,
             ha="right", va="bottom", fontweight="bold", alpha=0.7,
             fontfamily="monospace")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 1 — Shot Type Conversion
# ══════════════════════════════════════════════════════════════════════════════
LEAGUE_AVG = 13.57   # %

shot_data = [
    # label,        attempts, conv_pct, small_sample
    ("Snap",        11,       18.2,     False),
    ("Backhand",    16,       12.5,     False),
    ("Wrist",       60,        6.7,     False),
    ("Tip-in",       5,        0.0,     True),
    ("Deflected",    3,        0.0,     True),
]

# Sort descending by conversion rate for readability
shot_data_sorted = sorted(shot_data, key=lambda x: x[2])

labels   = [d[0] for d in shot_data_sorted]
attempts = [d[1] for d in shot_data_sorted]
convs    = [d[2] for d in shot_data_sorted]
small    = [d[3] for d in shot_data_sorted]

# Color logic: small sample → grey, above avg → green, below → red
colors = []
for i, d in enumerate(shot_data_sorted):
    if d[3]:                      # small sample
        colors.append("#555E6B")
    elif d[2] >= LEAGUE_AVG:
        colors.append(GREEN)
    else:
        colors.append(RED)

fig1, ax1 = plt.subplots(figsize=(10, 5.5))
apply_brand(fig1, ax1)

y_pos = np.arange(len(labels))
bars = ax1.barh(y_pos, convs, height=0.55, color=colors,
                edgecolor="#30363D", linewidth=0.6, zorder=3)

# League avg dashed line
ax1.axvline(LEAGUE_AVG, color="#DDDD55", linestyle="--", linewidth=1.6,
            alpha=0.85, zorder=4, label=f"League avg {LEAGUE_AVG}%")

# Bar labels
for i, (bar, att, cv, sm) in enumerate(zip(bars, attempts, convs, small)):
    label_x = cv + 0.4
    flag = "  ⚠ small sample" if sm else ""
    ax1.text(label_x, bar.get_y() + bar.get_height() / 2,
             f"{att} att  |  {cv:.1f}%{flag}",
             va="center", ha="left", fontsize=9.5, color=TEXT_COLOR, zorder=5)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=11, color=TEXT_COLOR)
ax1.set_xlim(0, 32)
ax1.set_xlabel("Conversion Rate (%)", fontsize=10, color=SUBTLE_COLOR, labelpad=8)
ax1.grid(axis="x", color="#30363D", linewidth=0.5, zorder=0)
ax1.set_axisbelow(True)

# Legend
legend = ax1.legend(fontsize=9, facecolor=PANEL_COLOR, edgecolor="#30363D",
                    labelcolor=TEXT_COLOR, loc="lower right")

# Patch legend for colors
green_patch = mpatches.Patch(color=GREEN,     label="Above league avg")
red_patch   = mpatches.Patch(color=RED,       label="Below league avg")
grey_patch  = mpatches.Patch(color="#555E6B", label="Small sample (<6 att)")
ax1.legend(handles=[green_patch, red_patch, grey_patch],
           fontsize=8.5, facecolor=PANEL_COLOR, edgecolor="#30363D",
           labelcolor=TEXT_COLOR, loc="lower right")

# Second legend for dashed line — overlay manually
from matplotlib.lines import Line2D
avg_line = Line2D([0], [0], color="#DDDD55", linestyle="--", linewidth=1.6,
                  label=f"League avg ({LEAGUE_AVG}%)")
handles2, _ = ax1.get_legend_handles_labels()
ax1.legend(handles=[green_patch, red_patch, grey_patch, avg_line],
           fontsize=8.5, facecolor=PANEL_COLOR, edgecolor="#30363D",
           labelcolor=TEXT_COLOR, loc="lower right")

# Titles
fig1.text(0.5, 0.97, "McDavid Net-Front Conversion by Shot Type",
          ha="center", va="top", fontsize=14, fontweight="bold", color=TEXT_COLOR)
fig1.text(0.5, 0.91, "Even strength 5v5  |  6 seasons  |  League avg = 13.57%",
          ha="center", va="top", fontsize=10, color=SUBTLE_COLOR)

add_logo_text(fig1)
plt.tight_layout(rect=[0, 0.03, 1, 0.90])
fig1.savefig("mcdavid_shot_type_chart.png", dpi=160, bbox_inches="tight",
             facecolor=BG_COLOR)
plt.close(fig1)
print("Chart 1 saved → mcdavid_shot_type_chart.png")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Wrist Rebound Conversion by Season
# ══════════════════════════════════════════════════════════════════════════════
seasons   = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
mcd_vals  = [40.0,       5.0,       0.0,       10.0,       0.0,       0.0]
lg_vals   = [18.7,      17.0,      14.1,       11.2,      11.7,      10.0]

MC_COLOR  = "#FF6B35"
LG_COLOR  = "#4AB3E8"

x = np.arange(len(seasons))

fig2, ax2 = plt.subplots(figsize=(10, 5.5))
apply_brand(fig2, ax2)

# Shaded area: red when McDavid below league avg
for i in range(len(seasons) - 1):
    xs = [x[i], x[i+1]]
    mc = [mcd_vals[i], mcd_vals[i+1]]
    lg = [lg_vals[i],  lg_vals[i+1]]
    if mcd_vals[i] < lg_vals[i] or mcd_vals[i+1] < lg_vals[i+1]:
        ax2.fill_between(xs, mc, lg, where=[m < l for m, l in zip(mc, lg)],
                         interpolate=True, color=RED, alpha=0.18, zorder=2)

# Lines
ax2.plot(x, lg_vals,  color=LG_COLOR, linestyle="--", linewidth=2.2,
         marker="o", markersize=6, zorder=4, label="League average")
ax2.plot(x, mcd_vals, color=MC_COLOR, linestyle="-",  linewidth=2.5,
         marker="o", markersize=7, zorder=5, label="McDavid")

# Data labels
for i, (mv, lv) in enumerate(zip(mcd_vals, lg_vals)):
    ax2.text(x[i], mv + 1.0, f"{mv:.1f}%", ha="center", va="bottom",
             fontsize=8.5, color=MC_COLOR, fontweight="bold", zorder=6)
    ax2.text(x[i], lv + 1.0, f"{lv:.1f}%", ha="center", va="bottom",
             fontsize=8.5, color=LG_COLOR, zorder=6)

ax2.set_xticks(x)
ax2.set_xticklabels(seasons, fontsize=10, color=TEXT_COLOR)
ax2.set_ylabel("Conversion Rate (%)", fontsize=10, color=SUBTLE_COLOR, labelpad=8)
ax2.set_ylim(-3, 52)
ax2.grid(axis="y", color="#30363D", linewidth=0.5, zorder=0)
ax2.set_axisbelow(True)

# Legend
ax2.legend(fontsize=9.5, facecolor=PANEL_COLOR, edgecolor="#30363D",
           labelcolor=TEXT_COLOR, loc="upper right")

# Red area legend patch
red_area = mpatches.Patch(color=RED, alpha=0.35, label="McDavid below league avg")
handles, _ = ax2.get_legend_handles_labels()
ax2.legend(handles=handles + [red_area], fontsize=9, facecolor=PANEL_COLOR,
           edgecolor="#30363D", labelcolor=TEXT_COLOR, loc="upper right")

# Titles
fig2.text(0.5, 0.97, "McDavid Wrist Rebound Conversion vs League Average",
          ha="center", va="top", fontsize=14, fontweight="bold", color=TEXT_COLOR)
fig2.text(0.5, 0.91, "Season by season  |  Even strength 5v5",
          ha="center", va="top", fontsize=10, color=SUBTLE_COLOR)

add_logo_text(fig2)
plt.tight_layout(rect=[0, 0.03, 1, 0.90])
fig2.savefig("mcdavid_wrist_trend_chart.png", dpi=160, bbox_inches="tight",
             facecolor=BG_COLOR)
plt.close(fig2)
print("Chart 2 saved → mcdavid_wrist_trend_chart.png")
