"""Regenerate kuemper_analysis.png with brand colors on the right panel only.

Left panel (Save % by Shot Type) is preserved exactly as-is.
Right panel (Weakness Trends by Season) is restyled with HockeyROI brand colors.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Right-panel palette: reuse left-side red/green + brand ROI Blue for 3rd bar
# ---------------------------------------------------------------------------
RED = "#ff4444"     # matches left side
GREEN = "#44ff44"   # matches left side
BLUE = "#2E7DC4"    # brand ROI Blue
TEXT = "white"
GRID = "#FFFFFF15"

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor("#1a1a2e")

# ===========================================================================
# LEFT PANEL — Save % by Shot Type (UNCHANGED from original notebook)
# ===========================================================================
shot_types = ["wrap-around", "deflected", "slap", "wrist", "snap", "tip-in", "backhand"]
save_pcts = [0.926, 0.921, 0.918, 0.911, 0.881, 0.866, 0.850]
his_avg = 0.898
colors_left = ["#44ff44" if v >= his_avg else "#ff4444" for v in save_pcts]

bars = ax1.barh(shot_types, save_pcts, color=colors_left)
ax1.set_xlim(0.80, 0.95)
ax1.axvline(x=0.898, color="white", linestyle="--", alpha=0.7, label="His average .898")
ax1.axvline(x=0.910, color="yellow", linestyle="--", alpha=0.7, label="League avg .910")

for bar, val in zip(bars, save_pcts):
    ax1.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", color="white", fontweight="bold")

ax1.set_facecolor("#1a1a2e")
ax1.tick_params(colors="white")
ax1.spines["bottom"].set_color("white")
ax1.spines["left"].set_color("white")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_title("Save % by Shot Type", color="white", fontsize=13, fontweight="bold", pad=15)
ax1.legend(loc="lower right", facecolor="#1a1a2e", labelcolor="white", fontsize=8)
ax1.invert_yaxis()  # match original ordering: wrap-around top, backhand bottom

# ===========================================================================
# RIGHT PANEL — Weakness Trends by Season (BRAND COLORS)
# ===========================================================================
seasons = ["2023-24", "2024-25", "2025-26"]
backhand_svpct = [0.846, 0.876, 0.823]
snap_svpct = [0.851, 0.908, 0.882]
tipin_svpct = [0.894, 0.884, 0.832]

x = np.arange(len(seasons))
width = 0.25

bars1 = ax2.bar(x - width, backhand_svpct, width, label="Backhand", color=RED,   alpha=0.85)
bars2 = ax2.bar(x,         snap_svpct,     width, label="Snap",     color=GREEN, alpha=0.85)
bars3 = ax2.bar(x + width, tipin_svpct,    width, label="Tip-in",   color=BLUE,  alpha=0.85)

ax2.set_ylim(0.78, 0.95)
ax2.axhline(y=0.898, color="white",  linestyle="--", alpha=0.7, label="His avg .898")
ax2.axhline(y=0.910, color="yellow", linestyle="--", alpha=0.7, label="League avg .910")
ax2.set_xticks(x)
ax2.set_xticklabels(seasons)
ax2.set_facecolor("#1a1a2e")
ax2.tick_params(colors="white")
ax2.spines["bottom"].set_color("white")
ax2.spines["left"].set_color("white")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_title("Weakness Trends by Season", color="white", fontsize=13, fontweight="bold", pad=15)
ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

for bar_group in (bars1, bars2, bars3):
    for bar in bar_group:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h + 0.002,
                 f"{h:.3f}", ha="center", va="bottom",
                 color="white", fontsize=7, fontweight="bold")

# ---------------------------------------------------------------------------
# Title + save
# ---------------------------------------------------------------------------
plt.suptitle("Darcy Kuemper Weakness Analysis\n132 Games | 3,466 Shots | 3 Seasons",
             color="white", fontsize=15, fontweight="bold", y=1.02)

plt.tight_layout()

out_path = Path(__file__).resolve().parents[1] / "Images" / "kuemper_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor="#1a1a2e", edgecolor="none")
print(f"Saved: {out_path}")
