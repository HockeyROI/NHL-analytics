#!/usr/bin/env python3
"""
NFI-Score Step 6 (DEFENSEMEN) — Off/Def decomposition chart for D.

Parallel to 06_chart_off_def_decomposition.py. Reads D league avg by state.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd

ROOT = Path("/Users/ashgarg/Documents/HockeyROI")
SRC = ROOT / "NFI" / "NFI_Score_Adj" / "2026" / "05" / "Output" / "league_avg_by_state_D.csv"

CHART_DIR = Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" / \
            "NHL Analysis" / "2026 posts" / "Charts"
CHART_PATH = CHART_DIR / "off_def_decomposition_by_state_D.png"

LOGO_CANDIDATES = [
    Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" /
    "NHL Analysis" / "Brand" / "Logos" / "hockeyroi_banner.png",
    Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" /
    "NHL Analysis" / "Brand" / "hockeyroi_banner.png",
]

STATES = ["Down2", "Down1", "Tied", "Up1", "Up2"]
LABELS = ["Down 2+", "Down 1", "Tied", "Up 1", "Up 2+"]
COLOR_OFF = "#2E7DC4"
COLOR_DEF = "#FF6B35"
COLOR_GREY = "#888888"


def stop(msg: str, code: int = 2) -> int:
    print(f"[step6D] STOP — {msg}", file=sys.stderr)
    return code


def main() -> int:
    if not SRC.exists():
        return stop(f"input not found: {SRC} (run Step 5D first)")

    df = pd.read_csv(SRC).set_index("state").reindex(STATES)
    off = df["league_avg_Off"].to_numpy()
    dfn = df["league_avg_Def"].to_numpy()
    net = df["league_avg_Net"].to_numpy()
    print(f"[step6D] read {SRC}")
    for s, o, d, n in zip(STATES, off, dfn, net):
        print(f"  {s:<6}  Off {o:>5.2f}  Def {d:>5.2f}  Net {n:>+5.2f}")

    logo = next((p for p in LOGO_CANDIDATES if p.exists()), None)

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = list(range(len(STATES)))

    ax.plot(x, off, color=COLOR_OFF, linewidth=2.6, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=1.5, label="Off (FOR / 60)", zorder=3)
    ax.plot(x, dfn, color=COLOR_DEF, linewidth=2.6, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=1.5, label="Def (AGAINST / 60)", zorder=3)

    for i in x:
        y_lo, y_hi = (off[i], dfn[i]) if off[i] > dfn[i] else (dfn[i], off[i])
        color = "#D9EDDA" if off[i] > dfn[i] else "#F4D5D5"
        ax.fill_betweenx([y_lo, y_hi], i - 0.18, i + 0.18,
                         color=color, alpha=0.65, zorder=1, edgecolor="none")

    for i, (o, d) in enumerate(zip(off, dfn)):
        ax.text(i, o + 0.10, f"{o:.2f}", ha="center", va="bottom",
                fontsize=9, color=COLOR_OFF, fontfamily="Arial", fontweight="bold")
        ax.text(i, d - 0.10, f"{d:.2f}", ha="center", va="top",
                fontsize=9, color=COLOR_DEF, fontfamily="Arial", fontweight="bold")

    y_min = min(off.min(), dfn.min())
    y_max = max(off.max(), dfn.max())
    pad = (y_max - y_min) * 0.35
    ax.set_ylim(y_min - pad, y_max + pad * 0.6)

    for i, n in enumerate(net):
        ax.text(i, y_min - pad * 0.55, f"Net  {n:+.2f}",
                ha="center", va="center", fontsize=10.5,
                color="#222222", fontfamily="Arial", fontweight="bold",
                bbox=dict(facecolor="#F4F4F4", edgecolor=COLOR_GREY,
                          linewidth=0.8, boxstyle="round,pad=0.35"))

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11, fontfamily="Arial", color="#333333")
    ax.set_ylabel("Net-front events per 60 (CNFI + MNFI)",
                  fontsize=11, fontfamily="Arial", color="#333333", labelpad=10)
    ax.set_xlabel("Score state (player's team perspective)",
                  fontsize=11, fontfamily="Arial", color="#333333", labelpad=22)

    ax.set_title("Defensemen: net-front activity by score state",
                 loc="left", fontsize=22, fontfamily="Impact",
                 color="#1A1A1A", pad=20)
    ax.text(0, 1.02,
            "League TOI-weighted averages across 198 defensemen · 5v5, regulation, "
            "regular season, 6 seasons (2020-21 → 2025-26)",
            transform=ax.transAxes, fontsize=9, fontfamily="Arial",
            color=COLOR_GREY)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_GREY)
    ax.spines["bottom"].set_color(COLOR_GREY)
    ax.tick_params(colors="#333333", labelsize=10)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.8, zorder=0)

    leg = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color("#333333")
        t.set_fontfamily("Arial")

    if logo is not None:
        try:
            img = plt.imread(str(logo))
            ab = AnnotationBbox(OffsetImage(img, zoom=0.10),
                                (1.0, -0.22), xycoords="axes fraction",
                                box_alignment=(1.0, 0.0), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f"[step6D] WARNING: could not embed logo ({e})")

    plt.tight_layout()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=200, facecolor="#FFFFFF",
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[step6D] wrote {CHART_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
