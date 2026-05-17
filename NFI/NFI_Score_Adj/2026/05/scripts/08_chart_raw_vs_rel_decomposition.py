#!/usr/bin/env python3
"""
NFI-Score Step 7b — Raw NFI Net vs RelNFI Net by score state.

Reads:  Output/league_avg_by_state.csv         (Step 5)
        Output/relnfi_per_state_league_avg.csv (Step 7a)
Writes: ~/Library/CloudStorage/OneDrive-Personal/NHL Analysis/2026 posts/Charts/raw_vs_rel_score_state_comparison.png

Two lines on a single panel: raw NFI Net (blue) and RelNFI Net (orange) across
the 5 states. If both rise in parallel, the score effect is real at the player
level. If raw rises but Rel is flat or inverted, the score effect was mostly
team-quality driven and gets stripped out by the WOWY relative.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
import numpy as np

ROOT = Path("/Users/ashgarg/Documents/HockeyROI")
RAW_CSV = ROOT / "NFI" / "NFI_Score_Adj" / "2026" / "05" / "Output" / "league_avg_by_state.csv"
REL_CSV = ROOT / "NFI" / "NFI_Score_Adj" / "2026" / "05" / "Output" / "relnfi_per_state_league_avg.csv"

CHART_DIR = Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" / \
            "NHL Analysis" / "2026 posts" / "Charts"
CHART_PATH = CHART_DIR / "raw_vs_rel_score_state_comparison.png"

LOGO_CANDIDATES = [
    Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" /
    "NHL Analysis" / "Brand" / "Logos" / "hockeyroi_banner.png",
    Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" /
    "NHL Analysis" / "Brand" / "hockeyroi_banner.png",
]

STATES = ["Down2", "Down1", "Tied", "Up1", "Up2"]
LABELS = ["Down 2+", "Down 1", "Tied", "Up 1", "Up 2+"]
COLOR_RAW = "#2E7DC4"   # blue
COLOR_REL = "#FF6B35"   # orange
COLOR_GREY = "#888888"


def stop(msg: str, code: int = 2) -> int:
    print(f"[step7b] STOP — {msg}", file=sys.stderr)
    return code


def main() -> int:
    if not RAW_CSV.exists():
        return stop(f"input not found: {RAW_CSV}")
    if not REL_CSV.exists():
        return stop(f"input not found: {REL_CSV}")

    raw = pd.read_csv(RAW_CSV).set_index("state").reindex(STATES)
    rel = pd.read_csv(REL_CSV).set_index("state").reindex(STATES)

    raw_net = raw["league_avg_Net"].to_numpy()
    rel_net = rel["RelNFI_Net"].to_numpy()
    print(f"[step7b] raw NFI Net by state:    {raw_net}")
    print(f"[step7b] RelNFI Net by state:     {rel_net}")
    raw_slope = float(np.diff(raw_net).mean())
    rel_slope = float(np.diff(rel_net).mean())
    print(f"[step7b] slopes — raw: {raw_slope:+.3f}/step   rel: {rel_slope:+.3f}/step")

    logo = next((p for p in LOGO_CANDIDATES if p.exists()), None)

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = list(range(len(STATES)))
    ax.plot(x, raw_net, color=COLOR_RAW, linewidth=2.8, marker="o", markersize=9,
            markeredgecolor="white", markeredgewidth=1.5,
            label="Raw NFI Net  (slope " + f"{raw_slope:+.3f}" + "/step)", zorder=3)
    ax.plot(x, rel_net, color=COLOR_REL, linewidth=2.8, marker="s", markersize=8.5,
            markeredgecolor="white", markeredgewidth=1.5,
            label="RelNFI Net  (slope " + f"{rel_slope:+.3f}" + "/step)", zorder=3)

    # value labels above each marker
    for i, v in enumerate(raw_net):
        ax.text(i, v + 0.06, f"{v:+.2f}", ha="center", va="bottom",
                fontsize=10, color=COLOR_RAW, fontfamily="Arial", fontweight="bold")
    for i, v in enumerate(rel_net):
        ax.text(i, v - 0.07, f"{v:+.2f}", ha="center", va="top",
                fontsize=10, color=COLOR_REL, fontfamily="Arial", fontweight="bold")

    ax.axhline(0, color="#222222", linewidth=0.7, linestyle=":")

    y_min = min(raw_net.min(), rel_net.min())
    y_max = max(raw_net.max(), rel_net.max())
    pad = (y_max - y_min) * 0.30
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11, fontfamily="Arial", color="#333333")
    ax.set_xlabel("Score state (player's team perspective)",
                  fontsize=11, fontfamily="Arial", color="#333333", labelpad=10)
    ax.set_ylabel("Net per 60 (Off − Def)",
                  fontsize=11, fontfamily="Arial", color="#333333", labelpad=10)

    ax.set_title("How much of the score effect is team-driven?",
                 loc="left", fontsize=22, fontfamily="Impact",
                 color="#1A1A1A", pad=20)
    ax.text(0, 1.02,
            "Raw NFI Net rises with a lead; RelNFI Net (off-ice WOWY) does not.  "
            "League TOI-weighted, 379 forwards · 5v5, regulation, regular season, "
            "6 seasons (2020-21 → 2025-26)",
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
                                (1.0, -0.18), xycoords="axes fraction",
                                box_alignment=(1.0, 0.0), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f"[step7b] WARNING: could not embed logo ({e})")

    plt.tight_layout()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=200, facecolor="#FFFFFF",
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[step7b] wrote {CHART_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
