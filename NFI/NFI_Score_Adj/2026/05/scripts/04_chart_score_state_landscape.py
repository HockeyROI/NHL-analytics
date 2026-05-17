#!/usr/bin/env python3
"""
NFI-Score Step 4 — Diagnostic chart: league-average NFI-Score-{State}-Net by state.

Reads: NFI/NFI_Score_Adj/2026/05/Output/nfi_score_player_rates.csv
Writes: ~/Library/CloudStorage/OneDrive-Personal/NHL Analysis/2026 posts/Charts/score_state_landscape.png

Logo path: spec lists ~/.../Brand/hockeyroi_banner.png. The file actually lives
at ~/.../Brand/Logos/hockeyroi_banner.png; this script searches the spec path
first, then falls back to /Logos/. If neither is found, the chart is generated
without the logo and a warning is printed.

Style: white background, blue #2E7DC4, light blue #4AB3E8, orange #FF6B35
sparingly, grey #888888. Headline in Impact (Bebas Neue not installed); body
text in Arial.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

ROOT = Path("/Users/ashgarg/Documents/HockeyROI")
OUT_DIR = ROOT / "NFI" / "NFI_Score_Adj" / "2026" / "05" / "Output"
WIDE = OUT_DIR / "nfi_score_player_rates.csv"
TOI = OUT_DIR / "per_state_toi.csv"

CHART_DIR = Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" / \
            "NHL Analysis" / "2026 posts" / "Charts"
CHART_PATH = CHART_DIR / "score_state_landscape.png"

LOGO_CANDIDATES = [
    Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" /
    "NHL Analysis" / "Brand" / "hockeyroi_banner.png",          # spec path
    Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" /
    "NHL Analysis" / "Brand" / "Logos" / "hockeyroi_banner.png",  # actual
]

STATE_ORDER = ["Down2", "Down1", "Tied", "Up1", "Up2"]
STATE_LABELS = ["Down 2+", "Down 1", "Tied", "Up 1", "Up 2+"]
COLOR_BLUE = "#2E7DC4"
COLOR_LIGHT_BLUE = "#4AB3E8"
COLOR_ORANGE = "#FF6B35"
COLOR_GREY = "#888888"


def stop(msg: str, code: int = 2) -> int:
    print(f"[step4] STOP — {msg}", file=sys.stderr)
    return code


def main() -> int:
    if not WIDE.exists():
        return stop(f"input not found: {WIDE}")
    if not TOI.exists():
        return stop(f"input not found: {TOI}")

    print(f"[step4] reading {WIDE}")
    df = pd.read_csv(WIDE)
    toi = pd.read_csv(TOI).set_index("player_id")

    # league-average NFI-Score-{State}-Net = TOI-weighted average across the 379
    # forwards, weighted by per-state TOI (each shot was credited to ~5 players,
    # so TOI weighting matches the league per-state pooled rate)
    means = {}
    for s in STATE_ORDER:
        col = f"NFI-Score-{s}-Net"
        w = df["player_id"].map(toi[f"TOI_{s}"]).fillna(0)
        v = df[col].fillna(0)
        means[s] = float((v * w).sum() / w.sum()) if w.sum() > 0 else 0.0
    print("[step4] league-average NFI-Score-{State}-Net (TOI-weighted across 379 F):")
    for s in STATE_ORDER:
        print(f"   {s:<6}  {means[s]:>+8.3f}")

    # logo
    logo_path = next((p for p in LOGO_CANDIDATES if p.exists()), None)
    if logo_path is None:
        print(f"[step4] WARNING: logo not found at any of: "
              f"{[str(p) for p in LOGO_CANDIDATES]} — chart will be drawn without it")
    else:
        print(f"[step4] using logo at: {logo_path}")

    # plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    vals = [means[s] for s in STATE_ORDER]
    # color: trail (Down) light blue, tied grey, lead (Up) blue; highlight max in orange
    colors = []
    max_idx = int(np.argmax([abs(v) for v in vals]))
    for i, s in enumerate(STATE_ORDER):
        if i == max_idx:
            colors.append(COLOR_ORANGE)
        elif s.startswith("Down"):
            colors.append(COLOR_LIGHT_BLUE)
        elif s == "Tied":
            colors.append(COLOR_GREY)
        else:
            colors.append(COLOR_BLUE)

    bars = ax.bar(STATE_LABELS, vals, color=colors, width=0.65,
                  edgecolor="white", linewidth=1.2)

    # value labels
    for bar, v in zip(bars, vals):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + (0.05 if h >= 0 else -0.05),
                f"{v:+.2f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=11, fontfamily="Arial", color="#222222", fontweight="bold")

    # zero line
    ax.axhline(0, color="#222222", linewidth=0.8)

    # styling
    ax.set_title("NFI-Score-Net by Score State",
                 loc="left", fontsize=22, fontfamily="Impact",
                 color="#1A1A1A", pad=18)
    ax.text(0, 1.02,
            "League-average net-front events per 60 (Off − Def), "
            "TOI-weighted across 379 forwards · 6 seasons (2020-21 → 2025-26) · "
            "ES, regulation, regular-season, CNFI+MNFI Fenwick",
            transform=ax.transAxes, fontsize=9, fontfamily="Arial",
            color=COLOR_GREY)

    ax.set_xlabel("Score state (shooter's team perspective)",
                  fontsize=11, fontfamily="Arial", color="#333333", labelpad=10)
    ax.set_ylabel("NFI-Score-Net  (events / 60)",
                  fontsize=11, fontfamily="Arial", color="#333333", labelpad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_GREY)
    ax.spines["bottom"].set_color(COLOR_GREY)
    ax.tick_params(colors="#333333", labelsize=10)

    # logo bottom-right via OffsetImage
    if logo_path is not None:
        try:
            img = plt.imread(str(logo_path))
            zoom = 0.10
            imagebox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imagebox, (1.0, -0.18), xycoords="axes fraction",
                                box_alignment=(1.0, 0.0), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f"[step4] WARNING: could not embed logo ({e})")

    plt.tight_layout()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=200, facecolor="#FFFFFF",
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[step4] wrote {CHART_PATH}")

    # monotonicity diagnostic
    seq = [means[s] for s in STATE_ORDER]
    is_monotonic_dec = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
    is_monotonic_inc = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
    print(f"[step4] monotonic across Down2 → Up2? "
          f"{'decreasing' if is_monotonic_dec else 'increasing' if is_monotonic_inc else 'NEITHER'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
