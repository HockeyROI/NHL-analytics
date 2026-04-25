"""Build 8 charts for HockeyROI posts / Sloan paper.

Output: NFI/output/charts/chart_*.png
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "axes.titleweight": "bold",
})

CHARTS = Path("/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/NFI/output/charts")
CHARTS.mkdir(parents=True, exist_ok=True)

# Brand colors
BLUE       = "#2E7DC4"
LIGHT_BLUE = "#4AB3E8"
ORANGE     = "#FF6B35"
GREEN      = "#44AA66"
YELLOW     = "#FFB700"
RED        = "#CC3333"
GREY       = "#888888"
DARK_TEXT  = "#1B3A5C"
WHITE_BG   = "#FFFFFF"

def save(fig, name):
    fp = CHARTS / name
    fig.savefig(fp, dpi=180, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return fp

# =====================================================================
# CHART A — Fenwick Factor Comparison
# =====================================================================
def chart_a():
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.set_facecolor("white")
    metrics = ["NFI%", "Fenwick", "Corsi", "HD Corsi"]
    vals    = [10.71,  11.91,     3.89,    1.82]
    colors  = [ORANGE, BLUE,      GREY,    GREY]
    ann     = ["(Spatial Fenwick)",  "Under-corrects by 71%",
               "Roughly valid (+10%)","Over-corrects by 93%"]
    y = np.arange(len(metrics))[::-1]
    bars = ax.barh(y, vals, color=colors, height=0.65, edgecolor="#222", linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels(metrics, fontsize=12)
    ax.axvline(3.5, color=RED, linestyle="--", linewidth=1.8, zorder=1)
    ax.text(3.5, len(metrics)-0.4, " Traditional\n 3.5 assumption",
            color=RED, fontsize=9.5, va="top", ha="left", fontweight="bold")
    for bar, v, a in zip(bars, vals, ann):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2, f"{v:.2f} pp  —  {a}",
                va="center", ha="left", fontsize=10.5, color=DARK_TEXT)
    ax.set_xlim(0, 18)
    ax.set_xlabel("Empirical OZ − DZ pct-point gap", fontsize=11)
    ax.set_title("The Fenwick Zone Adjustment Factor Is Wrong",
                 fontsize=15, color=DARK_TEXT, loc="left", pad=14)
    ax.text(0, -0.85,
            "Empirical factors vs traditional 3.5 assumption — OZ/DZ shift analysis, N = 807K shifts",
            transform=ax.get_yaxis_transform(), fontsize=10, color="#555555", ha="left")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)
    fig.tight_layout()
    return save(fig, "chart_fenwick_factor.png")

# =====================================================================
# CHART B — Three-Pillar Stacked R²
# =====================================================================
def chart_b():
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="white")
    ax.set_facecolor("white")
    segs   = [("Forward generation",  0.380, BLUE),
              ("Suppression adds",    0.111, LIGHT_BLUE),
              ("Goalie adds",         0.051, ORANGE),
              ("Unexplained",         0.458, GREY)]
    left = 0
    for name, val, col in segs:
        ax.barh([0], [val], left=left, color=col, height=0.55, edgecolor="white", linewidth=1.5)
        ax.text(left + val/2, 0, f"{name}\n{val:.3f}", ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold")
        left += val
    # Benchmarks
    benchmarks = [("Corsi 0.397", 0.397, GREY),
                  ("Fenwick 0.413", 0.413, BLUE),
                  ("xG% 0.538", 0.538, ORANGE)]
    for label, x, color in benchmarks:
        ax.axvline(x, ymin=0.05, ymax=0.95, color=color, linestyle="--", linewidth=1.6)
        ax.text(x, 0.45, label, rotation=90, color=color, fontsize=9,
                va="bottom", ha="center", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color, pad=2, boxstyle="round,pad=0.2"))
    ax.set_yticks([])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Cumulative R² vs standings points", fontsize=11)
    ax.set_title("Three Pillars Explain 65.5% of Standings Variance",
                 fontsize=15, color=DARK_TEXT, loc="left", pad=14)
    ax.text(0, -0.85,
            "Incremental R² contribution — N = 126 team-seasons pooled",
            transform=ax.get_yaxis_transform(), fontsize=10, color="#555555", ha="left")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--", zorder=0)
    fig.tight_layout()
    return save(fig, "chart_three_pillar.png")

# =====================================================================
# CHART C — Carolina block asymmetry
# =====================================================================
def chart_c():
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor="white")
    ax.set_facecolor("white")
    groups = ["Carolina", "League avg"]
    for_rate = [20, 27]
    ag_rate  = [36, 27]
    x = np.arange(len(groups))
    w = 0.36
    bars_f = ax.bar(x - w/2, for_rate, width=w, color=BLUE,  label="For (team's shots blocked)")
    bars_a = ax.bar(x + w/2, ag_rate,  width=w, color=ORANGE, label="Against (opp's shots blocked)")
    for b, v in list(zip(bars_f, for_rate)) + list(zip(bars_a, ag_rate)):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v}%",
                ha="center", fontsize=10.5, color=DARK_TEXT, fontweight="bold")
    # 16pp gap arrow
    ax.annotate("", xy=(x[0]+w/2, 36), xytext=(x[0]-w/2, 20),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    ax.text(x[0], 29, " 16pp gap\nlargest in dataset",
            ha="center", fontsize=10.5, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel("% of shot attempts blocked", fontsize=11)
    ax.set_ylim(0, 45)
    ax.set_title("Carolina's Block Rate Asymmetry",
                 fontsize=15, color=DARK_TEXT, loc="left", pad=14)
    ax.text(0, -0.18,
            "% of shot attempts blocked For vs Against — 5 seasons pooled",
            transform=ax.transAxes, fontsize=10, color="#555555", ha="left")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", zorder=0)
    fig.tight_layout()
    return save(fig, "chart_carolina_block.png")

# =====================================================================
# CHART D — Carolina percentile profile
# =====================================================================
def chart_d():
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor="white")
    ax.set_facecolor("white")
    metrics = ["NFI%_ZA", "RelNFI_A%", "Two-way F score",
               "Defensive F suppression", "Block asymmetry"]
    # percentile (1-100, where 1 = best; spec says "top 5%" etc.)
    # Show as percentile RANK displayed (100 − top percentile) so bars grow to right = better.
    top_pct = [5, 3, 5, 3, 1]  # "top X%"
    percentile_shown = [100 - p for p in top_pct]  # 95, 97, 95, 97, 99
    y = np.arange(len(metrics))[::-1]
    bars = ax.barh(y, percentile_shown, color=GREEN, height=0.58,
                   edgecolor="#222", linewidth=0.8)
    for bar, p, pct in zip(bars, top_pct, percentile_shown):
        ax.text(pct + 0.6, bar.get_y() + bar.get_height()/2,
                f"Top {p}%  (p{pct})", va="center", fontsize=10.5, color=DARK_TEXT)
    ax.axvline(50, color=GREY, linestyle="--", linewidth=1.4)
    ax.text(50, len(metrics)-0.5, " League avg (p50)", color=GREY, fontsize=9.5,
            va="top", ha="left", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(metrics, fontsize=11.5)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Percentile rank (higher = better)", fontsize=11)
    ax.set_title("Carolina — Elite Across Every Spatial Metric",
                 fontsize=15, color=DARK_TEXT, loc="left", pad=14)
    ax.text(0, -0.18,
            "Percentile rank across NFI framework metrics — 5 seasons pooled",
            transform=ax.transAxes, fontsize=10, color="#555555", ha="left")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--", zorder=0)
    fig.tight_layout()
    return save(fig, "chart_carolina_profile.png")

# =====================================================================
# CHART E — TNZI pipeline diagram
# =====================================================================
def chart_e():
    fig, ax = plt.subplots(figsize=(14, 3.2), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 14); ax.set_ylim(0, 3); ax.axis("off")
    steps = ["NHL API", "x/y\nCoordinates", "Faceoff\nClassification\n(OZ/DZ/NZ)",
             "Shift\nAttribution", "TNZI\nCalculation",
             "ZQoC / ZQoL\nAdjustment", "Streamlit\nDashboard"]
    n = len(steps)
    margin = 0.2
    box_w = (14 - margin*(n+1)) / n
    arrow_len = margin
    box_h = 1.6
    y_mid = 1.5
    centers = []
    for i, s in enumerate(steps):
        x = margin + i*(box_w + margin)
        rect = mpatches.FancyBboxPatch((x, y_mid - box_h/2), box_w, box_h,
                                        boxstyle="round,pad=0.06,rounding_size=0.12",
                                        facecolor=BLUE, edgecolor=BLUE, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + box_w/2, y_mid, s, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        centers.append((x, x + box_w))
    # Arrows between boxes
    for i in range(n-1):
        _, x_end_prev = centers[i]
        x_start_next, _ = centers[i+1]
        arrow = FancyArrowPatch((x_end_prev + 0.02, y_mid),
                                 (x_start_next - 0.02, y_mid),
                                 arrowstyle="-|>", mutation_scale=18,
                                 color=ORANGE, linewidth=2.2)
        ax.add_patch(arrow)
    ax.text(7.0, 2.85, "Automated Zone Impact Pipeline",
            ha="center", va="top", fontsize=15,
            color=DARK_TEXT, fontweight="bold")
    fig.tight_layout()
    return save(fig, "chart_tnzi_pipeline.png")

# =====================================================================
# CHART F — TNZI benchmark comparison
# =====================================================================
def chart_f():
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor="white")
    ax.set_facecolor("white")
    items = [("NFI%_ZA", 0.764, ORANGE),
             ("xG%",     0.733, GREY),
             ("Corsi",   0.651, GREY),
             ("HD Corsi",0.651, GREY),
             ("Fenwick", 0.643, GREY),
             ("TNZI_L",  0.589, BLUE),
             ("PDO",     0.580, GREY)]
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    labels  = [it[0] for it in items_sorted]
    vals    = [it[1] for it in items_sorted]
    colors  = [it[2] for it in items_sorted]
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, vals, color=colors, height=0.62, edgecolor="#222", linewidth=0.7)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.008, bar.get_y() + bar.get_height()/2,
                f"r = {v:.3f}", va="center", fontsize=10.5, color=DARK_TEXT)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11.5)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Pearson r to standings points", fontsize=11)
    ax.set_title("How Zone Time Compares to Conventional Metrics",
                 fontsize=15, color=DARK_TEXT, loc="left", pad=14)
    ax.text(0, -0.15,
            "Pearson correlation to team standings points — N = 126 team-seasons",
            transform=ax.transAxes, fontsize=10, color="#555555", ha="left")
    # Bracket around HD Corsi and TNZI_L
    try:
        hd_idx = labels.index("HD Corsi"); tn_idx = labels.index("TNZI_L")
    except ValueError:
        hd_idx = tn_idx = 0
    y_top = y[min(hd_idx, tn_idx)] + 0.32
    y_bot = y[max(hd_idx, tn_idx)] - 0.32
    x_br  = 0.82
    ax.plot([x_br, x_br+0.012, x_br+0.012, x_br],
            [y_top, y_top, y_bot, y_bot], color=BLUE, linewidth=1.6)
    ax.text(x_br+0.02, (y_top+y_bot)/2, "TNZI_L\nterritory",
            color=BLUE, fontsize=10, fontweight="bold", va="center")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--", zorder=0)
    fig.tight_layout()
    return save(fig, "chart_tnzi_benchmark.png")

# =====================================================================
# CHART G — NFI discovery timeline
# =====================================================================
def chart_g():
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor="white")
    ax.set_facecolor("white")
    milestones = [
        "Wide shots don't\nbeat goalies",
        "921K shots\nmapped",
        "CNFI 13.04%\nconfirmed",
        "Center lane\neffect found",
        "NFI% beats\nxG%",
        "Fenwick factor\n11.91 pp",
        "RelNFI split\nbuilt",
        "PP/PK\nvalidated",
    ]
    n = len(milestones)
    x = np.linspace(0.7, 13.3, n)
    y_line = 2.2
    ax.hlines(y_line, x[0]-0.3, x[-1]+0.3, colors=BLUE, linewidth=2.5, zorder=1)
    # alternate label positions above/below
    for i, (xi, ms) in enumerate(zip(x, milestones)):
        ax.scatter([xi], [y_line], s=180, color=ORANGE, zorder=3,
                   edgecolor="white", linewidth=1.8)
        ax.text(xi, y_line, str(i+1), ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold", zorder=4)
        ty = y_line + 0.95 if i % 2 == 0 else y_line - 1.05
        va = "bottom" if i % 2 == 0 else "top"
        ax.annotate(ms, xy=(xi, y_line), xytext=(xi, ty),
                     ha="center", va=va, fontsize=10, color=DARK_TEXT,
                     arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.5); ax.axis("off")
    ax.text(0.3, 4.1, "The NFI Discovery Sequence",
            fontsize=15, color=DARK_TEXT, fontweight="bold")
    fig.tight_layout()
    return save(fig, "chart_nfi_journey.png")

# =====================================================================
# CHART H — Top 100 metric scoreboard
# =====================================================================
def chart_h():
    fig, ax = plt.subplots(figsize=(13, 5.2), facecolor="white")
    ax.set_facecolor("white"); ax.axis("off")
    boxes = [("NFI%_ZA", 0.764, 0.583, ORANGE),
             ("xG%",     0.733, 0.538, BLUE),
             ("FF%_ZA",  0.655, 0.429, LIGHT_BLUE),
             ("CF%_ZA",  0.631, 0.398, GREY)]
    n = len(boxes)
    margin = 0.4
    total_w = 13 - margin*(n+1)
    w = total_w / n
    h = 3.5
    y = 0.55
    for i, (name, r, r2, col) in enumerate(boxes):
        x = margin + i*(w + margin)
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.08,rounding_size=0.18",
                                        facecolor=col, edgecolor=col, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.45, name, ha="center", va="center",
                color="white", fontsize=16, fontweight="bold")
        ax.text(x + w/2, y + h/2 + 0.25, f"r = {r:.3f}", ha="center", va="center",
                color="white", fontsize=24, fontweight="bold")
        ax.text(x + w/2, y + h/2 - 0.45, f"R² = {r2:.3f}", ha="center", va="center",
                color="white", fontsize=14)
        ax.text(x + w/2, y + 0.45, "Correlation to Winning", ha="center", va="center",
                color="white", fontsize=9.5, style="italic", alpha=0.9)
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2)
    ax.text(0.3, 4.75, "Four Metrics. Four Different Rankings. Which Do You Trust?",
            fontsize=15, color=DARK_TEXT, fontweight="bold")
    fig.tight_layout()
    return save(fig, "chart_top100_scoreboard.png")

# =====================================================================
# Run all
# =====================================================================
if __name__ == "__main__":
    results = []
    for name, fn in [("A chart_fenwick_factor",    chart_a),
                     ("B chart_three_pillar",      chart_b),
                     ("C chart_carolina_block",    chart_c),
                     ("D chart_carolina_profile",  chart_d),
                     ("E chart_tnzi_pipeline",     chart_e),
                     ("F chart_tnzi_benchmark",    chart_f),
                     ("G chart_nfi_journey",       chart_g),
                     ("H chart_top100_scoreboard", chart_h)]:
        try:
            fp = fn()
            size = fp.stat().st_size
            results.append((name, "OK", fp.name, size))
        except Exception as e:
            results.append((name, f"FAIL: {type(e).__name__}: {e}", None, 0))
    print()
    print(f"{'chart':<34} {'status':<10} {'file':<36} {'bytes':>10}")
    for name, status, fn, size in results:
        print(f"{name:<34} {status:<10} {fn or '—':<36} {size:>10,}")
