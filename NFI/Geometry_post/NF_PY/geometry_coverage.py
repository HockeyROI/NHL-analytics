"""
Publication-ready NHL Goalie Net Coverage Geometry Diagram  (v2)
Layout:
  Top row    – Standing overlay   |  Butterfly overlay
  Bottom row – 3.6× callout panel |  Coverage % bar chart
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon
from matplotlib.path import Path
import matplotlib.ticker as ticker

# ── Constants ────────────────────────────────────────────────────────────────
NET_W, NET_H = 72.0, 48.0          # inches
POST_LW  = 3.0                      # post linewidth (pts)
NET_BG   = "#EEF2F7"
NET_GRID = "#D4DCE8"

# ── Archetypes ───────────────────────────────────────────────────────────────
ARCHETYPES = dict(
    tall_lean = dict(
        label   = "Tall & Lean",
        spec    = "79″ (6′7″)  ·  195 lbs",
        height  = 79, weight = 195,
        fill    = "#1565C0", alpha = 0.55,
        edge    = "#0D47A1", ealpha = 0.90,
    ),
    wide_short = dict(
        label   = "Short & Wide",
        spec    = "71″ (5′11″)  ·  220 lbs",
        height  = 71, weight = 220,
        fill    = "#C62828", alpha = 0.48,
        edge    = "#B71C1C", ealpha = 0.90,
    ),
)

# ── Geometry ─────────────────────────────────────────────────────────────────
def metrics(h, w):
    shoulder_w       = h * 0.26
    hip_w            = w * 0.070
    stance_h         = min(h - 4, NET_H)
    pad_len          = min(h * 0.46, 43.0)
    pad_horiz        = pad_len * np.cos(np.radians(22))
    butterfly_w      = min(hip_w + 2 * pad_horiz, NET_W)
    butterfly_body_h = h * 0.43
    butterfly_pad_h  = 10.5
    return dict(shoulder_w=shoulder_w, hip_w=hip_w, stance_h=stance_h,
                pad_len=pad_len, butterfly_w=butterfly_w,
                butterfly_body_h=butterfly_body_h,
                butterfly_pad_h=butterfly_pad_h)

def standing_poly(m, cx=NET_W/2):
    hw = m["hip_w"]      / 2
    sw = m["shoulder_w"] / 2
    h  = m["stance_h"]
    ww = hw + (sw - hw) * 0.38          # waist level
    hd = sw * 0.50                      # head half-width
    xs = [cx-hw,  cx+hw,  cx+ww,  cx+sw,   cx+sw*0.88, cx+hd,
          cx-hd,  cx-sw*0.88, cx-sw,  cx-ww]
    ys = [0.0,    0.0,    h*0.42, h*0.60,  h*0.82,     h,
          h,      h*0.82,     h*0.60, h*0.42]
    return np.column_stack([xs, ys])

def butterfly_poly(m, cx=NET_W/2):
    pw = m["butterfly_w"]    / 2
    sw = m["shoulder_w"]     / 2
    ph = m["butterfly_pad_h"]
    bh = m["butterfly_body_h"]
    # T-shape with slightly rounded inner shoulder transition
    xs = [cx-pw, cx+pw, cx+pw, cx+sw*1.05, cx+sw, cx-sw, cx-sw*1.05, cx-pw]
    ys = [0.0,   0.0,   ph,    ph,          bh,    bh,    ph,          ph   ]
    return np.column_stack([xs, ys])

def poly_coverage_pct(poly):
    path = Path(poly)
    xs = np.linspace(0, NET_W, 500)
    ys = np.linspace(0, NET_H, 380)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    return path.contains_points(pts).mean() * 100

# ── Drawing helpers ──────────────────────────────────────────────────────────
def draw_net(ax):
    """Draw net background, crosshatch unblocked area, posts, crossbar, ice."""
    # net background
    ax.add_patch(mpatches.Rectangle((0, 0), NET_W, NET_H,
                 fc=NET_BG, ec="none", zorder=0))
    # subtle grid
    for x in np.arange(0, NET_W+1, 12):
        ax.axvline(x, color=NET_GRID, lw=0.6, zorder=1)
    for y in np.arange(0, NET_H+1, 12):
        ax.axhline(y, color=NET_GRID, lw=0.6, zorder=1)
    # ice line
    ax.axhline(0, color="#AAAAAA", lw=1.2, zorder=4)
    # posts & crossbar
    for x in [0, NET_W]:
        ax.plot([x, x], [0, NET_H], color="#222222", lw=POST_LW, zorder=8,
                solid_capstyle="butt")
    ax.plot([0, NET_W], [NET_H, NET_H], color="#222222", lw=POST_LW, zorder=8,
            solid_capstyle="butt")
    ax.set_xlim(-15, NET_W + 15)
    ax.set_ylim(-14, NET_H + 8)
    ax.set_aspect("equal")
    ax.axis("off")

def draw_goalie(ax, m, poly_fn, spec, zorder_base=3):
    poly = poly_fn(m)
    fill  = spec["fill"]
    alpha = spec["alpha"]
    edge  = spec["edge"]
    patch = plt.Polygon(poly, closed=True, fc=fill, ec=edge,
                        lw=1.8, alpha=alpha, zorder=zorder_base)
    # edge re-draw at full opacity for crisp outline
    edge_patch = plt.Polygon(poly, closed=True, fc="none", ec=edge,
                             lw=1.8, alpha=spec["ealpha"], zorder=zorder_base+1)
    ax.add_patch(patch)
    ax.add_patch(edge_patch)
    return poly

def dim_arrow(ax, x1, y1, x2, y2, label, color, fontsize=8, lw=1.2,
              offset=(0,0), ha="center", va="center"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="<->", color=color, lw=lw,
                                mutation_scale=10))
    mx = (x1+x2)/2 + offset[0]
    my = (y1+y2)/2 + offset[1]
    ax.text(mx, my, label, fontsize=fontsize, color=color, ha=ha, va=va,
            fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))

def coverage_badge(ax, pct, color, x=NET_W-1, y=NET_H-1):
    ax.text(x, y, f"{pct:.1f}%\ncoverage",
            ha="right", va="top", fontsize=9.5, color=color,
            fontweight="bold",
            bbox=dict(fc="white", ec=color, lw=1.4,
                      boxstyle="round,pad=0.45", alpha=0.95),
            zorder=10)

# ── Panel A – Standing overlay ───────────────────────────────────────────────
def panel_standing(ax):
    draw_net(ax)
    cx = NET_W / 2
    results = {}
    for key, spec in ARCHETYPES.items():
        m    = metrics(spec["height"], spec["weight"])
        poly = draw_goalie(ax, m, standing_poly, spec,
                           zorder_base=3 if key=="tall_lean" else 2)
        results[key] = (m, poly_coverage_pct(poly))

    # dim arrows for TALL/LEAN (outer)
    m_tl, cov_tl = results["tall_lean"]
    m_ws, cov_ws = results["wide_short"]

    # shoulder width – tall/lean (upper arrow)
    sw = m_tl["shoulder_w"]
    dim_arrow(ax, cx-sw/2, m_tl["stance_h"]*0.80+3,
                  cx+sw/2, m_tl["stance_h"]*0.80+3,
              f"Shoulder  {sw:.1f}\"",
              color=ARCHETYPES["tall_lean"]["edge"],
              offset=(0, 2.5), fontsize=8)

    # shoulder width – short/wide (lower arrow)
    sw2 = m_ws["shoulder_w"]
    dim_arrow(ax, cx-sw2/2, m_ws["stance_h"]*0.80-6,
                  cx+sw2/2, m_ws["stance_h"]*0.80-6,
              f"Shoulder  {sw2:.1f}\"",
              color=ARCHETYPES["wide_short"]["edge"],
              offset=(0, -2.5), fontsize=8)

    # hip width – short/wide
    hw = m_ws["hip_w"]
    dim_arrow(ax, cx-hw/2, 5, cx+hw/2, 5,
              f"Hip  {hw:.1f}\"",
              color=ARCHETYPES["wide_short"]["edge"],
              offset=(0, 2.5), fontsize=7.5)

    # net height tick on left
    ax.annotate("", xy=(-8, NET_H), xytext=(-8, 0),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.0,
                                mutation_scale=8))
    ax.text(-11, NET_H/2, '48"', fontsize=8, color="#555",
            ha="right", va="center", fontweight="bold")

    # net width tick on top
    ax.annotate("", xy=(NET_W, NET_H+5), xytext=(0, NET_H+5),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.0,
                                mutation_scale=8))
    ax.text(NET_W/2, NET_H+7, '72"', fontsize=8, color="#555",
            ha="center", va="bottom", fontweight="bold")

    # coverage badges
    coverage_badge(ax, cov_tl, ARCHETYPES["tall_lean"]["edge"],
                   x=NET_W-1, y=NET_H-1)
    coverage_badge(ax, cov_ws, ARCHETYPES["wide_short"]["edge"],
                   x=NET_W-1, y=NET_H-12)

    ax.set_title("Standing Position", fontsize=13, fontweight="bold",
                 color="#1a1a2e", pad=10)
    return results

# ── Panel B – Butterfly overlay ──────────────────────────────────────────────
def panel_butterfly(ax):
    draw_net(ax)
    cx = NET_W / 2
    results = {}
    for key, spec in ARCHETYPES.items():
        m    = metrics(spec["height"], spec["weight"])
        poly = draw_goalie(ax, m, butterfly_poly, spec,
                           zorder_base=3 if key=="tall_lean" else 2)
        results[key] = (m, poly_coverage_pct(poly))

    m_tl, cov_tl = results["tall_lean"]
    m_ws, cov_ws = results["wide_short"]

    # butterfly pad spread – tall/lean
    bw = m_tl["butterfly_w"]
    dim_arrow(ax, max(cx-bw/2, 1), -5.5, min(cx+bw/2, NET_W-1), -5.5,
              f"Pad spread  {bw:.1f}\"",
              color=ARCHETYPES["tall_lean"]["edge"],
              offset=(0, -4), fontsize=7.8)

    # body height in butterfly – tall/lean
    dim_arrow(ax, -9, 0, -9, m_tl["butterfly_body_h"],
              f'{m_tl["butterfly_body_h"]:.0f}"',
              color=ARCHETYPES["tall_lean"]["edge"],
              offset=(-2, 0), ha="right", fontsize=7.5)

    # body height in butterfly – short/wide
    mid_ws = m_ws["butterfly_body_h"]
    ax.annotate("", xy=(NET_W+9, mid_ws), xytext=(NET_W+9, 0),
                arrowprops=dict(arrowstyle="<->", color=ARCHETYPES["wide_short"]["edge"],
                                lw=1.1, mutation_scale=8))
    ax.text(NET_W+12, mid_ws/2, f'{mid_ws:.0f}"',
            fontsize=7.5, color=ARCHETYPES["wide_short"]["edge"],
            ha="left", va="center", fontweight="bold")

    # hip width difference callout
    hw_tl = m_tl["hip_w"]
    hw_ws = m_ws["hip_w"]
    ph = m_tl["butterfly_pad_h"]
    dim_arrow(ax, cx-hw_ws/2, ph/2, cx+hw_ws/2, ph/2,
              f'Hip  {hw_ws:.1f}"',
              color=ARCHETYPES["wide_short"]["edge"],
              offset=(0, 2.5), fontsize=7.2)

    # coverage badges
    coverage_badge(ax, cov_tl, ARCHETYPES["tall_lean"]["edge"],
                   x=NET_W-1, y=NET_H-1)
    coverage_badge(ax, cov_ws, ARCHETYPES["wide_short"]["edge"],
                   x=NET_W-1, y=NET_H-12)

    ax.set_title("Butterfly Position", fontsize=13, fontweight="bold",
                 color="#1a1a2e", pad=10)
    return results

# ── Panel C – 3.6× callout ───────────────────────────────────────────────────
def panel_callout(ax):
    draw_net(ax)
    cx = NET_W / 2

    # Average shoulder width
    avg_sw = np.mean([
        metrics(s["height"], s["weight"])["shoulder_w"]
        for s in ARCHETYPES.values()
    ])
    ratio  = NET_W / avg_sw

    # Draw the average goalie silhouette (greyed)
    m_avg = dict(
        shoulder_w       = avg_sw,
        hip_w            = avg_sw * 0.68,
        stance_h         = NET_H,
        pad_len          = 38,
        butterfly_w      = NET_W,
        butterfly_body_h = 33,
        butterfly_pad_h  = 10.5,
    )
    poly = standing_poly(m_avg)
    ax.add_patch(plt.Polygon(poly, closed=True,
                             fc="#78909C", ec="#455A64", lw=1.5,
                             alpha=0.45, zorder=3))

    # ── 3.6× hero callout ──
    # Arrow spanning full net width at mid-height
    arrow_y = NET_H * 0.52
    ax.annotate("", xy=(NET_W, arrow_y), xytext=(0, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="#1a1a2e", lw=2.0,
                                mutation_scale=14),
                zorder=9)
    # Shoulder-width bracket centred on goalie
    bracket_y = arrow_y
    lx = cx - avg_sw/2
    rx = cx + avg_sw/2
    for bx in [lx, rx]:
        ax.plot([bx, bx], [bracket_y-2, bracket_y+2],
                color="#1565C0", lw=2.0, zorder=10)
    ax.plot([lx, rx], [bracket_y, bracket_y],
            color="#1565C0", lw=2.0, zorder=10)

    # 3.6× text box — hero element
    ax.text(cx, NET_H*0.72,
            f"  Net width = {ratio:.1f}×\n  goalie shoulder width  ",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#1a1a2e", zorder=12,
            bbox=dict(fc="#FFF9C4", ec="#F9A825", lw=2.0,
                      boxstyle="round,pad=0.6", alpha=0.97))

    # sub-labels
    ax.text(cx, NET_H * 0.52 + 4.5,
            f'Net: {NET_W:.0f}"', fontsize=8.5, ha="center",
            color="#1a1a2e", fontweight="bold")
    ax.text(cx, bracket_y - 5.5,
            f'Avg shoulder: {avg_sw:.1f}"', fontsize=8,
            ha="center", color="#1565C0", fontweight="bold")

    # left & right uncovered zones
    for xz, xl, ha in [(0, cx-avg_sw/2, "center"), (cx+avg_sw/2, NET_W, "center")]:
        mid = (xz + xl) / 2
        ax.text(mid, NET_H * 0.25, "Exposed\nnet",
                ha="center", va="center", fontsize=8, color="#C62828",
                alpha=0.8, style="italic")

    # net size label
    ax.text(cx, NET_H + 6, '72"  wide  ×  48"  tall',
            ha="center", va="bottom", fontsize=9, color="#555",
            fontweight="bold")

    ax.set_title("Why Frame Size Matters", fontsize=13,
                 fontweight="bold", color="#1a1a2e", pad=10)

# ── Panel D – Coverage bar chart ─────────────────────────────────────────────
def panel_barchart(ax, stand_results, fly_results):
    positions  = ["Standing", "Butterfly"]
    bar_w      = 0.30
    x          = np.array([0.0, 1.0])
    colors     = {
        "tall_lean":  ARCHETYPES["tall_lean"]["fill"],
        "wide_short": ARCHETYPES["wide_short"]["fill"],
    }

    bars = {}
    for i, (pos, res) in enumerate(zip(positions, [stand_results, fly_results])):
        for j, (key, (m, cov)) in enumerate(res.items()):
            offset = (j - 0.5) * bar_w * 1.15
            b = ax.bar(x[i] + offset, cov, bar_w,
                       color=colors[key], alpha=0.82,
                       edgecolor="white", lw=1.2, zorder=3)
            ax.text(x[i] + offset, cov + 0.6, f"{cov:.1f}%",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color=colors[key])

    ax.set_xticks(x)
    ax.set_xticklabels(positions, fontsize=12, fontweight="bold")
    ax.set_ylabel("Net Coverage (%)", fontsize=10, color="#333")
    ax.set_ylim(0, 48)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.grid(axis="y", which="minor", alpha=0.15, zorder=1)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_facecolor("#FAFAFA")

    # delta annotations
    for i, res in enumerate([stand_results, fly_results]):
        covs = [v[1] for v in res.values()]
        diff = covs[0] - covs[1]      # tall_lean minus wide_short
        sign = "+" if diff >= 0 else ""
        ax.text(x[i], max(covs) + 4.5,
                f"Δ {sign}{diff:.1f}%",
                ha="center", fontsize=8.5, color="#555", style="italic")

    # legend
    handles = [
        mpatches.Patch(fc=colors["tall_lean"],  ec="white", lw=1,
                       label=ARCHETYPES["tall_lean"]["label"]),
        mpatches.Patch(fc=colors["wide_short"], ec="white", lw=1,
                       label=ARCHETYPES["wide_short"]["label"]),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left",
              framealpha=0.9, edgecolor="#CCC")
    ax.set_title("Coverage % Comparison", fontsize=13,
                 fontweight="bold", color="#1a1a2e", pad=10)

# ── Main ─────────────────────────────────────────────────────────────────────
def make_figure():
    fig = plt.figure(figsize=(18, 13), facecolor="white")

    # ── title block ──
    fig.text(0.5, 0.984,
             "NHL Goalie Net Coverage Geometry",
             ha="center", va="top", fontsize=20, fontweight="bold",
             color="#1a1a2e")
    fig.text(0.5, 0.963,
             f"Lean/Tall vs Short/Wide  ·  Standing & Butterfly Positions  ·  "
             f"Net: {int(NET_W)}\" wide × {int(NET_H)}\" tall  ·  All dimensions in inches",
             ha="center", va="top", fontsize=10.5, color="#555")

    # ── goalie legend (sits just below subtitle) ──
    leg_elements = [
        mpatches.Patch(fc=ARCHETYPES["tall_lean"]["fill"],
                       ec=ARCHETYPES["tall_lean"]["edge"], lw=1.5, alpha=0.75,
                       label=f"Tall & Lean — {ARCHETYPES['tall_lean']['spec']}"),
        mpatches.Patch(fc=ARCHETYPES["wide_short"]["fill"],
                       ec=ARCHETYPES["wide_short"]["edge"], lw=1.5, alpha=0.68,
                       label=f"Short & Wide — {ARCHETYPES['wide_short']['spec']}"),
        mpatches.Patch(fc=NET_BG, ec="#333", lw=1.5,
                       label="Uncovered net"),
    ]
    fig.legend(handles=leg_elements, loc="upper center",
               bbox_to_anchor=(0.5, 0.950), ncol=3,
               fontsize=9.5, framealpha=0.95, edgecolor="#CCC",
               handlelength=1.8)

    # ── 4-panel grid ──
    gs = fig.add_gridspec(2, 2,
                          left=0.04, right=0.97,
                          top=0.895, bottom=0.09,
                          wspace=0.10, hspace=0.36)

    ax_stand  = fig.add_subplot(gs[0, 0])
    ax_fly    = fig.add_subplot(gs[0, 1])
    ax_call   = fig.add_subplot(gs[1, 0])
    ax_bar    = fig.add_subplot(gs[1, 1])

    stand_results = panel_standing(ax_stand)
    fly_results   = panel_butterfly(ax_fly)
    panel_callout(ax_call)
    panel_barchart(ax_bar, stand_results, fly_results)

    # ── assumptions footnote ──
    assumptions = (
        "Assumptions:  shoulder width = height × 0.26  ·  hip width = weight × 0.070  ·  "
        "effective stance height = height − 4″  ·  "
        "pad length = min(height × 0.46, 43″)  ·  butterfly splay angle = 22° from ice  ·  "
        "butterfly body height = height × 0.43"
    )
    fig.text(0.5, 0.005, assumptions,
             ha="center", va="bottom", fontsize=7.5,
             color="#999", style="italic")

    out = ("/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/"
           "NHL analysis/geometry_coverage.png")
    plt.savefig(out, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved → {out}")

    # ── print summary ──
    print("\nCoverage summary:")
    print(f"  {'Position':<12}  {'Goalie':<18}  {'Coverage':>8}")
    print("  " + "─" * 42)
    for pos, res in [("Standing", stand_results), ("Butterfly", fly_results)]:
        for key, (m, cov) in res.items():
            lbl = ARCHETYPES[key]["label"]
            print(f"  {pos:<12}  {lbl:<18}  {cov:>7.1f}%")

    m_tl = metrics(ARCHETYPES["tall_lean"]["height"],
                   ARCHETYPES["tall_lean"]["weight"])
    avg_sw = np.mean([metrics(s["height"], s["weight"])["shoulder_w"]
                      for s in ARCHETYPES.values()])
    print(f"\n  Net width / avg shoulder width = {NET_W:.0f} / {avg_sw:.1f} = {NET_W/avg_sw:.2f}×")


if __name__ == "__main__":
    make_figure()
