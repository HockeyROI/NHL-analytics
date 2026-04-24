#!/usr/bin/env python3
"""
HockeyROI — Lukas Dostal 3-chart package
  Chart 1: Shot type SV% vs League (2025-26 ES horizontal bars)
  Chart 2: Medium-danger 3-season trend line
  Chart 3: Scouting card summary
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# ─── FONTS ─────────────────────────────────────────────────────────────────────
BEBAS_PATH = "/tmp/BebasNeue-Regular.ttf"
INTER_PATH = "/tmp/Inter-Regular.ttf"

def load_font(path, fallback="Arial"):
    if os.path.exists(path):
        fm.fontManager.addfont(path)
        prop = fm.FontProperties(fname=path)
        return prop.get_name()
    return fallback

BEBAS = load_font(BEBAS_PATH, "Arial Black")
INTER = load_font(INTER_PATH, "Arial")

# ─── BRAND PALETTE ──────────────────────────────────────────────────────────────
BG        = '#0B1D2E'
CARD_BG   = '#1B3A5C'
DOSTAL    = '#2E7DC4'
LEAGUE    = '#FFA940'
LABEL_FG  = '#F0F4F8'
GREY      = '#888888'
GREEN     = '#44AA66'
RED       = '#CC3333'
FOOTER    = '@HockeyROI | hockeyROI.substack.com'

OUT_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Charts/Dostal"

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def pp_color(diff):
    return GREEN if diff >= 0 else RED

def pp_str(diff):
    return f"+{diff*100:.1f}pp" if diff >= 0 else f"{diff*100:.1f}pp"

def set_dark_bg(fig, axes=None):
    fig.patch.set_facecolor(BG)
    if axes is not None:
        for ax in (axes if hasattr(axes, '__iter__') else [axes]):
            ax.set_facecolor(BG)

def add_footer(fig, text=FOOTER, y=0.018):
    fig.text(0.5, y, text, ha='center', va='bottom',
             color=GREY, fontsize=9, fontname=INTER, style='italic')

def spine_style(ax, keep_bottom=False):
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_visible(keep_bottom)
    if keep_bottom:
        ax.spines['bottom'].set_color(GREY)
        ax.spines['bottom'].set_linewidth(0.6)

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Shot Type SV% Grouped Horizontal Bars
# ═══════════════════════════════════════════════════════════════════════════════
def chart1():
    shot_types = ['Wrist', 'Snap', 'Slap', 'Tip-In', 'Deflected']
    ns         = [411, 391, 119, 110, 20]
    dostal     = [.922, .885, .908, .855, .850]
    league     = [.919, .896, .930, .881, .849]
    diffs      = [d - l for d, l in zip(dostal, league)]

    fig, ax = plt.subplots(figsize=(1200/150, 700/150), dpi=150)
    set_dark_bg(fig, ax)

    n      = len(shot_types)
    y      = np.arange(n)
    height = 0.30
    gap    = 0.06

    # Dostal bars (top of each pair)
    bars_d = ax.barh(y + height/2 + gap/2, dostal, height,
                     color=DOSTAL, alpha=0.92, label='Dostal', zorder=3)
    # League bars (bottom)
    bars_l = ax.barh(y - height/2 - gap/2, league, height,
                     color=LEAGUE, alpha=0.90, label='League Avg', zorder=3)

    # Gridlines
    for v in np.arange(0.80, 0.96, 0.02):
        ax.axvline(v, color='#FFFFFF', alpha=0.05, linewidth=0.6, zorder=1)

    # X axis range
    ax.set_xlim(0.820, 0.960)
    ax.set_ylim(-0.65, n - 0.35)

    # Y-tick labels: shot type + n= in grey
    ax.set_yticks(y)
    ax.set_yticklabels([])
    for i, (st, n_val) in enumerate(zip(shot_types, ns)):
        ax.text(-0.001, i, f'{st}  ', ha='right', va='center',
                color=LABEL_FG, fontsize=11, fontname=INTER,
                fontweight='bold', transform=ax.get_yaxis_transform())
        ax.text(-0.001, i - 0.22, f'n={n_val}', ha='right', va='center',
                color=GREY, fontsize=8.5, fontname=INTER,
                transform=ax.get_yaxis_transform())

    # Value labels on bars + differential
    for i, (dv, lv, diff) in enumerate(zip(dostal, league, diffs)):
        # Dostal value
        ax.text(dv + 0.0008, i + height/2 + gap/2, f'{dv:.3f}',
                va='center', color=LABEL_FG, fontsize=9, fontname=INTER,
                fontweight='bold', zorder=5)
        # League value
        ax.text(lv + 0.0008, i - height/2 - gap/2, f'{lv:.3f}',
                va='center', color=LABEL_FG, fontsize=8.5, fontname=INTER,
                alpha=0.75, zorder=5)
        # Differential badge — right side
        col  = pp_color(diff)
        text = pp_str(diff)
        ax.text(0.956, i, text, ha='right', va='center',
                color=col, fontsize=10.5, fontname=INTER,
                fontweight='bold', zorder=5,
                transform=ax.get_yaxis_transform())

    # X axis
    ax.tick_params(axis='x', colors=GREY, labelsize=8)
    ax.set_xlabel('Save Percentage', color=GREY, fontsize=9, fontname=INTER, labelpad=6)
    spine_style(ax, keep_bottom=True)
    ax.tick_params(axis='y', left=False)

    # Title
    fig.text(0.06, 0.95, 'DOSTAL vs LEAGUE — SHOT TYPE SV% (2025-26 ES)',
             ha='left', va='top', color=LABEL_FG, fontsize=18,
             fontname=BEBAS, fontweight='bold')
    fig.text(0.06, 0.89, 'Even Strength | Backhands Excluded | 2025-26 Regular Season',
             ha='left', va='top', color=GREY, fontsize=9.5, fontname=INTER)

    # Legend — below the chart, centred
    fig.legend(
        handles=[
            mpatches.Patch(facecolor=DOSTAL, alpha=0.92, label='Dostal'),
            mpatches.Patch(facecolor=LEAGUE, alpha=0.90, label='League Avg'),
        ],
        loc='lower center', ncol=2, frameon=False,
        labelcolor=LABEL_FG, fontsize=10,
        prop={'family': INTER},
        bbox_to_anchor=(0.55, 0.01)
    )

    add_footer(fig, y=0.072)
    plt.tight_layout(rect=[0.13, 0.10, 1.0, 0.88])

    path = os.path.join(OUT_DIR, 'dostal_shot_type_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Chart 1 saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Medium-Danger 3-Season Trend
# ═══════════════════════════════════════════════════════════════════════════════
def chart2():
    seasons  = ['2023-24', '2024-25', '2025-26']
    xpos     = [0, 1, 2]
    d_vals   = [.936, .923, .884]
    l_vals   = [.910, .905, .896]
    diffs    = [d - l for d, l in zip(d_vals, l_vals)]

    fig, ax = plt.subplots(figsize=(1200/150, 700/150), dpi=150)
    set_dark_bg(fig, ax)

    # Shaded fill between lines
    for i in range(len(xpos) - 1):
        x_seg  = np.linspace(xpos[i], xpos[i+1], 100)
        d_seg  = np.interp(x_seg, xpos, d_vals)
        l_seg  = np.interp(x_seg, xpos, l_vals)
        above  = d_seg >= l_seg
        # Shade where Dostal above
        ax.fill_between(x_seg, d_seg, l_seg,
                        where=above, color=GREEN, alpha=0.12, zorder=1)
        # Shade where Dostal below
        ax.fill_between(x_seg, d_seg, l_seg,
                        where=~above, color=RED, alpha=0.14, zorder=1)

    # Lines
    ax.plot(xpos, d_vals, color=DOSTAL, linewidth=2.8, zorder=4,
            marker='o', markersize=9, markerfacecolor=DOSTAL,
            markeredgecolor=LABEL_FG, markeredgewidth=1.5, label='Dostal')
    ax.plot(xpos, l_vals, color=LEAGUE, linewidth=2.0, zorder=3,
            linestyle='--', marker='o', markersize=7,
            markerfacecolor=LEAGUE, markeredgecolor=LABEL_FG,
            markeredgewidth=1.2, label='League Avg', alpha=0.80)

    # Gridlines
    for v in np.arange(0.870, 0.950, 0.010):
        ax.axhline(v, color='#FFFFFF', alpha=0.05, linewidth=0.6, zorder=0)

    # Annotations
    for i, (dv, lv, diff) in enumerate(zip(d_vals, l_vals, diffs)):
        col  = pp_color(diff)
        text = pp_str(diff)

        # Dostal point label
        offset = 0.0028 if dv >= lv else -0.0036
        ax.annotate(f'{dv:.3f}', xy=(xpos[i], dv),
                    xytext=(xpos[i], dv + offset),
                    ha='center', va='bottom' if offset > 0 else 'top',
                    color=LABEL_FG, fontsize=10, fontname=INTER,
                    fontweight='bold', zorder=6)

        # League point label
        loffset = -0.0028 if dv >= lv else 0.0028
        ax.annotate(f'{lv:.3f}', xy=(xpos[i], lv),
                    xytext=(xpos[i], lv + loffset),
                    ha='center', va='top' if loffset < 0 else 'bottom',
                    color=LEAGUE, fontsize=9, fontname=INTER,
                    fontweight='bold', zorder=6)

        # Differential badge
        badge_y = max(dv, lv) + 0.010
        ax.text(xpos[i], badge_y, text,
                ha='center', va='bottom', color=col,
                fontsize=11, fontname=INTER, fontweight='bold', zorder=6)

    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0.860, 0.958)
    ax.set_xticks(xpos)
    ax.set_xticklabels(seasons, color=LABEL_FG, fontsize=11,
                       fontname=INTER, fontweight='bold')
    ax.set_ylabel('ES Save Percentage', color=GREY, fontsize=10,
                  fontname=INTER, labelpad=8)
    ax.tick_params(axis='y', colors=GREY, labelsize=9)
    ax.tick_params(axis='x', bottom=False)
    spine_style(ax, keep_bottom=False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color(GREY)
    ax.spines['left'].set_linewidth(0.6)

    ax.legend(loc='upper right', frameon=False,
              labelcolor=LABEL_FG, fontsize=9.5,
              prop={'family': INTER})

    fig.text(0.06, 0.93,
             'DOSTAL MEDIUM-DANGER SV% — THE EDGE THAT DISAPPEARED',
             ha='left', va='top', color=LABEL_FG, fontsize=17,
             fontname=BEBAS, fontweight='bold')
    fig.text(0.06, 0.87,
             'Even Strength | Medium-Danger Zone (20–40 ft from net)',
             ha='left', va='top', color=GREY, fontsize=9.5, fontname=INTER)

    add_footer(fig)
    plt.tight_layout(rect=[0.07, 0.04, 1.0, 0.88])

    path = os.path.join(OUT_DIR, 'dostal_medium_danger_trend.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Chart 2 saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Scouting Card
# ═══════════════════════════════════════════════════════════════════════════════
def chart3():
    fig = plt.figure(figsize=(1200/150, 900/150), dpi=150)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG)
    ax.axis('off')

    # ── outer card ─────────────────────────────────────────────────────────────
    card = FancyBboxPatch((0.03, 0.07), 0.94, 0.87,
                          boxstyle="round,pad=0.012",
                          facecolor=CARD_BG, edgecolor='#2E7DC4',
                          linewidth=1.5, zorder=1)
    ax.add_patch(card)

    # ── title block ────────────────────────────────────────────────────────────
    ax.text(0.5, 0.905, 'HOW TO BEAT DOSTAL — ROUND 1 SCOUTING REPORT',
            ha='center', va='center', color=LABEL_FG, fontsize=16,
            fontname=BEBAS, fontweight='bold', zorder=5)
    ax.text(0.5, 0.872, '2025-26 Even Strength  |  Backhands Excluded',
            ha='center', va='center', color=GREY, fontsize=9,
            fontname=INTER, zorder=5)

    # Divider line under title
    ax.plot([0.06, 0.94], [0.858, 0.858], color='#2E7DC4',
            linewidth=0.8, alpha=0.55, zorder=4)

    # ── column headers ─────────────────────────────────────────────────────────
    #  ATTACK
    ax.text(0.26, 0.835, 'ATTACK',
            ha='center', va='center', color=GREEN, fontsize=15,
            fontname=BEBAS, fontweight='bold', zorder=5)
    ax.plot([0.07, 0.46], [0.822, 0.822], color=GREEN,
            linewidth=1.8, alpha=0.70, zorder=4)

    #  AVOID
    ax.text(0.74, 0.835, 'AVOID',
            ha='center', va='center', color=RED, fontsize=15,
            fontname=BEBAS, fontweight='bold', zorder=5)
    ax.plot([0.54, 0.93], [0.822, 0.822], color=RED,
            linewidth=1.8, alpha=0.70, zorder=4)

    # Center divider
    ax.plot([0.50, 0.50], [0.82, 0.12], color='#2E7DC4',
            linewidth=0.6, alpha=0.35, zorder=4)

    # ── ATTACK items ────────────────────────────────────────────────────────────
    attack_items = [
        ('Snap shots from the circles',
         'Med-danger snaps: .849 vs LGE .879 (−3.0pp)'),
        ('Tip-ins with net-front traffic',
         'HD tip-ins: .830 vs LGE .865 (−3.5pp)'),
        ('Point shots through screens',
         'Slap shots: .908 vs LGE .930 (−2.3pp)'),
        ('PP through the middle',
         'PK SV%: .820 vs LGE .860 (−4.0pp)'),
    ]

    y_start = 0.785
    y_step  = 0.155
    for i, (headline, stat) in enumerate(attack_items):
        y = y_start - i * y_step

        # Checkmark bullet
        ax.text(0.075, y + 0.008, '✓', ha='left', va='center',
                color=GREEN, fontsize=13, fontname=INTER,
                fontweight='bold', zorder=5)

        ax.text(0.105, y + 0.010, headline,
                ha='left', va='center', color=LABEL_FG,
                fontsize=10.5, fontname=INTER, fontweight='bold', zorder=5)

        ax.text(0.105, y - 0.016, stat,
                ha='left', va='center', color=GREEN,
                fontsize=8.5, fontname=INTER, alpha=0.85, zorder=5)

    # ── AVOID items ─────────────────────────────────────────────────────────────
    avoid_items = [
        ('Rush chances',
         'Rush SV%: .966 vs LGE .892 (+7.4pp)'),
        ('Clean wristers from outside',
         'Wrist SV%: .922 vs LGE .919 (+0.3pp)'),
        ('Forcing plays to the crease',
         'High-danger SV%: .836 vs LGE .826 (+1.0pp)'),
    ]

    y_start_avoid = 0.755
    y_step_avoid  = 0.175
    for i, (headline, stat) in enumerate(avoid_items):
        y = y_start_avoid - i * y_step_avoid

        ax.text(0.555, y + 0.008, '✗', ha='left', va='center',
                color=RED, fontsize=13, fontname=INTER,
                fontweight='bold', zorder=5)

        ax.text(0.585, y + 0.010, headline,
                ha='left', va='center', color=LABEL_FG,
                fontsize=10.5, fontname=INTER, fontweight='bold', zorder=5)

        ax.text(0.585, y - 0.016, stat,
                ha='left', va='center', color=RED,
                fontsize=8.5, fontname=INTER, alpha=0.85, zorder=5)

    # ── Key stat callout boxes ──────────────────────────────────────────────────
    # Left callout: worst shot type
    lbox = FancyBboxPatch((0.07, 0.115), 0.385, 0.075,
                          boxstyle="round,pad=0.008",
                          facecolor='#0B2E1A', edgecolor=GREEN,
                          linewidth=1.0, zorder=4)
    ax.add_patch(lbox)
    ax.text(0.263, 0.163, 'BIGGEST WEAKNESS',
            ha='center', va='center', color=GREEN,
            fontsize=8, fontname=BEBAS, zorder=5)
    ax.text(0.263, 0.143, 'Snap shots  .885  (−1.1pp vs LGE)',
            ha='center', va='center', color=LABEL_FG,
            fontsize=9.5, fontname=INTER, fontweight='bold', zorder=5)

    # Right callout: biggest strength
    rbox = FancyBboxPatch((0.545, 0.115), 0.385, 0.075,
                          boxstyle="round,pad=0.008",
                          facecolor='#2E0B0B', edgecolor=RED,
                          linewidth=1.0, zorder=4)
    ax.add_patch(rbox)
    ax.text(0.737, 0.163, 'AVOID AT ALL COSTS',
            ha='center', va='center', color=RED,
            fontsize=8, fontname=BEBAS, zorder=5)
    ax.text(0.737, 0.143, 'Rush shots  .966  (+7.4pp vs LGE)',
            ha='center', va='center', color=LABEL_FG,
            fontsize=9.5, fontname=INTER, fontweight='bold', zorder=5)

    # ── Footer ─────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.045,
            '@HockeyROI | hockeyROI.substack.com  |  Round 1: EDM vs ANA — Game 1 Monday April 20',
            ha='center', va='center', color=GREY,
            fontsize=8.5, fontname=INTER, style='italic', zorder=5)

    path = os.path.join(OUT_DIR, 'dostal_scouting_card.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Chart 3 saved: {path}")


# ─── RUN ALL ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    chart1()
    chart2()
    chart3()
    print("\nAll 3 charts complete.")
