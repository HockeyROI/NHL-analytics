#!/usr/bin/env python3
"""
HockeyROI — Ville Husso chart package
  Chart 1: Shot type SV% vs League (2025-26 ES horizontal bars)
  Chart 2: Scouting card summary
Exact style match to make_dostal_charts.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
import shutil

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

# ─── BRAND PALETTE ─────────────────────────────────────────────────────────────
BG       = '#0B1D2E'
CARD_BG  = '#1B3A5C'
PLAYER   = '#2E7DC4'   # Husso blue (same brand blue)
LEAGUE   = '#FFA940'   # orange — matches fixed Dostal charts
LABEL_FG = '#F0F4F8'
GREY     = '#888888'
GREEN    = '#44AA66'
RED      = '#CC3333'
FOOTER   = '@HockeyROI | hockeyROI.substack.com'

DUCKS_DIR = ("/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/"
             "NHL analysis/Goalies/Ducks")
DESKTOP   = os.path.expanduser("~/Desktop")

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

def save_both(fig, filename):
    """Save to Ducks folder and Desktop."""
    ducks_path   = os.path.join(DUCKS_DIR, filename)
    desktop_path = os.path.join(DESKTOP, filename)
    fig.savefig(ducks_path,   dpi=150, bbox_inches='tight', facecolor=BG)
    fig.savefig(desktop_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"  Saved → {ducks_path}")
    print(f"  Saved → {desktop_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Shot Type SV% Grouped Horizontal Bars
# ═══════════════════════════════════════════════════════════════════════════════
def chart1():
    print("Building Chart 1 — Shot Type Bars...")

    # Data ordered worst → best differential (Dostal chart is sorted worst → best)
    shot_types = ['Wrist', 'Snap', 'Tip-In', 'Slap', 'Deflected']
    ns         = [126,    173,    35,       32,     5]
    husso      = [.881,   .884,   .886,     .938,   1.000]
    league     = [.919,   .896,   .881,     .930,   .849]
    diffs      = [h - l for h, l in zip(husso, league)]

    # Sort by diff ascending (worst first, like Dostal chart)
    order = sorted(range(len(diffs)), key=lambda i: diffs[i])
    shot_types = [shot_types[i] for i in order]
    ns         = [ns[i]         for i in order]
    husso      = [husso[i]      for i in order]
    league     = [league[i]     for i in order]
    diffs      = [diffs[i]      for i in order]

    fig, ax = plt.subplots(figsize=(1200/150, 700/150), dpi=150)
    set_dark_bg(fig, ax)

    n      = len(shot_types)
    y      = np.arange(n)
    height = 0.30
    gap    = 0.06

    # Husso bars (top of each pair)
    ax.barh(y + height/2 + gap/2, husso, height,
            color=PLAYER, alpha=0.92, label='Husso', zorder=3)
    # League bars (bottom)
    ax.barh(y - height/2 - gap/2, league, height,
            color=LEAGUE, alpha=0.90, label='League Avg', zorder=3)

    # Gridlines
    x_max = 1.015   # extend to fit 1.000 Deflected bar cleanly
    for v in np.arange(0.80, 1.02, 0.02):
        ax.axvline(v, color='#FFFFFF', alpha=0.05, linewidth=0.6, zorder=1)

    ax.set_xlim(0.820, x_max)
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

    # Value labels on bars + differential badges
    for i, (hv, lv, diff) in enumerate(zip(husso, league, diffs)):
        # Husso value — small caveat label for 1.000 (n=5)
        label = f'{hv:.3f}' if hv < 1.0 else '1.000 (n=5)'
        bar_end = min(hv, x_max - 0.002)
        ax.text(bar_end + 0.0008, i + height/2 + gap/2, label,
                va='center', color=LABEL_FG, fontsize=9, fontname=INTER,
                fontweight='bold', zorder=5)
        # League value
        ax.text(lv + 0.0008, i - height/2 - gap/2, f'{lv:.3f}',
                va='center', color=LABEL_FG, fontsize=8.5, fontname=INTER,
                alpha=0.75, zorder=5)
        # Differential badge — right side
        col  = pp_color(diff)
        text = pp_str(diff)
        ax.text(0.987, i, text, ha='right', va='center',
                color=col, fontsize=10.5, fontname=INTER,
                fontweight='bold', zorder=5,
                transform=ax.get_yaxis_transform())

    # X axis
    ax.tick_params(axis='x', colors=GREY, labelsize=8)
    ax.set_xlabel('Save Percentage', color=GREY, fontsize=9,
                  fontname=INTER, labelpad=6)
    spine_style(ax, keep_bottom=True)
    ax.tick_params(axis='y', left=False)

    # Title block
    fig.text(0.06, 0.95, 'HUSSO vs LEAGUE — SHOT TYPE SV% (2025-26 ES)',
             ha='left', va='top', color=LABEL_FG, fontsize=18,
             fontname=BEBAS, fontweight='bold')
    fig.text(0.06, 0.89, 'Even Strength  |  Backhands Excluded  |  2025-26 Regular Season  |  20 Starts',
             ha='left', va='top', color=GREY, fontsize=9.5, fontname=INTER)

    # Legend — below chart, centred (same position as Dostal)
    fig.legend(
        handles=[
            mpatches.Patch(facecolor=PLAYER, alpha=0.92, label='Husso'),
            mpatches.Patch(facecolor=LEAGUE, alpha=0.90, label='League Avg'),
        ],
        loc='lower center', ncol=2, frameon=False,
        labelcolor=LABEL_FG, fontsize=10,
        prop={'family': INTER},
        bbox_to_anchor=(0.55, 0.01)
    )

    add_footer(fig, y=0.072)
    plt.tight_layout(rect=[0.13, 0.10, 1.0, 0.88])

    save_both(fig, 'husso_shot_type_chart.png')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Scouting Card
# ═══════════════════════════════════════════════════════════════════════════════
def chart2():
    print("Building Chart 2 — Scouting Card...")

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
                          facecolor=CARD_BG, edgecolor=PLAYER,
                          linewidth=1.5, zorder=1)
    ax.add_patch(card)

    # ── title block ────────────────────────────────────────────────────────────
    ax.text(0.5, 0.905, 'HOW TO BEAT HUSSO — ROUND 1 SCOUTING REPORT',
            ha='center', va='center', color=LABEL_FG, fontsize=16,
            fontname=BEBAS, fontweight='bold', zorder=5)
    ax.text(0.5, 0.872, '2025-26 Even Strength  |  Backhands Excluded  |  20 Games',
            ha='center', va='center', color=GREY, fontsize=9,
            fontname=INTER, zorder=5)

    # Divider under title
    ax.plot([0.06, 0.94], [0.858, 0.858], color=PLAYER,
            linewidth=0.8, alpha=0.55, zorder=4)

    # ── column headers ─────────────────────────────────────────────────────────
    ax.text(0.26, 0.835, 'ATTACK',
            ha='center', va='center', color=GREEN, fontsize=15,
            fontname=BEBAS, fontweight='bold', zorder=5)
    ax.plot([0.07, 0.46], [0.822, 0.822], color=GREEN,
            linewidth=1.8, alpha=0.70, zorder=4)

    ax.text(0.74, 0.835, 'AVOID',
            ha='center', va='center', color=RED, fontsize=15,
            fontname=BEBAS, fontweight='bold', zorder=5)
    ax.plot([0.54, 0.93], [0.822, 0.822], color=RED,
            linewidth=1.8, alpha=0.70, zorder=4)

    # Center divider
    ax.plot([0.50, 0.50], [0.82, 0.20], color=PLAYER,
            linewidth=0.6, alpha=0.35, zorder=4)

    # ── ATTACK items ────────────────────────────────────────────────────────────
    attack_items = [
        ('Wrist shots from in tight',
         'HD wristers: .697 vs LGE .820 (−12.3pp)'),
        ('Snap shots from the slot',
         'Med-danger snaps: .850 vs LGE .879 (−2.9pp)'),
        ('Right lateral shots',
         'Right lateral: .878 vs LGE .959 (−8.0pp)'),
        ('PP through the middle',
         'PK SV%: .841 vs LGE .860 (−1.9pp)'),
    ]

    y_start = 0.785
    y_step  = 0.148
    for i, (headline, stat) in enumerate(attack_items):
        y = y_start - i * y_step
        # Green filled circle bullet
        ax.plot(0.082, y + 0.010, 'o', color=GREEN,
                markersize=7, zorder=5)
        ax.text(0.105, y + 0.011, headline,
                ha='left', va='center', color=LABEL_FG,
                fontsize=10.5, fontname=INTER, fontweight='bold', zorder=5)
        ax.text(0.105, y - 0.015, stat,
                ha='left', va='center', color=GREEN,
                fontsize=8.5, fontname=INTER, alpha=0.90, zorder=5)

    # ── AVOID items ─────────────────────────────────────────────────────────────
    avoid_items = [
        ('Slap shots',
         'Slap SV%: .938 vs LGE .930 (+0.7pp)'),
        ('Low-danger perimeter shots',
         'Low-danger SV%: .978 vs LGE .975 (+0.4pp)'),
        ('Tip-ins (he handles these)',
         'Tip-in SV%: .886 vs LGE .881 (+0.5pp)'),
    ]

    y_start_avoid = 0.755
    y_step_avoid  = 0.175
    for i, (headline, stat) in enumerate(avoid_items):
        y = y_start_avoid - i * y_step_avoid
        # Red X drawn as two diagonal lines
        cx, cy, r = 0.563, y + 0.010, 0.010
        ax.plot([cx - r, cx + r], [cy - r, cy + r], color=RED,
                linewidth=2.0, solid_capstyle='round', zorder=5)
        ax.plot([cx - r, cx + r], [cy + r, cy - r], color=RED,
                linewidth=2.0, solid_capstyle='round', zorder=5)
        ax.text(0.585, y + 0.011, headline,
                ha='left', va='center', color=LABEL_FG,
                fontsize=10.5, fontname=INTER, fontweight='bold', zorder=5)
        ax.text(0.585, y - 0.015, stat,
                ha='left', va='center', color=RED,
                fontsize=8.5, fontname=INTER, alpha=0.90, zorder=5)

    # ── Key callout boxes ───────────────────────────────────────────────────────
    # Left — BIGGEST WEAKNESS
    lbox = FancyBboxPatch((0.07, 0.185), 0.385, 0.072,
                          boxstyle="round,pad=0.008",
                          facecolor='#0B2E1A', edgecolor=GREEN,
                          linewidth=1.0, zorder=4)
    ax.add_patch(lbox)
    ax.text(0.263, 0.232, 'BIGGEST WEAKNESS',
            ha='center', va='center', color=GREEN,
            fontsize=8, fontname=BEBAS, zorder=5)
    ax.text(0.263, 0.212, 'Wrist shots  .881  (−3.8pp vs LGE)',
            ha='center', va='center', color=LABEL_FG,
            fontsize=9.5, fontname=INTER, fontweight='bold', zorder=5)

    # Right — AVOID AT ALL COSTS
    rbox = FancyBboxPatch((0.545, 0.185), 0.385, 0.072,
                          boxstyle="round,pad=0.008",
                          facecolor='#2E0B0B', edgecolor=RED,
                          linewidth=1.0, zorder=4)
    ax.add_patch(rbox)
    ax.text(0.737, 0.232, 'AVOID AT ALL COSTS',
            ha='center', va='center', color=RED,
            fontsize=8, fontname=BEBAS, zorder=5)
    ax.text(0.737, 0.212, 'Rush chances: n=11, too small to draw conclusions',
            ha='center', va='center', color=LABEL_FG,
            fontsize=8.5, fontname=INTER, fontweight='bold', zorder=5)

    # ── Small disclaimer note ───────────────────────────────────────────────────
    ax.text(0.5, 0.138,
            '* Rush chances excluded from analysis — sample size (n=11) too small for reliable conclusions',
            ha='center', va='center', color=GREY,
            fontsize=7.5, fontname=INTER, style='italic', zorder=5)

    # ── Footer ─────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.095,
            '@HockeyROI | hockeyROI.substack.com  |  Round 1: EDM vs ANA',
            ha='center', va='center', color=GREY,
            fontsize=8.5, fontname=INTER, style='italic', zorder=5)

    save_both(fig, 'husso_scouting_card.png')
    plt.close()


# ─── RUN ALL ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(DUCKS_DIR, exist_ok=True)
    chart1()
    chart2()
    print("\nBoth Husso charts complete.")
