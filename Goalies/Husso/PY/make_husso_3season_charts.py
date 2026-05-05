#!/usr/bin/env python3
"""
HockeyROI — Ville Husso 3-season chart package
  Chart 1: Shot type SV% vs League (3-season combined ES horizontal bars)
  Chart 2: Scouting card — 3-season summary
Exact style match to make_husso_charts.py / make_dostal_charts.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
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

# ─── BRAND PALETTE ─────────────────────────────────────────────────────────────
BG       = '#0B1D2E'
CARD_BG  = '#1B3A5C'
PLAYER   = '#2E7DC4'
LEAGUE   = '#FFA940'
LABEL_FG = '#F0F4F8'
GREY     = '#888888'
GREEN    = '#44AA66'
RED      = '#CC3333'
FOOTER   = '@HockeyROI | hockeyROI.substack.com'

DUCKS_DIR = "/Users/ashgarg/Documents/HockeyROI/Goalies/Husso/Images"
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
    os.makedirs(DUCKS_DIR, exist_ok=True)
    ducks_path   = os.path.join(DUCKS_DIR, filename)
    desktop_path = os.path.join(DESKTOP, filename)
    fig.savefig(ducks_path,   dpi=150, bbox_inches='tight', facecolor=BG)
    fig.savefig(desktop_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"  Saved → {ducks_path}")
    print(f"  Saved → {desktop_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Shot Type SV% Grouped Horizontal Bars (3-season)
# ═══════════════════════════════════════════════════════════════════════════════
def chart1():
    print("Building Chart 1 — Shot Type Bars (3-season)...")

    # 3-season combined ES data (backhands excluded)
    # Wrist n=509  Husso .919  League .924  diff −0.005
    # Snap  n=276  Husso .891  League .893  diff −0.002
    # Slap  n=94   Husso .894  League .938  diff −0.044
    # Tip-In n=92  Husso .870  League .873  diff −0.003
    # Deflected n=12 Husso 1.000 League .860 diff +0.140
    shot_types = ['Wrist',  'Snap',  'Slap',  'Tip-In', 'Deflected']
    ns         = [509,       276,     94,       92,        12]
    husso      = [.919,      .891,    .894,     .870,      1.000]
    league     = [.924,      .893,    .938,     .873,       .860]
    diffs      = [h - l for h, l in zip(husso, league)]

    # Sort worst diff first (same ordering convention as Dostal chart)
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

    ax.barh(y + height/2 + gap/2, husso,  height,
            color=PLAYER, alpha=0.92, label='Husso',      zorder=3)
    ax.barh(y - height/2 - gap/2, league, height,
            color=LEAGUE, alpha=0.90, label='League Avg', zorder=3)

    x_max = 1.015
    for v in np.arange(0.80, 1.02, 0.02):
        ax.axvline(v, color='#FFFFFF', alpha=0.05, linewidth=0.6, zorder=1)

    ax.set_xlim(0.820, x_max)
    ax.set_ylim(-0.65, n - 0.35)

    ax.set_yticks(y)
    ax.set_yticklabels([])
    for i, (st, n_val) in enumerate(zip(shot_types, ns)):
        ax.text(-0.001, i, f'{st}  ', ha='right', va='center',
                color=LABEL_FG, fontsize=11, fontname=INTER,
                fontweight='bold', transform=ax.get_yaxis_transform())
        ax.text(-0.001, i - 0.22, f'n={n_val}', ha='right', va='center',
                color=GREY, fontsize=8.5, fontname=INTER,
                transform=ax.get_yaxis_transform())

    for i, (hv, lv, diff) in enumerate(zip(husso, league, diffs)):
        label = f'{hv:.3f}' if hv < 1.0 else '1.000 (n=12)'
        bar_end = min(hv, x_max - 0.002)
        ax.text(bar_end + 0.0008, i + height/2 + gap/2, label,
                va='center', color=LABEL_FG, fontsize=9, fontname=INTER,
                fontweight='bold', zorder=5)
        ax.text(lv + 0.0008, i - height/2 - gap/2, f'{lv:.3f}',
                va='center', color=LABEL_FG, fontsize=8.5, fontname=INTER,
                alpha=0.75, zorder=5)
        col  = pp_color(diff)
        text = pp_str(diff)
        ax.text(0.987, i, text, ha='right', va='center',
                color=col, fontsize=10.5, fontname=INTER,
                fontweight='bold', zorder=5,
                transform=ax.get_yaxis_transform())

    ax.tick_params(axis='x', colors=GREY, labelsize=8)
    ax.set_xlabel('Save Percentage', color=GREY, fontsize=9,
                  fontname=INTER, labelpad=6)
    spine_style(ax, keep_bottom=True)
    ax.tick_params(axis='y', left=False)

    fig.text(0.06, 0.95, 'HUSSO vs LEAGUE — SHOT TYPE SV% (3-SEASON ES)',
             ha='left', va='top', color=LABEL_FG, fontsize=18,
             fontname=BEBAS, fontweight='bold')
    fig.text(0.06, 0.89,
             'Even Strength  |  Backhands Excluded  |  2023-24 to 2025-26  |  47 Starts',
             ha='left', va='top', color=GREY, fontsize=9.5, fontname=INTER)

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

    save_both(fig, 'husso_shot_type_chart_3season.png')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Scouting Card (3-season)
# ═══════════════════════════════════════════════════════════════════════════════
def chart2():
    print("Building Chart 2 — Scouting Card (3-season)...")

    fig = plt.figure(figsize=(1200/150, 900/150), dpi=150)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG)
    ax.axis('off')

    # Outer card
    card = FancyBboxPatch((0.03, 0.07), 0.94, 0.87,
                          boxstyle="round,pad=0.012",
                          facecolor=CARD_BG, edgecolor=PLAYER,
                          linewidth=1.5, zorder=1)
    ax.add_patch(card)

    # Title block
    ax.text(0.5, 0.905, 'HOW TO BEAT HUSSO — 3-SEASON SCOUTING REPORT',
            ha='center', va='center', color=LABEL_FG, fontsize=16,
            fontname=BEBAS, fontweight='bold', zorder=5)
    ax.text(0.5, 0.872, '2023-24 to 2025-26  |  Even Strength  |  Backhands Excluded  |  47 Games',
            ha='center', va='center', color=GREY, fontsize=9,
            fontname=INTER, zorder=5)

    ax.plot([0.06, 0.94], [0.858, 0.858], color=PLAYER,
            linewidth=0.8, alpha=0.55, zorder=4)

    # Column headers
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

    ax.plot([0.50, 0.50], [0.82, 0.20], color=PLAYER,
            linewidth=0.6, alpha=0.35, zorder=4)

    # ATTACK items — backed by n=94+ for slap, n=305 HD, n=247 right
    attack_items = [
        ('Slap shots from the point',
         'Slap SV%: .894 vs LGE .938 (−4.4pp, n=94)'),
        ('Rush chances',
         'Rush SV%: .853 vs LGE .901 (−4.8pp, n=34)'),
        ('High-danger zone shots',
         'High-danger: .800 vs LGE .832 (−3.2pp, n=305)'),
        ('Right lateral entries',
         'Right lateral: .943 vs LGE .964 (−2.1pp, n=247)'),
    ]

    y_start = 0.785
    y_step  = 0.148
    for i, (headline, stat) in enumerate(attack_items):
        y = y_start - i * y_step
        ax.plot(0.082, y + 0.010, 'o', color=GREEN, markersize=7, zorder=5)
        ax.text(0.105, y + 0.011, headline,
                ha='left', va='center', color=LABEL_FG,
                fontsize=10.5, fontname=INTER, fontweight='bold', zorder=5)
        ax.text(0.105, y - 0.015, stat,
                ha='left', va='center', color=GREEN,
                fontsize=8.5, fontname=INTER, alpha=0.90, zorder=5)

    # AVOID items
    avoid_items = [
        ('Left side shots',
         'Left lateral: .973 vs LGE .962 (+1.1pp, n=292)'),
        ('Rebounds (handles them well)',
         'Rebound SV%: .842 vs LGE .841 (+0.1pp, n=57)'),
        ('Deflections / wrap-arounds',
         '1.000 SV% — but n=12/5, no conclusions'),
    ]

    y_start_avoid = 0.755
    y_step_avoid  = 0.175
    for i, (headline, stat) in enumerate(avoid_items):
        y = y_start_avoid - i * y_step_avoid
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

    # Callout boxes
    lbox = FancyBboxPatch((0.07, 0.185), 0.385, 0.072,
                          boxstyle="round,pad=0.008",
                          facecolor='#0B2E1A', edgecolor=GREEN,
                          linewidth=1.0, zorder=4)
    ax.add_patch(lbox)
    ax.text(0.263, 0.232, 'BIGGEST WEAKNESS',
            ha='center', va='center', color=GREEN,
            fontsize=8, fontname=BEBAS, zorder=5)
    ax.text(0.263, 0.212, 'Slap shots  .894  (−4.4pp vs LGE, n=94)',
            ha='center', va='center', color=LABEL_FG,
            fontsize=9.5, fontname=INTER, fontweight='bold', zorder=5)

    rbox = FancyBboxPatch((0.545, 0.185), 0.385, 0.072,
                          boxstyle="round,pad=0.008",
                          facecolor='#2E0B0B', edgecolor=RED,
                          linewidth=1.0, zorder=4)
    ax.add_patch(rbox)
    ax.text(0.737, 0.232, 'AVOID AT ALL COSTS',
            ha='center', va='center', color=RED,
            fontsize=8, fontname=BEBAS, zorder=5)
    ax.text(0.737, 0.212, 'Wrap-arounds / deflections — sample too small',
            ha='center', va='center', color=LABEL_FG,
            fontsize=8.5, fontname=INTER, fontweight='bold', zorder=5)

    # Disclaimer
    ax.text(0.5, 0.138,
            '* Rush (n=34), deflected (n=12), wrap-around (n=5) — treat with caution. Slap (n=94) is the most reliable finding.',
            ha='center', va='center', color=GREY,
            fontsize=7.5, fontname=INTER, style='italic', zorder=5)

    # Footer
    ax.text(0.5, 0.095,
            '@HockeyROI | hockeyROI.substack.com  |  Round 1: EDM vs ANA',
            ha='center', va='center', color=GREY,
            fontsize=8.5, fontname=INTER, style='italic', zorder=5)

    save_both(fig, 'husso_scouting_card_3season.png')
    plt.close()


# ─── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    chart1()
    chart2()
    print("\nBoth 3-season Husso charts complete.")
