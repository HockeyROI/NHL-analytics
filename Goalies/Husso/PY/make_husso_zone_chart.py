#!/usr/bin/env python3
"""
HockeyROI — Husso zone bubble chart  (v2 — matches reference layout)
Even strength · 3-season combined
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Arc, FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import os

# ── Fonts ──────────────────────────────────────────────────────────────────────
def load_font(path, fallback="Arial"):
    if os.path.exists(path):
        fm.fontManager.addfont(path)
        return fm.FontProperties(fname=path).get_name()
    return fallback

BEBAS = load_font("/tmp/BebasNeue-Regular.ttf", "Arial Black")
INTER = load_font("/tmp/Inter-Regular.ttf",      "Arial")

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = '#FFFFFF'          # card background — white
ICE      = '#D9EEF8'          # rink surface — very soft pastel blue (matches reference)
ICE_BORD = '#88B8D8'          # rink border — soft mid-blue
GOAL_RED = '#D4706A'          # goal line — muted salmon-red
LABEL_FG = '#1A1A1A'          # near-black text on white
DARK_TXT = '#1A1A1A'
GREY     = '#666666'
GREEN    = '#2E8B57'

# Bubble colours — muted/pastel to match reference exactly
C_HIGH   = '#D4706A'          # lighter rose-red   (20%+  danger)
C_MED    = '#E8BE55'          # lighter amber-gold  (10-20% vulnerable)
C_LOW    = '#7BAED4'          # steel blue          (<10%  managed)

# Text inside bubbles — dark (reference uses dark text, not white)
BUBBLE_TXT = '#1A1A1A'

DESKTOP = os.path.expanduser("~/Desktop")
OUT_DIR = "/Users/ashgarg/Documents/HockeyROI/Goalies/Husso/Images"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & zone-classify data ──────────────────────────────────────────────────
husso = pd.read_csv("/Users/ashgarg/Documents/HockeyROI/Goalies/Husso/husso_3seasons.csv")
BENCH = "/Users/ashgarg/Documents/HockeyROI/Goalies/Benchmarks Goalies/Data"
lg    = pd.read_csv(f"{BENCH}/all_goalie_shots_3seasons.csv",
                    usecols=['situation','is_goal','distance_ft','lateral','danger_zone'])

def assign_zone(d, lat, dz):
    try:    d = float(d)
    except: return 'Unknown'
    if str(lat) in ('nan','None'): return 'Unknown'
    if d > 50:
        return 'L point' if lat == 'Left' else 'R point'
    if lat == 'Left':
        return 'L wall' if dz == 'Low' else 'L circle'
    if lat == 'Right':
        return 'R wall' if dz == 'Low' else 'R circle'
    return 'crease' if d <= 15 else 'slot'

es    = husso[husso['situation'] == 'Even Strength'].copy()
lg_es = lg[lg['situation']       == 'Even Strength'].copy()

es['zone']    = es.apply(   lambda r: assign_zone(r['distance_ft'], r['lateral'], r['danger_zone']), axis=1)
lg_es['zone'] = lg_es.apply(lambda r: assign_zone(r['distance_ft'], r['lateral'], r['danger_zone']), axis=1)

def zone_stats(df):
    g = df[df['zone'] != 'Unknown'].groupby('zone').agg(
        n=('is_goal','count'), goals=('is_goal','sum')).reset_index()
    g['sv_pct']   = 1 - g['goals'] / g['n']
    g['goal_pct'] = g['goals'] / g['goals'].sum() * 100
    return g.set_index('zone')

h  = zone_stats(es)
lz = zone_stats(lg_es)
h['league_sv'] = lz['sv_pct']
h['diff_pp']   = (h['sv_pct'] - h['league_sv']) * 100

# ── Zone positions on rink canvas (x: -44→44, y: 0=goal line, 64=blue line) ──
ZONE_POS = {
    'L point':  (-18, 56),
    'R point':  ( 18, 56),
    'L wall':   (-36, 37),
    'R wall':   ( 36, 37),
    'L circle': (-22, 23),
    'R circle': ( 22, 23),
    'slot':     (  0, 21),
    'crease':   (  0,  7),
}

def bubble_color(gp):
    if gp >= 20:  return C_HIGH
    if gp >= 10:  return C_MED
    return C_LOW

def bubble_radius(n, n_min, n_max):
    return 3.5 + 5.5 * (np.sqrt(n) - np.sqrt(n_min)) / max(np.sqrt(n_max) - np.sqrt(n_min), 1)

n_vals = [h.loc[z,'n'] for z in ZONE_POS if z in h.index]
N_MIN, N_MAX = min(n_vals), max(n_vals)

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9.0, 10.2), dpi=150)
fig.patch.set_facecolor(BG)

# ── Title ──────────────────────────────────────────────────────────────────────
ax_t = fig.add_axes([0.0, 0.91, 1.0, 0.09])
ax_t.set_facecolor(BG); ax_t.axis('off')
ax_t.text(0.5, 0.70, 'Husso PK/EV — where ANA gets scored on',
          ha='center', va='center', color=LABEL_FG,
          fontsize=17, fontname=BEBAS, fontweight='bold')
ax_t.text(0.5, 0.18, '2023-24 to 2025-26  ·  Even Strength  ·  47 Starts  ·  105 Goals Against',
          ha='center', va='center', color=GREY, fontsize=9, fontname=INTER)

# ── Rink axes ─────────────────────────────────────────────────────────────────
ax = fig.add_axes([0.06, 0.31, 0.88, 0.60])
ax.set_xlim(-48, 48)
ax.set_ylim(-7,  68)
ax.set_aspect('equal')
ax.axis('off')

# Ice surface
ice = FancyBboxPatch((-44, -4), 88, 70,
                     boxstyle="round,pad=2.5",
                     facecolor=ICE, edgecolor=ICE_BORD,
                     linewidth=2.2, zorder=1)
ax.add_patch(ice)

# Blue line
ax.plot([-44, 44], [64, 64], color=ICE_BORD, linewidth=3.5, alpha=0.9, zorder=2)
ax.text(-45.5, 64, 'blue line', color=ICE_BORD, fontsize=7.5,
        fontname=INTER, va='center', alpha=0.85)

# Goal line
ax.plot([-20, 20], [0, 0], color=GOAL_RED, linewidth=1.8,
        linestyle='--', alpha=0.75, zorder=2)
ax.text(-21.5, 0, 'goal line', color=GOAL_RED, fontsize=7.5,
        fontname=INTER, va='center', alpha=0.75)

# Net
net = FancyBboxPatch((-3.0, -4.2), 6.0, 4.2,
                     boxstyle="round,pad=0.3",
                     facecolor='#C8DFF0', edgecolor='#88B8D8',
                     linewidth=1.0, zorder=3)
ax.add_patch(net)

# Crease arc
ax.add_patch(Arc((0,0), 12, 9, angle=0, theta1=0, theta2=180,
                 color='#88B8D8', linewidth=1.0, alpha=0.6, zorder=2))

# Faceoff circles
for cx in [-22, 22]:
    ax.add_patch(plt.Circle((cx, 22), 15, fill=False,
                             edgecolor='#88B8D8', linewidth=0.9, alpha=0.40, zorder=2))
    ax.add_patch(plt.Circle((cx, 22), 0.8, color='#88B8D8', alpha=0.55, zorder=2))

# ── Bubbles ────────────────────────────────────────────────────────────────────
for zone, (bx, by) in ZONE_POS.items():
    if zone not in h.index: continue
    row  = h.loc[zone]
    n    = int(row['n'])
    gp   = row['goal_pct']
    diff = row['diff_pp']
    col  = bubble_color(gp)
    r    = bubble_radius(n, N_MIN, N_MAX)

    # Shadow for depth
    ax.add_patch(plt.Circle((bx+0.5, by-0.5), r, color='#000000',
                             alpha=0.18, zorder=4))
    # Main bubble
    ax.add_patch(plt.Circle((bx, by), r, color=col, alpha=0.90, zorder=5))
    # Thin ring (slightly lighter than bubble)
    ax.add_patch(plt.Circle((bx, by), r, fill=False,
                             edgecolor='white', linewidth=0.8, alpha=0.45, zorder=6))

    # Percentage (bold, large)
    ax.text(bx, by + r*0.18, f'{gp:.1f}%',
            ha='center', va='center', color=BUBBLE_TXT,
            fontsize=13, fontname=INTER, fontweight='bold', zorder=7)
    # Zone label (smaller, below %)
    ax.text(bx, by - r*0.30, zone,
            ha='center', va='center', color=BUBBLE_TXT,
            fontsize=10, fontname=INTER, alpha=0.85, zorder=7)

# ── Legend row ─────────────────────────────────────────────────────────────────
ax_leg = fig.add_axes([0.06, 0.24, 0.88, 0.07])
ax_leg.set_facecolor(BG); ax_leg.axis('off')
ax_leg.set_xlim(0,1); ax_leg.set_ylim(0,1)

legend_items = [
    (C_HIGH, 'Danger (20%+)'),
    (C_MED,  'Vulnerable (10–20%)'),
    (C_LOW,  'Managed (<10%)'),
]
positions = [0.18, 0.50, 0.78]
for (col, lbl), xp in zip(legend_items, positions):
    ax_leg.add_patch(plt.Circle((xp - 0.04, 0.50), 0.030, color=col,
                                 alpha=0.90, transform=ax_leg.transData, zorder=3))
    ax_leg.text(xp - 0.005, 0.50, lbl, ha='left', va='center',
                color=LABEL_FG, fontsize=9, fontname=INTER)

# ── Stats row ─────────────────────────────────────────────────────────────────
ax_stats = fig.add_axes([0.06, 0.15, 0.88, 0.09])
ax_stats.set_facecolor(BG); ax_stats.axis('off')
ax_stats.set_xlim(0,1); ax_stats.set_ylim(0,1)

# Divider line
ax_stats.plot([0.02, 0.98], [0.92, 0.92], color='#CCCCCC', linewidth=0.8)

crease_slot_pct = h.loc['crease','goal_pct'] + h.loc['slot','goal_pct']
r_side_pct      = h.loc['R circle','goal_pct'] + (h.loc['R wall','goal_pct'] if 'R wall' in h.index else 0)
total_goals     = int(es['is_goal'].sum())

stats = [
    (f"{crease_slot_pct:.0f}%",  'crease + slot',        C_HIGH),
    (f"{r_side_pct:.0f}%",       'from the right side',  C_MED),
    (f"{total_goals}",            'ES goals against',     LABEL_FG),
]
for i, (val, lbl, col) in enumerate(stats):
    xp = 0.15 + i * 0.35
    ax_stats.text(xp, 0.65, val, ha='center', va='center',
                  color=col, fontsize=22, fontname=BEBAS, fontweight='bold')
    ax_stats.text(xp, 0.18, lbl, ha='center', va='center',
                  color=GREY, fontsize=8.5, fontname=INTER)

# ── Strength / Weakness callout ────────────────────────────────────────────────
ax_note = fig.add_axes([0.06, 0.04, 0.88, 0.11])
ax_note.set_facecolor(BG); ax_note.axis('off')
ax_note.set_xlim(0,1); ax_note.set_ylim(0,1)

ax_note.plot([0.02, 0.98], [0.96, 0.96], color='#CCCCCC', linewidth=0.8)

# Weakness box (left half)
ax_note.add_patch(FancyBboxPatch((0.02, 0.04), 0.455, 0.82,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#FEF0F0', edgecolor=C_HIGH,
                                  linewidth=1.2, zorder=1))
ax_note.text(0.245, 0.78, 'WEAKNESS',
             ha='center', va='center', color=C_HIGH,
             fontsize=10, fontname=BEBAS, fontweight='bold', zorder=2)
ax_note.text(0.245, 0.42, 'Wrists through the middle — high-danger\ncenter and right circle are his soft spots',
             ha='center', va='center', color='#222222',
             fontsize=8, fontname=INTER, linespacing=1.55, zorder=2)

# Strength box (right half)
ax_note.add_patch(FancyBboxPatch((0.525, 0.04), 0.455, 0.82,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#F0FAF4', edgecolor=GREEN,
                                  linewidth=1.2, zorder=1))
ax_note.text(0.752, 0.78, 'STRENGTH',
             ha='center', va='center', color=GREEN,
             fontsize=10, fontname=BEBAS, fontweight='bold', zorder=2)
ax_note.text(0.752, 0.42, 'Consistently strong on the left side —\nL circle is his best zone over 3 seasons',
             ha='center', va='center', color='#222222',
             fontsize=8, fontname=INTER, linespacing=1.55, zorder=2)

# Footer
fig.text(0.5, 0.008, '@HockeyROI | hockeyROI.substack.com',
         ha='center', va='bottom', color=GREY,
         fontsize=8, fontname=INTER, style='italic')

# ── Save ───────────────────────────────────────────────────────────────────────
for path in [os.path.join(DESKTOP, 'husso_zone_chart.png')]:
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"Saved → {path}")
plt.close()
