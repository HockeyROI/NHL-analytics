#!/usr/bin/env python3
"""
HockeyROI — Brand Palette Chart Rebuild
Regenerates all 7 charts with HockeyROI design system.
Data/layout unchanged; only colors, fonts, backgrounds updated.
"""

import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.patches import Arc, FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE    = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
GEO     = os.path.join(BASE, "Charts", "geometry_post")
PLY     = os.path.join(BASE, "Charts", "playoff_preview")

# ── PALETTE ───────────────────────────────────────────────────────────────────
BG       = "#0B1D2E"
SURFACE  = "#1B3A5C"
BLUE     = "#2E7DC4"
ICE_BLUE = "#4AB3E8"
ORANGE   = "#FF6B35"
TEXT     = "#F0F4F8"

BG_T     = (11,  29,  46)
SURF_T   = (27,  58,  92)
BLUE_T   = (46, 125, 196)
ICE_T    = (74, 179, 232)
ORANGE_T = (255, 107,  53)
TEXT_T   = (240, 244, 248)

# ── FONTS ─────────────────────────────────────────────────────────────────────
IMPACT = "/System/Library/Fonts/Supplemental/Impact.ttf"
ARIAL  = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIALB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

for p in [IMPACT, ARIAL, ARIALB]:
    fm.fontManager.addfont(p)

imp_prop  = fm.FontProperties(fname=IMPACT)
arl_prop  = fm.FontProperties(fname=ARIAL)
arlb_prop = fm.FontProperties(fname=ARIALB)

def mpl(rgb255, a=1.0):
    """Convert 0-255 RGB tuple + alpha float → matplotlib RGBA tuple."""
    return (rgb255[0]/255, rgb255[1]/255, rgb255[2]/255, a)

plt.rcParams.update({
    "font.family"     : "Arial",
    "text.color"      : TEXT,
    "axes.labelcolor" : TEXT,
    "xtick.color"     : TEXT,
    "ytick.color"     : TEXT,
    "axes.facecolor"  : BG,
    "figure.facecolor": BG,
    "axes.edgecolor"  : BLUE,
    "axes.linewidth"  : 0.6,
    "grid.color"      : BLUE,
    "grid.alpha"      : 0.2,
    "grid.linewidth"  : 0.5,
    "legend.facecolor": SURFACE,
    "legend.edgecolor": BLUE,
})

def wm(fig, x=0.985, y=0.018):
    fig.text(x, y, "HockeyROI", ha="right", va="bottom",
             fontsize=11, color=ICE_BLUE, fontproperties=imp_prop, alpha=0.9)

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, which="both", length=3)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLUE); sp.set_alpha(0.35)
    ax.yaxis.grid(True, color=BLUE, alpha=0.2, lw=0.5)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. REBOUND CORRIDOR  (rink diagram)
# ─────────────────────────────────────────────────────────────────────────────
def make_rebound_corridor():
    fig, ax = plt.subplots(figsize=(7.6, 9.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # coordinate system: x=-42..42, y=0(blue line)..75(end boards)
    ax.set_xlim(-48, 48)
    ax.set_ylim(-6, 82)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── boards ──
    rink = FancyBboxPatch((-42, -2), 84, 77,
                          boxstyle="round,pad=3",
                          linewidth=2.8, edgecolor=BLUE,
                          facecolor=SURFACE, zorder=1)
    ax.add_patch(rink)

    # ── blue line ──
    ax.plot([-40, 40], [0, 0], color=ICE_BLUE, lw=5, zorder=4,
            solid_capstyle="butt")
    ax.text(-46, -1.5, "Blue\nline", fontsize=6.5, color=ICE_BLUE,
            va="top", ha="center", fontproperties=arl_prop)

    # ── goal line ──
    GOAL_Y = 65
    ax.plot([-40, 40], [GOAL_Y, GOAL_Y], color=ORANGE, lw=2.2, zorder=4)

    # ── face-off circles (left & right) ──
    for cx in [-20, 20]:
        cy = 42
        c = plt.Circle((cx, cy), 15, fill=False,
                        edgecolor=ICE_BLUE, lw=1.4, zorder=3)
        ax.add_patch(c)
        ax.plot(cx, cy, "o", color=ICE_BLUE, ms=3, zorder=4)
        # hash marks
        for sign in [-1, 1]:
            ax.plot([cx+sign*4, cx+sign*4], [cy+13, cy+17],
                    color=ICE_BLUE, lw=1.0, zorder=4)
            ax.plot([cx+sign*4, cx+sign*4], [cy-17, cy-13],
                    color=ICE_BLUE, lw=1.0, zorder=4)

    # ── net ──
    NET_W, NET_D = 12, 4.5
    net = mpatches.Rectangle((-NET_W/2, GOAL_Y), NET_W, NET_D,
                              linewidth=2, edgecolor=TEXT,
                              facecolor=mpl(SURF_T, 0.7), zorder=7)
    ax.add_patch(net)
    for px in [-NET_W/2, NET_W/2]:
        ax.plot([px, px], [GOAL_Y, GOAL_Y+NET_D],
                color=TEXT, lw=3.5, zorder=8, solid_capstyle="round")

    # ── crease ──
    crease_fill = mpatches.Wedge((0, GOAL_Y), 8.5, 180, 360,
                                  facecolor=mpl(ICE_T, 0.2), edgecolor="none", zorder=5)
    ax.add_patch(crease_fill)
    crease_edge = Arc((0, GOAL_Y), 17, 12, angle=0,
                      theta1=180, theta2=360,
                      color=ICE_BLUE, lw=1.5, zorder=6)
    ax.add_patch(crease_edge)

    # ── doorstep zone ──
    DS_X, DS_Y, DS_W, DS_H = -9, 56, 18, 10
    ds_bg = mpatches.FancyBboxPatch((DS_X, DS_Y), DS_W, DS_H,
                                     boxstyle="round,pad=0.8",
                                     facecolor=mpl(ORANGE_T, 0.4),
                                     edgecolor=ORANGE, lw=1.8, zorder=7)
    ax.add_patch(ds_bg)
    # X arrows
    cx_d, cy_d = DS_X + DS_W/2, DS_Y + DS_H/2
    for dx, dy in [(-3, -2.5), (3, 2.5)]:
        ax.annotate("", xy=(cx_d-dx, cy_d-dy),
                    xytext=(cx_d+dx, cy_d+dy),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                   lw=1.1, mutation_scale=8), zorder=11)
    ax.text(cx_d, cy_d+0.5, "Doorstep Zone",
            ha="center", va="center", fontsize=6.5, color=TEXT,
            fontproperties=arlb_prop, zorder=12)

    # ── "85%" annotation ──
    ax.text(0, 49, "85% of dangerous rebounds originate here",
            ha="center", va="center", fontsize=8, color=TEXT,
            fontproperties=arlb_prop,
            bbox=dict(fc=BG, ec="none", pad=2, alpha=0.7), zorder=13)

    # ── shot origin glows ──
    def glow(x, y, col, outer_a=0.06, mid_a=0.14, dot_a=0.75):
        ax.scatter(x, y, s=900, color=col, alpha=outer_a, zorder=5)
        ax.scatter(x, y, s=380, color=col, alpha=mid_a,   zorder=6)
        ax.scatter(x, y, s=120, color=col, alpha=dot_a,   zorder=7)

    # left & right circles (blue glow = shot origin)
    glow(-20, 42, BLUE)
    ax.text(-38, 42, "Left circle", fontsize=7.5, color=TEXT,
            ha="center", va="center", fontproperties=arl_prop)

    glow(20, 42, BLUE)
    ax.text(37, 42, "Right circle", fontsize=7.5, color=TEXT,
            ha="center", va="center", fontproperties=arl_prop)

    # high slot (orange — these feed doorstep)
    glow(0, 30, ORANGE)
    ax.text(10, 30, "High slot", fontsize=7.5, color=TEXT,
            ha="left", va="center", fontproperties=arl_prop)

    # point shot (bottom)
    glow(0, 5, ORANGE)
    ax.text(10, 5, "Point shot", fontsize=7.5, color=TEXT,
            ha="left", va="center", fontproperties=arl_prop)

    # ── arrows: point → high slot → doorstep ──
    arr = dict(arrowstyle="-|>", mutation_scale=13, lw=2.0,
               color=ORANGE, zorder=9)
    ax.annotate("", xy=(0, 26), xytext=(0, 9),
                arrowprops=dict(**arr, alpha=0.65))
    ax.annotate("", xy=(0, DS_Y+0.5), xytext=(0, 34),
                arrowprops=dict(**arr, alpha=0.95))

    # ── title ──
    ax.set_title("The Rebound Corridor", fontsize=20, color=TEXT,
                 fontproperties=imp_prop, pad=10)
    fig.text(0.5, 0.915, "All roads lead to the doorstep",
             ha="center", fontsize=10.5, color=ICE_BLUE,
             fontproperties=arl_prop)
    wm(fig)

    fig.savefig(os.path.join(GEO, "rebound_corridor.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓  rebound_corridor.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVERSION MULTIPLIER  (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
def make_conversion_multiplier():
    labels = ["All other ES shots", "Tip-ins", "Deflections", "Rebound shots"]
    values = [1.00, 1.59, 1.75, 2.00]
    # brand spec colors per bar
    colors = [BLUE, ICE_BLUE, BLUE, ORANGE]  # baseline blue, ice blue, blue-mid, orange signal

    fig, ax = plt.subplots(figsize=(14, 7.5))
    style_ax(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=BLUE, alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)

    bars = ax.barh(labels, values, color=colors, height=0.52,
                   edgecolor="none", zorder=3)

    # value labels on bars
    bold_vals = {1.00, 2.00}
    for bar, val, lbl in zip(bars, values, labels):
        w = bar.get_width()
        ax.text(w + 0.015, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}x",
                va="center", ha="left", fontsize=20,
                color=TEXT, fontproperties=arlb_prop)

    # baseline dashed line at 1.0
    ax.axvline(1.0, color=TEXT, alpha=0.45, lw=1.4,
               linestyle="--", zorder=5)
    ax.text(1.0, -0.65, "baseline", ha="center", va="top",
            fontsize=9, color=TEXT, alpha=0.55, fontproperties=arl_prop)

    ax.set_xlim(0, 2.18)
    ax.set_ylim(-0.6, 3.6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=13)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_alpha(0.3)

    # title
    fig.text(0.08, 0.95,
             "Net-Front Attempts Convert at Nearly Double the Rate",
             ha="left", va="top", fontsize=18,
             color=TEXT, fontproperties=arlb_prop)
    fig.text(0.08, 0.885,
             "Goal rate multiplier vs all other even strength shots  |  6 seasons  |  921,141 shots",
             ha="left", va="top", fontsize=11, color=ICE_BLUE,
             fontproperties=arl_prop)
    wm(fig)

    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(os.path.join(GEO, "conversion_multiplier.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓  conversion_multiplier.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. HYMAN PROFILE  (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
def make_hyman_profile():
    seasons  = ["20-21","21-22","22-23","23-24","24-25","25-26"]
    nf_rates = [0.086, 0.079, 0.077, 0.104, 0.096, 0.110]
    timing   = [0.50,  0.78,  0.23,  0.72,  0.55,  0.42]
    avg_nf   = sum(nf_rates) / len(nf_rates)   # 0.092
    avg_tim  = sum(timing)   / len(timing)      # 0.533

    x   = np.arange(len(seasons))
    w   = 0.38

    fig, ax = plt.subplots(figsize=(14, 7.2))
    style_ax(ax)
    ax.xaxis.grid(False)

    b1 = ax.bar(x - w/2, nf_rates, w, color=BLUE,     label="NF attempt rate",  zorder=3)
    b2 = ax.bar(x + w/2, timing,   w, color=ORANGE,   label="Timing score (0-1)", zorder=3)

    # value labels
    for bar, val in zip(b1, nf_rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                color=BLUE, fontproperties=arlb_prop)
    for bar, val in zip(b2, timing):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.006,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                color=ORANGE, fontproperties=arlb_prop)

    # career avg lines (orange dashed = the ONE key stat signal)
    ax.axhline(avg_nf,  color=ORANGE, lw=1.6, ls="--", zorder=5)
    ax.axhline(avg_tim, color=BLUE,   lw=1.6, ls="--", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(seasons, fontsize=13)
    ax.set_ylabel("Rate / Score", fontsize=11, color=TEXT)
    ax.set_ylim(0, 0.90)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)

    legend = ax.legend(fontsize=11, loc="upper right",
                       framealpha=0.9, edgecolor=BLUE)
    for txt in legend.get_texts():
        txt.set_color(TEXT)

    # title
    fig.text(0.5, 0.97, "Zach Hyman — Net-Front Profile by Season",
             ha="center", va="top", fontsize=20,
             color=TEXT, fontproperties=imp_prop)
    fig.text(0.5, 0.925, "Attempt rate and timing score  |  Even strength 5v5",
             ha="center", va="top", fontsize=11, color=ICE_BLUE,
             fontproperties=arl_prop)
    wm(fig)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(os.path.join(GEO, "hyman_profile.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓  hyman_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. THREE PILLARS  (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
def make_three_pillars():
    groups   = ["0 pillars","1 pillar","2 pillars","3 pillars"]
    playoff  = [5.1,  32.5, 70.0, 92.5]
    cup_fin  = [0.0,   5.0,  7.5, 12.5]
    x = np.arange(len(groups))
    w = 0.38

    fig, ax = plt.subplots(figsize=(13, 7.8))
    style_ax(ax)
    ax.xaxis.grid(False)

    b1 = ax.bar(x - w/2, playoff,  w, color=BLUE,   label="% making playoffs",         zorder=3)
    b2 = ax.bar(x + w/2, cup_fin,  w, color=ORANGE, label="% reaching Cup Final / winning", zorder=3)

    # value labels on bars
    for bar, val in zip(b1, playoff):
        if val > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=14,
                    color=BLUE, fontproperties=arlb_prop)
    for bar, val in zip(b2, cup_fin):
        if val > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=14,
                    color=ORANGE, fontproperties=arlb_prop)

    # annotation arrow for 92.5% bar
    ax.annotate("92.5% playoff rate\n6 seasons, n=40",
                xy=(x[3] - w/2 + w/2, 92.5),
                xytext=(x[3] - 0.55, 78),
                fontsize=9, color=TEXT, fontproperties=arl_prop,
                arrowprops=dict(arrowstyle="->", color=TEXT,
                                lw=1.0, mutation_scale=10))

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=14)
    ax.set_ylabel("% of teams", fontsize=12, color=TEXT)
    ax.set_ylim(0, 108)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)

    # legend
    handles = [
        mpatches.Patch(color=BLUE,   label="% making playoffs"),
        mpatches.Patch(color=ORANGE, label="% reaching Cup Final / winning"),
    ]
    legend = ax.legend(handles=handles, fontsize=10, loc="upper left",
                       framealpha=0.9, edgecolor=BLUE)
    for txt in legend.get_texts():
        txt.set_color(TEXT)

    # title block
    fig.text(0.07, 0.97, "The Three-Pillar Blueprint",
             ha="left", va="top", fontsize=22,
             color=TEXT, fontproperties=arlb_prop)
    fig.text(0.07, 0.925,
             "Playoff and Cup Final rates by number of model pillars met  |  2020-21 through 2025-26",
             ha="left", va="top", fontsize=10, color=ICE_BLUE,
             fontproperties=arl_prop)
    fig.text(0.07, 0.895,
             "Pillars: NF rate + Goals against + Special teams  |  n=159 team-seasons",
             ha="left", va="top", fontsize=10, color=ICE_BLUE,
             fontproperties=arl_prop)
    wm(fig)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(GEO, "three_pillars.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓  three_pillars.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PILLAR SCORECARD  (PIL table)
# ─────────────────────────────────────────────────────────────────────────────
def make_pillar_scorecard():
    # thresholds: NF >= 15.2%  GA <= 3.00  ST >= 50.9%
    THR_NF = 15.2; THR_GA = 3.00; THR_ST = 50.9

    rows = [
        # (team, nf, ga, st, pillars, note)
        ("COL", 29.1, 2.593, 50.9, 3, ""),
        ("CAR", 32.0, 2.963, 52.7, 3, ""),
        ("DAL", 37.1, 2.841, 54.5, 3, ""),
        ("TBL", 33.9, 2.866, 51.6, 3, ""),
        ("MIN", 37.6, 3.000, 52.5, 3, ""),
        ("BUF", 34.0, 3.000, 50.7, 2, "← misses by 0.15%"),
        ("PIT", 36.9, 3.378, 52.8, 2, ""),
        ("VGK", 33.3, 3.110, 53.0, 2, ""),
        ("EDM", 37.4, 3.358, 54.2, 2, ""),
        ("UTA", 35.8, 2.901, 49.1, 2, ""),
        ("MTL", 34.2, 3.207, 50.7, 1, ""),
        ("BOS", 31.0, 3.073, 50.2, 1, ""),
        ("OTT", 34.7, 3.012, 49.8, 1, ""),
        ("PHI", 36.4, 3.024, 46.6, 1, ""),
        ("ANA", 35.5, 3.568, 47.5, 1, ""),
        ("LAK", 33.4, 3.136, 45.8, 1, ""),
    ]

    # PIL dimensions
    W, H = 730, 880
    img  = Image.new("RGB", (W, H), BG_T)
    draw = ImageDraw.Draw(img)

    fnt_title  = ImageFont.truetype(IMPACT, 20)
    fnt_hdr    = ImageFont.truetype(ARIALB, 12)
    fnt_team   = ImageFont.truetype(ARIALB, 13)
    fnt_val    = ImageFont.truetype(ARIALB, 13)
    fnt_small  = ImageFont.truetype(ARIAL,  10)
    fnt_badge  = ImageFont.truetype(ARIALB, 14)
    fnt_sect   = ImageFont.truetype(ARIALB, 10)

    def tw(text, font):
        bb = draw.textbbox((0,0), text, font=font)
        return bb[2] - bb[0]

    # ── title ──
    title = "2025-26 Playoff Teams — Four-Pillar Model"
    draw.text(((W - tw(title, fnt_title))//2, 14),
              title, font=fnt_title, fill=TEXT_T)

    # ── column headers ──
    ROW_H   = 42
    HDR_Y   = 50
    COL_X   = [30, 160, 310, 450, 595]   # Team, NF Rate, GA/gm, ST Score, Pillars

    hdr_bg = [0, HDR_Y, W, HDR_Y + ROW_H]
    draw.rectangle(hdr_bg, fill=SURF_T)

    headers = ["Team", "NF Rate", "GA / gm", "ST Score", "Pillars"]
    sub_hdrs = ["", f"≥ {THR_NF:.1f}%", f"≤ {THR_GA:.2f}", f"≥ {THR_ST:.1f}%", ""]
    for col, (hdr, sub, cx) in enumerate(zip(headers, sub_hdrs, COL_X)):
        xt = cx + (COL_X[col+1] - cx)//2 if col < 4 else cx + 30
        draw.text(((W - tw(hdr, fnt_hdr))//2 if col == 0 else xt - tw(hdr, fnt_hdr)//2,
                   HDR_Y + 6), hdr, font=fnt_hdr, fill=ICE_T)
        if sub:
            draw.text((xt - tw(sub, fnt_small)//2, HDR_Y + 24),
                      sub, font=fnt_small, fill=(*ICE_T[:3], 180))

    DATA_Y0 = HDR_Y + ROW_H
    section_drawn = {2: False, 1: False}
    cur_y = DATA_Y0
    row_idx = 0

    for (team, nf, ga, st, pillars, note) in rows:
        # section divider
        if pillars == 2 and not section_drawn[2]:
            section_drawn[2] = True
            draw.rectangle([0, cur_y, W, cur_y+20], fill=(*BLUE_T, 55))
            draw.text((36, cur_y+4), "2 PILLARS", font=fnt_sect, fill=(*ICE_T,))
            cur_y += 20
        if pillars == 1 and not section_drawn[1]:
            section_drawn[1] = True
            draw.rectangle([0, cur_y, W, cur_y+20], fill=(*BLUE_T, 55))
            draw.text((36, cur_y+4), "1 PILLAR", font=fnt_sect, fill=(*ICE_T,))
            cur_y += 20

        # alternating row background
        row_bg = SURF_T if row_idx % 2 == 1 else BG_T
        draw.rectangle([0, cur_y, W, cur_y + ROW_H], fill=row_bg)

        # subtle row divider
        draw.line([(0, cur_y), (W, cur_y)], fill=(*BLUE_T, 30), width=1)

        vcx = cur_y + ROW_H//2   # vertical center of row

        # Team name
        draw.text((36, vcx - 8), team, font=fnt_team, fill=TEXT_T)

        # NF Rate — all teams meet threshold
        nf_s = f"{nf:.1f}%"
        col = ICE_T  # met
        nf_x = COL_X[1] + (COL_X[2]-COL_X[1])//2
        draw.text((nf_x - tw(nf_s, fnt_val)//2, vcx-8), nf_s, font=fnt_val, fill=col)

        # GA/gm
        ga_s = f"{ga:.3f}"
        ga_col = ICE_T if ga <= THR_GA else ORANGE_T
        ga_x = COL_X[2] + (COL_X[3]-COL_X[2])//2
        draw.text((ga_x - tw(ga_s, fnt_val)//2, vcx-8), ga_s, font=fnt_val, fill=ga_col)

        # ST Score (+ optional note)
        st_s = f"{st:.1f}%"
        st_col = ICE_T if st >= THR_ST else ORANGE_T
        st_x = COL_X[3] + (COL_X[4]-COL_X[3])//2 - (30 if note else 0)
        draw.text((st_x - tw(st_s, fnt_val)//2, vcx-8), st_s, font=fnt_val, fill=st_col)
        if note:
            draw.text((st_x + tw(st_s, fnt_val)//2 + 6, vcx-6),
                      note, font=fnt_small, fill=ORANGE_T)

        # Pillars badge
        badge_col = ORANGE_T if pillars == 3 else (BLUE_T if pillars == 2 else (160,160,170))
        bx = COL_X[4] + 30
        draw.text((bx - tw(str(pillars), fnt_badge)//2, vcx-10),
                  str(pillars), font=fnt_badge, fill=badge_col)

        cur_y += ROW_H
        row_idx += 1

    # watermark
    wm_text = "HockeyROI"
    draw.text((W - tw(wm_text, fnt_small) - 12, H - 24),
              wm_text, font=fnt_small, fill=(*ICE_T[:3], 180))

    img.save(os.path.join(PLY, "pillar_scorecard.png"))
    print("  ✓  pillar_scorecard.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MATCHUP PICKS  (bracket — PIL)
# ─────────────────────────────────────────────────────────────────────────────
def make_matchup_picks():
    W, H = 1400, 780
    img  = Image.new("RGB", (W, H), BG_T)
    draw = ImageDraw.Draw(img)

    fnt_title  = ImageFont.truetype(IMPACT, 26)
    fnt_sub    = ImageFont.truetype(ARIAL,  12)
    fnt_conf   = ImageFont.truetype(ARIALB, 11)
    fnt_rnd    = ImageFont.truetype(ARIALB, 10)
    fnt_team_w = ImageFont.truetype(ARIALB, 14)   # winner
    fnt_team_l = ImageFont.truetype(ARIAL,  13)   # loser
    fnt_seed   = ImageFont.truetype(ARIAL,  9)
    fnt_badge  = ImageFont.truetype(ARIALB, 10)
    fnt_wm     = ImageFont.truetype(ARIAL,  10)
    fnt_cup    = ImageFont.truetype(ARIALB, 11)

    def tw(t, f):
        bb = draw.textbbox((0,0), t, font=f)
        return bb[2]-bb[0]

    CARD_W, CARD_H = 188, 112
    LINE_C  = (*BLUE_T,)
    LOSER_C = (120, 132, 144)    # dim TEXT

    # confidence badge colors
    CONF_COLORS = {
        "High":    ICE_T,
        "Medium":  ORANGE_T,
        "Med-Low": (220, 80, 30),
        "Low":     (200, 50, 50),
    }

    def draw_card(x, y, top_team, top_seed, bot_team, bot_seed, winner, conf):
        """Draw one matchup card. winner is 'top' or 'bot'."""
        draw.rounded_rectangle([x, y, x+CARD_W, y+CARD_H],
                                radius=6, fill=SURF_T,
                                outline=(*BLUE_T, 80), width=1)
        mid_y = y + CARD_H // 2

        # divider between two teams
        draw.line([(x+8, mid_y), (x+CARD_W-8, mid_y)],
                  fill=(*BLUE_T, 40), width=1)

        for team, seed, row_y, is_winner in [
            (top_team, top_seed, y + CARD_H//4 - 8,   winner == "top"),
            (bot_team, bot_seed, y + 3*CARD_H//4 - 8, winner == "bot"),
        ]:
            col  = ORANGE_T if is_winner else LOSER_C
            font = fnt_team_w if is_winner else fnt_team_l
            # seed
            draw.text((x+8, row_y+2), str(seed), font=fnt_seed, fill=col)
            # team name
            draw.text((x+22, row_y), team, font=font, fill=col)

        # confidence badge below card
        badge_c = CONF_COLORS.get(conf, ICE_T)
        CIR_R = 5
        badge_y = y + CARD_H + 8
        draw.ellipse([x+8-CIR_R, badge_y-CIR_R, x+8+CIR_R, badge_y+CIR_R],
                     fill=badge_c)
        draw.text((x+18, badge_y - 7), conf, font=fnt_badge, fill=(*badge_c,))

    def bracket_line(x1, y1, x2, y2, elbow_x=None):
        """Draw L-shaped bracket connector."""
        if elbow_x is None:
            elbow_x = (x1 + x2) // 2
        pts = [(x1,y1),(elbow_x,y1),(elbow_x,y2),(x2,y2)]
        draw.line(pts, fill=LINE_C, width=2)

    def q_box(x, y, w=110, h=56):
        """Round 2 / Conf Finals placeholder box."""
        draw.rounded_rectangle([x, y, x+w, y+h],
                                radius=5, fill=SURF_T,
                                outline=(*BLUE_T, 80), width=1)
        draw.text((x + w//2 - tw("?", fnt_team_w)//2, y + h//2 - 10),
                  "?", font=fnt_team_w, fill=(*TEXT_T[:3], 90))

    # ── East matchups (left side) ──
    EAST_X = 15
    # vertical top positions for each matchup card
    EM_Y = [72, 228, 404, 560]

    east_data = [
        ("BUF",1,"BOS",8,"top","High"),
        ("TBL",2,"MTL",7,"top","High"),
        ("CAR",3,"OTT",6,"top","Medium"),
        ("PIT",4,"PHI",5,"top","Medium"),
    ]
    for i, (t1,s1,t2,s2,win,conf) in enumerate(east_data):
        draw_card(EAST_X, EM_Y[i], t1,s1, t2,s2, win, conf)

    # ── West matchups (right side) ──
    WEST_X = W - 15 - CARD_W
    west_data = [
        ("COL",1,"LAK",8,"top","High"),
        ("DAL",2,"MIN",7,"top","Medium"),
        ("EDM",3,"ANA",6,"top","Med-Low"),
        ("VGK",4,"UTA",5,"bot","Low"),   # UTA is upset pick (bottom)
    ]
    WM_Y = EM_Y[:]
    for i, (t1,s1,t2,s2,win,conf) in enumerate(west_data):
        draw_card(WEST_X, WM_Y[i], t1,s1, t2,s2, win, conf)

    # ── Bracket connectors — East ──
    card_mid_y = [y + CARD_H//2 for y in EM_Y]
    # Pairs: (0,1) → R2a, (2,3) → R2b
    R2_X = EAST_X + CARD_W + 30

    R2A_Y = (card_mid_y[0] + card_mid_y[1]) // 2 - 28
    R2B_Y = (card_mid_y[2] + card_mid_y[3]) // 2 - 28
    R2_W, R2_H = 110, 56

    bracket_line(EAST_X+CARD_W, card_mid_y[0], R2_X, R2A_Y+R2_H//2)
    bracket_line(EAST_X+CARD_W, card_mid_y[1], R2_X, R2A_Y+R2_H//2)
    q_box(R2_X, R2A_Y, R2_W, R2_H)

    bracket_line(EAST_X+CARD_W, card_mid_y[2], R2_X, R2B_Y+R2_H//2)
    bracket_line(EAST_X+CARD_W, card_mid_y[3], R2_X, R2B_Y+R2_H//2)
    q_box(R2_X, R2B_Y, R2_W, R2_H)

    # Conf Finals East
    CF_X = R2_X + R2_W + 30
    CF_W, CF_H = 100, 56
    CF_E_Y = (R2A_Y + R2_H//2 + R2B_Y + R2_H//2)//2 - CF_H//2
    bracket_line(R2_X+R2_W, R2A_Y+R2_H//2, CF_X, CF_E_Y+CF_H//2)
    bracket_line(R2_X+R2_W, R2B_Y+R2_H//2, CF_X, CF_E_Y+CF_H//2)
    q_box(CF_X, CF_E_Y, CF_W, CF_H)

    # ── Bracket connectors — West ──
    R2W_X = WEST_X - 30 - R2_W
    R2WA_Y = R2A_Y; R2WB_Y = R2B_Y

    bracket_line(WEST_X, card_mid_y[0], R2W_X+R2_W, R2WA_Y+R2_H//2)
    bracket_line(WEST_X, card_mid_y[1], R2W_X+R2_W, R2WA_Y+R2_H//2)
    q_box(R2W_X, R2WA_Y, R2_W, R2_H)

    bracket_line(WEST_X, card_mid_y[2], R2W_X+R2_W, R2WB_Y+R2_H//2)
    bracket_line(WEST_X, card_mid_y[3], R2W_X+R2_W, R2WB_Y+R2_H//2)
    q_box(R2W_X, R2WB_Y, R2_W, R2_H)

    CF_W_X = R2W_X - 30 - CF_W
    CF_W_Y = CF_E_Y
    bracket_line(R2W_X, R2WA_Y+R2_H//2, CF_W_X+CF_W, CF_W_Y+CF_H//2)
    bracket_line(R2W_X, R2WB_Y+R2_H//2, CF_W_X+CF_W, CF_W_Y+CF_H//2)
    q_box(CF_W_X, CF_W_Y, CF_W, CF_H)

    # ── Stanley Cup Final (center) ──
    CUP_W, CUP_H = 170, 100
    CUP_X = (W - CUP_W) // 2
    CUP_Y = CF_E_Y + CF_H//2 - CUP_H//2
    draw.rounded_rectangle([CUP_X, CUP_Y, CUP_X+CUP_W, CUP_Y+CUP_H],
                            radius=8, fill=SURF_T,
                            outline=ORANGE_T, width=2)
    # trophy "T" shape
    tx, ty = CUP_X + CUP_W//2, CUP_Y + 32
    draw.rectangle([tx-18, ty-10, tx+18, ty+6], fill=ORANGE_T)
    draw.rectangle([tx-6, ty+6, tx+6, ty+28], fill=ORANGE_T)
    # label
    cup_lbl = "STANLEY CUP FINAL"
    draw.text((CUP_X + CUP_W//2 - tw(cup_lbl, fnt_cup)//2, CUP_Y+CUP_H-22),
              cup_lbl, font=fnt_cup, fill=ORANGE_T)

    # Connect Conf Finals to Cup Final
    bracket_line(CF_X+CF_W, CF_E_Y+CF_H//2, CUP_X, CUP_Y+CUP_H//2)
    bracket_line(CF_W_X, CF_W_Y+CF_H//2, CUP_X+CUP_W, CUP_Y+CUP_H//2)

    # ── Conference labels ──
    draw.text((EAST_X + CARD_W//2 - tw("EASTERN CONFERENCE",fnt_conf)//2, 28),
              "EASTERN CONFERENCE", font=fnt_conf, fill=(*ICE_T,))
    draw.text((WEST_X + CARD_W//2 - tw("WESTERN CONFERENCE",fnt_conf)//2, 28),
              "WESTERN CONFERENCE", font=fnt_conf, fill=(*ICE_T,))

    # ── Round labels at bottom ──
    ROUND_Y = H - 46
    round_labels = [
        (EAST_X + CARD_W//2,       "ROUND 1"),
        (R2_X + R2_W//2,           "ROUND 2"),
        (CF_X + CF_W//2,           "CONF. FINALS"),
        (W//2,                     ""),
        (CF_W_X + CF_W//2,         "CONF. FINALS"),
        (R2W_X + R2_W//2,          "ROUND 2"),
        (WEST_X + CARD_W//2,       "ROUND 1"),
    ]
    for rx, rl in round_labels:
        if rl:
            draw.text((rx - tw(rl, fnt_rnd)//2, ROUND_Y), rl,
                      font=fnt_rnd, fill=(*ICE_T[:3], 180))

    # ── Title ──
    ttl = "HockeyROI — 2025-26 Playoff Predictions"
    draw.text(((W - tw(ttl, fnt_title))//2, 4),
              ttl, font=fnt_title, fill=TEXT_T)
    sub = "Round 1 picks based on four-pillar model  |  Later rounds TBD"
    draw.text(((W - tw(sub, fnt_sub))//2, 38),
              sub, font=fnt_sub, fill=(*ICE_T[:3], 180))

    # watermark
    draw.text((W - tw("HockeyROI", fnt_wm) - 12, H - 22),
              "HockeyROI", font=fnt_wm, fill=(*ICE_T[:3], 180))

    img.save(os.path.join(PLY, "matchup_picks.png"))
    print("  ✓  matchup_picks.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. SPECIAL TEAMS RANKINGS  (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
def make_special_teams():
    data = [
        ("LAK", 45.8), ("PHI", 46.6), ("ANA", 47.5), ("UTA", 49.1),
        ("OTT", 49.8), ("BOS", 50.2), ("MTL", 50.7), ("BUF", 50.7),
        ("COL", 50.9), ("TBL", 51.6), ("MIN", 52.5), ("CAR", 52.7),
        ("PIT", 52.8), ("VGK", 53.0), ("EDM", 54.2), ("DAL", 54.5),
    ]
    THR = 50.9
    teams  = [d[0] for d in data]
    scores = [d[1] for d in data]
    colors = [ORANGE if s >= THR else BLUE for s in scores]

    fig, ax = plt.subplots(figsize=(11.5, 9.8))
    style_ax(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=BLUE, alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)

    bars = ax.barh(teams, scores, color=colors, height=0.62,
                   left=min(scores)-1.5,
                   edgecolor="none", zorder=3)

    # value labels
    for bar, val in zip(bars, scores):
        ax.text(val + 0.08, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=11, color=TEXT, fontproperties=arlb_prop)

    # threshold line
    ax.axvline(THR, color=ICE_BLUE, lw=1.6, ls="--", zorder=5)
    ax.text(THR + 0.08, len(teams) - 0.3,
            f"Top-third\nthreshold\n{THR:.1f}%",
            fontsize=8.5, color=ICE_BLUE, va="top",
            fontproperties=arl_prop)

    ax.set_xlim(min(scores)-1.8, max(scores)+1.6)
    ax.tick_params(axis="y", labelsize=12, left=False)
    ax.tick_params(axis="x", labelsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    for sp in ["top","right","left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_alpha(0.3)

    # team name styling on y-axis — bold all
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(arlb_prop)
        lbl.set_color(TEXT)

    # legend
    handles = [
        mpatches.Patch(color=ORANGE, label="Meets top-third threshold"),
        mpatches.Patch(color=BLUE,   label="Below threshold"),
    ]
    legend = ax.legend(handles=handles, fontsize=10, loc="lower right",
                       framealpha=0.9, edgecolor=BLUE)
    for txt in legend.get_texts():
        txt.set_color(TEXT)

    # title
    fig.text(0.5, 0.975, "Special Teams Score — 2025-26 Playoff Teams",
             ha="center", va="top", fontsize=20,
             color=TEXT, fontproperties=imp_prop)
    fig.text(0.5, 0.935,
             "Average of PP% and PK%  |  Orange = meets top-third threshold",
             ha="center", va="top", fontsize=11, color=ICE_BLUE,
             fontproperties=arl_prop)
    wm(fig)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(PLY, "special_teams_rankings.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓  special_teams_rankings.png")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nHockeyROI — Rebuilding 7 charts with brand palette\n")
    make_rebound_corridor()
    make_conversion_multiplier()
    make_hyman_profile()
    make_three_pillars()
    make_pillar_scorecard()
    make_matchup_picks()
    make_special_teams()
    print("\nAll 7 charts saved.\n")
