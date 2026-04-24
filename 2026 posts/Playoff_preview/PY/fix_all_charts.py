#!/usr/bin/env python3
"""
Fix three charts:
  1. Charts/covers/playoff_cover.png   — cup image: scale-to-fit (no crop)
  2. Charts/playoff_preview/pillar_scorecard.png — full rebuild
  3. Charts/playoff_preview/special_teams_rankings.png — x-axis from 0
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

BASE     = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
COVERS   = os.path.join(BASE, "Charts", "covers")
PLY      = os.path.join(BASE, "Charts", "playoff_preview")

# ── Shared palette ─────────────────────────────────────────────────────────────
BG_T     = (11,  29,  46)    # #0B1D2E
SURF_T   = (27,  58,  92)    # #1B3A5C
BLUE_T   = (46, 125, 196)    # #2E7DC4
ICE_T    = (74, 179, 232)    # #4AB3E8
ORANGE_T = (255, 107,  53)   # #FF6B35
TEXT_T   = (240, 244, 248)   # #F0F4F8

BG       = "#0B1D2E"
SURFACE  = "#1B3A5C"
BLUE     = "#2E7DC4"
ICE_BLUE = "#4AB3E8"
ORANGE   = "#FF6B35"
TEXT     = "#F0F4F8"

IMPACT = "/System/Library/Fonts/Supplemental/Impact.ttf"
ARIAL  = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIALB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

for p in [IMPACT, ARIAL, ARIALB]:
    fm.fontManager.addfont(p)
imp_prop  = fm.FontProperties(fname=IMPACT)
arl_prop  = fm.FontProperties(fname=ARIAL)
arlb_prop = fm.FontProperties(fname=ARIALB)

plt.rcParams.update({
    "font.family": "Arial", "text.color": TEXT,
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.facecolor": BG, "figure.facecolor": BG,
    "axes.edgecolor": BLUE, "axes.linewidth": 0.6,
    "grid.color": BLUE, "grid.alpha": 0.2, "grid.linewidth": 0.5,
    "legend.facecolor": SURFACE, "legend.edgecolor": BLUE,
})

def wm_mpl(fig, x=0.985, y=0.018):
    fig.text(x, y, "HockeyROI", ha="right", va="bottom",
             fontsize=11, color=ICE_BLUE, fontproperties=imp_prop, alpha=0.9)

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, which="both", length=3)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLUE); sp.set_alpha(0.35)
    ax.yaxis.grid(True, color=BLUE, alpha=0.2, lw=0.5)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1 — playoff_cover.png  (cup image: scale-to-fit, no crop)
# ═══════════════════════════════════════════════════════════════════════════════
def fh(size): return ImageFont.truetype(IMPACT, size)
def fb(size): return ImageFont.truetype(ARIALB, size)
def fr(size): return ImageFont.truetype(ARIAL, size)

def tw_pil(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]

def draw_wordmark_pil(draw, W, H, size=26):
    font = fh(size)
    t = "HOCKEYROI"
    x = W - tw_pil(draw, t, font) - 44
    draw.text((x, H - 48), t, font=font, fill=ICE_T)

def orange_bar_pil(draw, x, y, w=160, h=5):
    draw.rectangle([x, y, x + w, y + h], fill=ORANGE_T)

def bracket_overlay(W, H, opacity=16):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c = (*BLUE_T, opacity)
    bx = 580; col_w, col_h = 120, 160; gap = 40
    for row in range(4):
        y0 = 70 + row * (col_h + gap); y1 = y0 + col_h
        d.rounded_rectangle([bx, y0, bx + col_w, y1], radius=8, outline=c, width=2)
        mid_y = (y0 + y1) // 2
        if row % 2 == 0 and row + 1 < 4:
            next_y0 = 70 + (row + 1) * (col_h + gap)
            next_mid = (next_y0 + next_y0 + col_h) // 2
            c2 = (*BLUE_T, opacity - 4)
            d.line([(bx+col_w, mid_y), (bx+col_w+80, mid_y)], fill=c2, width=2)
            d.line([(bx+col_w+80, mid_y), (bx+col_w+80, next_mid)], fill=c2, width=2)
    bx2 = bx + col_w + 160
    for row in range(2):
        y0 = 70 + row * (2*(col_h+gap)+gap//2); y1 = y0 + col_h
        d.rounded_rectangle([bx2, y0, bx2+col_w, y1], radius=8, outline=c, width=2)
    tx, ty = W - 160, 140
    d.rounded_rectangle([tx, ty, tx+90, ty+130], radius=10, outline=c, width=2)
    d.rectangle([tx+25, ty+130, tx+65, ty+170], outline=c, width=2)
    d.line([(tx+10, ty+170), (tx+80, ty+170)], fill=c, width=3)
    return layer

def fix_playoff_cover():
    W, H = 1200, 630
    RIGHT_X = W * 2 // 3   # 800
    RIGHT_W = W - RIGHT_X  # 400

    # Load cup
    cup_src = Image.open(os.path.join(COVERS, "cup.jpg")).convert("RGBA")
    src_w, src_h = cup_src.size   # 1116 × 999

    # Scale to FIT within right third (400 × 630) — maintain aspect ratio, no crop
    scale   = min(RIGHT_W / src_w, H / src_h)   # 400/1116=0.358 is the binding constraint
    fit_w   = int(src_w * scale)                  # 400
    fit_h   = int(src_h * scale)                  # 358

    cup_fit = cup_src.resize((fit_w, fit_h), Image.LANCZOS)

    # Centre in right-third region
    paste_x = RIGHT_X + (RIGHT_W - fit_w) // 2   # 800 + 0 = 800
    paste_y = (H - fit_h) // 2                    # (630-358)//2 = 136

    # Build base
    base = Image.new("RGB", (W, H), BG_T)
    base = Image.alpha_composite(
        base.convert("RGBA"), bracket_overlay(W, H)
    ).convert("RGB")

    base_rgba = base.convert("RGBA")
    base_rgba.paste(cup_fit, (paste_x, paste_y), cup_fit)
    base = base_rgba.convert("RGB")

    # Draw text (unchanged from original)
    draw = ImageDraw.Draw(base)
    L = 70
    draw.text((L, 72), "2025-26 PLAYOFF PREVIEW", font=fb(14), fill=ICE_T)
    y = 112
    draw.text((L, y), "ROUND 1",     font=fh(88), fill=TEXT_T); y += 96
    draw.text((L, y), "PREDICTIONS", font=fh(88), fill=TEXT_T); y += 100
    orange_bar_pil(draw, L, y, w=180, h=5); y += 28
    draw.text((L, y), "Three-pillar model. 16 teams. One upset pick.",
              font=fr(22), fill=TEXT_T)
    draw_wordmark_pil(draw, W, H)

    base.save(os.path.join(COVERS, "playoff_cover.png"))
    print(f"  ✓  playoff_cover.png  (cup: {fit_w}×{fit_h}, no crop)")


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2 — pillar_scorecard.png  (full rebuild)
# ═══════════════════════════════════════════════════════════════════════════════
def fix_pillar_scorecard():
    THR_NF = 15.2; THR_GA = 3.00; THR_ST = 50.9

    rows = [
        # (team, nf%, ga, st%, pillars, miss_note)
        ("COL", 29.1, 2.593, 50.9,  3, ""),
        ("CAR", 32.0, 2.963, 52.7,  3, ""),
        ("DAL", 37.1, 2.841, 54.5,  3, ""),
        ("TBL", 33.9, 2.866, 51.6,  3, ""),
        ("MIN", 37.6, 3.000, 52.5,  3, ""),
        ("BUF", 34.0, 3.000, 50.7,  2, "−0.2%"),
        ("PIT", 36.9, 3.378, 52.8,  2, ""),
        ("VGK", 33.3, 3.110, 53.0,  2, ""),
        ("EDM", 37.4, 3.358, 54.2,  2, ""),
        ("UTA", 35.8, 2.901, 49.1,  2, ""),
        ("MTL", 34.2, 3.207, 50.7,  1, ""),
        ("BOS", 31.0, 3.073, 50.2,  1, ""),
        ("OTT", 34.7, 3.012, 49.8,  1, ""),
        ("PHI", 36.4, 3.024, 46.6,  1, ""),
        ("ANA", 35.5, 3.568, 47.5,  1, ""),
        ("LAK", 33.4, 3.136, 45.8,  1, ""),
    ]

    # ── Canvas ────────────────────────────────────────────────────────────────
    W, H   = 780, 980
    ROW_H  = 44
    DIV_H  = 26   # section divider height

    img  = Image.new("RGB", (W, H), BG_T)
    draw = ImageDraw.Draw(img)

    # ── Fonts ─────────────────────────────────────────────────────────────────
    fnt_title = ImageFont.truetype(IMPACT,  20)
    fnt_hdr   = ImageFont.truetype(ARIALB, 13)
    fnt_team  = ImageFont.truetype(ARIALB, 14)
    fnt_val   = ImageFont.truetype(ARIALB, 13)
    fnt_note  = ImageFont.truetype(ARIAL,  10)
    fnt_badge = ImageFont.truetype(ARIALB, 16)
    fnt_sect  = ImageFont.truetype(ARIALB, 11)
    fnt_sub   = ImageFont.truetype(ARIAL,   9)
    fnt_wm    = ImageFont.truetype(ARIALB, 11)

    def tw(text, font):
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0]

    # Column layout: [left_x, width]  — Team | NF Rate | GA/gm | ST Score | Pillars
    COLS = [(28, 130), (158, 148), (306, 150), (456, 152), (608, 140)]

    def col_cx(i):
        return COLS[i][0] + COLS[i][1] // 2

    # ── Title ─────────────────────────────────────────────────────────────────
    title = "2025-26 Playoff Teams — Four-Pillar Model"
    draw.text(((W - tw(title, fnt_title)) // 2, 14),
              title, font=fnt_title, fill=TEXT_T)

    # ── Column headers ─────────────────────────────────────────────────────────
    HDR_Y = 50
    draw.rectangle([0, HDR_Y, W, HDR_Y + ROW_H], fill=SURF_T)

    headers  = ["Team", "NF Rate",   "GA / gm",            "ST Score",           "Pillars"]
    subhdr   = ["",     f"≥{THR_NF:.0f}%", f"≤{THR_GA:.2f}", f"≥{THR_ST:.0f}%", ""]
    for i, (hdr, sub) in enumerate(zip(headers, subhdr)):
        cx = col_cx(i)
        draw.text((cx - tw(hdr, fnt_hdr) // 2, HDR_Y + 6),
                  hdr, font=fnt_hdr, fill=TEXT_T)
        if sub:
            draw.text((cx - tw(sub, fnt_note) // 2, HDR_Y + 26),
                      sub, font=fnt_note, fill=(*ICE_T[:3], 180))

    # ── Data rows ─────────────────────────────────────────────────────────────
    # Colours
    RED   = (204,  51,  51)   # #CC3333 — missed / weak
    GREEN = ( 68, 170, 102)   # #44AA66 — 3 pillars
    GOLD  = (255, 183,   0)   # #FFB700 — 2 pillars

    SECT_BG    = BLUE_T                  # #2E7DC4 row for divider
    SECT_COLS  = {3: TEXT_T, 2: TEXT_T, 1: TEXT_T}
    BADGE_COLS = {3: GREEN,  2: GOLD,   1: RED}
    SECT_LABELS = {3: "3 PILLARS", 2: "2 PILLARS", 1: "1 PILLAR"}

    cur_y  = HDR_Y + ROW_H
    row_idx = 0
    section_drawn = {3: False, 2: False, 1: False}

    for (team, nf, ga, st, pillars, note) in rows:

        # Section divider
        if not section_drawn[pillars]:
            section_drawn[pillars] = True
            draw.rectangle([0, cur_y, W, cur_y + DIV_H], fill=SECT_BG)
            lbl = SECT_LABELS[pillars]
            draw.text(((W - tw(lbl, fnt_sect)) // 2, cur_y + 5),
                      lbl, font=fnt_sect, fill=TEXT_T)
            cur_y += DIV_H

        # Alternating row background
        row_bg = SURF_T if row_idx % 2 == 1 else BG_T
        draw.rectangle([0, cur_y, W, cur_y + ROW_H], fill=row_bg)
        draw.line([(0, cur_y), (W, cur_y)], fill=(*BLUE_T, 20), width=1)

        vcx = cur_y + ROW_H // 2   # vertical centre of row

        # Team name — always white bold
        draw.text((COLS[0][0] + 10, vcx - 9), team, font=fnt_team, fill=TEXT_T)

        # Helper: draw a value cell
        def draw_cell(col_idx, text, met, note_str=""):
            cx = col_cx(col_idx)
            color = TEXT_T if met else RED
            ty = vcx - (9 if not note_str else 13)
            draw.text((cx - tw(text, fnt_val) // 2, ty),
                      text, font=fnt_val, fill=color)
            if note_str:
                draw.text((cx - tw(note_str, fnt_note) // 2, ty + 16),
                          note_str, font=fnt_note, fill=RED)

        # NF Rate (all meet ≥ 15.2 in this dataset)
        draw_cell(1, f"{nf:.1f}%",  nf >= THR_NF)

        # GA/gm
        draw_cell(2, f"{ga:.3f}",   ga <= THR_GA)

        # ST Score — show BUF near-miss note
        st_note = note if (note and st < THR_ST) else ""
        draw_cell(3, f"{st:.1f}%",  st >= THR_ST, st_note)

        # Pillars badge
        bcol = BADGE_COLS[pillars]
        bx   = col_cx(4)
        draw.text((bx - tw(str(pillars), fnt_badge) // 2, vcx - 11),
                  str(pillars), font=fnt_badge, fill=bcol)

        cur_y  += ROW_H
        row_idx += 1

    # ── Footnote ──────────────────────────────────────────────────────────────
    fn_y = cur_y + 10
    fn   = f"Thresholds (top-third of 32 teams): NF ≥{THR_NF}%  ·  GA/gm ≤{THR_GA:.2f}  ·  ST ≥{THR_ST}%"
    draw.text(((W - tw(fn, fnt_note)) // 2, fn_y), fn, font=fnt_note, fill=(*ICE_T[:3], 160))

    # ── Watermark ─────────────────────────────────────────────────────────────
    draw.text((W - tw("HockeyROI", fnt_wm) - 14, H - 26),
              "HockeyROI", font=fnt_wm, fill=(*ICE_T[:3], 220))

    img.save(os.path.join(PLY, "pillar_scorecard.png"))
    print("  ✓  pillar_scorecard.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3 — special_teams_rankings.png  (x-axis starts at 0)
# ═══════════════════════════════════════════════════════════════════════════════
def fix_special_teams():
    data = [
        ("LAK", 45.8), ("PHI", 46.6), ("ANA", 47.5), ("UTA", 49.1),
        ("OTT", 49.8), ("BOS", 50.2), ("MTL", 50.7), ("BUF", 50.7),
        ("COL", 50.9), ("TBL", 51.6), ("MIN", 52.5), ("CAR", 52.7),
        ("PIT", 52.8), ("VGK", 53.0), ("EDM", 54.2), ("DAL", 54.5),
    ]
    THR    = 50.9
    teams  = [d[0] for d in data]
    scores = [d[1] for d in data]
    # Gold above threshold, blue below — matches existing style
    colors = ["#FFB700" if s >= THR else BLUE for s in scores]

    fig, ax = plt.subplots(figsize=(11.5, 9.8))
    style_ax(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=BLUE, alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)

    # Bars start at 0 — no 'left=' offset
    bars = ax.barh(teams, scores, color=colors, height=0.62,
                   edgecolor="none", zorder=3)

    # Value labels
    for bar, val in zip(bars, scores):
        ax.text(val + 0.18, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=11, color=TEXT, fontproperties=arlb_prop)

    # Threshold line
    ax.axvline(THR, color=ICE_BLUE, lw=1.6, ls="--", zorder=5)
    ax.text(THR + 0.22, len(teams) - 0.3,
            f"Top-third threshold\n{THR:.1f}%",
            fontsize=8.5, color=ICE_BLUE, va="top", fontproperties=arl_prop)

    # X-axis from 0
    ax.set_xlim(0, max(scores) + 3.0)
    ax.tick_params(axis="y", labelsize=12, left=False)
    ax.tick_params(axis="x", labelsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_alpha(0.3)

    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(arlb_prop)
        lbl.set_color(TEXT)

    handles = [
        mpatches.Patch(color="#FFB700", label="Meets top-third threshold"),
        mpatches.Patch(color=BLUE,      label="Below threshold"),
    ]
    legend = ax.legend(handles=handles, fontsize=10, loc="lower right",
                       framealpha=0.9, edgecolor=BLUE)
    for txt in legend.get_texts():
        txt.set_color(TEXT)

    fig.text(0.5, 0.975, "Special Teams Score — 2025-26 Playoff Teams",
             ha="center", va="top", fontsize=20, color=TEXT, fontproperties=imp_prop)
    fig.text(0.5, 0.935,
             "Average of PP% and PK%  |  Gold = meets top-third threshold",
             ha="center", va="top", fontsize=11, color=ICE_BLUE, fontproperties=arl_prop)
    wm_mpl(fig)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(PLY, "special_teams_rankings.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓  special_teams_rankings.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    fix_playoff_cover()
    fix_pillar_scorecard()
    fix_special_teams()
    print("\nDone.\n")
