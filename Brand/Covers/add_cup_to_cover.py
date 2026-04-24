#!/usr/bin/env python3
"""Rebuild playoff_cover.png — adds cup.jpg to the right third."""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE    = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT_DIR = os.path.join(BASE, "Charts", "covers")
CUP_IMG = os.path.join(OUT_DIR, "cup.jpg")

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = (11,  29,  46)
SURFACE  = (27,  58,  92)
BLUE     = (46, 125, 196)
ICE_BLUE = (74, 179, 232)
ORANGE   = (255, 107,  53)
TEXT     = (240, 244, 248)

IMPACT = "/System/Library/Fonts/Supplemental/Impact.ttf"
ARIAL  = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIALB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

def fh(size): return ImageFont.truetype(IMPACT, size)
def fb(size): return ImageFont.truetype(ARIALB, size)
def fr(size): return ImageFont.truetype(ARIAL, size)

def text_w(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]

def draw_wordmark(draw, W, H, size=26):
    font = fh(size)
    t = "HOCKEYROI"
    x = W - text_w(draw, t, font) - 44
    draw.text((x, H - 48), t, font=font, fill=ICE_BLUE)

def orange_bar(draw, x, y, w=160, h=5):
    draw.rectangle([x, y, x + w, y + h], fill=ORANGE)

def bracket_overlay(W, H, opacity=16):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c = (*BLUE, opacity)

    bx = 580
    col_w, col_h = 120, 160
    gap = 40

    for row in range(4):
        y0 = 70 + row * (col_h + gap)
        y1 = y0 + col_h
        d.rounded_rectangle([bx, y0, bx + col_w, y1], radius=8, outline=c, width=2)
        mid_y = (y0 + y1) // 2
        if row % 2 == 0 and row + 1 < 4:
            next_y0 = 70 + (row + 1) * (col_h + gap)
            next_mid = (next_y0 + next_y0 + col_h) // 2
            c2 = (*BLUE, opacity - 4)
            d.line([(bx + col_w, mid_y), (bx + col_w + 80, mid_y)], fill=c2, width=2)
            d.line([(bx + col_w + 80, mid_y), (bx + col_w + 80, next_mid)], fill=c2, width=2)

    bx2 = bx + col_w + 160
    for row in range(2):
        y0 = 70 + row * (2 * (col_h + gap) + gap // 2)
        y1 = y0 + col_h
        d.rounded_rectangle([bx2, y0, bx2 + col_w, y1], radius=8, outline=c, width=2)

    # trophy silhouette hint
    tx, ty = W - 160, 140
    d.rounded_rectangle([tx, ty, tx + 90, ty + 130], radius=10, outline=c, width=2)
    d.rectangle([tx + 25, ty + 130, tx + 65, ty + 170], outline=c, width=2)
    d.line([(tx + 10, ty + 170), (tx + 80, ty + 170)], fill=c, width=3)

    return layer


def make_playoff_cover():
    W, H = 1200, 630

    # ── Right-third dimensions ─────────────────────────────────────────────────
    RIGHT_X = W * 2 // 3   # 800  — left edge of cup region
    RIGHT_W = W - RIGHT_X  # 400  — width of cup region

    # ── 1. Load and scale cup image to card height ─────────────────────────────
    cup_src = Image.open(CUP_IMG).convert("RGBA")
    src_w, src_h = cup_src.size                    # 1116 × 999

    scale   = H / src_h                            # 630 / 999 ≈ 0.630
    scaled_w = int(src_w * scale)                  # ≈ 703
    scaled_h = H                                   # 630

    cup_scaled = cup_src.resize((scaled_w, scaled_h), Image.LANCZOS)

    # ── 2. Crop to RIGHT_W wide (centre-crop to keep trophy centred) ──────────
    if scaled_w > RIGHT_W:
        crop_l = (scaled_w - RIGHT_W) // 2
        cup_crop = cup_scaled.crop((crop_l, 0, crop_l + RIGHT_W, scaled_h))
    else:
        # Cup narrower than region — pad left with BG
        pad = Image.new("RGBA", (RIGHT_W, H), (*BG, 255))
        pad.paste(cup_scaled, ((RIGHT_W - scaled_w) // 2, 0), cup_scaled)
        cup_crop = pad

    # ── 3. No overlay, no fade — use the cropped image as-is ─────────────────
    cup_final = cup_crop

    # ── 5. Build base (BG + bracket decoration) ────────────────────────────────
    base = Image.new("RGB", (W, H), BG)
    base = Image.alpha_composite(
        base.convert("RGBA"), bracket_overlay(W, H)
    ).convert("RGB")

    # ── 6. Composite cup onto right third ─────────────────────────────────────
    base_rgba = base.convert("RGBA")
    base_rgba.paste(cup_final, (RIGHT_X, 0), cup_final)
    base = base_rgba.convert("RGB")

    # ── 7. Draw all text (unchanged from original) ────────────────────────────
    draw = ImageDraw.Draw(base)
    L    = 70

    draw.text((L, 72), "2025-26 PLAYOFF PREVIEW", font=fb(14), fill=ICE_BLUE)

    y = 112
    draw.text((L, y), "ROUND 1",     font=fh(88), fill=TEXT)
    y += 96
    draw.text((L, y), "PREDICTIONS", font=fh(88), fill=TEXT)
    y += 100

    orange_bar(draw, L, y, w=180, h=5)
    y += 28

    draw.text((L, y), "Four-pillar model. 16 teams. One upset pick.",
              font=fr(22), fill=TEXT)

    draw_wordmark(draw, W, H)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    out = os.path.join(OUT_DIR, "playoff_cover.png")
    base.save(out)
    print(f"  ✓  playoff_cover.png saved → {out}")


make_playoff_cover()
