#!/usr/bin/env python3
"""
HockeyROI — Social Media Asset Generator
Generates 6 branded assets → Charts/covers/

Fonts used:
  Headlines : Impact  (Bebas Neue fallback per brand.md)
  Body/labels: Arial Bold / Arial  (Inter fallback per brand.md)
"""

import os
import math
from PIL import Image, ImageDraw, ImageFont

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE    = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT_DIR = os.path.join(BASE, "Charts", "covers")
os.makedirs(OUT_DIR, exist_ok=True)

# ── PALETTE ───────────────────────────────────────────────────────────────────
BG       = (11,  29,  46)    # #0B1D2E
SURFACE  = (27,  58,  92)    # #1B3A5C
BLUE     = (46, 125, 196)    # #2E7DC4
ICE_BLUE = (74, 179, 232)    # #4AB3E8
ORANGE   = (255, 107,  53)   # #FF6B35
TEXT     = (240, 244, 248)   # #F0F4F8

# ── FONT HELPERS ──────────────────────────────────────────────────────────────
IMPACT = "/System/Library/Fonts/Supplemental/Impact.ttf"
ARIAL  = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIALB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

def fh(size):
    """Headline font (Impact)."""
    return ImageFont.truetype(IMPACT, size)

def fb(size):
    """Body bold font (Arial Bold)."""
    return ImageFont.truetype(ARIALB, size)

def fr(size):
    """Body regular font (Arial)."""
    return ImageFont.truetype(ARIAL, size)

# ── DRAW UTILITIES ────────────────────────────────────────────────────────────
def text_w(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]

def text_h(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[3] - bb[1]

def draw_centered(draw, y, text, font, color, canvas_w):
    x = (canvas_w - text_w(draw, text, font)) // 2
    draw.text((x, y), text, font=font, fill=color)

def draw_wordmark(draw, W, H, size=26):
    font = fh(size)
    t = "HOCKEYROI"
    x = W - text_w(draw, t, font) - 44
    draw.text((x, H - 48), t, font=font, fill=ICE_BLUE)

def orange_bar(draw, x, y, w=160, h=5):
    draw.rectangle([x, y, x + w, y + h], fill=ORANGE)

def url_bar(draw, W, H, font_size=13):
    font = fb(font_size)
    t = "HOCKEYROI.SUBSTACK.COM"
    draw_centered(draw, H - 44, t, font, ICE_BLUE, W)

# ── BACKGROUND DECORATIONS ───────────────────────────────────────────────────
def rink_overlay(W, H, offset_x=520, opacity=18):
    """Draw a faint rink outline on the right half."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c = (*BLUE, opacity)

    rx1, ry1 = offset_x, 60
    rx2, ry2 = W - 40, H - 60
    corner   = 70

    # Rink outline (3 nested)
    for i in range(3):
        d.rounded_rectangle(
            [rx1 + i*18, ry1 + i*12, rx2 - i*18, ry2 - i*12],
            radius=corner - i*8, outline=c, width=2,
        )

    cx = (rx1 + rx2) // 2
    cy = (ry1 + ry2) // 2

    # Center red line
    d.line([(cx, ry1 + 20), (cx, ry2 - 20)], fill=c, width=3)

    # Blue lines
    for bx in [rx1 + (rx2 - rx1) // 3, rx1 + 2 * (rx2 - rx1) // 3]:
        d.line([(bx, ry1 + 20), (bx, ry2 - 20)], fill=c, width=2)

    # Center circle + dot
    r = 65
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=c, width=2)
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=c)

    # Crease semicircle (goal end right)
    d.arc([rx2 - 120, cy - 55, rx2 - 20, cy + 55], start=90, end=270, fill=c, width=2)

    return layer


def bracket_overlay(W, H, opacity=16):
    """Draw faint bracket shapes on the right side."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c = (*BLUE, opacity)

    # Two bracket 'columns'
    bx = 580
    col_w, col_h = 120, 160
    gap = 40

    for row in range(4):
        y0 = 70 + row * (col_h + gap)
        y1 = y0 + col_h
        # Left bracket pair
        d.rounded_rectangle([bx, y0, bx + col_w, y1],
                             radius=8, outline=c, width=2)
        # Connector line to right
        mid_y = (y0 + y1) // 2
        if row % 2 == 0 and row + 1 < 4:
            next_y0 = 70 + (row + 1) * (col_h + gap)
            next_mid = (next_y0 + next_y0 + col_h) // 2
            c2 = (*BLUE, opacity - 4)
            d.line([(bx + col_w, mid_y), (bx + col_w + 80, mid_y)], fill=c2, width=2)
            d.line([(bx + col_w + 80, mid_y), (bx + col_w + 80, next_mid)], fill=c2, width=2)

    # Second column further right
    bx2 = bx + col_w + 160
    for row in range(2):
        y0 = 70 + row * (2 * (col_h + gap) + gap // 2)
        y1 = y0 + col_h
        d.rounded_rectangle([bx2, y0, bx2 + col_w, y1],
                             radius=8, outline=c, width=2)

    # Trophy silhouette hint (very subtle) far right
    tx, ty = W - 160, 140
    d.rounded_rectangle([tx, ty, tx + 90, ty + 130], radius=10, outline=c, width=2)
    d.rectangle([tx + 25, ty + 130, tx + 65, ty + 170], outline=c, width=2)
    d.line([(tx + 10, ty + 170), (tx + 80, ty + 170)], fill=c, width=3)

    return layer


# ─────────────────────────────────────────────────────────────────────────────
#  ASSET 1 — geometry_cover.png  1200 × 630
# ─────────────────────────────────────────────────────────────────────────────
def make_geometry_cover():
    W, H = 1200, 630
    base = Image.new("RGB", (W, H), BG)
    base = Image.alpha_composite(base.convert("RGBA"),
                                  rink_overlay(W, H)).convert("RGB")
    draw = ImageDraw.Draw(base)

    L = 70

    # Category label
    draw.text((L, 72), "NET-FRONT ANALYSIS", font=fb(14), fill=ICE_BLUE)

    # Headline
    y = 112
    draw.text((L, y), "THE GEOMETRY", font=fh(88), fill=TEXT)
    y += 96
    draw.text((L, y), "OF WINNING", font=fh(88), fill=TEXT)
    y += 100

    # Orange accent bar
    orange_bar(draw, L, y, w=180, h=5)
    y += 28

    # Subtext
    draw.text((L, y), "921,000 NHL shots. One finding that changes everything.",
              font=fr(22), fill=TEXT)

    draw_wordmark(draw, W, H)

    out = os.path.join(OUT_DIR, "geometry_cover.png")
    base.save(out)
    print(f"  ✓  geometry_cover.png")


# ─────────────────────────────────────────────────────────────────────────────
#  ASSET 2 — rebound_stat_card.png  1080 × 1080
# ─────────────────────────────────────────────────────────────────────────────
def make_rebound_stat_card():
    W, H = 1080, 1080
    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Subtle radial glow behind number
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    for r in range(280, 0, -8):
        alpha = int(22 * (1 - r / 280))
        gd.ellipse([(W // 2 - r, H // 2 - r - 60),
                    (W // 2 + r, H // 2 + r - 60)],
                   fill=(*ORANGE, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Top label
    label = "REBOUND SHOTS  ·  GOAL RATE MULTIPLIER"
    draw_centered(draw, 88, label, fb(13), ICE_BLUE, W)

    # Thin divider
    draw.rectangle([(W // 2 - 200, 124), (W // 2 + 200, 126)],
                   fill=(*SURFACE, ))

    # Hero number
    hero_font = fh(210)
    hero      = "2.00×"
    hx = (W - text_w(draw, hero, hero_font)) // 2
    draw.text((hx, 340), hero, font=hero_font, fill=ORANGE)

    # Context line
    ctx = "vs 1.00× baseline  |  921,141 even strength shots"
    draw_centered(draw, 610, ctx, fr(22), TEXT, W)

    # Thin bottom divider
    draw.rectangle([(W // 2 - 300, 680), (W // 2 + 300, 682)],
                   fill=(*SURFACE,))

    url_bar(draw, W, H)

    out = os.path.join(OUT_DIR, "rebound_stat_card.png")
    img.save(out)
    print(f"  ✓  rebound_stat_card.png")


# ─────────────────────────────────────────────────────────────────────────────
#  ASSET 3 — playoff_cover.png  1200 × 630
# ─────────────────────────────────────────────────────────────────────────────
def make_playoff_cover():
    W, H = 1200, 630
    base = Image.new("RGB", (W, H), BG)
    base = Image.alpha_composite(base.convert("RGBA"),
                                  bracket_overlay(W, H)).convert("RGB")
    draw = ImageDraw.Draw(base)

    L = 70

    draw.text((L, 72), "2025-26 PLAYOFF PREVIEW", font=fb(14), fill=ICE_BLUE)

    y = 112
    draw.text((L, y), "ROUND 1", font=fh(88), fill=TEXT)
    y += 96
    draw.text((L, y), "PREDICTIONS", font=fh(88), fill=TEXT)
    y += 100

    orange_bar(draw, L, y, w=180, h=5)
    y += 28

    draw.text((L, y), "Four-pillar model. 16 teams. One upset pick.",
              font=fr(22), fill=TEXT)

    draw_wordmark(draw, W, H)

    out = os.path.join(OUT_DIR, "playoff_cover.png")
    base.save(out)
    print(f"  ✓  playoff_cover.png")


# ─────────────────────────────────────────────────────────────────────────────
#  ASSET 4 — upset_stat_card.png  1080 × 1080
# ─────────────────────────────────────────────────────────────────────────────
def make_upset_stat_card():
    W, H = 1080, 1080
    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Glow
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    for r in range(260, 0, -8):
        alpha = int(20 * (1 - r / 260))
        gd.ellipse([(W // 2 - r, H // 2 - r - 100),
                    (W // 2 + r, H // 2 + r - 100)],
                   fill=(*ORANGE, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Label
    draw_centered(draw, 88, "MODEL UPSET PICK  ·  ROUND 1", fb(13), ICE_BLUE, W)

    # Hero team name
    hero_font = fh(200)
    hero = "UTAH"
    draw_centered(draw, 300, hero, hero_font, ORANGE, W)

    # "over Vegas" line
    draw_centered(draw, 530, "over Vegas", fh(60), TEXT, W)

    # Thin divider
    draw.rectangle([(W // 2 - 260, 618), (W // 2 + 260, 620)],
                   fill=SURFACE)

    # Context
    ctx = "2 pillars vs 2 pillars  |  Goals against wins the tiebreaker"
    draw_centered(draw, 644, ctx, fr(20), ICE_BLUE, W)

    url_bar(draw, W, H)

    out = os.path.join(OUT_DIR, "upset_stat_card.png")
    img.save(out)
    print(f"  ✓  upset_stat_card.png")


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED: X thread opener card builder
# ─────────────────────────────────────────────────────────────────────────────
def make_x_thread_card(W, H, tweet_text, out_filename):
    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Card
    PAD    = 52
    CARD_W = W - PAD * 2
    CARD_H = H - PAD * 2
    CARD_X = PAD
    CARD_Y = PAD
    R      = 20

    draw.rounded_rectangle(
        [CARD_X, CARD_Y, CARD_X + CARD_W, CARD_Y + CARD_H],
        radius=R, fill=SURFACE,
    )

    # Subtle card border
    draw.rounded_rectangle(
        [CARD_X, CARD_Y, CARD_X + CARD_W, CARD_Y + CARD_H],
        radius=R, outline=(*BLUE, 80), width=1,
    )

    # Avatar circle
    AV_X, AV_Y, AV_R = CARD_X + 44, CARD_Y + 44, 32
    draw.ellipse(
        [AV_X - AV_R, AV_Y - AV_R, AV_X + AV_R, AV_Y + AV_R],
        fill=BLUE,
    )
    init_font = fh(32)
    ix = AV_X - text_w(draw, "H", init_font) // 2
    iy = AV_Y - text_h(draw, "H", init_font) // 2 - 2
    draw.text((ix, iy), "H", font=init_font, fill=TEXT)

    # Username
    draw.text((CARD_X + 44 + AV_R + 14, CARD_Y + 44 - 10),
              "@HockeyROI", font=fb(16), fill=TEXT)
    draw.text((CARD_X + 44 + AV_R + 14, CARD_Y + 44 + 12),
              "HockeyROI", font=fr(13), fill=(*ICE_BLUE,))

    # Thread counter top right
    counter_font = fb(14)
    ct = "1/8"
    draw.text((CARD_X + CARD_W - text_w(draw, ct, counter_font) - 24,
               CARD_Y + 44 - 8),
              ct, font=counter_font, fill=ICE_BLUE)

    # Divider under header
    div_y = CARD_Y + 100
    draw.rectangle([CARD_X + 24, div_y, CARD_X + CARD_W - 24, div_y + 1],
                   fill=(*BLUE, 60))

    # Tweet text — word-wrap to ~56 chars per line
    tweet_font = fr(24)
    words      = tweet_text.split()
    lines      = []
    current    = ""
    for w in words:
        test = f"{current} {w}".strip()
        if text_w(draw, test, tweet_font) > CARD_W - 80:
            if current:
                lines.append(current)
            current = w
        else:
            current = test
    if current:
        lines.append(current)

    ty = div_y + 26
    LH = 38
    for line in lines:
        draw.text((CARD_X + 40, ty), line, font=tweet_font, fill=TEXT)
        ty += LH

    # Bottom bar
    BAR_H  = 52
    bar_y  = CARD_Y + CARD_H - BAR_H
    draw.rounded_rectangle(
        [CARD_X, bar_y, CARD_X + CARD_W, CARD_Y + CARD_H],
        radius=R, fill=(*BG, 255),
    )
    # Cover the top edge of the rounded bar (it's inside the card)
    draw.rectangle(
        [CARD_X, bar_y, CARD_X + CARD_W, bar_y + R],
        fill=BG,
    )

    bar_label = "HOCKEYROI.COM  ·  DATA-DRIVEN NHL ANALYTICS"
    bar_font  = fb(12)
    draw_centered(draw, bar_y + (BAR_H - text_h(draw, bar_label, bar_font)) // 2,
                  bar_label, bar_font, ICE_BLUE, W)

    out = os.path.join(OUT_DIR, out_filename)
    img.save(out)
    print(f"  ✓  {out_filename}")


# ─────────────────────────────────────────────────────────────────────────────
#  ASSET 5 — x_thread_opener_geometry.png  1200 × 675
# ─────────────────────────────────────────────────────────────────────────────
def make_x_thread_geometry():
    make_x_thread_card(
        W=1200, H=675,
        tweet_text=(
            "Modern basketball solved offense with drive and kick. "
            "The data says hockey has an equivalent play. "
            "Nobody does it deliberately enough. 🧵"
        ),
        out_filename="x_thread_opener_geometry.png",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ASSET 6 — x_thread_opener_playoff.png  1200 × 675
# ─────────────────────────────────────────────────────────────────────────────
def make_x_thread_playoff():
    make_x_thread_card(
        W=1200, H=675,
        tweet_text=(
            "Round 1 playoff preview — 6 seasons of NHL data says which teams "
            "are built to go deep. Five three-pillar teams. One upset pick. 🧵"
        ),
        out_filename="x_thread_opener_playoff.png",
    )


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nHockeyROI — Generating social media assets → Charts/covers/\n")
    make_geometry_cover()
    make_rebound_stat_card()
    make_playoff_cover()
    make_upset_stat_card()
    make_x_thread_geometry()
    make_x_thread_playoff()
    print(f"\nDone. 6 assets saved to {OUT_DIR}\n")
