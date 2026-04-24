#!/usr/bin/env python3
"""
1. Redraw hockeyroi_banner.png at a narrower width so it fits the Substack
   theme cover image slot (less horizontal whitespace around the wordmark).
2. Re-render dostal_cover_card_v2.png using the same shift-right approach
   applied to the other 6 covers, so Substack's thumbnail crop doesn't clip
   the leftmost text.
"""

import os
from PIL import Image, ImageDraw, ImageFont

BASE = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT_DIR = os.path.join(BASE, "Brand", "Covers", "New")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Narrow banner
# ---------------------------------------------------------------------------
BG = (255, 255, 255)
DARK_BLUE = (27, 58, 92)
ORANGE = (255, 107, 53)

# Substack theme cover: 3:1 aspect, at least 1200px wide.
COVER_W, COVER_H = 1200, 400
FONT_SIZE = 140
GAP = 32
DIV_W = 6
SAFETY = 10
# PAD is unused for the final cover but kept so the original inner-render
# code below still measures correctly.
PAD = 0
H = None  # computed from wordmark

try:
    font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", FONT_SIZE)
except IOError:
    font = ImageFont.load_default()

# Measure each word by rendering it to a scratch canvas and taking its
# actual non-transparent bounds — this catches right-side bearing that
# textbbox at origin can undercount.
def measured_size(text, color):
    scratch = Image.new("RGBA", (FONT_SIZE * len(text) * 2, FONT_SIZE * 3),
                        (0, 0, 0, 0))
    ImageDraw.Draw(scratch).text((FONT_SIZE, FONT_SIZE), text,
                                 font=font, fill=color)
    bbox = scratch.getbbox()
    return scratch, bbox  # bbox = (l, t, r, b) of rendered pixels

h_img, h_bbox = measured_size("HOCKEY", DARK_BLUE + (255,))
r_img, r_bbox = measured_size("ROI",    ORANGE    + (255,))
hw = h_bbox[2] - h_bbox[0]
hh = h_bbox[3] - h_bbox[1]
rw = r_bbox[2] - r_bbox[0]
rh = r_bbox[3] - r_bbox[1]

# Build the wordmark on a tight transparent canvas, then center it on the
# 1200x400 Substack cover.
content_w = hw + GAP + DIV_W + GAP + rw + SAFETY
content_h = max(hh, rh)
mark = Image.new("RGBA", (content_w, content_h), (0, 0, 0, 0))
mdraw = ImageDraw.Draw(mark)

mark.paste(h_img.crop(h_bbox), (0, (content_h - hh) // 2),
           h_img.crop(h_bbox))
div_x = hw + GAP
mdraw.rectangle(
    [div_x, int(content_h * 0.12),
     div_x + DIV_W, int(content_h * 0.88)],
    fill=ORANGE,
)
r_x = div_x + DIV_W + GAP
mark.paste(r_img.crop(r_bbox), (r_x, (content_h - rh) // 2),
           r_img.crop(r_bbox))

img = Image.new("RGBA", (COVER_W, COVER_H), BG + (255,))
paste_x = (COVER_W - content_w) // 2
paste_y = (COVER_H - content_h) // 2
img.paste(mark, (paste_x, paste_y), mark)
W, H = COVER_W, COVER_H

banner_out = os.path.join(OUT_DIR, "hockeyroi_banner.png")
img.save(banner_out, "PNG")
print(f"banner: {W}x{H} -> {banner_out}")


# ---------------------------------------------------------------------------
# 2. Shift Dostal cover right (same as the other 6)
# ---------------------------------------------------------------------------
W_PX, H_PX = 1200, 675
SIDE_MARGIN = 140
BG_RGB = (11, 29, 46)  # brand dark

src = os.path.join(BASE, "Goalies All", "Dostal", "Images",
                   "dostal_cover_card_v2.png")
dostal = Image.open(src).convert("RGB")
sw, sh = dostal.size
target_w = W_PX - 2 * SIDE_MARGIN
scale = target_w / sw
new_w = target_w
new_h = int(sh * scale)
small = dostal.resize((new_w, new_h), Image.LANCZOS)

canvas = Image.new("RGB", (W_PX, H_PX), BG_RGB)
canvas.paste(small, (SIDE_MARGIN, (H_PX - new_h) // 2))

dostal_out = os.path.join(OUT_DIR, "dostal_cover_card_v2.png")
canvas.save(dostal_out)
print(f"dostal: {W_PX}x{H_PX} -> {dostal_out}")
