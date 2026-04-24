#!/usr/bin/env python3
"""
HockeyROI — Dostal cover card (16:9, 1200×675px)
Player photo fades into dark background with a horizontal gradient mask.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
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

# ─── PALETTE ───────────────────────────────────────────────────────────────────
BG     = '#0B1D2E'
BLUE   = '#2E7DC4'
WHITE  = '#F0F4F8'
GREY   = '#888888'
ORANGE = '#FF8C00'

IMG_PATH = ("/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/"
            "NHL analysis/Charts/Covers/Dostal.jpeg")
OUT_PATH = ("/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/"
            "NHL analysis/Charts/Dostal/dostal_cover_card.png")

W_PX, H_PX = 1200, 675
DPI = 150

# ─── LOAD & PREP PLAYER IMAGE ──────────────────────────────────────────────────
img_pil = Image.open(IMG_PATH).convert("RGBA")

# Target: fill right 40% of card at full card height
img_w_px  = int(W_PX * 0.50)   # slightly wider to give fade room
img_h_px  = H_PX

# Resize preserving aspect, then centre-crop to target box
src_w, src_h = img_pil.size
scale = max(img_w_px / src_w, img_h_px / src_h)
new_w = int(src_w * scale)
new_h = int(src_h * scale)
img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)

# Centre-crop
left   = (new_w - img_w_px) // 2
top    = max(0, new_h - img_h_px)   # anchor to bottom
right  = left + img_w_px
bottom = top  + img_h_px
img_pil = img_pil.crop((left, top, right, bottom))

# Apply horizontal gradient mask: left edge fully transparent → right fully opaque
# Fade zone: left 45% of the cropped image
img_arr = np.array(img_pil).astype(float)   # H × W × 4
fade_px = int(img_w_px * 0.45)

alpha_mask = np.ones(img_w_px, dtype=float)
alpha_mask[:fade_px] = np.linspace(0.0, 1.0, fade_px)
# Apply mask to alpha channel
img_arr[:, :, 3] *= alpha_mask[np.newaxis, :]
img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
img_faded = Image.fromarray(img_arr, 'RGBA')

# ─── COMPOSE FIGURE ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_PX)
ax.set_ylim(0, H_PX)
ax.set_facecolor(BG)
ax.axis('off')

# Place the faded player image anchored to the right-bottom corner
# imshow extent: [xmin, xmax, ymin, ymax]  (ymin < ymax = bottom-up)
img_x0 = W_PX - img_w_px
ax.imshow(img_faded, extent=[img_x0, W_PX, 0, H_PX],
          aspect='auto', origin='upper', zorder=2)

# Subtle vignette overlay so text side stays dark
grad = np.zeros((H_PX, W_PX, 4), dtype=np.uint8)
vignette_w = int(W_PX * 0.68)
alpha_vig = np.linspace(210, 0, vignette_w, dtype=np.uint8)
bg_rgb = tuple(int(BG.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
grad[:, :vignette_w, 0] = bg_rgb[0]
grad[:, :vignette_w, 1] = bg_rgb[1]
grad[:, :vignette_w, 2] = bg_rgb[2]
grad[:, :vignette_w, 3] = alpha_vig[np.newaxis, :]
vignette_img = Image.fromarray(grad, 'RGBA')
ax.imshow(vignette_img, extent=[0, W_PX, 0, H_PX],
          aspect='auto', origin='upper', zorder=3)

# ─── TEXT CONTENT (left side) ─────────────────────────────────────────────────
# Data units: x 0–1200, y 0–675 (y=0 bottom)
tx = 72    # left margin

# @HockeyROI tag — near top
ax.text(tx, 610, '@HockeyROI', color=GREY, fontsize=11,
        fontname=BEBAS, va='center', zorder=6)

# Main title — two lines, generously spaced
ax.text(tx, 530, 'HOW TO BEAT', color=WHITE, fontsize=54,
        fontname=BEBAS, fontweight='bold', va='center', zorder=6)
ax.text(tx, 440, 'LUKAS DOSTAL', color=WHITE, fontsize=54,
        fontname=BEBAS, fontweight='bold', va='center', zorder=6)

# Subtitle — clear gap below title
ax.text(tx, 362, 'ROUND 1 SCOUTING REPORT', color=BLUE, fontsize=22,
        fontname=BEBAS, va='center', zorder=6)

# Thin divider line
ax.plot([tx, tx + 420], [332, 332], color=BLUE, linewidth=1.0,
        alpha=0.80, zorder=6)

# Series info — below divider
ax.text(tx, 302, 'EDM vs ANA  |  Game 1  |  Monday April 20',
        color=GREY, fontsize=11, fontname=INTER, va='center', zorder=6)

# Substack handle — bottom
ax.text(tx, 52, 'hockeyROI.substack.com',
        color=GREY, fontsize=10, fontname=INTER,
        style='italic', va='center', zorder=6)

# ─── SAVE ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor=BG,
            pad_inches=0)
plt.close()
print(f"Cover card saved: {OUT_PATH}")
