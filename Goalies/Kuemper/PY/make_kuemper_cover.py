#!/usr/bin/env python3
"""
HockeyROI — Kuemper cover card (16:9, 1200x675px)
Player photo fades into dark background with a horizontal gradient mask.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from PIL import Image
import os

# --- FONTS -------------------------------------------------------------------
BEBAS_PATH = "/tmp/BebasNeue-Regular.ttf"
INTER_PATH = "/tmp/Inter-Regular.ttf"

def load_font(path, fallback="Arial"):
    if os.path.exists(path):
        fm.fontManager.addfont(path)
        return fm.FontProperties(fname=path).get_name()
    return fallback

BEBAS = load_font(BEBAS_PATH, "Arial Black")
INTER = load_font(INTER_PATH, "Arial")

# --- PALETTE -----------------------------------------------------------------
BG    = '#0B1D2E'
BLUE  = '#2E7DC4'
WHITE = '#F0F4F8'
GREY  = '#888888'

BASE = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
IMG_PATH = os.path.join(BASE, "Covers", "Kuemper.jpg")
OUT_PATH = os.path.join(BASE, "Goalies All", "Kuemper", "Images", "kuemper_cover.png")

W_PX, H_PX = 1200, 675
DPI = 150

# --- LOAD & PREP PLAYER IMAGE ------------------------------------------------
img_pil = Image.open(IMG_PATH).convert("RGBA")

img_w_px = int(W_PX * 0.40)
img_h_px = H_PX

src_w, src_h = img_pil.size
scale = max(img_w_px / src_w, img_h_px / src_h)
new_w = int(src_w * scale)
new_h = int(src_h * scale)
img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)

left   = (new_w - img_w_px) // 2
top    = max(0, new_h - img_h_px)
right  = left + img_w_px
bottom = top  + img_h_px
img_pil = img_pil.crop((left, top, right, bottom))

# Horizontal gradient mask: left edge transparent -> right edge opaque
img_arr = np.array(img_pil).astype(float)
fade_px = int(img_w_px * 0.45)
alpha_mask = np.ones(img_w_px, dtype=float)
alpha_mask[:fade_px] = np.linspace(0.0, 1.0, fade_px)
img_arr[:, :, 3] *= alpha_mask[np.newaxis, :]
img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
img_faded = Image.fromarray(img_arr, 'RGBA')

# --- COMPOSE FIGURE ----------------------------------------------------------
fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_PX)
ax.set_ylim(0, H_PX)
ax.set_facecolor(BG)
ax.axis('off')

img_x0 = W_PX - img_w_px
ax.imshow(img_faded, extent=[img_x0, W_PX, 0, H_PX],
          aspect='auto', origin='upper', zorder=2)

# Vignette so text side stays dark
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

# --- TEXT CONTENT (left side) ------------------------------------------------
tx = 72

ax.text(tx, 575, '@HockeyROI', color=GREY, fontsize=11,
        fontname=BEBAS, va='center', zorder=6)

ax.text(tx, 510, 'HOW TO BEAT', color=WHITE, fontsize=32,
        fontname=BEBAS, fontweight='bold', va='center', zorder=6)
ax.text(tx, 415, 'DARCY', color=WHITE, fontsize=42,
        fontname=BEBAS, fontweight='bold', va='center', zorder=6)
ax.text(tx, 315, 'KUEMPER', color=WHITE, fontsize=42,
        fontname=BEBAS, fontweight='bold', va='center', zorder=6)

ax.text(tx, 220, 'WEAKNESS ANALYSIS', color=BLUE, fontsize=20,
        fontname=BEBAS, va='center', zorder=6)

ax.plot([tx, tx + 340], [192, 192], color=BLUE, linewidth=1.0,
        alpha=0.80, zorder=6)

ax.text(tx, 165, '132 Games  |  3,466 Shots  |  3 Seasons',
        color=GREY, fontsize=11, fontname=INTER, va='center', zorder=6)

ax.text(tx, 52, 'hockeyROI.substack.com',
        color=GREY, fontsize=10, fontname=INTER,
        style='italic', va='center', zorder=6)

# --- SAVE --------------------------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor=BG, pad_inches=0)
plt.close()
print(f"Cover card saved: {OUT_PATH}")
