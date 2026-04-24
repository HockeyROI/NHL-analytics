#!/usr/bin/env python3
"""
Shift content right on all 6 HockeyROI cover cards so Substack's thumbnail
crop doesn't cut off the text on the left. Output saved into Covers/New/.

Strategy
--------
1. For the 3 cards with generator scripts (Kuemper / Forsberg / Oilers-vs-Avs),
   re-render them with a larger left margin + narrower photo.
2. For the 3 cards with only a PNG (McDavid / Playoff / Geometry), open the
   PNG and shift its contents right using PIL: pad the left with the brand
   dark background and crop the rightmost SHIFT_PX pixels.

Everything ends up 1200x675 @ the same size/aspect.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

# --- PATHS -------------------------------------------------------------------
BASE = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT_DIR = os.path.join(BASE, "Covers", "New")
os.makedirs(OUT_DIR, exist_ok=True)

# --- BRAND -------------------------------------------------------------------
BG    = '#0B1D2E'
BLUE  = '#2E7DC4'
WHITE = '#F0F4F8'
GREY  = '#888888'
BG_RGB = tuple(int(BG.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

W_PX, H_PX = 1200, 675
DPI = 150

# How much to shift content to the right so Substack thumbnails don't clip.
SHIFT_PX = 140

# --- FONTS -------------------------------------------------------------------
BEBAS_PATH = "/tmp/BebasNeue-Regular.ttf"
INTER_PATH = "/tmp/Inter-Regular.ttf"

def load_font(path, fallback):
    if os.path.exists(path):
        fm.fontManager.addfont(path)
        return fm.FontProperties(fname=path).get_name()
    return fallback

BEBAS = load_font(BEBAS_PATH, "Arial Black")
INTER = load_font(INTER_PATH, "Arial")


# =============================================================================
# Helper: re-render goalie / game cover with a shifted left margin
# =============================================================================
def render_photo_cover(photo_path, out_path,
                       top_text, title_line1, title_line2,
                       subtitle, stat_line,
                       tx=212):
    """
    Generic renderer matching the Kuemper/Forsberg/Oilers template but with
    configurable left-text x-origin (tx) and proportionally narrower photo.
    """
    # Photo width: keep the photo right-aligned and a bit narrower so it
    # still clears the shifted text block.
    img_w_px = int(W_PX * 0.36)
    img_h_px = H_PX

    img_pil = Image.open(photo_path).convert("RGBA")
    src_w, src_h = img_pil.size
    scale = max(img_w_px / src_w, img_h_px / src_h)
    img_pil = img_pil.resize((int(src_w * scale), int(src_h * scale)), Image.LANCZOS)
    new_w, new_h = img_pil.size
    left = (new_w - img_w_px) // 2
    top  = max(0, new_h - img_h_px)
    img_pil = img_pil.crop((left, top, left + img_w_px, top + img_h_px))

    arr = np.array(img_pil).astype(float)
    fade_px = int(img_w_px * 0.45)
    alpha_mask = np.ones(img_w_px, dtype=float)
    alpha_mask[:fade_px] = np.linspace(0.0, 1.0, fade_px)
    arr[:, :, 3] *= alpha_mask[np.newaxis, :]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    faded = Image.fromarray(arr, 'RGBA')

    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PX); ax.set_ylim(0, H_PX)
    ax.set_facecolor(BG); ax.axis('off')

    img_x0 = W_PX - img_w_px
    ax.imshow(faded, extent=[img_x0, W_PX, 0, H_PX],
              aspect='auto', origin='upper', zorder=2)

    # Vignette
    grad = np.zeros((H_PX, W_PX, 4), dtype=np.uint8)
    vignette_w = int(W_PX * 0.68)
    alpha_vig = np.linspace(210, 0, vignette_w, dtype=np.uint8)
    grad[:, :vignette_w, 0] = BG_RGB[0]
    grad[:, :vignette_w, 1] = BG_RGB[1]
    grad[:, :vignette_w, 2] = BG_RGB[2]
    grad[:, :vignette_w, 3] = alpha_vig[np.newaxis, :]
    ax.imshow(Image.fromarray(grad, 'RGBA'),
              extent=[0, W_PX, 0, H_PX], aspect='auto',
              origin='upper', zorder=3)

    # Text block
    ax.text(tx, 575, '@HockeyROI', color=GREY, fontsize=11,
            fontname=BEBAS, va='center', zorder=6)
    ax.text(tx, 510, top_text, color=WHITE, fontsize=32,
            fontname=BEBAS, fontweight='bold', va='center', zorder=6)
    ax.text(tx, 415, title_line1, color=WHITE, fontsize=42,
            fontname=BEBAS, fontweight='bold', va='center', zorder=6)
    ax.text(tx, 315, title_line2, color=WHITE, fontsize=42,
            fontname=BEBAS, fontweight='bold', va='center', zorder=6)
    ax.text(tx, 220, subtitle, color=BLUE, fontsize=20,
            fontname=BEBAS, va='center', zorder=6)
    ax.plot([tx, tx + 340], [192, 192], color=BLUE, linewidth=1.0,
            alpha=0.80, zorder=6)
    ax.text(tx, 165, stat_line, color=GREY, fontsize=11,
            fontname=INTER, va='center', zorder=6)
    ax.text(tx, 52, 'hockeyROI.substack.com', color=GREY, fontsize=10,
            fontname=INTER, style='italic', va='center', zorder=6)

    plt.savefig(out_path, dpi=DPI, bbox_inches='tight',
                facecolor=BG, pad_inches=0)
    plt.close()
    print(f"  wrote {out_path}")


# =============================================================================
# Helper: scale an existing PNG down and center it on a 1200x675 dark canvas.
# Preserves ALL original content (no edge loss); just gives both sides equal
# breathing room so Substack's thumbnail crop can't clip the leftmost text.
# =============================================================================
def shift_png_right(src_path, out_path, side_margin=SHIFT_PX):
    img = Image.open(src_path).convert("RGB")
    src_w, src_h = img.size
    target_w = W_PX - 2 * side_margin
    scale = target_w / src_w
    new_w = target_w
    new_h = int(src_h * scale)
    img_small = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (W_PX, H_PX), BG_RGB)
    paste_x = side_margin
    paste_y = (H_PX - new_h) // 2
    canvas.paste(img_small, (paste_x, paste_y))
    canvas.save(out_path)
    print(f"  wrote {out_path}")


# =============================================================================
# Run all 6
# =============================================================================
if __name__ == "__main__":
    # --- 1-3: re-render photo covers with wider left margin --------------
    render_photo_cover(
        photo_path=os.path.join(BASE, "Covers", "Kuemper.jpg"),
        out_path=os.path.join(OUT_DIR, "kuemper_cover.png"),
        top_text='HOW TO BEAT',
        title_line1='DARCY',
        title_line2='KUEMPER',
        subtitle='WEAKNESS ANALYSIS',
        stat_line='132 Games  |  3,466 Shots  |  3 Seasons',
    )
    render_photo_cover(
        photo_path=os.path.join(BASE, "Covers", "Foresberg.jpg"),
        out_path=os.path.join(OUT_DIR, "forsberg_cover.png"),
        top_text='HOW TO BEAT',
        title_line1='ANTON',
        title_line2='FORSBERG',
        subtitle='WEAKNESS ANALYSIS',
        stat_line='Career Sample  |  All Situations  |  3 Seasons',
    )
    render_photo_cover(
        photo_path=os.path.join(BASE, "Covers", "Nate.jpg"),
        out_path=os.path.join(OUT_DIR, "oilers_avs_cover.png"),
        top_text='APRIL 13 2026',
        title_line1='OILERS',
        title_line2='VS AVS',
        subtitle='GAME ANALYSIS',
        stat_line='Regular Season  |  Game Recap',
    )

    # --- 4-6: shift-existing-PNG covers ----------------------------------
    shift_png_right(
        src_path=os.path.join(BASE, "2026 posts", "Mcdavid_post",
                              "Images", "Mcd cover.png"),
        out_path=os.path.join(OUT_DIR, "mcdavid_cover.png"),
    )
    shift_png_right(
        src_path=os.path.join(BASE, "2026 posts", "Playoff_preview",
                              "Images", "Playoff 2026 cover.png"),
        out_path=os.path.join(OUT_DIR, "playoff_round1_cover.png"),
    )
    shift_png_right(
        src_path=os.path.join(BASE, "NFI", "Geometry_post",
                              "Images", "Geometry of Winning Cover.png"),
        out_path=os.path.join(OUT_DIR, "geometry_cover.png"),
    )

    print("\nAll 6 covers written to:", OUT_DIR)
