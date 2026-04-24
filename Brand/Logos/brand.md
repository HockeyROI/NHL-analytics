# HockeyROI Brand Style Guide

## Color Palette
| Name | Hex | Usage |
|------|-----|-------|
| Ice Black | #0B1D2E | Dark headers, footers |
| Deep Ice | #1B3A5C | Primary background |
| ROI Blue | #2E7DC4 | Primary accent, bars, lines |
| Ice Blue | #4AB3E8 | Secondary accent, labels, highlights |
| Signal Orange | #FF6B35 | Key stats only — use sparingly |
| Off White | #F0F4F8 | Body text, axis labels |

## Typography
- **Headlines:** Bebas Neue (fall back to Impact or Anton if unavailable)
- **Body / labels / annotations:** Inter (fall back to DM Sans or Helvetica)

## Chart Defaults (matplotlib / seaborn)
- Figure background: #0B1D2E
- Axes background: #1B3A5C
- Primary bar/line color: #2E7DC4
- Highlight color (key stat): #FF6B35
- Text / tick labels: #F0F4F8
- Grid lines: #ffffff15 (very subtle)
- Spine color: #ffffff20
- Font: Inter for all labels and annotations

## Chart Style Code Snippet
```python
import matplotlib.pyplot as plt
import matplotlib as mpl

BACKGROUND = "#0B1D2E"
SURFACE = "#1B3A5C"
BLUE = "#2E7DC4"
ICE_BLUE = "#4AB3E8"
ORANGE = "#FF6B35"
TEXT = "#F0F4F8"
GRID = "#FFFFFF15"

def apply_hockeyroi_style(fig, ax):
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
```

## Logo / Wordmark
- Text: HOCKEYROI in Bebas Neue
- Always on dark background
- Never place on white or light backgrounds

## General Rules
- Orange (#FF6B35) is for ONE key stat per chart only — never use as a general color
- Keep charts clean and minimal — no chartjunk
- Always include "HockeyROI" or "hockeyROI.substack.com" as a footer annotation on exported charts
- Wilson 95% confidence intervals on all sample-based metrics
- Pool minimum 3 seasons of data before publishing findings
