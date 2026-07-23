"""Shot charts for players / goalies / teams — matplotlib rink + shot dots.

Fed by the committable per-season parquets (Data/shot_events_by_season/), so it
works on Streamlit Cloud where the raw CSV is gitignored.

Coordinates: x_coord_norm in [~25, 99] (net at +89), y_coord_norm in [-42, 42],
already normalized so the shooter attacks toward +x. We draw the offensive half
with the net at the RIGHT. Perspective:
  - shooter/team view: y plotted as-is (shooter attacking rightward)
  - goalie view ("where I get scored on"): the goalie faces the shooter, so
    left/right mirror — we flip y so the labels read from the goalie's side.
Goals get a thick dark outer border; non-goal shots have no border. A light
hexbin density layer underneath is the "grouping" layer.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PARQUET_DIR = APP_DIR.parent / "Data" / "shot_events_by_season"

# Team primary / accent colors (abbrevs match the data; ARI + UTA both present).
TEAM_COLORS: dict[str, tuple[str, str]] = {
    "ANA": ("#F47A38", "#B09862"), "ARI": ("#8C2633", "#E2D6B5"),
    "BOS": ("#FFB81C", "#111111"), "BUF": ("#002654", "#FCB514"),
    "CGY": ("#C8102E", "#F1BE48"), "CAR": ("#CC0000", "#111111"),
    "CHI": ("#CF0A2C", "#FF671B"), "COL": ("#6F263D", "#236192"),
    "CBJ": ("#002654", "#CE1126"), "DAL": ("#006847", "#8F8F8C"),
    "DET": ("#CE1126", "#111111"), "EDM": ("#FF4C00", "#041E42"),
    "FLA": ("#C8102E", "#B9975B"), "LAK": ("#111111", "#A2AAAD"),
    "MIN": ("#154734", "#A6192E"), "MTL": ("#AF1E2D", "#192168"),
    "NSH": ("#FFB81C", "#041E42"), "NJD": ("#CE1126", "#111111"),
    "NYI": ("#00539B", "#F47D30"), "NYR": ("#0038A8", "#CE1126"),
    "OTT": ("#C52032", "#C2912C"), "PHI": ("#F74902", "#111111"),
    "PIT": ("#FCB514", "#111111"), "SJS": ("#006D75", "#EA7200"),
    "SEA": ("#001628", "#99D9D9"), "STL": ("#002F87", "#FCB514"),
    "TBL": ("#002868", "#8F8F8C"), "TOR": ("#00205B", "#8F8F8C"),
    "VAN": ("#00205B", "#00843D"), "VGK": ("#B4975A", "#333F42"),
    "WSH": ("#C8102E", "#041E42"), "WPG": ("#041E42", "#AC162C"),
    "UTA": ("#6CACE4", "#111111"),
}
_DEFAULT_COLOR = ("#4C6EF5", "#111111")

# Shot-type palette (stable, distinct)
SHOT_TYPE_COLORS = {
    "wrist": "#4C9BE8", "snap": "#2F5FBF", "slap": "#E8934A", "backhand": "#F2C744",
    "tip-in": "#5FBF7F", "deflected": "#2E8B57", "wrap-around": "#B05FD6",
    "bat": "#D65FA6", "between-legs": "#9C6B4A", "poke": "#8FA0A8",
    "cradle": "#8FA0A8", "unknown": "#9AA5AD",
}
ICE = "#FFFFFF"
ICE_LINE = "#9DBBD6"
GOAL_RED = "#C33"


def team_color(team: str) -> tuple[str, str]:
    return TEAM_COLORS.get(str(team).upper(), _DEFAULT_COLOR)


def load_shots(seasons: list[str] | None = None, game_type: str = "regular") -> pd.DataFrame:
    """Load per-season parquet shot data. seasons=None -> all available."""
    if not PARQUET_DIR.exists():
        return pd.DataFrame()
    files = sorted(PARQUET_DIR.glob("*.parquet"))
    if seasons is not None:
        want = {str(s) for s in seasons}
        files = [f for f in files if f.stem in want]
    if not files:
        return pd.DataFrame()
    cols = ["season", "game_type", "event_type", "shooter_player_id", "goalie_id",
            "shooting_team_abbrev", "x_coord_norm", "y_coord_norm", "is_goal", "shot_type"]
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=cols)
        d = d[(d["game_type"] == game_type)
              & d["event_type"].isin(["shot-on-goal", "missed-shot", "goal"])]
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _draw_rink(ax) -> None:
    """Offensive half, net at the right. Data coords (x ~25-99, y -42..42)."""
    ax.set_xlim(24, 100)
    ax.set_ylim(-43, 43)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((24, -43), 76, 86, facecolor=ICE, edgecolor=ICE_LINE, lw=1.5, zorder=0))
    ax.axvline(25, color=ICE_LINE, lw=3, alpha=0.8, zorder=1)      # blue line
    ax.text(25, 40, "BLUE LINE", color=ICE_LINE, fontsize=7, ha="left", va="top", weight="bold")
    ax.axvline(89, color=GOAL_RED, lw=2, alpha=0.85, zorder=1)     # goal line
    ax.text(89, 40, "GOAL LINE", color=GOAL_RED, fontsize=7, ha="right", va="top", weight="bold")
    ax.add_patch(Rectangle((89, -3), 4, 6, facecolor="#C33", alpha=0.75, zorder=2))  # net
    ax.add_patch(Arc((89, 0), 16, 12, theta1=90, theta2=270, color=ICE_LINE, lw=1, alpha=0.6))
    for cy in (-22, 22):                                            # faceoff circles
        ax.add_patch(plt.Circle((69, cy), 15, fill=False, edgecolor=ICE_LINE, lw=0.8, alpha=0.4))
        ax.add_patch(plt.Circle((69, cy), 0.8, color=ICE_LINE, alpha=0.5))


def shot_chart(shots: pd.DataFrame, title: str, subtitle: str = "",
               team: str | None = None, goalie_view: bool = False,
               show_bubbles: bool = True, goals_only: bool = False,
               footer: str = "@HockeyROI | hockeyROI.substack.com"):
    """Render a shot chart figure. Returns the matplotlib Figure (or None).
    goals_only: plot just the goals (still bordered); otherwise all shots."""
    if shots is None or shots.empty:
        return None
    d = shots.dropna(subset=["x_coord_norm", "y_coord_norm"]).copy()
    if d.empty:
        return None
    x = d["x_coord_norm"].astype(float).to_numpy()
    y = d["y_coord_norm"].astype(float).to_numpy()
    if goalie_view:                     # goalie faces the shooter -> mirror sides
        y = -y
    goal = d["is_goal"].astype(int).to_numpy() == 1
    prim, accent = team_color(team) if team else _DEFAULT_COLOR
    st = d["shot_type"].fillna("unknown").str.lower()
    cols = st.map(SHOT_TYPE_COLORS).fillna("#9AA5AD").to_numpy()
    # density layer uses whatever points are shown (all shots, or goals only)
    bx, by = (x[goal], y[goal]) if goals_only else (x, y)

    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=150)
    fig.patch.set_facecolor("white")
    _draw_rink(ax)

    if show_bubbles and len(bx) >= 12:   # grouping/density layer
        ax.hexbin(bx, by, gridsize=16, extent=(25, 99, -42, 42), mincnt=1,
                  cmap="Blues", alpha=0.35, zorder=1, linewidths=0)

    if not goals_only:
        # non-goal shots: colored by type, no border. Size/alpha shrink with
        # volume so high-count goalies/teams read as a heat cloud, not a blob.
        n = len(x)
        ds, da = (42, 0.72) if n < 400 else (24, 0.5) if n < 1200 else (13, 0.4)
        ax.scatter(x[~goal], y[~goal], s=ds, c=cols[~goal], alpha=da,
                   edgecolors="none", zorder=3)
    # goals: same fill, thick dark outer border — always prominent
    ax.scatter(x[goal], y[goal], s=70, c=cols[goal], alpha=0.95,
               edgecolors=accent, linewidths=1.8, zorder=4)

    ax.set_title(title, fontsize=15, weight="bold", color="#1A1A1A", loc="left", pad=10)
    if subtitle:
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, fontsize=9,
                color="#666", ha="left", va="bottom")
    # legend: shot types present + goal marker
    present = [t for t in SHOT_TYPE_COLORS if t in set(st)]
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=SHOT_TYPE_COLORS[t],
                          mec="none", ms=7, label=t.title()) for t in present[:8]]
    handles.append(plt.Line2D([0], [0], marker="o", ls="", mfc="#ccc",
                              mec=accent, mew=1.8, ms=8, label="Goal"))
    ax.legend(handles=handles, loc="lower left", ncol=4, fontsize=7,
              frameon=False, bbox_to_anchor=(0.0, -0.16))
    fig.text(0.5, 0.005, footer, ha="center", fontsize=7, color="#999", style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig
