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

# Brand (match the app's Altair charts)
NAVY = "#1B3A5C"        # PALETTE["text"] — titles, numbers, wordmark "HOCKEY"
ORANGE = "#FF6B35"      # PALETTE["orange"] — wordmark "ROI"
GREY = "#888888"        # text_secondary
GOAL_EDGE = "#111111"   # goal dots: black perimeter

# Shot-type palette — per user: snap=orange, slap=yellow, backhand=purple
SHOT_TYPE_COLORS = {
    "wrist": "#4C9BE8", "snap": "#FF6B35", "slap": "#F2C744", "backhand": "#8E5BD6",
    "tip-in": "#3FB37F", "deflected": "#2E8B57", "wrap-around": "#D65FA6",
    "bat": "#E86AA6", "between-legs": "#9C6B4A", "poke": "#8FA0A8",
    "cradle": "#8FA0A8", "unknown": "#9AA5AD",
}
ICE = "#F4F8FC"         # super-light blue — no border needed
ICE_LINE = "#BFD4E6"    # subtle interior lines (blue line / goal line / circles)
GOAL_RED = "#D4706A"


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
            "shooting_team_abbrev", "x_coord_norm", "y_coord_norm", "is_goal",
            "shot_type", "situation_code", "shooting_team_id", "home_team_id"]
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=cols)
        d = d[(d["game_type"] == game_type)
              & d["event_type"].isin(["shot-on-goal", "missed-shot", "goal"])]
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # situation_code = [away_goalie, away_skaters, home_skaters, home_goalie]
    sc = out["situation_code"].astype(str).str.zfill(4)
    shoot_home = out["shooting_team_id"] == out["home_team_id"]
    # Empty net = the DEFENDING team's goalie digit is 0.
    def_goalie = np.where(shoot_home, sc.str[0].astype(int), sc.str[3].astype(int))
    out["empty_net"] = def_goalie == 0
    # Shooter-perspective strength state, e.g. "5v5" / "5v4" / "4v5" — lets the
    # chart honour the app's Situation filter.
    ask, hsk = sc.str[1].astype(int), sc.str[2].astype(int)
    own = np.where(shoot_home, hsk, ask)
    opp = np.where(shoot_home, ask, hsk)
    out["situation"] = [f"{a}v{b}" for a, b in zip(own, opp)]
    return out


def _draw_rink(ax) -> None:
    """Offensive half, net at the right. No border — the ice is just the light
    axes background (data coords x ~25-99, y -42..42)."""
    # Show back to centre ice: 100 units wide vs 86 tall keeps the chart
    # LANDSCAPE with the aspect still equal, so the whole image fits on screen
    # (a 24-100 window is taller than it is wide once the aspect is equalised).
    ax.set_xlim(0, 100)
    ax.set_ylim(-43, 43)
    ax.set_aspect("equal")
    ax.set_facecolor(ICE)                 # ice = light background, no perimeter
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axvline(25, color="#8FB2D4", lw=1.8, alpha=0.7, zorder=1)   # blue line
    ax.axvline(89, color=GOAL_RED, lw=1.4, alpha=0.8, zorder=1)    # goal line
    ax.add_patch(Rectangle((89, -3), 3.5, 6, facecolor=GOAL_RED, alpha=0.8, zorder=2))
    ax.add_patch(Arc((89, 0), 14, 11, theta1=90, theta2=270, color=ICE_LINE, lw=0.8, alpha=0.7))
    for cy in (-22, 22):                                           # faceoff circles
        ax.add_patch(plt.Circle((69, cy), 15, fill=False, edgecolor=ICE_LINE, lw=0.7, alpha=0.6))
        ax.add_patch(plt.Circle((69, cy), 0.7, color=ICE_LINE, alpha=0.7))


def shot_chart(shots: pd.DataFrame, name: str, season: str = "", stat: str = "",
               team: str | None = None, goalie_view: bool = False,
               show_bubbles: bool = True, goals_only: bool = False,
               url: str = "hockeyROI.substack.com"):
    """Render a shot chart figure. Returns the matplotlib Figure (or None).
    Header stacks: NAME (big) / season / stat line. goals_only: just goals."""
    if shots is None or shots.empty:
        return None
    d = shots.dropna(subset=["x_coord_norm", "y_coord_norm"]).copy()
    if d.empty:
        return None
    # Empty-net shots are excluded entirely (no goalie to beat, and they're
    # usually taken from the shooter's own end) — noted on the chart instead.
    if "empty_net" in d.columns:
        _en_mask = d["empty_net"].astype(bool)
        n_en_goals = int((_en_mask & (d["is_goal"].astype(int) == 1)).sum())
        d = d[~_en_mask]
        if d.empty:
            return None
    else:
        n_en_goals = 0
    x = d["x_coord_norm"].astype(float).to_numpy()
    y = d["y_coord_norm"].astype(float).to_numpy()
    if goalie_view:                     # goalie faces the shooter -> mirror sides
        y = -y
    goal = d["is_goal"].astype(int).to_numpy() == 1
    # Only the points actually drawn count toward the legend / long-range note.
    shown = goal if goals_only else np.ones(len(x), dtype=bool)
    # ~3% of shots (and goals — e.g. empty-netters from a player's own end) are
    # taken outside the offensive zone we draw. Pin them to the left edge rather
    # than silently clipping them off-chart, so the dot count matches reality.
    n_far = int((x[shown] < 1).sum())
    x = np.clip(x, 1.0, 99.0)
    y = np.clip(y, -41.0, 41.0)
    prim, accent = team_color(team) if team else _DEFAULT_COLOR
    st = d["shot_type"].fillna("unknown").str.lower()
    cols = st.map(SHOT_TYPE_COLORS).fillna("#9AA5AD").to_numpy()
    fig, ax = plt.subplots(figsize=(4.4, 3.1), dpi=220)
    fig.patch.set_facecolor("white")
    _draw_rink(ax)

    # dot size/alpha shrink with volume so high-count goalies/teams read as a
    # heat cloud rather than a blob.
    n = len(x)
    ds, da = (22, 0.75) if n < 400 else (12, 0.5) if n < 1200 else (7, 0.4)
    if not goals_only:
        # non-goal shots: colored by type, no border
        ax.scatter(x[~goal], y[~goal], s=ds, c=cols[~goal], alpha=da,
                   edgecolors="none", zorder=3)
    # goals: SAME size as shots, with a very thin dark ring
    ax.scatter(x[goal], y[goal], s=ds, c=cols[goal], alpha=0.95,
               edgecolors=GOAL_EDGE, linewidths=0.35, zorder=4)

    # header: name (+ position), top-LEFT, sized to match the app's other charts
    ax.text(0.0, 1.015, name, transform=ax.transAxes, fontsize=5.5,
            weight="bold", color=NAVY, ha="left", va="bottom", family="sans-serif")
    # Legend = shot types only. In goals-only view every dot IS a goal, so the
    # "Goal" swatch is redundant; in all-shots view the ring is explained by a
    # note instead of a swatch.
    # Legend covers EVERY shot type the player has, not just the drawn subset,
    # so Goals-only and All-shots produce the same legend height and therefore
    # the same cropped image size.
    _st_shown = set(st)
    present = [t for t in SHOT_TYPE_COLORS if t in _st_shown and t != "unknown"]
    # 3 columns -> the legend wraps to 3-4 rows but stays NARROWER than the ice,
    # so the axes is the widest artist and the brand really does land in the
    # bottom-right corner (with 5 cols it overflowed and pulled the brand in).
    _NCOL = 3
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=SHOT_TYPE_COLORS[t],
                          mec="none", ms=4.5,
                          label=t.replace("-", " ").title())
               for t in present[:12]]
    # left-aligned with the ice + the name above it
    leg = ax.legend(handles=handles, loc="upper left", ncol=_NCOL, fontsize=5.5,
                    frameon=False, bbox_to_anchor=(0.0, -0.01),
                    handletextpad=0.25, columnspacing=0.8, borderaxespad=0.0)
    for txt in leg.get_texts():
        txt.set_color(NAVY)

    # Everything below the legend, so nothing can overlap it however many rows
    # the legend wraps onto.
    _rows = max(1, int(np.ceil(len(handles) / _NCOL)))
    _y = -0.045 - 0.046 * _rows

    # Constant note text: a mode-dependent string changes the tight-crop width,
    # which made Goals-only and All-shots render at different sizes.
    ax.text(0.0, _y, "circled dot = goal · empty-net goals excluded",
            transform=ax.transAxes, fontsize=4.5, color=GREY, ha="left",
            va="top", style="italic", family="sans-serif")
    # HOCKEY-ROI wordmark + url on their OWN line below the notes (right side)
    # so a long note can never run into them. "HOCKEY" ends at the junction x
    # and "ROI" starts there, so the pair reads as one wordmark.
    _yb = _y - 0.075
    _jx = 0.93
    ax.text(_jx, _yb, "HOCKEY", transform=ax.transAxes, ha="right",
            va="top", color=NAVY, weight="bold", fontsize=5, family="sans-serif")
    ax.text(_jx, _yb, "ROI", transform=ax.transAxes, ha="left",
            va="top", color=ORANGE, weight="bold", fontsize=5, family="sans-serif")
    if url:
        ax.text(1.0, _yb - 0.055, url, transform=ax.transAxes, ha="right",
                va="top", color=GREY, fontsize=3.8, style="italic",
                family="sans-serif")
    # extra bottom room per wrapped legend row
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93,
                        bottom=0.16 + 0.045 * (_rows - 1))
    return fig
