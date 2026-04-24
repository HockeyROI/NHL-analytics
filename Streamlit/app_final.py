"""HockeyROI — NHL Zone Time Metrics Explorer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
ZONES = REPO_ROOT / "Zones"
ADJ = ZONES / "adjusted_rankings"
VARS_DIR = ZONES / "zone_variations"

NHL_TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH",
]

SEASON_OPTIONS = [
    "Pooled (4 seasons)",
    "Current Season (2025-26)",
    "2024-25",
    "2023-24",
    "2022-23",
]

SEASON_TO_DOZI_COL = {
    "Current Season (2025-26)": "DOZI_25_26",
    "2024-25": "DOZI_24_25",
    "2023-24": "DOZI_23_24",
}

FLAG_EMOJI = {"RISING": "🟢", "STABLE": "🟡", "DECLINING": "🔴"}
FLAG_COLOR = {"RISING": "#44AA66", "STABLE": "#FFB700", "DECLINING": "#CC3333"}

TERTILE_METRICS = ["OZI", "DZI", "NZI", "TNZI", "TNZI_C", "TNZI_L", "TNZI_CL"]

DISPLAY_COLUMNS = [
    "player_name", "team", "pos", "GP", "seasons_qualified",
    "OZI", "DZI", "NZI",
    "TNZI", "TNZI_C", "TNZI_L", "TNZI_CL",
    "IOZC", "IOZL",
    "DOZI_23/24", "DOZI_24/25", "DOZI_25/26",
    "DOZI_trend", "DOZI_recent",
    "DOZI_flag",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_last_updated(kind: str = "regular") -> str:
    fp = ZONES / f"last_updated_{kind}.txt"
    if not fp.exists():
        return "unknown"
    return fp.read_text().strip()


@st.cache_data(show_spinner=False)
def load_adjusted(position: str) -> pd.DataFrame:
    """Load tnzi_adjusted_{forwards|defense}.csv — primary data source."""
    fp = ADJ / f"tnzi_adjusted_{position}.csv"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_csv(fp)


@st.cache_data(show_spinner=False)
def load_playoffs_raw(position: str) -> pd.DataFrame:
    """Best-effort load of playoffs zone data (oze_dze_nze_*)."""
    fp = ZONES / f"oze_dze_nze_{position}_playoffs.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    return df


@st.cache_data(show_spinner=False)
def load_combined_regular() -> pd.DataFrame:
    """Union of forwards + defense adjusted files, regular season."""
    frames = []
    for pos_file in ("forwards", "defense"):
        d = load_adjusted(pos_file)
        if not d.empty:
            d = d.copy()
            d["_pos_group"] = pos_file
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_combined_playoffs() -> pd.DataFrame:
    frames = []
    for pos_file in ("forwards", "defense"):
        d = load_playoffs_raw(pos_file)
        if not d.empty:
            d = d.copy()
            d["_pos_group"] = pos_file
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def tertile_cutoffs(metric: str) -> tuple[float, float] | None:
    """Compute 33rd/67th percentile cutoffs on the full unfiltered regular
    season dataset so colors remain stable as the user filters."""
    df = load_combined_regular()
    if df.empty or metric not in df.columns:
        return None
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(values) < 3:
        return None
    low, high = np.percentile(values, [33.333, 66.666])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#1B3A5C",
    "surface": "#0B1D2E",
    "blue": "#2E7DC4",
    "lightblue": "#4AB3E8",
    "text": "#F0F4F8",
    "orange": "#FF6B35",
    "rising": "#44AA66",
    "stable": "#FFB700",
    "declining": "#CC3333",
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&display=swap');

        html, body, [data-testid="stAppViewContainer"], .main {{
            background-color: {PALETTE['bg']} !important;
            color: {PALETTE['text']} !important;
            font-family: 'Inter', Arial, sans-serif;
        }}
        [data-testid="stHeader"] {{ background: {PALETTE['surface']}; }}
        [data-testid="stSidebar"] {{ background-color: {PALETTE['surface']} !important; }}
        [data-testid="stSidebar"] * {{ color: {PALETTE['text']} !important; }}

        h1, h2, h3, h4 {{
            font-family: 'Bebas Neue', Impact, 'Arial Black', sans-serif;
            color: {PALETTE['text']};
            letter-spacing: 0.5px;
        }}

        .hockeyroi-brand {{
            font-family: 'Bebas Neue', Impact, sans-serif;
            font-size: 3.2rem;
            line-height: 1;
            letter-spacing: 2px;
        }}
        .hockeyroi-brand .hockey {{ color: {PALETTE['lightblue']}; }}
        .hockeyroi-brand .roi {{ color: {PALETTE['orange']}; }}

        .tagline {{
            color: {PALETTE['text']};
            opacity: 0.85;
            font-size: 1rem;
            margin-top: -0.25rem;
        }}
        .timestamp {{
            color: {PALETTE['lightblue']};
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }}

        .disclaimer-box {{
            border: 2px solid {PALETTE['orange']};
            background: rgba(11, 29, 46, 0.6);
            padding: 1rem 1.25rem;
            border-radius: 6px;
            margin: 1rem 0 1.5rem 0;
            color: {PALETTE['text']};
            font-size: 0.95rem;
            line-height: 1.5;
        }}
        .disclaimer-box strong {{ color: {PALETTE['orange']}; }}

        /* Dataframe styling */
        [data-testid="stDataFrame"] {{ background-color: {PALETTE['surface']}; }}

        /* Expander */
        [data-testid="stExpander"] {{
            background-color: {PALETTE['surface']};
            border: 1px solid {PALETTE['blue']};
            border-radius: 4px;
        }}
        [data-testid="stExpander"] summary {{ color: {PALETTE['lightblue']} !important; }}

        /* Footer */
        .hockeyroi-footer {{
            border-top: 1px solid {PALETTE['blue']};
            margin-top: 2rem;
            padding-top: 1rem;
            color: {PALETTE['text']};
            opacity: 0.8;
            font-size: 0.85rem;
            text-align: center;
        }}
        .hockeyroi-footer a {{ color: {PALETTE['lightblue']}; text-decoration: none; }}

        /* Buttons and widgets */
        .stButton > button {{
            background-color: {PALETTE['blue']};
            color: {PALETTE['text']};
            border: 0;
        }}
        .stButton > button:hover {{ background-color: {PALETTE['lightblue']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    f = df.copy()

    position = st.session_state.get("f_position", "All")
    if position == "Forwards":
        f = f[f["_pos_group"] == "forwards"]
    elif position == "Defensemen":
        f = f[f["_pos_group"] == "defense"]

    season = st.session_state.get("f_season", SEASON_OPTIONS[0])
    if season in SEASON_TO_DOZI_COL:
        col = SEASON_TO_DOZI_COL[season]
        if col in f.columns:
            f = f[f[col].notna()]
    elif season == "2022-23":
        # DOZI not tracked for 22-23 — filter to players with 4 qualified seasons
        if "seasons_qualified" in f.columns:
            f = f[pd.to_numeric(f["seasons_qualified"], errors="coerce") >= 4]

    teams = st.session_state.get("f_teams", [])
    if teams:
        f = f[f["team"].isin(teams)]

    name_q = (st.session_state.get("f_name", "") or "").strip().lower()
    if name_q:
        f = f[f["player_name"].str.lower().str.contains(name_q, na=False)]

    min_gp = st.session_state.get("f_min_gp", 0)
    if "GP" in f.columns and min_gp > 0:
        f = f[pd.to_numeric(f["GP"], errors="coerce").fillna(0) >= min_gp]

    flag = st.session_state.get("f_flag", "All")
    if flag != "All" and "DOZI_flag" in f.columns:
        f = f[f["DOZI_flag"] == flag]

    return f


def prepare_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    rename = {"DOZI_23_24": "DOZI_23/24", "DOZI_24_25": "DOZI_24/25", "DOZI_25_26": "DOZI_25/26"}
    out = out.rename(columns=rename)

    if "DOZI_flag" in out.columns:
        def _fmt(v):
            if pd.isna(v) or v == "" or v is None:
                return ""
            return f"{FLAG_EMOJI.get(v, '')} {v}".strip()
        out["DOZI_flag"] = out["DOZI_flag"].apply(_fmt)

    cols = [c for c in DISPLAY_COLUMNS if c in out.columns]
    return out[cols]


def style_frame(display_df: pd.DataFrame):
    if display_df.empty:
        return display_df

    cutoffs = {m: tertile_cutoffs(m) for m in TERTILE_METRICS}

    def color_metric(col_name):
        cut = cutoffs.get(col_name)
        if not cut:
            return [""] * len(display_df)
        low, high = cut
        out = []
        for v in display_df[col_name]:
            try:
                x = float(v)
            except (TypeError, ValueError):
                out.append("")
                continue
            if x >= high:
                out.append(f"background-color: {PALETTE['rising']}; color: white;")
            elif x >= low:
                out.append(f"background-color: {PALETTE['stable']}; color: black;")
            else:
                out.append(f"background-color: {PALETTE['declining']}; color: white;")
        return out

    def color_flag(col):
        out = []
        for v in col:
            s = str(v) if v is not None else ""
            if "RISING" in s:
                out.append(f"background-color: {PALETTE['rising']}; color: white;")
            elif "STABLE" in s:
                out.append(f"background-color: {PALETTE['stable']}; color: black;")
            elif "DECLINING" in s:
                out.append(f"background-color: {PALETTE['declining']}; color: white;")
            else:
                out.append("")
        return out

    styler = display_df.style
    for m in TERTILE_METRICS:
        if m in display_df.columns:
            styler = styler.apply(lambda _c, name=m: color_metric(name), subset=[m])
    if "DOZI_flag" in display_df.columns:
        styler = styler.apply(color_flag, subset=["DOZI_flag"])

    fmt = {}
    for m in TERTILE_METRICS:
        if m in display_df.columns:
            fmt[m] = "{:.1f}"
    for m in ("IOZC", "IOZL"):
        if m in display_df.columns:
            fmt[m] = "{:.3f}"
    for m in ("DOZI_23/24", "DOZI_24/25", "DOZI_25/26", "DOZI_trend", "DOZI_recent"):
        if m in display_df.columns:
            fmt[m] = "{:+.3f}"
    styler = styler.format(fmt, na_rep="—")
    return styler


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def render_header() -> None:
    ts = load_last_updated("regular")
    st.markdown(
        f"""
        <div class="hockeyroi-brand"><span class="hockey">HOCKEY</span><span class="roi">ROI</span></div>
        <div class="tagline">Automated Zone Time Tracking — Updated Daily</div>
        <div class="timestamp">Last updated: {ts}</div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer-box">
        <strong>IMPORTANT:</strong> These metrics measure <em>eventful zone time</em> — where meaningful
        hockey plays occur after each type of faceoff deployment. They are context tools for player
        evaluation, not winning predictors. Zone time metrics do not outperform Corsi or xG% as team
        winning predictors in multi-season testing.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    st.sidebar.header("Filters")

    st.session_state.setdefault("f_position", "All")
    st.session_state.setdefault("f_season", SEASON_OPTIONS[0])
    st.session_state.setdefault("f_game_type", "Regular Season")
    st.session_state.setdefault("f_teams", [])
    st.session_state.setdefault("f_name", "")
    st.session_state.setdefault("f_min_gp", 0)
    st.session_state.setdefault("f_flag", "All")
    st.session_state.setdefault("sort_col", "TNZI_L")
    st.session_state.setdefault("sort_desc", True)

    st.sidebar.radio("Position", ["All", "Forwards", "Defensemen"], key="f_position")
    st.sidebar.selectbox("Season", SEASON_OPTIONS, key="f_season")
    st.sidebar.radio("Game type", ["Regular Season", "Playoffs"], key="f_game_type")
    st.sidebar.multiselect("Teams", NHL_TEAMS, key="f_teams")
    st.sidebar.text_input("Player name contains", key="f_name")
    st.sidebar.slider("Minimum GP", min_value=0, max_value=400, step=5, key="f_min_gp")
    st.sidebar.selectbox("DOZI flag", ["All", "RISING", "STABLE", "DECLINING"], key="f_flag")


def render_table() -> None:
    game_type = st.session_state.get("f_game_type", "Regular Season")

    if game_type == "Playoffs":
        raw = load_combined_playoffs()
        if raw.empty:
            st.info(
                "No playoffs data available yet for the selected view. "
                "Playoffs metrics use a different schema and are tracked separately."
            )
            return
        st.warning(
            "Playoffs view shows raw zone-event metrics (OZE / DZE / NZE) only. "
            "Adjusted TNZI and DOZI are computed on regular season data."
        )
        filtered = apply_filters(raw)
        if filtered.empty:
            st.info("No players match the current filters.")
            return
        st.dataframe(filtered, width="stretch", hide_index=True)
        st.caption(f"Showing {len(filtered):,} players")
        return

    data = load_combined_regular()
    if data.empty:
        st.error("Adjusted ranking files not found under Zones/adjusted_rankings/.")
        return

    filtered = apply_filters(data)
    if filtered.empty:
        st.info("No players match the current filters. Widen position, team, or GP filters.")
        return

    display_df = prepare_display(filtered)

    sort_col = st.session_state.get("sort_col", "TNZI_L")
    if sort_col in display_df.columns:
        display_df = display_df.sort_values(
            by=sort_col,
            ascending=not st.session_state.get("sort_desc", True),
            na_position="last",
        )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        pick = st.selectbox(
            "Sort by",
            [c for c in display_df.columns if c not in ("player_name", "team", "pos")],
            index=[c for c in display_df.columns if c not in ("player_name", "team", "pos")].index(
                sort_col
            ) if sort_col in display_df.columns else 0,
            key="sort_col",
        )
    with col_b:
        st.checkbox("Descending", key="sort_desc")

    display_df = display_df.sort_values(
        by=pick, ascending=not st.session_state.get("sort_desc", True), na_position="last"
    )

    st.dataframe(style_frame(display_df), width="stretch", hide_index=True)
    st.caption(f"Showing {len(display_df):,} players")


def render_explainers() -> None:
    with st.expander("What each metric means"):
        st.markdown(
            """
- **OZI**: After an offensive zone faceoff, what % of the shift generated events in the offensive zone. Measures zone retention.
- **DZI**: After a defensive zone faceoff, what % of the shift generated events in the offensive zone. Measures complete eventful transition from own end to attack.
- **NZI**: After a neutral zone faceoff, what % of the shift generated events in the offensive zone. Measures offensive zone reach from equal footing.
- **TNZI**: After a neutral zone faceoff, net difference between offensive and defensive zone events. Positive = net offensive driver. Negative = net drag.
- **TNZI_C**: TNZI adjusted for quality of competition faced. Higher competition = more credit.
- **TNZI_L**: TNZI adjusted for quality of linemates. Better linemates = score adjusted down. Best measure of individual contribution.
- **TNZI_CL**: TNZI adjusted for both competition and linemates.
- **IOZC**: Impact of Zone Competition — weighted average zone metric score of opponents faced. Higher = tougher competition.
- **IOZL**: Impact of Zone Linemates — weighted average zone metric score of teammates. Higher = better linemates.
- **DOZI**: Delta on Zone Impact — year over year change in TNZI score. Positive = improving. Negative = declining.
            """
        )

    with st.expander("Methodology"):
        st.markdown(
            """
- Zone tracking from NHL API x/y coordinates
- All events with xCoord contribute — shots, hits, blocked shots, faceoffs, giveaways, takeaways
- Zone boundaries: OZ `x > 25`, NZ `-25 to 25`, DZ `x < -25`
- Faceoff shifts only — line changes excluded
- 5v5 situations only via `situationCode 1551`
- Wilson CI at 95% confidence using faceoff shift count as `n`
- Minimum 50 faceoff shifts per zone type
- Normalized 0–10 within position group separately
- Full methodology and source code: [github.com/HockeyROI/nhl-analytics](https://github.com/HockeyROI/nhl-analytics)
            """
        )

    with st.expander("Known limitations"):
        st.markdown(
            """
- These are **team outcome** metrics — individual skill AND team system both contribute.
- Carolina Hurricanes system effect is clearly visible — multiple Hurricanes in top 20 across all metrics.
- Current season IOZL uses pooled linemate data as approximation — players traded mid-season (e.g. Quinn Hughes Dec 2025, Brent Burns) may show stale linemate context. Historical season views use exact linemate data.
- Brent Burns shows #1 current season D in TNZI_L — this is an artifact of his historical Carolina IOZL. His current raw TNZI (7.2) does not support a #1 ranking.
- Y coordinate not used — corner vs slot play treated identically.
- Zone tracking between events is approximation — last known coordinate assumed.
- These metrics do not predict team winning better than Corsi or xG in multi-season testing.
            """
        )


def render_footer() -> None:
    st.markdown(
        """
        <div class="hockeyroi-footer">
        HockeyROI — <a href="https://hockeyROI.substack.com">hockeyROI.substack.com</a> |
        <a href="https://twitter.com/HockeyROI">@HockeyROI</a> |
        <a href="https://github.com/HockeyROI/nhl-analytics">github.com/HockeyROI/nhl-analytics</a><br/>
        Data updated daily via NHL API. All methodology transparent and open source.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="HockeyROI — Zone Time Metrics",
        page_icon="🏒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_header()
    render_disclaimer()
    render_sidebar()
    render_table()
    render_explainers()
    render_footer()


if __name__ == "__main__":
    main()
