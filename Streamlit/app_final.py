"""HockeyROI — NHL Zone Time + Net Front Impact Metrics Explorer."""
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
NFI_ADJ = REPO_ROOT / "NFI" / "output" / "fully_adjusted"

NHL_TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH",
]

# Two-dropdown structure: Game Type x Season.
# Regular season options shown when Game Type = "Regular Season".
# Playoff options shown when Game Type = "Playoffs". Per-year playoff
# breakdowns are not yet available (playoff CSVs are pooled with no per-year
# season column), so only "All Playoffs" is shown for now — additional entries
# are added automatically once breakdown data is published.
REGULAR_SEASON_OPTIONS = [
    "Pooled (2022–2026)",
    "2025-26",
    "2024-25",
    "2023-24",
    "2022-23",
]
SEASON_OPTIONS = REGULAR_SEASON_OPTIONS  # back-compat alias for TNZI filter code

GAME_TYPE_OPTIONS = ["Regular Season", "Playoffs"]
PLAYOFF_SEASON_OPTIONS_BASE = ["All Playoffs (2022–2025)"]

SEASON_TO_DOZI_COL = {
    "2025-26": "DOZI_25_26",
    "2024-25": "DOZI_24_25",
    "2023-24": "DOZI_23_24",
}

FLAG_EMOJI = {"RISING": "🟢", "STABLE": "🟡", "DECLINING": "🔴"}

TERTILE_METRICS = ["OZI", "DZI", "NZI", "TNZI", "TNZI_C", "TNZI_L", "TNZI_CL"]

DISPLAY_COLUMNS = [
    "player_name", "team", "pos", "GP", "seasons_qualified",
    "OZI", "DZI", "NZI",
    "TNZI", "TNZI_C", "TNZI_L", "TNZI_CL",
    "ZQoC", "ZQoL",
    "DOZI_23/24", "DOZI_24/25", "DOZI_25/26",
    "DOZI_trend", "DOZI_recent",
    "DOZI_flag",
]

NFI_SEASON_KEY = {
    "Pooled (2022–2026)": "pooled",
    "2025-26": "20252026",
    "2024-25": "20242025",
    "2023-24": "20232024",
    "2022-23": "20222023",
}

# NFI TOI thresholds per season context
NFI_TOI_DEFAULT = {"pooled": 2000, "season": 500}
NFI_TOI_MAX = 7500  # fixed slider range prevents cross-season state conflicts

# Style block injected at the top of each expander. Uses `details[open] *`
# selector with !important — beats inline color and any inherited CSS,
# regardless of Streamlit's internal DOM nesting. Summary stays orange.
EXPANDER_STYLE = (
    "<style>"
    "details[open] > div > div > div > div { color: #F0F4F8 !important; }"
    "details[open] * { color: #F0F4F8 !important; }"
    "details[open] summary { color: #F0F4F8 !important; }"
    "details[open] summary * { color: #F0F4F8 !important; }"
    "details[open] summary svg, details[open] summary svg path {"
    " fill: #F0F4F8 !important; stroke: #F0F4F8 !important; }"
    "</style>"
)


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
    fp = ADJ / f"tnzi_adjusted_{position}.csv"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_csv(fp)


@st.cache_data(show_spinner=False)
def load_combined_regular() -> pd.DataFrame:
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
    """Load TNZI playoff-adjusted data (forwards + defense) from
    Zones/output/playoffs/. Adds the _pos_group helper column so the same
    position filter used for regular-season works unchanged."""
    frames = []
    for pos_file in ("forwards", "defense"):
        fp = ZONES / "output" / "playoffs" / f"tnzi_adjusted_{pos_file}_playoffs.csv"
        if not fp.exists():
            continue
        try:
            d = pd.read_csv(fp)
        except Exception:
            continue
        if d.empty:
            continue
        d = d.copy()
        d["_pos_group"] = pos_file
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_nfi_playoffs() -> pd.DataFrame:
    """Load NFI playoff player table, if present."""
    fp = NFI_ADJ / "player_fully_adjusted_playoffs.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)
    if "toi_min" not in df.columns and "toi_sec" in df.columns:
        df["toi_min"] = df["toi_sec"] / 60
    return df


@st.cache_data(show_spinner=False)
def playoff_season_options(df: pd.DataFrame) -> list[str]:
    """Return the playoff season options to show in the Season dropdown.
    Always starts with 'All Playoffs (2022–2025)'. If the data has a
    distinguishable per-year season column (something other than the
    literal 'playoffs' bucket), add per-year options too — currently the
    files are pooled so just 'All' is shown."""
    options = list(PLAYOFF_SEASON_OPTIONS_BASE)
    if df is None or df.empty or "season" not in df.columns:
        return options
    distinct = sorted(set(df["season"].astype(str).unique()) - {"playoffs", ""})
    label_map = {
        "20222023": "2022-23 Playoffs",
        "20232024": "2023-24 Playoffs",
        "20242025": "2024-25 Playoffs",
        "20252026": "2025-26 Playoffs",
    }
    for s in distinct:
        lbl = label_map.get(s, f"{s} Playoffs")
        if lbl not in options:
            options.append(lbl)
    return options


@st.cache_data(show_spinner=False)
def nfi_playoffs_available() -> bool:
    """Check whether NFI playoff data file exists with at least one row."""
    fp = NFI_ADJ / "player_fully_adjusted_playoffs.csv"
    if not fp.exists():
        return False
    try:
        # Header-only or empty files don't count
        with open(fp) as f:
            return sum(1 for _ in f) > 1
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def playoffs_available() -> bool:
    """Check whether playoff-adjusted TNZI files exist with data."""
    candidates = [
        ZONES / "output" / "playoffs" / "tnzi_adjusted_forwards_playoffs.csv",
        ZONES / "output" / "playoffs" / "tnzi_adjusted_defense_playoffs.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            with open(p) as f:
                if sum(1 for _ in f) > 1:
                    return True
        except Exception:
            continue
    return False


@st.cache_data(show_spinner=False)
def tertile_cutoffs(metric: str) -> tuple[float, float] | None:
    df = load_combined_regular()
    if df.empty or metric not in df.columns:
        return None
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(values) < 3:
        return None
    low, high = np.percentile(values, [33.333, 66.666])
    return float(low), float(high)


@st.cache_data(show_spinner=False)
def load_nfi_player() -> pd.DataFrame:
    fp = NFI_ADJ / "player_fully_adjusted.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    if "toi_min" not in df.columns and "toi_sec" in df.columns:
        df["toi_min"] = df["toi_sec"] / 60
    return df


# ---------------------------------------------------------------------------
# Styling — palette + CSS
# ---------------------------------------------------------------------------
PALETTE = {
    "bg":          "#0B1D2E",   # uniform dark background
    "surface":     "#0B1D2E",
    "panel":       "#122C44",   # slightly lighter for sidebar/expander contrast
    "blue":        "#2E7DC4",
    "lightblue":   "#4AB3E8",
    "text":        "#F0F4F8",
    "text_dark":   "#1B3A5C",   # for dark-on-light dropdowns
    "input_bg":    "#F0F4F8",
    "orange":      "#FF6B35",
    "rising":      "#5DAA7A",   # softer green
    "stable":      "#D4A843",   # softer yellow
    "declining":   "#C05555",   # softer red
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
        [data-testid="stHeader"] {{ background: {PALETTE['bg']}; }}
        [data-testid="stSidebar"] {{ background-color: {PALETTE['panel']} !important; }}
        /* Sidebar label text stays white, but EXCLUDE input controls (handled below) */
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p {{ color: {PALETTE['text']} !important; }}

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
        .hockeyroi-brand .hockey {{ color: {PALETTE['text']}; }}
        .hockeyroi-brand .roi    {{ color: {PALETTE['orange']}; }}

        .tagline {{ color: {PALETTE['text']}; opacity: 0.85; font-size: 1rem; margin-top: -0.25rem; }}
        .timestamp {{ color: {PALETTE['lightblue']}; font-size: 0.85rem; margin-top: 0.25rem; }}

        /* Disclaimer box — orange border, orange heading, white body */
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

        /* Dataframe */
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] table {{
            background-color: {PALETTE['surface']} !important;
            color: {PALETTE['text']} !important;
        }}

        /* Expander — orange border, white summary + body text. Body text color
           set inline in each expander (see render_*_explainers) to avoid CSS
           specificity conflicts with Streamlit's own styles. */
        [data-testid="stExpander"] {{
            background-color: {PALETTE['panel']};
            border: 1px solid {PALETTE['orange']};
            border-radius: 4px;
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {{
            color: {PALETTE['text']} !important;
            font-weight: 600;
        }}
        /* Chevron SVG stays white for contrast */
        [data-testid="stExpander"] summary svg,
        [data-testid="stExpander"] summary svg path,
        [data-testid="stExpander"] summary svg polygon {{
            fill: {PALETTE['text']} !important;
            stroke: {PALETTE['text']} !important;
        }}

        /* Tabs — text white, active tab white underline */
        [data-testid="stTabs"] {{ background-color: {PALETTE['bg']}; }}
        [data-testid="stTabs"] button {{
            color: {PALETTE['text']} !important;
            background-color: transparent !important;
        }}
        [data-testid="stTabs"] button[aria-selected="true"] {{
            color: {PALETTE['text']} !important;
            border-bottom: 3px solid {PALETTE['text']} !important;
        }}

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

        /* Buttons */
        .stButton > button {{
            background-color: {PALETTE['blue']};
            color: {PALETTE['text']};
            border: 0;
        }}
        .stButton > button:hover {{ background-color: {PALETTE['lightblue']}; }}

        /* --- DROPDOWN / INPUT READABILITY (ISSUE 2 FIX) --- */
        /* Selectbox displayed value */
        [data-testid="stSelectbox"] > div > div {{
            color: {PALETTE['text_dark']} !important;
            background-color: {PALETTE['input_bg']} !important;
        }}
        /* Multiselect container */
        [data-testid="stMultiSelect"] > div > div {{
            color: {PALETTE['text_dark']} !important;
            background-color: {PALETTE['input_bg']} !important;
        }}
        /* Base select — selected value text */
        [data-baseweb="select"] span,
        [data-baseweb="select"] div[role="button"] span {{
            color: {PALETTE['text_dark']} !important;
        }}
        /* Dropdown menu options */
        [data-baseweb="menu"] {{
            background-color: {PALETTE['input_bg']} !important;
        }}
        [data-baseweb="menu"] li,
        [data-baseweb="menu"] li div {{
            color: {PALETTE['text_dark']} !important;
            background-color: {PALETTE['input_bg']} !important;
        }}
        [data-baseweb="menu"] li:hover {{
            background-color: #D9E2EC !important;
        }}
        /* Multiselect selected tags */
        [data-baseweb="tag"] {{
            background-color: {PALETTE['lightblue']} !important;
        }}
        [data-baseweb="tag"] span {{
            color: {PALETTE['text_dark']} !important;
        }}
        /* Text input */
        input[type="text"] {{
            color: {PALETTE['text_dark']} !important;
            background-color: {PALETTE['input_bg']} !important;
        }}
        [data-testid="stTextInput"] input {{
            color: {PALETTE['text_dark']} !important;
            background-color: {PALETTE['input_bg']} !important;
        }}
        /* Radio option text stays white in sidebar */
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] label p {{
            color: {PALETTE['text']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# TNZI — filtering, display, styling
# ---------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filters for TNZI tab. Case-insensitive name search; season / position /
    team / GP / flag filters all applied to raw columns."""
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
        if "seasons_qualified" in f.columns:
            f = f[pd.to_numeric(f["seasons_qualified"], errors="coerce") >= 4]

    teams = st.session_state.get("f_teams", [])
    if teams:
        f = f[f["team"].isin(teams)]

    name_q = (st.session_state.get("f_name", "") or "").strip().lower()
    if name_q:
        f = f[f["player_name"].fillna("").str.lower().str.contains(name_q, na=False)]

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
    rename = {
        "DOZI_23_24": "DOZI_23/24",
        "DOZI_24_25": "DOZI_24/25",
        "DOZI_25_26": "DOZI_25/26",
        "IOZC": "ZQoC",
        "IOZL": "ZQoL",
    }
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
    for m in ("ZQoC", "ZQoL"):
        if m in display_df.columns:
            fmt[m] = "{:.3f}"
    for m in ("DOZI_23/24", "DOZI_24/25", "DOZI_25/26", "DOZI_trend", "DOZI_recent"):
        if m in display_df.columns:
            fmt[m] = "{:+.3f}"
    styler = styler.format(fmt, na_rep="—")
    return styler


# ---------------------------------------------------------------------------
# Common UI
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
# TNZI tab
# ---------------------------------------------------------------------------
def render_tnzi_disclaimer() -> None:
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


def render_tnzi_sidebar() -> None:
    st.sidebar.header("TNZI Filters")

    st.session_state.setdefault("f_position", "All")
    st.session_state.setdefault("game_type_tnzi", "Regular Season")
    st.session_state.setdefault("f_season", REGULAR_SEASON_OPTIONS[0])
    st.session_state.setdefault("f_teams", [])
    st.session_state.setdefault("f_name", "")
    st.session_state.setdefault("f_min_gp", 0)
    st.session_state.setdefault("f_flag", "All")

    st.sidebar.radio("Position", ["All", "Forwards", "Defensemen"], key="f_position")

    # --- Two-dropdown selector: Game Type then Season ---
    game_type = st.sidebar.selectbox("Game Type", GAME_TYPE_OPTIONS, key="game_type_tnzi")
    if game_type == "Regular Season":
        season_opts = REGULAR_SEASON_OPTIONS
    else:
        # Playoffs — always show "All Playoffs" + any per-year breakdown the
        # data supports. If TNZI playoff CSVs aren't present, hide the section.
        if not playoffs_available():
            st.sidebar.markdown(
                f"<div style='color:{PALETTE['lightblue']};opacity:0.7;font-style:italic;"
                "margin:-0.3rem 0 0.6rem 0;font-size:0.85rem;'>"
                "Playoffs — Coming Soon</div>",
                unsafe_allow_html=True,
            )
            season_opts = ["All Playoffs (2022–2025)"]
        else:
            tnzi_play = load_combined_playoffs()
            season_opts = playoff_season_options(tnzi_play)
    # Coerce stale state into the current option list.
    if st.session_state.get("f_season") not in season_opts:
        st.session_state["f_season"] = season_opts[0]
    st.sidebar.selectbox("Season", season_opts, key="f_season")
    if game_type == "Playoffs" and len(season_opts) == 1:
        st.sidebar.markdown(
            f"<div style='color:{PALETTE['lightblue']};opacity:0.7;font-style:italic;"
            "margin:-0.3rem 0 0.6rem 0;font-size:0.82rem;'>"
            "Season breakdown coming in next update</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.multiselect("Teams", NHL_TEAMS, key="f_teams")
    st.sidebar.text_input("Player name contains", key="f_name")
    st.sidebar.slider("Minimum GP", min_value=0, max_value=400, step=5, key="f_min_gp")
    st.sidebar.selectbox("DOZI flag", ["All", "RISING", "STABLE", "DECLINING"], key="f_flag")


def render_tnzi_table() -> None:
    game_type = st.session_state.get("game_type_tnzi", "Regular Season")
    season = st.session_state.get("f_season", REGULAR_SEASON_OPTIONS[0])
    if game_type == "Playoffs":
        data = load_combined_playoffs()
        if data.empty:
            st.caption("🏒 TNZI playoff data coming soon — populates automatically as games are played")
            return
    else:
        data = load_combined_regular()
        if data.empty:
            st.error("Adjusted ranking files not found under Zones/adjusted_rankings/.")
            return
    filtered = apply_filters(data)
    if filtered.empty:
        st.info("No players match the current filters. Widen position, team, or GP filters.")
        return

    display_df = prepare_display(filtered)
    # Hardcoded default sort: TNZI_L descending (best individual-contribution metric).
    # Playoff data lacks TNZI_L/CL — fall back to TNZI, then to first column.
    default_sort = next(
        (c for c in ("TNZI_L", "TNZI") if c in display_df.columns),
        display_df.columns[0],
    )
    display_df = display_df.sort_values(
        by=default_sort, ascending=False, na_position="last"
    )

    st.dataframe(style_frame(display_df), width="stretch", hide_index=True)
    st.caption(f"Showing {len(display_df):,} players — sorted by {default_sort} desc")


def render_tnzi_explainers() -> None:
    W = "#F0F4F8"  # body text color — inline to bypass CSS specificity
    with st.expander("What each metric means"):
        st.markdown(EXPANDER_STYLE, unsafe_allow_html=True)
        st.markdown(
            f'<ul style="color:{W};">'
            f'<li style="color:{W};"><strong>OZI</strong>: After an offensive zone faceoff, what % of the shift generated events in the offensive zone. Measures zone retention.</li>'
            f'<li style="color:{W};"><strong>DZI</strong>: After a defensive zone faceoff, what % of the shift generated events in the offensive zone. Measures complete eventful transition from own end to attack.</li>'
            f'<li style="color:{W};"><strong>NZI</strong>: After a neutral zone faceoff, what % of the shift generated events in the offensive zone. Measures offensive zone reach from equal footing.</li>'
            f'<li style="color:{W};"><strong>TNZI</strong>: After a neutral zone faceoff, net difference between offensive and defensive zone events. Positive = net offensive driver. Negative = net drag.</li>'
            f'<li style="color:{W};"><strong>TNZI_C</strong>: TNZI adjusted for quality of competition faced. Higher competition = more credit.</li>'
            f'<li style="color:{W};"><strong>TNZI_L</strong>: TNZI adjusted for quality of linemates. Better linemates = score adjusted down. Best measure of individual contribution.</li>'
            f'<li style="color:{W};"><strong>TNZI_CL</strong>: TNZI adjusted for both competition and linemates.</li>'
            f'<li style="color:{W};"><strong>ZQoC</strong>: Zone Quality of Competition — weighted average zone metric score of opponents faced. Higher = tougher competition.</li>'
            f'<li style="color:{W};"><strong>ZQoL</strong>: Zone Quality of Linemates — weighted average zone metric score of teammates. Higher = better linemates.</li>'
            f'<li style="color:{W};"><strong>DOZI</strong>: Delta on Zone Impact — year over year change in TNZI score. Positive = improving. Negative = declining.</li>'
            f'<li style="color:#F0F4F8;"><strong>DOZI vs NFI%_3A_MOM</strong>: DOZI (Zone Impact) and NFI%_3A_MOM (Net Front Impact) both measure year-over-year player trajectory but through different frameworks. DOZI tracks zone possession change. NFI%_3A_MOM tracks dangerous zone shot quality change. Use both together for the most complete picture of player momentum.</li>'
            f'</ul>',
            unsafe_allow_html=True,
        )

    with st.expander("Methodology"):
        st.markdown(EXPANDER_STYLE, unsafe_allow_html=True)
        st.markdown(
            f'<ul style="color:{W};">'
            f'<li style="color:{W};">Zone tracking from NHL API x/y coordinates</li>'
            f'<li style="color:{W};">All events with xCoord contribute — shots, hits, blocked shots, faceoffs, giveaways, takeaways</li>'
            f'<li style="color:{W};">Zone boundaries: OZ <code>x &gt; 25</code>, NZ <code>-25 to 25</code>, DZ <code>x &lt; -25</code></li>'
            f'<li style="color:{W};">Faceoff shifts only — line changes excluded</li>'
            f'<li style="color:{W};">5v5 situations only via <code>situationCode 1551</code></li>'
            f'<li style="color:{W};">Wilson CI at 95% confidence using faceoff shift count as <code>n</code></li>'
            f'<li style="color:{W};">Minimum 50 faceoff shifts per zone type</li>'
            f'<li style="color:{W};">Normalized 0–10 within position group separately</li>'
            f'<li style="color:{W};">Full methodology and source: '
              f'<a href="https://github.com/HockeyROI/nhl-analytics" style="color:#4AB3E8;">'
              f'github.com/HockeyROI/nhl-analytics</a></li>'
            f'</ul>',
            unsafe_allow_html=True,
        )

    with st.expander("Known limitations"):
        st.markdown(EXPANDER_STYLE, unsafe_allow_html=True)
        st.markdown(
            f'<ul style="color:{W};">'
            f'<li style="color:{W};">These are <strong>team outcome</strong> metrics — individual skill AND team system both contribute.</li>'
            f'<li style="color:{W};">Carolina Hurricanes system effect is clearly visible — multiple Hurricanes in top 20 across all metrics.</li>'
            f'<li style="color:{W};">Current season ZQoL uses pooled linemate data as approximation — players traded mid-season (e.g. Quinn Hughes Dec 2025, Brent Burns) may show stale linemate context. Historical season views use exact linemate data.</li>'
            f'<li style="color:{W};">Brent Burns shows #1 current season D in TNZI_L — this is an artifact of his historical Carolina ZQoL. His current raw TNZI (7.2) does not support a #1 ranking.</li>'
            f'<li style="color:{W};">Y coordinate not used — corner vs slot play treated identically.</li>'
            f'<li style="color:{W};">Zone tracking between events is approximation — last known coordinate assumed.</li>'
            f'<li style="color:{W};">These metrics do not predict team winning better than Corsi or xG in multi-season testing.</li>'
            f'</ul>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# NFI tab
# ---------------------------------------------------------------------------
def render_nfi_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer-box">
        <strong>NFI metrics use ES regulation data only.</strong> Minimum 2000 ES minutes for pooled
        rankings. Small sample players flagged with an asterisk. All data from public NHL API.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_nfi_sidebar() -> None:
    st.sidebar.header("NFI Filters")

    st.session_state.setdefault("nfi_position", "All")
    st.session_state.setdefault("game_type_nfi", "Regular Season")
    st.session_state.setdefault("nfi_season", REGULAR_SEASON_OPTIONS[0])
    st.session_state.setdefault("nfi_teams", [])
    st.session_state.setdefault("nfi_name", "")
    st.session_state.setdefault("nfi_min_toi", NFI_TOI_DEFAULT["pooled"])
    st.session_state.setdefault("nfi_last_season", REGULAR_SEASON_OPTIONS[0])

    # Widgets
    st.sidebar.radio("Position", ["All", "Forwards", "Defensemen"], key="nfi_position")

    # --- Game Type then Season (two-dropdown structure) ---
    game_type = st.sidebar.selectbox("Game Type", GAME_TYPE_OPTIONS, key="game_type_nfi")
    if game_type == "Regular Season":
        season_opts = REGULAR_SEASON_OPTIONS
    else:
        if not nfi_playoffs_available():
            st.sidebar.markdown(
                f"<div style='color:{PALETTE['lightblue']};opacity:0.7;font-style:italic;"
                "margin:-0.3rem 0 0.6rem 0;font-size:0.85rem;'>"
                "NFI Playoffs — Coming Soon</div>",
                unsafe_allow_html=True,
            )
            season_opts = ["All Playoffs (2022–2025)"]
        else:
            nfi_play = load_nfi_playoffs()
            season_opts = playoff_season_options(nfi_play)
    if st.session_state.get("nfi_season") not in season_opts:
        st.session_state["nfi_season"] = season_opts[0]
    chosen_season = st.sidebar.selectbox("Season", season_opts, key="nfi_season")
    if game_type == "Playoffs" and len(season_opts) == 1:
        st.sidebar.markdown(
            f"<div style='color:{PALETTE['lightblue']};opacity:0.7;font-style:italic;"
            "margin:-0.3rem 0 0.6rem 0;font-size:0.82rem;'>"
            "Season breakdown coming in next update</div>",
            unsafe_allow_html=True,
        )

    # On season change, reset TOI threshold to appropriate default.
    if st.session_state.get("nfi_last_season") != chosen_season:
        is_pooled_view = chosen_season in ("Pooled (2022–2026)", "All Playoffs (2022–2025)")
        st.session_state["nfi_min_toi"] = (
            NFI_TOI_DEFAULT["pooled"] if is_pooled_view else NFI_TOI_DEFAULT["season"]
        )
        st.session_state["nfi_last_season"] = chosen_season

    st.sidebar.multiselect("Teams", NHL_TEAMS, key="nfi_teams")
    st.sidebar.text_input("Player name contains", key="nfi_name")
    st.sidebar.slider(
        "Minimum ES TOI (min)",
        min_value=0, max_value=NFI_TOI_MAX, step=50, key="nfi_min_toi",
    )


def _aggregate_nfi_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """Career TOI-weighted means per player for pooled view."""
    if df.empty:
        return df
    rows = []
    for (pid, name, pos), g in df.groupby(["player_id", "player_name", "position"]):
        toi = g["toi_min"].astype(float)
        toi_total = float(toi.sum())
        if toi_total <= 0:
            continue

        def tw(col: str) -> float:
            if col not in g.columns:
                return np.nan
            vals = pd.to_numeric(g[col], errors="coerce")
            m = vals.notna() & (toi > 0)
            return float(np.average(vals[m], weights=toi[m])) if m.any() else np.nan

        rows.append({
            "player_id": int(pid),
            "player_name": name,
            "position": pos,
            "team": g.sort_values("season")["team"].iloc[-1],
            "toi_min": toi_total,
            "NFI_pct_ZA": tw("NFI_pct_ZA"),
            "NFI_pct_3A": tw("NFI_pct_3A"),
            "RelNFI_F_pct": tw("RelNFI_F_pct"),
            "RelNFI_A_pct": tw("RelNFI_A_pct"),
            "RelNFI_pct":   tw("RelNFI_pct"),
            "NFQOC":        tw("NFQOC"),
            "NFQOL":        tw("NFQOL"),
            "NFI_pct_3A_MOM": np.nan,  # MOM meaningful only per-season
        })
    return pd.DataFrame(rows)


def _filter_nfi(df: pd.DataFrame, season_opt: str) -> pd.DataFrame:
    if df.empty:
        return df

    position = st.session_state.get("nfi_position", "All")
    if position == "Forwards":
        df = df[df["position"] == "F"]
    elif position == "Defensemen":
        df = df[df["position"] == "D"]

    season_key = NFI_SEASON_KEY[season_opt]
    if season_key == "pooled":
        out = _aggregate_nfi_pooled(df)
    else:
        sub = df[df["season"] == season_key].copy()
        keep_cols = [
            "player_id", "player_name", "position", "team", "toi_min",
            "NFI_pct_ZA", "NFI_pct_3A",
            "RelNFI_F_pct", "RelNFI_A_pct", "RelNFI_pct",
            "NFQOC", "NFQOL",
            "NFI_pct_3A_MOM",
        ]
        out = sub[[c for c in keep_cols if c in sub.columns]].copy()

    teams = st.session_state.get("nfi_teams", [])
    if teams:
        out = out[out["team"].isin(teams)]

    name_q = (st.session_state.get("nfi_name", "") or "").strip().lower()
    if name_q:
        out = out[out["player_name"].fillna("").str.lower().str.contains(name_q, na=False)]

    min_toi = st.session_state.get("nfi_min_toi", NFI_TOI_DEFAULT["pooled"])
    out = out[out["toi_min"].fillna(0) >= min_toi]

    threshold = NFI_TOI_DEFAULT["pooled"] if season_key == "pooled" else NFI_TOI_DEFAULT["season"]
    out["small_sample"] = out["toi_min"] < threshold
    return out.reset_index(drop=True)


def _filter_nfi_playoffs(df: pd.DataFrame, season_opt: str) -> pd.DataFrame:
    """Filter the NFI playoff table by sidebar widgets. Per-year breakdown
    isn't available in the current playoff CSV (it's pooled), so any season
    option falls through to the full pooled view; once per-year data lands,
    this filter will branch on `season` column values."""
    if df.empty:
        return df
    out = df.copy()
    position = st.session_state.get("nfi_position", "All")
    if position == "Forwards":
        out = out[out["position"] == "F"]
    elif position == "Defensemen":
        out = out[out["position"] == "D"]
    # Per-year filter (only applies once data column has per-year values)
    if season_opt != "All Playoffs (2022–2025)" and "season" in out.columns:
        season_label_to_key = {
            "2022-23 Playoffs": "20222023",
            "2023-24 Playoffs": "20232024",
            "2024-25 Playoffs": "20242025",
            "2025-26 Playoffs": "20252026",
        }
        key = season_label_to_key.get(season_opt)
        if key is not None:
            out = out[out["season"].astype(str) == key]
    teams = st.session_state.get("nfi_teams", [])
    if teams:
        out = out[out["team"].isin(teams)]
    name_q = (st.session_state.get("nfi_name", "") or "").strip().lower()
    if name_q:
        out = out[out["player_name"].fillna("").str.lower().str.contains(name_q, na=False)]
    min_toi = st.session_state.get("nfi_min_toi", NFI_TOI_DEFAULT["pooled"])
    if "toi_min" in out.columns:
        out = out[out["toi_min"].fillna(0) >= min_toi]
        # Playoff sample is small — flag anything under 100 ES min as small_sample
        out["small_sample"] = out["toi_min"] < 100
    return out.reset_index(drop=True)


def _nfi_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "small_sample" in out.columns:
        out["player_name"] = out.apply(
            lambda r: f"{r['player_name']} *" if r["small_sample"] else r["player_name"],
            axis=1,
        )
    out = out.rename(columns={
        "player_name": "Player", "position": "Pos", "team": "Team", "toi_min": "TOI",
        "NFI_pct_ZA": "NFI%_ZA", "NFI_pct_3A": "NFI%_3A",
        "RelNFI_F_pct": "RelNFI_F%", "RelNFI_A_pct": "RelNFI_A%", "RelNFI_pct": "RelNFI%",
        "NFQOC": "NFQOC", "NFQOL": "NFQOL",
        "NFI_pct_3A_MOM": "NFI%_3A_MOM",
    })
    cols = ["Player", "Pos", "Team", "TOI",
            "NFI%_ZA", "NFI%_3A",
            "RelNFI_F%", "RelNFI_A%", "RelNFI%",
            "NFQOC", "NFQOL", "NFI%_3A_MOM"]
    cols = [c for c in cols if c in out.columns]
    return out[cols]


def _style_nfi(display_df: pd.DataFrame):
    if display_df.empty:
        return display_df

    def color_mom(col):
        out = []
        for v in col:
            try:
                x = float(v)
            except (TypeError, ValueError):
                out.append("")
                continue
            if np.isnan(x):
                out.append("")
            elif x > 0:
                out.append(f"color: {PALETTE['rising']}; font-weight: 600;")
            elif x < 0:
                out.append(f"color: {PALETTE['declining']}; font-weight: 600;")
            else:
                out.append("")
        return out

    styler = display_df.style
    if "NFI%_3A_MOM" in display_df.columns:
        styler = styler.apply(color_mom, subset=["NFI%_3A_MOM"])

    fmt = {}
    for c in ("NFI%_ZA", "NFI%_3A", "NFQOC", "NFQOL"):
        if c in display_df.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("RelNFI_F%", "RelNFI_A%", "RelNFI%"):
        if c in display_df.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:+.2f}"
    if "NFI%_3A_MOM" in display_df.columns:
        fmt["NFI%_3A_MOM"] = lambda x: "—" if pd.isna(x) else f"{x:+.3f}"
    if "TOI" in display_df.columns:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"

    styler = styler.format(fmt, na_rep="—")
    return styler


def render_nfi_table() -> None:
    game_type = st.session_state.get("game_type_nfi", "Regular Season")
    season = st.session_state.get("nfi_season", REGULAR_SEASON_OPTIONS[0])

    if game_type == "Playoffs":
        if not nfi_playoffs_available():
            st.caption("🏒 Playoff data coming soon — populates automatically as games are played")
            st.stop()
        df = load_nfi_playoffs()
        if df.empty:
            st.error("NFI playoff file is empty.")
            return
        filtered = _filter_nfi_playoffs(df, season)
    else:
        df = load_nfi_player()
        if df.empty:
            st.error(
                "NFI player file not found at "
                "`NFI/output/fully_adjusted/player_fully_adjusted.csv`."
            )
            return
        filtered = _filter_nfi(df, season)
    if filtered.empty:
        st.info("No players match the current filters. Widen position, team, or TOI filters.")
        return

    filtered = filtered.sort_values("NFI_pct_ZA", ascending=False, na_position="last").reset_index(drop=True)
    filtered.insert(0, "Rank", np.arange(1, len(filtered) + 1))
    display_df = _nfi_display(filtered)
    display_df.insert(0, "#", filtered["Rank"].values)

    st.dataframe(_style_nfi(display_df), width="stretch", hide_index=True)
    caption = f"Showing {len(display_df):,} players — sorted by NFI%_ZA descending"
    if "small_sample" in filtered.columns and filtered["small_sample"].any():
        n_small = int(filtered["small_sample"].sum())
        caption += f"  |  {n_small} small-sample players flagged with *"
    st.caption(caption)


def render_nfi_explainers() -> None:
    W = "#F0F4F8"
    with st.expander("What each NFI metric means"):
        st.markdown(EXPANDER_STYLE, unsafe_allow_html=True)
        st.markdown(
            f'<ul style="color:{W};">'
            f'<li style="color:{W};"><strong>NFI%</strong> — <strong>Net Front Impact %</strong>. Fenwick attempts (shots on goal + misses + goals, blocks excluded) filtered to CNFI (central net-front) and MNFI (mid net-front) zones. Team CNFI+MNFI for / (for + against) while the player is on ice. <strong>R² = 0.583 vs standings</strong> — beats xG% (0.538) and Corsi (0.397).</li>'
            f'<li style="color:{W};"><strong>NFI%_ZA</strong> — Zone-Adjusted using the <strong>empirical OZ/DZ factor of +10.71 pp</strong> (not the traditional 3.5 pp, which under-corrects by ~67%).</li>'
            f'<li style="color:{W};"><strong>NFI%_3A</strong> — <strong>Three-Adjusted</strong>: zone adjustment + NFQOC + NFQOL.</li>'
            f'<li style="color:{W};"><strong>RelNFI_F%</strong> — on-ice minus off-ice team CNFI+MNFI For per 60. Positive = team generates more dangerous shots with player on ice.</li>'
            f'<li style="color:{W};"><strong>RelNFI_A%</strong> — off-ice minus on-ice CNFI+MNFI Against per 60. Positive = suppresses more.</li>'
            f'<li style="color:{W};"><strong>RelNFI%</strong> — net two-way dangerous-zone impact = RelNFI_F% + RelNFI_A%.</li>'
            f'<li style="color:{W};"><strong>NFQOC</strong> — <strong>Net Front Quality of Competition</strong> — shared-TOI weighted mean of opponents\' NFI%, computed linemate-without-me to avoid shared-event collinearity.</li>'
            f'<li style="color:{W};"><strong>NFQOL</strong> — <strong>Net Front Quality of Linemates</strong> — same approach for teammates.</li>'
            f'<li style="color:{W};"><strong>NFI%_3A_MOM</strong> — year-over-year change in NFI%_3A. Positive = ascending.</li>'
            f'<li style="color:#F0F4F8;"><strong>NFI%_3A_MOM vs DOZI</strong>: NFI%_3A_MOM tracks year-over-year change in quality-adjusted dangerous zone performance. DOZI in the Zone Impact tab tracks zone possession change. They measure analogous momentum concepts through different methodologies — check both tabs for the most complete player trajectory picture.</li>'
            f'</ul>',
            unsafe_allow_html=True,
        )

    with st.expander("Methodology"):
        st.markdown(EXPANDER_STYLE, unsafe_allow_html=True)
        st.markdown(
            f'<ul style="color:{W};">'
            f'<li style="color:{W};">NFI zones (CNFI, MNFI) derived from shot-density clustering of NHL API x/y coordinates</li>'
            f'<li style="color:{W};">Fenwick events only (shots on goal + missed + goals). Blocks excluded — their coordinates are recorded at the blocker\'s location, not the shooter\'s.</li>'
            f'<li style="color:{W};">5v5 ES regulation only (state = ES in <code>shots_tagged.csv</code>)</li>'
            f'<li style="color:{W};">Per-player on-ice attribution: each ES shot event counts for all skaters on-ice</li>'
            f'<li style="color:{W};">Empirical zone factor from league-pooled OZ/DZ faceoff-shift analysis = <strong>+10.71 pp</strong></li>'
            f'<li style="color:{W};">NFQOC / NFQOL use a <strong>linemate-without-me</strong> correction — teammate <code>j</code>\'s rating is recomputed excluding events where both <code>i</code> and <code>j</code> were on ice, preventing β_QoL ≈ 1.0</li>'
            f'<li style="color:{W};">3A = raw − zone × (OZ_ratio − 0.5) − β_NFQOC × (NFQOC − mean) − β_NFQOL × (NFQOL − mean)</li>'
            f'<li style="color:{W};">Minimum thresholds: 2000 ES minutes pooled, 500 ES minutes current season</li>'
            f'<li style="color:{W};">Source: <a href="https://github.com/HockeyROI/nhl-analytics" style="color:#4AB3E8;">github.com/HockeyROI/nhl-analytics</a></li>'
            f'</ul>',
            unsafe_allow_html=True,
        )

    with st.expander("Known limitations"):
        st.markdown(EXPANDER_STYLE, unsafe_allow_html=True)
        st.markdown(
            f'<ul style="color:{W};">'
            f'<li style="color:{W};">NFI% is an <strong>on-ice</strong> metric — shared-event context effects are partially corrected via linemate-without-me but residual team-system effects remain.</li>'
            f'<li style="color:{W};">Carolina forwards (Fast, Staal, Martinook) rank high on NFI%_ZA because of CAR\'s system. 3A adjusts for it but doesn\'t fully eliminate it.</li>'
            f'<li style="color:{W};">2022-23 through 2025-26 available. 2021-22 not included (raw PBP starts 2022-23).</li>'
            f'<li style="color:{W};"><strong>Rel-NFI metrics do not aggregate to team points</strong> (they demean to zero within each team). Use Rel-NFI for individual ranking; use NFI%_ZA / 3A for team-level inference.</li>'
            f'<li style="color:{W};">MOM for 2025-26 is partial-season through the latest update.</li>'
            f'</ul>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="HockeyROI — Zone Time + Net Front Impact",
        page_icon="🏒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_header()

    # Sidebar framework selector — replaces the previous top-tab navigation.
    # Renders above all framework-specific filters; the picked section drives
    # which sidebar group + main content block is shown.
    st.sidebar.markdown(
        '<p style="color:#FF6B35; font-weight:700; font-size:1rem; margin-bottom:4px;">FRAMEWORK</p>',
        unsafe_allow_html=True,
    )
    section = st.sidebar.selectbox(
        "Framework",
        ["Zone Impact (TNZI)", "Net Front Impact (NFI)"],
        index=0,
        key="framework_selector",
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")

    if section == "Zone Impact (TNZI)":
        render_tnzi_disclaimer()
        render_tnzi_sidebar()
        render_tnzi_table()
        render_tnzi_explainers()
    elif section == "Net Front Impact (NFI)":
        render_nfi_disclaimer()
        render_nfi_sidebar()
        render_nfi_table()
        render_nfi_explainers()

    render_footer()


if __name__ == "__main__":
    main()
