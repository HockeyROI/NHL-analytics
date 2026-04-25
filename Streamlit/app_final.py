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
    "Pooled (4 seasons)": "pooled",
    "Current Season (2025-26)": "20252026",
    "2024-25": "20242025",
    "2023-24": "20232024",
    "2022-23": "20222023",
}

# NFI TOI thresholds per season context
NFI_TOI_DEFAULT = {"pooled": 2000, "season": 500}
NFI_TOI_MAX = 7500  # fixed slider range prevents cross-season state conflicts


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
def playoffs_available() -> bool:
    """Check whether any playoff-adjusted or playoff-event files exist."""
    candidates = [
        ZONES / "oze_dze_nze_forwards_playoffs.csv",
        ZONES / "oze_dze_nze_defense_playoffs.csv",
        ADJ / "tnzi_adjusted_forwards_playoffs.csv",
        ADJ / "tnzi_adjusted_defense_playoffs.csv",
    ]
    return any(p.exists() for p in candidates)


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

        /* Expander — orange border, orange header */
        [data-testid="stExpander"] {{
            background-color: {PALETTE['panel']};
            border: 1px solid {PALETTE['orange']};
            border-radius: 4px;
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p {{
            color: {PALETTE['orange']} !important;
            font-weight: 600;
        }}
        [data-testid="stExpander"] div[role="region"] {{ color: {PALETTE['text']}; }}
        [data-testid="stExpander"] li {{ color: {PALETTE['text']}; }}

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
    st.session_state.setdefault("f_season", SEASON_OPTIONS[0])
    st.session_state.setdefault("f_teams", [])
    st.session_state.setdefault("f_name", "")
    st.session_state.setdefault("f_min_gp", 0)
    st.session_state.setdefault("f_flag", "All")
    st.session_state.setdefault("sort_col", "TNZI_L")
    st.session_state.setdefault("sort_desc", True)

    st.sidebar.radio("Position", ["All", "Forwards", "Defensemen"], key="f_position")
    st.sidebar.selectbox("Season", SEASON_OPTIONS, key="f_season")
    if not playoffs_available():
        st.sidebar.markdown(
            f"<div style='color:{PALETTE['lightblue']};opacity:0.7;font-style:italic;"
            "margin:-0.3rem 0 0.6rem 0;font-size:0.85rem;'>Playoffs — Coming Soon</div>",
            unsafe_allow_html=True,
        )
    st.sidebar.multiselect("Teams", NHL_TEAMS, key="f_teams")
    st.sidebar.text_input("Player name contains", key="f_name")
    st.sidebar.slider("Minimum GP", min_value=0, max_value=400, step=5, key="f_min_gp")
    st.sidebar.selectbox("DOZI flag", ["All", "RISING", "STABLE", "DECLINING"], key="f_flag")


def render_tnzi_table() -> None:
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
    if sort_col not in display_df.columns:
        sort_col = "TNZI_L" if "TNZI_L" in display_df.columns else display_df.columns[0]

    col_a, col_b = st.columns([3, 1])
    sortable = [c for c in display_df.columns if c not in ("player_name", "team", "pos")]
    with col_a:
        pick = st.selectbox(
            "Sort by", sortable,
            index=sortable.index(sort_col) if sort_col in sortable else 0,
            key="sort_col",
        )
    with col_b:
        st.checkbox("Descending", key="sort_desc")

    display_df = display_df.sort_values(
        by=pick, ascending=not st.session_state.get("sort_desc", True), na_position="last"
    )

    st.dataframe(style_frame(display_df), width="stretch", hide_index=True)
    st.caption(f"Showing {len(display_df):,} players")


def render_tnzi_explainers() -> None:
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
- **ZQoC**: Zone Quality of Competition — weighted average zone metric score of opponents faced. Higher = tougher competition.
- **ZQoL**: Zone Quality of Linemates — weighted average zone metric score of teammates. Higher = better linemates.
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
- Full methodology and source: [github.com/HockeyROI/nhl-analytics](https://github.com/HockeyROI/nhl-analytics)
            """
        )

    with st.expander("Known limitations"):
        st.markdown(
            """
- These are **team outcome** metrics — individual skill AND team system both contribute.
- Carolina Hurricanes system effect is clearly visible — multiple Hurricanes in top 20 across all metrics.
- Current season ZQoL uses pooled linemate data as approximation — players traded mid-season (e.g. Quinn Hughes Dec 2025, Brent Burns) may show stale linemate context. Historical season views use exact linemate data.
- Brent Burns shows #1 current season D in TNZI_L — this is an artifact of his historical Carolina ZQoL. His current raw TNZI (7.2) does not support a #1 ranking.
- Y coordinate not used — corner vs slot play treated identically.
- Zone tracking between events is approximation — last known coordinate assumed.
- These metrics do not predict team winning better than Corsi or xG in multi-season testing.
            """
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
    st.session_state.setdefault("nfi_season", SEASON_OPTIONS[0])
    st.session_state.setdefault("nfi_teams", [])
    st.session_state.setdefault("nfi_name", "")
    st.session_state.setdefault("nfi_min_toi", NFI_TOI_DEFAULT["pooled"])
    st.session_state.setdefault("nfi_last_season", SEASON_OPTIONS[0])

    # Widgets
    st.sidebar.radio("Position", ["All", "Forwards", "Defensemen"], key="nfi_position")
    chosen_season = st.sidebar.selectbox("Season", SEASON_OPTIONS, key="nfi_season")

    # --- FIX ISSUE 1 (slider bug): on season change, reset TOI threshold to
    # the appropriate default BEFORE the slider is instantiated this run.
    if st.session_state.get("nfi_last_season") != chosen_season:
        default = NFI_TOI_DEFAULT["pooled"] if chosen_season == "Pooled (4 seasons)" else NFI_TOI_DEFAULT["season"]
        st.session_state["nfi_min_toi"] = default
        st.session_state["nfi_last_season"] = chosen_season

    st.sidebar.multiselect("Teams", NHL_TEAMS, key="nfi_teams")
    st.sidebar.text_input("Player name contains", key="nfi_name")
    # Fixed 0..NFI_TOI_MAX range so max_value never conflicts with stored state
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
    df = load_nfi_player()
    if df.empty:
        st.error(
            "NFI player file not found at "
            "`NFI/output/fully_adjusted/player_fully_adjusted.csv`."
        )
        return
    season = st.session_state.get("nfi_season", SEASON_OPTIONS[0])
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
    with st.expander("What each NFI metric means"):
        st.markdown(
            """
- **NFI%** — **Net Front Impact %**. Fenwick attempts (shots on goal + misses + goals, blocks
  excluded) filtered to CNFI (central net-front) and MNFI (mid net-front) zones. Team CNFI+MNFI
  for / (for + against) while the player is on ice. **R² = 0.583 vs standings** — beats
  xG% (0.538) and Corsi (0.397).
- **NFI%_ZA** — Zone-Adjusted using the **empirical OZ/DZ factor of +10.71 pp**
  (not the traditional 3.5 pp, which under-corrects by ~67%).
- **NFI%_3A** — **Three-Adjusted**: zone adjustment + NFQOC + NFQOL.
- **RelNFI_F%** — on-ice minus off-ice team CNFI+MNFI For per 60. Positive = team generates
  more dangerous shots with player on ice.
- **RelNFI_A%** — off-ice minus on-ice CNFI+MNFI Against per 60. Positive = suppresses more.
- **RelNFI%** — net two-way dangerous-zone impact = RelNFI_F% + RelNFI_A%.
- **NFQOC** — **Net Front Quality of Competition** — shared-TOI weighted mean of opponents'
  NFI%, computed linemate-without-me to avoid shared-event collinearity.
- **NFQOL** — **Net Front Quality of Linemates** — same approach for teammates.
- **NFI%_3A_MOM** — year-over-year change in NFI%_3A. Positive = ascending.
            """
        )

    with st.expander("Methodology"):
        st.markdown(
            """
- NFI zones (CNFI, MNFI) derived from shot-density clustering of NHL API x/y coordinates
- Fenwick events only (shots on goal + missed + goals). Blocks excluded — their coordinates
  are recorded at the blocker's location, not the shooter's.
- 5v5 ES regulation only (state = ES in `shots_tagged.csv`)
- Per-player on-ice attribution: each ES shot event counts for all skaters on-ice
- Empirical zone factor from league-pooled OZ/DZ faceoff-shift analysis = **+10.71 pp**
- NFQOC / NFQOL use a **linemate-without-me** correction — teammate `j`'s rating is
  recomputed excluding events where both `i` and `j` were on ice, preventing β_QoL ≈ 1.0
- 3A = raw − zone × (OZ_ratio − 0.5) − β_NFQOC × (NFQOC − mean) − β_NFQOL × (NFQOL − mean)
- Minimum thresholds: 2000 ES minutes pooled, 500 ES minutes current season
- Source: [github.com/HockeyROI/nhl-analytics](https://github.com/HockeyROI/nhl-analytics)
            """
        )

    with st.expander("Known limitations"):
        st.markdown(
            """
- NFI% is an **on-ice** metric — shared-event context effects are partially corrected via
  linemate-without-me but residual team-system effects remain.
- Carolina forwards (Fast, Staal, Martinook) rank high on NFI%_ZA because of CAR's system.
  3A adjusts for it but doesn't fully eliminate it.
- 2022-23 through 2025-26 available. 2021-22 not included (raw PBP starts 2022-23).
- **Rel-NFI metrics do not aggregate to team points** (they demean to zero within each team).
  Use Rel-NFI for individual ranking; use NFI%_ZA / 3A for team-level inference.
- MOM for 2025-26 is partial-season through the latest update.
            """
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

    tab_tnzi, tab_nfi = st.tabs(["Zone Impact (TNZI)", "Net Front Impact (NFI)"])

    with tab_tnzi:
        render_tnzi_disclaimer()
        render_tnzi_sidebar()
        render_tnzi_table()
        render_tnzi_explainers()

    with tab_nfi:
        render_nfi_disclaimer()
        render_nfi_sidebar()
        render_nfi_table()
        render_nfi_explainers()

    render_footer()


if __name__ == "__main__":
    main()
