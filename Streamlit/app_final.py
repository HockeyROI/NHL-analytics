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
FLAG_COLOR = {"RISING": "#44AA66", "STABLE": "#FFB700", "DECLINING": "#CC3333"}

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

# NFI season key mapping
NFI_SEASON_KEY = {
    "Pooled (4 seasons)": "pooled",
    "Current Season (2025-26)": "20252026",
    "2024-25": "20242025",
    "2023-24": "20232024",
    "2022-23": "20222023",
}


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


@st.cache_data(show_spinner=False)
def load_nfi_player() -> pd.DataFrame:
    """Load full NFI player dataset."""
    fp = NFI_ADJ / "player_fully_adjusted.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    if "toi_min" not in df.columns and "toi_sec" in df.columns:
        df["toi_min"] = df["toi_sec"] / 60
    return df


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

        /* Tabs */
        [data-testid="stTabs"] button {{
            color: {PALETTE['text']} !important;
            background-color: {PALETTE['surface']};
        }}
        [data-testid="stTabs"] button[aria-selected="true"] {{
            color: {PALETTE['orange']} !important;
            border-bottom: 3px solid {PALETTE['orange']};
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
# Filtering (Zone Impact tab)
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
# UI sections — common
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
# Zone Impact (TNZI) tab
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


def render_tnzi_table() -> None:
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
- Full methodology and source code: [github.com/HockeyROI/nhl-analytics](https://github.com/HockeyROI/nhl-analytics)
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
    st.session_state.setdefault("nfi_min_toi", 2000)

    st.sidebar.radio("Position", ["All", "Forwards", "Defensemen"], key="nfi_position")
    season = st.sidebar.selectbox("Season", SEASON_OPTIONS, key="nfi_season")
    st.sidebar.multiselect("Teams", NHL_TEAMS, key="nfi_teams")
    st.sidebar.text_input("Player name contains", key="nfi_name")
    # TOI slider default adapts to season
    default_min = 2000 if season == "Pooled (4 seasons)" else 500
    max_min = 7500 if season == "Pooled (4 seasons)" else 1600
    step = 100
    st.sidebar.slider(
        "Minimum ES TOI (min)",
        min_value=0,
        max_value=max_min,
        value=default_min,
        step=step,
        key="nfi_min_toi",
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
            vals = pd.to_numeric(g[col], errors="coerce") if col in g.columns else pd.Series(dtype=float)
            m = vals.notna() & (toi > 0)
            if m.sum() == 0:
                return np.nan
            return float(np.average(vals[m], weights=toi[m]))

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
            "RelNFI_pct": tw("RelNFI_pct"),
            "NFQOC": tw("NFQOC"),
            "NFQOL": tw("NFQOL"),
            # MOM only valid on single-season view; leave NaN in pooled
            "NFI_pct_3A_MOM": np.nan,
        })
    return pd.DataFrame(rows)


def _filter_nfi(df: pd.DataFrame, season_opt: str) -> pd.DataFrame:
    """Apply sidebar filters and return the prepared NFI table."""
    if df.empty:
        return df

    # Position filter
    position = st.session_state.get("nfi_position", "All")
    if position == "Forwards":
        df = df[df["position"] == "F"]
    elif position == "Defensemen":
        df = df[df["position"] == "D"]

    # Season filter / pooling
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

    # Team / name filters
    teams = st.session_state.get("nfi_teams", [])
    if teams:
        out = out[out["team"].isin(teams)]
    name_q = (st.session_state.get("nfi_name", "") or "").strip().lower()
    if name_q:
        out = out[out["player_name"].str.lower().str.contains(name_q, na=False)]

    # TOI filter
    min_toi = st.session_state.get("nfi_min_toi", 2000)
    out = out[out["toi_min"].fillna(0) >= min_toi]

    # Small-sample flag (for pooled: under 2000; for single season: under 500)
    threshold = 2000 if season_key == "pooled" else 500
    out["small_sample"] = out["toi_min"] < threshold

    return out.reset_index(drop=True)


def _nfi_display(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns + format for display."""
    if df.empty:
        return df
    out = df.copy()
    # Add asterisk to names of small-sample players
    if "small_sample" in out.columns:
        out["player_name"] = out.apply(
            lambda r: f"{r['player_name']} *" if r["small_sample"] else r["player_name"],
            axis=1,
        )
    out = out.rename(columns={
        "player_name": "Player",
        "position":    "Pos",
        "team":        "Team",
        "toi_min":     "TOI",
        "NFI_pct_ZA":  "NFI%_ZA",
        "NFI_pct_3A":  "NFI%_3A",
        "RelNFI_F_pct":"RelNFI_F%",
        "RelNFI_A_pct":"RelNFI_A%",
        "RelNFI_pct":  "RelNFI%",
        "NFQOC":       "NFQOC",
        "NFQOL":       "NFQOL",
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

    # Sort default: NFI_pct_ZA descending
    filtered = filtered.sort_values("NFI_pct_ZA", ascending=False, na_position="last").reset_index(drop=True)
    filtered.insert(0, "Rank", np.arange(1, len(filtered) + 1))

    display_df = _nfi_display(filtered)
    display_df.insert(0, "#", filtered["Rank"].values)

    st.dataframe(_style_nfi(display_df), width="stretch", hide_index=True)
    caption = f"Showing {len(display_df):,} players — sorted by NFI%_ZA descending"
    if filtered["small_sample"].any():
        n_small = int(filtered["small_sample"].sum())
        caption += f"  |  {n_small} small-sample players flagged with *"
    st.caption(caption)


def render_nfi_explainers() -> None:
    with st.expander("What each NFI metric means"):
        st.markdown(
            """
- **NFI%** — **Net Front Impact %**. Fenwick attempts (shots on goal + misses + goals, blocks
  excluded) filtered to the empirically-derived high-danger zones CNFI (central net-front) and
  MNFI (mid net-front). Measured as team CNFI+MNFI for / (for + against) while the player is on
  the ice. Beats xG% (R² = 0.538) and Corsi (0.397) at predicting team points — **R² = 0.583
  against standings**.
- **NFI%_ZA** — Zone-Adjusted using the **empirical OZ/DZ factor of +10.71 pp** (not the
  traditional 3.5 pp Corsi factor — applying 3.5 pp here under-corrects by ~67%).
- **NFI%_3A** — **Three-Adjusted**: zone adjustment + NFQOC + NFQOL. Individual player contribution
  net of context.
- **RelNFI_F%** — on-ice minus off-ice team CNFI+MNFI **For** rate per 60. Positive = team generates
  more dangerous shots with player on the ice.
- **RelNFI_A%** — off-ice minus on-ice team CNFI+MNFI **Against** rate per 60. Positive = team
  allows fewer dangerous shots with player on the ice.
- **RelNFI%** — net two-way dangerous-zone impact = RelNFI_F% + RelNFI_A%.
- **NFQOC** — **Net Front Quality of Competition** — shared-TOI weighted average of opposing
  skaters' NFI%, computed linemate-without-me to avoid shared-event collinearity.
- **NFQOL** — **Net Front Quality of Linemates** — same approach for teammates.
- **NFI%_3A_MOM** — year-over-year change in NFI%_3A. Positive = ascending. Negative = declining.
            """
        )

    with st.expander("Methodology"):
        st.markdown(
            """
- NFI zones (CNFI, MNFI) derived from shot-density clustering of NHL API x/y coordinates
- Fenwick events only: shot-on-goal + missed-shot + goal. Blocked shots excluded (blocker-side
  coordinates would mis-tag the zone).
- 5v5 even-strength regulation only (state = ES in `shots_tagged.csv`)
- Per-player attribution: each ES shot event counted for all 10 on-ice skaters (Corsi-style)
- Zone adjustment factor derived empirically from OZ/DZ faceoff-shift analysis:
  league-pooled OZ% − DZ% in own-share space = **+10.71 pp** for NFI%
- NFQOC / NFQOL use a **linemate-without-me** correction: when computing teammate `j`'s
  contribution to player `i`'s context, `j`'s rating is recomputed excluding events where
  both `i` and `j` were on the ice — otherwise β_QoL ≈ 1.0 from shared-event collinearity
- 3A = raw − zone adjustment − β_NFQOC × (NFQOC − mean) − β_NFQOL × (NFQOL − mean)
- Minimum thresholds: 2000 ES minutes pooled, 500 ES minutes current season
- Full methodology and source:
  [github.com/HockeyROI/nhl-analytics](https://github.com/HockeyROI/nhl-analytics)
            """
        )

    with st.expander("Known limitations"):
        st.markdown(
            """
- NFI% is an **on-ice** metric — shared-event effects (your shot counts for all 4 linemates too)
  are corrected via linemate-without-me in NFQOC/NFQOL, but residual context effects remain.
- Team-system effects are real and partially persist into 3A: Carolina forwards (Fast, Staal,
  Martinook) rank high on NFI%_ZA because of CAR's system. 3A adjusts for it but doesn't fully
  eliminate it.
- 2022-23 through 2025-26 available. 2021-22 not included (raw PBP data starts 2022-23).
- NFI% regresses well to points at team aggregate (R² = 0.58) but **Rel-NFI metrics do not**
  aggregate to team points (they demean to zero within each team). Use Rel-NFI for individual
  ranking, NFI%_ZA/3A for team-level inference.
- MOM values for 2025-26 are partial-season through the latest update.
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
