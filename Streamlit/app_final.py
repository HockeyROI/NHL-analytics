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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=3600)
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
# Goalie loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_goalie_nfi() -> pd.DataFrame:
    """Pooled goalie GSAx — canonical source.

    Reads the v2 pooled file (NFI/output/goalie_nfi_gsax_pooled_v2.csv)
    built by NFI/scripts/22_pool_goalie_gsax.py. The v2 file is itself
    aggregated up from goalie_nfi_gsax_by_season.csv so the pooled view
    and per-season view share the same CNFI+MNFI denominator and the
    same shots-faced threshold (300 pooled / 100 per season).

    Native columns of the v2 file:
        goalie_id, goalie_name, team, n_seasons, games, total_faced,
        es_toi_min, GSAx, GSAx_per60

    Optional Tier joined from publication_goalies_top60.csv.
    Optional CNFI_rebound_goal_rate / rebound_z joined from
    goalie_rebound_control.csv when present.
    """
    base = REPO_ROOT / "NFI" / "output" / "goalie_nfi_gsax_pooled_v2.csv"
    if not base.exists():
        return pd.DataFrame()
    df = pd.read_csv(base)

    # Backwards-compatible alias so any caller still expecting the legacy
    # column names doesn't break (the live goalie renderer uses GSAx /
    # GSAx_per60 directly, but external consumers may still reference the
    # old NFI_GSAx_cumulative / NFI_GSAx_per60 / ES_TOI_min names).
    if "GSAx" in df.columns:
        df["NFI_GSAx_cumulative"] = df["GSAx"]
    if "GSAx_per60" in df.columns:
        df["NFI_GSAx_per60"] = df["GSAx_per60"]
    if "es_toi_min" in df.columns:
        df["ES_TOI_min"] = df["es_toi_min"]

    # Tier from publication file (still useful colour-tagging)
    pub_fp = REPO_ROOT / "NFI" / "output" / "publication_goalies_top60.csv"
    if pub_fp.exists():
        pub = pd.read_csv(pub_fp)
        df = df.merge(
            pub[["goalie_name", "tier_label_text"]].rename(columns={"tier_label_text": "Tier"}),
            on="goalie_name", how="left",
        )
    if "Tier" not in df.columns:
        df["Tier"] = np.nan

    # Optional rebound-control join
    reb_fp = REPO_ROOT / "NFI" / "output" / "goalie_rebound_control.csv"
    if reb_fp.exists():
        reb = pd.read_csv(reb_fp)
        df = df.merge(
            reb[["goalie_id", "CNFI_rebound_goal_rate", "z_score"]].rename(
                columns={"z_score": "rebound_z"}
            ),
            on="goalie_id", how="left",
        )

    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_goalie_nfi_by_season() -> pd.DataFrame:
    """Per-season goalie GSAx (one row per goalie-season).

    Source: NFI/output/goalie_nfi_gsax_by_season.csv (built by
    NFI/scripts/21_goalie_gsax_by_season.py). CNFI+MNFI denominator,
    minimum 100 dangerous-zone shots-faced per season.
    """
    fp = REPO_ROOT / "NFI" / "output" / "goalie_nfi_gsax_by_season.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    if "season" in df.columns:
        df["season"] = df["season"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Styling — palette + CSS
# ---------------------------------------------------------------------------
PALETTE = {
    # --- White theme (migrated from dark navy, June 2026) ---
    "bg":          "#FFFFFF",   # app background
    "surface":     "#FFFFFF",
    "panel":       "#F0F4F8",   # light blue-grey — filter rows / expanders / cards
    "border":      "#D9E2EC",
    "blue":        "#2E7DC4",
    "lightblue":   "#4AB3E8",
    "text":        "#1B3A5C",   # primary text + headers (navy)
    "text_dark":   "#1B3A5C",   # dropdown / input text (key kept for back-compat)
    "text_secondary": "#888888",
    "input_bg":    "#FFFFFF",
    "orange":      "#FF6B35",
    # momentum (+/-) text colours that read on white
    "rising":      "#2E8B57",
    "stable":      "#888888",
    "declining":   "#C8504F",
    # heatmap / tertile shading (diverging, reads on white)
    "high":        "#3FA66B",
    "mid":         "#E8B43C",
    "low":         "#C8504F",
    # primary-metric emphasis (brand orange family, white text)
    "primary_top": "#F2622E",
    "primary_mid": "#C68A3A",
    "primary_low": "#A23B3B",
}

# Per-series line-chart colors. Orange (primary) is line 1; light blue is the
# 2nd line on every chart; the 3rd line on the 3-line charts (RelNFI, Zone) is
# brand blue. NZI is the orange line in the Zone chart, by request.
_CHART_PRIMARY = PALETTE["orange"]       # #FF6B35
_CHART_SECOND = PALETTE["lightblue"]     # #4AB3E8 light blue
_CHART_THIRD = PALETTE["blue"]           # #2E7DC4 brand blue (reads blue, not black)
_CHART_COLORS = {
    "NFI%": _CHART_PRIMARY,
    "RelNFI%": _CHART_PRIMARY, "RelNFI-A%": _CHART_SECOND, "RelNFI-S%": _CHART_THIRD,
    "NFI-A/60": _CHART_PRIMARY, "NFI-S/60": _CHART_SECOND,
    "NZI": _CHART_PRIMARY, "DZI": _CHART_SECOND, "OZI": _CHART_THIRD,
    "NFI_QG%": _CHART_PRIMARY, "xG_QG%": _CHART_SECOND,
    "NFI-GSAx/60": _CHART_PRIMARY, "QNFS%": _CHART_PRIMARY, "QS-GSAx%": _CHART_SECOND,
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
        /* Sidebar removed — filters live in the main page body. Hide the panel
           and its expand/collapse control entirely. */
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"] {{ display: none !important; }}
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
            background: {PALETTE['panel']};
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
            border-bottom: 3px solid {PALETTE['orange']} !important;
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

        /* FIX 1 — Expander summary stays dark navy with orange text after open */
        [data-testid="stExpander"] details summary {{
            background-color: {PALETTE['bg']} !important;
        }}
        [data-testid="stExpander"] details[open] summary {{
            background-color: {PALETTE['bg']} !important;
            color: {PALETTE['orange']} !important;
        }}
        [data-testid="stExpander"] details[open] summary * {{
            color: {PALETTE['orange']} !important;
        }}
        [data-testid="stExpander"] details[open] summary svg,
        [data-testid="stExpander"] details[open] summary svg path,
        [data-testid="stExpander"] details[open] summary svg polygon {{
            fill: {PALETTE['orange']} !important;
            stroke: {PALETTE['orange']} !important;
        }}

        /* FIX 5 — Metric stat box styling: dark navy bg, white labels, orange values */
        [data-testid="stMetric"] {{
            background-color: {PALETTE['bg']} !important;
            border: 1px solid {PALETTE['orange']};
            border-radius: 6px;
            padding: 0.85rem 1rem;
        }}
        [data-testid="stMetric"] label,
        [data-testid="stMetricLabel"] {{
            color: {PALETTE['text']} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {PALETTE['orange']} !important;
            font-weight: 700 !important;
        }}
        [data-testid="stMetricValue"] * {{
            color: {PALETTE['orange']} !important;
        }}

        /* FIX 11 — Framework toggle buttons */
        [data-testid="stButton"] button[kind="primary"] {{
            background-color: {PALETTE['orange']} !important;
            color: {PALETTE['text']} !important;
            border: none !important;
            font-weight: 700 !important;
        }}
        [data-testid="stButton"] button[kind="secondary"] {{
            background-color: {PALETTE['bg']} !important;
            color: {PALETTE['text']} !important;
            border: 1px solid {PALETTE['blue']} !important;
        }}
        /* FIX 4 — ensure buttons remain tappable on mobile (no overlay layer
           is intercepting clicks; force pointer events + tap optimisations). */
        [data-testid="stButton"] button {{
            pointer-events: auto !important;
            touch-action: manipulation !important;
            -webkit-tap-highlight-color: rgba(0,0,0,0) !important;
            cursor: pointer !important;
            position: relative !important;
            z-index: 100 !important;
        }}

        /* FIX 11 — small mobile-only filter hint */
        @media (max-width: 768px) {{
            .mobile-filter-hint {{
                display: block !important;
                color: {PALETTE['lightblue']};
                font-size: 0.8rem;
                margin-bottom: 0.5rem;
            }}
        }}
        @media (min-width: 769px) {{
            .mobile-filter-hint {{ display: none !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Common UI
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap;">
          <div class="hockeyroi-brand"><span class="hockey">HOCKEY</span><span class="roi">ROI</span></div>
          <div style="color:#2E7DC4; font-size:0.9rem; padding-bottom:0.45rem;">How these metrics work → <strong>Methodology</strong> tab (far right)</div>
        </div>
        <div class="tagline">NHL Net-Front Impact, Zone Impact, Quality Games &amp; Quality Starts (GSAx)</div>
        <div style="color:#888888; font-size:0.85rem; margin-top:0.15rem;">
          <a href="https://github.com/HockeyROI/NHL-analytics/blob/main/docs/METHODOLOGY.md" style="color:#2E7DC4;">Methodology on GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="hockeyroi-footer">
        HockeyROI — <a href="https://hockeyROI.substack.com">hockeyROI.substack.com</a> |
        <a href="https://twitter.com/HockeyROI">@HockeyROI</a> |
        <a href="https://github.com/HockeyROI/NHL-analytics">github.com/HockeyROI/NHL-analytics</a><br/>
        Built from NHL play-by-play data. Methodology open source.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sort_hint() -> None:
    """Small note above a table explaining the native header-sort cycle."""
    st.caption("↕ Click any column header to sort — 1st click ascending, "
               "2nd descending, 3rd clears.")


def _apply_ranks(disp, fmt, cohort, rank_cols, lower_better=()):
    """Append ' (rank)' to each ranked column's DISPLAY string while leaving the
    underlying cell value numeric, so header-sort still orders by the real value.

    Ranks are computed over `cohort` (a frame sharing the display column names),
    #1 = best; columns in `lower_better` rank lowest-value-first. NaN cells get
    no rank. Works via a value→rank map per column (ties share a rank, so the
    map is unambiguous)."""
    lower = set(lower_better)
    for col in rank_cols:
        if col not in disp.columns or col not in cohort.columns or col not in fmt:
            continue
        s = pd.to_numeric(cohort[col], errors="coerce")
        ranks = s.rank(ascending=(col in lower), method="min")
        vmap = {v: int(r) for v, r in zip(s.values, ranks.values)
                if pd.notna(v) and pd.notna(r)}

        def _mk(base_f, vm):
            def f(x):
                if pd.isna(x):
                    return base_f(x)
                r = vm.get(x)
                return f"{base_f(x)} ({r})" if r is not None else base_f(x)
            return f

        fmt[col] = _mk(fmt[col], vmap)
    return fmt


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

        # Only the columns the Players tab displays — raw NFI% + RelNFI family.
        rows.append({
            "player_id": int(pid),
            "player_name": name,
            "position": pos,
            "team": g.sort_values("season")["team"].iloc[-1],
            "toi_min": toi_total,
            "NFI_pct":      tw("NFI_pct"),
            "RelNFI_F_pct": tw("RelNFI_F_pct"),
            "RelNFI_A_pct": tw("RelNFI_A_pct"),
            "RelNFI_pct":   tw("RelNFI_pct"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Methodology tab
# ---------------------------------------------------------------------------
GITHUB_METHODOLOGY_URL = (
    "https://github.com/HockeyROI/NHL-analytics/blob/main/docs/METHODOLOGY.md"
)
TAB_LABELS = ["Player List", "Player Detail", "Goalie List", "Goalie Detail",
              "Trade Analyzer", "Teams", "Referees", "Methodology"]


def _meth_framework(name: str, body: str) -> str:
    return (
        f"<div style='margin:0.55rem 0; max-width:62rem;'>"
        f"<span style='color:{PALETTE['blue']}; font-weight:700;'>{name}</span>"
        f"<span style='color:{PALETTE['text']};'> — {body}</span></div>"
    )


def render_methodology() -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.25rem;'>Methodology</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color:{PALETTE['text']}; font-size:0.97rem; line-height:1.55; max-width:62rem;'>"
        "HockeyROI re-defines high-danger scoring chances around the geometry where shot location "
        "actually predicts conversion, then applies that lens across players, teams, goalies, and "
        "zone deployment. Short summaries are below; the full canonical methodology — definitions, "
        "derivations, locked spot-check values, and verification work — lives in a single "
        "source-of-truth document on GitHub.</p>",
        unsafe_allow_html=True,
    )
    st.link_button("Full methodology →", GITHUB_METHODOLOGY_URL, type="primary")

    st.markdown(
        f"<h3 style='color:{PALETTE['text']}; margin-top:1.2rem;'>The frameworks</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        _meth_framework(
            "NFI — Net-Front Impact",
            "Fenwick shot share (shots + misses + goals, blocks excluded) in the CNFI "
            "(close net-front) and MNFI (mid / high-slot) zones while a player is on ice. "
            "<b>RelNFI%</b> measures net-front impact relative to a player's own team "
            "(on-ice vs off-ice), isolating individual contribution from team strength.",
        )
        + _meth_framework(
            "Quality Games (QG)",
            "Per-game consistency — the share of a player's 5v5 regulation games where their "
            "on-ice danger share beat the position median, on two bases (MoneyPuck xG and NFI), "
            "with a half-credit rule for exact-median ties.",
        )
        + _meth_framework(
            "Teams",
            "Team-level CNFI+MNFI share, plus a roster-talent-vs-on-ice-result “two ways” "
            "comparison that flags teams whose talent and results diverge.",
        )
        + _meth_framework(
            "Goalies",
            "GSAx-based, three lenses: <b>NFI-GSAx</b> (net-front goals saved above expected), "
            "<b>QNFS%</b> (consistency of beating expected on net-front shots), and "
            "<b>QS-GSAx</b> (same idea on all shots). Qualifying floors differ by metric, so the "
            "cohorts differ — by design.",
        )
        + _meth_framework(
            "Zone Impact",
            "DZI / NZI / OZI — three position-normalized 0–10 lenses for offensive-zone time after "
            "defensive / neutral / offensive faceoffs. Independent lenses, not a hierarchy; a complete "
            "player rates well across all three.",
        )
        + _meth_framework(
            "Referees",
            "Penalty-call environment by official across 2023-24 → 2025-26.",
        ),
        unsafe_allow_html=True,
    )

    # Vollman disambiguation callout
    st.markdown(
        f"""
        <div style="border:2px solid {PALETTE['orange']}; background:{PALETTE['panel']};
             border-radius:6px; padding:1rem 1.25rem; margin:1.25rem 0; max-width:62rem;">
          <div style="color:{PALETTE['orange']}; font-weight:700; margin-bottom:0.3rem;">
            A note on &ldquo;Quality Starts&rdquo;</div>
          <div style="color:{PALETTE['text']}; font-size:0.94rem; line-height:1.5;">
            HockeyROI's goalie quality-start metrics are <b>not</b> Robert Vollman's Quality Starts
            (~2009, defined on save% vs league average). Here a quality game is
            <b>per-game GSAx &ge; 0</b> — the goalie beat expected on a danger / xG-weighted basis —
            and there are <b>two</b> parallel definitions (QNFS% on net-front shots, QS-GSAx on all
            shots). Don't map these to Vollman's metric, or to each other.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<p style='color:{PALETTE['text_secondary']}; font-size:0.85rem; max-width:62rem;'>"
        "Cohorts differ by metric by design — qualifying floors vary (e.g. pooled goalie NFI-GSAx "
        "requires ≥300 net-front shots; QNFS% requires ≥25 games in a season), so a player or "
        "goalie may appear in one table and not another. Data is current through the 2025-26 season; "
        "daily auto-updates are paused, with refreshes moving to roughly every 10 games.</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Players tab — NFI + Quality Games (+ Zone Impact in the Pooled view)
# ---------------------------------------------------------------------------
SEASON_KEY = {
    "2025-26": "20252026",
    "2024-25": "20242025",
    "2023-24": "20232024",
    "2022-23": "20222023",
    "2yr (2024–2026)": "pooled_2yr",
    "4yr (2022-2026)": "pooled",
}

# Seasons covered by the "2yr (2024–2026)" pooled-style view. Loaders that
# aggregate across seasons restrict to these when the season key is
# "pooled_2yr"; tabs without per-season support for 2yr fall back gracefully.
POOLED_2YR_SEASONS = ["20242025", "20252026"]


@st.cache_data(show_spinner=False, ttl=3600)
def load_qg_player_season() -> pd.DataFrame:
    """Quality Games per (player, season) — xG_QG% / NFI_QG% / GP / qual GP."""
    fp = REPO_ROOT / "Quality_Games" / "output" / "per_player_season.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype("Int64")
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_zone_pooled() -> pd.DataFrame:
    """Pooled NZI / DZI / OZI (0–10) from tnzi_adjusted_{forwards,defense}.csv.
    Name-keyed (the zone files carry no player_id); a `_pos_group` column is
    added so the merge to NFI is on (player_name, pos-group) — this guards the
    rare same-name / different-position case (e.g. the two Sebastian Ahos)."""
    frames = []
    for pos_file, grp in (("forwards", "F"), ("defense", "D")):
        fp = ADJ / f"tnzi_adjusted_{pos_file}.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        keep = [c for c in ("player_name", "NZI", "DZI", "OZI") if c in d.columns]
        d = d[keep].copy()
        d["_pos_group"] = grp
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    z = pd.concat(frames, ignore_index=True)
    return z.drop_duplicates(subset=["player_name", "_pos_group"], keep="first")


@st.cache_data(show_spinner=False, ttl=3600)
def load_zone_2yr() -> pd.DataFrame:
    """2-year (2024-25 + 2025-26) pooled NZI/DZI/OZI (0–10) from the uncapped
    2yr_recent_{NZI,DZI,OZI}_{forwards,defense}.csv files. Lens is in the
    filename; `raw_score` is the 0–10 value. Name-keyed (no player_id), so the
    merge mirrors load_zone_pooled exactly: on (player_name, _pos_group).

    DUPLICATE-NAME GUARD: a name can repeat within a position group (two
    distinct players, e.g. Sam/Samuel splits). Within each lens file we keep the
    higher-GP_in_scope row deterministically before merging the three lenses, so
    the final frame is unique per (player_name, _pos_group) and a left-join to
    the Players frame never multiplies rows or attaches the wrong player's zone.
    """
    sub = ADJ / "per_season"
    frames = []
    for pos_file, grp in (("forwards", "F"), ("defense", "D")):
        merged = None
        for m in ("NZI", "DZI", "OZI"):
            fp = sub / f"2yr_recent_{m}_{pos_file}.csv"
            if not fp.exists():
                continue
            d = pd.read_csv(fp)
            if "player_name" not in d.columns or "raw_score" not in d.columns:
                continue
            gp = d["GP_in_scope"] if "GP_in_scope" in d.columns else 0
            d = pd.DataFrame({"player_name": d["player_name"], "_gp": gp,
                              m: d["raw_score"]})
            d = (d.sort_values("_gp", ascending=False)
                   .drop_duplicates("player_name", keep="first")
                   .drop(columns="_gp"))
            merged = d if merged is None else merged.merge(d, on="player_name",
                                                           how="outer")
        if merged is not None:
            merged["_pos_group"] = grp
            frames.append(merged)
    if not frames:
        return pd.DataFrame()
    z = pd.concat(frames, ignore_index=True)
    return z.drop_duplicates(subset=["player_name", "_pos_group"], keep="first")


def _qg_pooled(qg: pd.DataFrame) -> pd.DataFrame:
    """Career-pooled QG: rates = total quality games / total qualifying GP."""
    if qg.empty:
        return qg
    g = qg.groupby("player_id").agg(
        GP=("GP", "sum"),
        qualifying_GP=("qualifying_GP", "sum"),
        _xc=("xG_QG_count", "sum"), _xq=("xG_qual_GP", "sum"),
        _nc=("NFI_QG_count", "sum"), _nq=("NFI_qual_GP", "sum"),
    ).reset_index()
    g["xG_QG_pct"] = np.where(g["_xq"] > 0, g["_xc"] / g["_xq"], np.nan)
    g["NFI_QG_pct"] = np.where(g["_nq"] > 0, g["_nc"] / g["_nq"], np.nan)
    return g[["player_id", "GP", "qualifying_GP", "xG_QG_pct", "NFI_QG_pct"]]


@st.cache_data(show_spinner=False, ttl=3600)
def load_as_counts() -> pd.DataFrame:
    """Per-(player_id, season) building blocks for raw attack/suppress per-60.

    Source: NFI/output/player_counts_by_state_zone_per_season.csv (Gate D).
    Keep ES, zone in {CNFI, MNFI}; sum on-ice for/against attempts per
    (player_id, season). ES TOI is per-(player, season, state) — identical
    across the zone rows — so it's taken once (`first`), NOT summed over zones.
    Returns counts + ES TOI (additive) so any scope's rate is the correct
    ratio-of-sums: sum(counts) / sum(ES TOI) * 60.
    """
    fp = REPO_ROOT / "NFI" / "output" / "player_counts_by_state_zone_per_season.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df = df[(df["state"] == "ES") & (df["zone"].isin(["CNFI", "MNFI"]))].copy()
    if df.empty:
        return pd.DataFrame()
    df["season"] = df["season"].astype(str)
    g = df.groupby(["player_id", "season"]).agg(
        for_att=("onice_for_att", "sum"),
        ag_att=("onice_ag_att", "sum"),
        es_toi_min=("toi_min", "first"),   # ES TOI constant across CNFI/MNFI rows
    ).reset_index()
    return g


def _as_rates(scope_key: str) -> pd.DataFrame:
    """Raw attack/suppress per-60 for one scope, by ratio-of-sums.

    scope_key is the SEASON_KEY value: "pooled" (4yr 2022-26, EXCLUDES 2021-22),
    "pooled_2yr" (2024-26), or a single-season string like "20252026". Counts
    and ES TOI are summed across the scope's seasons, then divided — pooling on
    rates would be wrong; pooling on counts+TOI is correct.
    """
    g = load_as_counts()
    if g.empty:
        return pd.DataFrame()
    if scope_key == "pooled":
        sub = g[g["season"].isin(POOLED_SEASONS)]            # 4yr, no 2021-22
    elif scope_key == "pooled_2yr":
        sub = g[g["season"].isin(POOLED_2YR_SEASONS)]        # 2024-25 + 2025-26
    else:
        sub = g[g["season"] == scope_key]                    # single season
    if sub.empty:
        return pd.DataFrame()
    agg = sub.groupby("player_id").agg(
        for_att=("for_att", "sum"), ag_att=("ag_att", "sum"),
        es_toi_min=("es_toi_min", "sum"),
    ).reset_index()
    ok = agg["es_toi_min"] > 0
    agg["NFI_A_rate"] = np.where(ok, agg["for_att"] / agg["es_toi_min"] * 60.0, np.nan)
    agg["NFI_S_rate"] = np.where(ok, agg["ag_att"] / agg["es_toi_min"] * 60.0, np.nan)
    return agg[["player_id", "NFI_A_rate", "NFI_S_rate"]]


# --- In-tab profiles (Gate F): per-season trends ----------------------------
PROFILE_SEASONS = ["20222023", "20232024", "20242025", "20252026"]  # 4yr, excl 2021-22
SEASON_DISPLAY = {"20222023": "2022-23", "20232024": "2023-24",
                  "20242025": "2024-25", "20252026": "2025-26"}


@st.cache_data(show_spinner=False, ttl=3600)
def load_zone_per_season(season: str | None = None) -> pd.DataFrame:
    """Per-season NZI/DZI/OZI (0–10), name-keyed. Same higher-GP duplicate-name
    guard and (player_name, _pos_group) keying as load_zone_2yr.

    season=None → long multi-season frame (season, player_name, _pos_group,
    NZI, DZI, OZI) used by the per-player trend. season="20252026" → just that
    season's rows keyed on (player_name, _pos_group), season column dropped, for
    a single-season leaderboard join (mirrors load_zone_pooled's shape)."""
    sub = ADJ / "per_season"
    out = []
    seasons = [season] if season is not None else PROFILE_SEASONS
    for ssn in seasons:
        for pos_file, grp in (("forwards", "F"), ("defense", "D")):
            merged = None
            for m in ("NZI", "DZI", "OZI"):
                fp = sub / f"{ssn}_{m}_{pos_file}.csv"
                if not fp.exists():
                    continue
                d = pd.read_csv(fp)
                if "player_name" not in d.columns or "raw_score" not in d.columns:
                    continue
                gp = d["GP_in_scope"] if "GP_in_scope" in d.columns else 0
                d = pd.DataFrame({"player_name": d["player_name"], "_gp": gp,
                                  m: d["raw_score"]})
                d = (d.sort_values("_gp", ascending=False)
                       .drop_duplicates("player_name", keep="first")
                       .drop(columns="_gp"))
                merged = d if merged is None else merged.merge(d, on="player_name",
                                                               how="outer")
            if merged is not None:
                merged["season"] = ssn
                merged["_pos_group"] = grp
                out.append(merged)
    if not out:
        return pd.DataFrame()
    full = pd.concat(out, ignore_index=True)
    if season is not None:
        return full.drop(columns=["season"]).reset_index(drop=True)
    return full


def _player_trend(pid: int) -> pd.DataFrame:
    """Per-season (2022-23..2025-26, excl 2021-22) metric trend for one
    player_id. CRITICAL: every source's season is normalized to an 8-digit
    STRING before merging (player_fully_adjusted is str, load_as_counts is str,
    QG is str) — a str/int mismatch would silently empty the merge. Ordered
    ascending by season."""
    pid = int(pid)
    nfi = load_nfi_player()
    if nfi.empty:
        return pd.DataFrame()
    p = nfi[nfi["player_id"] == pid].copy()
    if p.empty:
        return pd.DataFrame()
    p["season"] = p["season"].astype(str)
    p = p[p["season"].isin(PROFILE_SEASONS)]
    trend = p[["season", "NFI_pct", "RelNFI_pct", "RelNFI_F_pct", "RelNFI_A_pct"]].rename(
        columns={"NFI_pct": "NFI%", "RelNFI_pct": "RelNFI%",
                 "RelNFI_F_pct": "RelNFI-A%", "RelNFI_A_pct": "RelNFI-S%"})

    # Raw attack/suppress per-60 (ES CNFI+MNFI on-ice for/against) per season.
    g = load_as_counts()
    if not g.empty:
        a = g[g["player_id"] == pid].copy()
        a["season"] = a["season"].astype(str)
        a = a[a["season"].isin(PROFILE_SEASONS)]
        ok = a["es_toi_min"] > 0
        a["NFI-A/60"] = np.where(ok, a["for_att"] / a["es_toi_min"] * 60.0, np.nan)
        a["NFI-S/60"] = np.where(ok, a["ag_att"] / a["es_toi_min"] * 60.0, np.nan)
        trend = trend.merge(a[["season", "NFI-A/60", "NFI-S/60"]], on="season", how="outer")

    # Zone (name-keyed) — resolve this player's (name, pos-group) from the NFI
    # row; Pettersson/Aho split by pos-group, no same-name same-position dupes.
    name = p["player_name"].iloc[0]
    pos_group = "D" if str(p["position"].iloc[0]) == "D" else "F"
    z = load_zone_per_season()
    if not z.empty:
        zz = z[(z["player_name"] == name) & (z["_pos_group"] == pos_group)]
        if not zz.empty:
            zcols = ["season"] + [c for c in ("NZI", "DZI", "OZI") if c in zz.columns]
            trend = trend.merge(zz[zcols], on="season", how="outer")

    # Quality Games per season.
    qg = load_qg_player_season()
    if not qg.empty:
        q = qg[qg["player_id"] == pid].copy()
        q["season"] = q["season"].astype(str)
        q = q[q["season"].isin(PROFILE_SEASONS)]
        keep = ["season"] + [c for c in ("NFI_QG_pct", "xG_QG_pct") if c in q.columns]
        q = q[keep].rename(columns={"NFI_QG_pct": "NFI_QG%", "xG_QG_pct": "xG_QG%"})
        trend = trend.merge(q, on="season", how="outer")

    trend = trend[trend["season"].isin(PROFILE_SEASONS)].copy()
    # Always show every season 2022-23 → 2025-26 as a row, even ones before the
    # player debuted (blank cells), so the trend table has a consistent shape.
    missing = [s for s in PROFILE_SEASONS if s not in set(trend["season"])]
    if missing:
        trend = pd.concat([trend, pd.DataFrame({"season": missing})], ignore_index=True)
    trend["Season"] = trend["season"].map(SEASON_DISPLAY).fillna(trend["season"])
    return trend.sort_values("season").reset_index(drop=True)


def _league_rank(series, value, lower=False):
    """Competition rank of `value` within the full-league `series` (#1 = best,
    ties share a rank). lower=True → lowest value is best. None if no value."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if pd.isna(value) or s.empty:
        return None
    better = int((s < value).sum()) if lower else int((s > value).sum())
    return better + 1


def _player_season_ranks(pid: int) -> dict:
    """For one player, the per-season LEAGUE rank of each display metric among
    ALL skaters that season (#1 = best; NFI-S/60 lowest = #1). Returns
    {display_col: {season_str: rank}}."""
    pid = int(pid)
    out = {}
    nfi = load_nfi_player()
    if not nfi.empty:
        nfi = nfi.copy()
        nfi["season"] = nfi["season"].astype(str)
        for disp_c, src in (("NFI%", "NFI_pct"), ("RelNFI%", "RelNFI_pct"),
                            ("RelNFI-A%", "RelNFI_F_pct"), ("RelNFI-S%", "RelNFI_A_pct")):
            if src not in nfi.columns:
                continue
            d = {}
            for ssn in PROFILE_SEASONS:
                sub = nfi[nfi["season"] == ssn]
                pv = sub.loc[sub["player_id"] == pid, src]
                if len(pv):
                    d[ssn] = _league_rank(sub[src], pv.iloc[0])
            out[disp_c] = d
    g = load_as_counts()
    if not g.empty:
        g = g.copy()
        g["season"] = g["season"].astype(str)
        ok = g["es_toi_min"] > 0
        g["NFI-A/60"] = np.where(ok, g["for_att"] / g["es_toi_min"] * 60.0, np.nan)
        g["NFI-S/60"] = np.where(ok, g["ag_att"] / g["es_toi_min"] * 60.0, np.nan)
        for disp_c, low in (("NFI-A/60", False), ("NFI-S/60", True)):
            d = {}
            for ssn in PROFILE_SEASONS:
                sub = g[g["season"] == ssn]
                pv = sub.loc[sub["player_id"] == pid, disp_c]
                if len(pv):
                    d[ssn] = _league_rank(sub[disp_c], pv.iloc[0], lower=low)
            out[disp_c] = d
    z = load_zone_per_season()
    if not z.empty and not nfi.empty:
        prow = nfi[nfi["player_id"] == pid]
        if len(prow):
            name = prow["player_name"].iloc[0]
            pos_group = "D" if str(prow["position"].iloc[0]) == "D" else "F"
            for m in ("NZI", "DZI", "OZI"):
                if m not in z.columns:
                    continue
                d = {}
                for ssn in PROFILE_SEASONS:
                    sub = z[z["season"] == ssn]  # all skaters (F+D) that season
                    pv = sub.loc[(sub["player_name"] == name)
                                 & (sub["_pos_group"] == pos_group), m]
                    if len(pv) and pd.notna(pv.iloc[0]):
                        d[ssn] = _league_rank(sub[m], pv.iloc[0])
                out[m] = d
    qg = load_qg_player_season()
    if not qg.empty:
        qg = qg.copy()
        qg["season"] = qg["season"].astype(str)
        for disp_c, src in (("NFI_QG%", "NFI_QG_pct"), ("xG_QG%", "xG_QG_pct")):
            if src not in qg.columns:
                continue
            d = {}
            for ssn in PROFILE_SEASONS:
                sub = qg[qg["season"] == ssn]
                pv = sub.loc[sub["player_id"] == pid, src]
                if len(pv) and pd.notna(pv.iloc[0]):
                    d[ssn] = _league_rank(sub[src], pv.iloc[0])
            out[disp_c] = d
    return out


def _player_profile_table(pid: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Rank-annotated per-season trend table for a player. Returns
    (display_df, trend, metric_cols): display_df has string cells (value + league
    rank); trend is the numeric frame (for charts). Empty display_df if no data."""
    trend = _player_trend(pid)
    if trend.empty:
        return pd.DataFrame(), trend, []
    share_cols = ["RelNFI%", "RelNFI-A%", "RelNFI-S%", "NFI%"]
    rate_cols = ["NFI-A/60", "NFI-S/60"]
    zone_cols = ["NZI", "DZI", "OZI"]
    qg_cols = ["NFI_QG%", "xG_QG%"]
    metric_cols = [c for c in share_cols + rate_cols + zone_cols + qg_cols
                   if c in trend.columns]

    # Per-season LEAGUE rank (all skaters that season) appended to each cell.
    ranks = _player_season_ranks(pid)
    _b = {}
    for c in ("NFI%", "NFI_QG%", "xG_QG%"):
        _b[c] = lambda v: f"{v * 100:.1f}%"
    for c in ("RelNFI%", "RelNFI-A%", "RelNFI-S%"):
        _b[c] = lambda v: f"{v:+.2f}"
    for c in ("NFI-A/60", "NFI-S/60", "NZI", "DZI", "OZI"):
        _b[c] = lambda v: f"{v:.1f}"
    rows = []
    for _, r in trend.iterrows():
        ssn = r["season"]
        row = {"Season": r["Season"]}
        for c in metric_cols:
            v = r[c]
            if pd.isna(v):
                row[c] = "—"
            else:
                txt = _b.get(c, lambda v: f"{v}")(v)
                rk = ranks.get(c, {}).get(ssn)
                row[c] = f"{txt} ({rk})" if rk is not None else txt
        rows.append(row)
    return pd.DataFrame(rows, columns=["Season"] + metric_cols), trend, metric_cols


def _render_player_profile(pid: int) -> None:
    """Per-season trend table + auto-showing line charts for one player."""
    disp, trend, metric_cols = _player_profile_table(pid)
    if disp.empty:
        st.info("No per-season data available for this player.")
        return
    st.caption("Each value shows its **(rank)** — league rank among all skaters "
               "that season. NFI-S/60 (shots against): lowest = #1.")
    st.dataframe(disp, width="stretch", hide_index=True)

    def _chart(title: str, cols: list[str]) -> None:
        ys = [c for c in cols if c in trend.columns and trend[c].notna().any()]
        if not ys:
            return
        st.caption(title)
        st.line_chart(trend.set_index("Season")[ys],
                      color=[_CHART_COLORS.get(c, _CHART_SECOND) for c in ys])

    # Scales differ across families — one chart per scale so none flattens.
    # NFI% (0–1 absolute share) is split from the RelNFI family (points, ~±5).
    _chart("NFI% (share)", ["NFI%"])
    _chart("RelNFI family (RelNFI%, RelNFI-A%, RelNFI-S%)",
           ["RelNFI%", "RelNFI-A%", "RelNFI-S%"])
    _chart("Raw net-front rate per 60 (NFI-A/60, NFI-S/60)", ["NFI-A/60", "NFI-S/60"])
    _chart("Zone Impact 0–10 (NZI, DZI, OZI)", ["NZI", "DZI", "OZI"])
    _chart("Quality Games % (NFI_QG%, xG_QG%)", ["NFI_QG%", "xG_QG%"])


# ===========================================================================
# Playoff loaders — every producer wrote per playoff season PLUS an
# `all_playoffs` pooled row. The app shows only the pooled view (per the spec);
# the Min-TOI / Min-Shots sliders do the thresholding. Each loader returns the
# all_playoffs rows in the same schema its regular counterpart yields.
# The playoff player NFI build now derives the team-relative RelNFI family
# (same on-ice − off-ice per-60 definition as the regular build), so the
# playoff Players view shows RelNFI%/-A%/-S% and sorts by RelNFI% like regular.
# ===========================================================================
PLAYOFF_SCOPE = "all_playoffs"


@st.cache_data(show_spinner=False, ttl=3600)
def load_nfi_player_playoffs() -> pd.DataFrame:
    fp = NFI_ADJ / "player_fully_adjusted_playoffs.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    if "toi_min" not in df.columns and "toi_sec" in df.columns:
        df["toi_min"] = df["toi_sec"] / 60
    return df[df["season"] == PLAYOFF_SCOPE].copy()


@st.cache_data(show_spinner=False, ttl=3600)
def load_qg_player_playoffs() -> pd.DataFrame:
    fp = REPO_ROOT / "Quality_Games" / "output" / "per_player_season_playoffs.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    df = df[df["season"] == PLAYOFF_SCOPE].copy()
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype("Int64")
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_as_counts_playoffs() -> pd.DataFrame:
    """Raw attack/suppress per-60 (ES CNFI+MNFI on-ice for/against) per player,
    pooled all_playoffs, by ratio-of-sums — mirrors _as_rates."""
    fp = REPO_ROOT / "NFI" / "output" / "player_counts_by_state_zone_playoffs.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    df = df[(df["season"] == PLAYOFF_SCOPE) & (df["state"] == "ES")
            & (df["zone"].isin(["CNFI", "MNFI"]))].copy()
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("player_id").agg(
        for_att=("onice_for_att", "sum"),
        ag_att=("onice_ag_att", "sum"),
        es_toi_min=("toi_min", "first"),   # ES TOI constant across CNFI/MNFI rows
    ).reset_index()
    ok = g["es_toi_min"] > 0
    g["NFI_A_rate"] = np.where(ok, g["for_att"] / g["es_toi_min"] * 60.0, np.nan)
    g["NFI_S_rate"] = np.where(ok, g["ag_att"] / g["es_toi_min"] * 60.0, np.nan)
    return g[["player_id", "NFI_A_rate", "NFI_S_rate"]]


@st.cache_data(show_spinner=False, ttl=3600)
def load_zone_playoffs() -> pd.DataFrame:
    """Name-keyed playoff NZI/DZI/OZI for the all_playoffs pool, (player_name,
    _pos_group)-keyed exactly like load_zone_pooled."""
    zp = ZONES / "output" / "playoffs"
    frames = []
    for pos_file, grp in (("forwards", "F"), ("defense", "D")):
        fp = zp / f"tnzi_adjusted_{pos_file}_playoffs.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        d["season"] = d["season"].astype(str)
        d = d[d["season"] == PLAYOFF_SCOPE]
        keep = [c for c in ("player_name", "NZI", "DZI", "OZI") if c in d.columns]
        d = d[keep].copy()
        d["_pos_group"] = grp
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    z = pd.concat(frames, ignore_index=True)
    return z.drop_duplicates(subset=["player_name", "_pos_group"], keep="first")


def _build_players_frame(season_label: str, playoffs: bool = False) -> tuple[pd.DataFrame, bool]:
    """Return (long per-player frame, is_pooled). NFI + QG, plus NZI/DZI/OZI in
    the pooled view only (zone data has no season axis)."""
    if playoffs:
        nfi = load_nfi_player_playoffs()
        if nfi.empty:
            return pd.DataFrame(), True
        base = nfi.copy()
        qg = load_qg_player_playoffs()
        if not qg.empty:
            qcols = ["player_id", "GP", "qualifying_GP", "xG_QG_pct", "NFI_QG_pct"]
            base = base.merge(qg[[c for c in qcols if c in qg.columns]],
                              on="player_id", how="left")
        zone = load_zone_playoffs()
        if not zone.empty:
            base["_pos_group"] = np.where(base["position"] == "D", "D", "F")
            base = base.merge(zone, on=["player_name", "_pos_group"], how="left")
        as_df = load_as_counts_playoffs()
        if not as_df.empty:
            base = base.merge(as_df, on="player_id", how="left")
        return base, True

    nfi = load_nfi_player()
    if nfi.empty:
        return pd.DataFrame(), False
    qg = load_qg_player_season()
    key = SEASON_KEY.get(season_label, "pooled")
    # "pooled_2yr" is treated like the full pooled view but restricted to the
    # 2024–2026 seasons for the per-season-aware sources (NFI, QG). Zone Impact
    # has no season axis, so the 2yr view shows the all-season pooled zone
    # values (noted in the caption) — FOLLOW-UP: a true 2yr zone build.
    is_pooled = key in ("pooled", "pooled_2yr")

    if is_pooled:
        nfi_src = nfi if key == "pooled" else nfi[nfi["season"].isin(POOLED_2YR_SEASONS)]
        base = _aggregate_nfi_pooled(nfi_src)
        if not qg.empty:
            qg_src = qg if key == "pooled" else qg[qg["season"].isin(POOLED_2YR_SEASONS)]
            base = base.merge(_qg_pooled(qg_src), on="player_id", how="left")
        # Zone Impact: the 2yr view uses the real 2024-25+2025-26 pool
        # (2yr_recent files); the full pooled view uses the all-season build.
        zone = load_zone_2yr() if key == "pooled_2yr" else load_zone_pooled()
        if not zone.empty and not base.empty:
            base["_pos_group"] = np.where(base["position"] == "D", "D", "F")
            base = base.merge(zone, on=["player_name", "_pos_group"], how="left")
    else:
        base = nfi[nfi["season"] == SEASON_KEY[season_label]].copy()
        if not qg.empty:
            qcols = ["player_id", "season", "GP", "qualifying_GP",
                     "xG_QG_pct", "NFI_QG_pct"]
            base = base.merge(qg[[c for c in qcols if c in qg.columns]],
                              on=["player_id", "season"], how="left")
        # Per-season Zone Impact (uncapped per-season files), name-keyed on
        # (player_name, pos-group) like the pooled/2yr zone joins.
        zone = load_zone_per_season(SEASON_KEY[season_label])
        if not zone.empty and not base.empty:
            base["_pos_group"] = np.where(base["position"] == "D", "D", "F")
            base = base.merge(zone, on=["player_name", "_pos_group"], how="left")

    # Raw attack/suppress per-60 (ES CNFI+MNFI on-ice for/against), scoped to the
    # same seasons as the view via ratio-of-sums. Joins on player_id.
    as_df = _as_rates(key)
    if not as_df.empty and not base.empty:
        base = base.merge(as_df, on="player_id", how="left")
    return base, is_pooled


# Player List metric families — the collapse filter toggles each group's columns
# (display names, post-rename). Identity columns (Player/Pos/Team/GP/TOI) always
# show.
PLAYER_FAMILY_COLS = {
    "Net Front Impact": ["RelNFI%", "RelNFI-A%", "RelNFI-S%", "NFI%",
                         "NFI-A/60", "NFI-S/60"],
    "Zone Impact": ["NZI", "DZI", "OZI"],
    "Quality Games": ["xG_QG%", "NFI_QG%"],
}


def render_players(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Player List</h2>",
        unsafe_allow_html=True,
    )
    playoffs = game_type == "Playoffs"
    scope_label = "all playoffs (2022-2025 pooled)" if playoffs else season_label
    if playoffs:
        st.caption("Playoff view — all playoff games (2022-23 → 2024-25) pooled. "
                   "Use the Min ES TOI slider to threshold small samples.")
    frame, is_pooled = _build_players_frame(season_label, playoffs=playoffs)
    if frame.empty:
        st.error("Player data not found "
                 "(`NFI/output/fully_adjusted/player_fully_adjusted"
                 f"{'_playoffs' if playoffs else ''}.csv`).")
        return

    c1, c2, c3 = st.columns([1.0, 1.6, 1.3])
    with c1:
        pos = st.radio("Position", ["All", "F", "D"], horizontal=True, key="players_pos")
        collapse_fam = st.radio(
            "Collapse a Metric Family", ["None"] + list(PLAYER_FAMILY_COLS),
            horizontal=True, key="players_collapse",
            help="Hide a metric group's columns (Net Front Impact, Zone Impact, "
                 "Quality Games).")
    with c2:
        if playoffs:
            min_toi = st.slider("Min ES TOI (min)", 0, 1500, 300, 25,
                                key="players_toi_playoffs")
        else:
            toi_key = "players_toi_pooled" if is_pooled else "players_toi_season"
            default_toi = 2000 if is_pooled else 500
            min_toi = st.slider("Min ES TOI (min)", 0, 7500, default_toi, 50, key=toi_key)
    with c3:
        team_opts = ["All"] + sorted(frame["team"].dropna().unique().tolist())
        team_sel = st.selectbox("Team", team_opts, key="players_team")

    df = frame.copy()
    if pos in ("F", "D"):
        df = df[df["position"] == pos]
    else:
        df = df[df["position"].isin(["F", "D"])]
    df = df[df["toi_min"].fillna(0) >= min_toi]
    rank_cohort = df.copy()   # position + Min-TOI cohort — the ranking denominator
    if team_sel != "All":
        df = df[df["team"] == team_sel]
    if df.empty:
        st.markdown(
            f"<p style='color:{PALETTE['text']};'>No players match the current filters. "
            "Widen Position or Min TOI, or change the Season in the sidebar.</p>",
            unsafe_allow_html=True,
        )
        return

    df = df.sort_values("RelNFI_pct", ascending=False, na_position="last").reset_index(drop=True)
    # Storage → display: RelNFI_F (attack / for) shows as "RelNFI-A%",
    # RelNFI_A (suppress / against) shows as "RelNFI-S%". Do NOT sign-flip — the
    # underlying _F/_A columns are unchanged; only the display labels swap A/S.
    _ren = {
        "player_name": "Player", "position": "Pos", "team": "Team", "toi_min": "TOI",
        "NFI_pct": "NFI%", "RelNFI_pct": "RelNFI%",
        "RelNFI_F_pct": "RelNFI-A%", "RelNFI_A_pct": "RelNFI-S%",
        "NFI_A_rate": "NFI-A/60", "NFI_S_rate": "NFI-S/60",
        "xG_QG_pct": "xG_QG%", "NFI_QG_pct": "NFI_QG%",
    }
    df = df.rename(columns=_ren)
    rank_cohort = rank_cohort.rename(columns=_ren)

    # Always show the full column set (Compact view removed; Qual GP dropped).
    # NFI-A/60 / NFI-S/60 are RAW per-60 rates; RelNFI-A% / RelNFI-S% are the
    # relative (vs own-team) versions — both coexist, placed side by side.
    cols = ["Player", "Pos", "Team", "GP", "TOI", "RelNFI%", "RelNFI-A%",
            "RelNFI-S%", "NFI%", "NFI-A/60", "NFI-S/60", "NZI", "DZI", "OZI",
            "xG_QG%", "NFI_QG%"]
    # Zone now populates for single seasons too (per-season files), so it is no
    # longer stripped; the in-frame filter below drops it only if truly absent.
    cols = [c for c in cols if c in df.columns]
    # Metric-family collapse: hide the selected family's columns (identity columns
    # and the other families stay).
    _fam_of = {col: fam for fam, fcols in PLAYER_FAMILY_COLS.items() for col in fcols}
    cols = [c for c in cols if _fam_of.get(c) != collapse_fam]
    disp = df[cols].copy()

    fmt = {}
    for c in ("NFI%", "xG_QG%", "NFI_QG%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("RelNFI%", "RelNFI-A%", "RelNFI-S%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:+.2f}"
    for c in ("NFI-A/60", "NFI-S/60"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    for c in ("NZI", "DZI", "OZI"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    if "TOI" in disp.columns:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    for c in ("GP",):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    _player_rank = ["NFI%", "RelNFI%", "RelNFI-A%", "RelNFI-S%", "NFI-A/60",
                    "NFI-S/60", "NZI", "DZI", "OZI", "xG_QG%", "NFI_QG%"]
    _apply_ranks(disp, fmt, rank_cohort, _player_rank, lower_better={"NFI-S/60"})
    _cohort_label = {"All": "all skaters (F + D)", "F": "forwards",
                     "D": "defense"}[pos]
    st.caption(f"Each metric shows its **(rank)** within "
               f"**{_cohort_label}** (set by the Position filter; players meeting "
               f"Min-TOI). NFI-S/60 (shots against): lowest = #1.")
    _sort_hint()
    st.dataframe(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)

    if playoffs:
        zone_note = " · NZI/DZI/OZI pooled across playoffs"
    elif SEASON_KEY.get(season_label) == "pooled_2yr":
        zone_note = " · NZI/DZI/OZI pooled 2024-25 + 2025-26"
    elif is_pooled:
        zone_note = " · NZI/DZI/OZI pooled across all seasons"
    else:
        zone_note = " · NZI/DZI/OZI for this season"
    st.caption(
        f"{len(disp):,} players · {scope_label} · sorted by RelNFI% descending · "
        f"min {min_toi:,} ES min{zone_note}"
    )


def _render_player_playoff_summary(frame: pd.DataFrame, pid: int) -> None:
    """Pooled all-playoffs metric summary for one player (the playoff Detail view
    — no per-season trend; playoff producers publish the pooled view)."""
    row = frame[frame["player_id"] == pid]
    if row.empty:
        st.info("No pooled playoff data for this player.")
        return
    r = row.iloc[0]

    def _f(col, kind):
        v = r.get(col)
        if pd.isna(v):
            return "—"
        if kind == "pct":
            return f"{v * 100:.1f}%"
        if kind == "rate":
            return f"{v:.1f}"
        if kind == "toi":
            return f"{v:,.0f}"
        return f"{v}"

    def _rel(col):
        v = r.get(col)
        return "—" if pd.isna(v) else f"{v:+.2f}"

    items = [
        ("RelNFI%", _rel("RelNFI_pct")),
        ("RelNFI-A%", _rel("RelNFI_F_pct")),
        ("RelNFI-S%", _rel("RelNFI_A_pct")),
        ("NFI%", _f("NFI_pct", "pct")),
        ("NFI-A/60", _f("NFI_A_rate", "rate")),
        ("NFI-S/60", _f("NFI_S_rate", "rate")),
        ("NZI", _f("NZI", "rate")),
        ("DZI", _f("DZI", "rate")),
        ("OZI", _f("OZI", "rate")),
        ("xG_QG%", _f("xG_QG_pct", "pct")),
        ("NFI_QG%", _f("NFI_QG_pct", "pct")),
        ("ES TOI (min)", _f("toi_min", "toi")),
    ]
    st.caption(f"**{r['player_name']} ({r['position']})** · pooled playoffs "
               "(2022-23 → 2024-25).")
    st.dataframe(pd.DataFrame(items, columns=["Metric", "Value"]),
                 width="stretch", hide_index=True)


def render_player_detail(season_label: str, game_type: str) -> None:
    """Player Detail tab — searchable selector → per-season trend table + charts.
    Reuses _player_trend / _render_player_profile; selector options come from the
    full (unfiltered) player frame."""
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Player Detail</h2>",
        unsafe_allow_html=True,
    )
    playoffs = game_type == "Playoffs"
    frame, _ = _build_players_frame(season_label, playoffs=playoffs)
    if frame.empty:
        st.error("Player data not found "
                 "(`NFI/output/fully_adjusted/player_fully_adjusted"
                 f"{'_playoffs' if playoffs else ''}.csv`).")
        return
    popts = (frame[["player_id", "player_name", "position"]]
             .dropna(subset=["player_id"]).drop_duplicates("player_id")
             .sort_values("player_name"))
    pid_list = [int(x) for x in popts["player_id"].tolist()]
    plabel = {int(r.player_id): f"{r.player_name} ({r.position})"
              for r in popts.itertuples()}
    _prompt = ("Select a player for their pooled playoff profile"
               if playoffs else
               "Select a player for a per-season trend (2022-23 → 2025-26)")
    sel = st.selectbox(
        _prompt, pid_list, index=None, placeholder="— select a player —",
        format_func=lambda i: plabel.get(i, str(i)), key="players_profile")
    if sel is None:
        st.caption("Pick a player to see their "
                   + ("pooled playoff metrics." if playoffs
                      else "season-by-season trend and charts."))
    elif playoffs:
        _render_player_playoff_summary(frame, int(sel))
    else:
        _render_player_profile(int(sel))


# ---------------------------------------------------------------------------
# Teams tab — team NFI% (CNFI+MNFI share) + Attack/Suppress + Quality Games
# ---------------------------------------------------------------------------
POOLED_SEASONS = ["20222023", "20232024", "20242025", "20252026"]


@st.cache_data(show_spinner=False, ttl=3600)
def load_team_level() -> pd.DataFrame:
    fp = REPO_ROOT / "NFI" / "output" / "team_level_all_metrics.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    df["team"] = df["team"].replace({"ARI": "UTA"})  # legacy code → current
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_team_qg() -> pd.DataFrame:
    fp = REPO_ROOT / "Quality_Games" / "output" / "per_team_season.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp).rename(columns={"team_abbrev": "team"})
    df["season"] = df["season"].astype(str)
    df["team"] = df["team"].replace({"ARI": "UTA"})
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_team_attack_suppress() -> pd.DataFrame:
    """Per-(season, team) team Attack/Suppress: raw counts + games + per-game
    rates, all seasons 2022-23..2025-26. Counts/games are additive, so pooled
    windows use ratio-of-sums (see _team_attack_suppress)."""
    fp = REPO_ROOT / "NFI" / "output" / "team_nfi_verification_and_attack_suppress.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    df["team"] = df["team"].replace({"ARI": "UTA"})  # franchise continuity for pools
    keep = ["season", "team", "attack_count", "suppress_count",
            "games_played", "attack_per_game", "suppress_per_game"]
    return df[[c for c in keep if c in df.columns]].copy()


def _team_attack_suppress(key: str) -> pd.DataFrame:
    """Team Attack/Suppress per-game for a scope, named for direct merge.
    key: 'pooled' (4yr 2022-26), 'pooled_2yr' (2024-26), or a season string.
    Pools are ratio-of-sums: sum(counts)/sum(games) — NOT averaged per-game."""
    a = load_team_attack_suppress()
    if a.empty:
        return pd.DataFrame()
    if key == "pooled":
        sub = a[a["season"].isin(POOLED_SEASONS)]
    elif key == "pooled_2yr":
        sub = a[a["season"].isin(POOLED_2YR_SEASONS)]
    else:
        sub = a[a["season"] == key]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("team").agg(_ac=("attack_count", "sum"),
                                _sc=("suppress_count", "sum"),
                                _gp=("games_played", "sum")).reset_index()
    ok = g["_gp"] > 0
    g["Attack events"] = np.where(ok, g["_ac"] / g["_gp"], np.nan)
    g["Suppress events"] = np.where(ok, g["_sc"] / g["_gp"], np.nan)
    return g[["team", "Attack events", "Suppress events"]]


@st.cache_data(show_spinner=False, ttl=3600)
def load_team_zone() -> pd.DataFrame:
    """Team Zone Impact (NZI/OZI/DZI + composite) per window from
    NFI/output/team_zone.csv (built by NFI/scripts/build_team_zone.py).
    Windows: '4y_pool' (2022-26), '2y_2426' (2024-26). TOI-weighted."""
    fp = REPO_ROOT / "NFI" / "output" / "team_zone.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["team"] = df["team"].replace({"ARI": "UTA"})
    return df


def _team_zone_window(key: str) -> str:
    """Season key → team-zone window. No raw single-season team zone (2024-25 is
    hit-zoneCode distorted), so single seasons map to their pool: 2024-26 era →
    2y_2426, 2022-24 era + full pooled → 4y_pool."""
    if key in ("pooled_2yr", "20242025", "20252026"):
        return "2y_2426"
    return "4y_pool"


def _team_nfi_share(df: pd.DataFrame) -> np.ndarray:
    """Post-audit team NFI% = CNFI+MNFI Fenwick for / (for + against). Excludes FNFI."""
    ffor = df["CNFI_FF"] + df["MNFI_FF"]
    fagn = df["CNFI_FA"] + df["MNFI_FA"]
    denom = ffor + fagn
    return np.where(denom > 0, ffor / denom, np.nan)


@st.cache_data(show_spinner=False, ttl=3600)
def load_team_playoffs() -> pd.DataFrame:
    """Pooled all_playoffs team NFI% + Attack/Suppress events (per game) from
    team_nfi_verification_and_attack_suppress_playoffs.csv. NFI% is the
    post-audit CNFI+MNFI share (fraction), Attack/Suppress are per-game counts."""
    fp = REPO_ROOT / "NFI" / "output" / "team_nfi_verification_and_attack_suppress_playoffs.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    df = df[df["season"] == PLAYOFF_SCOPE].copy()
    df["team"] = df["team"].replace({"ARI": "UTA"})
    return df.rename(columns={"games_played": "GP",
                              "team_nfi_pct_post_audit": "NFI%",
                              "attack_per_game": "Attack events",
                              "suppress_per_game": "Suppress events"})[
        ["team", "GP", "NFI%", "Attack events", "Suppress events"]]


@st.cache_data(show_spinner=False, ttl=3600)
def load_team_qg_playoffs() -> pd.DataFrame:
    fp = REPO_ROOT / "Quality_Games" / "output" / "per_team_season_playoffs.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp).rename(columns={"team_abbrev": "team"})
    df["season"] = df["season"].astype(str)
    df["team"] = df["team"].replace({"ARI": "UTA"})
    return df[df["season"] == PLAYOFF_SCOPE].copy()


@st.cache_data(show_spinner=False, ttl=3600)
def load_team_zone_playoffs() -> pd.DataFrame:
    fp = REPO_ROOT / "NFI" / "output" / "team_zone_playoffs.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["window"] = df["window"].astype(str)
    df["team"] = df["team"].replace({"ARI": "UTA"})
    return df[df["window"] == PLAYOFF_SCOPE].copy()


def _render_teams_playoffs(season_label: str) -> None:
    """Teams tab, pooled all-playoffs view (2022-23 → 2024-25)."""
    st.caption("Playoff view — all playoff games (2022-23 → 2024-25) pooled.")
    team = load_team_playoffs()
    if team.empty:
        st.error("Playoff team data not found "
                 "(`NFI/output/team_nfi_verification_and_attack_suppress_playoffs.csv`).")
        return

    qg = load_team_qg_playoffs()
    if not qg.empty:
        q = qg[["team", "total_team_TOI_min", "team_xG_QG_pct", "team_NFI_QG_pct"]].rename(
            columns={"total_team_TOI_min": "TOI",
                     "team_xG_QG_pct": "xG_QG%", "team_NFI_QG_pct": "NFI_QG%"})
        team = team.merge(q, on="team", how="left")

    zcols = ["NZI", "DZI", "OZI"]
    tz = load_team_zone_playoffs()
    if not tz.empty:
        team = team.merge(tz[["team", "NZI", "DZI", "OZI"]], on="team", how="left")

    for c in ["TOI", "xG_QG%", "NFI_QG%"] + zcols:
        if c not in team.columns:
            team[c] = np.nan

    team = team.rename(columns={"team": "Team"})
    team = team.sort_values("NFI%", ascending=False, na_position="last").reset_index(drop=True)
    cols = (["Team", "GP", "TOI", "NFI%", "Attack events", "Suppress events"]
            + zcols + ["xG_QG%", "NFI_QG%"])
    disp = team[[c for c in cols if c in team.columns]].copy()

    fmt = {}
    for c in ("NFI%", "xG_QG%", "NFI_QG%"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("Attack events", "Suppress events"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    for c in zcols:
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    if "TOI" in disp:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    if "GP" in disp:
        fmt["GP"] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    _team_rank = ["NFI%", "Attack events", "Suppress events"] + zcols + ["xG_QG%", "NFI_QG%"]
    _apply_ranks(disp, fmt, disp, _team_rank, lower_better={"Suppress events"})
    st.caption("Each metric shows its **(rank)** across playoff teams. "
               "Suppress events (shots against): lowest = #1.")
    _sort_hint()
    st.dataframe(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
    st.caption(
        f"{len(disp)} teams · all playoffs (2022-2025 pooled) · sorted by NFI% "
        "(CNFI+MNFI share) descending · Zone Impact (NZI/DZI/OZI) is TOI-weighted."
    )


def render_teams(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Teams</h2>",
        unsafe_allow_html=True,
    )
    playoffs = game_type == "Playoffs"
    if playoffs:
        _render_teams_playoffs(season_label)
        return

    tl = load_team_level()
    if tl.empty:
        st.error("Team data not found (`NFI/output/team_level_all_metrics.csv`).")
        return
    qg = load_team_qg()
    key = SEASON_KEY.get(season_label, "pooled")
    is_pooled = key in ("pooled", "pooled_2yr")
    # "pooled_2yr" aggregates only 2024–2026; full pooled aggregates all four.
    pooled_seasons = POOLED_2YR_SEASONS if key == "pooled_2yr" else POOLED_SEASONS

    def _wmean(g: pd.DataFrame, col: str) -> float:
        w = g["total_team_TOI_min"].astype(float)
        v = pd.to_numeric(g[col], errors="coerce")
        m = v.notna() & (w > 0)
        return float(np.average(v[m], weights=w[m])) if m.any() else np.nan

    if is_pooled:
        sub = tl[tl["season"].isin(pooled_seasons)]
        agg = sub.groupby("team").agg(
            CNFI_FF=("CNFI_FF", "sum"), MNFI_FF=("MNFI_FF", "sum"),
            CNFI_FA=("CNFI_FA", "sum"), MNFI_FA=("MNFI_FA", "sum"),
            GP=("gp", "sum"),
        ).reset_index()
        agg["NFI%"] = _team_nfi_share(agg)
        team = agg[["team", "GP", "NFI%"]]
        if not qg.empty:
            q = qg[qg["season"].isin(pooled_seasons)]
            qrows = [{"team": t, "TOI": g["total_team_TOI_min"].sum(),
                      "xG_QG%": _wmean(g, "team_xG_QG_pct"),
                      "NFI_QG%": _wmean(g, "team_NFI_QG_pct")}
                     for t, g in q.groupby("team")]
            team = team.merge(pd.DataFrame(qrows), on="team", how="left")
    else:
        sk = SEASON_KEY[season_label]
        sub = tl[tl["season"] == sk].copy()
        sub["NFI%"] = _team_nfi_share(sub)
        team = sub[["team", "gp", "NFI%"]].rename(columns={"gp": "GP"})
        if not qg.empty:
            q = qg[qg["season"] == sk][
                ["team", "total_team_TOI_min", "team_xG_QG_pct", "team_NFI_QG_pct"]
            ].rename(columns={"total_team_TOI_min": "TOI",
                              "team_xG_QG_pct": "xG_QG%", "team_NFI_QG_pct": "NFI_QG%"})
            team = team.merge(q, on="team", how="left")

    if team.empty:
        st.info("No team data for this season.")
        return

    # Attack / Suppress events — all seasons + pools (ratio-of-sums for pools).
    a = _team_attack_suppress(key)
    if not a.empty:
        team = team.merge(a, on="team", how="left")

    # Team Zone Impact (NZI/DZI/OZI). Single seasons show their pooled window —
    # no raw single-season team zone (2024-25 is hit-distorted). Headers carry
    # the window suffix so the displayed pool is unambiguous.
    zwin = _team_zone_window(key)
    zsfx = "2yr" if zwin == "2y_2426" else "4yr"
    zcols = [f"NZI ({zsfx})", f"DZI ({zsfx})", f"OZI ({zsfx})"]
    tz = load_team_zone()
    if not tz.empty:
        tzw = (tz[tz["window"] == zwin][["team", "NZI", "DZI", "OZI"]]
               .rename(columns={"NZI": zcols[0], "DZI": zcols[1], "OZI": zcols[2]}))
        team = team.merge(tzw, on="team", how="left")

    for c in ["TOI", "xG_QG%", "NFI_QG%", "Attack events", "Suppress events"] + zcols:
        if c not in team.columns:
            team[c] = np.nan

    team = team.rename(columns={"team": "Team"})
    team = team.sort_values("NFI%", ascending=False, na_position="last").reset_index(drop=True)
    cols = (["Team", "GP", "TOI", "NFI%", "Attack events", "Suppress events"]
            + zcols + ["xG_QG%", "NFI_QG%"])
    disp = team[[c for c in cols if c in team.columns]].copy()

    fmt = {}
    for c in ("NFI%", "xG_QG%", "NFI_QG%"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("Attack events", "Suppress events"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    for c in zcols:
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    if "TOI" in disp:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    if "GP" in disp:
        fmt["GP"] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    _team_rank = (["NFI%", "Attack events", "Suppress events"] + zcols
                  + ["xG_QG%", "NFI_QG%"])
    _apply_ranks(disp, fmt, disp, _team_rank, lower_better={"Suppress events"})
    st.caption("Each metric shows its **(rank)** across all 32 teams. "
               "Suppress events (shots against): lowest = #1.")
    _sort_hint()
    st.dataframe(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)

    zwin_label = "4-year pool (2022-26)" if zwin == "4y_pool" else "2-year pool (2024-26)"
    cap = (f"{len(disp)} teams · {season_label} · sorted by NFI% (CNFI+MNFI share) "
           f"descending · Zone Impact (NZI/DZI/OZI) is TOI-weighted, shown as the "
           f"{zwin_label}; single seasons display their pooled window "
           f"(2022-24 → 4yr, 2024-26 → 2yr) since single-season team zone isn't published.")
    st.caption(cap)


# ---------------------------------------------------------------------------
# Goalies tab — NFI-GSAx + QNFS% + QS-GSAx (union of qualified cohorts)
# ---------------------------------------------------------------------------
GOALIE_SEASON_INT = {"2025-26": 20252026, "2024-25": 20242025,
                     "2023-24": 20232024, "2022-23": 20222023}
_QC = REPO_ROOT / "NFI" / "goalie_consistency" / "output"


@st.cache_data(show_spinner=False, ttl=3600)
def load_qnfs_pooled() -> pd.DataFrame:
    fp = _QC / "qnfs_2022-2026.csv"
    return pd.read_csv(fp) if fp.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_qnfs_by_season() -> pd.DataFrame:
    fp = _QC / "qnfs_per_season_2022-2026.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(int)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_qs_pooled() -> pd.DataFrame:
    fp = _QC / "qs_gsax_2022-2026.csv"
    return pd.read_csv(fp) if fp.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_qs_by_season() -> pd.DataFrame:
    fp = _QC / "qs_gsax_per_season_2022-2026.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(int)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_goalie_nfi_playoffs() -> pd.DataFrame:
    fp = REPO_ROOT / "NFI" / "output" / "goalie_nfi_gsax_by_season_playoffs.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    return df[df["season"] == PLAYOFF_SCOPE].copy()


@st.cache_data(show_spinner=False, ttl=3600)
def load_qnfs_playoffs() -> pd.DataFrame:
    fp = _QC / "qnfs_per_season_playoffs.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    return df[df["season"] == PLAYOFF_SCOPE].copy()


@st.cache_data(show_spinner=False, ttl=3600)
def load_qs_playoffs() -> pd.DataFrame:
    fp = _QC / "qs_gsax_per_season_playoffs.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    return df[df["season"] == PLAYOFF_SCOPE].copy()


def _goalie_trend(gid: int) -> pd.DataFrame:
    """Per-season (2022-23..2025-26) NFI-GSAx/60, QNFS%, QS-GSAx% for one
    goalie_id, outer-merged on season. Season normalized to INT before merging
    (all three by-season loaders cast to int) to avoid silent empty merges. The
    NFI-GSAx file includes a 2021-22 row; it's dropped here."""
    gid = int(gid)
    seasons_int = [20222023, 20232024, 20242025, 20252026]
    parts = []
    n = load_goalie_nfi_by_season()
    if not n.empty:
        parts.append(n[n["goalie_id"] == gid][["season", "GSAx_per60"]]
                     .rename(columns={"GSAx_per60": "NFI-GSAx/60"}))
    q = load_qnfs_by_season()
    if not q.empty:
        parts.append(q[q["goalie_id"] == gid][["season", "QNFS_pct"]]
                     .rename(columns={"QNFS_pct": "QNFS%"}))
    s = load_qs_by_season()
    if not s.empty:
        parts.append(s[s["goalie_id"] == gid][["season", "QS_GSAx_pct"]]
                     .rename(columns={"QS_GSAx_pct": "QS-GSAx%"}))
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    base = None
    for p in parts:
        p = p.copy()
        p["season"] = p["season"].astype(int)
        base = p if base is None else base.merge(p, on="season", how="outer")
    base = base[base["season"].isin(seasons_int)].copy()
    base["Season"] = (base["season"].astype(str).map(SEASON_DISPLAY)
                      .fillna(base["season"].astype(str)))
    return base.sort_values("season").reset_index(drop=True)


def _goalie_season_ranks(gid: int) -> dict:
    """Per-season LEAGUE rank of each metric among all goalies that season
    (#1 = best). Returns {display_col: {season_int: rank}}."""
    gid = int(gid)
    out = {}
    seasons_int = [20222023, 20232024, 20242025, 20252026]
    for loader, src, disp_c in (
        (load_goalie_nfi_by_season, "GSAx_per60", "NFI-GSAx/60"),
        (load_qnfs_by_season, "QNFS_pct", "QNFS%"),
        (load_qs_by_season, "QS_GSAx_pct", "QS-GSAx%"),
    ):
        df = loader()
        if df.empty or src not in df.columns:
            continue
        df = df.copy()
        df["season"] = df["season"].astype(int)
        d = {}
        for ssn in seasons_int:
            sub = df[df["season"] == ssn]
            pv = sub.loc[sub["goalie_id"] == gid, src]
            if len(pv) and pd.notna(pv.iloc[0]):
                d[ssn] = _league_rank(sub[src], pv.iloc[0])
        out[disp_c] = d
    return out


def _render_goalie_profile(gid: int) -> None:
    """Per-season trend table + line charts for one goalie."""
    trend = _goalie_trend(gid)
    if trend.empty:
        st.info("No per-season data available for this goalie.")
        return
    metric_cols = [c for c in ("NFI-GSAx/60", "QNFS%", "QS-GSAx%")
                   if c in trend.columns]
    ranks = _goalie_season_ranks(gid)
    _b = {"NFI-GSAx/60": lambda v: f"{v:+.3f}",
          "QNFS%": lambda v: f"{v:.1f}%", "QS-GSAx%": lambda v: f"{v:.1f}%"}
    rows = []
    for _, r in trend.iterrows():
        ssn = int(r["season"])
        row = {"Season": r["Season"]}
        for c in metric_cols:
            v = r[c]
            if pd.isna(v):
                row[c] = "—"
            else:
                txt = _b.get(c, lambda v: f"{v}")(v)
                rk = ranks.get(c, {}).get(ssn)
                row[c] = f"{txt} ({rk})" if rk is not None else txt
        rows.append(row)
    st.caption("Each value shows its **(rank)** — league rank among all goalies "
               "that season.")
    st.dataframe(pd.DataFrame(rows, columns=["Season"] + metric_cols),
                 width="stretch", hide_index=True)

    # CHOICE: split into 2 small multiples. NFI-GSAx/60 is a per-60 rate (~±0.3);
    # QNFS%/QS-GSAx% are percentages (~0–100). On a single shared axis the rate
    # collapses to a flat line near zero, so the rate gets its own chart and the
    # two percentages share one.
    if "NFI-GSAx/60" in trend.columns and trend["NFI-GSAx/60"].notna().any():
        st.caption("NFI-GSAx per 60")
        st.line_chart(trend.set_index("Season")[["NFI-GSAx/60"]],
                      color=[_CHART_COLORS.get("NFI-GSAx/60", _CHART_SECOND)])
    pct = [c for c in ("QNFS%", "QS-GSAx%")
           if c in trend.columns and trend[c].notna().any()]
    if pct:
        st.caption("Consistency % (QNFS%, QS-GSAx%)")
        st.line_chart(trend.set_index("Season")[pct],
                      color=[_CHART_COLORS.get(c, _CHART_SECOND) for c in pct])


def render_goalies(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Goalie List</h2>",
        unsafe_allow_html=True,
    )
    playoffs = game_type == "Playoffs"
    if playoffs:
        st.caption("Playoff view — all playoff games (2022-23 → 2024-25) pooled. "
                   "Use Min Shots Faced to threshold small samples.")

    # FOLLOW-UP: no 2-year pooled goalie build exists. Goalie GSAx is published
    # as full-pooled (2022–2026) or per single season; a faithful 2yr pool needs
    # re-derived denominators (not a season average), so fall back gracefully.
    if not playoffs and SEASON_KEY.get(season_label) == "pooled_2yr":
        st.info("2-season (2024–2026) goalie view isn't available yet — pick a "
                "single season or the full 4yr (2022-2026) view.")
        return

    is_pooled = (not playoffs) and SEASON_KEY.get(season_label, "pooled") == "pooled"
    if playoffs:
        n = load_goalie_nfi_playoffs()
        nfi = (n[["goalie_id", "goalie_name", "team", "games", "total_faced", "GSAx_per60"]]
               .rename(columns={"games": "GP_nfi", "GSAx_per60": "NFIG60"})
               if not n.empty else pd.DataFrame())
        q = load_qnfs_playoffs()
        qn = (q[["goalie_id", "goalie_name", "GP", "QNFS_pct", "QNFS_lo", "QNFS_hi"]]
              .rename(columns={"GP": "GP_qn"}) if not q.empty else pd.DataFrame())
        s = load_qs_playoffs()
        qs = (s[["goalie_id", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo"]]
              .rename(columns={"GP": "GP_qs"}) if not s.empty else pd.DataFrame())
    elif is_pooled:
        n = load_goalie_nfi()
        nfi = (n[["goalie_id", "goalie_name", "team", "games", "total_faced", "GSAx_per60"]]
               .rename(columns={"games": "GP_nfi", "GSAx_per60": "NFIG60"})
               if not n.empty else pd.DataFrame())
        q = load_qnfs_pooled()
        if not q.empty and "qualified" in q.columns:
            q = q[q["qualified"] == True]  # noqa: E712  union of QUALIFIED cohorts
        qn = (q[["goalie_id", "goalie_name", "GP", "QNFS_pct", "QNFS_lo", "QNFS_hi"]]
              .rename(columns={"GP": "GP_qn"}) if not q.empty else pd.DataFrame())
        s = load_qs_pooled()
        qs = (s[["goalie_id", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo"]]
              .rename(columns={"GP": "GP_qs"}) if not s.empty else pd.DataFrame())
    else:
        sk = GOALIE_SEASON_INT.get(season_label)
        bs = load_goalie_nfi_by_season()
        nfi = (bs[bs["season"] == sk][["goalie_id", "goalie_name", "team", "games", "total_faced", "GSAx_per60"]]
               .rename(columns={"games": "GP_nfi", "GSAx_per60": "NFIG60"})
               if (not bs.empty and sk) else pd.DataFrame())
        q0 = load_qnfs_by_season()
        qn = (q0[q0["season"] == sk][["goalie_id", "goalie_name", "GP", "QNFS_pct", "QNFS_lo", "QNFS_hi"]]
              .rename(columns={"GP": "GP_qn"}) if (not q0.empty and sk) else pd.DataFrame())
        s0 = load_qs_by_season()
        qs = (s0[s0["season"] == sk][["goalie_id", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo"]]
              .rename(columns={"GP": "GP_qs"}) if (not s0.empty and sk) else pd.DataFrame())

    frames = [f for f in (nfi, qn, qs) if not f.empty]
    if not frames:
        st.info("No goalie data available for this view.")
        return

    # Name lookup across sources; merge metrics on goalie_id (drop name first to
    # avoid suffix collisions, then map a single canonical name back).
    name_src = pd.concat([f[["goalie_id", "goalie_name"]] for f in frames
                          if "goalie_name" in f.columns], ignore_index=True)
    name_map = name_src.dropna().drop_duplicates("goalie_id").set_index("goalie_id")["goalie_name"]
    frames2 = [f.drop(columns=[c for c in ["goalie_name"] if c in f.columns]) for f in frames]
    base = frames2[0]
    for r in frames2[1:]:
        base = base.merge(r, on="goalie_id", how="outer")

    base["Goalie"] = base["goalie_id"].map(name_map)
    gp_cols = [c for c in ("GP_nfi", "GP_qn", "GP_qs") if c in base.columns]
    base["GP"] = base[gp_cols].bfill(axis=1).iloc[:, 0] if gp_cols else np.nan
    base["Team"] = base["team"] if "team" in base.columns else np.nan

    c1, c2 = st.columns([2, 1])
    with c1:
        if playoffs:
            default_shots, shots_key, smax = 50, "goalies_minshots_playoffs", 1500
        elif is_pooled:
            default_shots, shots_key, smax = 500, "goalies_minshots_pooled", 3000
        else:
            default_shots, shots_key, smax = 150, "goalies_minshots_season", 3000
        min_shots = st.slider("Min Shots Faced", 0, smax, default_shots, 50, key=shots_key)
        st.caption("Min Shots Faced filter suppresses small-sample noise in per-60 "
                   "rates. Defaults match the methodology's qualifying floors and the "
                   "previous app's discipline.")
    with c2:
        name_q = st.text_input("Goalie name contains", key="goalies_name").strip().lower()
    base = base[base["total_faced"].fillna(0) >= min_shots]
    rank_cohort = base.copy()   # Min-Shots cohort (pre name-filter) — rank denom
    if name_q:
        base = base[base["Goalie"].str.lower().str.contains(name_q, na=False)]
    if base.empty:
        st.info("No goalies match the current filters.")
        return

    def _qnfs_ci(r):
        if pd.isna(r.get("QNFS_lo")) or pd.isna(r.get("QNFS_hi")):
            return np.nan
        return f"({r['QNFS_lo']:.1f}–{r['QNFS_hi']:.1f})"
    base["QNFS 95% CI"] = base.apply(_qnfs_ci, axis=1)
    _gren = {
        "NFIG60": "NFI-GSAx/60", "QNFS_pct": "QNFS%",
        "QS_GSAx_pct": "QS-GSAx%", "QS_GSAx_lo": "QS-GSAx (95% lower)",
    }
    base = base.rename(columns=_gren)
    rank_cohort = rank_cohort.rename(columns=_gren)
    base = base.sort_values("NFI-GSAx/60", ascending=False, na_position="last").reset_index(drop=True)

    cols = ["Goalie", "Team", "GP", "NFI-GSAx/60", "QNFS%", "QS-GSAx%"]
    disp = base[[c for c in cols if c in base.columns]].copy()

    fmt = {}
    if "NFI-GSAx/60" in disp:
        fmt["NFI-GSAx/60"] = lambda x: "—" if pd.isna(x) else f"{x:+.3f}"
    for c in ("QNFS%", "QS-GSAx%", "QS-GSAx (95% lower)"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}%"
    if "GP" in disp:
        fmt["GP"] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    _apply_ranks(disp, fmt, rank_cohort, ["NFI-GSAx/60", "QNFS%", "QS-GSAx%"])
    st.caption("Each metric shows its **(rank)** across all goalies "
               "meeting the Min-Shots filter. Blanks (below a metric's floor) are "
               "unranked.")
    _sort_hint()
    st.dataframe(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
    _goalie_scope = "all playoffs (2022-2025 pooled)" if playoffs else season_label
    st.caption(
        f"{len(disp)} goalies · {_goalie_scope} · sorted by NFI-GSAx/60 descending · "
        "blanks = below that metric's qualifying floor (not zero)"
    )
    if playoffs:
        st.markdown(
            f"<p style='color:{PALETTE['text_secondary']}; font-size:0.82rem; max-width:62rem;'>"
            "Pooled across all playoff games (2022-23 → 2024-25). No qualifying "
            "floor is applied — every goalie with playoff data appears; use Min "
            "Shots Faced to threshold. Per-game metric definitions (QNFS ≥3 "
            "net-front shots/game; QS-GSAx ≥10 shots/game) are retained.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<p style='color:{PALETTE['text_secondary']}; font-size:0.82rem; max-width:62rem;'>"
            "Goalies shown are the union of qualified cohorts across the three metrics. "
            "Backup goalies appearing only in unqualified QNFS rows are excluded — see "
            "Methodology for full qualifying floors. Qualifying floors differ by metric "
            "(NFI-GSAx ≥300 net-front shots pooled / ≥100 per season; QNFS% ≥25 GP/season "
            "with ≥3 net-front shots/game; QS-GSAx ≥10 shots/game, ≥25 GP/season).</p>",
            unsafe_allow_html=True,
        )


def _render_goalie_playoff_summary(gid: int) -> None:
    """Pooled all-playoffs metric summary for one goalie (playoff Detail view)."""
    n, q, s = load_goalie_nfi_playoffs(), load_qnfs_playoffs(), load_qs_playoffs()

    def pick(df, col):
        if df.empty or col not in df.columns:
            return np.nan
        r = df[df["goalie_id"] == gid]
        return r[col].iloc[0] if len(r) else np.nan

    name = next((str(v) for v in (pick(n, "goalie_name"), pick(q, "goalie_name"),
                                  pick(s, "goalie_name")) if pd.notna(v)), str(gid))

    def fmt(v, kind):
        if pd.isna(v):
            return "—"
        if kind == "gsax":
            return f"{v:+.3f}"
        if kind == "pct":
            return f"{v:.1f}%"
        return f"{int(v):,}"

    items = [
        ("NFI-GSAx/60", fmt(pick(n, "GSAx_per60"), "gsax")),
        ("QNFS%", fmt(pick(q, "QNFS_pct"), "pct")),
        ("QS-GSAx%", fmt(pick(s, "QS_GSAx_pct"), "pct")),
        ("Games (GSAx)", fmt(pick(n, "games"), "int")),
        ("Shots faced", fmt(pick(n, "total_faced"), "int")),
    ]
    st.caption(f"**{name}** · pooled playoffs (2022-23 → 2024-25).")
    st.dataframe(pd.DataFrame(items, columns=["Metric", "Value"]),
                 width="stretch", hide_index=True)


def render_goalie_detail(season_label: str, game_type: str) -> None:
    """Goalie Detail tab — searchable selector → per-season trend table + charts.
    Reuses _goalie_trend / _render_goalie_profile; options span the full goalie
    universe (by-season GSAx file), independent of the leaderboard filters."""
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Goalie Detail</h2>",
        unsafe_allow_html=True,
    )
    playoffs = game_type == "Playoffs"
    n = load_goalie_nfi_playoffs() if playoffs else load_goalie_nfi_by_season()
    if n.empty:
        st.info("No goalie data available.")
        return
    nn = n[["goalie_id", "goalie_name"]].dropna().drop_duplicates("goalie_id")
    glabel = {int(r.goalie_id): r.goalie_name for r in nn.itertuples()}
    gid_list = sorted(glabel, key=lambda i: glabel[i])
    if playoffs:
        gsel = st.selectbox(
            "Select a goalie for their pooled playoff profile",
            gid_list, index=None, placeholder="— select a goalie —",
            format_func=lambda i: glabel.get(i, str(i)), key="goalies_profile")
        if gsel is not None:
            _render_goalie_playoff_summary(int(gsel))
        else:
            st.caption("Pick a goalie to see their pooled playoff metrics.")
        return
    gsel = st.selectbox(
        "Select a goalie for a per-season trend (2022-23 → 2025-26)",
        gid_list, index=None, placeholder="— select a goalie —",
        format_func=lambda i: glabel.get(i, str(i)), key="goalies_profile")
    if gsel is not None:
        _render_goalie_profile(int(gsel))
    else:
        st.caption("Pick a goalie to see their season-by-season trend and charts.")


# ---------------------------------------------------------------------------
# Trade Analyzer tab — side-by-side player detail data (no charts), up to 5
# ---------------------------------------------------------------------------
TRADE_MAX_PLAYERS = 5


def render_trade_analyzer(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Trade Analyzer</h2>",
        unsafe_allow_html=True,
    )
    playoffs = game_type == "Playoffs"

    if playoffs:
        frame, _ = _build_players_frame(season_label, playoffs=True)
        src = (frame[["player_id", "player_name", "position"]]
               if not frame.empty else pd.DataFrame())
    else:
        nfi = load_nfi_player()
        # Most-recent name/position per player, spanning every season (so retired
        # or traded players are still selectable — useful for trade comparisons).
        src = (nfi.sort_values("season").drop_duplicates("player_id", keep="last")
               [["player_id", "player_name", "position"]]
               if not nfi.empty else pd.DataFrame())
    if src.empty:
        st.error("Player data not found.")
        return

    popts = (src.dropna(subset=["player_id"]).drop_duplicates("player_id")
             .sort_values("player_name"))
    pid_list = [int(x) for x in popts["player_id"].tolist()]
    plabel = {int(r.player_id): f"{r.player_name} ({r.position})"
              for r in popts.itertuples()}

    st.caption(f"Compare up to {TRADE_MAX_PLAYERS} players' detail data side by side "
               "(the same numbers as Player Detail — no charts)."
               + (" Pooled playoff view." if playoffs else ""))
    sel = st.multiselect(
        f"Players (max {TRADE_MAX_PLAYERS})", pid_list,
        format_func=lambda i: plabel.get(i, str(i)),
        max_selections=TRADE_MAX_PLAYERS, key="trade_players")
    if not sel:
        st.caption(f"Pick up to {TRADE_MAX_PLAYERS} players to compare.")
        return

    if playoffs:
        for pid in sel:
            _render_player_playoff_summary(frame, int(pid))
        return

    st.caption("Each value shows its **(rank)** — league rank among all skaters "
               "that season. NFI-S/60 (shots against): lowest = #1.")
    for pid in sel:
        st.markdown(f"**{plabel.get(int(pid), str(pid))}**")
        disp, _, _ = _player_profile_table(int(pid))
        if disp.empty:
            st.info("No per-season data available for this player.")
        else:
            st.dataframe(disp, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Referees tab — penalty-call tendencies (league-wide, 2023-24 → 2025-26)
# ---------------------------------------------------------------------------
REF_SEASON_INT = {"2025-26": 20252026, "2024-25": 20242025, "2023-24": 20232024}
REF_TYPES = ["Tripping", "Roughing", "Hooking", "Holding", "Slashing", "Interference"]
REF_MIN_GAMES = 40


@st.cache_data(show_spinner=False, ttl=3600)
def load_ref_penalties() -> pd.DataFrame:
    """League-wide penalties (2023-24 → 2025-26), exploded to one row per
    (penalty, individual referee). The source `referee` field lists both
    officials comma-joined; we split so each penalty is attributed to both."""
    fp = REPO_ROOT / "Referees" / "output" / "all_teams_penalties_3seasons.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(int)
    df["ref"] = df["referee"].astype(str).str.split(r"\s*,\s*")
    df = df.explode("ref")
    df["ref"] = df["ref"].str.strip()
    return df[df["ref"] != ""].copy()


REF_TEAM_MIN_GAMES = 5   # smaller floor — a ref works any one team only a handful of times


def _ref_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per referee: games, pen/game, home-pen%, per-game type rates."""
    rows = []
    for ref, g in df.groupby("ref"):
        games = g["game_id"].nunique()
        pens = len(g)
        ha = g["home_or_away"]
        home = int((ha == "home").sum())
        away = int((ha == "away").sum())
        row = {"Referee": ref, "Games": games, "Pen/Game": pens / games if games else np.nan,
               "Home Pen%": (home / (home + away) * 100) if (home + away) else np.nan,
               "Away Pen%": (away / (home + away) * 100) if (home + away) else np.nan}
        for t in REF_TYPES:
            row[f"{t}/G"] = (g["penalty_type"] == t).sum() / games if games else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _delta_fmt(avg, dec=2, pct=False):
    """Styler formatter: 'value (±Δ vs avg)'. Keeps the cell numeric (so the
    column still header-sorts on the real value) while appending the bracket."""
    suf = "%" if pct else ""
    def f(x):
        if pd.isna(x):
            return "—"
        return f"{x:.{dec}f}{suf} ({x - avg:+.{dec}f})"
    return f


def _ref_season_scope(df: pd.DataFrame, season_label: str) -> pd.DataFrame:
    key = SEASON_KEY.get(season_label, "pooled")
    if key == "pooled_2yr":
        return df[df["season"].isin([20242025, 20252026])]
    if key != "pooled":
        return df[df["season"] == REF_SEASON_INT.get(season_label)]
    return df


def _render_ref_league(df: pd.DataFrame, season_label: str, name_q: str) -> None:
    """League-wide referee table. The top row is the highlighted LEAGUE AVERAGE
    (plain values); every referee cell shows its value with an inline
    (± vs league average) bracket."""
    tbl = _ref_table(df)
    tbl = tbl[tbl["Games"] >= REF_MIN_GAMES].copy()
    if tbl.empty:
        st.info(f"No referees meet the {REF_MIN_GAMES}-game floor for this view.")
        return
    metric_cols = ["Pen/Game", "Home Pen%", "Away Pen%"] + [f"{t}/G" for t in REF_TYPES]
    avg = {c: float(tbl[c].mean()) for c in metric_cols if c in tbl.columns}
    pct_cols = {"Home Pen%", "Away Pen%"}

    tbl = tbl.sort_values("Pen/Game", ascending=False).reset_index(drop=True)
    if name_q:
        tbl = tbl[tbl["Referee"].str.lower().str.contains(name_q, na=False)]
    if tbl.empty:
        st.info("No referees match the name filter.")
        return

    def _plain(v, c):
        if pd.isna(v):
            return "—"
        dec = 1 if c in pct_cols else 2
        return f"{v:.{dec}f}{'%' if c in pct_cols else ''}"

    def _val(v, c):
        if pd.isna(v):
            return "—"
        dec = 1 if c in pct_cols else 2
        return f"{v:.{dec}f}{'%' if c in pct_cols else ''} ({v - avg[c]:+.{dec}f})"

    # Highlighted league-average row first (plain values, no bracket), then refs.
    rows = [{"Referee": "LEAGUE AVERAGE", "Games": f"{int(tbl['Games'].sum()):,}",
             **{c: _plain(avg[c], c) for c in metric_cols}}]
    for _, r in tbl.iterrows():
        rows.append({"Referee": r["Referee"], "Games": f"{int(r['Games']):,}",
                     **{c: _val(r[c], c) for c in metric_cols}})
    disp = pd.DataFrame(rows, columns=["Referee", "Games"] + metric_cols)

    def _bold_avg(row):
        is_avg = row["Referee"] == "LEAGUE AVERAGE"
        return [f"font-weight:700; color:{PALETTE['blue']};" if is_avg else "" for _ in row]

    st.caption(
        "For every referee, the number in brackets is that referee "
        "**(± vs the league average)** — e.g. Pen/Game `7.92 (+1.06)` means 1.06 more "
        "penalties per game than the league-average referee. Home Pen% = share of a "
        "referee's penalties assessed to the home team."
    )
    st.dataframe(disp.style.apply(_bold_avg, axis=1), width="stretch", hide_index=True)


def _render_ref_team(df: pd.DataFrame, season_label: str, team: str, name_q: str) -> None:
    """Per-team view: (A) what each referee calls AGAINST this team — every rate
    (overall and per penalty type) carries a two-sided bracket (Δ vs league avg /
    Δ vs that ref's own average); (B) the team's penalties-taken per game by type,
    each with a (Δ vs league avg) bracket."""
    dft = df[(df["home_team"] == team) | (df["away_team"] == team)].copy()
    games_T = dft["game_id"].nunique()
    if games_T == 0:
        st.info(f"No referee data for {team} in this view.")
        return
    distinct_games = df["game_id"].nunique()
    against = dft[dft["penalized_team"] == team]          # penalties on this team

    # League baseline = average penalties against a single team per game. df is
    # exploded (one row per referee → every penalty twice), so unique penalties =
    # rows / 2, and team-games = 2 × distinct games. Per type, restrict the count.
    def _league_base(ptype=None):
        sub = df if ptype is None else df[df["penalty_type"] == ptype]
        return (len(sub) / 2) / (2 * distinct_games) if distinct_games else np.nan
    L_overall = _league_base()
    L_type = {t: _league_base(t) for t in REF_TYPES}

    # Per-referee baselines over the FULL scope (all teams). A ref's total rows /
    # (2 × games) = their average penalties against a single team per game.
    ref_games_all = df.groupby("ref")["game_id"].nunique()
    ref_base_overall = (df.groupby("ref").size() / (2 * ref_games_all)).to_dict()
    ref_type_counts = df.groupby(["ref", "penalty_type"]).size().to_dict()

    def _two_sided(rate, lg, ref_own):
        if pd.isna(rate):
            return "—"
        own = f"{rate - ref_own:+.2f}" if pd.notna(ref_own) else "—"
        return f"{rate:.2f} ({rate - lg:+.2f} / {own})"

    # ---- Table A: referees in TEAM's games ----
    st.markdown(f"**Referees in {team}'s games — penalties called against {team}**")
    rate_col = f"Pen/G vs {team}"
    rows = []
    for ref, g in dft.groupby("ref"):
        games_RT = g["game_id"].nunique()
        if games_RT < REF_TEAM_MIN_GAMES:
            continue
        ag = against[against["ref"] == ref]
        rate = len(ag) / games_RT
        g_ref = ref_games_all.get(ref, np.nan)
        rec = {"Referee": ref, "Games": int(games_RT), "_sort": rate,
               rate_col: _two_sided(rate, L_overall, ref_base_overall.get(ref))}
        for t in REF_TYPES:
            rt = (ag["penalty_type"] == t).sum() / games_RT
            rb_t = (ref_type_counts.get((ref, t), 0) / (2 * g_ref)
                    if pd.notna(g_ref) and g_ref else np.nan)
            rec[f"{t}/G"] = _two_sided(rt, L_type[t], rb_t)
        rows.append(rec)

    if not rows:
        st.info(f"No referee worked ≥{REF_TEAM_MIN_GAMES} games involving {team} "
                "in this view.")
    else:
        ta = pd.DataFrame(rows).sort_values("_sort", ascending=False)
        if name_q:
            ta = ta[ta["Referee"].str.lower().str.contains(name_q, na=False)]
        if ta.empty:
            st.info("No referees match the name filter.")
        else:
            cols = ["Referee", "Games", rate_col] + [f"{t}/G" for t in REF_TYPES]
            st.caption(
                "Each cell is `rate (Δ vs league average / Δ vs that referee's own "
                "average)`. **First bracket number:** how this referee's rate against "
                f"{team} compares to the league-wide average rate against any team. "
                "**Second number:** how it compares to that same referee's own average "
                f"rate against all teams — positive means the referee calls more against "
                f"{team} than they normally do."
            )
            st.dataframe(ta[cols].reset_index(drop=True), width="stretch", hide_index=True)

    # ---- Table B: TEAM penalties taken per game by type ----
    st.markdown(f"**{team} penalties taken per game**")
    brows = []
    for t in REF_TYPES:
        rate = ((against["penalty_type"] == t).sum() / 2) / games_T
        lt = L_type[t]
        brows.append({"Penalty": t,
                      "Per game": f"{rate:.2f} ({rate - lt:+.2f})" if pd.notna(lt) else f"{rate:.2f}"})
    rate_all = (len(against) / 2) / games_T
    brows.append({"Penalty": "All penalties",
                  "Per game": f"{rate_all:.2f} ({rate_all - L_overall:+.2f})"
                  if pd.notna(L_overall) else f"{rate_all:.2f}"})
    st.caption(
        "Each cell is `rate (Δ vs league average)` — a positive bracket means "
        f"{team} takes more of that penalty per game than a league-average team."
    )
    st.dataframe(pd.DataFrame(brows), width="stretch", hide_index=True)


def render_referees(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Referees</h2>",
        unsafe_allow_html=True,
    )
    if game_type == "Playoffs":
        st.info("Referee data covers regular-season games only.")
        return
    if season_label == "2022-23":
        st.info("Referee data covers 2023-24 onward — no 2022-23 data.")
        return

    df = load_ref_penalties()
    if df.empty:
        st.error("Referee data not found "
                 "(`Referees/output/all_teams_penalties_3seasons.csv`).")
        return
    df = _ref_season_scope(df, season_label)
    if df.empty:
        st.info("No referee data for this season.")
        return

    st.markdown(
        f"<p style='color:{PALETTE['text_secondary']}; font-size:0.85rem; font-style:italic; "
        f"max-width:62rem;'>Each game has two referees and the NHL doesn't publish which "
        "official called a given penalty — so these are the penalty environment in games each "
        "referee worked (with a partner), not penalties personally assigned.</p>",
        unsafe_allow_html=True,
    )

    teams = sorted({t for t in set(df["home_team"]) | set(df["away_team"])
                    if isinstance(t, str) and len(t) == 3})
    c1, c2 = st.columns([1.0, 1.6])
    with c1:
        team_sel = st.selectbox("Team", ["All teams"] + teams, key="refs_team")
    with c2:
        name_q = st.text_input("Referee name contains", key="refs_name").strip().lower()

    if team_sel == "All teams":
        _render_ref_league(df, season_label, name_q)
    else:
        _render_ref_team(df, season_label, team_sel, name_q)


# ---------------------------------------------------------------------------
# Global sidebar (Season + Game type — apply to every tab)
# ---------------------------------------------------------------------------
POOLED_4YR_LABEL = "4yr (2022-2026)"


def render_global_filters() -> tuple[str, str]:
    """Season + game-type filters in the main page body (no sidebar).

    Playoffs only ship the pooled view (single-playoff-year samples are too
    small), so when Playoffs is selected the Season filter is locked to the
    4-year pooled view — shown disabled, with the user's regular-season pick
    preserved (separate widget key) for when they switch back."""
    st.session_state.setdefault("g_game_type", "Regular Season")
    # Non-widget mirror of the regular-season pick. Streamlit drops a widget's
    # state when it isn't rendered (i.e. while the Season box is hidden in
    # playoff mode), so we stash the choice here to restore it on the way back.
    st.session_state.setdefault("g_season_pick", "2025-26")
    is_playoffs = st.session_state.get("g_game_type") == "Playoffs"
    season_opts = list(SEASON_KEY.keys())
    c1, c2 = st.columns([1.2, 2.4])
    with c1:
        if is_playoffs:
            st.selectbox("Season", season_opts,
                         index=season_opts.index(POOLED_4YR_LABEL),
                         disabled=True, key="g_season_locked")
            season = POOLED_4YR_LABEL
        else:
            season = st.selectbox(
                "Season", season_opts,
                index=season_opts.index(st.session_state["g_season_pick"]),
                key="g_season")
            st.session_state["g_season_pick"] = season
    with c2:
        game_type = st.radio("Game type", ["Regular Season", "Playoffs"],
                             horizontal=True, key="g_game_type")
    if game_type == "Playoffs":
        st.caption("Playoffs pool all seasons (2022-23 → 2024-25) — single-season "
                   "samples are too small, so the Season filter is locked to the "
                   "pooled view.")
    else:
        st.caption("Season and game type apply across all tabs.")
    return season, game_type


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="HockeyROI — NHL Net-Front Impact, Zone Impact, Quality Games & Quality Starts (GSAx)",
        page_icon="🏒",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    render_header()
    season_label, game_type = render_global_filters()
    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    (player_list_tab, player_detail_tab, goalie_list_tab, goalie_detail_tab,
     trade_tab, teams_tab, refs_tab, meth_tab) = st.tabs(TAB_LABELS)
    with player_list_tab:
        render_players(season_label, game_type)
    with player_detail_tab:
        render_player_detail(season_label, game_type)
    with goalie_list_tab:
        render_goalies(season_label, game_type)
    with goalie_detail_tab:
        render_goalie_detail(season_label, game_type)
    with trade_tab:
        render_trade_analyzer(season_label, game_type)
    with teams_tab:
        render_teams(season_label, game_type)
    with refs_tab:
        render_referees(season_label, game_type)
    with meth_tab:
        render_methodology()

    render_footer()


if __name__ == "__main__":
    main()
