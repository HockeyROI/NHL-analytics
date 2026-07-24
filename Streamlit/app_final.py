"""HockeyROI — NHL Impact Analytics (Net-Front Impact, Zone, xG, Quality Games, Goalie GSAx)."""
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
EDGE_DIR = REPO_ROOT / "edge" / "output"


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
_CHART_FOURTH = "#7E57C2"                # purple — 4th line on 4-series charts
# Line-chart palette by ASPECT (matches the QG line chart): overall = orange,
# offense / For / attack = light-blue, defense / Against / suppress = blue.
_CHART_COLORS = {
    "NFI%": _CHART_PRIMARY,
    "RelNFI%": _CHART_PRIMARY, "RelNFI-A%": _CHART_SECOND, "RelNFI-S%": _CHART_THIRD,
    "NFI-A/60": _CHART_SECOND, "NFI-S/60": _CHART_THIRD,
    "NZI": _CHART_PRIMARY, "DZI": _CHART_SECOND, "OZI": _CHART_THIRD,
    "TZI": _CHART_FOURTH,
    "NFI-QG%": _CHART_PRIMARY, "xG-QG%": _CHART_SECOND,
    "RelNFI-QG%": _CHART_PRIMARY, "RelxG-QG%": _CHART_SECOND, "RelxG%": _CHART_PRIMARY,
    "NFI-GSAx/60": _CHART_PRIMARY, "QNFG%": _CHART_PRIMARY, "QG%": _CHART_SECOND,
    # xG family — For = orange, Against = blue (two shades of blue read too similar)
    "xGF/60": _CHART_PRIMARY, "xGA/60": _CHART_THIRD,
    "RelxG-F%": _CHART_SECOND, "RelxG-A%": _CHART_THIRD,
    # goalie extras (NFI SV% = purple; sQS% = blue/third)
    "NFI SV%": _CHART_FOURTH, "sQS%": _CHART_THIRD,
    "NFI-GSAx": _CHART_PRIMARY, "MP-GSAx": _CHART_THIRD,
    "MP-GSAx/60": _CHART_THIRD,
    # EDGE zone-time trio (same "3rd line = brand blue" convention as NZI/DZI/OZI);
    # the other EDGE charts are single-line, so they default to orange below.
    "EDGE OZ%": _CHART_PRIMARY, "EDGE NZ%": _CHART_SECOND, "EDGE DZ%": _CHART_THIRD,
    "EDGE Top Speed": _CHART_PRIMARY, "EDGE Bursts 20+": _CHART_PRIMARY,
    "EDGE Distance (mi)": _CHART_PRIMARY,
    "OZ Start%": _CHART_PRIMARY, "DZ Start%": _CHART_THIRD,
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
        <div class="tagline">NHL Impact Analytics — Net-Front Impact, Zone Impact, xG &amp; Quality Games, PDO, and NHL EDGE tracking, for skaters and goalies</div>
        <div style="color:#888888; font-size:0.85rem; margin-top:0.15rem;">
          <a href="https://github.com/HockeyROI/NHL-analytics/blob/main/docs/METHODOLOGY.md" style="color:#2E7DC4;">Methodology on GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "ℹ️ A **blank cell** on the Player/Goalie lists means that player or goalie "
        "fell below the metric's qualifying **sample-size** minimum for that scope — "
        "it's “not enough data”, not zero. Each metric shows **(league rank / team "
        "rank)** — rank within **all skaters (F + D)** league-wide, then within their "
        "own team. Only players with **≥ 500 ES minutes** are ranked; lower the Min ES "
        "TOI slider to reveal the rest as **(UR)** = unranked (same meaning as a blank "
        "cell — shown but unranked, not zero). NFI-S/60 (shots against): lowest = #1."
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


# Abbreviation → full name, shown as a hover tooltip on the column header of any
# data table (via st.column_config help). Covers the metric abbreviations used
# across the Players, Goalies, Teams and Referees tables. PDO is deliberately
# just "Luck" — the whole point of the metric is that it's a luck proxy.
_ABBR_FULL = {
    # Zone Impact index (0–100, 50 = position-group average)
    "OZI": "Offensive Zone Impact — O-zone time after offensive-zone faceoffs (0–100, 50 = average)",
    "DZI": "Defensive Zone Impact — O-zone time after defensive-zone faceoffs (0–100, 50 = average)",
    "NZI": "Neutral Zone Impact — O-zone time after neutral-zone faceoffs (0–100, 50 = average)",
    "TZI": "Transitional Zone Impact — O-zone minus D-zone time after neutral-zone faceoffs (0–100, 50 = average)",
    "OZ Start%": "Offensive-Zone Start % — share of faceoff-started shifts beginning in the O-zone",
    "DZ Start%": "Defensive-Zone Start % — share of faceoff-started shifts beginning in the D-zone",
    "NZ Start%": "Neutral-Zone Start % — share of faceoff-started shifts beginning in the N-zone",
    # Net Front Impact family
    "NFI%": "Net Front Impact %",
    "NFI-A/60": "Net Front Impact — Attack per 60 minutes",
    "NFI-S/60": "Net Front Impact — Suppress per 60 minutes",
    "RelNFI%": "Relative Net Front Impact % (vs own team)",
    "RelNFI-A%": "Relative Net Front Impact — Attack % (vs own team)",
    "RelNFI-S%": "Relative Net Front Impact — Suppress % (vs own team)",
    # Quality Games family
    "NFI-QG%": "Net Front Impact — Quality Games %",
    "NFI-QG-A%": "Net Front Impact — Quality Games (Attack) %",
    "NFI-QG-S%": "Net Front Impact — Quality Games (Suppress) %",
    "RelNFI-QG%": "Relative Net Front Impact — Quality Games %",
    "xG-QG%": "Expected Goals — Quality Games %",
    "xG-QG-F%": "Expected Goals — Quality Games (For) %",
    "xG-QG-A%": "Expected Goals — Quality Games (Against) %",
    "RelxG-QG%": "Relative Expected Goals — Quality Games %",
    "RelxG-QG-F%": "Relative Expected Goals — Quality Games (For) %, vs own team",
    "RelxG-QG-A%": "Relative Expected Goals — Quality Games (Against) %, vs own team",
    "RelNFI-QG-A%": "Relative Net Front Impact — Quality Games (Attack) %, vs own team",
    "RelNFI-QG-S%": "Relative Net Front Impact — Quality Games (Suppress) %, vs own team",
    # xG family
    "xGF/60": "Expected Goals For per 60 minutes",
    "xGA/60": "Expected Goals Against per 60 minutes",
    "xG%": "Expected Goals % (share of on-ice expected goals that are for)",
    "RelxG%": "Relative Expected Goals % (vs own team)",
    "RelxG-F%": "Relative Expected Goals — For % (vs own team)",
    "RelxG-A%": "Relative Expected Goals — Against % (vs own team)",
    "PDO": "Luck",
    "TOI/GP": "Even-strength minutes per game (ES TOI ÷ games played)",
    "PDOxG": "Luck, xG-adjusted — SH% & SV% net of expected goals (0-centered)",
    # NHL EDGE tracking
    "EDGE OZ%": "NHL EDGE — Offensive-Zone time %",
    "EDGE OZ% (EV)": "NHL EDGE — Offensive-Zone time % (even strength)",
    "EDGE NZ%": "NHL EDGE — Neutral-Zone time %",
    "EDGE DZ%": "NHL EDGE — Defensive-Zone time %",
    "EDGE Top Speed": "NHL EDGE — Top skating speed (mph)",
    "EDGE Bursts 20+": "NHL EDGE — Number of 20+ mph speed bursts (season total)",
    "EDGE Bursts/60": "NHL EDGE — 20+ mph speed bursts per 60 minutes (all situations)",
    "EZI": "EDGE Zone Impact — O-zone time earned above/below what O-zone faceoff starts predict (0–100, 50 = average)",
    "EDGE Distance (mi)": "NHL EDGE — Distance skated (miles)",
    "EDGE Distance/60": "NHL EDGE — Distance skated per 60 minutes (miles)",
    # Goalies
    "NFI-GSAx": "Net Front Impact — Goals Saved Above Expected (net-front shots)",
    "NFI-GSAx/60": "Net Front Impact — Goals Saved Above Expected per 60 minutes",
    "NFI SV%": "Net Front Impact — Save % (on the net-front shot set)",
    "QNFG%": "Quality Net-Front Games % — share of games beating expected on net-front shots",
    "QG": "Quality Games — share of games with all-shot Goals Saved Above Expected ≥ 0",
    "QGx": "Quality Games (GSAx-based) — share of games with GSAx ≥ 0",
    "sQS%": "Starter Quality Start % — share of games clearing that season's starter-tier save% baseline",
    "QG%s": "Quality Start % vs Starter-tier save% baseline",
    "QG%b": "Quality Start % vs Backup-tier save% baseline",
    "MP-GSAx": "MoneyPuck Goals Saved Above Expected",
    "MP-GSAx/60": "MoneyPuck Goals Saved Above Expected per 60 minutes",
    # Per-situation suite — all follow the Situation toggle (5v5/PP/PK/4v4/3v3/5v3/All)
    # Situation-driven columns (these follow the Situation filter). Listed after
    # the originals above so they override those entries with the fuller text.
    "CF/60": "On-ice Corsi (shot attempts) For per 60 — follows the Situation filter",
    "CA/60": "On-ice Corsi Against per 60 — follows the Situation filter",
    "CF%": "On-ice Corsi For % (CF/(CF+CA)) — follows the Situation filter",
    "FF/60": "On-ice Fenwick (unblocked attempts) For per 60 — Situation filter",
    "FA/60": "On-ice Fenwick Against per 60 — Situation filter",
    "FF%": "On-ice Fenwick For % — Situation filter",
    "xGF/60": "On-ice Expected Goals For per 60 (HockeyROI xG model) — Situation filter",
    "xGA/60": "On-ice Expected Goals Against per 60 — Situation filter",
    "xG%": "On-ice Expected Goals For % (xGF/(xGF+xGA)) — Situation filter",
    "GF/60": "On-ice Goals For per 60 — Situation filter",
    "GA/60": "On-ice Goals Against per 60 — Situation filter",
    "GF%": "On-ice Goals For % — Situation filter",
    "iCF/60": "Individual shot attempts per 60 — Situation filter",
    "ixG/60": "Individual Expected Goals per 60 — Situation filter",
    "iG/60": "Individual Goals per 60 — Situation filter",
    "ixG": "Individual Expected Goals (total) — Situation filter",
    "iG": "Individual Goals (total) — Situation filter",
    "PP Value": "Power-play value: on-ice xGF/60 + CF/60 (higher = better) — set Situation to PP",
    "PK Value": "Penalty-kill value: on-ice xGA/60 + CA/60 (LOWER = better) — set Situation to PK",
    "RelCF%": "Relative Corsi For % — on-ice CF% minus the team's CF% with the player OFF (Situation filter; season-aggregate on/off, exact for one-team players, approximate across trades)",
    "RelxGF%": "Relative xGF % — on-ice xGF% minus the team's xGF% with the player OFF (Situation filter; season-aggregate on/off)",
    "PDO": "PDO (luck) — on-ice SH% + SV% (SOG-based), Situation filter; ~100 = neutral, higher = running hot. Raw, NOT xG-adjusted (that's PDOxG).",
    "PDOxG": "PDOxG — PDO net of expected: (SH% − xSH%) + (SV% − xSV%) from the xG model, SOG-based. 0-centered; + = finishing/goaltending above xG.",
}


def _show_df(obj, **kwargs) -> None:
    """st.dataframe with the leading identity column pinned (frozen on the left)
    and columns sized to their content so numbers aren't clipped — the table
    scrolls horizontally instead of squeezing every column. Works for plain
    DataFrames and Stylers (Styler.data holds the underlying frame). Any column
    whose name is a known abbreviation (_ABBR_FULL) gets a hover-tooltip on its
    header spelling out the full metric name."""
    cols = obj.data.columns if hasattr(obj, "data") else obj.columns
    if len(cols):
        cc = dict(kwargs.pop("column_config", {}) or {})
        # The leading "Season" column doesn't size to content reliably (it clips),
        # so give it an explicit pixel width fitted to its longest label: tight for
        # single-season views ("2024-25") and just wide enough for the long pooled
        # "2yr avg (24-26)" row — no dead space, no clipping (important on mobile).
        if str(cols[0]) == "Season":
            _frame = obj.data if hasattr(obj, "data") else obj
            _vals = _frame[cols[0]].astype(str).tolist()
            _maxlen = max([len("Season")] + [len(v) for v in _vals])
            _w = min(190, max(50, round(_maxlen * 7.0) + 12))
        else:
            _w = None   # other leading columns stay content-sized
        def _abbr_help(name):
            """Full name for a column, matching the bare abbrev or a suffixed one
            like 'OZI (4yr)' → look up 'OZI'."""
            s = str(name)
            return _ABBR_FULL.get(s) or _ABBR_FULL.get(s.split(" (")[0])
        cc.setdefault(cols[0], st.column_config.Column(
            pinned=True, width=_w, help=_abbr_help(cols[0])))
        # Header hover-tooltips: spell out each abbreviation's full name.
        for _c in cols[1:]:
            _full = _abbr_help(_c)
            if _full and _c not in cc:
                cc[_c] = st.column_config.Column(help=_full)
        kwargs["column_config"] = cc
    kwargs["width"] = "content"   # size to content (no clipping) rather than stretch
    return st.dataframe(obj, **kwargs)   # returns selection state when on_select set


def _apply_ranks(disp, fmt, cohort, rank_cols, lower_better=(), second_cohort=None,
                 mark_unranked=False, qualified=None, team_rank_idx=None):
    """Append ' (rank)' to each ranked column's DISPLAY string while leaving the
    underlying cell value numeric, so header-sort still orders by the real value.

    Ranks are computed over `cohort` (a frame sharing the display column names),
    #1 = best; columns in `lower_better` rank lowest-value-first. NaN cells get
    no rank. If `second_cohort` is given (e.g. a single team), a second rank
    within it is appended as ' (league / team)'.

    Two modes:
    • qualified=None (default): value→rank map applied via the Styler formatter.
      Used for self-ranked tables (e.g. Teams) where every row is rankable.
    • qualified=<bool Series aligned to disp.index>: ROW-BASED. A row gets its
      rank only when its mask is True; otherwise ' (UR)' (unranked) when
      mark_unranked, else nothing. To keep the decision per-row (not per-value,
      which would mis-rank a sub-floor row whose fraction equals a qualified
      row's), cells are nudged by an index-scaled epsilon (≤1e-6, invisible to
      the formatted value and to sort order) so each row maps to its own string."""
    lower = set(lower_better)
    rowwise = qualified is not None
    for col in rank_cols:
        if col not in disp.columns or col not in cohort.columns or col not in fmt:
            continue
        s = pd.to_numeric(cohort[col], errors="coerce")
        ranks = s.rank(ascending=(col in lower), method="min")
        vmap = {v: int(r) for v, r in zip(s.values, ranks.values)
                if pd.notna(v) and pd.notna(r)}
        vmap2 = None
        if second_cohort is not None and col in second_cohort.columns:
            s2 = pd.to_numeric(second_cohort[col], errors="coerce")
            r2 = s2.rank(ascending=(col in lower), method="min")
            vmap2 = {v: int(r) for v, r in zip(s2.values, r2.values)
                     if pd.notna(v) and pd.notna(r)}

        if not rowwise:
            def _mk(base_f, vm, vm2, mark):
                def f(x):
                    if pd.isna(x):
                        return base_f(x)
                    r = vm.get(x)
                    if r is None:
                        return f"{base_f(x)} (UR)" if mark else base_f(x)
                    r2 = vm2.get(x) if vm2 is not None else None
                    return (f"{base_f(x)} ({r} / {r2})" if r2 is not None
                            else f"{base_f(x)} ({r})")
                return f
            fmt[col] = _mk(fmt[col], vmap, vmap2, mark_unranked)
            continue

        # Row-based: build each row's suffix from its OWN qualification, then make
        # cell values unique (real + i·1e-9) so the formatter keys back to it. The
        # second (team) rank comes from team_rank_idx[col][row] when given (per-row
        # own-team rank), else from second_cohort's value map.
        real = pd.to_numeric(disp[col], errors="coerce")
        qmask = qualified.reindex(disp.index).fillna(False).astype(bool)
        _trk = (team_rank_idx or {}).get(col, {})
        suffix = {}
        # A column that happens to be all-whole-number with no NaNs in the
        # filtered set (e.g. a Team filter shrinking the sample) can infer as
        # int64, which can't hold the epsilon-perturbed float below — force
        # float64 so the perturbation always has somewhere to go.
        perturbed = real.astype("float64").copy()
        for i, (idx, x) in enumerate(real.items()):
            if pd.isna(x):
                continue
            px = x + i * 1e-9
            perturbed.iloc[i] = px
            if qmask.iloc[i]:
                r = vmap.get(x)
                if team_rank_idx is not None:
                    r2 = _trk.get(idx)
                else:
                    r2 = vmap2.get(x) if vmap2 is not None else None
                if r is None:
                    suffix[px] = " (UR)" if mark_unranked else ""
                elif r2 is not None:
                    suffix[px] = f" ({r} / {r2})"
                else:
                    suffix[px] = f" ({r})"
            else:
                suffix[px] = " (UR)" if mark_unranked else ""
        disp[col] = perturbed

        def _mk_row(base_f, sfx):
            def f(x):
                if pd.isna(x):
                    return base_f(x)
                return f"{base_f(x)}{sfx.get(x, '')}"
            return f
        fmt[col] = _mk_row(fmt[col], suffix)
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
TAB_LABELS = ["Player List", "Goalie List",
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
            "on-ice danger share beat the position median, on two bases: MoneyPuck xG "
            "(<b>xG-QG%</b>) and NFI (<b>NFI-QG%</b>), with a half-credit rule for exact-median "
            "ties. <b>Relative QG</b> (<b>RelNFI-QG%</b>, <b>RelxG-QG%</b>) runs the same per-game "
            "median test on the player's <i>team-relative</i> danger share (on-ice vs off-ice), so "
            "it credits beating the bar after isolating individual contribution from team strength "
            "— the QG analog of RelNFI%. <b>Offense / Defense split</b> grades each game's offense "
            "and defense separately: the share of games where the player's on-ice offense rate per "
            "60 beat the league position-median, and where the on-ice defense rate per 60 was below "
            "it. xG uses For/Against labels (<b>xG-QG-F%</b>, <b>xG-QG-A%</b>); NFI uses "
            "Attack/Suppress to match the NFI family (<b>NFI-QG-A%</b> = attack/offense, "
            "<b>NFI-QG-S%</b> = suppress/defense). Higher is better on all four; available at team "
            "level too.",
        )
        + _meth_framework(
            "Situations (all game states)",
            "The <b>Situation</b> toggle (and the <b>Sit</b> columns / Situation-splits table) "
            "recomputes the on-ice possession/xG/individual suite for a chosen game state: "
            "<b>5v5</b>, <b>PP</b> (5v4+5v3+4v3), <b>PK</b> (4v5+3v5+3v4), <b>4v4</b>, <b>3v3</b> "
            "(regular-season OT), <b>5v3</b>, or <b>All</b>. Built from a fresh per-situation on-ice "
            "attribution over the raw shot + shift data: each strength state comes from the event "
            "<i>situation_code</i>, on-ice players from the shift intervals, and time-on-ice per "
            "situation is reconstructed the same way (so every rate is counts ÷ that situation's TOI, "
            "by ratio-of-sums). Columns: <b>CF/CA, FF/FA, xGF/xGA, GF/GA</b> per-60 and their shares "
            "(<b>CF%/FF%/xGF%/GF%</b>), individual <b>iCF/ixG/iG</b> per-60, on/off relatives "
            "(<b>Sit RelCF%/RelxGF%</b> — the player's share minus his team's share with him off, "
            "per situation; a season-aggregate on/off, exact for one-team players and approximate "
            "across mid-scope trades), plus dedicated special-teams value scores — "
            "<b>PP xGF+CF/60</b> (higher = better) and <b>PK xGA+CA/60</b> (lower = better), an "
            "NST-style single number for power-play offense and penalty-kill defense. xG is the "
            "HockeyROI model (not MoneyPuck), and a matching Situation toggle on the Teams tab gives "
            "team-level versions of all these. "
            "<b>Scope note:</b> the bespoke 5v5-native families — RelNFI, Quality Games, Zone Impact "
            "— stay on their 5v5 basis and are <i>not</i> re-derived per situation (their "
            "team-relative / per-game-median / faceoff-anchored constructions assume even strength). "
            "Per-situation league totals also carry an on-ice roster-size multiplier, so compare "
            "players on the per-60 rates and shares, not raw totals.",
        )
        + _meth_framework(
            "Expected Goals (xG) — our own model",
            "The xG behind <b>PDOxG</b> and the on-ice <b>xGF/xGA</b> (incl. the per-situation "
            "engine) is HockeyROI's own — a Fenwick-based gradient-boosted model "
            "(<code>xG/build_xg.py</code>) trained on shot geometry (distance / angle), shot type, "
            "and pre-shot context read from the <b>full play-by-play</b>: rebound, <b>rush</b>, time "
            "&amp; distance since the last event of any kind (faceoff / hit / turnover / shot), the "
            "last event's type and zone, plus running score &amp; strength state and home/away. "
            "Held-out <b>AUC 0.764</b>, tightly calibrated, and its player-season xG totals correlate "
            "<b>0.99 with MoneyPuck's</b> published xGoal — a fully in-house model that tracks the "
            "public benchmark closely. "
            "<i>(The Quality-Games xG is a separate, deliberately MoneyPuck-sourced input, labeled "
            "as such — not this model.)</i>",
        )
        + _meth_framework(
            "xG — Expected Goals",
            "On-ice expected goals, MoneyPuck-style, split into For and Against. <b>xGF/60</b> and "
            "<b>xGA/60</b> are the raw on-ice expected goals for / against per 60 while the player "
            "is on the ice. <b>RelxG%</b> is the season-level <i>relative</i> xG rate (on-ice − "
            "off-ice per 60) — the xG counterpart to RelNFI% — and <b>RelxG-F%</b> / <b>RelxG-A%</b> "
            "split that relative rate into its For and Against halves. All are built from MoneyPuck's "
            "raw shot data with HockeyROI's own qualifying filter, so they can differ from "
            "MoneyPuck's published columns — different filters/aggregation, not a question of "
            "accuracy. (Split out of Quality Games: QG is the per-game consistency %; xG is the "
            "underlying rate.)",
        )
        + _meth_framework(
            "Teams",
            "Team-level CNFI+MNFI share, plus a roster-talent-vs-on-ice-result “two ways” "
            "comparison that flags teams whose talent and results diverge.",
        )
        + _meth_framework(
            "Goalies",
            "Two families. GSAx-based: <b>NFI-GSAx</b> (net-front goals saved above expected), "
            "<b>QNFG%</b> (consistency of beating expected on net-front shots), and "
            "<b>QG</b> (share of games with all-shot GSAx ≥ 0 — goals-saved-above-expected, "
            "as a game rate; formerly GQG). <b>NFI SV%</b> is the one raw-save% exception in that "
            "family — an unadjusted save% on NFI-GSAx's own net-front shot set, a sanity check, "
            "not a replacement. Save%-based: <b>sQS%</b> — the little <b>s</b> is <b>Starter</b>. "
            "sQS% = share of games where per-game save% (5v5 shots on goal) cleared that season's "
            "starter-tier baseline — the volume-weighted save% of that season's top-32-GP "
            "(starter) or next-32-GP (backup) goalies. A traditional Quality Start, but against a "
            "population-specific bar recomputed every season instead of one fixed league-average "
            "line. Toggle above switches baseline/scope; the leaderboard always shows the "
            "currently-selected one. Qualifying floors differ by metric, so the cohorts differ.",
        )
        + _meth_framework(
            "Zone Impact",
            "<b>OZI / DZI / NZI / TZI</b> — four lenses for a player's offensive-zone time after 5v5 "
            "faceoffs, expressed as a <b>0–100 index where 50 = the position-group average</b> "
            "(above 50 = more O-zone time than an average forward / defenseman, below = less). "
            "OZI / DZI / NZI measure O-zone time after <b>o</b>ffensive / <b>d</b>efensive / "
            "<b>n</b>eutral-zone draws; <b>TZI</b> (Transitional Zone Impact) is the neutral-zone "
            "transition split — O-zone minus D-zone time after neutral draws — so it reads how a "
            "player tilts play out of the neutral zone. The index is built by taking each player's "
            "raw per-shift zone-time percentage and recentring it on the league-average percentage "
            "for their position; the natural spread of the underlying stat sets the spread of the "
            "index (no artificial stretch), so most players sit in a tight band around 50 and only "
            "genuine outliers reach the extremes — the same idea as a save% that lives between .880 "
            "and .920. Independent lenses, not a hierarchy; a complete player rates above 50 across "
            "all four. <b>D/N/O Start%</b> sits alongside them — the plain share of a player's "
            "faceoff-started shifts that began in each zone (my own play-by-play data). It's a "
            "presentation layer showing deployment context, not a new metric — it doesn't feed into "
            "or alter OZI/DZI/NZI/TZI.",
        )
        + _meth_framework(
            "PDO",
            "Shooting% + save% luck proxy, 5v5 or all-situations (toggle-able): SH% = on-ice goals-for "
            "÷ on-ice shots-on-goal-for; SV% = 1 − (on-ice goals-against ÷ on-ice shots-on-goal-"
            "against); <b>PDO</b> = (SH% + SV%) × 100. Shots-on-goal based (not Fenwick/Corsi) — the "
            "conventional definition. Centers on ~100 league-wide; well above/below is usually "
            "unsustainable shooting or save luck rather than skill. A raw descriptive column shown "
            "beside xG — no relative or Quality-Games version, and it isn't used for ranking.",
        )
        + _meth_framework(
            "PDOxG",
            "PDO's luck signal net of shot quality. Standard PDO treats every on-ice shot as a "
            "coin-flip against league-average conversion, but a player who generates (or concedes) "
            "better chances will run a high (or low) PDO by shot quality alone — not luck. PDOxG "
            "replaces the league-average baseline with an <b>expected</b> one from my SOG-conditional "
            "xG model (shot distance/angle/type): PDOxG = (SH% − xSH%) + (SV% − xSV%), where xSH% = "
            "on-ice xGF ÷ SOG-for and xSV% = 1 − (on-ice xGA ÷ SOG-against). <b>Centered on 0</b> — "
            "positive = finishing/goaltending beyond what shot quality predicts (still likely luck, "
            "but the shot-quality portion is removed); negative = the reverse. Same 5v5/all-"
            "situations toggle and ≥200-min floor as PDO. Descriptive, not a ranking. Note it does "
            "<b>not</b> adjust for the finishing talent of linemates or the goalie behind you — that "
            "would be a further, teammate-relative step.",
        )
        + _meth_framework(
            "NHL EDGE",
            "NHL's own player-tracking data (by player position, not puck position) — offensive / "
            "neutral / defensive-zone time share, top skating speed, 20+ mph speed bursts (NHL's "
            "raw season total <b>and</b> a per-60 rate, <b>EDGE Bursts/60</b>), and distance skated "
            "(also per-60). Both per-60 rates now divide by <b>all-situations</b> TOI (from the "
            "per-situation engine), the correct matching denominator for these all-situations "
            "tracking totals — fixing an earlier ES-only-TOI approximation that inflated "
            "special-teams players. The scope toggle defaults to even strength to match the rest "
            "of the page."
            "<br><br>"
            "A <b>different measurement basis</b> than the zone metrics above: EDGE tracks "
            "continuously across all-situations or even-strength TOI (toggle-able for OZ%); "
            "NZI/DZI/OZI and D/N/O Start% track puck/faceoff position after strict-5v5 faceoffs "
            "only — do not read EDGE zone-time% as the same metric as those. Each EDGE value shows "
            "a computed (league / team) rank rather than NHL's own percentile, matching every other "
            "ranked column in the app; pooled/2yr views are a games-played-weighted average across "
            "seasons, regular season only."
            "<br><br>"
            "<b>EZI (EDGE Zone Impact)</b> relates a player's O-zone time to their O-zone faceoff "
            "starts: a 0–100 index (50 = position-group average) built from a <b>regression "
            "residual</b>, not a plain difference. <code>OZ time% − OZ Start%</code> sounds right "
            "but is badly biased — Start% swings far more with deployment (std≈6.7) than O-zone "
            "time actually does (std≈2.4), so a plain subtraction correlated −0.94 with Start% "
            "itself, i.e. it was mostly just an inverted deployment metric. EZI instead fits "
            "<code>OZ time% ~ OZ Start%</code> per position group (slope ≈0.23–0.25 — starts "
            "predict time far more weakly than 1-for-1) and uses the residual: positive means a "
            "player earns more O-zone time than their own starts predict (driving play beyond "
            "deployment), negative means sheltered starts aren't converting into time. The residual "
            "is exactly uncorrelated with Start% by construction, then recentred onto the same "
            "0–100/50-average scale as OZI/DZI/NZI/TZI. Regular season only (no playoff zone-start "
            "data).",
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
            A note on QG / QNFG% vs. sQS% vs. &ldquo;Quality Starts&rdquo;</div>
          <div style="color:{PALETTE['text']}; font-size:0.94rem; line-height:1.5;">
            Robert Vollman's Quality Starts (~2009) flags a game where save% beats a single
            <b>league-average save%</b> line. <b>QG</b> (formerly GQG, formerly QS-GSAx) and
            <b>QNFG%</b> are <b>not</b> that — a quality game there is <b>per-game GSAx &ge; 0</b>,
            the goalie beating expected on a danger / xG-weighted basis, not on raw save%
            (QNFG% on net-front shots only, QG on all shots). Don't map either to Vollman's
            metric, or map them to each other.<br><br>
            <b>sQS% is save%-based</b>, in the spirit of Vollman — but instead of one fixed
            league-average line, it uses a baseline recomputed every season: the volume-weighted
            save% of that season's top-32-GP &ldquo;starter&rdquo; tier. So <b>sQS%</b> asks
            &ldquo;does this goalie perform like a starter?&rdquo; rather than merely beating a
            league-average line that a backup would clear too. Toggle the shot scope
            (5v5 / all situations) for sQS% — QG and QNFG% stay 5v5-only regardless.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<p style='color:{PALETTE['text_secondary']}; font-size:0.85rem; max-width:62rem;'>"
        "Cohorts differ by metric by design — qualifying floors vary (e.g. pooled goalie NFI-GSAx "
        "requires ≥300 net-front shots; QNFG% requires ≥25 games in a season), so a player or "
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
    # Referee data only exists 2023-24+; this pool is its 3-season view. The
    # metric tabs show a "not available" notice for it (see REF_ONLY_LABEL).
    "3yr (2023-2026) — Referees only": "ref_pooled",
    "4yr (2022-2026)": "pooled",
}
REF_ONLY_LABEL = "3yr (2023-2026) — Referees only"


def _block_ref_only(season_label: str) -> bool:
    """The 3yr Referees pool is not a metric scope. When it's selected, show a
    notice on a metric tab and signal the caller to stop. Returns True if blocked."""
    if season_label == REF_ONLY_LABEL:
        st.info("The **3yr (2023-2026)** pool is a Referees-only view — pick a single "
                "season or the 2yr / 4yr pool for this tab.")
        return True
    return False

# Seasons covered by the "2yr (2024–2026)" pooled-style view. Loaders that
# aggregate across seasons restrict to these when the season key is
# "pooled_2yr"; tabs without per-season support for 2yr fall back gracefully.
POOLED_2YR_SEASONS = ["20242025", "20252026"]


@st.cache_data(show_spinner=False, ttl=3600)
def load_qg_player_season() -> pd.DataFrame:
    """Quality Games per (player, season) — xG-QG% / NFI-QG% / GP / qual GP."""
    fp = REPO_ROOT / "Quality_Games" / "output" / "per_player_season.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype("Int64")
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_player_season_team_order() -> dict:
    """{(player_id:int, season:str): [team, team, ...]} — the teams a player
    suited up for that season in the order they first appeared (by game_id, which
    is chronological within a season). Source: per_player_game.csv. Used to show
    traded players as "EDM / COL" in play order rather than alphabetically."""
    fp = REPO_ROOT / "Quality_Games" / "output" / "per_player_game.csv"
    if not fp.exists():
        return {}
    df = pd.read_csv(fp, usecols=["player_id", "season", "game_id", "team_abbrev"])
    df["season"] = df["season"].astype(str)
    agg = (df.groupby(["player_id", "season", "team_abbrev"])
           .agg(first_game=("game_id", "min"), games=("game_id", "size"))
           .reset_index().sort_values(["player_id", "season", "first_game"]))
    out = {}
    for (pid, sn), g in agg.groupby(["player_id", "season"], sort=False):
        # Drop stray single-game teams (mislabeled game logs — e.g. a 1-game VGK
        # blip for a player who really split the season COL/CGY) when the player
        # has a real multi-game team; keep all if every team is a single game.
        _real = g[g["games"] >= 2]
        keep = _real if not _real.empty else g
        out[(int(pid), sn)] = keep["team_abbrev"].tolist()
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def load_team_rosters() -> dict:
    """{(season:str, team): set(player_id)} — who suited up for each team each
    season (any team they appeared for). Built from the chronological order map;
    used for within-team ranks on the player-detail page."""
    out = {}
    for (pid, sn), teams in load_player_season_team_order().items():
        for t in teams:
            out.setdefault((sn, t), set()).add(int(pid))
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def load_zone_pooled() -> pd.DataFrame:
    """Pooled OZI / DZI / NZI / TZI (0–100 index, 50 = position-group average)
    from zone_index100/pooled_{forwards,defense}.csv. Name-keyed (a `_pos_group`
    column is added so the merge to NFI is on (player_name, pos-group) — this
    guards the rare same-name / different-position case, e.g. the two Sebastian
    Ahos). See build_zone_index100.py for the 50=average rescale."""
    frames = []
    for pos_file, grp in (("forwards", "F"), ("defense", "D")):
        fp = ADJ / "zone_index100" / f"pooled_{pos_file}.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        keep = [c for c in ("player_name", "OZI", "DZI", "NZI", "TZI")
                if c in d.columns]
        d = d[keep].copy()
        d["_pos_group"] = grp
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    z = pd.concat(frames, ignore_index=True)
    return z.drop_duplicates(subset=["player_name", "_pos_group"], keep="first")


@st.cache_data(show_spinner=False, ttl=3600)
def load_zone_2yr() -> pd.DataFrame:
    """2-year (2024-25 + 2025-26) pooled OZI/DZI/NZI/TZI (0–100 index, 50 =
    position-group average) from zone_index100/2yr_{forwards,defense}.csv.
    Name-keyed, so the merge mirrors load_zone_pooled exactly: on
    (player_name, _pos_group). The 2yr index is recentred on the 2yr pool's own
    per-position average (not carried over from the 4-yr pool)."""
    frames = []
    for pos_file, grp in (("forwards", "F"), ("defense", "D")):
        fp = ADJ / "zone_index100" / f"2yr_{pos_file}.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        keep = [c for c in ("player_name", "OZI", "DZI", "NZI", "TZI")
                if c in d.columns]
        d = d[keep].copy()
        d["_pos_group"] = grp
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    z = pd.concat(frames, ignore_index=True)
    return z.drop_duplicates(subset=["player_name", "_pos_group"], keep="first")


def _qg_pooled(qg: pd.DataFrame) -> pd.DataFrame:
    """Career-pooled QG: rates = total quality games / total qualifying GP."""
    if qg.empty:
        return qg
    # QG-flag rates (xG/NFI/RelxG/RelNFI _QG) pool by ratio-of-sums on their
    # count/qual_GP denominators. RelxG_pct is a per-60 rate with no count
    # denominator, so it pools by TOI-weighted mean (weights = TOI_total_sec) —
    # the same method _aggregate_nfi_pooled uses for RelNFI_pct.
    aggs = dict(
        GP=("GP", "sum"),
        qualifying_GP=("qualifying_GP", "sum"),
        _xc=("xG_QG_count", "sum"), _xq=("xG_qual_GP", "sum"),
        _nc=("NFI_QG_count", "sum"), _nq=("NFI_qual_GP", "sum"),
    )
    _has_rel = {"RelxG_QG_count", "RelxG_qual_GP",
                "RelNFI_QG_count", "RelNFI_qual_GP"}.issubset(qg.columns)
    if _has_rel:
        aggs.update(_rxc=("RelxG_QG_count", "sum"), _rxq=("RelxG_qual_GP", "sum"),
                    _rnc=("RelNFI_QG_count", "sum"), _rnq=("RelNFI_qual_GP", "sum"))
    g = qg.groupby("player_id").agg(**aggs).reset_index()
    g["xG_QG_pct"] = np.where(g["_xq"] > 0, g["_xc"] / g["_xq"], np.nan)
    g["NFI_QG_pct"] = np.where(g["_nq"] > 0, g["_nc"] / g["_nq"], np.nan)
    out_cols = ["player_id", "GP", "qualifying_GP", "xG_QG_pct", "NFI_QG_pct"]
    if _has_rel:
        g["RelxG_QG_pct"] = np.where(g["_rxq"] > 0, g["_rxc"] / g["_rxq"], np.nan)
        g["RelNFI_QG_pct"] = np.where(g["_rnq"] > 0, g["_rnc"] / g["_rnq"], np.nan)
        out_cols += ["RelxG_QG_pct", "RelNFI_QG_pct"]
    # RelxG_pct / RelxG_F_pct / RelxG_A_pct — TOI-weighted means of per-season
    # values (no count denominator), mirroring _aggregate_nfi_pooled's RelNFI_pct.
    for _rc in ("RelxG_pct", "RelxG_F_pct", "RelxG_A_pct"):
        if {_rc, "TOI_total_sec"}.issubset(qg.columns):
            t = qg[["player_id", _rc, "TOI_total_sec"]].copy()
            t[_rc] = pd.to_numeric(t[_rc], errors="coerce")
            t["w"] = pd.to_numeric(t["TOI_total_sec"], errors="coerce")
            t = t[t[_rc].notna() & (t["w"] > 0)]
            t["_num"] = t[_rc] * t["w"]
            rx = t.groupby("player_id").agg(_num=("_num", "sum"),
                                            _den=("w", "sum")).reset_index()
            rx[_rc] = np.where(rx["_den"] > 0, rx["_num"] / rx["_den"], np.nan)
            g = g.merge(rx[["player_id", _rc]], on="player_id", how="left")
            out_cols.append(_rc)
    return g[out_cols]


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
        # Fenwick (blocks excluded) — the raw NFI-A/60 / NFI-S/60 source. The
        # _att (Corsi) columns are still present but the NFI definition is
        # Fenwick; see the matching fix in script 03's per-season writer.
        for_att=("onice_for_fen", "sum"),
        ag_att=("onice_ag_fen", "sum"),
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


# ---------------------------------------------------------------------------
# Per-situation engine (Data/player_situation_onice.csv + _toi.csv)
#   Built by NFI/scripts/build_situation_onice.py — per (player, season,
#   game_type, situation) on-ice + individual counts. The situation toggle rolls
#   the granular skater-matchups into display buckets and derives the full
#   possession/xG/individual suite by ratio-of-sums (sum counts / sum TOI).
# ---------------------------------------------------------------------------
SITUATION_BUCKETS: dict[str, list[str] | None] = {
    "All situations": None,           # every matchup
    "5v5": ["5v5"],
    "PP": ["5v4", "5v3", "4v3"],      # man-advantage (player's own team up)
    "PK": ["4v5", "3v5", "3v4"],      # shorthanded
    "4v4": ["4v4"],
    "3v3": ["3v3"],
    "5v3": ["5v3"],
}
# Buckets where the bespoke 5v5-native families (RelNFI/QG/Zone) still apply
# as-is. Outside this, those columns are shown labeled "5v5" or as N/A.
_SIT_IS_5V5 = "5v5"


@st.cache_data(show_spinner=False, ttl=3600)
def load_situation_onice() -> pd.DataFrame:
    fp = REPO_ROOT / "Data" / "player_situation_onice.csv"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_csv(fp, dtype={"season": str})


def _situation_toggle_state() -> str:
    """Shared situation-scope toggle (set by the widget in render_players, read
    wherever the per-situation suite is built this run — same shared-session
    pattern as _pdo_toggle_state). Defaults to 5v5 (the app's native basis)."""
    s = st.session_state.get("g_situation", "5v5")
    return s if s in SITUATION_BUCKETS else "5v5"


def _allsit_toi(scope_key: str, playoffs: bool = False) -> pd.DataFrame:
    """Per-player TOTAL all-situations TOI (sum over every situation) for one
    scope, by ratio-of-sums. This is the correct denominator for NHL EDGE
    all-situations totals (speed bursts, distance) — the NFI toi_min is ES-only
    and silently inflated EDGE per-60 rates for special-teams players."""
    df = load_situation_onice()
    if df.empty:
        return pd.DataFrame()
    df = df[df["game_type"] == ("playoff" if playoffs else "regular")]
    seasons = _situation_seasons(scope_key, playoffs)
    if seasons is not None:
        df = df[df["season"].isin(seasons)]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("player_id")["toi_min"].sum().reset_index()
    return g.rename(columns={"toi_min": "allsit_toi_min"})


def _situation_seasons(scope_key: str, playoffs: bool) -> list[str] | None:
    """Seasons included for a scope_key; None means 'all seasons in file'
    (used for the pooled playoff view)."""
    if playoffs:
        return None
    if scope_key == "pooled":
        return list(POOLED_SEASONS)
    if scope_key == "pooled_2yr":
        return list(POOLED_2YR_SEASONS)
    return [scope_key]


# internal count columns produced by the engine
_SIT_COUNTS = ["CF", "FF", "xGF", "GF", "CA", "FA", "xGA", "GA", "SOGF", "SOGA",
               "xGFs", "xGAs", "iCF", "iFF", "ixG", "iG", "toi_min"]


def _situation_metrics(scope_key: str, bucket_label: str,
                       playoffs: bool = False) -> pd.DataFrame:
    """Per-player possession/xG/individual suite for one scope + situation
    bucket, by ratio-of-sums over the scope's seasons and the bucket's
    matchups. Returns display-named rate/share columns keyed on player_id."""
    df = load_situation_onice()
    if df.empty:
        return pd.DataFrame()
    df = df[df["game_type"] == ("playoff" if playoffs else "regular")]
    seasons = _situation_seasons(scope_key, playoffs)
    if seasons is not None:
        df = df[df["season"].isin(seasons)]
    mats = SITUATION_BUCKETS.get(bucket_label, ["5v5"])
    if mats is not None:
        df = df[df["situation"].isin(mats)]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("player_id")[_SIT_COUNTS].sum().reset_index()
    g = g.merge(df.groupby("player_id")["gp"].max().reset_index(), on="player_id")
    toi = g["toi_min"]
    ok = toi > 0

    def per60(col):
        return np.where(ok, g[col] / toi * 60.0, np.nan)

    def share(f, a):
        d = g[f] + g[a]
        return np.where(d > 0, g[f] / d * 100.0, np.nan)

    # Display names are prefixed "Sit " — ASCII-safe, collision-free with the
    # 5v5 xG family's "xGF/60"/"xGA/60", and a clear signal these follow the
    # situation toggle. sit_TOI/sit_GP kept unprefixed for internal ranking use.
    out = pd.DataFrame({"player_id": g["player_id"]})
    out["sit_TOI"] = toi
    out["sit_GP"] = g["gp"]
    out["Sit TOI/GP"] = np.where(g["gp"] > 0, toi / g["gp"], np.nan)
    out["Sit CF/60"], out["Sit CA/60"] = per60("CF"), per60("CA")
    out["Sit FF/60"], out["Sit FA/60"] = per60("FF"), per60("FA")
    out["Sit xGF/60"], out["Sit xGA/60"] = per60("xGF"), per60("xGA")
    out["Sit GF/60"], out["Sit GA/60"] = per60("GF"), per60("GA")
    out["Sit CF%"], out["Sit FF%"] = share("CF", "CA"), share("FF", "FA")
    out["Sit xGF%"], out["Sit GF%"] = share("xGF", "xGA"), share("GF", "GA")
    out["Sit iCF/60"], out["Sit ixG/60"] = per60("iCF"), per60("ixG")
    out["Sit iG/60"] = per60("iG")
    out["Sit ixG"], out["Sit iG"] = g["ixG"].round(2), g["iG"].astype(int)
    # Dedicated special-teams player value (NST-style): PP offence = xGF/60 +
    # CF/60; PK defence = xGA/60 + CA/60 (lower is better on PK).
    out["Sit PP xGF+CF/60"] = out["Sit xGF/60"] + out["Sit CF/60"]
    out["Sit PK xGA+CA/60"] = out["Sit xGA/60"] + out["Sit CA/60"]
    # PDO (raw luck) = on-ice SH% + SV%, SOG-based, per situation (NST-style)
    _ok_pdo = (g["SOGF"] > 0) & (g["SOGA"] > 0)
    _sh = np.where(g["SOGF"] > 0, g["GF"] / g["SOGF"], np.nan)
    _sv = np.where(g["SOGA"] > 0, 1 - g["GA"] / g["SOGA"], np.nan)
    out["Sit PDO"] = np.where(_ok_pdo, (_sh + _sv) * 100, np.nan)
    # PDOxG = luck NET of expected: (SH% - xSH%) + (SV% - xSV%), SOG-based xG.
    # 0-centred; positive = finishing/goaltending above what xG predicts.
    _xsh = np.where(g["SOGF"] > 0, g["xGFs"] / g["SOGF"], np.nan)
    _xsv = np.where(g["SOGA"] > 0, 1 - g["xGAs"] / g["SOGA"], np.nan)
    out["Sit PDOxG"] = np.where(_ok_pdo, ((_sh - _xsh) + (_sv - _xsv)) * 100, np.nan)
    # raw on-ice counts kept (prefixed, not displayed) for the on/off Rel calc
    for c in ("CF", "CA", "xGF", "xGA"):
        out[f"_sr_{c}"] = g[c].values
    out["_sr_toi"] = toi.values
    return out


def _team_situation_raw(scope_key: str, bucket_label: str,
                        playoffs: bool = False) -> pd.DataFrame:
    """Per-team RAW summed on-ice counts for a scope+bucket (for the on/off
    team-without-player baseline). Keyed on 'team'."""
    df = load_team_situation_onice()
    if df.empty:
        return pd.DataFrame()
    df = df[df["game_type"] == ("playoff" if playoffs else "regular")]
    seasons = _situation_seasons(scope_key, playoffs)
    if seasons is not None:
        df = df[df["season"].isin(seasons)]
    mats = SITUATION_BUCKETS.get(bucket_label, ["5v5"])
    if mats is not None:
        df = df[df["situation"].isin(mats)]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("team")[["CF", "CA", "xGF", "xGA", "toi_min"]].sum().reset_index()
    return g.rename(columns={c: f"_tr_{c}" for c in ("CF", "CA", "xGF", "xGA")}
                    ).rename(columns={"toi_min": "_tr_toi"})


def _add_situation_rel(base: pd.DataFrame, scope_key: str, bucket_label: str,
                       playoffs: bool = False) -> pd.DataFrame:
    """Add per-situation on/off relatives (Sit RelCF% / Sit RelxGF%) =
    player's on-ice share minus the SAME team's share with the player OFF.
    Computed only where the without-player split is clean (single team,
    non-traded); traded players and multi-team pooled scopes -> NaN, since a
    single season-team total can't isolate the player's off-ice context there.
    Mirrors the existing 5v5 RelNFI's on/off definition, per situation."""
    need = [f"_sr_{c}" for c in ("CF", "CA", "xGF", "xGA")] + ["_sr_toi"]
    if base.empty or "team" not in base.columns or not all(c in base.columns for c in need):
        return base
    tr = _team_situation_raw(scope_key, bucket_label, playoffs)
    if tr.empty:
        return base
    base = base.merge(tr, on="team", how="left")
    wo_cf = base["_tr_CF"] - base["_sr_CF"]
    wo_ca = base["_tr_CA"] - base["_sr_CA"]
    wo_xgf = base["_tr_xGF"] - base["_sr_xGF"]
    wo_xga = base["_tr_xGA"] - base["_sr_xGA"]
    wo_toi = base["_tr_toi"] - base["_sr_toi"]
    valid = ((wo_toi > 1) & (wo_cf >= 0) & (wo_ca >= 0) & (wo_xgf >= 0)
             & (wo_xga >= 0) & (base["_sr_toi"] > 0))

    def _share(f, a):
        d = f + a
        return np.where(d > 0, f / d * 100.0, np.nan)

    p_cf = _share(base["_sr_CF"], base["_sr_CA"])
    p_xgf = _share(base["_sr_xGF"], base["_sr_xGA"])
    w_cf = _share(wo_cf, wo_ca)
    w_xgf = _share(wo_xgf, wo_xga)
    base["Sit RelCF%"] = np.where(valid, p_cf - w_cf, np.nan)
    base["Sit RelxGF%"] = np.where(valid, p_xgf - w_xgf, np.nan)
    return base.drop(columns=[c for c in base.columns
                              if c.startswith("_tr_") or c.startswith("_sr_")])


@st.cache_data(show_spinner=False, ttl=3600)
def load_team_situation_onice() -> pd.DataFrame:
    fp = REPO_ROOT / "Data" / "team_situation_onice.csv"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_csv(fp, dtype={"season": str})


_TEAM_SIT_COUNTS = ["CF", "FF", "xGF", "GF", "CA", "FA", "xGA", "GA", "toi_min"]


def _team_situation_metrics(scope_key: str, bucket_label: str,
                            playoffs: bool = False) -> pd.DataFrame:
    """Team-level per-situation possession/xG suite (team analog of
    _situation_metrics), keyed on 'Team'. Ratio-of-sums over the scope's
    seasons and the bucket's matchups."""
    df = load_team_situation_onice()
    if df.empty:
        return pd.DataFrame()
    df = df[df["game_type"] == ("playoff" if playoffs else "regular")]
    seasons = _situation_seasons(scope_key, playoffs)
    if seasons is not None:
        df = df[df["season"].isin(seasons)]
    mats = SITUATION_BUCKETS.get(bucket_label, ["5v5"])
    if mats is not None:
        df = df[df["situation"].isin(mats)]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("team")[_TEAM_SIT_COUNTS].sum().reset_index()
    toi = g["toi_min"]
    ok = toi > 0

    def per60(c):
        return np.where(ok, g[c] / toi * 60.0, np.nan)

    def share(f, a):
        d = g[f] + g[a]
        return np.where(d > 0, g[f] / d * 100.0, np.nan)

    out = pd.DataFrame({"Team": g["team"]})
    out["Sit TOI"] = toi
    out["Sit CF%"], out["Sit xGF%"] = share("CF", "CA"), share("xGF", "xGA")
    out["Sit GF%"] = share("GF", "GA")
    out["Sit xGF/60"], out["Sit xGA/60"] = per60("xGF"), per60("xGA")
    out["Sit CF/60"], out["Sit CA/60"] = per60("CF"), per60("CA")
    out["Sit GF/60"], out["Sit GA/60"] = per60("GF"), per60("GA")
    out["Sit PP xGF+CF/60"] = out["Sit xGF/60"] + out["Sit CF/60"]
    out["Sit PK xGA+CA/60"] = out["Sit xGA/60"] + out["Sit CA/60"]
    return out


# Compact team Situation column set shown on the Teams tab (kept tight — the
# team table shows every column at once, no family pills).
TEAM_SIT_COLS = ["Sit TOI", "Sit CF%", "Sit xGF%", "Sit GF%",
                 "Sit xGF/60", "Sit xGA/60", "Sit PP xGF+CF/60", "Sit PK xGA+CA/60"]


# The situation-aware values REPLACE the older 5v5-only columns under the plain
# names, so there is ONE xG/PDO column set and the Situation filter drives it.
_SIT_UNIFY = {
    # these REPLACE the older 5v5-only columns of the same concept
    "Sit xGF/60": "xGF/60", "Sit xGA/60": "xGA/60", "Sit xGF%": "xG%",
    "Sit PDO": "PDO", "Sit PDOxG": "PDOxG",
    # these are new (no old equivalent) — just drop the "Sit " prefix
    "Sit CF/60": "CF/60", "Sit CA/60": "CA/60", "Sit CF%": "CF%",
    "Sit FF/60": "FF/60", "Sit FA/60": "FA/60", "Sit FF%": "FF%",
    "Sit GF/60": "GF/60", "Sit GA/60": "GA/60", "Sit GF%": "GF%",
    "Sit iCF/60": "iCF/60", "Sit ixG/60": "ixG/60", "Sit iG/60": "iG/60",
    "Sit ixG": "ixG", "Sit iG": "iG",
    "Sit RelCF%": "RelCF%", "Sit RelxGF%": "RelxGF%",
    "Sit PP xGF+CF/60": "PP Value", "Sit PK xGA+CA/60": "PK Value",
}
# "Sit TOI/GP" is dropped — the frame already has a TOI/GP column.
_SIT_DROP = ["Sit TOI/GP"]
# The situation-driven columns, under their final plain names (shown in the xG
# family — there is no separate "Situations" family any more).
_SIT_PLAIN = ["CF/60", "CA/60", "CF%", "FF/60", "FA/60", "FF%",
              "GF/60", "GA/60", "GF%", "iCF/60", "ixG/60", "iG/60", "ixG", "iG",
              "RelCF%", "RelxGF%", "PP Value", "PK Value"]


def _unify_situation_xg(base: pd.DataFrame) -> pd.DataFrame:
    """Collapse the duplicate xG/PDO columns: drop the old 5v5-only versions and
    promote the situation-aware ones to the plain names (xGF/60, xG%, PDO, ...).
    Every downstream consumer keeps working — same names, now filter-driven."""
    if base is None or base.empty:
        return base
    base = base.drop(columns=[c for c in _SIT_DROP if c in base.columns],
                     errors="ignore")
    have = {k: v for k, v in _SIT_UNIFY.items() if k in base.columns}
    if not have:
        return base
    base = base.drop(columns=[v for v in have.values() if v in base.columns],
                     errors="ignore")
    return base.rename(columns=have)


_SIT_SPLIT_BUCKETS = ["5v5", "PP", "PK", "4v4", "3v3", "5v3"]


def _player_situation_table(pid: int, season_label: str | None = None,
                            playoffs: bool = False) -> pd.DataFrame:
    """One drilled player's key on-ice/individual metrics across every situation
    bucket at once (rows = 5v5/PP/PK/4v4/3v3/5v3), for the current scope."""
    key = SEASON_KEY.get(season_label, "pooled") if season_label else "pooled"
    rows = []
    for bucket in _SIT_SPLIT_BUCKETS:
        m = _situation_metrics(key, bucket, playoffs=playoffs)
        if m.empty:
            continue
        r = m[m["player_id"] == int(pid)]
        if not len(r):
            continue
        r = r.iloc[0]
        if not (r["sit_TOI"] > 0):
            continue
        rows.append({
            "Situation": bucket, "TOI": r["sit_TOI"], "TOI/GP": r["Sit TOI/GP"],
            "CF%": r["Sit CF%"], "xGF%": r["Sit xGF%"],
            "xGF/60": r["Sit xGF/60"], "xGA/60": r["Sit xGA/60"],
            "GF/60": r["Sit GF/60"], "iG": r["Sit iG"], "ixG": r["Sit ixG"],
        })
    return pd.DataFrame(rows)


# PDO shot scope — same 5v5-vs-all-situations pattern as the Goalies sQS%
# toggle (QG_SCOPE_SUFFIX). Built together in one pass by build_pdo_sog.py.
PDO_SCOPE_FILE = {"5v5": "player_pdo_5v5_per_season.csv",
                   "All situations": "player_pdo_allsit_per_season.csv"}


def _pdo_toggle_state() -> str:
    """Shared PDO shot-scope toggle state (set by the widget in
    render_players, read here so it applies wherever PDO is computed this
    run — same shared-session-state pattern as _qg_toggle_state)."""
    return st.session_state.get("players_pdo_scope", "5v5")


@st.cache_data(show_spinner=False, ttl=3600)
def load_pdo_counts(scope_label: str = "5v5") -> pd.DataFrame:
    """Per-(player_id, season) SOG/goals building blocks for PDO — raw
    shot-events computation (fresh on-ice attribution pass reusing the
    validated 03_onice_attribution_pillars.py logic; see
    NFI/scripts/build_pdo_sog.py). Descriptive only — not a canonical NFI
    metric, no rel/QG. scope_label: '5v5' or 'All situations'."""
    fn = PDO_SCOPE_FILE.get(scope_label, PDO_SCOPE_FILE["5v5"])
    fp = REPO_ROOT / "NFI" / "output" / fn
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_csv(fp, dtype={"season": str})


@st.cache_data(show_spinner=False, ttl=3600)
def load_pdo_counts_playoffs(scope_label: str = "5v5") -> pd.DataFrame:
    """Playoff counterpart of load_pdo_counts — already pooled across all
    playoff games (see NFI/scripts/build_pdo_sog_playoffs.py), one row per
    qualifying player, so no further season aggregation is needed."""
    fn = ("player_pdo_5v5_playoffs.csv" if scope_label == "5v5"
          else "player_pdo_allsit_playoffs.csv")
    fp = REPO_ROOT / "NFI" / "output" / fn
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_csv(fp)


def _pdo_rate_playoffs() -> pd.DataFrame:
    """PDO/PDOxG for the pooled all-playoffs scope — the file is already one
    row per qualifying player (>=200 min 5v5 TOI pooled across every playoff
    game), so this just picks the columns and renames to display names."""
    g = load_pdo_counts_playoffs(_pdo_toggle_state())
    if g.empty:
        return pd.DataFrame()
    return g[["player_id", "pdo", "pdoxg"]].rename(columns={"pdo": "PDO", "pdoxg": "PDOxG"})


def _pdo_rate(scope_key: str) -> pd.DataFrame:
    """PDO for one scope, by ratio-of-sums on SOG/goals — same pooling
    convention as _as_rates. Regular season only (no playoff PDO computed).
    Raw column — no rel, no QG. Shot scope (5v5/all situations) follows the
    shared toggle in _pdo_toggle_state()."""
    g = load_pdo_counts(_pdo_toggle_state())
    if g.empty:
        return pd.DataFrame()
    if scope_key == "pooled":
        sub = g[g["season"].isin(POOLED_SEASONS)]
    elif scope_key == "pooled_2yr":
        sub = g[g["season"].isin(POOLED_2YR_SEASONS)]
    else:
        sub = g[g["season"] == scope_key]
    if sub.empty:
        return pd.DataFrame()
    _agg_kw = dict(
        sog_for=("sog_for", "sum"), goals_for=("goals_for", "sum"),
        sog_against=("sog_against", "sum"), goals_against=("goals_against", "sum"),
    )
    # xgf/xga building blocks exist only if build_pdo_sog.py was rerun with the
    # xG merge — pool them too when present so PDOxG uses the same ratio-of-sums.
    _has_xg = {"xgf", "xga"}.issubset(g.columns)
    if _has_xg:
        _agg_kw.update(xgf=("xgf", "sum"), xga=("xga", "sum"))
    agg = sub.groupby("player_id").agg(**_agg_kw).reset_index()
    ok = (agg["sog_for"] > 0) & (agg["sog_against"] > 0)
    sh = np.where(ok, agg["goals_for"] / agg["sog_for"], np.nan)
    sv = np.where(ok, 1 - agg["goals_against"] / agg["sog_against"], np.nan)
    agg["PDO"] = np.where(ok, (sh + sv) * 100, np.nan)
    out_cols = ["player_id", "PDO"]
    if _has_xg:
        # PDOxG: SH% & SV% net of expected (xG), 0-centered. Positive = finishing
        # / goaltending luck beyond what shot quality predicts.
        x_sh = np.where(ok, agg["xgf"] / agg["sog_for"], np.nan)
        x_sv = np.where(ok, 1 - agg["xga"] / agg["sog_against"], np.nan)
        agg["PDOxG"] = np.where(ok, ((sh - x_sh) + (sv - x_sv)) * 100, np.nan)
        out_cols.append("PDOxG")
    return agg[out_cols]


@st.cache_data(show_spinner=False, ttl=3600)
def load_edge_player_season() -> pd.DataFrame:
    """NHL EDGE per-player-season tracking stats, regular season only —
    descriptive, source-separate from my zone metrics. See edge/README.md
    for scrape methodology and the basis-mismatch caveat vs TZI/NZI/DZI/OZI."""
    fp = EDGE_DIR / "edge_skater_stats.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, dtype={"season": str})
    df["player_id"] = df["player_id"].astype("Int64")
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_edge_player_playoffs() -> pd.DataFrame:
    """NHL EDGE per-player-PLAYOFF-season tracking stats — same schema as
    load_edge_player_season, source edge_skater_stats_playoffs.csv (pulled by
    the same edge/scripts/pull_edge_stats.py, game_type=3/playoffs)."""
    fp = EDGE_DIR / "edge_skater_stats_playoffs.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, dtype={"season": str})
    df["player_id"] = df["player_id"].astype("Int64")
    return df


_EDGE_COLS = ["oz_time_pct", "oz_time_pct_percentile",
              "nz_time_pct", "nz_time_pct_percentile",
              "dz_time_pct", "dz_time_pct_percentile",
              "top_skating_speed_mph", "top_skating_speed_percentile",
              "speed_bursts_over_20mph", "speed_bursts_over_20mph_percentile",
              "distance_skated_miles", "distance_skated_percentile"]

# Raw EDGE column -> display name. Shared by the Player List leaderboard and
# the per-player drill-in trend, so both stay in sync.
_EDGE_REN = {
    "oz_time_pct": "EDGE OZ%", "oz_time_pct_percentile": "EDGE OZ %ile",
    "nz_time_pct": "EDGE NZ%", "nz_time_pct_percentile": "EDGE NZ %ile",
    "dz_time_pct": "EDGE DZ%", "dz_time_pct_percentile": "EDGE DZ %ile",
    "top_skating_speed_mph": "EDGE Top Speed", "top_skating_speed_percentile": "EDGE Speed %ile",
    "speed_bursts_over_20mph": "EDGE Bursts 20+",
    "speed_bursts_over_20mph_percentile": "EDGE Bursts %ile",
    "distance_skated_miles": "EDGE Distance (mi)", "distance_skated_percentile": "EDGE Distance %ile",
    "edge_distance_per60": "EDGE Distance/60",
    "edge_bursts_per60": "EDGE Bursts/60",
}
# Value-only columns (excludes the raw NHL percentile columns) — displayed with
# a computed (league / team) rank bracket instead, same convention as every
# other ranked column in this table.
_EDGE_VALUE_RAW = [c for c in _EDGE_COLS if "percentile" not in c]
# EZI leads the EDGE family (the headline "did they earn their O-zone time"
# metric), then the raw NHL tracking columns, then the per-60 rates. Speed
# bursts and distance are ALL-SITUATIONS totals; both now divide by the
# all-situations TOI (Data/player_situation_onice.csv, summed over every
# situation) — the correct matching denominator (the old ES-only toi_min
# inflated special-teams players). NHL's raw season burst TOTAL is kept too.
_EDGE_VALUE_DISP = (["EZI"] + [_EDGE_REN[c] for c in _EDGE_VALUE_RAW]
                    + ["EDGE Bursts/60", "EDGE Distance/60"])

# EDGE OZ%-scope toggle — the ONLY EDGE stat with an even-strength split from
# NHL is offensive-zone time; NZ%/DZ% have just the one (all-situations)
# number no matter what, since that's all the API publishes for those two.
_EDGE_OZ_SCOPE_COL = {"Even Strength": ("oz_time_pct_ev", "oz_time_pct_ev_percentile"),
                      "All Situations": ("oz_time_pct", "oz_time_pct_percentile")}


def _edge_toggle_state() -> str:
    """Shared EDGE OZ% scope toggle state (set by the widget in
    render_players, read here so it applies wherever EDGE is computed this
    run — same shared-session-state pattern as _pdo_toggle_state). Defaults to
    Even Strength to match the rest of the page (OZI/DZI/NZI/TZI and OZ Start%
    are all strict 5v5) — and, importantly, so EZI compares EDGE OZ time% to
    OZ Start% on the SAME even-strength basis rather than letting power-play
    O-zone time leak into the residual."""
    return st.session_state.get("players_edge_scope", "Even Strength")


def _edge_rate(scope_key: str) -> pd.DataFrame:
    """EDGE tracking columns for one scope. Single seasons return that
    season's row as-is (value + NHL's own percentile). Pooled/2yr views
    return a games_played-weighted average across the player's available
    EDGE seasons, INCLUDING the percentile columns — NHL doesn't expose
    enough to recompute a true multi-season percentile, so a pooled
    percentile here is a descriptive approximation, not NHL's own number.
    Regular season only (matches the scrape; no playoff EDGE wiring here)."""
    g = load_edge_player_season()
    if g.empty:
        return pd.DataFrame()
    g = g.copy()
    _oz_col, _oz_pct_col = _EDGE_OZ_SCOPE_COL[_edge_toggle_state()]
    g["oz_time_pct"] = g[_oz_col]
    g["oz_time_pct_percentile"] = g[_oz_pct_col]
    if scope_key == "pooled":
        sub = g[g["season"].isin(POOLED_SEASONS)].copy()
    elif scope_key == "pooled_2yr":
        sub = g[g["season"].isin(POOLED_2YR_SEASONS)].copy()
    else:
        sub = g[g["season"] == scope_key].copy()
    if sub.empty:
        return pd.DataFrame()
    if scope_key not in ("pooled", "pooled_2yr"):
        return sub[["player_id"] + _EDGE_COLS].drop_duplicates("player_id")
    sub["_w"] = sub["games_played"].clip(lower=1)
    for c in _EDGE_COLS:
        sub[f"_wx_{c}"] = sub[c] * sub["_w"]
    agg = sub.groupby("player_id").agg(
        **{f"_wsum_{c}": (f"_wx_{c}", "sum") for c in _EDGE_COLS},
        _wtot=("_w", "sum"),
    ).reset_index()
    for c in _EDGE_COLS:
        agg[c] = np.where(agg["_wtot"] > 0, agg[f"_wsum_{c}"] / agg["_wtot"], np.nan)
    return agg[["player_id"] + _EDGE_COLS]


def _edge_rate_playoffs() -> pd.DataFrame:
    """EDGE tracking columns pooled across ALL playoff seasons a player
    appears in — games-played-weighted average, same construction as
    _edge_rate('pooled') but sourced from load_edge_player_playoffs()."""
    g = load_edge_player_playoffs()
    if g.empty:
        return pd.DataFrame()
    g = g.copy()
    _oz_col, _oz_pct_col = _EDGE_OZ_SCOPE_COL[_edge_toggle_state()]
    g["oz_time_pct"] = g[_oz_col]
    g["oz_time_pct_percentile"] = g[_oz_pct_col]
    g["_w"] = g["games_played"].clip(lower=1)
    for c in _EDGE_COLS:
        g[f"_wx_{c}"] = g[c] * g["_w"]
    agg = g.groupby("player_id").agg(
        **{f"_wsum_{c}": (f"_wx_{c}", "sum") for c in _EDGE_COLS},
        _wtot=("_w", "sum"),
    ).reset_index()
    for c in _EDGE_COLS:
        agg[c] = np.where(agg["_wtot"] > 0, agg[f"_wsum_{c}"] / agg["_wtot"], np.nan)
    # Raw distance total, summed across playoff seasons — under a DISTINCT
    # name so it doesn't collide with the games-weighted-average
    # "distance_skated_miles" column above (that's the display total, matching
    # regular-season _edge_rate's convention; this _sum column is only for the
    # ratio-of-sum per-60 rate below). Bursts stay as the games-weighted-average
    # raw total in `agg` above — no per-60 rate (see _EDGE_VALUE_DISP comment).
    raw = g.groupby("player_id").agg(
        _dist_sum=("distance_skated_miles", "sum")).reset_index()
    return agg[["player_id"] + _EDGE_COLS].merge(raw, on="player_id", how="left")


def _edge_distance_rate(scope_key: str) -> pd.DataFrame:
    """EDGE distance skated, normalized to a per-60-minutes rate. Computed as a
    ratio-of-sums (sum distance, sum toi_min across the scope's seasons, then
    divide once) rather than games-weighted-averaging the way _edge_rate
    pools its other columns — distance_skated_miles is a season TOTAL, and
    toi_min elsewhere in this app is a season-summed cumulative figure for
    pooled scopes, so averaging the numerator while the denominator stays
    summed would silently understate the pooled rate by roughly 1/N seasons.
    Both EDGE distance and the TOI denominator are now ALL-SITUATIONS
    (Data/player_situation_onice.csv, summed over every situation), fixing the
    old ES-only-TOI mismatch that inflated special-teams players' rates."""
    edge = load_edge_player_season()
    if edge.empty:
        return pd.DataFrame()
    e = edge[["player_id", "season", "distance_skated_miles"]].dropna()
    if scope_key == "pooled":
        e = e[e["season"].isin(POOLED_SEASONS)]
    elif scope_key == "pooled_2yr":
        e = e[e["season"].isin(POOLED_2YR_SEASONS)]
    else:
        e = e[e["season"] == scope_key]
    if e.empty:
        return pd.DataFrame()
    dist = e.groupby("player_id")["distance_skated_miles"].sum().reset_index()
    toi = _allsit_toi(scope_key)
    if toi.empty:
        return pd.DataFrame()
    g = dist.merge(toi, on="player_id", how="inner")
    ok = g["allsit_toi_min"] > 0
    g["edge_distance_per60"] = np.where(
        ok, g["distance_skated_miles"] / g["allsit_toi_min"] * 60.0, np.nan)
    return g[["player_id", "edge_distance_per60"]]


def _edge_bursts_rate(scope_key: str) -> pd.DataFrame:
    """EDGE 20+ mph speed bursts per 60, all-situations (bursts are an
    all-situations season total; denominator is all-situations TOI). Ratio-of-
    sums across the scope's seasons — restores the per-60 burst rate that was
    reverted when only ES-only TOI was available."""
    edge = load_edge_player_season()
    if edge.empty:
        return pd.DataFrame()
    e = edge[["player_id", "season", "speed_bursts_over_20mph"]].dropna()
    if scope_key == "pooled":
        e = e[e["season"].isin(POOLED_SEASONS)]
    elif scope_key == "pooled_2yr":
        e = e[e["season"].isin(POOLED_2YR_SEASONS)]
    else:
        e = e[e["season"] == scope_key]
    if e.empty:
        return pd.DataFrame()
    b = e.groupby("player_id")["speed_bursts_over_20mph"].sum().reset_index()
    toi = _allsit_toi(scope_key)
    if toi.empty:
        return pd.DataFrame()
    g = b.merge(toi, on="player_id", how="inner")
    ok = g["allsit_toi_min"] > 0
    g["edge_bursts_per60"] = np.where(
        ok, g["speed_bursts_over_20mph"] / g["allsit_toi_min"] * 60.0, np.nan)
    return g[["player_id", "edge_bursts_per60"]]


_EZI_MIN_TOI = 200.0   # stability floor on the position-group AVERAGE only


def _add_ezi(base: pd.DataFrame) -> pd.DataFrame:
    """EZI (EDGE Zone Impact) — a 0-100 index (50 = position-group average)
    measuring how much MORE (or less) EDGE offensive-zone TIME a player earns
    than their O-zone faceoff STARTS predict, ADJUSTED for how weakly starts
    actually predict time.

    A naive raw = OZ time% − OZ Start% implicitly assumes starts predict time
    1-for-1. They don't: OZ Start% has huge deployment-driven spread (std≈6.7,
    range 8–53) while OZ time% barely moves (std≈2.4, range 35–50.5) —
    possession/tempo dominates over the starting whistle far more than
    deployment does. An OLS fit of OZ time% on OZ Start% (position-group-
    specific, verified 2026-07) gives a slope of only ≈0.23-0.25: a 40-point
    gap in starts predicts only a ~9-10-point gap in time. The naive
    subtraction therefore over-penalizes high-Start% players and over-rewards
    low-Start% players by roughly 4x — confirmed empirically: naive-diff
    correlated −0.94 with OZ Start% itself, meaning it was mostly just an
    inverted deployment metric, not a real skill residual (the leaderboard
    was dominated by low-event defensive players; zero offensive stars).

    Fix: raw = OZ time% − (intercept + slope × OZ Start%), i.e. the actual
    regression RESIDUAL, fit separately per position group (least squares,
    OZ Start% as the sole predictor). By construction this residual is
    exactly uncorrelated with OZ Start% (verified: 0.0000), so it isolates
    "time beyond what your own starts predict" rather than restating
    deployment. Positive = converts O-zone time beyond what the starts alone
    would suggest; negative = starts aren't converting into time.

    Recentred exactly like OZI/DZI/NZI/TZI: EZI = clip(50 + (raw − position-
    group average raw), 0, 100) — and since OLS residuals average to 0 within
    each fitted group by construction, this recentring step is a no-op on
    the population used to fit the regression; it only matters for making
    the final scale consistent with the other Zone Impact metrics. Requires
    "oz_time_pct" (raw EDGE fraction, pre-rename), "OZ Start%", "position",
    "toi_min" (>=200 ES-min floor on which rows fit the regression, for
    stability); no-ops if the inputs are missing (e.g. playoffs, which has
    no zone-start data)."""
    if not {"oz_time_pct", "OZ Start%", "position", "toi_min"}.issubset(base.columns):
        return base
    base = base.copy()
    oz_time = base["oz_time_pct"] * 100.0
    oz_start = base["OZ Start%"]
    pos_group = np.where(base["position"] == "D", "D", "F")
    floor_ok = base["toi_min"].fillna(0) >= _EZI_MIN_TOI
    ezi = pd.Series(np.nan, index=base.index)
    for grp in ("F", "D"):
        m = (pos_group == grp) & oz_time.notna() & oz_start.notna()
        if not m.any():
            continue
        fit_m = m & floor_ok
        if fit_m.sum() < 10:   # too few points to fit a stable regression
            fit_m = m
        slope, intercept = np.polyfit(oz_start[fit_m], oz_time[fit_m], 1)
        raw = oz_time[m] - (intercept + slope * oz_start[m])   # indexed like m's True rows
        # raw already averages ~0 over the fitting population by OLS
        # construction, but recentre over the SAME floor-qualified subset
        # (not the full group) so the displayed scale matches every other
        # Zone Impact metric's convention exactly.
        grp_avg = raw[fit_m[m]].mean()
        if pd.isna(grp_avg):
            grp_avg = raw.mean()
        ezi.loc[m] = (50.0 + (raw - grp_avg)).clip(0, 100)
    base["EZI"] = ezi.round(1)
    return base


@st.cache_data(show_spinner=False, ttl=3600)
def _load_xg_game(playoffs: bool = False) -> pd.DataFrame:
    """Per-game on-ice xG For/Against + on-ice TOI, from the MoneyPuck-derived
    per_player_game file (regular or playoffs)."""
    fn = "per_player_game_playoffs.csv" if playoffs else "per_player_game.csv"
    fp = REPO_ROOT / "Quality_Games" / "output" / fn
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, usecols=["player_id", "season", "xG_for", "xG_ag", "TOI_on_sec"])
    df["season"] = df["season"].astype(str)
    return df


def _xg_onice_rates(scope_key: str, playoffs: bool = False) -> pd.DataFrame:
    """On-ice xGF/60 and xGA/60 per player for one scope (MoneyPuck style), by
    ratio-of-sums: sum xG-for, xG-against and on-ice TOI across the scope's seasons,
    then rate = xG / TOI * 3600. scope_key: 'pooled', 'pooled_2yr', or a season."""
    df = _load_xg_game(playoffs)
    if df.empty:
        return pd.DataFrame()
    if playoffs:
        sub = df
    elif scope_key == "pooled":
        sub = df[df["season"].isin([str(s) for s in POOLED_SEASONS])]
    elif scope_key == "pooled_2yr":
        sub = df[df["season"].isin([str(s) for s in POOLED_2YR_SEASONS])]
    else:
        sub = df[df["season"] == str(scope_key)]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("player_id").agg(_xgf=("xG_for", "sum"), _xga=("xG_ag", "sum"),
                                     _toi=("TOI_on_sec", "sum")).reset_index()
    ok = g["_toi"] > 0
    g["xGF/60"] = np.where(ok, g["_xgf"] / g["_toi"] * 3600.0, np.nan)
    g["xGA/60"] = np.where(ok, g["_xga"] / g["_toi"] * 3600.0, np.nan)
    # Raw on-ice xG share (xGF% = xGF / (xGF + xGA)) — NOT RelxG%, which is
    # relative to teammates. This is the plain Corsi%-style share, same
    # pattern as NFI% alongside RelNFI%.
    _tot = g["_xgf"] + g["_xga"]
    g["xG%"] = np.where(_tot > 0, g["_xgf"] / _tot * 100.0, np.nan)
    return g[["player_id", "xGF/60", "xGA/60", "xG%"]]


# Quality-Games For/Against split (built by 03_quality_game_for_against.py): the
# share of a player's games where OFFENSE (For) or DEFENSE (Against) was quality,
# for xG and NFI. Display names → source count/qual_GP column stems.
_QG_FA = {"xG-QG-F%": "xG_QG_F", "xG-QG-A%": "xG_QG_A",
          "NFI-QG-A%": "NFI_QG_F", "NFI-QG-S%": "NFI_QG_A",
          # Relative (on/off, team-without-player) counterparts of the four
          # For/Against splits — same 0-1 Quality-Games scale as the raw splits.
          "RelxG-QG-F%": "RelxG_QG_F", "RelxG-QG-A%": "RelxG_QG_A",
          "RelNFI-QG-A%": "RelNFI_QG_F", "RelNFI-QG-S%": "RelNFI_QG_A"}


@st.cache_data(show_spinner=False, ttl=3600)
def load_qg_fa_player(playoffs: bool = False) -> pd.DataFrame:
    fn = "per_player_qg_fa_playoffs.csv" if playoffs else "per_player_qg_fa.csv"
    fp = REPO_ROOT / "Quality_Games" / "output" / fn
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_qg_fa_team(playoffs: bool = False) -> pd.DataFrame:
    fn = "per_team_qg_fa_playoffs.csv" if playoffs else "per_team_qg_fa.csv"
    fp = REPO_ROOT / "Quality_Games" / "output" / fn
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    df["team_abbrev"] = df["team_abbrev"].replace({"ARI": "UTA"})
    return df


def _qg_fa_rates(scope_key: str, playoffs: bool = False) -> pd.DataFrame:
    """Player QG For/Against rates (0-1) for a scope, pooled by ratio-of-sums
    (sum counts / sum qual_GP across the scope's seasons). Columns are already
    display-named (xG-QG-F%, xG-QG-A%, NFI-QG-A%, NFI-QG-S%)."""
    df = load_qg_fa_player(playoffs)
    if df.empty:
        return pd.DataFrame()
    if playoffs:
        sub = df[df["season"] == PLAYOFF_SCOPE] if (df["season"] == PLAYOFF_SCOPE).any() else df
    elif scope_key == "pooled":
        sub = df[df["season"].isin([str(s) for s in POOLED_SEASONS])]
    elif scope_key == "pooled_2yr":
        sub = df[df["season"].isin([str(s) for s in POOLED_2YR_SEASONS])]
    else:
        sub = df[df["season"] == str(scope_key)]
    if sub.empty:
        return pd.DataFrame()
    agg = {}
    for base in _QG_FA.values():
        agg[f"{base}_count"] = (f"{base}_count", "sum")
        agg[f"{base}_qual_GP"] = (f"{base}_qual_GP", "sum")
    g = sub.groupby("player_id").agg(**agg).reset_index()
    out = g[["player_id"]].copy()
    for disp, base in _QG_FA.items():
        q = g[f"{base}_qual_GP"]
        out[disp] = np.where(q > 0, g[f"{base}_count"] / q, np.nan)
    return out


_TEAM_QG_FA = {"team_xG_QG_F_pct": "xG-QG-F%", "team_xG_QG_A_pct": "xG-QG-A%",
               "team_NFI_QG_F_pct": "NFI-QG-A%", "team_NFI_QG_A_pct": "NFI-QG-S%"}


def _team_qg_fa(scope_key: str) -> pd.DataFrame:
    """Team QG For/Against rates (0-1) for a scope, TOI-weighted across the scope's
    seasons (mirrors render_teams' xG-QG%/NFI-QG% pooling). Keyed on 'team'."""
    df = load_qg_fa_team()
    if df.empty:
        return pd.DataFrame()
    if scope_key == "pooled":
        sub = df[df["season"].isin([str(s) for s in POOLED_SEASONS])]
    elif scope_key == "pooled_2yr":
        sub = df[df["season"].isin([str(s) for s in POOLED_2YR_SEASONS])]
    else:
        sub = df[df["season"] == str(scope_key)]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for t, g in sub.groupby("team_abbrev"):
        w = pd.to_numeric(g["total_team_TOI_min"], errors="coerce").astype(float)
        row = {"team": t}
        for src, disp in _TEAM_QG_FA.items():
            v = pd.to_numeric(g[src], errors="coerce")
            m = v.notna() & (w > 0)
            row[disp] = float(np.average(v[m], weights=w[m])) if m.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --- In-tab profiles (Gate F): per-season trends ----------------------------
PROFILE_SEASONS = ["20222023", "20232024", "20242025", "20252026"]  # 4yr, excl 2021-22
SEASON_DISPLAY = {"20222023": "2022-23", "20232024": "2023-24",
                  "20242025": "2024-25", "20252026": "2025-26"}


@st.cache_data(show_spinner=False, ttl=3600)
def load_zone_per_season(season: str | None = None) -> pd.DataFrame:
    """Per-season OZI/DZI/NZI/TZI (0–100 index, 50 = position-group average),
    name-keyed on (player_name, _pos_group). Each season's index is recentred on
    that season's own per-position average, so 50 always means league-average.

    season=None → long multi-season frame (season, player_name, _pos_group,
    OZI, DZI, NZI, TZI) used by the per-player trend. season="20252026" → just
    that season's rows keyed on (player_name, _pos_group), season column dropped,
    for a single-season leaderboard join (mirrors load_zone_pooled's shape)."""
    sub = ADJ / "zone_index100"
    out = []
    seasons = [season] if season is not None else PROFILE_SEASONS
    for ssn in seasons:
        label = SEASON_DISPLAY.get(ssn, ssn)   # files are named "2024-25" etc.
        for pos_file, grp in (("forwards", "F"), ("defense", "D")):
            fp = sub / f"{label}_{pos_file}.csv"
            if not fp.exists():
                continue
            d = pd.read_csv(fp)
            keep = [c for c in ("player_name", "OZI", "DZI", "NZI", "TZI")
                    if c in d.columns]
            d = d[keep].copy()
            d = d.drop_duplicates("player_name", keep="first")
            d["season"] = ssn
            d["_pos_group"] = grp
            out.append(d)
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
    p = p[p["season"].isin(PROFILE_SEASONS)].copy()
    # Team per season in chronological play order ("EDM / COL" for traded players),
    # falling back to the single NFI team if the order map lacks that season.
    _order = load_player_season_team_order()

    def _team_disp(r):
        teams = _order.get((pid, r["season"]))
        if teams:
            return " / ".join(teams)
        return r["team"] if isinstance(r["team"], str) and r["team"] else None

    p["Team"] = p.apply(_team_disp, axis=1)
    trend = p[["season", "Team", "NFI_pct", "RelNFI_pct", "RelNFI_F_pct", "RelNFI_A_pct"]].rename(
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
            zcols = ["season"] + [c for c in ("OZI", "DZI", "NZI", "TZI")
                                  if c in zz.columns]
            trend = trend.merge(zz[zcols], on="season", how="outer")

    # Quality Games per season.
    qg = load_qg_player_season()
    if not qg.empty:
        q = qg[qg["player_id"] == pid].copy()
        q["season"] = q["season"].astype(str)
        q = q[q["season"].isin(PROFILE_SEASONS)]
        keep = ["season"] + [c for c in ("GP", "NFI_QG_pct", "xG_QG_pct",
                "RelNFI_QG_pct", "RelxG_QG_pct", "RelxG_pct",
                "RelxG_F_pct", "RelxG_A_pct") if c in q.columns]
        q = q[keep].rename(columns={"NFI_QG_pct": "NFI-QG%", "xG_QG_pct": "xG-QG%",
                "RelNFI_QG_pct": "RelNFI-QG%", "RelxG_QG_pct": "RelxG-QG%",
                "RelxG_pct": "RelxG%", "RelxG_F_pct": "RelxG-F%",
                "RelxG_A_pct": "RelxG-A%"})
        trend = trend.merge(q, on="season", how="outer")

    # Quality-Games For/Against per season (xG-QG-F/A%, NFI-QG-F/A%).
    qgfa = load_qg_fa_player()
    if not qgfa.empty:
        qf = qgfa[qgfa["player_id"] == pid].copy()
        qf["season"] = qf["season"].astype(str)
        qf = qf[qf["season"].isin(PROFILE_SEASONS)]
        _fa_ren = {f"{b}_pct": disp for disp, b in _QG_FA.items()}
        _fk = ["season"] + [c for c in _fa_ren if c in qf.columns]
        if len(_fk) > 1:
            trend = trend.merge(qf[_fk].rename(columns=_fa_ren), on="season", how="outer")

    # On-ice xGF/60 and xGA/60 (MoneyPuck-style) per season.
    xgame = _load_xg_game()
    if not xgame.empty:
        xa = xgame[(xgame["player_id"] == pid)
                   & (xgame["season"].isin(PROFILE_SEASONS))].copy()
        if not xa.empty:
            xg = xa.groupby("season").agg(_xgf=("xG_for", "sum"),
                                          _xga=("xG_ag", "sum"),
                                          _toi=("TOI_on_sec", "sum")).reset_index()
            _ok = xg["_toi"] > 0
            xg["xGF/60"] = np.where(_ok, xg["_xgf"] / xg["_toi"] * 3600.0, np.nan)
            xg["xGA/60"] = np.where(_ok, xg["_xga"] / xg["_toi"] * 3600.0, np.nan)
            _tot = xg["_xgf"] + xg["_xga"]
            xg["xG%"] = np.where(_tot > 0, xg["_xgf"] / _tot * 100.0, np.nan)
            trend = trend.merge(xg[["season", "xGF/60", "xGA/60", "xG%"]],
                                on="season", how="outer")

    # PDO (SOG-based; scope follows the shared 5v5/all-situations toggle) per season.
    pdo_counts = load_pdo_counts(_pdo_toggle_state())
    if not pdo_counts.empty:
        pc = pdo_counts[(pdo_counts["player_id"] == pid)
                         & (pdo_counts["season"].isin(PROFILE_SEASONS))].copy()
        if not pc.empty:
            _pcols = ["season", "pdo"] + (["pdoxg"] if "pdoxg" in pc.columns else [])
            trend = trend.merge(
                pc[_pcols].rename(columns={"pdo": "PDO", "pdoxg": "PDOxG"}),
                on="season", how="outer")

    # NHL EDGE tracking per season (regular season only; see edge/README.md).
    edge_season = load_edge_player_season()
    if not edge_season.empty:
        ea = edge_season[(edge_season["player_id"] == pid)
                          & (edge_season["season"].isin(PROFILE_SEASONS))].copy()
        if not ea.empty:
            _oz_col, _oz_pct_col = _EDGE_OZ_SCOPE_COL[_edge_toggle_state()]
            ea["oz_time_pct"] = ea[_oz_col]
            ea["oz_time_pct_percentile"] = ea[_oz_pct_col]
            # Per-60 EDGE distance rate BEFORE the rename below — same ratio
            # (this player's own season rows) that the league-wide
            # _edge_distance_rate function uses, so the per-season and pooled
            # 2yr-avg rows land on an identical basis. Bursts stay as NHL's
            # raw season total (see _EDGE_VALUE_DISP comment) — no rate here.
            _rate_src = ea[["season", "distance_skated_miles"]].merge(
                p[["season", "toi_min"]], on="season", how="left")
            _ok_toi = _rate_src["toi_min"] > 0
            _rate_src["EDGE Distance/60"] = np.where(
                _ok_toi, _rate_src["distance_skated_miles"] / _rate_src["toi_min"] * 60.0, np.nan)
            ea = ea[["season"] + _EDGE_VALUE_RAW].rename(columns=_EDGE_REN)
            trend = trend.merge(ea, on="season", how="outer")
            trend = trend.merge(
                _rate_src[["season", "EDGE Distance/60"]],
                on="season", how="outer")

    # D/N/O Start% — real per-season data (Zones/output/zone_start_per_season.csv).
    _zs = load_zone_start_per_season_raw()
    if not _zs.empty:
        _zsp = _zs[_zs["player_id"] == pid].copy()
        if not _zsp.empty:
            _tot = (_zsp["oz_faceoff_shifts"] + _zsp["dz_faceoff_shifts"]
                    + _zsp["nz_faceoff_shifts"])
            _okz = _tot > 0
            _zsp["OZ Start%"] = np.where(_okz, _zsp["oz_faceoff_shifts"] / _tot * 100, np.nan)
            _zsp["DZ Start%"] = np.where(_okz, _zsp["dz_faceoff_shifts"] / _tot * 100, np.nan)
            _zsp["NZ Start%"] = np.where(_okz, _zsp["nz_faceoff_shifts"] / _tot * 100, np.nan)
            trend = trend.merge(_zsp[["season", "OZ Start%", "DZ Start%", "NZ Start%"]],
                                on="season", how="outer")
    for _c in ("DZ Start%", "NZ Start%", "OZ Start%"):
        if _c not in trend.columns:
            trend[_c] = np.nan

    trend = trend[trend["season"].isin(PROFILE_SEASONS)].copy()
    # The raw per-60s (NFI-A/60, NFI-S/60) come from an UNfloored count file, so a
    # season below the NFI build's TOI floor would otherwise show them alone with
    # everything else blank. Blank them too when the season has no NFI data, so a
    # row is all-or-nothing rather than per-60-only.
    if "NFI%" in trend.columns:
        _no_nfi = trend["NFI%"].isna()
        for _c in ("NFI-A/60", "NFI-S/60"):
            if _c in trend.columns:
                trend.loc[_no_nfi, _c] = np.nan
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


def _player_season_ranks(pid: int, same_pos: bool = False, team=None) -> dict:
    """For one player, the per-season rank of each display metric (#1 = best;
    NFI-S/60 lowest = #1). Cohort is all skaters by default, or the player's own
    position group (F or D) when same_pos=True. If `team` is given, the cohort is
    further restricted to that team's roster each season (within-team rank).
    Returns {display_col: {season_str: rank}}."""
    pid = int(pid)
    out = {}
    nfi = load_nfi_player()
    # Player's position group + a player_id→F/D map for sources lacking position.
    pos_group, pos_map = None, {}
    if not nfi.empty:
        _n = nfi.dropna(subset=["player_id"]).copy()
        _pg = np.where(_n["position"].astype(str) == "D", "D", "F")
        pos_map = dict(zip(_n["player_id"].astype(int), _pg))
        pos_group = pos_map.get(pid)
    restrict = same_pos and pos_group is not None

    # Per-season team roster (player_ids + names) for within-team ranks. `team`
    # may be a specific abbrev, or "__own__" to use the player's OWN team(s) that
    # season (union of rosters across teams in a traded season).
    team_pids, team_names = {}, {}
    if team and not nfi.empty:
        _rosters = load_team_rosters()
        _nm = nfi.dropna(subset=["player_id"]).copy()
        _nm["season"] = _nm["season"].astype(str)
        _order = load_player_season_team_order() if team == "__own__" else {}
        _nfi_team = (dict(zip(zip(_nm["player_id"].astype(int), _nm["season"]), _nm["team"]))
                     if team == "__own__" else {})
        for ssn in PROFILE_SEASONS:
            if team == "__own__":
                _ts = _order.get((pid, ssn)) or []
                if not _ts:
                    _t = _nfi_team.get((pid, ssn))
                    _ts = [_t] if isinstance(_t, str) and _t else []
                pids = set().union(*[_rosters.get((ssn, t), set()) for t in _ts]) if _ts else set()
            else:
                pids = _rosters.get((ssn, team), set())
            team_pids[ssn] = pids
            team_names[ssn] = set(
                _nm[(_nm["season"] == ssn)
                    & (_nm["player_id"].astype(int).isin(pids))]["player_name"])

    def _byid(sub, ssn):  # restrict a player_id-keyed cohort to position / team
        if restrict:
            sub = sub[sub["player_id"].astype(int).map(pos_map) == pos_group]
        if team:
            sub = sub[sub["player_id"].astype(int).isin(team_pids.get(ssn, set()))]
        return sub

    if not nfi.empty:
        nfi = nfi.copy()
        nfi["season"] = nfi["season"].astype(str)
        for disp_c, src in (("NFI%", "NFI_pct"), ("RelNFI%", "RelNFI_pct"),
                            ("RelNFI-A%", "RelNFI_F_pct"), ("RelNFI-S%", "RelNFI_A_pct")):
            if src not in nfi.columns:
                continue
            d = {}
            for ssn in PROFILE_SEASONS:
                sub = _byid(nfi[nfi["season"] == ssn], ssn)
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
                sub = _byid(g[g["season"] == ssn], ssn)
                pv = sub.loc[sub["player_id"] == pid, disp_c]
                if len(pv):
                    d[ssn] = _league_rank(sub[disp_c], pv.iloc[0], lower=low)
            out[disp_c] = d
    z = load_zone_per_season()
    if not z.empty and pos_group is not None:
        prow = nfi[nfi["player_id"] == pid]
        if len(prow):
            name = prow["player_name"].iloc[0]
            for m in ("OZI", "DZI", "NZI", "TZI"):
                if m not in z.columns:
                    continue
                d = {}
                for ssn in PROFILE_SEASONS:
                    sub = z[z["season"] == ssn]  # all skaters that season
                    if restrict:
                        sub = sub[sub["_pos_group"] == pos_group]
                    if team:
                        sub = sub[sub["player_name"].isin(team_names.get(ssn, set()))]
                    pv = sub.loc[(sub["player_name"] == name)
                                 & (sub["_pos_group"] == pos_group), m]
                    if len(pv) and pd.notna(pv.iloc[0]):
                        d[ssn] = _league_rank(sub[m], pv.iloc[0])
                out[m] = d
    zs = load_zone_start_per_season_raw()
    if not zs.empty:
        zs = zs.copy()
        _tot = zs["oz_faceoff_shifts"] + zs["dz_faceoff_shifts"] + zs["nz_faceoff_shifts"]
        _okz = _tot > 0
        zs["OZ Start%"] = np.where(_okz, zs["oz_faceoff_shifts"] / _tot * 100, np.nan)
        zs["DZ Start%"] = np.where(_okz, zs["dz_faceoff_shifts"] / _tot * 100, np.nan)
        zs["NZ Start%"] = np.where(_okz, zs["nz_faceoff_shifts"] / _tot * 100, np.nan)
        for m in ("OZ Start%", "DZ Start%", "NZ Start%"):
            d = {}
            for ssn in PROFILE_SEASONS:
                sub = _byid(zs[zs["season"] == ssn], ssn)
                pv = sub.loc[sub["player_id"] == pid, m]
                if len(pv) and pd.notna(pv.iloc[0]):
                    d[ssn] = _league_rank(sub[m], pv.iloc[0])
            out[m] = d
    qg = load_qg_player_season()
    if not qg.empty:
        qg = qg.copy()
        qg["season"] = qg["season"].astype(str)
        for disp_c, src in (("NFI-QG%", "NFI_QG_pct"), ("xG-QG%", "xG_QG_pct"),
                            ("RelNFI-QG%", "RelNFI_QG_pct"), ("RelxG-QG%", "RelxG_QG_pct"),
                            ("RelxG%", "RelxG_pct")):
            if src not in qg.columns:
                continue
            d = {}
            for ssn in PROFILE_SEASONS:
                sub = _byid(qg[qg["season"] == ssn], ssn)
                pv = sub.loc[sub["player_id"] == pid, src]
                if len(pv) and pd.notna(pv.iloc[0]):
                    d[ssn] = _league_rank(sub[src], pv.iloc[0])
            out[disp_c] = d
    pdo = load_pdo_counts(_pdo_toggle_state())
    if not pdo.empty:
        pdo = pdo.copy()
        pdo["season"] = pdo["season"].astype(str)
        d = {}
        for ssn in PROFILE_SEASONS:
            sub = _byid(pdo[pdo["season"] == ssn], ssn)
            pv = sub.loc[sub["player_id"] == pid, "pdo"]
            if len(pv) and pd.notna(pv.iloc[0]):
                d[ssn] = _league_rank(sub["pdo"], pv.iloc[0])
        out["PDO"] = d
        if "pdoxg" in pdo.columns:
            dx = {}
            for ssn in PROFILE_SEASONS:
                sub = _byid(pdo[pdo["season"] == ssn], ssn)
                pv = sub.loc[sub["player_id"] == pid, "pdoxg"]
                if len(pv) and pd.notna(pv.iloc[0]):
                    dx[ssn] = _league_rank(sub["pdoxg"], pv.iloc[0])
            out["PDOxG"] = dx
    edge = load_edge_player_season()
    if not edge.empty:
        edge = edge.copy()
        edge["season"] = edge["season"].astype(str)
        _oz_col, _oz_pct_col = _EDGE_OZ_SCOPE_COL[_edge_toggle_state()]
        edge["oz_time_pct"] = edge[_oz_col]
        edge["oz_time_pct_percentile"] = edge[_oz_pct_col]
        for raw_c in _EDGE_VALUE_RAW:
            disp_c = _EDGE_REN[raw_c]
            if raw_c not in edge.columns:
                continue
            d = {}
            for ssn in PROFILE_SEASONS:
                sub = _byid(edge[edge["season"] == ssn], ssn)
                pv = sub.loc[sub["player_id"] == pid, raw_c]
                if len(pv) and pd.notna(pv.iloc[0]):
                    d[ssn] = _league_rank(sub[raw_c], pv.iloc[0])
            out[disp_c] = d
    return out


# Storage→display column map for the appended 2yr pooled row.
_P2YR_MAP = {"NFI%": "NFI_pct", "RelNFI%": "RelNFI_pct", "RelNFI-A%": "RelNFI_F_pct",
             "RelNFI-S%": "RelNFI_A_pct", "NFI-A/60": "NFI_A_rate", "NFI-S/60": "NFI_S_rate",
             "NZI": "NZI", "DZI": "DZI", "OZI": "OZI", "TZI": "TZI",
             "NFI-QG%": "NFI_QG_pct",
             "xG-QG%": "xG_QG_pct", "RelNFI-QG%": "RelNFI_QG_pct",
             "RelxG-QG%": "RelxG_QG_pct", "RelxG%": "RelxG_pct",
             # xG family + QG For/Against — the 2yr frame carries these under their
             # display names (or storage names for RelxG-F/A).
             "RelxG-F%": "RelxG_F_pct", "RelxG-A%": "RelxG_A_pct",
             "xGF/60": "xGF/60", "xGA/60": "xGA/60", "xG%": "xG%",
             "xG-QG-F%": "xG-QG-F%", "xG-QG-A%": "xG-QG-A%",
             "NFI-QG-A%": "NFI-QG-A%", "NFI-QG-S%": "NFI-QG-S%",
             # Relative QG For/Against splits (2yr frame carries display names).
             "RelxG-QG-F%": "RelxG-QG-F%", "RelxG-QG-A%": "RelxG-QG-A%",
             "RelNFI-QG-A%": "RelNFI-QG-A%", "RelNFI-QG-S%": "RelNFI-QG-S%",
             "PDO": "PDO", "PDOxG": "PDOxG",
             "DZ Start%": "DZ Start%", "NZ Start%": "NZ Start%", "OZ Start%": "OZ Start%",
             # EDGE — the 2yr frame carries these under their RAW column names
             # (the display rename only happens in render_players' own table).
             **{disp: raw for raw, disp in _EDGE_REN.items()}}


@st.cache_data(show_spinner=False, ttl=3600)
def _players_2yr_frame() -> pd.DataFrame:
    """The 2-year (2024-26) denominator-based pooled player frame — reused to
    append a '2yr avg' row to the detail/trade trend."""
    f, _ = _build_players_frame("2yr (2024–2026)")
    return f


def _player_profile_table(pid: int, same_pos: bool = False, families=None,
                          team=None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Rank-annotated per-season trend table for a player. Returns
    (display_df, trend, metric_cols): display_df has string cells (value + rank);
    trend is the numeric frame (for charts). Empty display_df if no data.
    same_pos ranks within the player's position group instead of all skaters.
    families: optional list of metric families to show (None/empty = all).
    team: when set, each cell also shows the within-team rank as (league / team)."""
    trend = _player_trend(pid)
    if trend.empty:
        return pd.DataFrame(), trend, []
    share_cols = ["RelNFI%", "RelNFI-A%", "RelNFI-S%", "NFI%"]
    rate_cols = ["NFI-A/60", "NFI-S/60"]
    zone_cols = ["DZ Start%", "NZ Start%", "OZ Start%", "OZI", "DZI", "NZI", "TZI"]
    # Grouped raw+rel so the table above the charts mirrors the paired bars:
    # each Quality-Games metric sits next to its relative (on/off) counterpart.
    qg_cols = ["NFI-QG%", "RelNFI-QG%", "NFI-QG-A%", "RelNFI-QG-A%",
               "NFI-QG-S%", "RelNFI-QG-S%",
               "xG-QG%", "RelxG-QG%", "xG-QG-F%", "RelxG-QG-F%",
               "xG-QG-A%", "RelxG-QG-A%"]
    xg_cols = ["xGF/60", "xGA/60", "xG%", "RelxG%", "RelxG-F%", "RelxG-A%", "PDO", "PDOxG"]
    edge_cols = _EDGE_VALUE_DISP
    metric_cols = [c for c in qg_cols + xg_cols + share_cols + rate_cols + zone_cols + edge_cols
                   if c in trend.columns]
    if families:   # narrow to the selected metric families (a column may belong
                   # to more than one family, e.g. D/N/O Start% under both Zone
                   # Impact and EDGE — show it if ANY selected family claims it)
        _fam_of = {}
        for fam, fcols in PLAYER_FAMILY_COLS.items():
            for col in fcols:
                _fam_of.setdefault(col, set()).add(fam)
        _wanted = set(families)
        metric_cols = [c for c in metric_cols if _fam_of.get(c) and _fam_of[c] & _wanted]

    # Per-season rank (cohort per same_pos) appended to each cell. When a team is
    # given, also compute the within-team rank → cells read "(league / team)".
    ranks = _player_season_ranks(pid, same_pos=same_pos)
    team_ranks = _player_season_ranks(pid, same_pos=same_pos, team=team) if team else {}
    _b = {}
    for c in ("NFI%", "NFI-QG%", "xG-QG%", "RelNFI-QG%", "RelxG-QG%",
              "xG-QG-F%", "xG-QG-A%", "NFI-QG-A%", "NFI-QG-S%",
              "RelxG-QG-F%", "RelxG-QG-A%", "RelNFI-QG-A%", "RelNFI-QG-S%"):
        _b[c] = lambda v: f"{v * 100:.1f}%"
    for c in ("RelNFI%", "RelNFI-A%", "RelNFI-S%", "RelxG%", "RelxG-F%", "RelxG-A%"):
        _b[c] = lambda v: f"{v:+.2f}"
    for c in ("NFI-A/60", "NFI-S/60", "OZI", "DZI", "NZI", "TZI"):
        _b[c] = lambda v: f"{v:.1f}"
    for c in ("DZ Start%", "NZ Start%", "OZ Start%"):
        _b[c] = lambda v: f"{v:.1f}%"
    for c in ("xGF/60", "xGA/60"):
        _b[c] = lambda v: f"{v:.2f}"
    _b["xG%"] = lambda v: f"{v:.1f}%"
    _b["PDO"] = lambda v: f"{v:.1f}"
    _b["PDOxG"] = lambda v: f"{v:+.1f}"
    for c in ("EDGE OZ%", "EDGE OZ% (EV)", "EDGE NZ%", "EDGE DZ%"):
        _b[c] = lambda v: f"{v * 100:.1f}%"
    _b["EDGE Top Speed"] = lambda v: f"{v:.1f} mph"
    _b["EDGE Bursts 20+"] = lambda v: f"{v:.0f}"
    _b["EDGE Distance (mi)"] = lambda v: f"{v:.1f} mi"
    _b["EDGE Distance/60"] = lambda v: f"{v:.2f} mi/60"
    has_team = "Team" in trend.columns
    has_gp = "GP" in trend.columns
    rows = []
    for _, r in trend.iterrows():
        ssn = r["season"]
        row = {"Season": r["Season"]}
        if has_team:
            row["Team"] = r["Team"] if pd.notna(r["Team"]) else "—"
        if has_gp:
            row["GP"] = f"{int(r['GP'])}" if pd.notna(r["GP"]) else "—"
        for c in metric_cols:
            v = r[c]
            if pd.isna(v):
                row[c] = "—"
            else:
                txt = _b.get(c, lambda v: f"{v}")(v)
                rk = ranks.get(c, {}).get(ssn)
                if rk is None:
                    # Below that metric's OWN qualifying floor that season (not a
                    # missing value) — mark it explicitly rather than showing a
                    # bare number with no indication it's unranked.
                    row[c] = f"{txt} (UR)"
                elif team:
                    trk = team_ranks.get(c, {}).get(ssn)
                    row[c] = f"{txt} ({rk} / {trk})" if trk is not None else f"{txt} ({rk})"
                else:
                    row[c] = f"{txt} ({rk})"
        rows.append(row)

    # Append a "2yr avg (24-26)" row — the denominator-based 2-season pool, with
    # the same (league / team) rank brackets, ranked within the 2yr cohort.
    _f2 = _players_2yr_frame()
    if not _f2.empty and (_f2["player_id"] == pid).any():
        _pr = _f2[_f2["player_id"] == pid].iloc[0]
        _pos = str(_pr.get("position"))
        _coh = (_f2[_f2["position"] == _pos] if same_pos
                else _f2[_f2["position"].isin(["F", "D"])])
        _t2 = _pr.get("team")
        _tcoh = (_coh[_coh["team"] == _t2] if (team and isinstance(_t2, str)) else None)
        _lower = {"NFI-S/60"}
        r2 = {"Season": "2yr avg (24-26)"}
        if has_team:
            r2["Team"] = _t2 if isinstance(_t2, str) and _t2 else "—"
        if has_gp:
            r2["GP"] = f"{int(_pr['GP'])}" if pd.notna(_pr.get("GP")) else "—"
        for c in metric_cols:
            sc = _P2YR_MAP.get(c)
            v = _pr.get(sc) if sc else None
            if sc is None or pd.isna(v):
                r2[c] = "—"
                continue
            txt = _b.get(c, lambda v: f"{v}")(v)

            def _rk(fr, _v=v, _c=c, _sc=sc):
                s = pd.to_numeric(fr[_sc], errors="coerce")
                better = (s < _v).sum() if _c in _lower else (s > _v).sum()
                return int(better) + 1
            lg = _rk(_coh)
            if team and _tcoh is not None and len(_tcoh):
                r2[c] = f"{txt} ({lg} / {_rk(_tcoh)})"
            else:
                r2[c] = f"{txt} ({lg})"
        rows.append(r2)

    lead = ["Season"] + (["Team"] if has_team else []) + (["GP"] if has_gp else [])
    return pd.DataFrame(rows, columns=lead + metric_cols), trend, metric_cols


_QG_BAR_METRICS = ["NFI-QG%", "NFI-QG-A%", "NFI-QG-S%", "RelNFI-QG%",
                   "RelNFI-QG-A%", "RelNFI-QG-S%",
                   "xG-QG%", "xG-QG-F%", "xG-QG-A%", "RelxG-QG%",
                   "RelxG-QG-F%", "RelxG-QG-A%"]

# The two paired-bar panels: (concept label, raw QG metric, relative QG metric).
# Each concept renders a Raw+Rel pair grouped together, spaced from the next
# concept. NFI panel and xG panel go side by side.
_QG_PAIR_PANELS = {
    "NFI": [("Attack", "NFI-QG-A%", "RelNFI-QG-A%"),
            ("Suppress", "NFI-QG-S%", "RelNFI-QG-S%"),
            ("Overall", "NFI-QG%", "RelNFI-QG%")],
    "xG": [("For", "xG-QG-F%", "RelxG-QG-F%"),
           ("Against", "xG-QG-A%", "RelxG-QG-A%"),
           ("Overall", "xG-QG%", "RelxG-QG%")],
}
# Flat Raw+Rel-paired metric order per family — used by the Trade Analyzer's
# faceted per-player bar compare so it carries the SAME 12 metrics (incl. the
# relative QG splits) and the same Attack/Suppress/Overall order as the
# single-player drill-in's paired panels.
_QG_BAR_ORDER_NFI = [m for _c, raw, rel in _QG_PAIR_PANELS["NFI"] for m in (raw, rel)]
_QG_BAR_ORDER_XG = [m for _c, raw, rel in _QG_PAIR_PANELS["xG"] for m in (raw, rel)]
_ZONE_BAR_METRICS_ORDER = ["OZI", "DZI", "NZI", "TZI", "EZI"]
_BRAND_DEEP = "#0A1A2F"          # "Hockey" — deeper than the chart navy
_BRAND_ROI = PALETTE["orange"]   # "ROI" — brand orange (#FF6B35)


_BRAND_URL = "hockeyroi.streamlit.app"


# Set by drill-in views so a downloaded chart carries whose data it is (rendered as
# an in-chart title so it's part of the browser's native Save-as-PNG). Reset to None
# on leaderboard views.
_CHART_TITLE = None


def _set_dl_title(name) -> None:
    global _CHART_TITLE
    _CHART_TITLE = str(name) if name else None


def _brand_row(width_px: int = None):
    """A standalone one-row chart carrying the two-colour HockeyROI wordmark + URL,
    right-aligned. vconcat'd beneath multi-panel (faceted/concat) charts so the brand
    is part of the Vega spec — and therefore part of the browser's native Save-as-PNG
    export — the same way single-panel charts get it from the in-plot _brand_layer.
    (Faceted charts can't embed the in-plot version: value-positioned marks don't
    resolve to a panel's coordinates.)"""
    import altair as alt
    W = int(width_px) if width_px else 720
    b = alt.Chart(pd.DataFrame([{"_": 0}]))
    _y_mark = alt.value(16)      # wordmark line
    _y_url = alt.value(31)       # URL line, just beneath

    def _word(text, xpx, align, color, size, yv):
        return b.mark_text(align=align, baseline="bottom", fontSize=size,
                           fontWeight="bold", color=color).encode(
            x=alt.value(xpx), y=yv, text=alt.value(text))

    # Literal pixel x (the row's width is known) — the "width" signal doesn't resolve
    # in a standalone value-only layer, so the wordmark is anchored to W directly.
    h = _word("Hockey", W - 27, "right", _BRAND_DEEP, 12, _y_mark)
    r = _word("ROI", W - 27, "left", _BRAND_ROI, 12, _y_mark)
    u = _word(_BRAND_URL, W, "right", "#7A8694", 10, _y_url)
    return alt.layer(h, r, u).properties(width=W, height=34)


def _brand_layer(lift: int = 6):
    """The HockeyROI footer drawn INSIDE the plot, bottom-right, just ABOVE the
    x-axis, as two stacked right-aligned lines: the two-colour 'HockeyROI' wordmark
    on top, the site URL beneath it. Staying within the plot bounds (y < height)
    means it never distorts the axes (Streamlit's 'fit' autosize only shrinks the
    plot for marks placed BELOW it) and is never clipped — and it's part of the
    Save-as-PNG image. Each glyph gets a white halo (a white-outlined copy drawn
    behind) so it stays legible over dark bars as well as on the white background.
    The wordmark's two words abut at a shared anchor (Hockey right-aligned, ROI
    left-aligned) so 'HockeyROI' has no gap whatever its width. lift: pixels the
    URL (bottom line) sits above the x-axis; the wordmark rides one line higher.
    (Faceted charts can't embed this — value-positioned marks don't resolve to a
    panel's coordinates — so they get the vconcat'd _brand_row instead.)"""
    import altair as alt
    b = alt.Chart(pd.DataFrame([{"_": 0}]))
    _y_url = alt.value(alt.ExprRef(f"height - {int(lift)}"))        # bottom line: URL
    _y_mark = alt.value(alt.ExprRef(f"height - {int(lift) + 14}"))  # line above: wordmark

    def _word(text, x_expr, align, color, size, yv):
        x = alt.value(alt.ExprRef(x_expr))
        halo = b.mark_text(align=align, baseline="bottom", fontSize=size, fontWeight="bold",
                           color="white", stroke="white", strokeWidth=3, opacity=0.9).encode(
            x=x, y=yv, text=alt.value(text))
        fg = b.mark_text(align=align, baseline="bottom", fontSize=size, fontWeight="bold",
                         color=color).encode(x=x, y=yv, text=alt.value(text))
        return halo, fg

    h_h, h_f = _word("Hockey", "width - 27", "right", _BRAND_DEEP, 12, _y_mark)
    r_h, r_f = _word("ROI", "width - 27", "left", _BRAND_ROI, 12, _y_mark)
    u_h, u_f = _word(_BRAND_URL, "width", "right", "#7A8694", 10, _y_url)  # under the mark
    # All halos first (behind), then the crisp coloured glyphs on top.
    return alt.layer(u_h, h_h, r_h, u_f, h_f, r_f)


def _strip_tooltips(obj) -> None:
    """Recursively drop 'tooltip' from every encoding block so charts don't show a
    hover popup (not needed — the Save button is right below)."""
    if isinstance(obj, dict):
        enc = obj.get("encoding")
        if isinstance(enc, dict):
            enc.pop("tooltip", None)
        for v in obj.values():
            _strip_tooltips(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_tooltips(v)


def _show_chart(chart, dl_name: str, brand_width: int = None, brand_lift: int = 6,
                keep_tooltip: bool = False, brand_embedded: bool = False) -> None:
    """Render a chart with the HockeyROI brand baked INTO the Vega spec, so the
    browser's native "Save as PNG" (the chart's ··· hover menu) carries the brand
    with zero server-side rendering. Non-faceted charts get the two-colour wordmark
    + URL layered inside the plot (bottom-right, just above the x-axis) — it shows
    on-screen AND in a saved PNG without distorting the axes. Faceted/multi-panel
    charts can't embed it that way (value-positioned marks don't resolve to a
    panel's coordinates), so a _brand_row() wordmark is vconcat'd beneath them —
    UNLESS brand_embedded=True, meaning the caller already layered _brand_layer()
    onto one of its own sub-panels, so the footer is skipped to avoid a duplicate.
    brand_lift raises the embedded footer when data crowds the bottom. brand_width
    sizes the vconcat brand row to sit under a fixed-width faceted chart's right
    edge. _CHART_TITLE (set by drill-ins) becomes an in-chart title so a saved
    image names whose data it is. keep_tooltip: most charts strip on-screen
    tooltips, but a chart with no other way to identify a point (e.g. a league-wide
    scatter of unlabeled dots) sets this True so hovering still reveals which is
    which. dl_name is retained as a stable per-chart key hint (no download button
    is rendered — the native menu handles saving)."""
    import altair as alt
    _cd = chart.to_dict()
    _multi = any(k in _cd for k in ("facet", "hconcat", "vconcat", "concat", "repeat"))
    if _multi and not brand_embedded:
        # Can't embed the in-plot wordmark in a faceted/concat chart, so append a
        # brand row beneath the whole thing — still part of the spec, so a native
        # Save-as-PNG includes it. Transparent view stroke drops the border box
        # around the brand row (and matches the panels to the single-chart look).
        disp = alt.vconcat(chart, _brand_row(brand_width),
                           spacing=4).configure_view(stroke=None)
    elif brand_embedded:
        disp = chart                       # caller already layered the wordmark in
    else:
        disp = alt.layer(chart, _brand_layer(brand_lift))
    _on_screen = disp.to_dict()
    # On-screen (and in the native export): strip hover tooltips unless this chart
    # needs them to identify an otherwise-unlabeled point.
    if not keep_tooltip:
        _strip_tooltips(_on_screen)
    # Name whose data it is, as a top-level title (valid on any spec shape) so the
    # saved PNG is self-labeled. Only drill-in views set _CHART_TITLE.
    if _CHART_TITLE:
        _on_screen["title"] = {"text": _CHART_TITLE, "anchor": "start",
                               "fontSize": 15, "color": PALETTE["text"],
                               "fontWeight": "bold", "offset": 8}
    if not _multi:
        # Vega-Lite's default top padding isn't enough room for a titled chart's
        # ascenders at this font size — the SVG renders with overflow:hidden, so
        # without extra padding the top few px of the title's tallest characters
        # get clipped (confirmed via the rendered DOM: title top sat ~7px above
        # the SVG's own top edge).
        _on_screen["padding"] = {"left": 10, "top": 15, "right": 10, "bottom": 10}
    # Render the native "Save as PNG" at 2× so downloads are crisp (not the 1× screen
    # resolution vega-embed defaults to). vega-embed reads export options from the
    # spec's usermeta.embedOptions, so this needs no Streamlit API support and still
    # renders entirely in the browser (zero server cost). Save-as-SVG stays vector.
    _on_screen.setdefault("usermeta", {}).setdefault(
        "embedOptions", {})["scaleFactor"] = 2
    st.vega_lite_chart(_on_screen, use_container_width=True)


# Diverging gradient: a light tint near the 50% midline → the FULL brand colour
# further out (blue navy above 50%, brand orange below) — same colours as the
# solid version, just softened toward the middle.
_BAR_BLUE_LIGHT, _BAR_BLUE_STRONG = "#BCD0E2", PALETTE["text"]      # → #1B3A5C navy
_BAR_ORG_LIGHT, _BAR_ORG_STRONG = "#FF9D6B", PALETTE["orange"]      # → #FF6B35 orange


def _hex_lerp(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a[k] + (b[k] - a[k]) * t):02X}" for k in range(3))


def _bar_color(value: float) -> str:
    """Gradient: closer to the 50% midline → lighter, further → darker; blue above
    50%, orange below. Full saturation at ±20 points (i.e. 30%/70%)."""
    inten = min(1.0, abs(value - 50.0) / 20.0)
    return (_hex_lerp(_BAR_BLUE_LIGHT, _BAR_BLUE_STRONG, inten) if value >= 50
            else _hex_lerp(_BAR_ORG_LIGHT, _BAR_ORG_STRONG, inten))


def _qg_axis_domain(values) -> list:
    vv = [v for v in values if pd.notna(v)]
    if not vv:
        return [30, 70]
    return [min(30, int(np.floor(min(vv))) - 2), max(70, int(np.ceil(max(vv))) + 2)]


def _player_qg_vals(pid: int, trend: pd.DataFrame, label: str) -> dict:
    """The 5 bar metrics (0-100) for one player at a season label or the 2yr row."""
    if label == "2yr avg (24-26)":
        _p2 = _players_2yr_frame()
        _pr = _p2[_p2["player_id"] == int(pid)] if not _p2.empty else _p2
        return {m: (float(_pr[_P2YR_MAP[m]].iloc[0]) * 100
                    if len(_pr) and _P2YR_MAP.get(m) in _pr.columns
                    and pd.notna(_pr[_P2YR_MAP[m]].iloc[0]) else np.nan)
                for m in _QG_BAR_METRICS}
    _tr = trend[trend["Season"].astype(str) == str(label)]
    return {m: (float(_tr[m].iloc[0]) * 100 if len(_tr) and m in _tr.columns
                and pd.notna(_tr[m].iloc[0]) else np.nan) for m in _QG_BAR_METRICS}


# The Zone-Impact-family index metrics for the hard-locked zone bar (0-100
# scale, 50 = position-group average). OZI/DZI/NZI/TZI are already on a 0-100
# basis (unlike the QG metrics, which are stored as 0-1 fractions), so they're
# read straight through with NO ×100 rescale. EZI joins them here too — it's
# the same 0-100/50-avg scale, just EDGE-time-vs-starts instead of PBP-time
# share, so it belongs alongside the others as a 5th lens.
_ZONE_BAR_METRICS = ["OZI", "DZI", "NZI", "TZI", "EZI"]


def _player_zone_vals(pid: int, trend: pd.DataFrame, label: str) -> dict:
    """The Zone-Impact-family index values (OZI/DZI/NZI/TZI + EZI, 0-100) for
    one player at a season label or the 2yr row. 50 = league-average for that
    position group. EZI isn't in the per-season trend table (its position-group
    regression fit needs the WHOLE cohort, not a single player's row), so it's
    looked up fresh from the leaderboard frame (_build_players_frame) for the
    matching scope instead — cached, so repeat lookups are cheap. Comes back
    NaN wherever EZI isn't computed (e.g. playoffs — no zone-start data)."""
    if label == "2yr avg (24-26)":
        _p2 = _players_2yr_frame()
        _pr = _p2[_p2["player_id"] == int(pid)] if not _p2.empty else _p2
        return {m: (float(_pr[m].iloc[0])
                    if len(_pr) and m in _pr.columns
                    and pd.notna(_pr[m].iloc[0]) else np.nan)
                for m in _ZONE_BAR_METRICS}
    _tr = trend[trend["Season"].astype(str) == str(label)]
    out = {m: (float(_tr[m].iloc[0]) if len(_tr) and m in _tr.columns
                and pd.notna(_tr[m].iloc[0]) else np.nan)
           for m in _ZONE_BAR_METRICS if m != "EZI"}
    out["EZI"] = np.nan
    if label in SEASON_KEY:
        _b, _ = _build_players_frame(label)
        if not _b.empty and "EZI" in _b.columns:
            _r = _b[_b["player_id"] == int(pid)]
            if len(_r) and pd.notna(_r["EZI"].iloc[0]):
                out["EZI"] = float(_r["EZI"].iloc[0])
    return out


def _default_qg_year(season_label, seasons) -> str:
    """Row label to default the bar to, from the global Season filter."""
    key = SEASON_KEY.get(season_label) if season_label else None
    if key == "pooled_2yr" and "2yr avg (24-26)" in seasons:
        return "2yr avg (24-26)"
    if season_label in seasons:
        return season_label
    _non2 = [s for s in seasons if s != "2yr avg (24-26)"]
    return _non2[-1] if _non2 else (seasons[-1] if seasons else None)


# Fixed y-axis range for every bar chart on the player-profile page (QG pairs +
# Zone Impact) so they're visually comparable at a glance instead of each
# auto-zooming to its own data.
_PROFILE_BAR_YDOM = [30, 75]


def _qg_bar_chart(vals: dict, label: str, caption: str = None,
                  dl_prefix: str = "QG-bars", ydomain: list = None,
                  title: str = None) -> None:
    """Diverging bar of metric %s vs a 50% baseline (50% = league-median: bar up
    when above, down when below). vals maps display-metric → value on a 0-100
    scale. caption overrides the default (player NFI%+QG) caption; dl_prefix names
    the download file. ydomain fixes the y-axis range (defaults to auto-fit).
    title: chart description (e.g. "Zone Impact Index") — the filtered year/scope
    (label) is appended automatically so the baked-in chart image is self-labeled."""
    import altair as alt
    rows = [{"Metric": m, "value": float(v), "base": 50.0, "color": _bar_color(v)}
            for m, v in vals.items() if pd.notna(v)]
    if not rows:
        st.caption("No values for this selection.")
        return
    d = pd.DataFrame(rows)
    _dom = ydomain or _qg_axis_domain([r["value"] for r in rows])
    st.caption(caption or (f"**{label}** — Quality-Games % vs the **50% "
               "baseline** (bar up = above 50%, down = below; darker = further from 50%). "
               "**NFI** family in blue, **xG** family in orange."))
    # Colour labels by metric family (xG-* orange, NFI-* blue) rather than fixed
    # index positions — scales to any bar count instead of assuming exactly 5.
    _orange_lbls = "[" + ",".join(f"'{r['Metric']}'" for r in rows
                                  if r["Metric"].startswith("xG")) + "]"
    _label_color = {"expr": f"indexof({_orange_lbls}, datum.value) >= 0 "
                            f"? '{PALETTE['orange']}' : '{PALETTE['text']}'"}
    bars = alt.Chart(d).mark_bar(size=30, clip=True).encode(
        x=alt.X("Metric:N", sort=[r["Metric"] for r in rows],
                axis=alt.Axis(labelAngle=0, title=None, labelFontWeight="bold",
                              labelFontSize=12, labelColor=_label_color)),
        y=alt.Y("base:Q", scale=alt.Scale(domain=_dom), title="%"),
        y2="value:Q",
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("value:Q", format=".1f", title="%")])
    rule = alt.Chart(pd.DataFrame({"y": [50.0]})).mark_rule(
        strokeDash=[4, 4], color=PALETTE["text_secondary"]).encode(y="y:Q")
    chart = bars + rule
    if title:
        chart = chart.properties(title=alt.TitleParams(
            text=f"{title} — {label}", color=PALETTE["text"], fontSize=13))
    _show_chart(chart, dl_name=f"{dl_prefix}-{label}")


def _qg_panel_chart(panels: list[tuple], vals: dict, title: str, embed_brand: bool = False):
    """Build (not render) one panel's diverging-bar Altair chart: each concept's
    Raw bar sits directly next to its Rel bar, groups spaced apart via blank
    spacer categories on the x-axis. Every bar keeps its OWN x-axis tick label
    (the metric's display name) — no opacity/legend distinction between Raw and
    Rel; colour is purely the original distance-from-50 diverging scale (darker
    = further from 50, blue above / orange below). embed_brand=True layers the
    HockeyROI wordmark into THIS panel's own bottom-right corner (used on the
    rightmost panel of a combined hconcat, so the logo reads as "bottom-right of
    the graph" rather than a separate footer below the whole combined chart).
    Returns an Altair chart object (caller combines panels + calls _show_chart
    once)."""
    import altair as alt
    order: list[str] = []
    rows = []
    for i, (_concept, raw_m, rel_m) in enumerate(panels):
        for m in (raw_m, rel_m):
            order.append(m)
            v = vals.get(m)
            if pd.notna(v):
                rows.append({"Metric": m, "value": float(v), "base": 50.0,
                            "color": _bar_color(v)})
        if i < len(panels) - 1:
            order.append(f"__spacer_{i}__")   # blank slot, no bar, no label
    d = pd.DataFrame(rows) if rows else pd.DataFrame(
        {"Metric": [], "value": [], "base": [], "color": []})
    bars = alt.Chart(d).mark_bar(size=30, clip=True).encode(
        x=alt.X("Metric:N", sort=order, scale=alt.Scale(domain=order),
                axis=alt.Axis(title=None, labelAngle=-40, labelFontSize=10,
                              labelFontWeight="bold", labelOverlap=False,
                              labelExpr="test('^__spacer', datum.value) ? '' : datum.value")),
        y=alt.Y("base:Q", scale=alt.Scale(domain=_PROFILE_BAR_YDOM), title="%"),
        y2="value:Q",
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("value:Q", format=".1f", title="%")])
    rule = alt.Chart(pd.DataFrame({"y": [50.0]})).mark_rule(
        strokeDash=[4, 4], color=PALETTE["text_secondary"]).encode(y="y:Q")
    layers = bars + rule
    if embed_brand:
        layers = layers + _brand_layer(6)
    # Widened so the two panels together spread toward the page's full content
    # width (near the brand watermark), not a cramped narrow chart.
    return layers.properties(
        width=max(340, 66 * len(order)), height=300,
        title=alt.TitleParams(text=title, color=PALETTE["text"], fontSize=13))


def _qg_paired_bar_chart(vals: dict, label: str, caption: str, dl_prefix: str) -> None:
    """Two SEPARATE Quality-Games bar charts — NFI and xG (MoneyPuck) — each its
    own downloadable image so either can be posted on its own without the other
    crowding it. Each shows Raw+Rel bars paired per concept, spaced between
    concepts, on the same fixed y-axis; the brand wordmark is embedded in each
    chart's own bottom-right corner."""
    st.caption(caption)
    for fam, title in (("NFI", "NFI Quality Games %"),
                       ("xG", "xG (MoneyPuck) Quality Games %")):
        ch = _qg_panel_chart(_QG_PAIR_PANELS[fam], vals, f"{title} — {label}",
                             embed_brand=True)
        _show_chart(ch, dl_name=f"{dl_prefix}-{fam}-{label}", brand_embedded=True)


def _qg_bar_chart_compare(players_vals: dict, label: str, metrics: list[str] = None,
                          caption: str = None, dl_name: str = "Trade-QG-bars",
                          title: str = None) -> None:
    """Side-by-side small-multiple bar charts (one panel per player) of the QG
    metrics vs the 50% baseline. players_vals: {player_name: {metric: 0-100}}.
    metrics restricts to a subset (e.g. NFI-only or xG-only) so the NFI and xG
    families render as two separate charts instead of mixed in one. title: chart
    description — the filtered year/scope (label) is appended automatically."""
    import altair as alt
    _metrics = metrics or _QG_BAR_METRICS
    rows, allv = [], []
    for pname, vals in players_vals.items():
        for m, v in vals.items():
            if m in _metrics and pd.notna(v):
                rows.append({"Player": pname, "Metric": m, "value": float(v),
                             "base": 50.0, "color": _bar_color(v)})
                allv.append(float(v))
    if not rows:
        st.caption("No values to compare for this selection.")
        return
    d = pd.DataFrame(rows)
    _dom = _qg_axis_domain(allv)
    st.caption(caption or (f"**{label}** — Quality-Games % vs the **50% baseline**, "
               "one panel per player (bar up = above 50%; darker = further from 50%)."))
    # Width per panel so the panels together fill a wide layout (faceted charts
    # ignore use_container_width, so size the panels up explicitly).
    _n = max(1, len(players_vals))
    _w = int(max(200, 1040 / _n))
    _ch = alt.Chart(d)   # shared data so a layered chart can be faceted
    bars = _ch.mark_bar(size=34).encode(
        x=alt.X("Metric:N", sort=_metrics,
                axis=alt.Axis(labelAngle=-30, title=None, labelFontWeight="bold",
                              labelFontSize=11, labelColor=PALETTE["text"])),
        y=alt.Y("base:Q", scale=alt.Scale(domain=_dom), title="%"),
        y2="value:Q",
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=[alt.Tooltip("Player:N"), alt.Tooltip("Metric:N"),
                 alt.Tooltip("value:Q", format=".1f", title="%")])
    rule = _ch.mark_rule(strokeDash=[4, 4],
                         color=PALETTE["text_secondary"]).encode(y=alt.datum(50))
    chart = alt.layer(bars, rule).properties(width=_w, height=300).facet(
        column=alt.Column("Player:N", title=None,
                          header=alt.Header(labelFontWeight="bold", labelFontSize=13)))
    if title:
        chart = chart.properties(title=alt.TitleParams(
            text=f"{title} — {label}", color=PALETTE["text"], fontSize=13))
    _show_chart(chart, dl_name=dl_name,
                brand_width=_w * _n + 24 * (_n - 1) + 55)


# Quality-Games line series: display name, colour, and dash per metric. NFI
# family orange / xG family blue; relative versions dashed.
# All 8 QG metrics on one chart. COLOUR = aspect (overall / offense / defense /
# relative); LINE STYLE = model (NFI solid, xG dashed) — so every (colour, dash)
# pair is unique. NFI offense/defense use Attack/Suppress labels; xG uses For/Against.
_QG_LINE_ORDER = ["NFI-QG%", "xG-QG%", "NFI-QG-A%", "xG-QG-F%",
                  "NFI-QG-S%", "xG-QG-A%", "RelNFI-QG%", "RelxG-QG%"]
_QG_SERIES = {"NFI-QG%": "NFI-QG%", "xG-QG%": "xG-QG%",
              "NFI-QG-A%": "NFI-QG-A%", "xG-QG-F%": "xG-QG-F%",
              "NFI-QG-S%": "NFI-QG-S%", "xG-QG-A%": "xG-QG-A%",
              "RelNFI-QG%": "Rel NFI-QG%", "RelxG-QG%": "Rel xG-QG%"}
_QG_SERIES_COLOR = {  # by aspect
    "NFI-QG%": _CHART_PRIMARY, "xG-QG%": _CHART_PRIMARY,          # overall — orange
    "NFI-QG-A%": _CHART_SECOND, "xG-QG-F%": _CHART_SECOND,        # offense — light blue
    "NFI-QG-S%": _CHART_THIRD, "xG-QG-A%": _CHART_THIRD,          # defense — blue
    "Rel NFI-QG%": _CHART_FOURTH, "Rel xG-QG%": _CHART_FOURTH}    # relative — purple
_QG_SERIES_DASH = {  # by model: NFI solid, xG dashed
    "NFI-QG%": [1, 0], "NFI-QG-A%": [1, 0], "NFI-QG-S%": [1, 0], "Rel NFI-QG%": [1, 0],
    "xG-QG%": [6, 4], "xG-QG-F%": [6, 4], "xG-QG-A%": [6, 4], "Rel xG-QG%": [6, 4]}


def _qg_line_long(trend: pd.DataFrame):
    cols = [c for c in _QG_LINE_ORDER if c in trend.columns and trend[c].notna().any()]
    if not cols:
        return None, []
    long = (trend[["Season"] + cols].melt("Season", var_name="Metric",
            value_name="value").dropna(subset=["value"]))
    long["Series"] = long["Metric"].map(_QG_SERIES)
    return long, [_QG_SERIES[c] for c in cols]


def _qg_line_encodings(series):
    """color + strokeDash both keyed on Series (same legend) so Altair merges them
    into ONE bottom legend whose symbols are coloured solid/dashed LINES (not the
    point marker)."""
    import altair as alt
    _leg = alt.Legend(title=None, orient="bottom", direction="horizontal",
                      symbolType="stroke", symbolStrokeWidth=2.5, symbolSize=260)
    return dict(
        color=alt.Color("Series:N", sort=series, legend=_leg,
                        scale=alt.Scale(domain=series,
                                        range=[_QG_SERIES_COLOR[s] for s in series])),
        strokeDash=alt.StrokeDash("Series:N", sort=series, legend=_leg,
                                  scale=alt.Scale(domain=series,
                                                  range=[_QG_SERIES_DASH[s] for s in series])))


def _qg_line_ydomain(values) -> list:
    """Zoom the QG line y-axis tightly to the data (with ~15% padding, min ±0.02) so
    season-to-season movement is visible rather than flattened by a wide band."""
    vv = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if vv.empty:
        return [0.35, 0.75]
    lo, hi = float(vv.min()), float(vv.max())
    pad = max(0.02, (hi - lo) * 0.15)
    return [lo - pad, hi + pad]


_QG_LINE_ORDER_NFI = ["NFI-QG%", "NFI-QG-A%", "NFI-QG-S%", "RelNFI-QG%"]
_QG_LINE_ORDER_XG = ["xG-QG%", "xG-QG-F%", "xG-QG-A%", "RelxG-QG%"]


def _qg_line_long_subset(trend: pd.DataFrame, cols: list[str]):
    cols = [c for c in cols if c in trend.columns and trend[c].notna().any()]
    if not cols:
        return None, []
    long = (trend[["Season"] + cols].melt("Season", var_name="Metric",
            value_name="value").dropna(subset=["value"]))
    long["Series"] = long["Metric"].map(_QG_SERIES)
    return long, [_QG_SERIES[c] for c in cols]


def _qg_split_line_chart(trend: pd.DataFrame, cols: list[str], caption: str,
                         dl_name: str) -> None:
    import altair as alt
    long, series = _qg_line_long_subset(trend, cols)
    if long is None:
        return
    st.caption(caption)
    _leg = alt.Legend(title=None, orient="bottom", direction="horizontal",
                      symbolType="stroke", symbolStrokeWidth=2.5, symbolSize=260)
    chart = alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("Season:N", title=None),
        y=alt.Y("value:Q", title=None,
                scale=alt.Scale(domain=_qg_line_ydomain(long["value"]))),
        color=alt.Color("Series:N", sort=series, legend=_leg,
                        scale=alt.Scale(domain=series,
                                        range=[_QG_SERIES_COLOR[s] for s in series])),
        tooltip=["Season:N", "Series:N", alt.Tooltip("value:Q", format=".3f")],
    ).properties(height=300)
    _show_chart(chart, dl_name=dl_name, brand_lift=33)


def _qg_combined_line(trend: pd.DataFrame) -> None:
    """Two line charts — NFI Quality-Games % and xG (MoneyPuck) Quality-Games %,
    split by model so each is readable on its own. Colour = aspect (overall
    orange, offense light-blue, defense blue, relative purple)."""
    _qg_split_line_chart(
        trend, _QG_LINE_ORDER_NFI,
        "**NFI** Quality Games % — colour = aspect (**overall** orange, "
        "**offense** light-blue, **defense** blue, **relative** purple).",
        "Quality-Games-line-NFI")
    _qg_split_line_chart(
        trend, _QG_LINE_ORDER_XG,
        "**xG (MoneyPuck)** Quality Games % — colour = aspect (**overall** orange, "
        "**offense** light-blue, **defense** blue, **relative** purple).",
        "Quality-Games-line-xG")


def _qg_line_chart_compare(players: dict, cols: list[str] = None, caption: str = None,
                           dl_name: str = "Trade-QG-line") -> None:
    """Side-by-side QG % line charts, one panel per player. players: {name: trend}.
    cols restricts to a subset (e.g. NFI-only or xG-only) so the two model
    families render as two separate charts instead of mixed in one."""
    import altair as alt
    parts, series = [], []
    for name, tr in players.items():
        long, ser = _qg_line_long_subset(tr, cols) if cols else _qg_line_long(tr)
        if long is None:
            continue
        long["Player"] = name
        parts.append(long)
        series = series or ser
    if not parts:
        return
    d = pd.concat(parts, ignore_index=True)
    _n = max(1, len(players))
    _w = int(max(200, 1040 / _n))
    st.caption(caption or ("Quality Games % over time, per player — **NFI** (orange) vs "
               "**xG** (blue); relative (**Rel**) versions **dashed**."))
    base = alt.Chart(d).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("Season:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("value:Q", title=None,
                scale=alt.Scale(domain=_qg_line_ydomain(d["value"]))),
        tooltip=["Player:N", "Season:N", "Series:N", alt.Tooltip("value:Q", format=".3f")],
        **_qg_line_encodings(series)).properties(width=_w, height=280)
    chart = base.facet(column=alt.Column("Player:N", title=None,
                       header=alt.Header(labelFontWeight="bold", labelFontSize=13)))
    _show_chart(chart, dl_name=dl_name,
                brand_width=_w * _n + 24 * (_n - 1) + 55)


def _zone_line_chart_compare(players: dict) -> None:
    """Side-by-side Zone Impact index (OZI/DZI/NZI/TZI, 0-100, 50 = average) line
    charts, one panel per player. players: {name: trend}. Mirrors the QG line
    comparison."""
    import altair as alt
    zcols = ["OZI", "DZI", "NZI", "TZI"]
    parts = []
    for name, tr in players.items():
        cols = [c for c in zcols if c in tr.columns and tr[c].notna().any()]
        if not cols:
            continue
        long = (tr[["Season"] + cols].melt("Season", var_name="Metric",
                value_name="value").dropna(subset=["value"]))
        long["Player"] = name
        parts.append(long)
    if not parts:
        return
    d = pd.concat(parts, ignore_index=True)
    ys = [c for c in zcols if c in set(d["Metric"])]
    _n = max(1, len(players))
    _w = int(max(200, 1040 / _n))
    st.caption("Zone Impact index 0–100 over time, per player — **OZI** / **DZI** / "
               "**NZI** / **TZI** (50 = league-average for the position).")
    base = alt.Chart(d).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("Season:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[30, 70])),
        color=alt.Color("Metric:N", sort=ys, legend=alt.Legend(
            orient="bottom", title=None, symbolType="stroke", symbolStrokeWidth=2.5),
            scale=alt.Scale(domain=ys,
                            range=[_CHART_COLORS.get(c, _CHART_SECOND) for c in ys])),
        tooltip=["Player:N", "Season:N", "Metric:N", alt.Tooltip("value:Q", format=".2f")]
        ).properties(width=_w, height=280)
    chart = base.facet(column=alt.Column("Player:N", title=None,
                       header=alt.Header(labelFontWeight="bold", labelFontSize=13)))
    _show_chart(chart, dl_name="Trade-Zone-line",
                brand_width=_w * _n + 24 * (_n - 1) + 55)


def _trade_line_compare(players: dict, cols: list[str], caption: str,
                        dl_name: str) -> None:
    """Generic faceted (one panel per player) year-over-year line chart for a set
    of trend columns — the multi-player analogue of the drill-in's inner _chart().
    players: {name: trend_df}. Colours follow _CHART_COLORS; y-domain is tightened
    to the data (shared across panels) so season-to-season movement is visible."""
    import altair as alt
    parts = []
    for name, tr in players.items():
        use = [c for c in cols if c in tr.columns and tr[c].notna().any()]
        if not use:
            continue
        long = (tr[["Season"] + use].melt("Season", var_name="Metric",
                value_name="value").dropna(subset=["value"]))
        long["Player"] = name
        parts.append(long)
    if not parts:
        return
    d = pd.concat(parts, ignore_index=True)
    ys = [c for c in cols if c in set(d["Metric"])]
    _n = max(1, len(players))
    _w = int(max(200, 1040 / _n))
    st.caption(caption)
    base = alt.Chart(d).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("Season:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=_tight_domain(d["value"]))),
        color=alt.Color("Metric:N", sort=ys, legend=alt.Legend(
            orient="bottom", title=None, symbolType="stroke", symbolStrokeWidth=2.5),
            scale=alt.Scale(domain=ys,
                            range=[_CHART_COLORS.get(c, _CHART_SECOND) for c in ys])),
        tooltip=["Player:N", "Season:N", "Metric:N", alt.Tooltip("value:Q", format=".2f")]
        ).properties(width=_w, height=280)
    chart = base.facet(column=alt.Column("Player:N", title=None,
                       header=alt.Header(labelFontWeight="bold", labelFontSize=13)))
    _show_chart(chart, dl_name=dl_name, brand_width=_w * _n + 24 * (_n - 1) + 55)


def _shot_chart_seasons(season_label: str | None, playoffs: bool):
    """Map the app scope to a list of season strings for the shot parquets
    (None = all seasons, used for the pooled playoff view)."""
    if playoffs:
        return None
    key = SEASON_KEY.get(season_label, "pooled")
    if key == "pooled":
        return list(POOLED_SEASONS)
    if key == "pooled_2yr":
        return list(POOLED_2YR_SEASONS)
    return [key]


@st.cache_data(show_spinner=False, ttl=3600)
def _load_shots_cached(seasons_tuple, game_type: str) -> pd.DataFrame:
    import shot_charts as _sc
    return _sc.load_shots(list(seasons_tuple) if seasons_tuple else None, game_type)


def _render_shot_chart(kind: str, ident, name: str, team: str | None,
                       season_label: str | None, playoffs: bool) -> None:
    """Shot chart for a player (shots taken), goalie (shots faced), or team
    (shots taken). Reads the committable per-season parquets."""
    try:
        import shot_charts as _sc
    except Exception:
        return
    seasons = _shot_chart_seasons(season_label, playoffs)
    shots = _load_shots_cached(tuple(seasons) if seasons else None,
                               "playoff" if playoffs else "regular")
    if shots.empty:
        return
    if kind == "player":
        shots, gv = shots[shots["shooter_player_id"] == ident], False
    elif kind == "goalie":
        shots, gv = shots[shots["goalie_id"] == ident], True
    else:
        shots, gv = shots[shots["shooting_team_abbrev"] == str(ident)], False
    if shots.empty:
        return
    ng = int(shots["is_goal"].sum())
    lbl = season_label or ("Playoffs" if playoffs else "")
    faced = kind == "goalie"
    nm = name + (" (shots faced)" if faced else "")
    stat = (f"{len(shots):,} {'shots faced' if faced else 'shots'} · "
            f"{ng} {'goals allowed' if faced else 'goals'}"
            + ("  ·  goalie's-eye view" if faced else ""))
    _goals_only = st.checkbox("Goals only", value=False,
                              key=f"shotgoals_{kind}_{ident}")
    fig = _sc.shot_chart(shots, nm, season=str(lbl), stat=stat, team=team,
                         goalie_view=gv, goals_only=_goals_only)
    if fig is not None:
        import io
        _buf = io.BytesIO()
        fig.savefig(_buf, format="png", dpi=220, bbox_inches="tight",
                    facecolor="white")
        # use_container_width=False keeps it at its natural (small) size —
        # otherwise Streamlit stretches the figure to the full column width.
        st.pyplot(fig, clear_figure=True, use_container_width=False)
        st.download_button(
            "⬇", _buf.getvalue(),
            file_name=f"{str(name).replace(' ', '-')}-shot-map.png",
            mime="image/png", key=f"shotdl_{kind}_{ident}",
            help="Save this shot map as a PNG")


def _render_player_profile(pid: int, same_pos: bool = False, families=None,
                           team=None, season_label=None) -> None:
    """Per-season trend table + auto-showing line charts for one player.
    families: optional list of metric families to show (None/empty = all).
    team: when set, cells also show the within-team rank as (league / team).
    season_label: the global Season filter — used to default-select the bar row."""
    disp, trend, metric_cols = _player_profile_table(pid, same_pos=same_pos,
                                                     families=families, team=team)
    if disp.empty:
        st.info("No per-season data available for this player.")
        return
    _show_fams = set(families) if families else set(PLAYER_FAMILY_COLS)
    cohort = "all skaters"
    nfi = load_nfi_player()
    _prow_any = nfi[nfi["player_id"] == int(pid)] if not nfi.empty else nfi
    _highlight_name = _prow_any["player_name"].iloc[0] if len(_prow_any) else None
    if same_pos:
        prow = _prow_any
        if len(prow):
            cohort = "defense" if str(prow["position"].iloc[0]) == "D" else "forwards"
    if team:
        _team_txt = "their own team's" if team == "__own__" else f"**{team}**"
        st.caption(f"Each value shows **(league / team)** rank — rank among "
                   f"**{cohort}** league-wide, then among {_team_txt} skaters that "
                   "season. NFI-S/60 (shots against): lowest = #1.")
    else:
        st.caption(f"Each value shows its **(rank)** — rank among **{cohort}** that "
                   "season. NFI-S/60 (shots against): lowest = #1.")
    _ev = _show_df(disp, width="stretch", hide_index=True, on_select="rerun",
                   selection_mode="single-row", key=f"pl_detail_{int(pid)}")
    _sel = getattr(getattr(_ev, "selection", None), "rows", None)
    _seasons = disp["Season"].astype(str).tolist()
    if _sel and 0 <= _sel[0] < len(_seasons):
        _yr = _seasons[_sel[0]]                 # user clicked a year
    else:
        _yr = _default_qg_year(season_label, _seasons)   # default to filter's year
        # If that row has no data for this player, fall back to the latest that does.
        if _yr and all(pd.isna(v) for v in _player_qg_vals(pid, trend, _yr).values()):
            _wd = [s for s in _seasons
                   if any(pd.notna(v) for v in _player_qg_vals(pid, trend, s).values())]
            if _wd:
                _yr = _wd[-1]
    # View-mode toggle — mutually exclusive, sits ABOVE the bar chart. "Show
    # current year data" (default) renders the bar charts for the selected row;
    # "Year over year" replaces them with the season-by-season line charts;
    # "Shot map" replaces them with the rink shot chart.
    with st.container(key="players_view_mode_box"):
        st.markdown(
            f"<div style='color:{PALETTE['text']}; font-size:1.15rem; "
            "font-weight:700; margin-bottom:0.2rem;'>View</div>",
            unsafe_allow_html=True)
        st.markdown(
            "<style>.st-key-players_view_mode_box "
            "[data-testid='stRadio'] label p{font-size:1.05rem !important; "
            "font-weight:600;}</style>",
            unsafe_allow_html=True)
        view_mode = st.radio(
            "View", ["Show current year data", "Year over year", "Shot map"],
            horizontal=True, key="players_view_mode", label_visibility="collapsed")

    if view_mode == "Shot map":
        _pteam = (str(_prow_any["team"].iloc[0])
                  if len(_prow_any) and "team" in _prow_any.columns
                  and pd.notna(_prow_any["team"].iloc[0]) else None)
        _ppos = (str(_prow_any["position"].iloc[0])
                 if len(_prow_any) and "position" in _prow_any.columns
                 and pd.notna(_prow_any["position"].iloc[0]) else None)
        _nm = _highlight_name or f"Player {pid}"
        if _ppos:
            _nm = f"{_nm} ({_ppos})"
        _render_shot_chart("player", int(pid), _nm, _pteam, season_label,
                           playoffs=False)

    def _chart(title: str, cols: list[str], ydomain=None) -> None:
        import altair as alt
        ys = [c for c in cols if c in trend.columns and trend[c].notna().any()]
        if not ys:
            return
        st.caption(title)
        long = (trend[["Season"] + ys].melt("Season", var_name="Metric",
                value_name="value").dropna(subset=["value"]))
        # ydomain (when given) fixes a comparable spread; otherwise tighten to
        # the actual data range so season-to-season movement is visible
        # instead of flattened by a wide auto-zero domain.
        _ysc = alt.Scale(domain=ydomain or _tight_domain(long["value"]))
        ch = alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("Season:N", title=None),
            y=alt.Y("value:Q", title=None, scale=_ysc),
            color=alt.Color("Metric:N", sort=ys, legend=alt.Legend(
                orient="bottom", title=None, symbolType="stroke", symbolStrokeWidth=2.5),
                scale=alt.Scale(domain=ys,
                                range=[_CHART_COLORS.get(c, _CHART_SECOND) for c in ys])),
            tooltip=["Season:N", "Metric:N", alt.Tooltip("value:Q", format=".2f")]
            ).properties(height=300)
        _show_chart(ch, dl_name=title.split(" (")[0].replace(" ", "-"))

    if view_mode == "Year over year":
        # Line charts — ALL of them still show regardless of which metric-family
        # pills are selected above (the pills filter the table + team scatters),
        # but whichever family is currently toggled on moves to the TOP of this
        # section, so the pills also control what you see first here.
        # One chart per scale so none flattens; y-axes zoom to each chart's own
        # data range so season-to-season movement is visible.
        _yoy_charts = [
            ("Quality Games", lambda: _qg_combined_line(trend)),
            ("xG", lambda: _chart("On-ice xG per 60 (xGF/60, xGA/60)", ["xGF/60", "xGA/60"])),
            ("xG", lambda: _chart("Relative xG % (RelxG%, RelxG-F%, RelxG-A%)",
                   ["RelxG%", "RelxG-F%", "RelxG-A%"])),
            ("xG", lambda: _chart("PDOxG (5v5) — luck net of shot quality", ["PDOxG"])),
            ("Net Front Impact", lambda: _chart(
                   "RelNFI family (RelNFI%, RelNFI-A%, RelNFI-S%)",
                   ["RelNFI%", "RelNFI-A%", "RelNFI-S%"])),
            ("Zone Impact", lambda: _chart(
                   "Zone Impact index 0–100 (OZI, DZI, NZI, TZI) — 50 = average",
                   ["OZI", "DZI", "NZI", "TZI"], ydomain=[30, 70])),
            ("Zone Impact", lambda: _chart(
                   "D/O Zone Start% (faceoff-started 5v5 shifts)",
                   ["OZ Start%", "DZ Start%"])),
            ("Net Front Impact", lambda: _chart(
                   "Raw net-front rate per 60 (NFI-A/60, NFI-S/60)",
                   ["NFI-A/60", "NFI-S/60"])),
            ("Net Front Impact", lambda: _chart("NFI% (net-front share)", ["NFI%"])),
            ("EDGE", lambda: _chart("EDGE Zone-Time % (OZ, DZ)", ["EDGE OZ%", "EDGE DZ%"])),
            ("EDGE", lambda: _chart("EDGE Top Speed (mph)", ["EDGE Top Speed"])),
            ("EDGE", lambda: _chart("EDGE Speed Bursts (20+ mph, season total)",
                   ["EDGE Bursts 20+"])),
            ("EDGE", lambda: _chart("EDGE Distance Skated (mi)", ["EDGE Distance (mi)"])),
        ]
        for _fam, _fn in sorted(_yoy_charts, key=lambda item: item[0] not in _show_fams):
            _fn()
    elif view_mode == "Show current year data":
        # Current-year bars: the combined NFI+xG Quality-Games panel (one
        # download covers both) and the hard-locked Zone Impact bar — both on
        # the same fixed 30–75 y-axis. 50 = baseline; above = better/more
        # O-zone time, below = worse/less.
        _all_qg_vals = _player_qg_vals(pid, trend, _yr)
        _qg_paired_bar_chart(
            _all_qg_vals, _yr,
            caption="Quality Games % vs **50**.",
            dl_prefix="QG-bars")
        _zone_vals = _player_zone_vals(pid, trend, _yr)
        _qg_bar_chart(_zone_vals, _yr,
                     caption="Zone Impact index vs **50** (league average).",
                     dl_prefix="Zone-bars", ydomain=_PROFILE_BAR_YDOM,
                     title="Zone Impact Index")
        st.caption("↕ Click a different year (or the 2yr row) above to change the bars.")

    # Team scatters — shown regardless of which family pills are selected,
    # auto-scoped to this player's own team, so a drill-in gives an immediate
    # team-context view. Suppressed in "Shot map" view, which is meant to be
    # the shot chart on its own.
    if view_mode != "Shot map":
        _render_team_scatters(trend, season_label, same_pos, cohort, _highlight_name)


def _render_team_scatters(trend: pd.DataFrame, season_label: str, same_pos: bool,
                          cohort: str, highlight_name: str, dl_suffix: str = "-drill",
                          heading_prefix: str = "") -> None:
    """The 5 team-scatter charts (PDO/xG%, NFI%/xG%, EDGE zone, Zone Starts,
    EDGE speed), auto-scoped to a player's own team. Shared by the single-
    player drill-in and the Trade Analyzer (one call per traded player)."""
    _my_team = None
    if "Team" in trend.columns:
        for _t in trend["Team"].dropna().iloc[::-1]:   # most recent season first
            if isinstance(_t, str) and _t:
                _my_team = _t.split(" / ")[0]           # traded mid-season: first team listed
                break
    if not _my_team:
        return
    # Uses the page's actual season filter now that D/N/O Start% has a
    # real per-season cut (previously force-pooled to always have Start%
    # data, which meant a team's roster here could include players from
    # any of the last 4 seasons — e.g. a long-retired player still
    # showing up under their old team when viewing a recent season).
    _team_frame = _team_scatter_frame(season_label or "4yr (2022-2026)", team=_my_team)
    # Keep the full-roster (both-position) copy for the speed scatter, which
    # always shows Forwards AND Defense sub-charts regardless of the drilled-in
    # player's position — the same_pos restriction below would otherwise strip
    # out the other position group.
    _team_frame_allpos = _team_frame
    # Respect the "Rank against Defense/Forwards only" choice above — a
    # same_pos view should scatter against the player's own position
    # group on the team, not the whole roster.
    if same_pos and not _team_frame.empty and "position" in _team_frame.columns:
        _pos_code = "D" if cohort == "defense" else "F"
        _team_frame = _team_frame[_team_frame["position"] == _pos_code]
    if _team_frame.empty:
        return
    # True league-wide frame (no team filter) — needed for any "league average"
    # crosshair on a team-scoped chart, since _team_frame itself is only this
    # player's own team and averaging it would silently compute a TEAM average
    # and mislabel it "league average".
    _league_frame = _team_scatter_frame(season_label or "4yr (2022-2026)")
    _scope_txt = f" ({cohort})" if same_pos else ""
    _yl = season_label or "4yr (2022-2026)"
    st.markdown(f"<h3 style='color:{PALETTE['text']}; margin-top:1.5rem;'>{heading_prefix}{_my_team} "
                f"Team Scatters{_scope_txt}</h3>", unsafe_allow_html=True)
    if {"PDOxG", "xG_QG_pct"}.issubset(_team_frame.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>PDOxG vs "
                    f"xG-QG%</h4>", unsafe_allow_html=True)
        _pdoxg_xgqg_scatter(_team_frame, True, dl_suffix=dl_suffix, highlight_name=highlight_name,
                            year_label=_yl)
    if {"PDOxG", "NFI_QG_pct"}.issubset(_team_frame.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>PDOxG vs "
                    f"NFI-QG%</h4>", unsafe_allow_html=True)
        _pdoxg_nfiqg_scatter(_team_frame, True, dl_suffix=dl_suffix, highlight_name=highlight_name,
                             year_label=_yl)
    if {"xG_QG_pct", "NFI_QG_pct"}.issubset(_team_frame.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>xG-QG% vs "
                    f"NFI-QG%</h4>", unsafe_allow_html=True)
        _xgqg_nfiqg_scatter(_team_frame, True, dl_suffix=dl_suffix, highlight_name=highlight_name,
                            year_label=_yl)
    if {"PDOxG", "xG%"}.issubset(_team_frame.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>PDOxG vs "
                    f"xG%</h4>", unsafe_allow_html=True)
        _pdo_xg_scatter(_team_frame, True, dl_suffix=dl_suffix, highlight_name=highlight_name,
                       year_label=_yl)
    if {"PDOxG", "NFI%"}.issubset(_team_frame.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>PDOxG vs "
                    f"NFI%</h4>", unsafe_allow_html=True)
        _pdo_nfi_scatter(_team_frame, True, dl_suffix=dl_suffix, highlight_name=highlight_name,
                        year_label=_yl)
    if {"NFI%", "xG%"}.issubset(_team_frame.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>NFI% vs "
                    f"xG%</h4>", unsafe_allow_html=True)
        _nfi_xg_scatter(_team_frame, True, dl_suffix=dl_suffix, highlight_name=highlight_name,
                       year_label=_yl)
    if {"EDGE DZ%", "EDGE OZ%"}.issubset(_team_frame.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: D-Zone vs "
                    f"O-Zone Time%</h4>", unsafe_allow_html=True)
        _edge_zone_scatter(_team_frame, True, dl_suffix=dl_suffix, highlight_name=highlight_name,
                          year_label=_yl)
    if {"DZ Start%", "OZ Start%"}.issubset(_team_frame.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>Zone Starts: D-Zone "
                    f"vs O-Zone</h4>", unsafe_allow_html=True)
        _zone_start_scatter(_team_frame, True, dl_suffix=dl_suffix, highlight_name=highlight_name,
                           year_label=_yl)
    if {"EDGE OZ%", "DZ Start%", "NZ Start%"}.issubset(_team_frame.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EZI: EDGE O-Zone "
                    f"Time vs Non-O-Zone Starts</h4>", unsafe_allow_html=True)
        _ezi_scatter(_team_frame, True, dl_suffix=dl_suffix, highlight_name=highlight_name,
                    year_label=_yl)
    if {"EDGE Top Speed", "EDGE Bursts 20+"}.issubset(_team_frame_allpos.columns):
        _edge_speed_scatter(_team_frame_allpos, True, dl_suffix=dl_suffix, highlight_name=highlight_name,
                           year_label=_yl, league_df=_league_frame)


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
        # Fenwick (blocks excluded) — the raw NFI-A/60 / NFI-S/60 source, matching
        # the regular path. Playoff counts file now carries the _fen columns.
        for_att=("onice_for_fen", "sum"),
        ag_att=("onice_ag_fen", "sum"),
        es_toi_min=("toi_min", "first"),   # ES TOI constant across CNFI/MNFI rows
    ).reset_index()
    ok = g["es_toi_min"] > 0
    g["NFI_A_rate"] = np.where(ok, g["for_att"] / g["es_toi_min"] * 60.0, np.nan)
    g["NFI_S_rate"] = np.where(ok, g["ag_att"] / g["es_toi_min"] * 60.0, np.nan)
    return g[["player_id", "NFI_A_rate", "NFI_S_rate"]]


@st.cache_data(show_spinner=False, ttl=3600)
def load_zone_playoffs() -> pd.DataFrame:
    """Name-keyed playoff OZI/DZI/NZI/TZI (0–100 index, 50 = position-group
    average) pooled across all playoff games, (player_name, _pos_group)-keyed
    exactly like load_zone_pooled. Built by build_zone_index100_playoffs.py.
    Playoff samples are short, so qualifying is by the 50-shift gate (no 20-GP
    floor) — a smaller set of players than the regular-season pool."""
    frames = []
    for pos_file, grp in (("forwards", "F"), ("defense", "D")):
        fp = ADJ / "zone_index100" / f"playoffs_{pos_file}.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        keep = [c for c in ("player_name", "OZI", "DZI", "NZI", "TZI")
                if c in d.columns]
        d = d[keep].copy()
        d["_pos_group"] = grp
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    z = pd.concat(frames, ignore_index=True)
    return z.drop_duplicates(subset=["player_name", "_pos_group"], keep="first")


@st.cache_data(show_spinner=False, ttl=3600)
def load_zone_start_pooled() -> pd.DataFrame:
    """Per-player faceoff-START zone split (D/N/O), pooled across all regular
    seasons, from Zones/output/zone_time_raw.csv — MY PBP data (faceoff-started
    5v5 shifts), player_id-keyed. Presentation only: shares of
    oz/dz/nz_faceoff_shifts — does not touch OZI/NZI/DZI."""
    fp = ZONES / "output" / "zone_time_raw.csv"
    if not fp.exists():
        return pd.DataFrame()
    d = pd.read_csv(fp, usecols=["player_id", "oz_faceoff_shifts",
                                  "dz_faceoff_shifts", "nz_faceoff_shifts"])
    total = d["oz_faceoff_shifts"] + d["dz_faceoff_shifts"] + d["nz_faceoff_shifts"]
    ok = total > 0
    d["OZ Start%"] = np.where(ok, d["oz_faceoff_shifts"] / total * 100, np.nan)
    d["DZ Start%"] = np.where(ok, d["dz_faceoff_shifts"] / total * 100, np.nan)
    d["NZ Start%"] = np.where(ok, d["nz_faceoff_shifts"] / total * 100, np.nan)
    return d[["player_id", "OZ Start%", "DZ Start%", "NZ Start%"]]


@st.cache_data(show_spinner=False, ttl=3600)
def load_zone_start_per_season_raw() -> pd.DataFrame:
    """Per-player, per-season faceoff-START zone split (D/N/O) RAW SHIFT
    COUNTS, from Zones/output/zone_start_per_season.csv — same underlying
    PBP faceoff-shift classification as load_zone_start_pooled, just kept
    per-season instead of pre-collapsed, so any scope (single season, 2yr,
    4yr pooled) can compute its own Start% via ratio-of-sums. No minimum-
    sample floor (a descriptive "who starts where" share, not a rating)."""
    fp = ZONES / "output" / "zone_start_per_season.csv"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_csv(fp, dtype={"season": str})


def _zone_start_rate(scope_key: str) -> pd.DataFrame:
    """D/N/O Start% for one scope, via ratio-of-sums over the per-season raw
    faceoff-shift counts (sum across the scope's seasons, then compute the
    share once) — the same pattern as _xg_onice_rates/_edge_distance_rate,
    so a single season, the 2yr pool, and the 4yr pool are all internally
    consistent instead of one being a different aggregation method."""
    d = load_zone_start_per_season_raw()
    if d.empty:
        return pd.DataFrame()
    if scope_key == "pooled":
        sub = d[d["season"].isin(POOLED_SEASONS)]
    elif scope_key == "pooled_2yr":
        sub = d[d["season"].isin(POOLED_2YR_SEASONS)]
    else:
        sub = d[d["season"] == str(scope_key)]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("player_id").agg(
        _oz=("oz_faceoff_shifts", "sum"), _dz=("dz_faceoff_shifts", "sum"),
        _nz=("nz_faceoff_shifts", "sum")).reset_index()
    total = g["_oz"] + g["_dz"] + g["_nz"]
    ok = total > 0
    g["OZ Start%"] = np.where(ok, g["_oz"] / total * 100, np.nan)
    g["DZ Start%"] = np.where(ok, g["_dz"] / total * 100, np.nan)
    g["NZ Start%"] = np.where(ok, g["_nz"] / total * 100, np.nan)
    return g[["player_id", "OZ Start%", "DZ Start%", "NZ Start%"]]


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
            qcols = ["player_id", "GP", "qualifying_GP", "xG_QG_pct", "NFI_QG_pct",
                     "RelNFI_QG_pct", "RelxG_QG_pct", "RelxG_pct",
                     "RelxG_F_pct", "RelxG_A_pct"]
            base = base.merge(qg[[c for c in qcols if c in qg.columns]],
                              on="player_id", how="left")
        zone = load_zone_playoffs()
        if not zone.empty:
            base["_pos_group"] = np.where(base["position"] == "D", "D", "F")
            base = base.merge(zone, on=["player_name", "_pos_group"], how="left")
        as_df = load_as_counts_playoffs()
        if not as_df.empty:
            base = base.merge(as_df, on="player_id", how="left")
        xg = _xg_onice_rates("pooled", playoffs=True)
        if not xg.empty and not base.empty:
            base = base.merge(xg, on="player_id", how="left")
        qgfa = _qg_fa_rates("pooled", playoffs=True)
        if not qgfa.empty and not base.empty:
            base = base.merge(qgfa, on="player_id", how="left")
        # PDO / PDOxG — pooled across all playoff games (build_pdo_sog_playoffs.py).
        pdo = _pdo_rate_playoffs()
        if not pdo.empty and not base.empty:
            base = base.merge(pdo, on="player_id", how="left")
        # NHL EDGE tracking — pooled across all playoff seasons a player appears
        # in (games-played-weighted average, raw totals ratio-of-summed).
        edge = _edge_rate_playoffs()
        if not edge.empty and not base.empty:
            base = base.merge(edge.rename(columns=_EDGE_REN), on="player_id", how="left")
            _ok_toi = base["toi_min"] > 0
            if "_dist_sum" in base.columns:
                base["EDGE Distance/60"] = np.where(
                    _ok_toi, base["_dist_sum"] / base["toi_min"] * 60.0, np.nan)
            base = base.drop(columns=[c for c in ("_dist_sum",) if c in base.columns])
        # Per-situation suite (situation toggle-driven). Playoffs pool all games.
        sit = _situation_metrics("pooled", _situation_toggle_state(), playoffs=True)
        if not sit.empty and not base.empty:
            base = base.merge(sit, on="player_id", how="left")
            base = _add_situation_rel(base, "pooled", _situation_toggle_state(), playoffs=True)
        base = base.drop(columns=[c for c in base.columns
                                  if c.startswith("_sr_") or c.startswith("_tr_")],
                         errors="ignore")
        base = _unify_situation_xg(base)
        _box = _box_score_scope("pooled", playoffs=True)
        if not _box.empty and not base.empty:
            base = base.merge(_box, on="player_id", how="left")
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
        zstart = _zone_start_rate(key)
        if not zstart.empty and not base.empty:
            base = base.merge(zstart, on="player_id", how="left")
    else:
        base = nfi[nfi["season"] == SEASON_KEY[season_label]].copy()
        if not qg.empty:
            qcols = ["player_id", "season", "GP", "qualifying_GP",
                     "xG_QG_pct", "NFI_QG_pct",
                     "RelNFI_QG_pct", "RelxG_QG_pct", "RelxG_pct",
                     "RelxG_F_pct", "RelxG_A_pct",
                     "teams_in_season"]  # for multi-team display + filter
            base = base.merge(qg[[c for c in qcols if c in qg.columns]],
                              on=["player_id", "season"], how="left")
        # Per-season Zone Impact (uncapped per-season files), name-keyed on
        # (player_name, pos-group) like the pooled/2yr zone joins.
        zone = load_zone_per_season(SEASON_KEY[season_label])
        if not zone.empty and not base.empty:
            base["_pos_group"] = np.where(base["position"] == "D", "D", "F")
            base = base.merge(zone, on=["player_name", "_pos_group"], how="left")
        zstart = _zone_start_rate(SEASON_KEY[season_label])
        if not zstart.empty and not base.empty:
            base = base.merge(zstart, on="player_id", how="left")

    # Raw attack/suppress per-60 (ES CNFI+MNFI on-ice for/against), scoped to the
    # same seasons as the view via ratio-of-sums. Joins on player_id.
    as_df = _as_rates(key)
    if not as_df.empty and not base.empty:
        base = base.merge(as_df, on="player_id", how="left")
    # On-ice xGF/60 and xGA/60 (MoneyPuck-style), same ratio-of-sums scoping.
    xg = _xg_onice_rates(key)
    if not xg.empty and not base.empty:
        base = base.merge(xg, on="player_id", how="left")
    # PDO (5v5, SOG-based shooting%+save% luck proxy) — raw context column
    # beside xG. Regular season only.
    pdo = _pdo_rate(key)
    if not pdo.empty and not base.empty:
        base = base.merge(pdo, on="player_id", how="left")
    # NHL EDGE tracking columns — a separate metric family, different basis
    # than NZI/DZI/OZI (see edge/README.md). Regular season only.
    edge = _edge_rate(key)
    if not edge.empty and not base.empty:
        base = base.merge(edge, on="player_id", how="left")
    # EZI (EDGE Zone Impact) — how much more/less EDGE O-zone TIME a player
    # gets than their O-zone faceoff STARTS predict. See _add_ezi docstring.
    base = _add_ezi(base)
    # EDGE distance skated, normalized to a per-60-minutes rate (ratio-of-sums,
    # not games-weighted averaging — see _edge_distance_rate docstring).
    edge_dist_rate = _edge_distance_rate(key)
    if not edge_dist_rate.empty and not base.empty:
        base = base.merge(edge_dist_rate, on="player_id", how="left")
    # EDGE 20+ mph speed bursts per 60 (all-situations bursts / all-situations
    # TOI) — restored now that per-situation TOI exists.
    edge_burst_rate = _edge_bursts_rate(key)
    if not edge_burst_rate.empty and not base.empty:
        base = base.merge(edge_burst_rate, on="player_id", how="left")
    # Quality-Games For/Against (xG-QG-F/A%, NFI-QG-F/A%), ratio-of-sums pooling.
    qgfa = _qg_fa_rates(key)
    if not qgfa.empty and not base.empty:
        base = base.merge(qgfa, on="player_id", how="left")
    # Per-situation possession/xG/individual suite — reflects the situation
    # toggle (5v5 / PP / PK / 4v4 / 3v3 / 5v3 / All). Additive "Sit " columns.
    sit = _situation_metrics(key, _situation_toggle_state(), playoffs=False)
    if not sit.empty and not base.empty:
        base = base.merge(sit, on="player_id", how="left")
        base = _add_situation_rel(base, key, _situation_toggle_state(), playoffs=False)
    base = base.drop(columns=[c for c in base.columns
                              if c.startswith("_sr_") or c.startswith("_tr_")],
                     errors="ignore")
    base = _unify_situation_xg(base)
    # Box-score counting stats (Box Score metric family).
    _box = _box_score_scope(key, playoffs=False)
    if not _box.empty and not base.empty:
        base = base.merge(_box, on="player_id", how="left")
    return base, is_pooled


# Box-score display columns (a metric family on the Player List, not a tab).
# Defined here so PLAYER_FAMILY_COLS below can reference it.
BOX_FAMILY_COLS = ["G", "A1", "A2", "A", "Pts", "PPP", "SHP", "Sh", "Sh%", "GWG",
                   "Reb Created", "TK", "GV", "Hits", "Hits Taken", "Blocks",
                   "Min Pen", "Maj Pen", "PIM", "Pen Drawn", "FO W", "FO L", "FO%"]

# Player List metric families — the collapse filter toggles each group's columns
# (display names, post-rename). Identity columns (Player/Pos/Team/GP/TOI) always
# show.
PLAYER_FAMILY_COLS = {
    # Quality Games = the "-QG%" metrics only (share of games that were "quality").
    "Quality Games": ["xG-QG%", "xG-QG-F%", "xG-QG-A%", "RelxG-QG%",
                      "RelxG-QG-F%", "RelxG-QG-A%",
                      "NFI-QG%", "NFI-QG-A%", "NFI-QG-S%", "RelNFI-QG%",
                      "RelNFI-QG-A%", "RelNFI-QG-S%"],
    # xG / possession — all situation-driven (they follow the Situation filter).
    "xG": (["xGF/60", "xGA/60", "xG%", "RelxG%", "RelxG-F%", "RelxG-A%",
            "PDO", "PDOxG"] + _SIT_PLAIN),
    "Net Front Impact": ["RelNFI%", "RelNFI-A%", "RelNFI-S%", "NFI%",
                         "NFI-A/60", "NFI-S/60"],
    "Zone Impact": ["DZ Start%", "NZ Start%", "OZ Start%",
                    "OZI", "DZI", "NZI", "TZI"],
    # NHL EDGE tracking — a separate basis than NZI/DZI/OZI (player-position,
    # all-situations/EV tracking vs strict 5v5 faceoff-started PBP). See
    # edge/README.md. D/N/O Start% is NOT EDGE data (it's my own PBP faceoff
    # data), but it's shown here too (as well as under Zone Impact) since it's
    # directly comparable context alongside EDGE's own zone-time% — the
    # methodology tab spells out the different source/definition so it's not
    # mistaken for an EDGE-API stat.
    "EDGE": _EDGE_VALUE_DISP + ["DZ Start%", "NZ Start%", "OZ Start%"],
    # Box score — NHL counting stats (all situations; not situation-filtered).
    "Box Score": BOX_FAMILY_COLS,
}


def _team_scatter_frame(season_label: str, team: str = None) -> pd.DataFrame:
    """Player-level frame with everything the 4 team-scatter charts need (PDO,
    xG, NFI%, D/N/O Start%, EDGE speed/bursts) — one shared build so the main
    leaderboard and the drill-in's auto team-scatters use identical data.
    Optionally filtered to one team."""
    base, _ = _build_players_frame(season_label)
    if base.empty:
        return pd.DataFrame()
    base = base.rename(columns={"player_name": "Player", "team": "Team",
                                "NFI_pct": "NFI%", "toi_min": "TOI",
                                "RelxG_pct": "RelxG%", "RelxG_F_pct": "RelxG-F%",
                                "RelxG_A_pct": "RelxG-A%", **_EDGE_REN})
    if team:
        base = base[base["Team"] == team]
    return base


def _scatter_with_labels(df: pd.DataFrame, x_col: str, y_col: str, x_title: str,
                         y_title: str, dl_name: str, caption: str,
                         team_scoped: bool, name_col: str = "Player",
                         extra_layer=None, color_col: str = None,
                         color_title: str = None, highlight_name: str = None,
                         domain_df: pd.DataFrame = None,
                         year_label: str = None) -> None:
    """Shared scatter renderer for the 5 team-scatter charts: tight (non-zero)
    axis domains so points aren't clustered in a corner, player-name labels
    shown directly ONLY when team_scoped (a small, readable point count) —
    otherwise names are hover-only (league-wide would be unreadable), and
    shown as LAST NAME only to keep the chart readable. Dots are always blue
    (uniform size), labels always orange. color_col (optional): a light
    (easy) -> dark (hard) color gradient by a 3rd metric — team-scoped views
    ONLY (league-wide always plain blue dots, since a color legend across
    hundreds of points isn't readable); silently falls back to plain dots if
    that column isn't available for the current scope. year_label (optional):
    the filtered season/scope (e.g. "2024-25", "4yr (2022-2026)", "Playoffs")
    — baked into the chart's own title (as "{y_title} vs {x_title} — {year}")
    so the downloaded image is self-labeled. highlight_name
    (optional): the searched/drilled-into player's full name — their label
    renders bold and slightly larger so they're easy to pick out among
    teammates."""
    import altair as alt
    d = df.dropna(subset=[x_col, y_col]).copy()
    if d.empty:
        return
    st.caption(caption)
    # Axis domain source: by default the plotted points, but when domain_df is
    # given (e.g. a 2-player trade comparison) scale to the full-league range so
    # a couple of points aren't zoomed in to a corner — a small real gap (e.g.
    # two elite skaters' top speed) then reads as small, not exaggerated.
    _dom = domain_df if (domain_df is not None and not domain_df.empty) else d
    _xdom = _tight_domain(_dom[x_col].dropna() if x_col in _dom else d[x_col],
                          pad_frac=0.15, min_pad=1e-6)
    _ydom = _tight_domain(_dom[y_col].dropna() if y_col in _dom else d[y_col],
                          pad_frac=0.15, min_pad=1e-6)
    _use_color = (team_scoped and color_col is not None and color_col in d.columns
                  and d[color_col].notna().any())
    # Trim to only the columns this chart actually plots/tooltips — the caller
    # often passes a wide leaderboard/team frame with dozens of unrelated
    # metrics, and embedding all of them would let Streamlit's chart-hover
    # "Show data" button (and any Vega data export) expose columns that were
    # never drawn on this chart.
    _keep = [c for c in (name_col, x_col, y_col, color_col if _use_color else None)
             if c is not None]
    d = d[_keep].copy()
    _tooltip = [alt.Tooltip(f"{name_col}:N"), alt.Tooltip(f"{x_col}:Q", format=".2f"),
                alt.Tooltip(f"{y_col}:Q", format=".2f")]
    if _use_color:
        _tooltip.append(alt.Tooltip(f"{color_col}:Q", title=color_title or color_col, format=".1f"))
        # Explicit light-blue -> deep-navy range (not a named "blues" scheme):
        # the scheme's pale end was reading as barely-off-white against the
        # chart background, making the gradient look flat. These two brand
        # colors (also used for the floating-bar charts) give real contrast.
        points = alt.Chart(d).mark_circle(size=90, opacity=0.95).encode(
            x=alt.X(f"{x_col}:Q", title=x_title, scale=alt.Scale(domain=_xdom, zero=False)),
            y=alt.Y(f"{y_col}:Q", title=y_title, scale=alt.Scale(domain=_ydom, zero=False)),
            color=alt.Color(f"{color_col}:Q", title=color_title or color_col,
                            scale=alt.Scale(range=[_BAR_BLUE_STRONG, _BAR_BLUE_LIGHT]),
                            legend=None),
            tooltip=_tooltip,
        )
    else:
        points = alt.Chart(d).mark_circle(size=90, opacity=0.75, color=PALETTE["blue"]).encode(
            x=alt.X(f"{x_col}:Q", title=x_title, scale=alt.Scale(domain=_xdom, zero=False)),
            y=alt.Y(f"{y_col}:Q", title=y_title, scale=alt.Scale(domain=_ydom, zero=False)),
            tooltip=_tooltip,
        )
    chart = points + extra_layer if extra_layer is not None else points
    if team_scoped:
        d["_label"] = d[name_col].astype(str).str.split().str[-1]
        _is_hl = (highlight_name is not None) and (d[name_col] == highlight_name).any()
        if _is_hl:
            _hl = d[d[name_col] == highlight_name]
            _rest = d[d[name_col] != highlight_name]
            labels = alt.Chart(_rest).mark_text(align="left", dx=6, dy=-6, fontSize=10,
                                                color=PALETTE["orange"]).encode(
                x=alt.X(f"{x_col}:Q", scale=alt.Scale(domain=_xdom, zero=False)),
                y=alt.Y(f"{y_col}:Q", scale=alt.Scale(domain=_ydom, zero=False)),
                text="_label:N",
            )
            hl_label = alt.Chart(_hl).mark_text(align="left", dx=7, dy=-7, fontSize=13,
                                                fontWeight="bold",
                                                color=PALETTE["orange"]).encode(
                x=alt.X(f"{x_col}:Q", scale=alt.Scale(domain=_xdom, zero=False)),
                y=alt.Y(f"{y_col}:Q", scale=alt.Scale(domain=_ydom, zero=False)),
                text="_label:N",
            )
            chart = chart + labels + hl_label
        else:
            labels = alt.Chart(d).mark_text(align="left", dx=6, dy=-6, fontSize=10,
                                            color=PALETTE["orange"]).encode(
                x=alt.X(f"{x_col}:Q", scale=alt.Scale(domain=_xdom, zero=False)),
                y=alt.Y(f"{y_col}:Q", scale=alt.Scale(domain=_ydom, zero=False)),
                text="_label:N",
            )
            chart = chart + labels
    if year_label:
        # Single line only — a 2-line array title was tried (splitting long
        # axis-title pairs like EZI's onto their own line) but Vega-Lite
        # doesn't reserve extra top margin for it in this render path, so
        # line 1 got clipped above the SVG's own top edge (confirmed via the
        # rendered DOM: title top -11px vs SVG top 0px). The in-app chart
        # width (1120px+, use_container_width) comfortably fits even the
        # longest of these titles on one line, so single-line is both
        # simpler and doesn't clip.
        _title_text = f"{y_title} vs {x_title} — {year_label}"
        chart = chart.properties(title=alt.TitleParams(
            text=_title_text, color=PALETTE["text"], fontSize=13))
    # League-wide (not team_scoped) scatters have no visible name label — hover
    # is the ONLY way to identify a point — so keep the tooltip there. Team-
    # scoped scatters already show a name label on every dot, so they stay
    # tooltip-free like other charts.
    _show_chart(chart, dl_name=dl_name, keep_tooltip=not team_scoped)


def _pdo_xg_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "", highlight_name: str = None,
                    domain_df: pd.DataFrame = None, year_label: str = None) -> None:
    import altair as alt
    if not {"PDOxG", "xG%"}.issubset(df.columns):
        return
    rule0 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color=PALETTE["text_secondary"], strokeDash=[4, 4]).encode(y="y:Q")
    _scatter_with_labels(
        df, "xG%", "PDOxG", "xG%", "PDOxG (luck net of shot quality)",
        f"pdoxg-vs-xg-pct{dl_suffix}",
        "Descriptive, not a ranking — see Methodology. Color = **OZ Start%** "
        "(light = easier, dark = harder).",
        team_scoped, extra_layer=rule0, color_col="OZ Start%",
        color_title="OZ Start% (light = easier, dark = harder)", highlight_name=highlight_name,
        domain_df=domain_df, year_label=year_label)


def _pdo_nfi_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "", highlight_name: str = None,
                     domain_df: pd.DataFrame = None, year_label: str = None) -> None:
    import altair as alt
    if not {"PDOxG", "NFI%"}.issubset(df.columns):
        return
    rule0 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color=PALETTE["text_secondary"], strokeDash=[4, 4]).encode(y="y:Q")
    _scatter_with_labels(
        df, "NFI%", "PDOxG", "NFI%", "PDOxG (luck net of shot quality)",
        f"pdoxg-vs-nfi-pct{dl_suffix}",
        "Descriptive, not a ranking — see Methodology. Color = **OZ Start%** "
        "(light = easier, dark = harder).",
        team_scoped, extra_layer=rule0, color_col="OZ Start%",
        color_title="OZ Start% (light = easier, dark = harder)", highlight_name=highlight_name,
        domain_df=domain_df, year_label=year_label)


def _zone_start_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "", highlight_name: str = None,
                        domain_df: pd.DataFrame = None, year_label: str = None) -> None:
    if not {"DZ Start%", "OZ Start%"}.issubset(df.columns):
        return
    _scatter_with_labels(
        df, "DZ Start%", "OZ Start%", "DZ Start%", "OZ Start%",
        f"zone-start-scatter{dl_suffix}",
        "Faceoff-started 5v5 shifts, one point per player.",
        team_scoped, highlight_name=highlight_name, domain_df=domain_df,
        year_label=year_label)


def _edge_zone_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "", highlight_name: str = None,
                       domain_df: pd.DataFrame = None, year_label: str = None) -> None:
    if not {"EDGE DZ%", "EDGE OZ%"}.issubset(df.columns):
        return
    _scatter_with_labels(
        df, "EDGE DZ%", "EDGE OZ%", "EDGE DZ%", "EDGE OZ%",
        f"EDGE-zone-scatter{dl_suffix}",
        "NHL EDGE tracking, one point per player. Color = **OZ Start%** "
        "(light = easier, dark = harder).",
        team_scoped, color_col="OZ Start%",
        color_title="OZ Start% (light = easier, dark = harder)", highlight_name=highlight_name,
        domain_df=domain_df, year_label=year_label)


def _ezi_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "", highlight_name: str = None,
                 domain_df: pd.DataFrame = None, year_label: str = None) -> None:
    """EDGE O-zone TIME vs non-O-zone faceoff STARTS (DZ Start% + NZ Start%) —
    the two raw ingredients behind EZI, plotted directly so a mismatch (low
    starts, high time, or the reverse) is visible as distance from the
    diagonal rather than collapsed into the single EZI number."""
    if not {"EDGE OZ%", "DZ Start%", "NZ Start%"}.issubset(df.columns):
        return
    d = df.copy()
    d["D/N Start%"] = d["DZ Start%"] + d["NZ Start%"]
    if domain_df is not None and not domain_df.empty and {"DZ Start%", "NZ Start%"}.issubset(domain_df.columns):
        domain_df = domain_df.copy()
        domain_df["D/N Start%"] = domain_df["DZ Start%"] + domain_df["NZ Start%"]
    _scatter_with_labels(
        d, "EDGE OZ%", "D/N Start%", "EDGE O-Zone Time%", "D/N-Zone Start% (100 − OZ Start%)",
        f"ezi-scatter{dl_suffix}",
        "NHL EDGE O-zone time vs non-O-zone faceoff starts — top-right = EZI "
        "outperformers (low O-zone starts, high O-zone time); bottom-left = "
        "underperformers (sheltered starts that aren't converting to time).",
        team_scoped, highlight_name=highlight_name, domain_df=domain_df,
        year_label=year_label)


def _edge_speed_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "", highlight_name: str = None,
                        domain_df: pd.DataFrame = None, year_label: str = None,
                        league_df: pd.DataFrame = None,
                        title_prefix: str = "EDGE: Speed Bursts vs Top Speed") -> None:
    """Skating speed vs speed-burst rate, rendered as TWO scatters — one for
    forwards, one for defense — ALWAYS both, regardless of which (if any)
    player is drilled into. NHL computes its EDGE speed percentiles WITHIN
    position group (its own published "league average" is 22.17 mph for F vs
    21.59 mph for D — forwards skate faster), so a single mixed crosshair made
    a top-quartile-among-D defenseman look "below average" against a line
    inflated by forwards. Each chart's dashed crosshair is therefore that
    position group's OWN average, matching NHL's per-position basis."""
    import altair as alt
    # Y axis is 20+ mph speed bursts per 60 (all-situations bursts ÷ all-
    # situations TOI) where available; the crosshair below auto-recomputes from
    # this same column, so the "average line" is the per-60 average. Playoffs
    # don't carry the per-60 column yet, so fall back to NHL's raw season total
    # there (keeps that scatter working rather than silently blanking).
    _y = ("EDGE Bursts/60" if "EDGE Bursts/60" in getattr(df, "columns", [])
          else "EDGE Bursts 20+")
    _y_title = ("Speed Bursts / 60 (20+ mph)" if _y == "EDGE Bursts/60"
                else "Speed Bursts (20+ mph, season total)")
    # The position group column is "position" on the team/trade/playoff frames
    # but renamed to "Pos" on the main leaderboard df — accept either.
    def _poscol(_f):
        return ("position" if "position" in getattr(_f, "columns", [])
                else ("Pos" if "Pos" in getattr(_f, "columns", []) else None))
    _pc = _poscol(df)
    if not {"EDGE Top Speed", _y}.issubset(df.columns) or _pc is None:
        return
    # Crosshair average MUST come from the true league-wide frame (per position
    # group), not the possibly team-filtered `df` this chart plots. league_df is
    # the explicit whole-league source (passed by team-scoped / trade callers);
    # domain_df is a fallback since the Trade Analyzer's domain_df is already the
    # unfiltered league frame; only falls back to `df` on the plain league-wide
    # leaderboard, where df already IS the league.
    _mean_src = (league_df if (league_df is not None and not league_df.empty)
                 else (domain_df if (domain_df is not None and not domain_df.empty) else df))
    _mpc = _poscol(_mean_src)
    # Shared axis domain across BOTH sub-charts so forwards vs defense are
    # directly comparable (the forward cloud visibly sits to the right).
    _dom = domain_df if (domain_df is not None and not domain_df.empty) else df
    for _grp, _lbl in (("F", "Forwards"), ("D", "Defense")):
        _sub = df[df[_pc] == _grp]
        if _sub.dropna(subset=["EDGE Top Speed", _y]).empty:
            continue
        _msrc = (_mean_src[_mean_src[_mpc] == _grp] if _mpc is not None else _mean_src)
        _mx = pd.to_numeric(_msrc.get("EDGE Top Speed"), errors="coerce").mean()
        _my = pd.to_numeric(_msrc.get(_y), errors="coerce").mean()
        _avg = []
        if pd.notna(_mx):
            _avg.append(alt.Chart(pd.DataFrame({"x": [float(_mx)]})).mark_rule(
                color=PALETTE["text_secondary"], strokeDash=[4, 4]).encode(x="x:Q"))
        if pd.notna(_my):
            _avg.append(alt.Chart(pd.DataFrame({"y": [float(_my)]})).mark_rule(
                color=PALETTE["text_secondary"], strokeDash=[4, 4]).encode(y="y:Q"))
        _extra = alt.layer(*_avg) if _avg else None
        # Own "EDGE: ..." heading PER position group (not one shared heading
        # above both) — a single heading above only the first (Forwards) chart
        # made the second (Defense) chart look untitled/orphaned below it.
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>{title_prefix} "
                    f"— {_lbl}</h4>", unsafe_allow_html=True)
        _yl = year_label
        _scatter_with_labels(
            _sub, "EDGE Top Speed", _y, "Top Speed (mph)",
            _y_title, f"EDGE-speed-burst-vs-top-speed-{_grp}{dl_suffix}",
            f"NHL EDGE tracking, one point per {_lbl.lower().rstrip('s')}. Dashed lines = "
            f"**{_lbl.lower()}** league average on each axis (F and D computed separately, "
            "matching NHL's per-position percentiles). Top-right = fast **and** frequent bursts.",
            team_scoped, extra_layer=_extra, highlight_name=highlight_name, domain_df=_dom,
            year_label=_yl)


def _nfi_xg_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "", highlight_name: str = None,
                    domain_df: pd.DataFrame = None, year_label: str = None) -> None:
    import altair as alt
    if not {"NFI%", "xG%"}.issubset(df.columns):
        return
    _scatter_with_labels(
        df, "xG%", "NFI%", "xG%", "NFI%",
        f"nfi-vs-xg-pct{dl_suffix}",
        "One point per player. Color = **OZ Start%** "
        "(light = easier, dark = harder).",
        team_scoped, color_col="OZ Start%",
        color_title="OZ Start% (light = easier, dark = harder)", highlight_name=highlight_name,
        domain_df=domain_df, year_label=year_label)


def _pdoxg_xgqg_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "", highlight_name: str = None,
                        domain_df: pd.DataFrame = None, year_label: str = None) -> None:
    """PDOxG (luck) vs xG-QG% (share of games clearing the xG floor) — does a
    player's quality-game rate come with over/under-shooting luck?"""
    import altair as alt
    if not {"PDOxG", "xG_QG_pct"}.issubset(df.columns):
        return
    rule0 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color=PALETTE["text_secondary"], strokeDash=[4, 4]).encode(y="y:Q")
    _scatter_with_labels(
        df, "xG_QG_pct", "PDOxG", "xG-QG%", "PDOxG (luck net of shot quality)",
        f"pdoxg-vs-xgqg-pct{dl_suffix}",
        "Descriptive, not a ranking — see Methodology. Color = **OZ Start%** "
        "(light = easier, dark = harder).",
        team_scoped, extra_layer=rule0, color_col="OZ Start%",
        color_title="OZ Start% (light = easier, dark = harder)", highlight_name=highlight_name,
        domain_df=domain_df, year_label=year_label)


def _pdoxg_nfiqg_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "", highlight_name: str = None,
                         domain_df: pd.DataFrame = None, year_label: str = None) -> None:
    """PDOxG (luck) vs NFI-QG% (share of games clearing the net-front-impact
    floor)."""
    import altair as alt
    if not {"PDOxG", "NFI_QG_pct"}.issubset(df.columns):
        return
    rule0 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color=PALETTE["text_secondary"], strokeDash=[4, 4]).encode(y="y:Q")
    _scatter_with_labels(
        df, "NFI_QG_pct", "PDOxG", "NFI-QG%", "PDOxG (luck net of shot quality)",
        f"pdoxg-vs-nfiqg-pct{dl_suffix}",
        "Descriptive, not a ranking — see Methodology. Color = **OZ Start%** "
        "(light = easier, dark = harder).",
        team_scoped, extra_layer=rule0, color_col="OZ Start%",
        color_title="OZ Start% (light = easier, dark = harder)", highlight_name=highlight_name,
        domain_df=domain_df, year_label=year_label)


def _xgqg_nfiqg_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "", highlight_name: str = None,
                        domain_df: pd.DataFrame = None, year_label: str = None) -> None:
    """xG-QG% vs NFI-QG% — do the two quality-game rates (shot quality vs net-
    front impact) agree for a player? 50/50 crosshair = an average qualifier
    on both."""
    import altair as alt
    if not {"xG_QG_pct", "NFI_QG_pct"}.issubset(df.columns):
        return
    # xG_QG_pct/NFI_QG_pct are stored as 0-1 fractions (not 0-100), so the 50/50
    # crosshair must be 0.5/0.5 — a literal 50 here previously blew the shared-scale
    # domain resolution out to ~1.25B pixels (data range ~0.1-0.9 unioned with a
    # value of 50), which silently failed to render at all.
    rule_v = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(
        color=PALETTE["text_secondary"], strokeDash=[4, 4]).encode(x="x:Q")
    rule_h = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
        color=PALETTE["text_secondary"], strokeDash=[4, 4]).encode(y="y:Q")
    _scatter_with_labels(
        df, "xG_QG_pct", "NFI_QG_pct", "xG-QG%", "NFI-QG%",
        f"nfiqg-vs-xgqg-pct{dl_suffix}",
        "One point per player — how often each clears the xG vs net-front-impact "
        "quality-game floor. Color = **OZ Start%** (light = easier, dark = harder).",
        team_scoped, extra_layer=alt.layer(rule_v, rule_h), color_col="OZ Start%",
        color_title="OZ Start% (light = easier, dark = harder)", highlight_name=highlight_name,
        domain_df=domain_df, year_label=year_label)


@st.cache_data(show_spinner=False, ttl=3600)
def load_box_score() -> pd.DataFrame:
    """NHL Stats API box score (build_box_score.py) + PBP supplement
    (build_box_score_pbp.py: A1/A2, hits_taken, rebounds_created, FO W/L),
    merged per (player_id, season, game_type)."""
    fp = REPO_ROOT / "Data" / "box_score_skaters.csv"
    if not fp.exists():
        return pd.DataFrame()
    d = pd.read_csv(fp, dtype={"season": str}).drop_duplicates()
    pbp_fp = REPO_ROOT / "Data" / "box_score_pbp.csv"
    if pbp_fp.exists():
        p = pd.read_csv(pbp_fp, dtype={"season": str})
        d = d.merge(p, on=["player_id", "season", "game_type"], how="left")
    return d


# Box-score count columns (summed across pooled scopes); rates recomputed after.
_BOX_SUMS = ["GP", "goals", "assists", "A1", "A2", "points", "shots", "ppGoals",
             "ppPoints", "shPoints", "hits", "hits_taken", "blockedShots",
             "takeaways", "giveaways", "minorPenalties", "majorPenalties",
             "penaltyMinutes", "penaltiesDrawn", "rebounds_created",
             "faceoffs_won", "faceoffs_lost", "gameWinningGoals"]

# Box-score display columns (a metric family on the Player List, not a tab).
_BOX_REN = {"goals": "G", "assists": "A", "points": "Pts", "shots": "Sh",
            "ppPoints": "PPP", "shPoints": "SHP", "hits": "Hits",
            "hits_taken": "Hits Taken", "blockedShots": "Blocks",
            "takeaways": "TK", "giveaways": "GV", "minorPenalties": "Min Pen",
            "majorPenalties": "Maj Pen", "penaltyMinutes": "PIM",
            "penaltiesDrawn": "Pen Drawn", "rebounds_created": "Reb Created",
            "faceoffs_won": "FO W", "faceoffs_lost": "FO L",
            "gameWinningGoals": "GWG"}


def _box_score_scope(scope_key: str, playoffs: bool = False) -> pd.DataFrame:
    """Per-player box-score counting stats for one scope (counts summed across
    the scope's seasons; Sh%/FO% recomputed from the sums). Keyed player_id."""
    d = load_box_score()
    if d.empty:
        return pd.DataFrame()
    d = d[d["game_type"] == ("playoff" if playoffs else "regular")]
    seasons = _situation_seasons(scope_key, playoffs)
    if seasons is not None:
        d = d[d["season"].isin(seasons)]
    if d.empty:
        return pd.DataFrame()
    # GP is excluded — the player frame already has its own GP column.
    sums = {c: (c, "sum") for c in _BOX_SUMS if c in d.columns and c != "GP"}
    agg = d.groupby("player_id").agg(**sums).reset_index()
    agg = agg.rename(columns=_BOX_REN)
    if {"G", "Sh"}.issubset(agg.columns):
        agg["Sh%"] = np.where(agg["Sh"] > 0, agg["G"] / agg["Sh"] * 100, np.nan)
    if {"FO W", "FO L"}.issubset(agg.columns):
        _d = agg["FO W"] + agg["FO L"]
        agg["FO%"] = np.where(_d > 0, agg["FO W"] / _d * 100, np.nan)
    keep = ["player_id"] + [c for c in BOX_FAMILY_COLS if c in agg.columns]
    return agg[keep]


def _unused_render_box_score() -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Box Score</h2>",
        unsafe_allow_html=True)
    season_label, game_type = render_scoped_filters("box")
    if _block_ref_only(season_label):
        return
    _set_dl_title(None)
    playoffs = game_type == "Playoffs"
    d = load_box_score()
    if d.empty:
        st.error("Box-score data not found (`Data/box_score_skaters.csv`). "
                 "Run `NFI/scripts/build_box_score.py` + `build_box_score_pbp.py`.")
        return
    seasons = _shot_chart_seasons(season_label, playoffs)      # None = all
    gt = "playoff" if playoffs else "regular"
    d = d[d["game_type"] == gt]
    if seasons is not None:
        d = d[d["season"].isin(seasons)]
    if d.empty:
        st.info("No box-score data for this scope.")
        return

    # aggregate across the scope's seasons (sum counts) — one row per player
    sums = {c: (c, "sum") for c in _BOX_SUMS if c in d.columns}
    agg = d.groupby("player_id").agg(
        player_name=("player_name", "last"), position=("position", "last"),
        team=("team", "last"), toi_sec=("toi_per_game_sec", "mean"), **sums
    ).reset_index()
    # recompute rate stats from summed counts
    agg["Sh%"] = np.where(agg["shots"] > 0, agg["goals"] / agg["shots"] * 100, np.nan)
    _fo = agg.get("faceoffs_won", 0) + agg.get("faceoffs_lost", 0)
    agg["FO%"] = np.where(_fo > 0, agg.get("faceoffs_won", 0) / _fo * 100, np.nan)
    agg["TOI/GP"] = agg["toi_sec"] / 60.0
    agg["team"] = agg["team"].astype(str).str.split(",").str[-1]   # most recent team

    c1, c2, c3 = st.columns([1.5, 1.0, 1.3])
    with c1:
        _psel = st.selectbox("Search a player", sorted(agg["player_name"].dropna().unique()),
                             index=None, placeholder="", key="box_search")
    with c2:
        pos = st.radio("Position", ["All", "F", "D"], horizontal=True, key="box_pos")
    with c3:
        min_gp = st.slider("Min GP", 0, 82, 20, 1, key="box_mingp")
    if pos == "F":
        agg = agg[agg["position"].isin(["C", "L", "R"])]
    elif pos == "D":
        agg = agg[agg["position"] == "D"]
    agg = agg[agg["GP"].fillna(0) >= min_gp]
    if _psel:
        agg = agg[agg["player_name"] == _psel]
    if agg.empty:
        st.info("No players match the filters.")
        return

    ren = {"player_name": "Player", "position": "Pos", "team": "Team", "goals": "G",
           "assists": "A", "points": "Pts", "shots": "Sh", "ppPoints": "PPP",
           "shPoints": "SHP", "hits": "Hits", "hits_taken": "Hits Taken",
           "blockedShots": "Blocks", "takeaways": "TK", "giveaways": "GV",
           "minorPenalties": "Min Pen", "majorPenalties": "Maj Pen",
           "penaltyMinutes": "PIM", "penaltiesDrawn": "Pen Drawn",
           "rebounds_created": "Reb Created", "faceoffs_won": "FO W",
           "faceoffs_lost": "FO L", "gameWinningGoals": "GWG"}
    agg = agg.rename(columns=ren)
    cols = ["Player", "Pos", "Team", "GP", "TOI/GP",
            "G", "A1", "A2", "A", "Pts", "PPP", "SHP", "Sh", "Sh%", "GWG",
            "Reb Created", "TK", "GV",
            "Hits", "Hits Taken", "Blocks", "Min Pen", "Maj Pen", "PIM", "Pen Drawn",
            "FO W", "FO L", "FO%"]
    cols = [c for c in cols if c in agg.columns]
    disp = agg.sort_values("Pts", ascending=False)[cols].reset_index(drop=True)
    fmt = {}
    for c in cols:
        if c in ("Sh%", "FO%"):
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}%"
        elif c == "TOI/GP":
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
        elif c not in ("Player", "Pos", "Team"):
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    st.caption(f"{len(disp)} skaters · {season_label} · {gt} · sorted by Points. "
               "Source: NHL Stats API; A1/A2, Hits Taken, Reb Created from play-by-play "
               "(2022-26 only). Cap hit coming soon.")
    _show_df(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)


def render_players() -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Player List</h2>",
        unsafe_allow_html=True,
    )
    season_label, game_type = render_scoped_filters("players", show_situation=True)
    if _block_ref_only(season_label):
        return
    _set_dl_title(None)                    # only drill-in charts get a name
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

    # Multi-team (traded) players: list each player's teams in the order they
    # played them that season (chronological by game), shown as "EDM / COL".
    # Single-season only (teams_in_season comes from the QG per-season build).
    _has_teams = "teams_in_season" in frame.columns
    if _has_teams:
        frame = frame.copy()
        _order = load_player_season_team_order()
        _ssn = SEASON_KEY.get(season_label)

        def _team_list(r):
            pid = r.get("player_id")
            if pd.notna(pid) and _ssn:
                teams = _order.get((int(pid), _ssn))
                if teams:
                    return teams
            ts = r.get("teams_in_season")   # fallback: QG's alphabetical set
            if isinstance(ts, str) and ts.strip():
                return [t.strip() for t in ts.split(",") if t.strip()]
            t = r.get("team")
            return [t] if isinstance(t, str) and t else []

        frame["_teams"] = frame.apply(_team_list, axis=1)
        frame["team"] = frame["_teams"].apply(lambda ts: " / ".join(ts) if ts else None)
        _all_teams = sorted({t for ts in frame["_teams"] for t in ts})
    else:
        _all_teams = sorted(frame["team"].dropna().unique().tolist())

    # Player search options (drill into one player's detail on this same page).
    _popts = (frame[["player_id", "player_name", "position"]]
              .dropna(subset=["player_id"]).drop_duplicates("player_id")
              .sort_values("player_name"))
    _pid_list = [int(x) for x in _popts["player_id"].tolist()]
    _plabel = {int(r.player_id): f"{r.player_name} ({r.position})"
               for r in _popts.itertuples()}

    # Default position for the Min ES TOI slider. The slider itself IS the
    # ranking floor — players below its current value are unranked "(UR)";
    # move it down to bring more players into the ranked cohort.
    rank_floor = 300 if playoffs else (2000 if is_pooled else 500)
    c1, c2, c3 = st.columns([1.5, 1.0, 1.3])
    with c1:
        player_sel = st.selectbox(
            "Search a player", _pid_list, index=None,
            placeholder="",
            format_func=lambda i: _plabel.get(i, str(i)), key="players_search",
            on_change=lambda: st.session_state.update(players_team="All"),
            help="Pick a player to see their season-by-season detail on this page.")
    with c2:
        pos = st.radio("Position", ["All", "F", "D"], horizontal=True, key="players_pos")
    with c3:
        team_opts = ["All"] + _all_teams
        # Picking a team exits any drill-in and clears the player search (the two
        # are mutually exclusive views).
        team_sel = st.selectbox(
            "Team", team_opts, key="players_team",
            on_change=lambda: st.session_state.update(_pl_drill=None, players_search=None))

    # Metric-family toggles first, then the Team filter. Families start with none
    # selected (only the identity columns show); click a family to display it.
    fcol, tcol, gcol = st.columns([2.4, 0.85, 0.85])
    # Quality Games shows by default; the user can toggle other families on/off.
    st.session_state.setdefault("players_display_seg", ["Quality Games"])
    with fcol:
        # Pills (not segmented_control): they wrap onto multiple lines and are more
        # reliable to tap on mobile than a connected button group.
        display_fams = st.pills(
            "**Display a Metric Family**", list(PLAYER_FAMILY_COLS),
            selection_mode="multi", key="players_display_seg",
            help="Tap a metric group to show its columns (Net Front Impact, "
                 "Zone Impact, Quality Games). Tap again to hide it.") or []
        if "Zone Impact" not in display_fams and "EDGE" not in display_fams:
            st.caption("↑ Raw DZ/NZ/OZ Start% values only appear in the table below "
                       "once **Zone Impact** (or **EDGE**) is tapped on.")
    if "EDGE" in display_fams:
        st.session_state.setdefault("players_edge_scope", "Even Strength")
        st.radio("EDGE OZ% scope", list(_EDGE_OZ_SCOPE_COL.keys()), horizontal=True,
                 key="players_edge_scope",
                 help="Scope for EDGE OZ% (and EZI) only — NZ%/DZ% always show their "
                      "one available (all-situations) number; NHL doesn't publish an "
                      "even-strength split for those two. Defaults to Even Strength to "
                      "match the rest of the page (all 5v5).")
        st.caption("The scope toggle applies only to **EDGE OZ%** and **EZI** — NZ%/DZ% "
                   "have no even-strength variant from NHL, so they're unaffected.")
    if "xG" in display_fams:
        st.session_state.setdefault("players_pdo_scope", "5v5")
        st.radio("PDO shot scope", list(PDO_SCOPE_FILE.keys()), horizontal=True,
                 key="players_pdo_scope",
                 help="Shot scope for PDO only — every other xG-group column stays "
                      "as-is regardless.")
        st.caption("The shot-scope toggle applies only to **PDO** — other xG columns "
                   "are unaffected.")
    if "xG" in display_fams:
        st.caption("xG / possession columns follow the **Situation** filter above (next "
                   "to Game type). PP=5v4+5v3+4v3, PK=4v5+3v5+3v4. NFI/QG/Zone stay 5v5.")
    with tcol:
        if playoffs:
            min_toi = st.slider("Min ES TOI (min)", 0, 1500, rank_floor, 25,
                                key="players_toi_playoffs")
        else:
            toi_key = "players_toi_pooled" if is_pooled else "players_toi_season"
            min_toi = st.slider("Min ES TOI (min)", 0, 7500, rank_floor, 50, key=toi_key)
    with gcol:
        # Min GP — a SEPARATE, adjustable floor from Min ES TOI, so a player who's
        # off the leaderboard purely on games played (rather than low per-game
        # minutes) can be brought on. Doesn't change the fixed rank_floor used for
        # ranking eligibility (still shows "(UR)" below that), only which rows show.
        _is_2yr_scope = (not playoffs) and SEASON_KEY.get(season_label) == "pooled_2yr"
        _gpmax = 30 if playoffs else (350 if is_pooled else (170 if _is_2yr_scope else 82))
        _gpkey = (f"players_mingp_{'playoffs' if playoffs else ('pooled' if is_pooled else ('2yr' if _is_2yr_scope else 'season'))}")
        min_gp = st.slider("Min GP", 0, _gpmax, min(25, _gpmax), 1, key=_gpkey)

    # Drill-in (via the search box OR clicking a leaderboard row): show one
    # player's detail (trend + charts) here. Clear/deselect to return to the list.
    def _drill(pid):
        _pname = _plabel.get(int(pid), str(pid))
        _set_dl_title(_pname)                     # name downloaded charts
        st.markdown(f"### {_pname}")
        if playoffs:
            _render_player_playoff_summary(frame, int(pid))
        else:
            _prow = _popts[_popts["player_id"] == int(pid)]
            _is_d = len(_prow) and str(_prow["position"].iloc[0]) == "D"
            _pos_label = "Defense only" if _is_d else "Forwards only"
            _rc = st.radio("Rank against", ["All skaters", _pos_label],
                           horizontal=True, key="players_rank_cohort")
            # Drill-in follows the metric-family filter (the pills above): only the
            # selected families' columns/charts show. Clear all pills to see every
            # family.
            _render_player_profile(int(pid), same_pos=(_rc != "All skaters"),
                                   families=display_fams, team="__own__",
                                   season_label=season_label)

    # Drill via the search box OR a clicked leaderboard row — either one collapses
    # the leaderboard to just that player's detail.
    if player_sel is not None:
        st.session_state["_pl_drill"] = None     # an explicit search overrides a click
        # Back button clears the search box (a selected selectbox value is a chip,
        # not free text, so this is the clean way to reset it). Use a callback so
        # the widget's state can be modified before the next run instantiates it.
        st.button("← Back to leaderboard", key="pl_back_search",
                  on_click=lambda: st.session_state.update(players_search=None))
        _drill(player_sel)
        return
    _drill_pid = st.session_state.get("_pl_drill")
    if _drill_pid is not None:
        if st.button("← Back to leaderboard", key="pl_back"):
            st.session_state["_pl_drill"] = None
            st.rerun()
        _drill(int(_drill_pid))
        return

    df = frame.copy()
    if pos in ("F", "D"):
        df = df[df["position"] == pos]
    else:
        df = df[df["position"].isin(["F", "D"])]
    # Ranking denominator is the position cohort clearing the Min ES TOI / Min GP
    # sliders — the sliders ARE the qualifying floor, so moving them changes who
    # ranks, not just who's visible.
    rank_cohort = df[(df["toi_min"].fillna(0) >= min_toi)
                     & (df["GP"].fillna(0) >= min_gp)].copy()
    df = df[df["toi_min"].fillna(0) >= min_toi]
    df = df[df["GP"].fillna(0) >= min_gp]
    if team_sel != "All":
        if _has_teams:
            df = df[df["_teams"].apply(lambda ts: team_sel in ts)]
        else:
            df = df[df["team"] == team_sel]
    if df.empty:
        st.markdown(
            f"<p style='color:{PALETTE['text']};'>No players match the current filters. "
            "Widen Position or Team, or change the Season in the sidebar.</p>",
            unsafe_allow_html=True,
        )
        return

    # Qualified players (≥ slider floor) lead, sorted by RelNFI%; sub-floor (UR)
    # players follow — so low-TOI noise can't dominate the top of the leaderboard.
    df["_qual"] = (df["toi_min"].fillna(0) >= min_toi) & (df["GP"].fillna(0) >= min_gp)
    df = df.sort_values(["_qual", "RelNFI_pct"], ascending=[False, False],
                        na_position="last").reset_index(drop=True)
    # Storage → display: RelNFI_F (attack / for) shows as "RelNFI-A%",
    # RelNFI_A (suppress / against) shows as "RelNFI-S%". Do NOT sign-flip — the
    # underlying _F/_A columns are unchanged; only the display labels swap A/S.
    _ren = {
        "player_name": "Player", "position": "Pos", "team": "Team", "toi_min": "TOI",
        "NFI_pct": "NFI%", "RelNFI_pct": "RelNFI%",
        "RelNFI_F_pct": "RelNFI-A%", "RelNFI_A_pct": "RelNFI-S%",
        "NFI_A_rate": "NFI-A/60", "NFI_S_rate": "NFI-S/60",
        "xG_QG_pct": "xG-QG%", "NFI_QG_pct": "NFI-QG%",
        "RelNFI_QG_pct": "RelNFI-QG%", "RelxG_QG_pct": "RelxG-QG%",
        "RelxG_pct": "RelxG%", "RelxG_F_pct": "RelxG-F%", "RelxG_A_pct": "RelxG-A%",
        **_EDGE_REN,
    }
    df = df.rename(columns=_ren)
    rank_cohort = rank_cohort.rename(columns=_ren)

    # ES minutes per game — a "how heavily are they used" context column beside
    # the cumulative TOI. GP is games played; TOI is the ES 5v5 total.
    if {"TOI", "GP"}.issubset(df.columns):
        _gp = pd.to_numeric(df["GP"], errors="coerce")
        df["TOI/GP"] = np.where(_gp > 0,
                                pd.to_numeric(df["TOI"], errors="coerce") / _gp, np.nan)

    # Always show the full column set (Compact view removed; Qual GP dropped).
    # NFI-A/60 / NFI-S/60 are RAW per-60 rates; RelNFI-A% / RelNFI-S% are the
    # relative (vs own-team) versions — both coexist, placed side by side.
    cols = ["Player", "Pos", "Team", "GP", "TOI", "TOI/GP",
            "xG-QG%", "xG-QG-F%", "xG-QG-A%", "RelxG-QG%",
            "NFI-QG%", "NFI-QG-A%", "NFI-QG-S%", "RelNFI-QG%",
            "xGF/60", "xGA/60", "xG%", "RelxG%", "RelxG-F%", "RelxG-A%", "PDO", "PDOxG",
            "RelNFI%", "RelNFI-A%", "RelNFI-S%", "NFI%", "NFI-A/60", "NFI-S/60",
            "DZ Start%", "NZ Start%", "OZ Start%", "OZI", "DZI", "NZI", "TZI",
            *_EDGE_VALUE_DISP, *_SIT_PLAIN, *BOX_FAMILY_COLS]
    # Zone now populates for single seasons too (per-season files), so it is no
    # longer stripped; the in-frame filter below drops it only if truly absent.
    cols = [c for c in cols if c in df.columns]
    # Metric-family display: identity columns (no family) always show; a family's
    # columns show only when that family is selected. Nothing selected = identity
    # columns only. A column may belong to more than one family (e.g. D/N/O
    # Start% under both Zone Impact and EDGE) — show it if ANY selected family
    # claims it.
    _fam_of = {}
    for _fam, _fcols in PLAYER_FAMILY_COLS.items():
        for _col in _fcols:
            _fam_of.setdefault(_col, set()).add(_fam)
    _shown = set(display_fams)
    cols = [c for c in cols if not _fam_of.get(c) or _fam_of[c] & _shown]
    disp = df[cols].copy()

    fmt = {}
    for c in ("NFI%", "xG-QG%", "NFI-QG%", "RelNFI-QG%", "RelxG-QG%",
              "xG-QG-F%", "xG-QG-A%", "NFI-QG-A%", "NFI-QG-S%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("RelNFI%", "RelNFI-A%", "RelNFI-S%", "RelxG%", "RelxG-F%", "RelxG-A%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:+.2f}"
    for c in ("NFI-A/60", "NFI-S/60"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    for c in ("xGF/60", "xGA/60"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    if "xG%" in disp.columns:
        fmt["xG%"] = lambda x: "—" if pd.isna(x) else f"{x:.1f}%"
    if "PDO" in disp.columns:
        fmt["PDO"] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    if "PDOxG" in disp.columns:
        fmt["PDOxG"] = lambda x: "—" if pd.isna(x) else f"{x:+.1f}"
    for c in ("OZI", "DZI", "NZI", "TZI"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    for c in ("OZ Start%", "DZ Start%", "NZ Start%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}%"
    for c in ("EDGE OZ%", "EDGE OZ% (EV)", "EDGE NZ%", "EDGE DZ%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x*100:.1f}%"
    if "EDGE Top Speed" in disp.columns:
        fmt["EDGE Top Speed"] = lambda x: "—" if pd.isna(x) else f"{x:.1f} mph"
    if "EDGE Bursts 20+" in disp.columns:
        fmt["EDGE Bursts 20+"] = lambda x: "—" if pd.isna(x) else f"{x:.0f}"
    if "EDGE Distance (mi)" in disp.columns:
        fmt["EDGE Distance (mi)"] = lambda x: "—" if pd.isna(x) else f"{x:.1f} mi"
    if "EDGE Distance/60" in disp.columns:
        fmt["EDGE Distance/60"] = lambda x: "—" if pd.isna(x) else f"{x:.2f} mi/60"
    if "EDGE Bursts/60" in disp.columns:
        fmt["EDGE Bursts/60"] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    if "EZI" in disp.columns:
        fmt["EZI"] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    if "TOI" in disp.columns:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    if "TOI/GP" in disp.columns:
        fmt["TOI/GP"] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    # Situation-driven suite formatters (plain names after unification).
    for c in ("CF/60", "CA/60", "FF/60", "FA/60", "GF/60", "GA/60",
              "iCF/60", "iG/60", "PP Value", "PK Value"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    for c in ("ixG/60", "ixG"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    for c in ("CF%", "FF%", "GF%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}%"
    for c in ("RelCF%", "RelxGF%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:+.1f}"
    if "iG" in disp.columns:
        fmt["iG"] = lambda x: "—" if pd.isna(x) else f"{x:.0f}"
    # Box-score counting stats: integers, except the two percentages.
    for c in BOX_FAMILY_COLS:
        if c in disp.columns:
            fmt[c] = ((lambda x: "—" if pd.isna(x) else f"{x:.1f}%")
                      if c in ("Sh%", "FO%")
                      else (lambda x: "—" if pd.isna(x) else f"{x:,.0f}"))
    for c in ("GP",):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    _player_rank = ["NFI%", "RelNFI%", "RelNFI-A%", "RelNFI-S%", "NFI-A/60",
                    "NFI-S/60", "OZI", "DZI", "NZI", "TZI",
                    "DZ Start%", "NZ Start%", "OZ Start%",
                    "RelNFI-QG%", "NFI-QG%", "RelxG%", "RelxG-QG%", "xG-QG%",
                    "xGF/60", "xGA/60", "RelxG-F%", "RelxG-A%",
                    "xG-QG-F%", "xG-QG-A%", "NFI-QG-A%", "NFI-QG-S%",
                    *_EDGE_VALUE_DISP]
    # lower value = better (rank ascending): shots/xG against
    _lower = {"NFI-S/60", "xGA/60", "RelxG-A%"}
    # Second bracket number = within-team rank. With a team selected, rank within
    # that team; otherwise within each player's own (most-recent) team. Computed
    # per-row over the qualified cohort and keyed to the displayed rows by id.
    _team_rank_idx = {}
    _df_pid = dict(zip(df.index, df["player_id"]))
    if team_sel != "All":
        _tc = (rank_cohort[rank_cohort["_teams"].apply(lambda ts: team_sel in ts)]
               if _has_teams else rank_cohort[rank_cohort["Team"] == team_sel])
        for col in _player_rank:
            if col not in _tc.columns:
                continue
            tr = pd.to_numeric(_tc[col], errors="coerce").rank(
                ascending=(col in _lower), method="min")
            p2t = dict(zip(_tc["player_id"], tr))
            _team_rank_idx[col] = {i: int(p2t[p]) for i, p in _df_pid.items()
                                   if pd.notna(p2t.get(p))}
    else:
        _coh_team = (rank_cohort["_teams"].apply(lambda ts: ts[-1] if isinstance(ts, list) and ts else None)
                     if _has_teams else rank_cohort.get("Team"))
        if _coh_team is not None:
            for col in _player_rank:
                if col not in rank_cohort.columns:
                    continue
                tr = pd.to_numeric(rank_cohort[col], errors="coerce").groupby(
                    _coh_team).rank(ascending=(col in _lower), method="min")
                p2t = dict(zip(rank_cohort["player_id"], tr))
                _team_rank_idx[col] = {i: int(p2t[p]) for i, p in _df_pid.items()
                                       if pd.notna(p2t.get(p))}
    _pl_qual = ((pd.to_numeric(disp["TOI"], errors="coerce").fillna(0) >= min_toi)
               & (pd.to_numeric(disp["GP"], errors="coerce").fillna(0) >= min_gp))
    _apply_ranks(disp, fmt, rank_cohort, _player_rank, lower_better=_lower,
                 mark_unranked=True, qualified=_pl_qual, team_rank_idx=_team_rank_idx)
    _lb_cohort_txt = {"F": "forwards", "D": "defense"}.get(pos, "all skaters")
    st.caption(f"Each value shows **(league / team)** rank — rank among "
               f"**{_lb_cohort_txt}** league-wide, then among their own team's skaters "
               "that season. NFI-S/60 (shots against): lowest = #1.")
    _sort_hint()
    st.caption("Click a row to open that player's detail (collapses the list).")
    _gen = st.session_state.get("_pl_tbl_gen", 0)
    _event = _show_df(disp.style.format(fmt, na_rep="—"), hide_index=True,
                      on_select="rerun", selection_mode="single-row",
                      key=f"players_tbl_{_gen}")

    if playoffs:
        zone_note = " · OZI/DZI/NZI/TZI (0–100, 50 = avg) pooled across playoffs"
    elif SEASON_KEY.get(season_label) == "pooled_2yr":
        zone_note = (" · OZI/DZI/NZI/TZI (0–100, 50 = avg) and D/N/O Start% "
                     "pooled 2024-25 + 2025-26")
    elif is_pooled:
        zone_note = (" · OZI/DZI/NZI/TZI (0–100, 50 = avg) and D/N/O Start% "
                     "pooled across all seasons")
    else:
        zone_note = (" · OZI/DZI/NZI/TZI (0–100, 50 = avg) and D/N/O Start% "
                     "for this season")
    st.caption(
        f"{len(disp):,} players (≥ {min_toi:,} ES min, ≥ {min_gp} GP) · {scope_label} · "
        f"sorted by RelNFI% descending · ranked at ≥ {min_toi:,} ES min / ≥ {min_gp} GP "
        f"(else UR){zone_note} · the Min GP / Min ES TOI sliders set the ranking floor "
        "directly — lower them to rank more players, raise them to tighten the cohort"
    )

    import altair as alt
    _team_scoped = team_sel != "All"
    # OZ Start% (the scatter color source) only exists on a pooled season, so
    # a team filter applied while a single season is selected silently lost
    # its color — same fix as the drill-in's team-scatters getting their own
    # always-pooled frame. Backfill it here from the pooled frame (color-only
    # use; doesn't touch anything else this season's df carries).
    if "OZ Start%" not in df.columns:
        _oz_backfill, _ = _build_players_frame("4yr (2022-2026)")
        if not _oz_backfill.empty and "OZ Start%" in _oz_backfill.columns:
            df = df.merge(_oz_backfill[["player_id", "OZ Start%"]], on="player_id", how="left")
    # Scatter/bar/distribution charts below use `df` (merged, renamed, but NOT
    # narrowed by the family-pill column filter that `disp` went through) so
    # they always show whenever their underlying data exists — regardless of
    # which metric-family pills are toggled. Only genuine data-availability
    # gates remain (e.g. is_pooled for Start%, since that data has no
    # per-season cut).
    if {"PDOxG", "xG_QG_pct"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>PDOxG vs xG-QG%</h4>",
                    unsafe_allow_html=True)
        _pdoxg_xgqg_scatter(df, _team_scoped, year_label=scope_label)

    if {"PDOxG", "NFI_QG_pct"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>PDOxG vs NFI-QG%</h4>",
                    unsafe_allow_html=True)
        _pdoxg_nfiqg_scatter(df, _team_scoped, year_label=scope_label)

    if {"xG_QG_pct", "NFI_QG_pct"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>xG-QG% vs NFI-QG%</h4>",
                    unsafe_allow_html=True)
        _xgqg_nfiqg_scatter(df, _team_scoped, year_label=scope_label)

    if {"PDOxG", "xG%"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>PDOxG vs xG%</h4>",
                    unsafe_allow_html=True)
        _pdo_xg_scatter(df, _team_scoped, year_label=scope_label)

    if {"PDOxG", "NFI%"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>PDOxG vs NFI%</h4>",
                    unsafe_allow_html=True)
        _pdo_nfi_scatter(df, _team_scoped, year_label=scope_label)

    if {"NFI%", "xG%"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>NFI% vs xG%</h4>",
                    unsafe_allow_html=True)
        _nfi_xg_scatter(df, _team_scoped, year_label=scope_label)

    if {"EDGE DZ%", "EDGE OZ%"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: D-Zone vs "
                    f"O-Zone Time%</h4>", unsafe_allow_html=True)
        _edge_zone_scatter(df, _team_scoped, year_label=scope_label)

    if {"DZ Start%", "OZ Start%"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>Zone Starts: D-Zone vs "
                    f"O-Zone</h4>", unsafe_allow_html=True)
        _zone_start_scatter(df, _team_scoped, year_label=scope_label)

    if {"EDGE OZ%", "DZ Start%", "NZ Start%"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EZI: EDGE O-Zone "
                    f"Time vs Non-O-Zone Starts</h4>", unsafe_allow_html=True)
        _ezi_scatter(df, _team_scoped, year_label=scope_label)

    if {"EDGE Top Speed", "EDGE Bursts 20+"}.issubset(df.columns):
        _edge_speed_scatter(df, _team_scoped)

    # Row click → drill into that player (collapse the list). Bump the table key
    # so the leaderboard re-renders without a stale selection when we come back.
    _sel_rows = getattr(getattr(_event, "selection", None), "rows", None)
    if _sel_rows:
        st.session_state["_pl_drill"] = int(df.iloc[_sel_rows[0]]["player_id"])
        st.session_state["_pl_tbl_gen"] = _gen + 1
        st.rerun()


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

    # Quality Games metrics first, then NFI family, Zone Impact, and TOI.
    items = [
        ("NFI-QG%", _f("NFI_QG_pct", "pct")),
        ("RelNFI-QG%", _f("RelNFI_QG_pct", "pct")),
        ("xG-QG%", _f("xG_QG_pct", "pct")),
        ("RelxG-QG%", _f("RelxG_QG_pct", "pct")),
        ("RelxG%", _rel("RelxG_pct")),
        ("RelNFI%", _rel("RelNFI_pct")),
        ("RelNFI-A%", _rel("RelNFI_F_pct")),
        ("RelNFI-S%", _rel("RelNFI_A_pct")),
        ("NFI%", _f("NFI_pct", "pct")),
        ("NFI-A/60", _f("NFI_A_rate", "rate")),
        ("NFI-S/60", _f("NFI_S_rate", "rate")),
        ("OZI", _f("OZI", "rate")),
        ("DZI", _f("DZI", "rate")),
        ("NZI", _f("NZI", "rate")),
        ("TZI", _f("TZI", "rate")),
        ("ES TOI (min)", _f("toi_min", "toi")),
    ]
    st.caption(f"**{r['player_name']} ({r['position']})** · pooled playoffs.")
    # Horizontal layout: metrics as columns, a single value row (matches the rest
    # of the app), rather than a tall two-column Metric/Value table.
    hdf = pd.DataFrame([{m: v for m, v in items}])[[m for m, _ in items]]
    _show_df(hdf, width="stretch", hide_index=True)

    # Combined NFI+xG Quality-Games bar (Raw+Rel paired) and the Zone Impact bar
    # — same charts as the regular-season drill-in, built from this player's
    # pooled playoff row, both on the shared fixed 30–75 y-axis.
    qg_vals = {m: (float(r[_P2YR_MAP[m]]) * 100
                   if _P2YR_MAP.get(m) in r.index and pd.notna(r.get(_P2YR_MAP[m]))
                   else np.nan)
               for m in _QG_BAR_METRICS}
    if any(pd.notna(v) for v in qg_vals.values()):
        _qg_paired_bar_chart(
            qg_vals, "Playoffs",
            caption="Quality Games % vs **50**.",
            dl_prefix="QG-bars-playoffs")
    zone_vals = {m: (float(r[m]) if m in r.index and pd.notna(r.get(m)) else np.nan)
                 for m in _ZONE_BAR_METRICS}
    if any(pd.notna(v) for v in zone_vals.values()):
        _qg_bar_chart(zone_vals, "Playoffs",
                     caption="Zone Impact index vs **50** (league average).",
                     dl_prefix="Zone-bars-playoffs", ydomain=_PROFILE_BAR_YDOM,
                     title="Zone Impact Index")

    # Team scatters, scoped to this player's playoff team. Zone-start% still
    # isn't computed for playoffs, so only 4 of the 5 regular-season scatters
    # are buildable here.
    _team = r.get("team")
    if isinstance(_team, str) and _team and "team" in frame.columns:
        _tf = (frame[frame["team"] == _team]
               .rename(columns={"player_name": "Player", "NFI_pct": "NFI%"}))
        _hi = r.get("player_name")
        if {"Player", "PDOxG", "xG%"}.issubset(_tf.columns):
            st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>"
                        f"{_team} PDOxG vs xG% (playoffs)</h4>", unsafe_allow_html=True)
            _pdo_xg_scatter(_tf, True, dl_suffix="-playoffs", highlight_name=_hi,
                          year_label="Playoffs")
        if {"Player", "PDOxG", "NFI%"}.issubset(_tf.columns):
            st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>"
                        f"{_team} PDOxG vs NFI% (playoffs)</h4>", unsafe_allow_html=True)
            _pdo_nfi_scatter(_tf, True, dl_suffix="-playoffs", highlight_name=_hi,
                           year_label="Playoffs")
        if {"Player", "NFI%", "xG%"}.issubset(_tf.columns):
            st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>"
                        f"{_team} NFI% vs xG% (playoffs)</h4>", unsafe_allow_html=True)
            _nfi_xg_scatter(_tf, True, dl_suffix="-playoffs", highlight_name=_hi,
                          year_label="Playoffs")
        if {"Player", "EDGE DZ%", "EDGE OZ%"}.issubset(_tf.columns):
            st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>"
                        f"{_team} EDGE: D-Zone vs O-Zone Time% (playoffs)</h4>",
                        unsafe_allow_html=True)
            _edge_zone_scatter(_tf, True, dl_suffix="-playoffs", highlight_name=_hi,
                              year_label="Playoffs")
        if {"Player", "EDGE Top Speed"}.issubset(_tf.columns):
            _edge_speed_scatter(_tf, True, dl_suffix="-playoffs", highlight_name=_hi,
                               year_label="Playoffs", league_df=frame,
                               title_prefix=f"{_team} EDGE: Speed Bursts vs Top Speed (playoffs)")


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
    """Team Zone Impact (OZI/DZI/NZI/TZI + composite, 0–100 index, 50 = average
    team) per window from NFI/output/team_zone.csv (built by
    NFI/scripts/build_team_zone.py). Windows: '4y_pool' (2022-26), '2y_2426'
    (2024-26). TOI-weighted mean of the per-player 0-100 index."""
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
                     "team_xG_QG_pct": "xG-QG%", "team_NFI_QG_pct": "NFI-QG%"})
        team = team.merge(q, on="team", how="left")

    zcols = ["OZI", "DZI", "NZI", "TZI"]
    tz = load_team_zone_playoffs()
    if not tz.empty:
        team = team.merge(tz[["team"] + zcols], on="team", how="left")

    for c in ["TOI", "xG-QG%", "NFI-QG%"] + zcols:
        if c not in team.columns:
            team[c] = np.nan

    team = team.rename(columns={"team": "Team"})
    team = team.sort_values("NFI%", ascending=False, na_position="last").reset_index(drop=True)
    cols = (["Team", "GP", "TOI", "NFI%", "Attack events", "Suppress events"]
            + zcols + ["xG-QG%", "NFI-QG%"])
    disp = team[[c for c in cols if c in team.columns]].copy()

    fmt = {}
    for c in ("NFI%", "xG-QG%", "NFI-QG%"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("Attack events", "Suppress events"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    for c in zcols:
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    if "TOI" in disp:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    if "GP" in disp:
        fmt["GP"] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    _team_rank = (["NFI%", "Attack events", "Suppress events"] + zcols
                  + ["xG-QG%", "xG-QG-F%", "xG-QG-A%", "NFI-QG%", "NFI-QG-A%", "NFI-QG-S%"])
    _apply_ranks(disp, fmt, disp, _team_rank, lower_better={"Suppress events"})
    st.caption("Each metric shows its **(rank)** across playoff teams. "
               "Suppress events (shots against): lowest = #1.")
    _sort_hint()
    _show_df(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
    st.caption(
        f"{len(disp)} teams · all playoffs (2022-2025 pooled) · sorted by NFI% "
        "(CNFI+MNFI share) descending · Zone Impact (OZI/DZI/NZI/TZI, 0–100 index "
        "where 50 = the average team) is TOI-weighted."
    )


def render_teams() -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Teams</h2>",
        unsafe_allow_html=True,
    )
    season_label, game_type = render_scoped_filters("teams", show_situation=True)
    if _block_ref_only(season_label):
        return
    _set_dl_title(None)
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
                      "xG-QG%": _wmean(g, "team_xG_QG_pct"),
                      "NFI-QG%": _wmean(g, "team_NFI_QG_pct")}
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
                              "team_xG_QG_pct": "xG-QG%", "team_NFI_QG_pct": "NFI-QG%"})
            team = team.merge(q, on="team", how="left")

    if team.empty:
        st.info("No team data for this season.")
        return

    # Quality-Games For/Against at team level (TOI-weighted like xG-QG%/NFI-QG%).
    tfa = _team_qg_fa(key)
    if not tfa.empty:
        team = team.merge(tfa, on="team", how="left")

    # Attack / Suppress events — all seasons + pools (ratio-of-sums for pools).
    a = _team_attack_suppress(key)
    if not a.empty:
        team = team.merge(a, on="team", how="left")

    # Team Zone Impact (OZI/DZI/NZI/TZI, 0–100 index, 50 = average team). Single
    # seasons show their pooled window — no raw single-season team zone (2024-25
    # is hit-distorted). Headers carry the window suffix so the pool is unambiguous.
    zwin = _team_zone_window(key)
    zsfx = "2yr" if zwin == "2y_2426" else "4yr"
    _zbase = ["OZI", "DZI", "NZI", "TZI"]
    zcols = [f"{m} ({zsfx})" for m in _zbase]
    tz = load_team_zone()
    if not tz.empty:
        tzw = (tz[tz["window"] == zwin][["team"] + _zbase]
               .rename(columns=dict(zip(_zbase, zcols))))
        team = team.merge(tzw, on="team", how="left")

    # Per-situation team suite — driven by the global Situation filter.
    tsit = _team_situation_metrics(key, _situation_toggle_state())
    if not tsit.empty:
        team = team.merge(tsit[["Team"] + TEAM_SIT_COLS].rename(columns={"Team": "team"}),
                          on="team", how="left")

    _fa_disp = list(_TEAM_QG_FA.values())   # xG-QG-F%, xG-QG-A%, NFI-QG-A%, NFI-QG-S%
    for c in ["TOI", "xG-QG%", "NFI-QG%", "Attack events",
              "Suppress events"] + zcols + _fa_disp:
        if c not in team.columns:
            team[c] = np.nan

    team = team.rename(columns={"team": "Team"})
    team = team.sort_values("NFI%", ascending=False, na_position="last").reset_index(drop=True)
    cols = (["Team", "GP", "TOI", "NFI%", "Attack events", "Suppress events"]
            + zcols + ["xG-QG%", "xG-QG-F%", "xG-QG-A%",
                       "NFI-QG%", "NFI-QG-A%", "NFI-QG-S%"]
            + [c for c in TEAM_SIT_COLS if c in team.columns])
    disp = team[[c for c in cols if c in team.columns]].copy()

    fmt = {}
    for c in ("NFI%", "xG-QG%", "NFI-QG%") + tuple(_fa_disp):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("Attack events", "Suppress events"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    for c in zcols:
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    if "TOI" in disp:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    if "GP" in disp:
        fmt["GP"] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"
    # Team Situation columns
    for c in ("Sit CF%", "Sit xGF%", "Sit GF%"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}%"
    for c in ("Sit xGF/60", "Sit xGA/60"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    for c in ("Sit PP xGF+CF/60", "Sit PK xGA+CA/60"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    if "Sit TOI" in disp:
        fmt["Sit TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"

    _team_rank = (["NFI%", "Attack events", "Suppress events"] + zcols
                  + ["xG-QG%", "NFI-QG%"])
    _apply_ranks(disp, fmt, disp, _team_rank, lower_better={"Suppress events"})
    st.caption("Each metric shows its **(rank)** across all 32 teams. "
               "Suppress events (shots against): lowest = #1.")
    _sort_hint()
    _show_df(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)

    zwin_label = "4-year pool (2022-26)" if zwin == "4y_pool" else "2-year pool (2024-26)"
    cap = (f"{len(disp)} teams · {season_label} · sorted by NFI% (CNFI+MNFI share) "
           f"descending · Zone Impact (OZI/DZI/NZI/TZI, 0–100 index where 50 = the "
           f"average team) is TOI-weighted, shown as the {zwin_label}; single seasons "
           f"display their pooled window (2022-24 → 4yr, 2024-26 → 2yr) since "
           f"single-season team zone isn't published.")
    st.caption(cap)

    # Team shot map — pick a team to see where it generates its shots.
    _teams_avail = sorted(team["Team"].dropna().unique().tolist())
    _tpick = st.selectbox("Team shot map", ["—"] + _teams_avail, index=0,
                          key="teams_shotmap_pick")
    if _tpick and _tpick != "—":
        _render_shot_chart("team", _tpick, _tpick, _tpick, season_label, playoffs=False)


# ---------------------------------------------------------------------------
# Goalies tab — NFI-GSAx + QNFG% + QG (union of qualified cohorts)
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


# sQS% — tiered save%-based Quality Games. Built in two shot scopes;
# suffix "" = 5v5 ES regulation, "_allsit" = all situations. Playoffs not yet
# built for this metric (see compute_qg_tiered.py).
QG_SCOPE_SUFFIX = {"5v5": "", "All situations": "_allsit"}


def _qg_toggle_state() -> tuple[bool, str, str]:
    """Shared sQS% baseline + shot-scope toggle state (set by the widgets
    in render_goalies, read here so the drill-in / Trade Analyzer respect
    whatever the user picked on the Goalies tab in this same run — same
    shared-session-state pattern as the global Season/Game-type filters).
    Returns (qg_starter, qg_scope_suffix, qg_label)."""
    scope_label = st.session_state.get("goalies_qg_scope", "5v5")
    qg_starter = True
    qg_scope_suffix = QG_SCOPE_SUFFIX.get(scope_label, "")
    qg_label = "sQS%"
    return qg_starter, qg_scope_suffix, qg_label


@st.cache_data(show_spinner=False, ttl=3600)
def load_qg_tiered_pooled(scope_suffix: str) -> pd.DataFrame:
    fp = _QC / f"qg_savepct_2022-2026{scope_suffix}.csv"
    return pd.read_csv(fp) if fp.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_qg_tiered_by_season(scope_suffix: str) -> pd.DataFrame:
    fp = _QC / f"qg_savepct_per_season_2022-2026{scope_suffix}.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(int)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_qg_starter_baseline(scope_suffix: str) -> dict:
    """{season_str: starter-tier baseline save% (0-100)} for the sQS% bar."""
    fp = _QC / f"qg_tier_baselines_by_season{scope_suffix}.csv"
    if not fp.exists():
        return {}
    df = pd.read_csv(fp)
    df = df[df["tier"] == "starter"]
    return {str(s): float(v) for s, v in zip(df["season"], df["baseline_save_pct"])}


@st.cache_data(show_spinner=False, ttl=3600)
def load_sqs_league_avg(scope_suffix: str) -> dict:
    """{season_str: GP-weighted league-average sQS% (QG_pct_s, 0-100)} — the
    real "average goalie" line for the sQS% consistency bar. sQS% is graded
    against the high STARTER-tier save% bar, so the league averages ~51-53%
    (NOT 50%): a hardcoded 50 line understates the bar and makes an average
    goalie read as above-average. Volume-weighted by GP to mirror how NFI SV%'s
    league-average-save% baseline is shots-weighted."""
    df = load_qg_tiered_by_season(scope_suffix)
    if df.empty or "QG_pct_s" not in df.columns:
        return {}
    d = df.copy()
    d["season"] = d["season"].astype(str)
    out = {}
    for s, g in d.groupby("season"):
        v = pd.to_numeric(g["QG_pct_s"], errors="coerce")
        w = pd.to_numeric(g.get("GP"), errors="coerce")
        m = v.notna() & (w > 0)
        if m.any():
            out[str(s)] = float(np.average(v[m], weights=w[m]))
    return out


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


@st.cache_data(show_spinner=False, ttl=3600)
def load_qg_tiered_playoffs(scope_suffix: str) -> pd.DataFrame:
    """Pooled all_playoffs sQS% — each playoff game judged against ITS
    season's regular-season starter/backup baseline (see
    compute_qg_tiered_playoffs.py); no fresh playoff-only tier split."""
    fp = _QC / f"qg_savepct_playoffs{scope_suffix}.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["season"] = df["season"].astype(str)
    return df[df["season"] == PLAYOFF_SCOPE].copy()


def _goalie_trend(gid: int, qg_scope_suffix: str = "", qg_starter: bool = True) -> pd.DataFrame:
    """Per-season (2022-23..2025-26) NFI-GSAx/60, NFI SV%, QNFG%, QG%, and
    sQS% (per the shared toggle) for one goalie_id, outer-merged on
    season. Season normalized to INT before merging (all loaders cast to int)
    to avoid silent empty merges. The NFI-GSAx file includes a 2021-22 row;
    it's dropped here. The save%-based column is labeled "sQS%" (starter-tier
    baseline; the backup tier was retired)."""
    gid = int(gid)
    seasons_int = [20222023, 20232024, 20242025, 20252026]
    qg_col = "QG_pct_s"
    qg_label = "sQS%"
    parts = []
    n = load_goalie_nfi_by_season()
    if not n.empty:
        _ncols = [c for c in ("GSAx_per60", "NFI_save_pct", "GSAx") if c in n.columns]
        parts.append(n[n["goalie_id"] == gid][["season"] + _ncols]
                     .rename(columns={"GSAx_per60": "NFI-GSAx/60", "NFI_save_pct": "NFI SV%",
                                      "GSAx": "NFI-GSAx"}))
    q = load_qnfs_by_season()
    if not q.empty:
        parts.append(q[q["goalie_id"] == gid][["season", "QNFS_pct"]]
                     .rename(columns={"QNFS_pct": "QNFG%"}))
    s = load_qs_by_season()
    if not s.empty:
        _scols = [c for c in ("QS_GSAx_pct", "GSAx_total") if c in s.columns]
        parts.append(s[s["goalie_id"] == gid][["season"] + _scols]
                     .rename(columns={"QS_GSAx_pct": "QG%", "GSAx_total": "MP-GSAx"}))
    t = load_qg_tiered_by_season(qg_scope_suffix)
    if not t.empty and qg_col in t.columns:
        parts.append(t[t["goalie_id"] == gid][["season", qg_col]]
                     .rename(columns={qg_col: qg_label}))
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
    # Games played per season (prefer NFI 'games', fall back to QNFG/QS/QG 'GP').
    _gp = {}
    for _df, _c in ((n, "games"), (q, "GP"), (s, "GP"), (t, "GP")):
        if _df.empty or _c not in _df.columns:
            continue
        for _, _rr in _df[_df["goalie_id"] == gid].iterrows():
            _s = int(_rr["season"])
            if _s not in _gp and pd.notna(_rr[_c]):
                _gp[_s] = int(_rr[_c])
    base["GP"] = base["season"].map(_gp)
    # All-shot GSAx per 60 ≈ cumulative all-shot GSAx / GP (goalies play ~full
    # games, so per-game ≈ per-60). NFI-GSAx/60 already comes as a true per-60.
    if "MP-GSAx" in base.columns:
        _gpn = pd.to_numeric(base["GP"], errors="coerce")
        base["MP-GSAx/60"] = np.where(_gpn > 0, base["MP-GSAx"] / _gpn, np.nan)
    return base.sort_values("season").reset_index(drop=True)


def _goalie_season_ranks(gid: int, qg_scope_suffix: str = "", qg_starter: bool = True) -> dict:
    """Per-season LEAGUE rank of each metric among the QUALIFIED goalies that
    season (#1 = best) — the same cohort the leaderboard ranks over, so the drill-in
    rank matches the list. A goalie below the metric's qualifying floor gets no rank
    (they show as unranked on the list too). Returns {display_col: {season: rank}}."""
    gid = int(gid)
    out = {}
    seasons_int = [20222023, 20232024, 20242025, 20252026]
    qg_col = "QG_pct_s"
    qg_label = "sQS%"
    # per metric: (loader, value col, display col, fallback floor col, floor)
    for loader, src, disp_c, floor_col, floor in (
        (load_goalie_nfi_by_season, "GSAx_per60", "NFI-GSAx/60", "total_faced", 100),
        (load_goalie_nfi_by_season, "NFI_save_pct", "NFI SV%", "total_faced", 100),
        (load_qnfs_by_season, "QNFS_pct", "QNFG%", "GP", 25),
        (load_qs_by_season, "QS_GSAx_pct", "QG%", "GP", 25),
        (load_qs_by_season, "GSAx_total", "MP-GSAx", "GP", 25),
        (lambda: load_qs_by_season().assign(
            _mp60=lambda _d: np.where(pd.to_numeric(_d.get("GP"), errors="coerce") > 0,
                                      pd.to_numeric(_d.get("GSAx_total"), errors="coerce")
                                      / pd.to_numeric(_d.get("GP"), errors="coerce"), np.nan)),
         "_mp60", "MP-GSAx/60", "GP", 25),
        (lambda: load_qg_tiered_by_season(qg_scope_suffix), qg_col, qg_label, "GP", 25),
    ):
        df = loader()
        if df.empty or src not in df.columns:
            continue
        df = df.copy()
        df["season"] = df["season"].astype(int)
        d = {}
        for ssn in seasons_int:
            sub = df[df["season"] == ssn]
            if sub.empty:
                continue
            # Qualified cohort = the producer's `qualified` flag when present,
            # else the in-app floor (matches render_goalies).
            if "qualified" in sub.columns:
                qmask = sub["qualified"].fillna(False).astype(bool)
            elif floor_col in sub.columns:
                qmask = sub[floor_col].fillna(0) >= floor
            else:
                qmask = pd.Series(True, index=sub.index)
            prow = sub[sub["goalie_id"] == gid]
            if (len(prow) and pd.notna(prow[src].iloc[0])
                    and bool(qmask.loc[prow.index[0]])):
                d[ssn] = _league_rank(sub[qmask][src], prow[src].iloc[0])
        out[disp_c] = d
    return out


def _goalie_profile_table(gid: int, qg_scope_suffix: str = "", qg_starter: bool = True
                          ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Rank-annotated per-season goalie trend + a '2yr avg (24-26)' pooled row.
    Returns (display_df, trend, metric_cols). Shared by the goalie detail and the
    Trade Analyzer's goalie mode. qg_scope_suffix/qg_starter select which ONE of
    sQS% shows (matches the leaderboard's toggle — no starter/backup split)."""
    qg_label = "sQS%"
    trend = _goalie_trend(gid, qg_scope_suffix, qg_starter)
    if trend.empty:
        return pd.DataFrame(), trend, []
    metric_cols = [c for c in ("NFI-GSAx", "NFI-GSAx/60", "MP-GSAx", "MP-GSAx/60",
                   "NFI SV%", "QNFG%", "QG%", qg_label) if c in trend.columns]
    has_gp = "GP" in trend.columns and trend["GP"].notna().any()
    ranks = _goalie_season_ranks(gid, qg_scope_suffix, qg_starter)
    _b = {"NFI-GSAx/60": lambda v: f"{v:+.3f}", "NFI SV%": lambda v: f"{v * 100:.1f}%",
          "QNFG%": lambda v: f"{v:.1f}%", "QG%": lambda v: f"{v:.1f}%",
          qg_label: lambda v: f"{v:.1f}%",
          "NFI-GSAx": lambda v: f"{v:+.1f}", "MP-GSAx": lambda v: f"{v:+.1f}",
          "MP-GSAx/60": lambda v: f"{v:+.3f}"}
    rows = []
    for _, r in trend.iterrows():
        ssn = int(r["season"])
        row = {"Season": r["Season"]}
        if has_gp:
            row["GP"] = f"{int(r['GP'])}" if pd.notna(r.get("GP")) else "—"
        for c in metric_cols:
            v = r[c]
            if pd.isna(v):
                row[c] = "—"
            else:
                txt = _b.get(c, lambda v: f"{v}")(v)
                rk = ranks.get(c, {}).get(ssn)
                # Below that metric's OWN qualifying floor that season (e.g. QNFG%/
                # QG%/sQS% need >=25 GP) — mark it explicitly rather than showing a
                # bare number with no indication it's unranked.
                row[c] = f"{txt} ({rk})" if rk is not None else f"{txt} (UR)"
        rows.append(row)

    # Append a "2yr avg (24-26)" row — denominator-based pool of the last two
    # seasons (ranked within the 2yr pool).
    _n2, _q2, _s2 = _pool_goalie_seasons((20242025, 20252026))
    _g2 = _pool_qg_tiered_seasons((20242025, 20252026), qg_scope_suffix)
    _qg_pool_col = "QG_pct_s"
    _src2 = {"NFI-GSAx/60": (_n2, "NFIG60"), "NFI SV%": (_n2, "NFISV"),
             "QNFG%": (_q2, "QNFS_pct"), "QG%": (_s2, "QS_GSAx_pct"),
             qg_label: (_g2, _qg_pool_col)}
    _v2 = {}
    for c, (fr, col) in _src2.items():
        rr = fr[fr["goalie_id"] == gid] if (not fr.empty and col in fr.columns) else pd.DataFrame()
        _v2[c] = rr[col].iloc[0] if len(rr) else np.nan
    if any(pd.notna(v) for v in _v2.values()):
        row = {"Season": "2yr avg (24-26)"}
        if has_gp:
            _gp2 = trend[trend["season"].isin((20242025, 20252026))]["GP"].dropna()
            row["GP"] = f"{int(_gp2.sum())}" if len(_gp2) else "—"
        for c in metric_cols:
            v = _v2.get(c)
            if pd.isna(v):
                row[c] = "—"
                continue
            txt = _b.get(c, lambda v: f"{v}")(v)
            fr, col = _src2[c]
            s = pd.to_numeric(fr[col], errors="coerce")
            row[c] = f"{txt} ({int((s > v).sum()) + 1})"
        rows.append(row)
    _lead = ["Season"] + (["GP"] if has_gp else [])
    return pd.DataFrame(rows, columns=_lead + metric_cols), trend, metric_cols


def _tight_domain(values, pad_frac: float = 0.12, min_pad: float = 0.5):
    """Data-tight y-domain [min-pad, max+pad] so a line's movement is visible."""
    vv = [float(v) for v in values if pd.notna(v)]
    if not vv:
        return None
    lo, hi = min(vv), max(vv)
    pad = max(min_pad, (hi - lo) * pad_frac)
    return [lo - pad, hi + pad]


@st.cache_data(show_spinner=False, ttl=3600)
def load_nfi_sv_baseline() -> dict:
    """{season_str: league-average NFI (net-front) save% (0-1)} — shots-faced-
    weighted mean of NFI_save_pct, the cut-off line for a goalie's NFI SV%."""
    df = load_goalie_nfi_by_season()
    if df.empty or "NFI_save_pct" not in df.columns:
        return {}
    out = {}
    for s, g in df.groupby("season"):
        w = pd.to_numeric(g.get("total_faced"), errors="coerce")
        v = pd.to_numeric(g["NFI_save_pct"], errors="coerce")
        m = v.notna() & (w > 0)
        out[str(s)] = float(np.average(v[m], weights=w[m])) if m.any() else np.nan
    return out


_GSAX_STARTER_N = 32   # top-N goalies by GP each season = "starter" tier (matches sQS%)


@st.cache_data(show_spinner=False, ttl=3600)
def load_gsax_league_avg() -> dict:
    """{(season_str, metric): STARTER-tier (top-32 GP) average GSAx} — the baseline
    the goalie GSAx bar diverges from, so a goalie reads above/below the average
    STARTER (not the whole league). Starter tier matches the sQS% definition.
    Metrics: NFI-GSAx, NFI-GSAx/60, MP-GSAx, MP-GSAx/60."""
    out = {}
    n = load_goalie_nfi_by_season()
    if not n.empty and "games" in n.columns:
        for s, g in n.groupby("season"):
            ss = str(int(s))
            st_ = g.assign(_gp=pd.to_numeric(g["games"], errors="coerce")) \
                   .sort_values("_gp", ascending=False).head(_GSAX_STARTER_N)
            out[(ss, "NFI-GSAx")] = float(pd.to_numeric(st_["GSAx"], errors="coerce").mean())
            p60 = pd.to_numeric(st_.get("GSAx_per60"), errors="coerce")
            w = pd.to_numeric(st_.get("total_faced"), errors="coerce")
            m = p60.notna() & (w > 0)
            out[(ss, "NFI-GSAx/60")] = float(np.average(p60[m], weights=w[m])) if m.any() else np.nan
    q = load_qs_by_season()
    if not q.empty and {"GSAx_total", "GP"}.issubset(q.columns):
        for s, g in q.groupby("season"):
            ss = str(int(s))
            st_ = g.assign(_gp=pd.to_numeric(g["GP"], errors="coerce")) \
                   .sort_values("_gp", ascending=False).head(_GSAX_STARTER_N)
            gt = pd.to_numeric(st_["GSAx_total"], errors="coerce")
            gp = pd.to_numeric(st_["GP"], errors="coerce")
            out[(ss, "MP-GSAx")] = float(gt.mean())
            m = gt.notna() & (gp > 0)
            out[(ss, "MP-GSAx/60")] = float(gt[m].sum() / gp[m].sum()) if m.any() and gp[m].sum() > 0 else np.nan
    return out


def _goalie_consistency_bar(row, qg_label: str, sv_baseline, sqs_baseline=None) -> None:
    """One-year diverging bar (like the player QG bar). QNFG% and QG% (GSAx≥0
    game shares) sit against the 50% line — a clean "beat expected half the time"
    reference. sQS% and NFI SV% each sit against their OWN season league-average
    line (sqs_baseline / sv_baseline) rather than 50%, because both are graded
    against a high save%-based bar where the league average is NOT 50%: sQS%
    averages ~51-53% (a hardcoded 50 would flatter every average goalie), and
    NFI SV% averages ~91%. Bar colour: blue above its own line, orange below."""
    import altair as alt
    # sQS% baseline: real league-average sQS% when supplied, else fall back to 50.
    _sqs_base = (float(sqs_baseline) if sqs_baseline is not None
                 and pd.notna(sqs_baseline) else 50.0)
    specs = [("QNFG%", 50.0, 1.0), ("QG%", 50.0, 1.0), (qg_label, _sqs_base, 1.0)]
    sv_base = sv_baseline * 100.0 if sv_baseline is not None and pd.notna(sv_baseline) else None
    if sv_base is not None and "NFI SV%" in row and pd.notna(row["NFI SV%"]):
        specs.append(("NFI SV%", sv_base, 100.0))
    rows = []
    for m, base, mul in specs:
        if m in row and pd.notna(row[m]):
            v = float(row[m]) * mul
            rows.append({"Metric": m, "value": v, "base": base,
                         "color": _bar_color(50.0 + (v - base))})   # colour by dist from its line
    if not rows:
        return
    d = pd.DataFrame(rows)
    _vals = [r["value"] for r in rows] + [r["base"] for r in rows]
    dom = [int(np.floor(min(_vals))) - 2, int(np.ceil(max(_vals))) + 2]
    _sqs_txt = (f"sQS% vs its **{_sqs_base:.0f}%** league average"
                if sqs_baseline is not None and pd.notna(sqs_baseline)
                else "sQS% vs 50")
    st.caption(f"**{row['Season']}** — QNFG% / QG% vs **50** (beat expected half the "
               f"time); {_sqs_txt}; NFI SV% vs league-average save%.")
    _sort = [r["Metric"] for r in rows]
    bars = alt.Chart(d).mark_bar(size=40).encode(
        x=alt.X("Metric:N", sort=_sort,
                axis=alt.Axis(labelAngle=0, title=None, labelFontWeight="bold")),
        y=alt.Y("base:Q", scale=alt.Scale(domain=dom), title="%"), y2="value:Q",
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=[alt.Tooltip("Metric:N"),
                 alt.Tooltip("base:Q", title="league avg", format=".1f"),
                 alt.Tooltip("value:Q", format=".1f")])
    # Each bar's baseline (its own league-average / 50 reference) drawn as a short
    # tick at that bar's base — not a full-width line, since the three metrics now
    # have three different baselines (~50 / ~52 / ~91).
    ticks = alt.Chart(d).mark_tick(color=_CHART_THIRD, thickness=2, size=44).encode(
        x=alt.X("Metric:N", sort=_sort), y="base:Q")
    _show_chart(bars + ticks, dl_name=f"Goalie-consistency-{row['Season']}")


def _goalie_gsax_bar(row, qg_scope_suffix: str = "") -> None:
    """One-year GSAx bar, its own chart: total (left) and per-60 (right). Bars
    span from that season's STARTER-tier average (not 0) to the goalie's raw
    GSAx — same y/y2 floating-bar encoding as _goalie_consistency_bar's sQS%
    bar, so a bar that's just above its starter-avg baseline reads as small,
    not as "starting from zero". NFI-GSAx = net-front, MP-GSAx = all-shot
    (MoneyPuck)."""
    import altair as alt
    _avg = load_gsax_league_avg()
    _ssn = str(int(row["season"])) if pd.notna(row.get("season")) else None

    def _panel(metrics, title, fmt):
        rows = []
        for m in metrics:
            if m in row and pd.notna(row[m]):
                base = _avg.get((_ssn, m), 0.0) if _ssn else 0.0
                raw = float(row[m])
                rows.append({"Metric": m, "raw": raw, "avg": base,
                             "color": _BAR_BLUE_STRONG if raw >= base else _BAR_ORG_STRONG})
        if not rows:
            return None
        d = pd.DataFrame(rows)
        order = [r["Metric"] for r in rows]
        _vals = [r["raw"] for r in rows] + [r["avg"] for r in rows]
        _pad = max(0.1, (max(_vals) - min(_vals)) * 0.15) if len(_vals) > 1 else max(0.1, abs(_vals[0]) * 0.2)
        dom = [min(_vals) - _pad, max(_vals) + _pad]
        bars = alt.Chart(d).mark_bar(size=44).encode(
            x=alt.X("Metric:N", sort=order, axis=alt.Axis(labelAngle=-20, title=None)),
            y=alt.Y("avg:Q", title=title, scale=alt.Scale(domain=dom)), y2="raw:Q",
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=["Metric:N", alt.Tooltip("raw:Q", title="GSAx", format=fmt),
                     alt.Tooltip("avg:Q", title="starter avg", format=fmt)])
        cuts = pd.DataFrame({"y": sorted({r["avg"] for r in rows})})
        line = alt.Chart(cuts).mark_rule(color=_CHART_THIRD, strokeDash=[4, 4]).encode(y="y:Q")
        return (bars + line).properties(width=320, height=300)

    total = _panel(["NFI-GSAx", "MP-GSAx"], "GSAx vs starter avg (total)", ".2f")
    per60 = _panel(["NFI-GSAx/60", "MP-GSAx/60"], "GSAx vs starter avg (/60)", ".3f")
    panels = [p for p in (total, per60) if p is not None]
    if not panels:
        return
    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
    st.caption(f"**{row['Season']}** — GSAx vs **starter-tier average** (total, then per-60).")
    # Note (blue) — the starter-average baseline for the open season, like the sQS% note.
    if _ssn:
        def _fmt(mtot, m60):
            a, b = _avg.get((_ssn, mtot)), _avg.get((_ssn, m60))
            return f"{a:+.1f} tot / {b:+.3f} /60" if a is not None and b is not None else "—"
        st.markdown(
            f"<div style='color:{_CHART_THIRD}; font-size:0.85rem; margin:0.1rem 0 0.4rem;'>"
            f"<b>Starter-tier (top-{_GSAX_STARTER_N} GP) average GSAx, {row['Season']}</b> "
            f"— NFI {_fmt('NFI-GSAx', 'NFI-GSAx/60')}; "
            f"MP {_fmt('MP-GSAx', 'MP-GSAx/60')} (the blue baseline).</div>",
            unsafe_allow_html=True)
        _sqs_bl = load_qg_starter_baseline(qg_scope_suffix).get(_ssn)
        if _sqs_bl is not None:
            _scope_label = "all situations" if qg_scope_suffix else "5v5"
            st.markdown(
                f"<div style='color:{_CHART_THIRD}; font-size:0.85rem; margin:0.1rem 0 0.4rem;'>"
                f"<b>sQS% starter baseline save% ({_scope_label})</b> — a game clears sQS% "
                f"when its save% beats this line: {row['Season']} {_sqs_bl:.1f}%.</div>",
                unsafe_allow_html=True)
    # Two separate charts (total, then per-60) — each its own downloadable image.
    if total is not None:
        _show_chart(total, dl_name=f"Goalie-GSAx-total-{row['Season']}", brand_width=340)
    if per60 is not None:
        _show_chart(per60, dl_name=f"Goalie-GSAx-per60-{row['Season']}", brand_width=340)


def _render_goalie_profile(gid: int, qg_scope_suffix: str = "", qg_starter: bool = True,
                           season_label: str = None) -> None:
    """Per-season trend table + line charts for one goalie."""
    qg_label = "sQS%"
    disp, trend, metric_cols = _goalie_profile_table(gid, qg_scope_suffix, qg_starter)
    if disp.empty:
        st.info("No per-season data available for this goalie.")
        return
    st.caption("Each value shows its **(rank)** — league rank among all goalies "
               "that season (2yr row ranks within the 2-season pool).")
    _show_df(disp, width="stretch", hide_index=True)

    # Shot map — shots faced (goalie's-eye view), in the goalie's team colours.
    _g = load_goalie_nfi()
    _grow = _g[_g["goalie_id"] == int(gid)] if not _g.empty else _g
    _gname = str(_grow["goalie_name"].iloc[0]) if len(_grow) else f"Goalie {gid}"
    _gteam = (str(_grow["team"].iloc[0]) if len(_grow) and "team" in _grow.columns
              and pd.notna(_grow["team"].iloc[0]) else None)
    _render_shot_chart("goalie", int(gid), _gname, _gteam, season_label, playoffs=False)

    # CHOICE: 3 small multiples. NFI-GSAx/60 is a per-60 rate (~±0.3); NFI SV%
    # is a raw save% (~85-95%) — a very different band from the "beat expected
    # X% of the time" consistency rates (~30-70%), so it gets its own chart
    # rather than squashing the consistency chart's zoomed axis. QNFG%/QG%/
    # sQS% are all "beat a bar X% of the time" rates and share one chart.
    import altair as alt

    def _gchart(title, ys, frame, ydomain=None):
        ys = [c for c in ys if c in frame.columns and frame[c].notna().any()]
        if not ys:
            return
        st.caption(title)
        long = (frame[["Season"] + ys].melt("Season", var_name="Metric",
                value_name="value").dropna(subset=["value"]))
        _yscale = alt.Scale(domain=ydomain) if ydomain else alt.Undefined
        ch = alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("Season:N", title=None),
            y=alt.Y("value:Q", title=None, scale=_yscale),
            color=alt.Color("Metric:N", sort=ys, legend=alt.Legend(
                orient="bottom", title=None, symbolType="stroke", symbolStrokeWidth=2.5),
                scale=alt.Scale(domain=ys,
                                range=[_CHART_COLORS.get(c, _CHART_SECOND) for c in ys])),
            tooltip=["Season:N", "Metric:N", alt.Tooltip("value:Q", format=".3f")]
            ).properties(height=300)
        _show_chart(ch, dl_name=title.split(" (")[0].replace(" ", "-"))

    # (1) One-year diverging bars: consistency+SV% (vs 50% / league-avg save%) and
    # GSAx (total + per-60, vs 0) — each on its own chart with blue cut-off lines.
    _bar_cols = [c for c in ("QNFG%", "QG%", qg_label, "NFI SV%", "NFI-GSAx",
                 "MP-GSAx", "NFI-GSAx/60", "MP-GSAx/60") if c in trend.columns]
    if _bar_cols:
        _bt = trend.dropna(subset=_bar_cols, how="all")
        if not _bt.empty:
            # Default to the globally-selected Season filter (same pattern as the
            # Player List bar chart's _default_qg_year) instead of always the
            # latest row — previously hardcoded to iloc[-1], so the bars never
            # changed when the Season filter changed.
            _bt_seasons = _bt["Season"].astype(str).tolist()
            _yr = _default_qg_year(season_label, _bt_seasons)
            _match = _bt[_bt["Season"].astype(str) == str(_yr)]
            _row = _match.iloc[0] if len(_match) else _bt.iloc[-1]
            _ssn_str = str(int(_row["season"])) if pd.notna(_row.get("season")) else None
            _svb = load_nfi_sv_baseline().get(_ssn_str) if _ssn_str else None
            _sqsb = load_sqs_league_avg(qg_scope_suffix).get(_ssn_str) if _ssn_str else None
            _goalie_consistency_bar(_row, qg_label, _svb, sqs_baseline=_sqsb)
            _goalie_gsax_bar(_row, qg_scope_suffix)

    # (2) Consistency % over time (no GSAx) — QNFG%, QG%, sQS% on one axis.
    _cons = [c for c in ("QNFG%", "QG%", qg_label) if c in trend.columns]
    if _cons:
        _cv = pd.concat([trend[c] for c in _cons], ignore_index=True).tolist()
        _gchart("Consistency % over time (QNFG%, QG%, sQS%)", _cons, trend,
                ydomain=_tight_domain(_cv, min_pad=1.0))

    # (3) GSAx over time — NFI-GSAx (net-front) vs MP-GSAx (all-shot / MoneyPuck),
    # shown both per-60 and cumulative.
    _gs60 = [c for c in ("NFI-GSAx/60", "MP-GSAx/60") if c in trend.columns]
    if _gs60:
        _v = pd.concat([trend[c] for c in _gs60], ignore_index=True).tolist()
        _gchart("GSAx per 60 over time (NFI-GSAx/60 vs MP-GSAx/60)", _gs60, trend,
                ydomain=_tight_domain(_v, min_pad=0.05))
    _gsc = [c for c in ("NFI-GSAx", "MP-GSAx") if c in trend.columns]
    if _gsc:
        _v = pd.concat([trend[c] for c in _gsc], ignore_index=True).tolist()
        _gchart("GSAx cumulative over time (NFI-GSAx vs MP-GSAx)", _gsc, trend,
                ydomain=_tight_domain(_v, min_pad=1.0))


def _wilson(k: float, n: float, lower: bool = True, z: float = 1.96) -> float:
    """Wilson score-interval bound for k successes in n trials (returns 0-1)."""
    if not n:
        return np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (center - margin) / denom if lower else (center + margin) / denom


@st.cache_data(show_spinner=False, ttl=3600)
def _pool_goalie_seasons(seasons: tuple) -> tuple:
    """Faithful denominator-based pool of the by-season goalie files across
    `seasons`: sum the raw counts, then recompute the rates (so each metric is
    computed over the combined sample — e.g. quality-games / GP across ~164 games,
    GSAx / pooled-TOI). Returns (nfi, qn, qs) shaped like the other goalie
    branches, with per-metric `qualified` flags (NFI-GSAx ≥100·n net-front shots;
    QNFG/QG: any single season with ≥25 GP)."""
    sset = set(seasons)
    n_seasons = len(seasons)
    bs = load_goalie_nfi_by_season()
    nfi = pd.DataFrame()
    if not bs.empty:
        b = bs[bs["season"].isin(sset)].copy()
        # Per-season ES TOI reconstructed from GSAx / per-60, then summed.
        b["_toi"] = np.where(b["GSAx_per60"].abs() > 1e-9,
                             b["GSAx"] / b["GSAx_per60"] * 60.0, np.nan)
        last_team = (b.sort_values("season").drop_duplicates("goalie_id", keep="last")
                     .set_index("goalie_id")["team"])
        g = (b.groupby(["goalie_id", "goalie_name"])
               .agg(GP_nfi=("games", "sum"), total_faced=("total_faced", "sum"),
                    _gsax=("GSAx", "sum"), _toi=("_toi", "sum"),
                    _goals=("total_goals", "sum")).reset_index())
        g["NFIG60"] = np.where(g["_toi"] > 0, g["_gsax"] / g["_toi"] * 60.0, np.nan).round(3)
        g["NFISV"] = np.where(g["total_faced"] > 0,
                              (g["total_faced"] - g["_goals"]) / g["total_faced"],
                              np.nan).round(4)
        g["team"] = g["goalie_id"].map(last_team)
        g["qual_gsax"] = g["total_faced"] >= 100 * n_seasons
        nfi = g[["goalie_id", "goalie_name", "team", "GP_nfi", "total_faced",
                 "NFIG60", "NFISV", "qual_gsax"]]
    q0 = load_qnfs_by_season()
    qn = pd.DataFrame()
    if not q0.empty:
        b = q0[q0["season"].isin(sset)]
        g = (b.groupby(["goalie_id", "goalie_name"])
               .agg(GP_qn=("GP", "sum"), _q=("quality_games", "sum"),
                    _maxgp=("GP", "max")).reset_index())
        g["QNFS_pct"] = g["_q"] / g["GP_qn"] * 100
        g["QNFS_lo"] = g.apply(lambda r: _wilson(r["_q"], r["GP_qn"], True) * 100, axis=1)
        g["QNFS_hi"] = g.apply(lambda r: _wilson(r["_q"], r["GP_qn"], False) * 100, axis=1)
        g["qual_qn"] = g["_maxgp"] >= 25
        qn = g[["goalie_id", "goalie_name", "GP_qn", "QNFS_pct", "QNFS_lo",
                "QNFS_hi", "qual_qn"]]
    s0 = load_qs_by_season()
    qs = pd.DataFrame()
    if not s0.empty:
        b = s0[s0["season"].isin(sset)]
        _agg = dict(GP_qs=("GP", "sum"), _q=("quality_games", "sum"),
                    _maxgp=("GP", "max"))
        if "GSAx_total" in b.columns:
            _agg["MP-GSAx"] = ("GSAx_total", "sum")   # all-shot GSAx pools additively
        g = b.groupby(["goalie_id", "goalie_name"]).agg(**_agg).reset_index()
        g["QS_GSAx_pct"] = g["_q"] / g["GP_qs"] * 100
        g["QS_GSAx_lo"] = g.apply(lambda r: _wilson(r["_q"], r["GP_qs"], True) * 100, axis=1)
        g["qual_qs"] = g["_maxgp"] >= 25
        qs = g[[c for c in ["goalie_id", "goalie_name", "GP_qs", "QS_GSAx_pct",
                            "QS_GSAx_lo", "MP-GSAx", "qual_qs"] if c in g.columns]]
    return nfi, qn, qs


@st.cache_data(show_spinner=False, ttl=3600)
def _pool_qg_tiered_seasons(seasons: tuple, scope_suffix: str) -> pd.DataFrame:
    """Faithful denominator-based pool of the sQS% by-season file across
    `seasons` (same sum-then-recompute pattern as _pool_goalie_seasons): sums
    QGs_games/QGb_games/GP across the pooled seasons, then recomputes the
    rates + Wilson bounds over the combined sample."""
    b0 = load_qg_tiered_by_season(scope_suffix)
    if b0.empty:
        return pd.DataFrame()
    sset = set(seasons)
    b = b0[b0["season"].isin(sset)]
    if b.empty:
        return pd.DataFrame()
    g = (b.groupby(["goalie_id", "goalie_name"])
           .agg(GP_qg=("GP", "sum"), _qs=("QGs_games", "sum"), _qb=("QGb_games", "sum"),
                _maxgp=("GP", "max")).reset_index())
    g["QG_pct_s"] = g["_qs"] / g["GP_qg"] * 100
    g["QG_pct_b"] = g["_qb"] / g["GP_qg"] * 100
    g["QG_pct_s_lo"] = g.apply(lambda r: _wilson(r["_qs"], r["GP_qg"], True) * 100, axis=1)
    g["qual_qg"] = g["_maxgp"] >= 25
    return g[["goalie_id", "goalie_name", "GP_qg", "QG_pct_s", "QG_pct_b", "QG_pct_s_lo", "qual_qg"]]


def render_goalies() -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Goalie List</h2>",
        unsafe_allow_html=True,
    )
    season_label, game_type = render_scoped_filters("goalies")
    if _block_ref_only(season_label):
        return
    _set_dl_title(None)                    # only drill-in charts get a name
    playoffs = game_type == "Playoffs"
    if playoffs:
        st.caption("Playoff view — all playoff games (2022-23 → 2024-25) pooled. "
                   "Small playoff samples: all goalies are ranked (no qualifying floor).")

    # sQS% (Starter Quality Start) — save%-based, judged against that season's
    # starter-tier baseline. Only the shot-scope toggle (5v5 vs. all situations)
    # remains; QNFG% and QG% stay 5v5-only regardless. sQS% isn't built for playoffs,
    # so each playoff game is judged against that season's REGULAR-SEASON starter
    # baseline (see compute_qg_tiered_playoffs.py).
    st.session_state.setdefault("goalies_qg_scope", "5v5")
    qgc1, _ = st.columns([1.4, 1.4])
    with qgc1:
        qg_scope_label = st.radio(
            "sQS% shot scope", list(QG_SCOPE_SUFFIX.keys()),
            horizontal=True, key="goalies_qg_scope",
            help="Shot scope for sQS% only — QNFG% and QG% are always 5v5.")
    if playoffs:
        st.caption("The shot-scope toggle applies only to **sQS%** — **QNFG% and QG% "
                   "stay 5v5-only** regardless. Each playoff game is graded against "
                   "**that season's regular-season starter baseline**.")
    else:
        st.caption("The shot-scope toggle applies only to **sQS%** — "
                   "**QNFG% and QG% stay 5v5-only** regardless.")
    qg_starter = True
    qg_scope_suffix = QG_SCOPE_SUFFIX[qg_scope_label]

    # State the sQS% starter baseline save% for the open season(s) + shot scope.
    _bl = load_qg_starter_baseline(qg_scope_suffix)
    if _bl:
        _sk = SEASON_KEY.get(season_label, "pooled")
        _bl_seasons = (POOLED_SEASONS if _sk in ("pooled",) or playoffs
                       else POOLED_2YR_SEASONS if _sk == "pooled_2yr" else [_sk])
        _bl_parts = [f"{SEASON_DISPLAY.get(s, s)} {_bl[s]:.1f}%"
                     for s in _bl_seasons if s in _bl]
        if _bl_parts:
            st.markdown(
                f"<div style='color:{_CHART_THIRD}; font-size:0.85rem; margin:0.1rem 0 0.4rem;'>"
                f"<b>sQS% starter baseline save% ({qg_scope_label})</b> — a game clears "
                "sQS% when its save% beats this line: " + " · ".join(_bl_parts) + "</div>",
                unsafe_allow_html=True)

    is_2yr = (not playoffs) and SEASON_KEY.get(season_label) == "pooled_2yr"
    is_pooled = (not playoffs) and SEASON_KEY.get(season_label, "pooled") == "pooled"
    if is_2yr:
        # Faithful denominator-based pool of 2024-25 + 2025-26 (counts summed,
        # rates recomputed over the combined ~164-game sample).
        nfi, qn, qs = _pool_goalie_seasons(tuple(int(s) for s in POOLED_2YR_SEASONS))
        qgt = _pool_qg_tiered_seasons(tuple(int(s) for s in POOLED_2YR_SEASONS), qg_scope_suffix)
    elif playoffs:
        n = load_goalie_nfi_playoffs()
        nfi = (n[[c for c in ["goalie_id", "goalie_name", "team", "games", "total_faced", "GSAx_per60", "NFI_save_pct"] if c in n.columns]]
               .rename(columns={"games": "GP_nfi", "GSAx_per60": "NFIG60", "NFI_save_pct": "NFISV"})
               if not n.empty else pd.DataFrame())
        q = load_qnfs_playoffs()
        qn = (q[["goalie_id", "goalie_name", "GP", "QNFS_pct", "QNFS_lo", "QNFS_hi"]]
              .rename(columns={"GP": "GP_qn"}) if not q.empty else pd.DataFrame())
        s = load_qs_playoffs()
        qs = (s[[c for c in ["goalie_id", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo", "GSAx_total"] if c in s.columns]]
              .rename(columns={"GP": "GP_qs", "GSAx_total": "MP-GSAx"}) if not s.empty else pd.DataFrame())
        t = load_qg_tiered_playoffs(qg_scope_suffix)
        qgt = (t[["goalie_id", "goalie_name", "GP", "QG_pct_s", "QG_pct_b"]]
              .rename(columns={"GP": "GP_qg"}) if not t.empty else pd.DataFrame())
    elif is_pooled:
        n = load_goalie_nfi()
        nfi = (n[[c for c in ["goalie_id", "goalie_name", "team", "games", "total_faced", "GSAx_per60", "NFI_save_pct", "qualified"] if c in n.columns]]
               .rename(columns={"games": "GP_nfi", "GSAx_per60": "NFIG60", "NFI_save_pct": "NFISV", "qualified": "qual_gsax"})
               if not n.empty else pd.DataFrame())
        q = load_qnfs_pooled()  # keep EVERY goalie; qualification gates ranking only
        qn = (q[[c for c in ["goalie_id", "goalie_name", "GP", "QNFS_pct", "QNFS_lo", "QNFS_hi", "qualified"] if c in q.columns]]
              .rename(columns={"GP": "GP_qn", "qualified": "qual_qn"}) if not q.empty else pd.DataFrame())
        s = load_qs_pooled()
        qs = (s[[c for c in ["goalie_id", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo", "GSAx_total", "qualified"] if c in s.columns]]
              .rename(columns={"GP": "GP_qs", "GSAx_total": "MP-GSAx", "qualified": "qual_qs"}) if not s.empty else pd.DataFrame())
        t = load_qg_tiered_pooled(qg_scope_suffix)  # keep EVERY goalie; qualification gates ranking only
        qgt = (t[[c for c in ["goalie_id", "goalie_name", "GP", "QG_pct_s", "QG_pct_b", "qualified"] if c in t.columns]]
               .rename(columns={"GP": "GP_qg", "qualified": "qual_qg"}) if not t.empty else pd.DataFrame())
    else:
        sk = GOALIE_SEASON_INT.get(season_label)
        bs = load_goalie_nfi_by_season()
        nfi = (bs[bs["season"] == sk][[c for c in ["goalie_id", "goalie_name", "team", "games", "total_faced", "GSAx_per60", "NFI_save_pct", "qualified"] if c in bs.columns]]
               .rename(columns={"games": "GP_nfi", "GSAx_per60": "NFIG60", "NFI_save_pct": "NFISV", "qualified": "qual_gsax"})
               if (not bs.empty and sk) else pd.DataFrame())
        q0 = load_qnfs_by_season()
        qn = (q0[q0["season"] == sk][[c for c in ["goalie_id", "goalie_name", "GP", "QNFS_pct", "QNFS_lo", "QNFS_hi", "qualified"] if c in q0.columns]]
              .rename(columns={"GP": "GP_qn", "qualified": "qual_qn"}) if (not q0.empty and sk) else pd.DataFrame())
        s0 = load_qs_by_season()
        qs = (s0[s0["season"] == sk][[c for c in ["goalie_id", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo", "GSAx_total", "qualified"] if c in s0.columns]]
              .rename(columns={"GP": "GP_qs", "GSAx_total": "MP-GSAx", "qualified": "qual_qs"}) if (not s0.empty and sk) else pd.DataFrame())
        t0 = load_qg_tiered_by_season(qg_scope_suffix)
        qgt = (t0[t0["season"] == sk][[c for c in ["goalie_id", "goalie_name", "GP", "QG_pct_s", "QG_pct_b", "qualified"] if c in t0.columns]]
               .rename(columns={"GP": "GP_qg", "qualified": "qual_qg"}) if (not t0.empty and sk) else pd.DataFrame())

    frames = [f for f in (nfi, qn, qs, qgt) if not f.empty]
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
    gp_cols = [c for c in ("GP_nfi", "GP_qn", "GP_qs", "GP_qg") if c in base.columns]
    base["GP"] = base[gp_cols].bfill(axis=1).iloc[:, 0] if gp_cols else np.nan
    base["Team"] = base["team"] if "team" in base.columns else np.nan
    # MoneyPuck all-shot GSAx per 60 ≈ cumulative all-shot GSAx / GP (goalies play
    # ~full games, so per-game ≈ per-60) — same approximation the drill-in uses.
    if "MP-GSAx" in base.columns:
        _gpn = pd.to_numeric(base["GP"], errors="coerce")
        base["MP-GSAx/60"] = np.where(_gpn > 0, base["MP-GSAx"] / _gpn, np.nan)
    _gid_of = dict(zip(base["Goalie"], base["goalie_id"]))   # name → id for drill-in

    c1, c2, c3, c4 = st.columns([1.8, 1.1, 1.1, 0.9])
    with c1:
        _gnames = sorted(base["Goalie"].dropna().unique().tolist())
        goalie_pick = st.selectbox(
            "Find a goalie", _gnames, index=None, placeholder="",
            key="goalies_name_pick",
            on_change=lambda: st.session_state.update(goalies_team="All"),
            help="Type to search by name; pick one to see their detail (trend + charts).")
    with c2:
        # Min Shots Faced FILTERS the list (hides small-sample goalies by default);
        # ranking is gated separately by each metric's qualified flag.
        if playoffs:
            _shdef, _shkey, _shmax = 50, "goalies_minshots_playoffs", 1500
        elif is_pooled:
            _shdef, _shkey, _shmax = 500, "goalies_minshots_pooled", 3000
        elif is_2yr:
            _shdef, _shkey, _shmax = 300, "goalies_minshots_2yr", 3000
        else:
            _shdef, _shkey, _shmax = 150, "goalies_minshots_season", 3000
        min_shots = st.slider("Min Shots Faced", 0, _shmax, _shdef, 50, key=_shkey)
    with c3:
        # Min GP — a SEPARATE, adjustable floor from the fixed 25-GP bar each of
        # QNFG%/QG%/sQS% requires to be individually ranked. Lowering this just
        # brings a low-GP goalie onto the leaderboard (still showing "(UR)" on
        # whichever of their columns don't clear that metric's own floor) — it
        # does not change what counts as "qualified" for ranking.
        _gpmax = 30 if playoffs else (100 if (is_pooled or is_2yr) else 82)
        _gpkey = f"goalies_mingp_{'playoffs' if playoffs else ('pooled' if is_pooled else ('2yr' if is_2yr else 'season'))}"
        min_gp = st.slider("Min GP", 0, _gpmax, min(25, _gpmax), 1, key=_gpkey)
    with c4:
        _gteam_opts = ["All"] + sorted(base["Team"].dropna().unique().tolist())
        # Picking a team exits any drill-in and clears the goalie search (mutually
        # exclusive views).
        goalie_team = st.selectbox(
            "Team", _gteam_opts, key="goalies_team",
            on_change=lambda: st.session_state.update(_gl_drill=None, goalies_name_pick=None))

    # Drill-in (via "Find a goalie" OR a clicked row) — either collapses the
    # leaderboard to just that goalie's detail.
    def _goalie_drill(gid, label):
        _set_dl_title(label)                     # name downloaded charts
        st.markdown(f"### {label}")
        if playoffs:
            _render_goalie_playoff_summary(int(gid), qg_scope_suffix, qg_starter)
        else:
            _render_goalie_profile(int(gid), qg_scope_suffix, qg_starter, season_label)

    if goalie_pick is not None:
        st.session_state["_gl_drill"] = None     # an explicit search overrides a click
        gid = _gid_of.get(goalie_pick)
        if gid is not None and pd.notna(gid):
            st.button("← Back to leaderboard", key="gl_back_search",
                      on_click=lambda: st.session_state.update(goalies_name_pick=None))
            _goalie_drill(int(gid), goalie_pick)
        return
    _gl_drill = st.session_state.get("_gl_drill")
    if _gl_drill is not None:
        if st.button("← Back to leaderboard", key="gl_back"):
            st.session_state["_gl_drill"] = None
            st.rerun()
        _goalie_drill(int(_gl_drill), name_map.get(int(_gl_drill), str(_gl_drill)))
        return

    # Per-metric qualification — the Min Shots Faced / Min GP sliders ARE the
    # qualifying floor (each metric is ranked only over goalies clearing its own
    # `qual_*` bar; others render "(UR)"). Playoffs have no floor → rank everyone.
    if playoffs:
        for _qc in ("qual_gsax", "qual_qn", "qual_qs", "qual_qg"):
            base[_qc] = True
    else:
        base["qual_gsax"] = (base.get("total_faced", pd.Series(np.nan, index=base.index))
                             .fillna(0) >= min_shots)
        for _qc, _col in (("qual_qn", "GP_qn"), ("qual_qs", "GP_qs"), ("qual_qg", "GP_qg")):
            base[_qc] = (base.get(_col, pd.Series(np.nan, index=base.index))
                        .fillna(0) >= min_gp)

    # Ranking pool = ALL goalies (set before the Min-Shots filter so it never
    # changes ranks a second time via visibility); each metric is ranked only
    # over goalies that clear ITS qualifying bar (per the `qual_*` flags above)
    # — others render "(UR)". The Min-Shots/Min-GP sliders then also filter which
    # rows are shown, and the Team filter narrows further.
    rank_pool = base.copy()
    base = base[base["total_faced"].fillna(0) >= min_shots]
    base = base[base["GP"].fillna(0) >= min_gp]
    if goalie_team != "All":
        base = base[base["Team"] == goalie_team]
    if base.empty:
        st.info("No goalies match the current filters.")
        return

    def _qnfs_ci(r):
        if pd.isna(r.get("QNFS_lo")) or pd.isna(r.get("QNFS_hi")):
            return np.nan
        return f"({r['QNFS_lo']:.1f}–{r['QNFS_hi']:.1f})"
    base["QNFG 95% CI"] = base.apply(_qnfs_ci, axis=1)
    _gren = {
        "NFIG60": "NFI-GSAx/60", "NFISV": "NFI SV%", "QNFS_pct": "QNFG%",
        "QS_GSAx_pct": "QG%", "QS_GSAx_lo": "QG (95% lower)",
    }
    base = base.rename(columns=_gren)
    rank_pool = rank_pool.rename(columns=_gren)

    # sQS%: both are always in the data (computed for every goalie
    # against both baselines); the toggle just picks which ONE is displayed,
    # under a header naming exactly which baseline is active.
    _qg_active, _qg_other = ("QG_pct_s", "QG_pct_b") if qg_starter else ("QG_pct_b", "QG_pct_s")
    _qg_label = "sQS%"
    for _df in (base, rank_pool):
        if _qg_active in _df.columns:
            _df.rename(columns={_qg_active: _qg_label}, inplace=True)
        if _qg_other in _df.columns:
            _df.drop(columns=[_qg_other], inplace=True)

    # Goalies qualified for any metric lead (sorted by NFI-GSAx/60); pure-noise
    # small samples sink to the bottom rather than topping the leaderboard.
    _qual_cols = [c for c in ("qual_gsax", "qual_qn", "qual_qs", "qual_qg") if c in base.columns]
    base["_qual_any"] = base[_qual_cols].any(axis=1)
    base = base.sort_values(["_qual_any", "NFI-GSAx/60"], ascending=[False, False],
                            na_position="last").reset_index(drop=True)

    cols = ["Goalie", "Team", "GP", "NFI-GSAx/60", "MP-GSAx/60", "MP-GSAx",
            "NFI SV%", "QNFG%", "QG%", _qg_label]
    disp = base[[c for c in cols if c in base.columns]].copy()

    fmt = {}
    if "NFI-GSAx/60" in disp:
        fmt["NFI-GSAx/60"] = lambda x: "—" if pd.isna(x) else f"{x:+.3f}"
    if "MP-GSAx/60" in disp:
        fmt["MP-GSAx/60"] = lambda x: "—" if pd.isna(x) else f"{x:+.3f}"
    if "MP-GSAx" in disp:
        fmt["MP-GSAx"] = lambda x: "—" if pd.isna(x) else f"{x:+.1f}"
    if "NFI SV%" in disp:
        fmt["NFI SV%"] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("QNFG%", "QG%", "QG (95% lower)", _qg_label):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}%"
    if "GP" in disp:
        fmt["GP"] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    # Each metric ranked only over goalies qualified for THAT metric (others UR).
    # NFI SV% shares NFI-GSAx's qualifying cohort — same CNFI+MNFI shots-faced
    # denominator, just an unadjusted rate instead of an xG-relative one.
    # MP-GSAx (all-shot GSAx) comes from the same QS/quality-start pipeline as
    # QG%, so it's gated on that pipeline's own cohort (qual_qs, ≥25 GP) — same
    # source, same floor, and consistent with the drill-in's per-metric ranks.
    _metric_qual = {"NFI-GSAx/60": "qual_gsax", "NFI SV%": "qual_gsax",
                     "MP-GSAx/60": "qual_qs", "MP-GSAx": "qual_qs",
                     "QNFG%": "qual_qn", "QG%": "qual_qs", _qg_label: "qual_qg"}
    for _m, _qc in _metric_qual.items():
        if _m not in disp.columns or _qc not in base.columns:
            continue
        _coh = rank_pool[rank_pool[_qc]]
        _tc = (_coh[_coh["Team"] == goalie_team] if goalie_team != "All" else None)
        _apply_ranks(disp, fmt, _coh, [_m], second_cohort=_tc, mark_unranked=True,
                     qualified=base[_qc])
    if goalie_team != "All":
        st.caption(f"Each metric shows **(league rank / {goalie_team} rank)**. Every "
                   f"goalie is listed; a metric is ranked only if the goalie clears its "
                   f"bar — **NFI-GSAx** ≥ {min_shots:,} shots faced, **QNFG / QG / "
                   f"{_qg_label}** ≥ {min_gp} GP — else **(UR)** = unranked. The Min Shots "
                   "Faced / Min GP sliders set these bars directly.")
    else:
        st.caption(f"Each metric shows its **(rank)**. Every goalie is listed; a metric "
                   f"is ranked only if the goalie clears its bar — **NFI-GSAx** "
                   f"≥ {min_shots:,} shots faced, **QNFG / QG / {_qg_label}** ≥ {min_gp} GP "
                   "— else **(UR)** = unranked. The Min Shots Faced / Min GP sliders set "
                   "these bars directly.")
    st.caption("**QG** = goals-saved-above-expected, as a game rate (formerly GQG) — "
               "the share of a goalie's games where their all-shot GSAx ≥ 0 (beat "
               "expected on a danger/xG-weighted basis), not raw save%.")
    st.caption("**NFI SV%** = raw (unadjusted) save% on the net-front danger-zone shot "
               "set only (CNFI+MNFI shots faced) — a sanity-check stat, not shot-quality "
               "adjusted like NFI-GSAx.")
    st.caption("**MP-GSAx** = MoneyPuck all-shot goals-saved-above-expected (total, and "
               "per-60) — the all-shot counterpart to net-front **NFI-GSAx**; ranked on the "
               "same shot-qualified cohort.")
    if not playoffs:
        _bar_txt = ("that season's" if not (is_pooled or is_2yr)
                    else f"each season's ({'2-season pool' if is_2yr else '4-season pool'})")
        st.caption(
            f"**{_qg_label}** = share of games where per-game save% ({qg_scope_label} shots "
            f"on goal) cleared {_bar_txt} **{'starter' if qg_starter else 'backup'}-tier** "
            f"baseline — the volume-weighted save% of that season's top-32-GP (starter) or "
            f"next-32-GP (backup) goalies. A traditional Quality Start, but against a "
            f"population-specific bar recomputed every season instead of one fixed "
            f"league-average line. Toggle above switches baseline/scope; the leaderboard "
            f"always shows the currently-selected one."
        )
    _sort_hint()
    st.caption("Click a row to open that goalie's detail (collapses the list).")
    _ggen = st.session_state.get("_gl_tbl_gen", 0)
    _gevent = _show_df(disp.style.format(fmt, na_rep="—"), hide_index=True,
                       on_select="rerun", selection_mode="single-row",
                       key=f"goalies_tbl_{_ggen}")
    _goalie_scope = "all playoffs (2022-2025 pooled)" if playoffs else season_label
    st.caption(
        f"{len(disp)} goalies (≥ {min_shots:,} shots faced, ≥ {min_gp} GP) · "
        f"{_goalie_scope} · sorted by NFI-GSAx/60 descending · (UR) = below that "
        "metric's own ranking floor — lower Min GP / Min Shots Faced to bring more "
        "goalies onto the list"
    )
    if playoffs:
        st.markdown(
            f"<p style='color:{PALETTE['text_secondary']}; font-size:0.82rem; max-width:62rem;'>"
            "Pooled across all playoff games (2022-23 → 2024-25). Every goalie with "
            "playoff data is shown and ranked — no qualifying floor is applied to the "
            "small playoff samples. Per-game metric definitions (QNFG ≥3 net-front "
            "shots/game; QG ≥10 shots/game; sQS% ≥10 shots/game) are retained. "
            "sQS% grade each playoff game against that season's regular-season "
            "baseline, not a fresh playoff-only tier split.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<p style='color:{PALETTE['text_secondary']}; font-size:0.82rem; max-width:62rem;'>"
            "Goalies above the Min-Shots filter are shown with a value for each "
            "metric (lower it toward 0 to see everyone). A metric is RANKED only "
            "when the goalie clears its qualifying minimum — otherwise the cell "
            "reads (UR), unranked. Floors differ by metric: NFI-GSAx ≥300 net-front "
            "shots pooled / ≥100 per season; QNFG% ≥25 GP/season (≥3 net-front "
            "shots/game); QG ≥25 GP/season (≥10 shots/game); sQS% ≥25 GP/season "
            "(≥10 shots/game). The regenerated data "
            "files carry a <code>qualified</code> flag per metric for downstream "
            "analysis.</p>",
            unsafe_allow_html=True,
        )

    # Row click → drill into that goalie (collapse the list); bump the table key
    # so it re-renders without a stale selection when we come back.
    _grows = getattr(getattr(_gevent, "selection", None), "rows", None)
    if _grows:
        _cgid = base.iloc[_grows[0]]["goalie_id"]
        if pd.notna(_cgid):
            st.session_state["_gl_drill"] = int(_cgid)
            st.session_state["_gl_tbl_gen"] = _ggen + 1
            st.rerun()


def _playoff_sv_baseline(n: pd.DataFrame) -> float:
    """Simple shots-weighted average NFI save% across ALL playoff goalies (no
    starter-tier subset) — playoffs are only 16 teams, so there's no
    meaningful bench population to exclude the way the regular-season
    top-32-GP starter tier does. Same weighting method as
    load_nfi_sv_baseline(), just the whole playoff pool instead of one
    regular season. Used as the cutoff line for both NFI SV% and (for display
    only) the sQS% baseline note."""
    if n.empty or "NFI_save_pct" not in n.columns:
        return np.nan
    w = pd.to_numeric(n.get("total_faced"), errors="coerce")
    v = pd.to_numeric(n["NFI_save_pct"], errors="coerce")
    m = v.notna() & (w > 0)
    return float(np.average(v[m], weights=w[m])) if m.any() else np.nan


def _render_goalie_playoff_summary(gid: int, qg_scope_suffix: str = "", qg_starter: bool = True) -> None:
    """Pooled all-playoffs metric summary for one goalie (playoff Detail view)."""
    qg_label = "sQS%"
    qg_col = "QG_pct_s"
    n, q, s = load_goalie_nfi_playoffs(), load_qnfs_playoffs(), load_qs_playoffs()
    g = load_qg_tiered_playoffs(qg_scope_suffix)

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
        if kind == "sv":
            return f"{v * 100:.1f}%"
        return f"{int(v):,}"

    # League rank (all playoff goalies, no floor) alongside each performance
    # value, same "(rank)" convention as every other ranked column in the app.
    def rk(df, col, val, lower=False):
        if df.empty or col not in df.columns or pd.isna(val):
            return None
        return _league_rank(df[col], val, lower=lower)

    v_gsax60 = pick(n, "GSAx_per60")
    v_sv = pick(n, "NFI_save_pct")
    v_qnfg = pick(q, "QNFS_pct")
    v_qg = pick(s, "QS_GSAx_pct")
    v_sqs = pick(g, qg_col)

    def with_rank(txt, r):
        return f"{txt} ({r})" if r is not None else txt

    items = [
        ("NFI-GSAx/60", with_rank(fmt(v_gsax60, "gsax"), rk(n, "GSAx_per60", v_gsax60))),
        ("NFI SV%", with_rank(fmt(v_sv, "sv"), rk(n, "NFI_save_pct", v_sv))),
        ("QNFG%", with_rank(fmt(v_qnfg, "pct"), rk(q, "QNFS_pct", v_qnfg))),
        ("QG%", with_rank(fmt(v_qg, "pct"), rk(s, "QS_GSAx_pct", v_qg))),
        (qg_label, with_rank(fmt(v_sqs, "pct"), rk(g, qg_col, v_sqs))),
        ("Games (GSAx)", fmt(pick(n, "games"), "int")),
        ("Shots faced", fmt(pick(n, "total_faced"), "int")),
    ]
    st.caption(f"**{name}** · pooled playoffs · **(rank)** among all playoff goalies.")
    _show_df(pd.DataFrame(items, columns=["Metric", "Value"]),
                 width="stretch", hide_index=True)

    # Consistency % bar — QNFG%/QG% vs 50, sQS% vs the all-playoff-goalie average
    # sQS%, and NFI SV% vs a simple all-playoff-goalie shots-weighted average
    # save% (16 teams' worth of playoff goalies — no top-N "starter tier" subset).
    _sv_bl = _playoff_sv_baseline(n)
    # League-average playoff sQS% (GP-weighted where GP is available).
    _sqs_bl = np.nan
    if not g.empty and "QG_pct_s" in g.columns:
        _sv = pd.to_numeric(g["QG_pct_s"], errors="coerce")
        _w = pd.to_numeric(g.get("GP"), errors="coerce") if "GP" in g.columns else None
        _mm = _sv.notna() & ((_w > 0) if _w is not None else True)
        if _mm.any():
            _sqs_bl = (float(np.average(_sv[_mm], weights=_w[_mm])) if _w is not None
                       else float(_sv[_mm].mean()))
    if pd.notna(_sv_bl):
        st.markdown(
            f"<div style='color:{_CHART_THIRD}; font-size:0.85rem; margin:0.1rem 0 0.4rem;'>"
            f"<b>Playoff save% baseline</b> (shots-weighted average, all playoff "
            f"goalies) — {_sv_bl * 100:.1f}%. Used as the cutoff for NFI SV%; "
            f"{qg_label} uses the all-playoff-goalie average sQS%"
            + (f" ({_sqs_bl:.0f}%)" if pd.notna(_sqs_bl) else "") + ".</div>",
            unsafe_allow_html=True)
    if any(pd.notna(v) for v in (v_qnfg, v_qg, v_sqs, v_sv)):
        _row = pd.Series({"Season": "Playoffs", "QNFG%": v_qnfg,
                          "QG%": v_qg, qg_label: v_sqs, "NFI SV%": v_sv})
        _goalie_consistency_bar(_row, qg_label, _sv_bl,
                               sqs_baseline=_sqs_bl if pd.notna(_sqs_bl) else None)


# ---------------------------------------------------------------------------
# Trade Analyzer tab — side-by-side player detail data (no charts), up to 5
# ---------------------------------------------------------------------------
TRADE_MAX_PLAYERS = 5


def _render_trade_goalies(playoffs: bool) -> None:
    """Goalie side of the Trade Analyzer — side-by-side goalie detail tables."""
    names = {}
    for _ld in (load_goalie_nfi_by_season, load_qnfs_by_season, load_qs_by_season):
        d = _ld()
        if not d.empty and "goalie_name" in d.columns:
            for gid, nm in zip(d["goalie_id"], d["goalie_name"]):
                if pd.notna(gid) and pd.notna(nm):
                    names[int(gid)] = str(nm)
    if not names:
        st.error("Goalie data not found.")
        return
    gid_list = sorted(names, key=lambda g: names[g])
    st.caption(f"Compare up to {TRADE_MAX_PLAYERS} goalies' detail data side by side "
               + ("(pooled playoff view)." if playoffs else "(no charts)."))
    sel = st.multiselect(
        f"Goalies (max {TRADE_MAX_PLAYERS})", gid_list,
        format_func=lambda g: names.get(g, str(g)),
        max_selections=TRADE_MAX_PLAYERS, key="trade_goalies")
    if not sel:
        st.caption(f"Pick up to {TRADE_MAX_PLAYERS} goalies to compare.")
        return
    qg_starter, qg_scope_suffix, _ = _qg_toggle_state()
    if playoffs:
        for gid in sel:
            _render_goalie_playoff_summary(int(gid), qg_scope_suffix, qg_starter)
        return
    st.caption("Each value shows its **(rank)** — league rank among all goalies "
               "that season (2yr row ranks within the 2-season pool).")
    for gid in sel:
        st.markdown(f"**{names.get(int(gid), str(gid))}**")
        disp, _, _ = _goalie_profile_table(int(gid), qg_scope_suffix, qg_starter)
        if disp.empty:
            st.info("No per-season data available for this goalie.")
        else:
            _show_df(disp, width="stretch", hide_index=True)


def render_trade_analyzer() -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Trade Analyzer</h2>",
        unsafe_allow_html=True,
    )
    season_label, game_type = render_scoped_filters("trade")
    if _block_ref_only(season_label):
        return
    _set_dl_title(None)
    playoffs = game_type == "Playoffs"

    if st.radio("Compare", ["Skaters", "Goalies"], horizontal=True,
                key="trade_mode") == "Goalies":
        _render_trade_goalies(playoffs)
        return

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

    # Data first: per-player detail tables.
    cohort = st.radio("Rank against", ["All skaters", "Same position"],
                      horizontal=True, key="trade_rank_cohort")
    same_pos = cohort != "All skaters"
    _cohort_txt = ("each player's own position group (F vs F, D vs D)"
                   if same_pos else "all skaters")
    st.caption(f"Each value shows **(league / team)** rank — among {_cohort_txt} "
               "league-wide, then within the player's own team that season. "
               "NFI-S/60 (shots against): lowest = #1.")
    for pid in sel:
        st.markdown(f"**{plabel.get(int(pid), str(pid))}**")
        disp, _, _ = _player_profile_table(int(pid), same_pos=same_pos, team="__own__")
        if disp.empty:
            st.info("No per-season data available for this player.")
        else:
            _show_df(disp, width="stretch", hide_index=True)

    # Then the side-by-side comparison charts — one panel per player. Bars = the
    # filter's year; the secondary line image = QG % over time. NFI and xG are
    # split into separate charts (each family read on its own basis) rather
    # than mixed together, matching the single-player drill-in's split.
    _seasons_all = [SEASON_DISPLAY.get(s, s) for s in PROFILE_SEASONS] + ["2yr avg (24-26)"]
    _cmp_yr = _default_qg_year(season_label, _seasons_all)
    _trends, _pv, _zv = {}, {}, {}
    for pid in sel:
        _nm = plabel.get(int(pid), str(pid))
        _tr = _player_trend(int(pid))
        if not _tr.empty:
            _trends[_nm] = _tr
            _pv[_nm] = _player_qg_vals(int(pid), _tr, _cmp_yr)
            _zv[_nm] = _player_zone_vals(int(pid), _tr, _cmp_yr)
    if _pv:
        _set_dl_title(" vs ".join(_pv.keys()))
        # Same 12-metric Raw+Rel paired set (and Attack/Suppress/Overall order)
        # as the single-player drill-in bars, faceted one panel per player.
        _qg_bar_chart_compare(
            _pv, _cmp_yr, metrics=_QG_BAR_ORDER_NFI,
            caption="**NFI** Quality-Games % vs the **50% baseline** "
                    "(Raw next to its Relative counterpart), one panel per player.",
            dl_name="Trade-QG-bars-NFI", title="NFI Quality Games %")
        _qg_bar_chart_compare(
            _pv, _cmp_yr, metrics=_QG_BAR_ORDER_XG,
            caption="**xG (MoneyPuck)** Quality-Games % vs the "
                    "**50% baseline** (Raw next to its Relative counterpart), one panel "
                    "per player.", dl_name="Trade-QG-bars-xG",
            title="xG (MoneyPuck) Quality Games %")
        # Zone Impact bar (OZI/DZI/NZI/TZI, 0-100, 50 = position average) — same
        # metric the drill-in shows, faceted per player.
        if any(any(pd.notna(v) for v in zv.values()) for zv in _zv.values()):
            _qg_bar_chart_compare(
                _zv, _cmp_yr, metrics=_ZONE_BAR_METRICS_ORDER,
                caption="**Zone Impact** index (OZI/DZI/NZI/TZI) vs the "
                        "**50 baseline** (50 = league-average for the position), one panel "
                        "per player.", dl_name="Trade-Zone-bars", title="Zone Impact Index")
    # Year-over-year line graphs behind a toggle (off by default), mirroring the
    # player drill-in — lead with the bars + scatters, reveal the season-by-season
    # lines on demand. Same full metric set as the drill-in (Quality Games, xG,
    # Net-Front Impact, Zone Impact, EDGE), one panel per player.
    if _trends:
        st.checkbox("Show year-over-year graphs", key="trade_show_yoy", value=False)
        if st.session_state.get("trade_show_yoy"):
            _set_dl_title(" vs ".join(_trends.keys()))
            # Quality Games (split NFI / xG by model)
            _qg_line_chart_compare(
                _trends, cols=_QG_LINE_ORDER_NFI,
                caption="**NFI** Quality Games % over time, per player — colour = aspect "
                        "(overall/offense/defense/relative); relative (**Rel**) dashed.",
                dl_name="Trade-QG-line-NFI")
            _qg_line_chart_compare(
                _trends, cols=_QG_LINE_ORDER_XG,
                caption="**xG (MoneyPuck)** Quality Games % over time, per player — colour = "
                        "aspect (overall/offense/defense/relative); relative (**Rel**) dashed.",
                dl_name="Trade-QG-line-xG")
            # xG family
            _trade_line_compare(_trends, ["xGF/60", "xGA/60"],
                "On-ice xG per 60 (xGF/60, xGA/60) over time, per player.",
                "Trade-xG-per60")
            _trade_line_compare(_trends, ["RelxG%", "RelxG-F%", "RelxG-A%"],
                "Relative xG % (RelxG%, RelxG-F%, RelxG-A%) over time, per player.",
                "Trade-RelxG")
            _trade_line_compare(_trends, ["PDOxG"],
                "PDOxG (5v5) — luck net of shot quality — over time, per player.",
                "Trade-PDOxG-line")
            # Net-Front Impact family
            _trade_line_compare(_trends, ["RelNFI%", "RelNFI-A%", "RelNFI-S%"],
                "RelNFI family (RelNFI%, RelNFI-A%, RelNFI-S%) over time, per player.",
                "Trade-RelNFI")
            _trade_line_compare(_trends, ["NFI-A/60", "NFI-S/60"],
                "Raw net-front rate per 60 (NFI-A/60, NFI-S/60) over time, per player.",
                "Trade-NFI-rate")
            _trade_line_compare(_trends, ["NFI%"],
                "NFI% (net-front share) over time, per player.", "Trade-NFI-pct")
            # Zone Impact + zone starts
            _zone_line_chart_compare(_trends)
            _trade_line_compare(_trends, ["OZ Start%", "DZ Start%"],
                "D/O Zone Start% (faceoff-started 5v5 shifts) over time, per player.",
                "Trade-ZoneStart")
            # EDGE tracking
            _trade_line_compare(_trends, ["EDGE OZ%", "EDGE DZ%"],
                "EDGE Zone-Time % (OZ, DZ) over time, per player.", "Trade-EDGE-zone")
            _trade_line_compare(_trends, ["EDGE Top Speed"],
                "EDGE Top Speed (mph) over time, per player.", "Trade-EDGE-topspeed")
            _trade_line_compare(_trends, ["EDGE Bursts 20+"],
                "EDGE Speed Bursts (20+ mph, season total) over time, per player.",
                "Trade-EDGE-bursts")
            _trade_line_compare(_trends, ["EDGE Distance (mi)"],
                "EDGE Distance Skated (mi) over time, per player.", "Trade-EDGE-dist")

    # Scatters — ONLY the selected trade players (not their whole teams). All
    # five scatter types show; the two xG-based ones (PDO/xG%, NFI%/xG%) render
    # Raw xG% | Rel xG% side by side in one faceted image. Axis convention
    # matches the leaderboard scatters (xG on x, the paired metric on y).
    if sel:
        # Full league-wide frame (unfiltered) drives the axis DOMAINS so a
        # 2-player comparison is placed in real context — the same axis numbers
        # the leaderboard/team scatters use — instead of auto-zooming to just the
        # picked players and exaggerating a small real gap. _sf is the subset
        # actually plotted.
        _full = _team_scatter_frame(season_label or "4yr (2022-2026)")
        _sf = (_full[_full["player_id"].isin([int(p) for p in sel])]
               if not _full.empty else _full)
        st.markdown(f"<h3 style='color:{PALETTE['text']}; margin-top:1.5rem;'>Selected "
                    "Players — Scatters</h3>", unsafe_allow_html=True)
        st.caption("Only the selected players are plotted, but the axes span the full "
                   "league range (same scale as the leaderboard) so a small real gap "
                   "isn't exaggerated. The xG-based scatters show **Raw xG%** vs **Rel "
                   "xG%** side by side; the others have no relative variant.")
        _tyl = season_label or "4yr (2022-2026)"
        if {"PDOxG", "xG%", "RelxG%"}.issubset(_sf.columns):
            st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>PDOxG vs "
                        "xG%</h4>", unsafe_allow_html=True)
            _trade_rawrel_scatter(_sf, "PDOxG", "PDOxG", "Trade-PDOxG-vs-xG", _full)
        if {"NFI%", "xG%", "RelxG%"}.issubset(_sf.columns):
            st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>NFI% vs "
                        "xG%</h4>", unsafe_allow_html=True)
            _trade_rawrel_scatter(_sf, "NFI%", "NFI%", "Trade-NFI-vs-xG", _full)
        if {"PDOxG", "NFI%"}.issubset(_sf.columns):
            st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>PDOxG vs "
                        "NFI%</h4>", unsafe_allow_html=True)
            _pdo_nfi_scatter(_sf, True, dl_suffix="-trade", domain_df=_full, year_label=_tyl)
        if {"EDGE DZ%", "EDGE OZ%"}.issubset(_sf.columns):
            st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: "
                        "D-Zone vs O-Zone Time%</h4>", unsafe_allow_html=True)
            _edge_zone_scatter(_sf, True, dl_suffix="-trade", domain_df=_full, year_label=_tyl)
        if {"DZ Start%", "OZ Start%"}.issubset(_sf.columns):
            st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>Zone "
                        "Starts: D-Zone vs O-Zone</h4>", unsafe_allow_html=True)
            _zone_start_scatter(_sf, True, dl_suffix="-trade", domain_df=_full, year_label=_tyl)
        if {"EDGE OZ%", "DZ Start%", "NZ Start%"}.issubset(_sf.columns):
            st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EZI: "
                        "EDGE O-Zone Time vs Non-O-Zone Starts</h4>", unsafe_allow_html=True)
            _ezi_scatter(_sf, True, dl_suffix="-trade", domain_df=_full, year_label=_tyl)
        if {"EDGE Top Speed", "EDGE Bursts 20+"}.issubset(_sf.columns):
            _edge_speed_scatter(_sf, True, dl_suffix="-trade", domain_df=_full, year_label=_tyl,
                               league_df=_full)


def _trade_rawrel_scatter(df_sel: pd.DataFrame, y_col: str, y_title: str,
                          dl_name: str, full_df: pd.DataFrame = None) -> None:
    """Two SEPARATE scatters for the selected trade players — Raw xG% and
    Rel xG%, each its own image with its own download button (previously one
    faceted image; split per user request so each half can be viewed/posted on
    its own). x = the xG measure (raw xG% or RelxG%), y = y_col (PDO or NFI%),
    matching the leaderboard scatters' xG-on-x orientation. full_df (the
    league-wide frame) sets each axis domain so the picked players sit in real
    context rather than auto-zoomed to their own span."""
    import altair as alt
    if df_sel is None or df_sel.empty or not {y_col, "xG%", "RelxG%", "Player"}.issubset(df_sel.columns):
        st.caption("No data for the selected players in this scope.")
        return
    _dom = full_df if (full_df is not None and not full_df.empty) else df_sel
    _ydom = _tight_domain(_dom[y_col].dropna() if y_col in _dom else df_sel[y_col].dropna(),
                          pad_frac=0.15, min_pad=1e-6)
    for meas, col, suf in (("Raw xG%", "xG%", "-raw"), ("Rel xG%", "RelxG%", "-rel")):
        rows = [{"Player": r["Player"], "yv": float(r[y_col]), "xv": float(r[col]),
                "_label": str(r["Player"]).split()[-1]}
               for _, r in df_sel.iterrows() if pd.notna(r.get(y_col)) and pd.notna(r.get(col))]
        if not rows:
            continue
        d = pd.DataFrame(rows)
        _xdom = _tight_domain(_dom[col].dropna() if col in _dom else d["xv"],
                              pad_frac=0.15, min_pad=1e-6)
        st.markdown(f"<h5 style='color:{PALETTE['text']}; margin-top:0.6rem;'>{meas}</h5>",
                   unsafe_allow_html=True)
        base = alt.Chart(d).encode(
            x=alt.X("xv:Q", title=meas, scale=alt.Scale(domain=_xdom, zero=False)),
            y=alt.Y("yv:Q", title=y_title, scale=alt.Scale(domain=_ydom, zero=False)))
        pts = base.mark_circle(size=150, opacity=0.8, color=PALETTE["blue"]).encode(
            tooltip=[alt.Tooltip("Player:N"), alt.Tooltip("xv:Q", format=".2f", title=meas),
                     alt.Tooltip("yv:Q", format=".3f", title=y_title)])
        txt = base.mark_text(align="left", dx=8, dy=-4, fontSize=12, fontWeight="bold",
                             color=PALETTE["orange"]).encode(text="_label:N")
        chart = (pts + txt).properties(width=340, height=320)
        _show_chart(chart, dl_name=dl_name + suf, brand_width=340)


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


def _ref_league_avg(df: pd.DataFrame, min_games: int = REF_MIN_GAMES) -> dict:
    """Per-referee table → league-average rate for every metric column,
    computed over referees clearing the min-games floor (mirrors the league
    table's own highlighted LEAGUE AVERAGE row). Shared by the league table
    and the single-referee bar chart so both quote the same number."""
    tbl = _ref_table(df)
    tbl = tbl[tbl["Games"] >= min_games]
    metric_cols = ["Pen/Game", "Home Pen%", "Away Pen%"] + [f"{t}/G" for t in REF_TYPES]
    return {c: float(tbl[c].mean()) for c in metric_cols if c in tbl.columns}


def _ref_bar_chart(ref: str, df: pd.DataFrame, season_label: str,
                   side: str = "All", team: str = None) -> None:
    """One referee's penalty-call rate per game, by type, as grouped bars from a
    0 baseline: a BLUE bar for the referee (darker blue the higher the rate) next
    to an ORANGE comparison-average bar. The comparison is the LEAGUE average with
    no team filter, or the TEAM average (all referees' rate against that team) when
    a team is selected. `side` (All/Home/Away) restricts to penalties on the home/
    away team (no team) or to the team's home/away games (team filtered)."""
    import altair as alt
    order = list(REF_TYPES)   # one group per penalty type (no overall "All" bar)
    sc = side.lower()   # "home" / "away" / "all"

    if team:
        # Team view: penalties called AGAINST the team; side picks the team's
        # home vs away games. Comparison = the team's own average (all refs).
        gscope = df[(df["home_team"] == team) | (df["away_team"] == team)]
        if sc == "home":
            gscope = gscope[gscope["home_team"] == team]
        elif sc == "away":
            gscope = gscope[gscope["away_team"] == team]
        ref_games = gscope[gscope["ref"] == ref]["game_id"].nunique()
        ref_pens = gscope[(gscope["ref"] == ref) & (gscope["penalized_team"] == team)]
        # Team average = POOLED penalties-taken-per-game (matches the per-team
        # Table B's "penalties taken per game"): unique penalties (exploded → /2)
        # over the team's distinct games in scope.
        avg_pens = gscope[gscope["penalized_team"] == team]
        avg_games = gscope["game_id"].nunique()
        avg_label, scope_txt = f"{team} average", f" vs **{team}**"

        def _avg_rate(pt):
            ap = avg_pens if pt is None else avg_pens[avg_pens["penalty_type"] == pt]
            return ((len(ap) / 2) / avg_games) if avg_games else np.nan
    else:
        # League view: every penalty in the referee's games; side picks penalties
        # on the home vs away team.
        dside = df if sc == "all" else df[df["home_or_away"] == sc]
        ref_games = df[df["ref"] == ref]["game_id"].nunique()
        ref_pens = dside[dside["ref"] == ref]
        avg_label, scope_txt = "League average", ""
        # League average = MEAN of each qualifying referee's own per-game rate
        # (refs with >= REF_MIN_GAMES) — the same basis as the league table's
        # highlighted LEAGUE AVERAGE row, so the bar matches the table.
        _ref_g = df.groupby("ref")["game_id"].nunique()
        _qrefs = _ref_g[_ref_g >= REF_MIN_GAMES].index

        def _avg_rate(pt):
            sub = dside if pt is None else dside[dside["penalty_type"] == pt]
            cnt = sub.groupby("ref").size()
            rates = [cnt.get(r, 0) / _ref_g[r] for r in _qrefs]
            return float(np.mean(rates)) if rates else np.nan

    if not ref_games:
        st.caption(f"{ref} has no games in this selection.")
        return
    rows = []
    for cat in order:
        pt = None if cat == "All" else cat
        rp = ref_pens if pt is None else ref_pens[ref_pens["penalty_type"] == pt]
        rows.append({"Metric": cat, "Series": "This referee",
                     "value": len(rp) / ref_games})
        rows.append({"Metric": cat, "Series": avg_label, "value": _avg_rate(pt)})
    d = pd.DataFrame(rows).dropna(subset=["value"])
    if d.empty:
        st.caption(f"No data for {ref} in this selection.")
        return

    # Both series are shaded light→dark by their rate — referee bars light-blue→
    # navy, average bars light-orange→burnt-orange. A SHARED max (over every bar)
    # drives the darkness, so shade = absolute rate and is comparable across the
    # two colours (a darker bar always means a higher per-game rate).
    _vmax = float(d["value"].max()) if len(d) else 0.0
    def _col(r):
        t = (r["value"] / _vmax) if _vmax > 0 else 0.0
        t = min(max(t, 0.0), 1.0)
        if r["Series"] == "This referee":
            return _hex_lerp("#BCD0E2", PALETTE["text"], t)   # light blue → navy
        # light orange → the original brand orange; a tight range so the fade is
        # subtle (the light end is already a fairly saturated orange, not a pale
        # peach) — keeps all average bars clearly in the brand-orange family.
        return _hex_lerp("#FFB88F", PALETTE["orange"], t)
    d["color"] = d.apply(_col, axis=1)
    _series_order = ["This referee", avg_label]

    # Legend labels: the blue swatch is the referee; the orange swatch is the
    # comparison average (league or team). The chart's own legend carries the
    # colour key, so the caption no longer spells the colours out.
    _avg_leg = "League average" if not team else f"{team} average"
    _leg_relabel = {"This referee": ref, avg_label: _avg_leg}
    d["Legend"] = d["Series"].map(_leg_relabel)
    _leg_order = [ref, _avg_leg]

    _side_txt = "" if sc == "all" else f" — {side} only"
    st.caption(f"**{ref}**{scope_txt} — penalty calls per game by type{_side_txt}. "
               "Each penalty type shows this referee next to the "
               f"{'league' if not team else team} average; darker shading = a higher rate.")
    _base = alt.Chart(d).encode(
        x=alt.X("Metric:N", sort=order,
                scale=alt.Scale(paddingInner=0.35, paddingOuter=0.2),
                axis=alt.Axis(labelAngle=0, title=None, labelFontWeight="bold")),
        # paddingInner=0 → the referee bar and the average bar in each group touch
        # (no gap); the gap between penalty types comes from the x band padding.
        xOffset=alt.XOffset("Series:N", sort=_series_order,
                            scale=alt.Scale(paddingInner=0.0, paddingOuter=0.0)),
        y=alt.Y("value:Q", title="Per game", scale=alt.Scale(zero=True)))
    bars = _base.mark_bar().encode(
        color=alt.Color("color:N", scale=None, legend=None),   # per-datum blue gradient
        tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("Legend:N", title=""),
                 alt.Tooltip("value:Q", format=".2f", title="Per game")])
    # Invisible layer whose sole job is to render a 2-swatch colour legend (the
    # bars use per-datum hex fills for the gradient, which can't emit a legend).
    _leg = alt.Chart(pd.DataFrame({"Legend": _leg_order})).mark_square(opacity=0).encode(
        color=alt.Color("Legend:N", sort=_leg_order,
                        scale=alt.Scale(domain=_leg_order,
                                        range=[PALETTE["blue"], PALETTE["orange"]]),
                        legend=alt.Legend(orient="top", title=None, symbolType="square",
                                          symbolSize=160, labelFontSize=12)))
    _ttl = f"{ref}{(' vs ' + team) if team else ''}"
    _ttl += f"{('' if sc == 'all' else ' (' + side + ')')} — Penalty Calls per Game — {season_label}"
    chart = alt.layer(bars, _leg).resolve_scale(color="independent").properties(
        title=alt.TitleParams(text=_ttl, color=PALETTE["text"], fontSize=13))
    _dl = f"Ref-bars-{ref.replace(' ', '-')}{('-' + team) if team else ''}{('' if sc == 'all' else '-' + side)}"
    _show_chart(chart, dl_name=_dl)


def _render_ref_league(df: pd.DataFrame, season_label: str):
    """League-wide referee table. The top row is the highlighted LEAGUE AVERAGE
    (plain values); every referee cell shows its value with an inline
    (± vs league average) bracket. Click a referee's row to drill into their
    penalty-call bar chart below. Returns the clicked referee's name, or None."""
    tbl = _ref_table(df)
    tbl = tbl[tbl["Games"] >= REF_MIN_GAMES].copy()
    if tbl.empty:
        st.info(f"No referees meet the {REF_MIN_GAMES}-game floor for this view.")
        return None
    metric_cols = ["Pen/Game", "Home Pen%", "Away Pen%"] + [f"{t}/G" for t in REF_TYPES]
    avg = _ref_league_avg(df)
    pct_cols = {"Home Pen%", "Away Pen%"}

    tbl = tbl.sort_values("Pen/Game", ascending=False).reset_index(drop=True)

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
    st.caption("Click a referee's row to see their penalty-call bar chart below.")
    _event = _show_df(disp.style.apply(_bold_avg, axis=1), width="stretch", hide_index=True,
                      on_select="rerun", selection_mode="single-row", key="ref_league_tbl")
    _sel = getattr(getattr(_event, "selection", None), "rows", None)
    if _sel and 0 <= _sel[0] < len(disp):
        _clicked = disp["Referee"].iloc[_sel[0]]
        if _clicked != "LEAGUE AVERAGE":
            return _clicked
    return None


def _render_ref_team(df: pd.DataFrame, season_label: str, team: str):
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
        if ta.empty:
            st.info(f"No referee worked ≥{REF_TEAM_MIN_GAMES} games involving {team}.")
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
            st.caption("Click a referee's row to see their Home/Away bar chart vs "
                      f"{team} below.")
            ta_disp = ta[cols].reset_index(drop=True)
            _event = _show_df(ta_disp, width="stretch", hide_index=True,
                              on_select="rerun", selection_mode="single-row",
                              key=f"ref_team_tbl_{team}")
            _sel = getattr(getattr(_event, "selection", None), "rows", None)

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
    _show_df(pd.DataFrame(brows), width="stretch", hide_index=True)

    if rows and not ta.empty and _sel and 0 <= _sel[0] < len(ta_disp):
        return ta_disp["Referee"].iloc[_sel[0]]
    return None


def render_referees() -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Referees</h2>",
        unsafe_allow_html=True,
    )
    season_label, game_type = render_scoped_filters("referees")
    _set_dl_title(None)
    # The Referees tab reads the SHARED global Season filter (the same one shown on
    # every tab).
    # Referee data only exists 2023-24+, so it supports the single seasons it has
    # plus its own "3yr (Referees only)" pool; for any other selection (2022-23,
    # 2yr, 4yr, Playoffs) it shows a "not available" notice.
    if game_type == "Playoffs":
        st.info("Referee data covers regular-season games only — switch Game type "
                "to Regular Season.")
        return
    # Map the global selection to referee seasons. Coverage is 2023-24+; the 2yr
    # pool (2024-25 + 2025-26) and the 3yr Referees pool are both inside coverage,
    # while 2022-23 and the 4yr pool include years with no referee data.
    if season_label == REF_ONLY_LABEL:
        _sel = set(REF_SEASON_INT.values())
    elif SEASON_KEY.get(season_label) == "pooled_2yr":
        _sel = {REF_SEASON_INT["2024-25"], REF_SEASON_INT["2025-26"]}
    elif season_label in REF_SEASON_INT:
        _sel = {REF_SEASON_INT[season_label]}
    else:
        st.info(f"Referee data isn't available for **{season_label}** "
                "(coverage is 2023-24 onward). Pick a single season from 2023-24 on, "
                f"the 2yr pool, or **{REF_ONLY_LABEL}** in the Season filter up top.")
        return

    df = load_ref_penalties()
    if df.empty:
        st.error("Referee data not found "
                 "(`Referees/output/all_teams_penalties_3seasons.csv`).")
        return
    df = df[df["season"].isin(_sel)]
    if df.empty:
        st.info("No referee data for this selection.")
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
    _refs_all = sorted(df["ref"].dropna().unique().tolist())
    c0, c1, c2 = st.columns([1.6, 0.9, 1.1])
    with c0:
        ref_pick = st.selectbox(
            "Find a referee", _refs_all, index=None, placeholder="", key="refs_search",
            help="Type to search by name; pick one to see their penalty-call chart.")
    with c1:
        side_sel = st.radio("Home / Away", ["All", "Home", "Away"], horizontal=True,
                            key="refs_side",
                            help="Filters the drill-in bar chart. With no team, Home/Away = "
                                 "penalties on the home vs away team. With a team, = the "
                                 "team's home vs away games.")
    with c2:
        team_sel = st.selectbox("Team", ["All teams"] + teams, key="refs_team")

    if ref_pick:
        st.session_state["_ref_drill"] = ref_pick

    if team_sel == "All teams":
        _clicked = _render_ref_league(df, season_label)
    else:
        _clicked = _render_ref_team(df, season_label, team_sel)
    if _clicked:
        st.session_state["_ref_drill"] = _clicked

    _drill = st.session_state.get("_ref_drill")
    if _drill and _drill not in _refs_all:
        _drill = None
        st.session_state["_ref_drill"] = None
    if _drill:
        st.markdown("---")
        _h, _b = st.columns([4, 1])
        with _h:
            st.markdown(f"### {_drill}")
        with _b:
            st.button("✕ Clear", key="refs_drill_clear",
                      on_click=lambda: st.session_state.update(
                          _ref_drill=None, refs_search=None))
        _set_dl_title(_drill)
        _team_arg = None if team_sel == "All teams" else team_sel
        _ref_bar_chart(_drill, df, season_label, side=side_sel, team=_team_arg)


# ---------------------------------------------------------------------------
# Global sidebar (Season + Game type — apply to every tab)
# ---------------------------------------------------------------------------
POOLED_4YR_LABEL = "4yr (2022-2026)"


def render_scoped_filters(scope: str, show_situation: bool = False) -> tuple[str, str]:
    """Season + game-type filters rendered INSIDE each tab (next to that tab's own
    filters), but kept GLOBAL: every tab writes to and reads from the same shared
    session_state, so changing the season on one tab changes it everywhere. Each
    tab gets its own widget keys (Streamlit renders every tab each run, so a single
    shared key would collide); on_change callbacks push the pick to the shared keys
    and each run pre-seeds this scope's widgets from them to stay in sync.

    Playoffs only ship the pooled view (single-playoff-year samples are too small),
    so when Playoffs is selected the Season box is locked to the 4-year pooled view,
    with the user's regular-season pick preserved for when they switch back."""
    st.session_state.setdefault("g_game_type", "Regular Season")
    st.session_state.setdefault("g_season_pick", "2025-26")
    _gt_key, _ss_key = f"g_gt_{scope}", f"g_ssn_{scope}"
    # Pre-seed this scope's widgets from the shared value (before instantiation).
    st.session_state[_gt_key] = st.session_state["g_game_type"]
    is_playoffs = st.session_state["g_game_type"] == "Playoffs"
    if not is_playoffs:
        st.session_state[_ss_key] = st.session_state["g_season_pick"]
    season_opts = list(SEASON_KEY.keys())

    def _sync_gt():
        st.session_state["g_game_type"] = st.session_state[_gt_key]

    def _sync_ssn():
        st.session_state["g_season_pick"] = st.session_state[_ss_key]

    _sit_key = f"g_sit_{scope}"

    def _sync_sit():
        st.session_state["g_situation"] = st.session_state[_sit_key]

    if show_situation:
        st.session_state.setdefault("g_situation", "5v5")
        st.session_state[_sit_key] = st.session_state["g_situation"]
        cols = st.columns([1.2, 1.7, 1.5])
    else:
        cols = st.columns([1.2, 2.4])
    c1, c2 = cols[0], cols[1]
    with c1:
        if is_playoffs:
            st.selectbox("Season", season_opts,
                         index=season_opts.index(POOLED_4YR_LABEL),
                         disabled=True, key=f"g_ssn_locked_{scope}")
            season = POOLED_4YR_LABEL
        else:
            season = st.selectbox("Season", season_opts, key=_ss_key,
                                  on_change=_sync_ssn)
    with c2:
        game_type = st.radio("Game type", ["Regular Season", "Playoffs"],
                             horizontal=True, key=_gt_key, on_change=_sync_gt)
    if show_situation:
        with cols[2]:
            st.selectbox("Situation", list(SITUATION_BUCKETS), key=_sit_key,
                         on_change=_sync_sit,
                         help="Applies to the 'Sit' metric columns / Situation-splits "
                              "(PP=5v4+5v3+4v3, PK=4v5+3v5+3v4). NFI/QG/Zone stay 5v5.")
    if game_type == "Playoffs":
        st.caption("Playoffs pool all seasons (2022-23 → 2024-25); the Season filter "
                   "is locked to the pooled view. Season & game type apply to all tabs.")
    else:
        st.caption("Season and game type apply across all tabs.")
    return season, game_type


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="HockeyROI — NHL Impact Analytics",
        page_icon="🏒",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    # The Vega "···" actions menu is KEPT — its "Save as PNG"/"Save as SVG" is now
    # how a chart is downloaded (rendered client-side in the browser, so it costs
    # the server nothing; the brand is baked into every chart's spec so the saved
    # image carries it). Fullscreen (expand) is also kept. Only Streamlit's OWN
    # native chart-hover toolbar button (aria-label="Show data") stays hidden — it
    # renders the chart's full source DataFrame as a table, including every column
    # passed to alt.Chart(d) even when only 2-4 are actually plotted, so hovering
    # never exposes more than what's drawn. Scoped to charts; st.dataframe toolbars
    # are untouched.
    _css = (
        "[data-testid='stElementContainer']:has([data-testid='stVegaLiteChart']) "
        "button[aria-label='Show data']{display:none !important;}"
    )
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)
    render_header()
    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    # The Season + Game-type filter now lives at the top of each tab (rendered by
    # render_scoped_filters), grouped with that tab's own filters but kept in sync
    # across tabs — rather than a standalone row above the tabs.
    (player_list_tab, goalie_list_tab,
     trade_tab, teams_tab, refs_tab, meth_tab) = st.tabs(TAB_LABELS)
    with player_list_tab:
        render_players()
    with goalie_list_tab:
        render_goalies()
    with trade_tab:
        render_trade_analyzer()
    with teams_tab:
        render_teams()
    with refs_tab:
        render_referees()
    with meth_tab:
        render_methodology()

    render_footer()


if __name__ == "__main__":
    main()
