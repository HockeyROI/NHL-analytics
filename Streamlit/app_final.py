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
        <div class="tagline">NHL Net-Front Impact &amp; Zone Analytics</div>
        <div style="color:#888888; font-size:0.85rem; margin-top:0.15rem;">
          Data through the 2025-26 season ·
          <a href="https://github.com/HockeyROI/NHL-analytics/blob/main/docs/METHODOLOGY.md" style="color:#2E7DC4;">methodology on GitHub</a>
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
TAB_LABELS = ["Players", "Teams", "Goalies", "Referees", "Methodology"]


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
    "Pooled (2022–2026)": "pooled",
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


def _build_players_frame(season_label: str) -> tuple[pd.DataFrame, bool]:
    """Return (long per-player frame, is_pooled). NFI + QG, plus NZI/DZI/OZI in
    the pooled view only (zone data has no season axis)."""
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
        zone = load_zone_pooled()
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
    return base, is_pooled


def render_players(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Players</h2>",
        unsafe_allow_html=True,
    )
    if game_type == "Playoffs":
        st.info("Playoff player metrics — coming soon.")
        return

    frame, is_pooled = _build_players_frame(season_label)
    if frame.empty:
        st.error("Player data not found "
                 "(`NFI/output/fully_adjusted/player_fully_adjusted.csv`).")
        return

    c1, c2, c3, c4 = st.columns([1.0, 1.5, 1.1, 1.5])
    with c1:
        pos = st.radio("Position", ["All", "F", "D"], horizontal=True, key="players_pos")
    with c2:
        toi_key = "players_toi_pooled" if is_pooled else "players_toi_season"
        default_toi = 2000 if is_pooled else 500
        min_toi = st.slider("Min ES TOI (min)", 0, 7500, default_toi, 50, key=toi_key)
    with c3:
        team_opts = ["All"] + sorted(frame["team"].dropna().unique().tolist())
        team_sel = st.selectbox("Team", team_opts, key="players_team")
    with c4:
        name_q = st.text_input("Player name contains", key="players_name").strip().lower()

    df = frame.copy()
    if pos in ("F", "D"):
        df = df[df["position"] == pos]
    else:
        df = df[df["position"].isin(["F", "D"])]
    df = df[df["toi_min"].fillna(0) >= min_toi]
    if team_sel != "All":
        df = df[df["team"] == team_sel]
    if name_q:
        df = df[df["player_name"].str.lower().str.contains(name_q, na=False)]
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
    df = df.rename(columns={
        "player_name": "Player", "position": "Pos", "team": "Team", "toi_min": "TOI",
        "NFI_pct": "NFI%", "RelNFI_pct": "RelNFI%",
        "RelNFI_F_pct": "RelNFI-A%", "RelNFI_A_pct": "RelNFI-S%",
        "xG_QG_pct": "xG_QG%", "NFI_QG_pct": "NFI_QG%",
    })

    # Always show the full column set (Compact view removed; Qual GP dropped).
    cols = ["Player", "Pos", "Team", "GP", "TOI", "NFI%", "RelNFI%", "RelNFI-A%",
            "RelNFI-S%", "NZI", "DZI", "OZI", "xG_QG%", "NFI_QG%"]
    if not is_pooled:  # zone metrics are pooled-only — hide for single-season views
        cols = [c for c in cols if c not in ("NZI", "DZI", "OZI")]
    cols = [c for c in cols if c in df.columns]
    disp = df[cols].copy()

    fmt = {}
    for c in ("NFI%", "xG_QG%", "NFI_QG%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("RelNFI%", "RelNFI-A%", "RelNFI-S%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:+.2f}"
    for c in ("NZI", "DZI", "OZI"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    if "TOI" in disp.columns:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    for c in ("GP",):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    st.dataframe(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)

    zone_note = (" · NZI/DZI/OZI pooled across all seasons" if is_pooled
                 else " · Zone Impact hidden (pooled-only) in single-season view")
    st.caption(
        f"{len(disp):,} players · {season_label} · sorted by RelNFI% descending · "
        f"min {min_toi:,} ES min{zone_note}"
    )


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
    """2025-26 post-audit Attack/Suppress per game (single-season snapshot)."""
    fp = REPO_ROOT / "NFI" / "output" / "team_nfi_verification_and_attack_suppress.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df["team"] = df["team"].replace({"ARI": "UTA"})
    return df[["team", "attack_per_game", "suppress_per_game"]].copy()


def _team_nfi_share(df: pd.DataFrame) -> np.ndarray:
    """Post-audit team NFI% = CNFI+MNFI Fenwick for / (for + against). Excludes FNFI."""
    ffor = df["CNFI_FF"] + df["MNFI_FF"]
    fagn = df["CNFI_FA"] + df["MNFI_FA"]
    denom = ffor + fagn
    return np.where(denom > 0, ffor / denom, np.nan)


def render_teams(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Teams</h2>",
        unsafe_allow_html=True,
    )
    if game_type == "Playoffs":
        st.info("Playoff team metrics — coming soon.")
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

    # Attack / Suppress events — 2025-26 snapshot only
    if season_label == "2025-26":
        a = load_team_attack_suppress().rename(
            columns={"attack_per_game": "Attack events", "suppress_per_game": "Suppress events"})
        team = team.merge(a, on="team", how="left")
    else:
        team["Attack events"] = np.nan
        team["Suppress events"] = np.nan

    for c in ("TOI", "xG_QG%", "NFI_QG%"):
        if c not in team.columns:
            team[c] = np.nan

    team = team.rename(columns={"team": "Team"})
    team = team.sort_values("NFI%", ascending=False, na_position="last").reset_index(drop=True)
    cols = ["Team", "GP", "TOI", "NFI%", "Attack events", "Suppress events",
            "xG_QG%", "NFI_QG%"]
    disp = team[[c for c in cols if c in team.columns]].copy()

    fmt = {}
    for c in ("NFI%", "xG_QG%", "NFI_QG%"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("Attack events", "Suppress events"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    if "TOI" in disp:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    if "GP" in disp:
        fmt["GP"] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    st.dataframe(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)

    cap = f"{len(disp)} teams · {season_label} · sorted by NFI% (CNFI+MNFI share) descending"
    if season_label != "2025-26":
        cap += (" · Attack and Suppress events are currently only computed for 2025-26. "
                "Per-season history requires a pipeline run not yet performed.")
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


def render_goalies(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Goalies</h2>",
        unsafe_allow_html=True,
    )
    if game_type == "Playoffs":
        st.info("Goalie playoff metrics aren't available yet — these are "
                "regular-season GSAx-based metrics.")
        return

    # FOLLOW-UP: no 2-year pooled goalie build exists. Goalie GSAx is published
    # as full-pooled (2022–2026) or per single season; a faithful 2yr pool needs
    # re-derived denominators (not a season average), so fall back gracefully.
    if SEASON_KEY.get(season_label) == "pooled_2yr":
        st.info("2-season (2024–2026) goalie view isn't available yet — pick a "
                "single season or the full Pooled (2022–2026) view.")
        return

    is_pooled = SEASON_KEY.get(season_label, "pooled") == "pooled"
    if is_pooled:
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
        default_shots = 500 if is_pooled else 150
        shots_key = "goalies_minshots_pooled" if is_pooled else "goalies_minshots_season"
        min_shots = st.slider("Min Shots Faced", 0, 3000, default_shots, 50, key=shots_key)
        st.caption("Min Shots Faced filter suppresses small-sample noise in per-60 "
                   "rates. Defaults match the methodology's qualifying floors and the "
                   "previous app's discipline.")
    with c2:
        name_q = st.text_input("Goalie name contains", key="goalies_name").strip().lower()
    base = base[base["total_faced"].fillna(0) >= min_shots]
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
    base = base.rename(columns={
        "NFIG60": "NFI-GSAx/60", "QNFS_pct": "QNFS%",
        "QS_GSAx_pct": "QS-GSAx%", "QS_GSAx_lo": "QS-GSAx (95% lower)",
    })
    base = base.sort_values("NFI-GSAx/60", ascending=False, na_position="last").reset_index(drop=True)

    cols = ["Goalie", "Team", "GP", "NFI-GSAx/60", "QNFS%", "QNFS 95% CI",
            "QS-GSAx%", "QS-GSAx (95% lower)"]
    disp = base[[c for c in cols if c in base.columns]].copy()

    fmt = {}
    if "NFI-GSAx/60" in disp:
        fmt["NFI-GSAx/60"] = lambda x: "—" if pd.isna(x) else f"{x:+.3f}"
    for c in ("QNFS%", "QS-GSAx%", "QS-GSAx (95% lower)"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}%"
    if "GP" in disp:
        fmt["GP"] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    st.dataframe(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
    st.caption(
        f"{len(disp)} goalies · {season_label} · sorted by NFI-GSAx/60 descending · "
        "blanks = below that metric's qualifying floor (not zero)"
    )
    st.markdown(
        f"<p style='color:{PALETTE['text_secondary']}; font-size:0.82rem; max-width:62rem;'>"
        "Goalies shown are the union of qualified cohorts across the three metrics. "
        "Backup goalies appearing only in unqualified QNFS rows are excluded — see "
        "Methodology for full qualifying floors. Qualifying floors differ by metric "
        "(NFI-GSAx ≥300 net-front shots pooled / ≥100 per season; QNFS% ≥25 GP/season "
        "with ≥3 net-front shots/game; QS-GSAx ≥10 shots/game, ≥25 GP/season).</p>",
        unsafe_allow_html=True,
    )


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


def _ref_league_avg(tbl: pd.DataFrame) -> pd.DataFrame:
    avg = {"Referee": "LEAGUE AVERAGE", "Games": tbl["Games"].sum()}
    for c in tbl.columns:
        if c not in ("Referee", "Games"):
            avg[c] = tbl[c].mean()
    return pd.DataFrame([avg])


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
    key = SEASON_KEY.get(season_label, "pooled")
    if key == "pooled_2yr":
        df = df[df["season"].isin([20242025, 20252026])]
    elif key != "pooled":
        sk = REF_SEASON_INT.get(season_label)
        df = df[df["season"] == sk]

    st.markdown(
        f"<p style='color:{PALETTE['text_secondary']}; font-size:0.85rem; font-style:italic; "
        f"max-width:62rem;'>Each game has two referees and the NHL doesn't publish which "
        "official called a given penalty — so these are the penalty environment in games each "
        "referee worked (with a partner), not penalties personally assigned.</p>",
        unsafe_allow_html=True,
    )

    name_q = st.text_input("Referee name contains", key="refs_name").strip().lower()

    tbl = _ref_table(df)
    tbl = tbl[tbl["Games"] >= REF_MIN_GAMES]
    if tbl.empty:
        st.info("No referees meet the 40-game floor for this view.")
        return
    avg = _ref_league_avg(tbl)
    tbl = tbl.sort_values("Pen/Game", ascending=False).reset_index(drop=True)
    if name_q:
        tbl = tbl[tbl["Referee"].str.lower().str.contains(name_q, na=False)]
    out = pd.concat([avg, tbl], ignore_index=True)

    cols = (["Referee", "Games", "Pen/Game", "Home Pen%", "Away Pen%"]
            + [f"{t}/G" for t in REF_TYPES])
    disp = out[[c for c in cols if c in out.columns]].copy()

    fmt = {"Games": lambda x: "—" if pd.isna(x) else f"{int(x):,}",
           "Pen/Game": lambda x: "—" if pd.isna(x) else f"{x:.2f}",
           "Home Pen%": lambda x: "—" if pd.isna(x) else f"{x:.1f}%",
           "Away Pen%": lambda x: "—" if pd.isna(x) else f"{x:.1f}%"}
    for t in REF_TYPES:
        c = f"{t}/G"
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"

    def _bold_avg(row):
        is_avg = row["Referee"] == "LEAGUE AVERAGE"
        return [f"font-weight:700; color:{PALETTE['blue']};" if is_avg else "" for _ in row]

    styler = disp.style.format(fmt, na_rep="—").apply(_bold_avg, axis=1)
    st.dataframe(styler, width="stretch", hide_index=True)
    n_refs = len(disp) - 1
    st.caption(
        f"{n_refs} referees · {season_label} · sorted by Pen/Game descending · "
        "minimum 40 games · top row = league average. Home Pen% = share of a "
        "referee's penalties assessed to the home team (league ≈ 47%)."
    )


# ---------------------------------------------------------------------------
# Global sidebar (Season + Game type — apply to every tab)
# ---------------------------------------------------------------------------
def render_global_sidebar() -> tuple[str, str]:
    st.sidebar.markdown(
        f"<div style='font-family:\"Bebas Neue\",Impact,sans-serif; font-size:1.4rem; "
        f"color:{PALETTE['text']}; letter-spacing:1px; margin-bottom:0.3rem;'>Filters</div>",
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("g_season", "2025-26")
    st.session_state.setdefault("g_game_type", "Regular Season")
    season = st.sidebar.selectbox("Season", list(SEASON_KEY.keys()), key="g_season")
    game_type = st.sidebar.radio("Game type", ["Regular Season", "Playoffs"],
                                 key="g_game_type")
    st.sidebar.caption(
        "Season and game type apply across all tabs."
    )
    return season, game_type


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="HockeyROI — NHL Net-Front Impact & Zone Analytics",
        page_icon="🏒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_header()
    season_label, game_type = render_global_sidebar()
    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    players_tab, teams_tab, goalies_tab, refs_tab, meth_tab = st.tabs(TAB_LABELS)
    with players_tab:
        render_players(season_label, game_type)
    with teams_tab:
        render_teams(season_label, game_type)
    with goalies_tab:
        render_goalies(season_label, game_type)
    with refs_tab:
        render_referees(season_label, game_type)
    with meth_tab:
        render_methodology()

    render_footer()


if __name__ == "__main__":
    main()
