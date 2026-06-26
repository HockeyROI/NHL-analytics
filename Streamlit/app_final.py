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
    "NFI-QG%": _CHART_PRIMARY, "xG-QG%": _CHART_SECOND,
    "RelNFI-QG%": _CHART_PRIMARY, "RelxG-QG%": _CHART_SECOND, "RelxG%": _CHART_PRIMARY,
    "NFI-GSAx/60": _CHART_PRIMARY, "QNFS%": _CHART_PRIMARY, "GQG%": _CHART_SECOND,
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


def _show_df(obj, **kwargs) -> None:
    """st.dataframe with the leading identity column pinned (frozen on the left)
    and columns sized to their content so numbers aren't clipped — the table
    scrolls horizontally instead of squeezing every column. Works for plain
    DataFrames and Stylers (Styler.data holds the underlying frame)."""
    cols = obj.data.columns if hasattr(obj, "data") else obj.columns
    if len(cols):
        cc = dict(kwargs.pop("column_config", {}) or {})
        # The detail/trade trend's leading "Season" column holds the long
        # "2yr avg (24-26)" label — give it a wider fixed width so it isn't
        # clipped; other leading columns stay content-sized.
        _w = "medium" if str(cols[0]) == "Season" else None
        cc.setdefault(cols[0], st.column_config.Column(pinned=True, width=_w))
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
        perturbed = real.copy()
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
            "— the QG analog of RelNFI%. <b>RelxG%</b> is the underlying season-level relative xG "
            "rate itself (relative xG per 60, on-ice − off-ice), the xG counterpart to RelNFI%. "
            "RelxG is built from MoneyPuck's raw shot data with HockeyROI's own qualifying filter, "
            "so it can differ from MoneyPuck's published relative-xG columns — different "
            "filters/aggregation, not a question of accuracy.",
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
            "<b>GQG</b> — Goalie Quality Games, the Quality-Start idea computed on GSAx "
            "(share of games with all-shot GSAx ≥ 0) rather than raw save%. Qualifying "
            "floors differ by metric, so the cohorts differ — by design.",
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
            A note on GQG vs. &ldquo;Quality Starts&rdquo;</div>
          <div style="color:{PALETTE['text']}; font-size:0.94rem; line-height:1.5;">
            <b>GQG</b> (Goalie Quality Games, formerly QS-GSAx) is <b>not</b> Robert Vollman's
            Quality Starts (~2009, defined on save% vs league average). In GQG a quality game is
            <b>per-game GSAx &ge; 0</b> — the goalie beat expected on a danger / xG-weighted basis,
            not on raw save%. <b>QNFS%</b> is the same idea on net-front shots only. Don't map
            GQG to Vollman's metric, or QNFS% and GQG to each other.
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
    first = (df.groupby(["player_id", "season", "team_abbrev"])["game_id"].min()
             .reset_index().sort_values(["player_id", "season", "game_id"]))
    out = {}
    for (pid, sn), g in first.groupby(["player_id", "season"], sort=False):
        out[(int(pid), sn)] = g["team_abbrev"].tolist()
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
    # RelxG_pct — TOI-weighted mean of per-season values (no count denominator),
    # mirroring _aggregate_nfi_pooled's pooling of RelNFI_pct.
    if {"RelxG_pct", "TOI_total_sec"}.issubset(qg.columns):
        t = qg[["player_id", "RelxG_pct", "TOI_total_sec"]].copy()
        t["RelxG_pct"] = pd.to_numeric(t["RelxG_pct"], errors="coerce")
        t["w"] = pd.to_numeric(t["TOI_total_sec"], errors="coerce")
        t = t[t["RelxG_pct"].notna() & (t["w"] > 0)]
        t["_num"] = t["RelxG_pct"] * t["w"]
        rx = t.groupby("player_id").agg(_num=("_num", "sum"),
                                        _den=("w", "sum")).reset_index()
        rx["RelxG_pct"] = np.where(rx["_den"] > 0, rx["_num"] / rx["_den"], np.nan)
        g = g.merge(rx[["player_id", "RelxG_pct"]], on="player_id", how="left")
        out_cols.append("RelxG_pct")
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
            zcols = ["season"] + [c for c in ("NZI", "DZI", "OZI") if c in zz.columns]
            trend = trend.merge(zz[zcols], on="season", how="outer")

    # Quality Games per season.
    qg = load_qg_player_season()
    if not qg.empty:
        q = qg[qg["player_id"] == pid].copy()
        q["season"] = q["season"].astype(str)
        q = q[q["season"].isin(PROFILE_SEASONS)]
        keep = ["season"] + [c for c in ("GP", "NFI_QG_pct", "xG_QG_pct",
                "RelNFI_QG_pct", "RelxG_QG_pct", "RelxG_pct") if c in q.columns]
        q = q[keep].rename(columns={"NFI_QG_pct": "NFI-QG%", "xG_QG_pct": "xG-QG%",
                "RelNFI_QG_pct": "RelNFI-QG%", "RelxG_QG_pct": "RelxG-QG%",
                "RelxG_pct": "RelxG%"})
        trend = trend.merge(q, on="season", how="outer")

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
            for m in ("NZI", "DZI", "OZI"):
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
    return out


# Storage→display column map for the appended 2yr pooled row.
_P2YR_MAP = {"NFI%": "NFI_pct", "RelNFI%": "RelNFI_pct", "RelNFI-A%": "RelNFI_F_pct",
             "RelNFI-S%": "RelNFI_A_pct", "NFI-A/60": "NFI_A_rate", "NFI-S/60": "NFI_S_rate",
             "NZI": "NZI", "DZI": "DZI", "OZI": "OZI", "NFI-QG%": "NFI_QG_pct",
             "xG-QG%": "xG_QG_pct", "RelNFI-QG%": "RelNFI_QG_pct",
             "RelxG-QG%": "RelxG_QG_pct", "RelxG%": "RelxG_pct"}


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
    zone_cols = ["NZI", "DZI", "OZI"]
    qg_cols = ["RelNFI-QG%", "NFI-QG%", "RelxG%", "RelxG-QG%", "xG-QG%"]
    metric_cols = [c for c in share_cols + rate_cols + zone_cols + qg_cols
                   if c in trend.columns]
    if families:   # narrow to the selected metric families
        _fam_of = {col: fam for fam, fcols in PLAYER_FAMILY_COLS.items() for col in fcols}
        metric_cols = [c for c in metric_cols if _fam_of.get(c) in set(families)]

    # Per-season rank (cohort per same_pos) appended to each cell. When a team is
    # given, also compute the within-team rank → cells read "(league / team)".
    ranks = _player_season_ranks(pid, same_pos=same_pos)
    team_ranks = _player_season_ranks(pid, same_pos=same_pos, team=team) if team else {}
    _b = {}
    for c in ("NFI%", "NFI-QG%", "xG-QG%", "RelNFI-QG%", "RelxG-QG%"):
        _b[c] = lambda v: f"{v * 100:.1f}%"
    for c in ("RelNFI%", "RelNFI-A%", "RelNFI-S%", "RelxG%"):
        _b[c] = lambda v: f"{v:+.2f}"
    for c in ("NFI-A/60", "NFI-S/60", "NZI", "DZI", "OZI"):
        _b[c] = lambda v: f"{v:.1f}"
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
                    row[c] = txt
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


_QG_BAR_METRICS = ["NFI%", "NFI-QG%", "xG-QG%", "RelNFI-QG%", "RelxG-QG%"]
_BRAND_DEEP = "#0A1A2F"          # "Hockey" — deeper than the chart navy
_BRAND_ROI = PALETTE["orange"]   # "ROI" — brand orange (#FF6B35)


def _chart_brand() -> None:
    """HockeyROI wordmark rendered just under a chart (Hockey deep-navy, ROI orange)."""
    st.markdown(
        "<div style='text-align:right; margin:-0.7rem 0 0.5rem 0; font-weight:800; "
        "font-size:0.95rem; letter-spacing:0.2px;'>"
        f"<span style='color:{_BRAND_DEEP};'>Hockey</span>"
        f"<span style='color:{_BRAND_ROI};'>ROI</span></div>",
        unsafe_allow_html=True)


# Diverging gradient: a light tint near the 50% midline → the FULL brand colour
# further out (blue navy above 50%, brand orange below) — same colours as the
# solid version, just softened toward the middle.
_BAR_BLUE_LIGHT, _BAR_BLUE_STRONG = "#BCD0E2", PALETTE["text"]      # → #1B3A5C navy
_BAR_ORG_LIGHT, _BAR_ORG_STRONG = "#FFCDB5", PALETTE["orange"]      # → #FF6B35 orange


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


def _default_qg_year(season_label, seasons) -> str:
    """Row label to default the bar to, from the global Season filter."""
    key = SEASON_KEY.get(season_label) if season_label else None
    if key == "pooled_2yr" and "2yr avg (24-26)" in seasons:
        return "2yr avg (24-26)"
    if season_label in seasons:
        return season_label
    _non2 = [s for s in seasons if s != "2yr avg (24-26)"]
    return _non2[-1] if _non2 else (seasons[-1] if seasons else None)


def _qg_bar_chart(vals: dict, label: str) -> None:
    """Diverging bar of NFI% + the four Quality-Games percentages vs a 50%
    baseline (50% = league-median consistency: bar up when above, down when
    below). vals maps display-metric → value on a 0-100 scale."""
    import altair as alt
    rows = [{"Metric": m, "value": float(v), "base": 50.0, "color": _bar_color(v)}
            for m, v in vals.items() if pd.notna(v)]
    if not rows:
        st.caption("No NFI% / Quality-Games values for this selection.")
        return
    d = pd.DataFrame(rows)
    _dom = _qg_axis_domain([r["value"] for r in rows])
    st.caption(f"**{label}** — NFI% + Quality-Games % vs the **50% baseline** "
               "(bar up = above 50%, down = below; darker = further from 50%).")
    bars = alt.Chart(d).mark_bar(size=40).encode(
        x=alt.X("Metric:N", sort=[r["Metric"] for r in rows],
                axis=alt.Axis(labelAngle=0, title=None, labelFontWeight="bold",
                              labelFontSize=12, labelColor=PALETTE["text"])),
        y=alt.Y("base:Q", scale=alt.Scale(domain=_dom), title="%"),
        y2="value:Q",
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("value:Q", format=".1f", title="%")])
    rule = alt.Chart(pd.DataFrame({"y": [50.0]})).mark_rule(
        strokeDash=[4, 4], color=PALETTE["text_secondary"]).encode(y="y:Q")
    st.altair_chart(bars + rule, use_container_width=True)
    _chart_brand()


def _qg_bar_chart_compare(players_vals: dict, label: str) -> None:
    """Side-by-side small-multiple bar charts (one panel per player) of the 5 QG
    metrics vs the 50% baseline. players_vals: {player_name: {metric: 0-100}}."""
    import altair as alt
    rows, allv = [], []
    for pname, vals in players_vals.items():
        for m, v in vals.items():
            if pd.notna(v):
                rows.append({"Player": pname, "Metric": m, "value": float(v),
                             "base": 50.0, "color": _bar_color(v)})
                allv.append(float(v))
    if not rows:
        st.caption("No NFI% / Quality-Games values to compare for this selection.")
        return
    d = pd.DataFrame(rows)
    _dom = _qg_axis_domain(allv)
    st.caption(f"**{label}** — NFI% + Quality-Games % vs the **50% baseline**, one "
               "panel per player (bar up = above 50%; darker = further from 50%).")
    # Width per panel so the panels together fill the container (two players
    # shouldn't be skinnier than the single-player chart).
    _n = max(1, len(players_vals))
    _w = int(max(160, 720 / _n))
    _ch = alt.Chart(d)   # shared data so a layered chart can be faceted
    bars = _ch.mark_bar(size=34).encode(
        x=alt.X("Metric:N", sort=_QG_BAR_METRICS,
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
    st.altair_chart(chart, use_container_width=True)
    _chart_brand()


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
    if same_pos:
        nfi = load_nfi_player()
        prow = nfi[nfi["player_id"] == int(pid)] if not nfi.empty else nfi
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
    _qg_bar_chart(_player_qg_vals(pid, trend, _yr), _yr)
    st.caption("↕ Click a different year (or the 2yr row) above to change the bars.")

    def _chart(title: str, cols: list[str]) -> None:
        ys = [c for c in cols if c in trend.columns and trend[c].notna().any()]
        if not ys:
            return
        st.caption(title)
        st.line_chart(trend.set_index("Season")[ys],
                      color=[_CHART_COLORS.get(c, _CHART_SECOND) for c in ys])
        _chart_brand()

    # Scales differ across families — one chart per scale so none flattens.
    # NFI% (0–1 absolute share) is split from the RelNFI family (points, ~±5).
    # Charts follow the family filter (selected families only; none = all).
    if "Net Front Impact" in _show_fams:
        _chart("NFI% (share)", ["NFI%"])
        _chart("RelNFI family (RelNFI%, RelNFI-A%, RelNFI-S%)",
               ["RelNFI%", "RelNFI-A%", "RelNFI-S%"])
        _chart("Raw net-front rate per 60 (NFI-A/60, NFI-S/60)", ["NFI-A/60", "NFI-S/60"])
    if "Zone Impact" in _show_fams:
        _chart("Zone Impact 0–10 (NZI, DZI, OZI)", ["NZI", "DZI", "OZI"])
    if "Quality Games" in _show_fams:
        _chart("Quality Games % — raw (NFI-QG%, xG-QG%)", ["NFI-QG%", "xG-QG%"])
        _chart("Quality Games % — relative (RelNFI-QG%, RelxG-QG%)",
               ["RelNFI-QG%", "RelxG-QG%"])
        _chart("Relative xG per 60 (RelxG%)", ["RelxG%"])


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
            qcols = ["player_id", "GP", "qualifying_GP", "xG_QG_pct", "NFI_QG_pct",
                     "RelNFI_QG_pct", "RelxG_QG_pct", "RelxG_pct"]
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
                     "xG_QG_pct", "NFI_QG_pct",
                     "RelNFI_QG_pct", "RelxG_QG_pct", "RelxG_pct",
                     "teams_in_season"]  # for multi-team display + filter
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
    "Quality Games": ["RelNFI-QG%", "NFI-QG%", "RelxG%", "RelxG-QG%", "xG-QG%"],
}


def render_players(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Player List</h2>",
        unsafe_allow_html=True,
    )
    if _block_ref_only(season_label):
        return
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

    # Fixed ES-TOI floor for RANKING: players below it are ranked "(UR)". The Min
    # ES TOI slider (default = this floor) FILTERS the list — by default it hides
    # the sub-floor players; slide it down to reveal them (shown as UR), up to
    # trim further. The slider never changes the ranking denominator.
    rank_floor = 300 if playoffs else (2000 if is_pooled else 500)
    c1, c2, c3 = st.columns([1.0, 1.3, 1.5])
    with c1:
        pos = st.radio("Position", ["All", "F", "D"], horizontal=True, key="players_pos")
    with c2:
        if playoffs:
            min_toi = st.slider("Min ES TOI (min)", 0, 1500, rank_floor, 25,
                                key="players_toi_playoffs")
        else:
            toi_key = "players_toi_pooled" if is_pooled else "players_toi_season"
            min_toi = st.slider("Min ES TOI (min)", 0, 7500, rank_floor, 50, key=toi_key)
    with c3:
        player_sel = st.selectbox(
            "Search a player", _pid_list, index=None,
            placeholder="",
            format_func=lambda i: _plabel.get(i, str(i)), key="players_search",
            on_change=lambda: st.session_state.update(players_team="All"),
            help="Pick a player to see their season-by-season detail on this page.")

    # Metric-family toggles first, then the Team filter. Families start with none
    # selected (only the identity columns show); click a family to display it.
    fcol, tcol = st.columns([2.8, 1.0])
    with fcol:
        display_fams = st.segmented_control(
            "**Display a Metric Family**", list(PLAYER_FAMILY_COLS),
            selection_mode="multi", key="players_display_seg",
            help="Click a metric group to show its columns (Net Front Impact, "
                 "Zone Impact, Quality Games). Click again to hide it.") or []
    with tcol:
        team_opts = ["All"] + _all_teams
        # Picking a team exits any drill-in and clears the player search (the two
        # are mutually exclusive views).
        team_sel = st.selectbox(
            "Team", team_opts, key="players_team",
            on_change=lambda: st.session_state.update(_pl_drill=None, players_search=None))

    # Drill-in (via the search box OR clicking a leaderboard row): show one
    # player's detail (trend + charts) here. Clear/deselect to return to the list.
    def _drill(pid):
        st.markdown(f"### {_plabel.get(int(pid), str(pid))}")
        if playoffs:
            _render_player_playoff_summary(frame, int(pid))
        else:
            _prow = _popts[_popts["player_id"] == int(pid)]
            _is_d = len(_prow) and str(_prow["position"].iloc[0]) == "D"
            _pos_label = "Defense only" if _is_d else "Forwards only"
            _rc = st.radio("Rank against", ["All skaters", _pos_label],
                           horizontal=True, key="players_rank_cohort")
            _render_player_profile(int(pid), same_pos=(_rc != "All skaters"),
                                   families=display_fams, team="__own__",
                                   season_label=season_label)

    # Drill via the search box OR a clicked leaderboard row — either one collapses
    # the leaderboard to just that player's detail.
    if player_sel is not None:
        st.session_state["_pl_drill"] = None     # an explicit search overrides a click
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
    # Ranking denominator is the position cohort clearing the FIXED floor (set
    # before the Min-TOI slider so the slider never changes ranks). The slider
    # then filters which rows are shown; sub-floor rows that survive it render UR.
    rank_cohort = df[df["toi_min"].fillna(0) >= rank_floor].copy()
    df = df[df["toi_min"].fillna(0) >= min_toi]
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

    # Qualified players (≥ floor) lead, sorted by RelNFI%; sub-floor (UR) players
    # follow — so low-TOI noise can't dominate the top of the leaderboard.
    df["_qual"] = df["toi_min"].fillna(0) >= rank_floor
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
        "RelxG_pct": "RelxG%",
    }
    df = df.rename(columns=_ren)
    rank_cohort = rank_cohort.rename(columns=_ren)

    # Always show the full column set (Compact view removed; Qual GP dropped).
    # NFI-A/60 / NFI-S/60 are RAW per-60 rates; RelNFI-A% / RelNFI-S% are the
    # relative (vs own-team) versions — both coexist, placed side by side.
    cols = ["Player", "Pos", "Team", "GP", "TOI", "RelNFI%", "RelNFI-A%",
            "RelNFI-S%", "NFI%", "NFI-A/60", "NFI-S/60", "NZI", "DZI", "OZI",
            "RelNFI-QG%", "NFI-QG%", "RelxG%", "RelxG-QG%", "xG-QG%"]
    # Zone now populates for single seasons too (per-season files), so it is no
    # longer stripped; the in-frame filter below drops it only if truly absent.
    cols = [c for c in cols if c in df.columns]
    # Metric-family display: identity columns (no family) always show; a family's
    # columns show only when that family is selected. Nothing selected = identity
    # columns only.
    _fam_of = {col: fam for fam, fcols in PLAYER_FAMILY_COLS.items() for col in fcols}
    _shown = set(display_fams)
    cols = [c for c in cols if _fam_of.get(c) is None or _fam_of.get(c) in _shown]
    disp = df[cols].copy()

    fmt = {}
    for c in ("NFI%", "xG-QG%", "NFI-QG%", "RelNFI-QG%", "RelxG-QG%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("RelNFI%", "RelNFI-A%", "RelNFI-S%", "RelxG%"):
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
                    "NFI-S/60", "NZI", "DZI", "OZI",
                    "RelNFI-QG%", "NFI-QG%", "RelxG%", "RelxG-QG%", "xG-QG%"]
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
                ascending=(col == "NFI-S/60"), method="min")
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
                    _coh_team).rank(ascending=(col == "NFI-S/60"), method="min")
                p2t = dict(zip(rank_cohort["player_id"], tr))
                _team_rank_idx[col] = {i: int(p2t[p]) for i, p in _df_pid.items()
                                       if pd.notna(p2t.get(p))}
    _pl_qual = pd.to_numeric(disp["TOI"], errors="coerce").fillna(0) >= rank_floor
    _apply_ranks(disp, fmt, rank_cohort, _player_rank, lower_better={"NFI-S/60"},
                 mark_unranked=True, qualified=_pl_qual, team_rank_idx=_team_rank_idx)
    _cohort_label = {"All": "all skaters (F + D)", "F": "forwards",
                     "D": "defense"}[pos]
    _team_txt = team_sel if team_sel != "All" else "their own team"
    st.caption(f"Each metric shows **(league rank / team rank)** — rank within "
               f"**{_cohort_label}** league-wide, then within **{_team_txt}**. Only "
               f"players with **≥ {rank_floor:,} ES minutes** are ranked; lower the "
               f"Min ES TOI slider to reveal the rest as **(UR)** = unranked. "
               f"NFI-S/60 (shots against): lowest = #1.")
    _sort_hint()
    st.caption("Click a row to open that player's detail (collapses the list).")
    _gen = st.session_state.get("_pl_tbl_gen", 0)
    _event = _show_df(disp.style.format(fmt, na_rep="—"), hide_index=True,
                      on_select="rerun", selection_mode="single-row",
                      key=f"players_tbl_{_gen}")

    if playoffs:
        zone_note = " · NZI/DZI/OZI pooled across playoffs"
    elif SEASON_KEY.get(season_label) == "pooled_2yr":
        zone_note = " · NZI/DZI/OZI pooled 2024-25 + 2025-26"
    elif is_pooled:
        zone_note = " · NZI/DZI/OZI pooled across all seasons"
    else:
        zone_note = " · NZI/DZI/OZI for this season"
    st.caption(
        f"{len(disp):,} players (≥ {min_toi:,} ES min) · {scope_label} · sorted by "
        f"RelNFI% descending · ranked at ≥ {rank_floor:,} ES min (else UR){zone_note}"
    )

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
        ("RelNFI-QG%", _f("RelNFI_QG_pct", "pct")),
        ("NFI-QG%", _f("NFI_QG_pct", "pct")),
        ("RelxG%", _rel("RelxG_pct")),
        ("RelxG-QG%", _f("RelxG_QG_pct", "pct")),
        ("xG-QG%", _f("xG_QG_pct", "pct")),
        ("ES TOI (min)", _f("toi_min", "toi")),
    ]
    st.caption(f"**{r['player_name']} ({r['position']})** · pooled playoffs "
               "(2022-23 → 2024-25).")
    _show_df(pd.DataFrame(items, columns=["Metric", "Value"]),
                 width="stretch", hide_index=True)


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
                     "team_xG_QG_pct": "xG-QG%", "team_NFI_QG_pct": "NFI-QG%"})
        team = team.merge(q, on="team", how="left")

    zcols = ["NZI", "DZI", "OZI"]
    tz = load_team_zone_playoffs()
    if not tz.empty:
        team = team.merge(tz[["team", "NZI", "DZI", "OZI"]], on="team", how="left")

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
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    if "TOI" in disp:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    if "GP" in disp:
        fmt["GP"] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    _team_rank = ["NFI%", "Attack events", "Suppress events"] + zcols + ["xG-QG%", "NFI-QG%"]
    _apply_ranks(disp, fmt, disp, _team_rank, lower_better={"Suppress events"})
    st.caption("Each metric shows its **(rank)** across playoff teams. "
               "Suppress events (shots against): lowest = #1.")
    _sort_hint()
    _show_df(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
    st.caption(
        f"{len(disp)} teams · all playoffs (2022-2025 pooled) · sorted by NFI% "
        "(CNFI+MNFI share) descending · Zone Impact (NZI/DZI/OZI) is TOI-weighted."
    )


def render_teams(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Teams</h2>",
        unsafe_allow_html=True,
    )
    if _block_ref_only(season_label):
        return
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

    for c in ["TOI", "xG-QG%", "NFI-QG%", "Attack events", "Suppress events"] + zcols:
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
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    if "TOI" in disp:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    if "GP" in disp:
        fmt["GP"] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    _team_rank = (["NFI%", "Attack events", "Suppress events"] + zcols
                  + ["xG-QG%", "NFI-QG%"])
    _apply_ranks(disp, fmt, disp, _team_rank, lower_better={"Suppress events"})
    st.caption("Each metric shows its **(rank)** across all 32 teams. "
               "Suppress events (shots against): lowest = #1.")
    _sort_hint()
    _show_df(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)

    zwin_label = "4-year pool (2022-26)" if zwin == "4y_pool" else "2-year pool (2024-26)"
    cap = (f"{len(disp)} teams · {season_label} · sorted by NFI% (CNFI+MNFI share) "
           f"descending · Zone Impact (NZI/DZI/OZI) is TOI-weighted, shown as the "
           f"{zwin_label}; single seasons display their pooled window "
           f"(2022-24 → 4yr, 2024-26 → 2yr) since single-season team zone isn't published.")
    st.caption(cap)


# ---------------------------------------------------------------------------
# Goalies tab — NFI-GSAx + QNFS% + GQG (union of qualified cohorts)
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
    """Per-season (2022-23..2025-26) NFI-GSAx/60, QNFS%, GQG% for one
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
                     .rename(columns={"QS_GSAx_pct": "GQG%"}))
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
        (load_qs_by_season, "QS_GSAx_pct", "GQG%"),
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


def _goalie_profile_table(gid: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Rank-annotated per-season goalie trend + a '2yr avg (24-26)' pooled row.
    Returns (display_df, trend, metric_cols). Shared by the goalie detail and the
    Trade Analyzer's goalie mode."""
    trend = _goalie_trend(gid)
    if trend.empty:
        return pd.DataFrame(), trend, []
    metric_cols = [c for c in ("NFI-GSAx/60", "QNFS%", "GQG%")
                   if c in trend.columns]
    ranks = _goalie_season_ranks(gid)
    _b = {"NFI-GSAx/60": lambda v: f"{v:+.3f}",
          "QNFS%": lambda v: f"{v:.1f}%", "GQG%": lambda v: f"{v:.1f}%"}
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

    # Append a "2yr avg (24-26)" row — denominator-based pool of the last two
    # seasons (ranked within the 2yr pool).
    _n2, _q2, _s2 = _pool_goalie_seasons((20242025, 20252026))
    _src2 = {"NFI-GSAx/60": (_n2, "NFIG60"), "QNFS%": (_q2, "QNFS_pct"),
             "GQG%": (_s2, "QS_GSAx_pct")}
    _v2 = {}
    for c, (fr, col) in _src2.items():
        rr = fr[fr["goalie_id"] == gid] if not fr.empty else fr
        _v2[c] = rr[col].iloc[0] if len(rr) else np.nan
    if any(pd.notna(v) for v in _v2.values()):
        row = {"Season": "2yr avg (24-26)"}
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
    return pd.DataFrame(rows, columns=["Season"] + metric_cols), trend, metric_cols


def _render_goalie_profile(gid: int) -> None:
    """Per-season trend table + line charts for one goalie."""
    disp, trend, metric_cols = _goalie_profile_table(gid)
    if disp.empty:
        st.info("No per-season data available for this goalie.")
        return
    st.caption("Each value shows its **(rank)** — league rank among all goalies "
               "that season (2yr row ranks within the 2-season pool).")
    _show_df(disp, width="stretch", hide_index=True)

    # CHOICE: split into 2 small multiples. NFI-GSAx/60 is a per-60 rate (~±0.3);
    # QNFS%/GQG% are percentages (~0–100). On a single shared axis the rate
    # collapses to a flat line near zero, so the rate gets its own chart and the
    # two percentages share one.
    if "NFI-GSAx/60" in trend.columns and trend["NFI-GSAx/60"].notna().any():
        st.caption("NFI-GSAx per 60")
        st.line_chart(trend.set_index("Season")[["NFI-GSAx/60"]],
                      color=[_CHART_COLORS.get("NFI-GSAx/60", _CHART_SECOND)])
        _chart_brand()
    pct = [c for c in ("QNFS%", "GQG%")
           if c in trend.columns and trend[c].notna().any()]
    if pct:
        st.caption("Consistency % (QNFS%, GQG%)")
        st.line_chart(trend.set_index("Season")[pct],
                      color=[_CHART_COLORS.get(c, _CHART_SECOND) for c in pct])
        _chart_brand()


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
    QNFS/GQG: any single season with ≥25 GP)."""
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
                    _gsax=("GSAx", "sum"), _toi=("_toi", "sum")).reset_index())
        g["NFIG60"] = np.where(g["_toi"] > 0, g["_gsax"] / g["_toi"] * 60.0, np.nan).round(3)
        g["team"] = g["goalie_id"].map(last_team)
        g["qual_gsax"] = g["total_faced"] >= 100 * n_seasons
        nfi = g[["goalie_id", "goalie_name", "team", "GP_nfi", "total_faced",
                 "NFIG60", "qual_gsax"]]
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
        g = (b.groupby(["goalie_id", "goalie_name"])
               .agg(GP_qs=("GP", "sum"), _q=("quality_games", "sum"),
                    _maxgp=("GP", "max")).reset_index())
        g["QS_GSAx_pct"] = g["_q"] / g["GP_qs"] * 100
        g["QS_GSAx_lo"] = g.apply(lambda r: _wilson(r["_q"], r["GP_qs"], True) * 100, axis=1)
        g["qual_qs"] = g["_maxgp"] >= 25
        qs = g[["goalie_id", "goalie_name", "GP_qs", "QS_GSAx_pct", "QS_GSAx_lo", "qual_qs"]]
    return nfi, qn, qs


def render_goalies(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Goalie List</h2>",
        unsafe_allow_html=True,
    )
    if _block_ref_only(season_label):
        return
    playoffs = game_type == "Playoffs"
    if playoffs:
        st.caption("Playoff view — all playoff games (2022-23 → 2024-25) pooled. "
                   "Small playoff samples: all goalies are ranked (no qualifying floor).")

    is_2yr = (not playoffs) and SEASON_KEY.get(season_label) == "pooled_2yr"
    is_pooled = (not playoffs) and SEASON_KEY.get(season_label, "pooled") == "pooled"
    if is_2yr:
        # Faithful denominator-based pool of 2024-25 + 2025-26 (counts summed,
        # rates recomputed over the combined ~164-game sample).
        nfi, qn, qs = _pool_goalie_seasons(tuple(int(s) for s in POOLED_2YR_SEASONS))
    elif playoffs:
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
        nfi = (n[[c for c in ["goalie_id", "goalie_name", "team", "games", "total_faced", "GSAx_per60", "qualified"] if c in n.columns]]
               .rename(columns={"games": "GP_nfi", "GSAx_per60": "NFIG60", "qualified": "qual_gsax"})
               if not n.empty else pd.DataFrame())
        q = load_qnfs_pooled()  # keep EVERY goalie; qualification gates ranking only
        qn = (q[[c for c in ["goalie_id", "goalie_name", "GP", "QNFS_pct", "QNFS_lo", "QNFS_hi", "qualified"] if c in q.columns]]
              .rename(columns={"GP": "GP_qn", "qualified": "qual_qn"}) if not q.empty else pd.DataFrame())
        s = load_qs_pooled()
        qs = (s[[c for c in ["goalie_id", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo", "qualified"] if c in s.columns]]
              .rename(columns={"GP": "GP_qs", "qualified": "qual_qs"}) if not s.empty else pd.DataFrame())
    else:
        sk = GOALIE_SEASON_INT.get(season_label)
        bs = load_goalie_nfi_by_season()
        nfi = (bs[bs["season"] == sk][[c for c in ["goalie_id", "goalie_name", "team", "games", "total_faced", "GSAx_per60", "qualified"] if c in bs.columns]]
               .rename(columns={"games": "GP_nfi", "GSAx_per60": "NFIG60", "qualified": "qual_gsax"})
               if (not bs.empty and sk) else pd.DataFrame())
        q0 = load_qnfs_by_season()
        qn = (q0[q0["season"] == sk][[c for c in ["goalie_id", "goalie_name", "GP", "QNFS_pct", "QNFS_lo", "QNFS_hi", "qualified"] if c in q0.columns]]
              .rename(columns={"GP": "GP_qn", "qualified": "qual_qn"}) if (not q0.empty and sk) else pd.DataFrame())
        s0 = load_qs_by_season()
        qs = (s0[s0["season"] == sk][[c for c in ["goalie_id", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo", "qualified"] if c in s0.columns]]
              .rename(columns={"GP": "GP_qs", "qualified": "qual_qs"}) if (not s0.empty and sk) else pd.DataFrame())

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
    # Per-metric qualification. PREFER the producer's `qualified` flag when the
    # data file carries it (correct, e.g. pooled requires a season with ≥25 GP,
    # not just accumulated games). Fall back to an in-app floor only when the
    # column is absent (older data file), so the tab never crashes. Playoffs have
    # no floor → rank everyone.
    _gsax_floor = 300 if is_pooled else (200 if is_2yr else 100)
    _fallback = {"qual_gsax": ("total_faced", _gsax_floor),
                 "qual_qn": ("GP_qn", 25), "qual_qs": ("GP_qs", 25)}
    for _qc, (_col, _flr) in _fallback.items():
        if playoffs:
            base[_qc] = True
        elif _qc in base.columns:
            base[_qc] = base[_qc].fillna(False).astype(bool)
        else:
            base[_qc] = base.get(_col, pd.Series(np.nan, index=base.index)).fillna(0) >= _flr
    _gid_of = dict(zip(base["Goalie"], base["goalie_id"]))   # name → id for drill-in

    c1, c2, c3 = st.columns([1.3, 2.0, 1.0])
    with c1:
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
    with c2:
        _gnames = sorted(base["Goalie"].dropna().unique().tolist())
        goalie_pick = st.selectbox(
            "Find a goalie", ["All goalies"] + _gnames, key="goalies_name_pick",
            on_change=lambda: st.session_state.update(goalies_team="All"),
            help="Type to search by name; pick one to see their detail (trend + charts).")
    with c3:
        _gteam_opts = ["All"] + sorted(base["Team"].dropna().unique().tolist())
        # Picking a team exits any drill-in and clears the goalie search (mutually
        # exclusive views).
        goalie_team = st.selectbox(
            "Team", _gteam_opts, key="goalies_team",
            on_change=lambda: st.session_state.update(_gl_drill=None, goalies_name_pick="All goalies"))

    # Drill-in (via "Find a goalie" OR a clicked row) — either collapses the
    # leaderboard to just that goalie's detail.
    def _goalie_drill(gid, label):
        st.markdown(f"### {label}")
        if playoffs:
            _render_goalie_playoff_summary(int(gid))
        else:
            _render_goalie_profile(int(gid))

    if goalie_pick != "All goalies":
        st.session_state["_gl_drill"] = None     # an explicit search overrides a click
        gid = _gid_of.get(goalie_pick)
        if gid is not None and pd.notna(gid):
            _goalie_drill(int(gid), goalie_pick)
        return
    _gl_drill = st.session_state.get("_gl_drill")
    if _gl_drill is not None:
        if st.button("← Back to leaderboard", key="gl_back"):
            st.session_state["_gl_drill"] = None
            st.rerun()
        _goalie_drill(int(_gl_drill), name_map.get(int(_gl_drill), str(_gl_drill)))
        return

    # Ranking pool = ALL goalies (set before the Min-Shots filter so it never
    # changes ranks); each metric is ranked only over goalies that clear ITS
    # qualifying bar (per the `qual_*` flags) — others render "(UR)". The Min-Shots
    # slider then filters which rows are shown, and the Team filter narrows further.
    rank_pool = base.copy()
    base = base[base["total_faced"].fillna(0) >= min_shots]
    if goalie_team != "All":
        base = base[base["Team"] == goalie_team]
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
        "QS_GSAx_pct": "GQG%", "QS_GSAx_lo": "GQG (95% lower)",
    }
    base = base.rename(columns=_gren)
    rank_pool = rank_pool.rename(columns=_gren)
    # Goalies qualified for any metric lead (sorted by NFI-GSAx/60); pure-noise
    # small samples sink to the bottom rather than topping the leaderboard.
    base["_qual_any"] = base[["qual_gsax", "qual_qn", "qual_qs"]].any(axis=1)
    base = base.sort_values(["_qual_any", "NFI-GSAx/60"], ascending=[False, False],
                            na_position="last").reset_index(drop=True)

    cols = ["Goalie", "Team", "GP", "NFI-GSAx/60", "QNFS%", "GQG%"]
    disp = base[[c for c in cols if c in base.columns]].copy()

    fmt = {}
    if "NFI-GSAx/60" in disp:
        fmt["NFI-GSAx/60"] = lambda x: "—" if pd.isna(x) else f"{x:+.3f}"
    for c in ("QNFS%", "GQG%", "GQG (95% lower)"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}%"
    if "GP" in disp:
        fmt["GP"] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    # Each metric ranked only over goalies qualified for THAT metric (others UR).
    _metric_qual = {"NFI-GSAx/60": "qual_gsax", "QNFS%": "qual_qn", "GQG%": "qual_qs"}
    for _m, _qc in _metric_qual.items():
        if _m not in disp.columns or _qc not in base.columns:
            continue
        _coh = rank_pool[rank_pool[_qc]]
        _tc = (_coh[_coh["Team"] == goalie_team] if goalie_team != "All" else None)
        _apply_ranks(disp, fmt, _coh, [_m], second_cohort=_tc, mark_unranked=True,
                     qualified=base[_qc])
    _shot_floor = ("300 net-front shots" if is_pooled
                   else "200 net-front shots" if is_2yr else "100 net-front shots")
    if goalie_team != "All":
        st.caption(f"Each metric shows **(league rank / {goalie_team} rank)**. Every "
                   f"goalie is listed; a metric is ranked only if the goalie clears its "
                   f"bar — **NFI-GSAx** ≥ {_shot_floor}, **QNFS / GQG** ≥ 25 GP — else "
                   f"**(UR)** = unranked.")
    else:
        st.caption(f"Each metric shows its **(rank)**. Every goalie is listed; a metric "
                   f"is ranked only if the goalie clears its bar — **NFI-GSAx** "
                   f"≥ {_shot_floor}, **QNFS / GQG** ≥ 25 GP — else **(UR)** = unranked.")
    st.caption("**GQG (Goalie Quality Games)** = the Quality-Start idea computed on "
               "**GSAx**, not raw save% — the share of a goalie's games where their "
               "all-shot GSAx ≥ 0 (beat expected on a danger/xG-weighted basis).")
    _sort_hint()
    st.caption("Click a row to open that goalie's detail (collapses the list).")
    _ggen = st.session_state.get("_gl_tbl_gen", 0)
    _gevent = _show_df(disp.style.format(fmt, na_rep="—"), hide_index=True,
                       on_select="rerun", selection_mode="single-row",
                       key=f"goalies_tbl_{_ggen}")
    _goalie_scope = "all playoffs (2022-2025 pooled)" if playoffs else season_label
    st.caption(
        f"{len(disp)} goalies (≥ {min_shots:,} shots faced) · {_goalie_scope} · sorted "
        "by NFI-GSAx/60 descending · (UR) = below that metric's ranking floor"
    )
    if playoffs:
        st.markdown(
            f"<p style='color:{PALETTE['text_secondary']}; font-size:0.82rem; max-width:62rem;'>"
            "Pooled across all playoff games (2022-23 → 2024-25). Every goalie with "
            "playoff data is shown and ranked — no qualifying floor is applied to the "
            "small playoff samples. Per-game metric definitions (QNFS ≥3 net-front "
            "shots/game; GQG ≥10 shots/game) are retained.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<p style='color:{PALETTE['text_secondary']}; font-size:0.82rem; max-width:62rem;'>"
            "Goalies above the Min-Shots filter are shown with a value for each "
            "metric (lower it toward 0 to see everyone). A metric is RANKED only "
            "when the goalie clears its qualifying minimum — otherwise the cell "
            "reads (UR), unranked. Floors differ by metric: NFI-GSAx ≥300 net-front "
            "shots pooled / ≥100 per season; QNFS% ≥25 GP/season (≥3 net-front "
            "shots/game); GQG ≥25 GP/season (≥10 shots/game). The regenerated data "
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
        ("GQG%", fmt(pick(s, "QS_GSAx_pct"), "pct")),
        ("Games (GSAx)", fmt(pick(n, "games"), "int")),
        ("Shots faced", fmt(pick(n, "total_faced"), "int")),
    ]
    st.caption(f"**{name}** · pooled playoffs (2022-23 → 2024-25).")
    _show_df(pd.DataFrame(items, columns=["Metric", "Value"]),
                 width="stretch", hide_index=True)


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
    if playoffs:
        for gid in sel:
            _render_goalie_playoff_summary(int(gid))
        return
    st.caption("Each value shows its **(rank)** — league rank among all goalies "
               "that season (2yr row ranks within the 2-season pool).")
    for gid in sel:
        st.markdown(f"**{names.get(int(gid), str(gid))}**")
        disp, _, _ = _goalie_profile_table(int(gid))
        if disp.empty:
            st.info("No per-season data available for this goalie.")
        else:
            _show_df(disp, width="stretch", hide_index=True)


def render_trade_analyzer(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Trade Analyzer</h2>",
        unsafe_allow_html=True,
    )
    if _block_ref_only(season_label):
        return
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

    # Side-by-side QG bar comparison — one panel per player, for the filter's year.
    _seasons_all = [SEASON_DISPLAY.get(s, s) for s in PROFILE_SEASONS] + ["2yr avg (24-26)"]
    _cmp_yr = _default_qg_year(season_label, _seasons_all)
    _pv = {}
    for pid in sel:
        _tr = _player_trend(int(pid))
        if not _tr.empty:
            _pv[plabel.get(int(pid), str(pid))] = _player_qg_vals(int(pid), _tr, _cmp_yr)
    if _pv:
        _qg_bar_chart_compare(_pv, _cmp_yr)

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
    _show_df(disp.style.apply(_bold_avg, axis=1), width="stretch", hide_index=True)


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
            _show_df(ta[cols].reset_index(drop=True), width="stretch", hide_index=True)

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


def render_referees(season_label: str, game_type: str) -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Referees</h2>",
        unsafe_allow_html=True,
    )
    # The Referees tab reads the SHARED global Season filter (no second picker).
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
    st.caption("ℹ️ A **blank cell** anywhere on this page means that player or goalie "
               "fell below the metric's qualifying **sample-size** minimum for that "
               "scope — it's “not enough data”, not zero. **(UR)** beside a value means "
               "the same: shown but unranked.")
    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    (player_list_tab, goalie_list_tab,
     trade_tab, teams_tab, refs_tab, meth_tab) = st.tabs(TAB_LABELS)
    with player_list_tab:
        render_players(season_label, game_type)
    with goalie_list_tab:
        render_goalies(season_label, game_type)
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
