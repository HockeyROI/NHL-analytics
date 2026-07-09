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
            "DZI / NZI / OZI — three position-normalized 0–10 lenses for offensive-zone time after "
            "defensive / neutral / offensive faceoffs. Independent lenses, not a hierarchy; a complete "
            "player rates well across all three. <b>D/N/O Start%</b> sits alongside them — the plain "
            "share of a player's faceoff-started shifts that began in each zone (my own play-by-play "
            "data, pooled across seasons). It's a presentation layer showing deployment context, not "
            "a new metric — it doesn't feed into or alter DZI/NZI/OZI.",
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
            "NHL EDGE",
            "NHL's own player-tracking data (by player position, not puck position) — offensive / "
            "neutral / defensive-zone time share, top skating speed, 20+ mph speed-burst count, and "
            "distance skated. A <b>different measurement basis</b> than the zone metrics above: EDGE "
            "tracks continuously across all-situations or even-strength TOI (toggle-able for OZ%); "
            "NZI/DZI/OZI track puck position after strict-5v5 faceoffs only. Each EDGE value shows a "
            "computed (league / team) rank rather than NHL's own percentile, matching every other "
            "ranked column in the app.",
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
    agg = sub.groupby("player_id").agg(
        sog_for=("sog_for", "sum"), goals_for=("goals_for", "sum"),
        sog_against=("sog_against", "sum"), goals_against=("goals_against", "sum"),
    ).reset_index()
    ok = (agg["sog_for"] > 0) & (agg["sog_against"] > 0)
    sh = np.where(ok, agg["goals_for"] / agg["sog_for"], np.nan)
    sv = np.where(ok, 1 - agg["goals_against"] / agg["sog_against"], np.nan)
    agg["PDO"] = np.where(ok, (sh + sv) * 100, np.nan)
    return agg[["player_id", "PDO"]]


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
    "edge_distance_per_min": "EDGE Distance/min",
}
# Value-only columns (excludes the raw NHL percentile columns) — displayed with
# a computed (league / team) rank bracket instead, same convention as every
# other ranked column in this table.
_EDGE_VALUE_RAW = [c for c in _EDGE_COLS if "percentile" not in c]
_EDGE_VALUE_DISP = [_EDGE_REN[c] for c in _EDGE_VALUE_RAW] + ["EDGE Distance/min"]

# EDGE OZ%-scope toggle — the ONLY EDGE stat with an even-strength split from
# NHL is offensive-zone time; NZ%/DZ% have just the one (all-situations)
# number no matter what, since that's all the API publishes for those two.
_EDGE_OZ_SCOPE_COL = {"Even Strength": ("oz_time_pct_ev", "oz_time_pct_ev_percentile"),
                      "All Situations": ("oz_time_pct", "oz_time_pct_percentile")}


def _edge_toggle_state() -> str:
    """Shared EDGE OZ% scope toggle state (set by the widget in
    render_players, read here so it applies wherever EDGE is computed this
    run — same shared-session-state pattern as _pdo_toggle_state)."""
    return st.session_state.get("players_edge_scope", "All Situations")


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


def _edge_distance_rate(scope_key: str) -> pd.DataFrame:
    """EDGE distance skated, normalized to a per-minute rate. Computed as a
    ratio-of-sums (sum distance, sum toi_min across the scope's seasons, then
    divide once) rather than games-weighted-averaging the way _edge_rate
    pools its other columns — distance_skated_miles is a season TOTAL, and
    toi_min elsewhere in this app is a season-summed cumulative figure for
    pooled scopes, so averaging the numerator while the denominator stays
    summed would silently understate the pooled rate by roughly 1/N seasons.
    EDGE distance is all-situations while toi_min is ES-only, so the rate
    itself is still an approximation — just an internally-consistent one."""
    edge = load_edge_player_season()
    nfi = load_nfi_player()
    if edge.empty or nfi.empty:
        return pd.DataFrame()
    e = edge[["player_id", "season", "distance_skated_miles"]].dropna()
    n = nfi[["player_id", "season", "toi_min"]].copy()
    n["season"] = n["season"].astype(str)
    m = e.merge(n, on=["player_id", "season"], how="inner")
    if scope_key == "pooled":
        sub = m[m["season"].isin(POOLED_SEASONS)]
    elif scope_key == "pooled_2yr":
        sub = m[m["season"].isin(POOLED_2YR_SEASONS)]
    else:
        sub = m[m["season"] == scope_key]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("player_id").agg(_dist=("distance_skated_miles", "sum"),
                                     _toi=("toi_min", "sum")).reset_index()
    ok = g["_toi"] > 0
    g["edge_distance_per_min"] = np.where(ok, g["_dist"] / g["_toi"], np.nan)
    return g[["player_id", "edge_distance_per_min"]]


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
    return g[["player_id", "xGF/60", "xGA/60"]]


# Quality-Games For/Against split (built by 03_quality_game_for_against.py): the
# share of a player's games where OFFENSE (For) or DEFENSE (Against) was quality,
# for xG and NFI. Display names → source count/qual_GP column stems.
_QG_FA = {"xG-QG-F%": "xG_QG_F", "xG-QG-A%": "xG_QG_A",
          "NFI-QG-A%": "NFI_QG_F", "NFI-QG-S%": "NFI_QG_A"}


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
            trend = trend.merge(xg[["season", "xGF/60", "xGA/60"]],
                                on="season", how="outer")

    # PDO (SOG-based; scope follows the shared 5v5/all-situations toggle) per season.
    pdo_counts = load_pdo_counts(_pdo_toggle_state())
    if not pdo_counts.empty:
        pc = pdo_counts[(pdo_counts["player_id"] == pid)
                         & (pdo_counts["season"].isin(PROFILE_SEASONS))].copy()
        if not pc.empty:
            trend = trend.merge(pc[["season", "pdo"]].rename(columns={"pdo": "PDO"}),
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
            # EDGE distance skated, normalized to a per-minute rate BEFORE the
            # rename below — same ratio (distance / toi_min, this player's own
            # season rows) that _edge_distance_rate uses league-wide, so the
            # per-season and pooled 2yr-avg rows land on an identical basis.
            _dist_toi = ea[["season", "distance_skated_miles"]].merge(
                p[["season", "toi_min"]], on="season", how="left")
            _ok_toi = _dist_toi["toi_min"] > 0
            _dist_toi["EDGE Distance/min"] = np.where(
                _ok_toi, _dist_toi["distance_skated_miles"] / _dist_toi["toi_min"], np.nan)
            ea = ea[["season"] + _EDGE_VALUE_RAW].rename(columns=_EDGE_REN)
            trend = trend.merge(ea, on="season", how="outer")
            trend = trend.merge(_dist_toi[["season", "EDGE Distance/min"]],
                                on="season", how="outer")

    # D/N/O Start% has no per-season cut (pooled-only, see load_zone_start_pooled) —
    # add the columns as all-NaN per-season placeholders so the "2yr avg" row
    # (which pulls from the separately-pooled _players_2yr_frame via _P2YR_MAP)
    # can still surface them; the per-season rows correctly show "—".
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
             "NZI": "NZI", "DZI": "DZI", "OZI": "OZI", "NFI-QG%": "NFI_QG_pct",
             "xG-QG%": "xG_QG_pct", "RelNFI-QG%": "RelNFI_QG_pct",
             "RelxG-QG%": "RelxG_QG_pct", "RelxG%": "RelxG_pct",
             # xG family + QG For/Against — the 2yr frame carries these under their
             # display names (or storage names for RelxG-F/A).
             "RelxG-F%": "RelxG_F_pct", "RelxG-A%": "RelxG_A_pct",
             "xGF/60": "xGF/60", "xGA/60": "xGA/60",
             "xG-QG-F%": "xG-QG-F%", "xG-QG-A%": "xG-QG-A%",
             "NFI-QG-A%": "NFI-QG-A%", "NFI-QG-S%": "NFI-QG-S%",
             "PDO": "PDO",
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
    zone_cols = ["DZ Start%", "NZ Start%", "OZ Start%", "NZI", "DZI", "OZI"]
    qg_cols = ["xG-QG%", "xG-QG-F%", "xG-QG-A%", "RelxG-QG%",
               "NFI-QG%", "NFI-QG-A%", "NFI-QG-S%", "RelNFI-QG%"]
    xg_cols = ["xGF/60", "xGA/60", "RelxG%", "RelxG-F%", "RelxG-A%", "PDO"]
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
              "xG-QG-F%", "xG-QG-A%", "NFI-QG-A%", "NFI-QG-S%"):
        _b[c] = lambda v: f"{v * 100:.1f}%"
    for c in ("RelNFI%", "RelNFI-A%", "RelNFI-S%", "RelxG%", "RelxG-F%", "RelxG-A%"):
        _b[c] = lambda v: f"{v:+.2f}"
    for c in ("NFI-A/60", "NFI-S/60", "NZI", "DZI", "OZI"):
        _b[c] = lambda v: f"{v:.1f}"
    for c in ("DZ Start%", "NZ Start%", "OZ Start%"):
        _b[c] = lambda v: f"{v:.1f}%"
    for c in ("xGF/60", "xGA/60"):
        _b[c] = lambda v: f"{v:.2f}"
    _b["PDO"] = lambda v: f"{v:.1f}"
    for c in ("EDGE OZ%", "EDGE OZ% (EV)", "EDGE NZ%", "EDGE DZ%"):
        _b[c] = lambda v: f"{v * 100:.1f}%"
    _b["EDGE Top Speed"] = lambda v: f"{v:.1f} mph"
    _b["EDGE Bursts 20+"] = lambda v: f"{v:.0f}"
    _b["EDGE Distance (mi)"] = lambda v: f"{v:.1f} mi"
    _b["EDGE Distance/min"] = lambda v: f"{v:.3f} mi/min"
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


_QG_BAR_METRICS = ["NFI-QG%", "NFI-QG-A%", "NFI-QG-S%", "RelNFI-QG%",
                   "xG-QG%", "xG-QG-F%", "xG-QG-A%", "RelxG-QG%"]
_BRAND_DEEP = "#0A1A2F"          # "Hockey" — deeper than the chart navy
_BRAND_ROI = PALETTE["orange"]   # "ROI" — brand orange (#FF6B35)


def _chart_brand(width_px: int = None) -> None:
    """HockeyROI footer rendered just under a (faceted) chart, stacked to match the
    embedded version: the two-colour wordmark (Hockey deep-navy, ROI orange) on top,
    the site URL beneath it, right-aligned. width_px caps the block so it sits under
    the chart's right edge (for fixed-width faceted charts) rather than the far
    container edge."""
    _w = f"max-width:{int(width_px)}px; " if width_px else ""
    st.markdown(
        f"<div style='{_w}text-align:right; margin:-0.3rem 0 0.5rem 0; "
        "line-height:1.05; letter-spacing:0.2px;'>"
        "<div style='font-weight:800; font-size:0.95rem;'>"
        f"<span style='color:{_BRAND_DEEP};'>Hockey</span>"
        f"<span style='color:{_BRAND_ROI};'>ROI</span></div>"
        f"<div style='color:#7A8694; font-size:0.78rem;'>{_BRAND_URL}</div></div>",
        unsafe_allow_html=True)


try:
    import vl_convert as _vlc            # PNG export backend (installed on deploy)
    _HAS_VLC = True
    # Register the bundled Inter TTFs so the server-side PNG renders in the SAME
    # font as the on-screen chart (Streamlit uses the Inter webfont; vl-convert
    # has no system access, so without this it falls back to its default font).
    try:
        _vlc.register_font_directory(str(APP_DIR / "fonts"))
    except Exception:
        pass
except Exception:
    _HAS_VLC = False

# The font the PNG is rendered with — must match the on-screen UI font (Inter).
_CHART_FONT = "Inter"


def _export_spec(spec_json: str) -> str:
    """Make a Vega-Lite spec render to a clean standalone PNG: padding + pad-
    autosize so nothing clips, an explicit width for single-view charts (the
    on-screen 'container' width can't resolve headless), and a light axis/view
    config so it doesn't get the default Vega axis box."""
    import json
    d = json.loads(spec_json)
    _multi = any(k in d for k in ("facet", "hconcat", "vconcat", "concat", "repeat"))
    d["padding"] = {"left": 10, "top": 10, "right": 28, "bottom": 18}
    if not _multi:
        d["autosize"] = {"type": "pad", "contains": "padding"}
        if d.get("width") in (None, "container"):
            d["width"] = 860
    cfg = d.setdefault("config", {})
    cfg.setdefault("axis", {"gridColor": "#ececec", "tickColor": "#cccccc",
                            "labelColor": PALETTE["text"], "titleColor": PALETTE["text"]})
    cfg.setdefault("axisY", {})
    cfg.setdefault("axisX", {"domainColor": "#cccccc"})
    cfg.setdefault("view", {"stroke": "transparent"})
    # Force the left y-axis line + ticks off (matches the on-screen chart). Must be
    # forced, NOT setdefault: a layered/composed spec can already carry an axisY
    # config, in which case setdefault would silently leave the domain line on.
    cfg["axisY"]["domain"] = False
    cfg["axisY"]["ticks"] = False
    # Force Inter everywhere text is drawn so the PNG matches the on-screen font.
    for _grp in ("axis", "axisX", "axisY", "legend", "header"):
        _g = cfg.setdefault(_grp, {})
        _g["labelFont"] = _CHART_FONT
        _g["titleFont"] = _CHART_FONT
    cfg.setdefault("title", {})["font"] = _CHART_FONT
    cfg.setdefault("title", {})["subtitleFont"] = _CHART_FONT
    cfg.setdefault("text", {})["font"] = _CHART_FONT   # mark_text (incl. wordmark/URL)
    cfg["font"] = _CHART_FONT
    return json.dumps(d)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=300)
def _alt_png(spec_json: str):
    """Vega-Lite spec → PNG bytes (None if export unavailable)."""
    if not _HAS_VLC:
        return None
    try:
        return _vlc.vegalite_to_png(_export_spec(spec_json), scale=2)
    except Exception:
        return None


_BRAND_URL = "hockeyroi.streamlit.app"


def _png_add_brand(png):
    """Composite the stacked HockeyROI wordmark + site URL into a white footer
    strip at the bottom of an already-rendered PNG (scale=2). Used for multi-panel
    (faceted) charts, whose brand can't be embedded inside the Vega plot — value-
    positioned marks don't resolve to a panel's coordinates — so the download still
    carries the brand, matching the on-screen HTML footer."""
    if not png:
        return png
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        im = Image.open(io.BytesIO(png)).convert("RGB")
        W, H = im.size
        _fd = str(APP_DIR / "fonts")
        f_mark = ImageFont.truetype(f"{_fd}/Inter-Bold.ttf", 24)
        f_url = ImageFont.truetype(f"{_fd}/Inter-Regular.ttf", 20)
        strip = 60
        out = Image.new("RGB", (W, H + strip), (255, 255, 255))
        out.paste(im, (0, 0))
        dr = ImageDraw.Draw(out)
        right = W - 24
        y_url = H + strip - 12               # bottom line: URL
        y_mark = y_url - 26                   # line above: wordmark
        _w_roi = dr.textlength("ROI", font=f_mark)
        dr.text((right, y_mark), "ROI", font=f_mark, fill=(255, 107, 53), anchor="rs")
        dr.text((right - _w_roi, y_mark), "Hockey", font=f_mark, fill=(10, 26, 47), anchor="rs")
        dr.text((right, y_url), _BRAND_URL, font=f_url, fill=(122, 134, 148), anchor="rs")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return png


# Set by drill-in views so a downloaded chart carries whose data it is (composited
# into a title strip at the top of the PNG). Reset to None on leaderboard views.
_CHART_TITLE = None


def _set_dl_title(name) -> None:
    global _CHART_TITLE
    _CHART_TITLE = str(name) if name else None


def _png_add_title(png, title):
    """Composite a bold title (the player/goalie/comparison name) into a white strip
    at the TOP of a rendered PNG, so a saved chart identifies whose data it is."""
    if not png or not title:
        return png
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        im = Image.open(io.BytesIO(png)).convert("RGB")
        W, H = im.size
        f = ImageFont.truetype(str(APP_DIR / "fonts" / "Inter-Bold.ttf"), 30)
        strip = 48
        out = Image.new("RGB", (W, H + strip), (255, 255, 255))
        out.paste(im, (0, strip))
        dr = ImageDraw.Draw(out)
        dr.text((16, strip // 2), title, font=f, fill=(27, 58, 92), anchor="lm")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return png


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
    panel's coordinates — so they use the stacked HTML footer in _chart_brand.)"""
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


def _show_chart(chart, dl_name: str, brand_width: int = None, brand_lift: int = 6) -> None:
    """Render an Altair chart + a 'Save PNG' download button. Non-faceted charts
    carry the two-colour HockeyROI wordmark + site URL embedded inside the plot
    (bottom-right, just above the x-axis) so it shows on-screen AND in the PNG
    without distorting the axes. Faceted/multi-panel charts can't embed it (value-
    positioned marks don't resolve to a panel's coordinates), so they show the same
    stacked wordmark + URL as an HTML footer on-screen and have it composited into
    the PNG. brand_lift raises the embedded footer when data crowds the bottom."""
    import altair as alt
    import hashlib
    _cd = chart.to_dict()
    _multi = any(k in _cd for k in ("facet", "hconcat", "vconcat", "concat", "repeat"))
    disp = chart if _multi else alt.layer(chart, _brand_layer(brand_lift))
    st.altair_chart(disp, use_container_width=True)
    if _multi:
        _chart_brand(brand_width)
    png = _alt_png(disp.to_json())
    if _multi:
        png = _png_add_brand(png)        # composite the brand into the download
    if _CHART_TITLE:
        png = _png_add_title(png, _CHART_TITLE)   # name whose data it is
    if png:
        _key = "dl_" + hashlib.md5(dl_name.encode()).hexdigest()[:12]
        _sp, _btn = st.columns([20, 1])
        with _btn:
            st.download_button("⬇", data=png, file_name=f"{dl_name}.png",
                               mime="image/png", key=_key)


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


def _default_qg_year(season_label, seasons) -> str:
    """Row label to default the bar to, from the global Season filter."""
    key = SEASON_KEY.get(season_label) if season_label else None
    if key == "pooled_2yr" and "2yr avg (24-26)" in seasons:
        return "2yr avg (24-26)"
    if season_label in seasons:
        return season_label
    _non2 = [s for s in seasons if s != "2yr avg (24-26)"]
    return _non2[-1] if _non2 else (seasons[-1] if seasons else None)


def _qg_bar_chart(vals: dict, label: str, caption: str = None,
                  dl_prefix: str = "QG-bars") -> None:
    """Diverging bar of metric %s vs a 50% baseline (50% = league-median: bar up
    when above, down when below). vals maps display-metric → value on a 0-100
    scale. caption overrides the default (player NFI%+QG) caption; dl_prefix names
    the download file."""
    import altair as alt
    rows = [{"Metric": m, "value": float(v), "base": 50.0, "color": _bar_color(v)}
            for m, v in vals.items() if pd.notna(v)]
    if not rows:
        st.caption("No values for this selection.")
        return
    d = pd.DataFrame(rows)
    _dom = _qg_axis_domain([r["value"] for r in rows])
    st.caption(caption or (f"**{label}** — Quality-Games % vs the **50% "
               "baseline** (bar up = above 50%, down = below; darker = further from 50%). "
               "**NFI** family in blue, **xG** family in orange."))
    # Colour labels by metric family (xG-* orange, NFI-* blue) rather than fixed
    # index positions — scales to any bar count instead of assuming exactly 5.
    _orange_lbls = "[" + ",".join(f"'{r['Metric']}'" for r in rows
                                  if r["Metric"].startswith("xG")) + "]"
    _label_color = {"expr": f"indexof({_orange_lbls}, datum.value) >= 0 "
                            f"? '{PALETTE['orange']}' : '{PALETTE['text']}'"}
    bars = alt.Chart(d).mark_bar(size=40).encode(
        x=alt.X("Metric:N", sort=[r["Metric"] for r in rows],
                axis=alt.Axis(labelAngle=0, title=None, labelFontWeight="bold",
                              labelFontSize=12, labelColor=_label_color)),
        y=alt.Y("base:Q", scale=alt.Scale(domain=_dom), title="%"),
        y2="value:Q",
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("value:Q", format=".1f", title="%")])
    rule = alt.Chart(pd.DataFrame({"y": [50.0]})).mark_rule(
        strokeDash=[4, 4], color=PALETTE["text_secondary"]).encode(y="y:Q")
    _show_chart(bars + rule, dl_name=f"{dl_prefix}-{label}")


def _qg_bar_chart_compare(players_vals: dict, label: str) -> None:
    """Side-by-side small-multiple bar charts (one panel per player) of the 8 QG
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
    # Width per panel so the panels together fill a wide layout (faceted charts
    # ignore use_container_width, so size the panels up explicitly).
    _n = max(1, len(players_vals))
    _w = int(max(200, 1040 / _n))
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
    _show_chart(chart, dl_name="Trade-QG-bars",
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


def _qg_line_chart_compare(players: dict) -> None:
    """Side-by-side QG % line charts, one panel per player. players: {name: trend}."""
    import altair as alt
    parts, series = [], []
    for name, tr in players.items():
        long, ser = _qg_line_long(tr)
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
    st.caption("Quality Games % over time, per player — **NFI** (orange) vs **xG** "
               "(blue); relative (**Rel**) versions **dashed**.")
    base = alt.Chart(d).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("Season:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("value:Q", title=None,
                scale=alt.Scale(domain=_qg_line_ydomain(d["value"]))),
        tooltip=["Player:N", "Season:N", "Series:N", alt.Tooltip("value:Q", format=".3f")],
        **_qg_line_encodings(series)).properties(width=_w, height=280)
    chart = base.facet(column=alt.Column("Player:N", title=None,
                       header=alt.Header(labelFontWeight="bold", labelFontSize=13)))
    _show_chart(chart, dl_name="Trade-QG-line",
                brand_width=_w * _n + 24 * (_n - 1) + 55)


def _zone_line_chart_compare(players: dict) -> None:
    """Side-by-side Zone Impact (NZI/DZI/OZI, 0-10) line charts, one panel per
    player. players: {name: trend}. Mirrors the QG line comparison."""
    import altair as alt
    zcols = ["NZI", "DZI", "OZI"]
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
    st.caption("Zone Impact 0–10 over time, per player — **NZI** / **DZI** / **OZI**.")
    base = alt.Chart(d).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("Season:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("value:Q", title=None),
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
    _all_qg_vals = _player_qg_vals(pid, trend, _yr)
    _nfi_vals = {m: v for m, v in _all_qg_vals.items() if m in _QG_LINE_ORDER_NFI}
    _xg_vals = {m: v for m, v in _all_qg_vals.items() if m in _QG_LINE_ORDER_XG}
    _qg_bar_chart(_nfi_vals, _yr,
                 caption=f"**{_yr}** — **NFI** Quality Games % vs the **50% baseline**.",
                 dl_prefix="QG-bars-NFI")
    _qg_bar_chart(_xg_vals, _yr,
                 caption=f"**{_yr}** — **xG (MoneyPuck)** Quality Games % vs the "
                         "**50% baseline**.",
                 dl_prefix="QG-bars-xG")
    st.caption("↕ Click a different year (or the 2yr row) above to change the bars.")

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

    # Line (year-over-year) charts are opt-in via a toggle — off by default so
    # the drill-in leads with the bar + team-scatter charts, not a wall of
    # line charts. When on, each still follows the family filter (selected
    # families only; none selected = all). One chart per scale so none
    # flattens; y-axes zoom to each chart's own data range (zero=False) so
    # season-to-season movement is visible instead of flattened by a wide
    # fixed domain.
    st.checkbox("Show year-over-year graphs", key="players_show_yoy", value=False)
    if st.session_state.get("players_show_yoy"):
        if "Quality Games" in _show_fams:
            _qg_combined_line(trend)
        if "xG" in _show_fams:
            _chart("On-ice xG per 60 (xGF/60, xGA/60)", ["xGF/60", "xGA/60"])
            _chart("Relative xG % (RelxG%, RelxG-F%, RelxG-A%)",
                   ["RelxG%", "RelxG-F%", "RelxG-A%"])
            _chart("PDO (5v5)", ["PDO"])
        if "Net Front Impact" in _show_fams:
            _chart("RelNFI family (RelNFI%, RelNFI-A%, RelNFI-S%)",
                   ["RelNFI%", "RelNFI-A%", "RelNFI-S%"])
        if "Zone Impact" in _show_fams:
            _chart("Zone Impact 0–10 (NZI, DZI, OZI)", ["NZI", "DZI", "OZI"])
        if "Net Front Impact" in _show_fams:
            _chart("Raw net-front rate per 60 (NFI-A/60, NFI-S/60)", ["NFI-A/60", "NFI-S/60"])
            _chart("NFI% (net-front share)", ["NFI%"])
        if "EDGE" in _show_fams:
            _chart("EDGE Zone-Time % (OZ, DZ)", ["EDGE OZ%", "EDGE DZ%"])
            _chart("EDGE Top Speed (mph)", ["EDGE Top Speed"])
            _chart("EDGE Speed Bursts (20+ mph)", ["EDGE Bursts 20+"])
            _chart("EDGE Distance Skated (mi)", ["EDGE Distance (mi)"])

    # 3 team scatters — ALWAYS shown here regardless of which family pills are
    # selected above, auto-scoped to this player's own team, so a drill-in
    # gives an immediate team-context view without needing the main
    # leaderboard's own Team filter set.
    _my_team = None
    if "Team" in trend.columns:
        for _t in trend["Team"].dropna().iloc[::-1]:   # most recent season first
            if isinstance(_t, str) and _t:
                _my_team = _t.split(" / ")[0]           # traded mid-season: first team listed
                break
    if _my_team:
        # Always pooled (4yr) here, regardless of the page's season filter —
        # Zone Start% (and therefore PDO's bubble sizing, which uses OZ Start%)
        # only exists on the pooled frame, so passing a single-season label
        # through silently dropped Zone Start% and produced a header with no
        # chart below it plus flat (unsized) PDO bubbles.
        _team_frame = _team_scatter_frame("4yr (2022-2026)", team=_my_team)
        if not _team_frame.empty:
            st.markdown(f"<h3 style='color:{PALETTE['text']}; margin-top:1.5rem;'>{_my_team} Team "
                        f"Scatters</h3>", unsafe_allow_html=True)
            st.caption(f"Auto-scoped to **{_my_team}** (this player's team) — shown regardless of "
                       "which metric families are selected above.")
            if {"PDO", "xGF/60", "xGA/60"}.issubset(_team_frame.columns):
                st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>PDO vs xG "
                            f"Differential</h4>", unsafe_allow_html=True)
                _pdo_xg_scatter(_team_frame, True, dl_suffix="-drill")
            if {"EDGE DZ%", "EDGE OZ%"}.issubset(_team_frame.columns):
                st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: D-Zone vs "
                            f"O-Zone Time%</h4>", unsafe_allow_html=True)
                _edge_zone_scatter(_team_frame, True, dl_suffix="-drill")
            if {"DZ Start%", "OZ Start%"}.issubset(_team_frame.columns):
                st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>Zone Starts: D-Zone "
                            f"vs O-Zone</h4>", unsafe_allow_html=True)
                _zone_start_scatter(_team_frame, True, dl_suffix="-drill")
            if {"NFI%", "xGF/60", "xGA/60"}.issubset(_team_frame.columns):
                st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>NFI% vs xG "
                            f"Differential</h4>", unsafe_allow_html=True)
                _nfi_xg_scatter(_team_frame, True, dl_suffix="-drill")
            if {"EDGE Distance/min", "xGF/60", "xGA/60"}.issubset(_team_frame.columns):
                st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: Distance/min "
                            f"vs xG Differential</h4>", unsafe_allow_html=True)
                _edge_distance_xg_scatter(_team_frame, True, dl_suffix="-drill")
            if {"EDGE Bursts 20+", "xGF/60", "xGA/60"}.issubset(_team_frame.columns):
                st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: Speed Bursts "
                            f"vs xG Differential</h4>", unsafe_allow_html=True)
                _edge_bursts_xg_scatter(_team_frame, True, dl_suffix="-drill")
            if {"EDGE Top Speed", "xGF/60", "xGA/60"}.issubset(_team_frame.columns):
                st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: Top Speed vs "
                            f"xG Differential</h4>", unsafe_allow_html=True)
                _edge_topspeed_xg_scatter(_team_frame, True, dl_suffix="-drill")
            if {"EDGE Distance/min", "TOI"}.issubset(_team_frame.columns):
                st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: Distance/min "
                            f"vs Minutes Played</h4>", unsafe_allow_html=True)
                _edge_distance_toi_scatter(_team_frame, True, dl_suffix="-drill")
            if {"EDGE Top Speed", "EDGE Bursts 20+"}.issubset(_team_frame.columns):
                st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: Speed Bursts "
                            f"vs Top Speed</h4>", unsafe_allow_html=True)
                _edge_speed_scatter(_team_frame, True, dl_suffix="-drill")


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
        zstart = load_zone_start_pooled()
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
    # EDGE distance skated, normalized to a per-minute rate (ratio-of-sums,
    # not games-weighted averaging — see _edge_distance_rate docstring).
    edge_dist_rate = _edge_distance_rate(key)
    if not edge_dist_rate.empty and not base.empty:
        base = base.merge(edge_dist_rate, on="player_id", how="left")
    # Quality-Games For/Against (xG-QG-F/A%, NFI-QG-F/A%), ratio-of-sums pooling.
    qgfa = _qg_fa_rates(key)
    if not qgfa.empty and not base.empty:
        base = base.merge(qgfa, on="player_id", how="left")
    return base, is_pooled


# Player List metric families — the collapse filter toggles each group's columns
# (display names, post-rename). Identity columns (Player/Pos/Team/GP/TOI) always
# show.
PLAYER_FAMILY_COLS = {
    # Quality Games = the "-QG%" metrics only (share of games that were "quality").
    "Quality Games": ["xG-QG%", "xG-QG-F%", "xG-QG-A%", "RelxG-QG%",
                      "NFI-QG%", "NFI-QG-A%", "NFI-QG-S%", "RelNFI-QG%"],
    # xG = the raw + relative expected-goals rate metrics (split out of QG).
    "xG": ["xGF/60", "xGA/60", "RelxG%", "RelxG-F%", "RelxG-A%", "PDO"],
    "Net Front Impact": ["RelNFI%", "RelNFI-A%", "RelNFI-S%", "NFI%",
                         "NFI-A/60", "NFI-S/60"],
    "Zone Impact": ["DZ Start%", "NZ Start%", "OZ Start%", "NZI", "DZI", "OZI"],
    # NHL EDGE tracking — a separate basis than NZI/DZI/OZI (player-position,
    # all-situations/EV tracking vs strict 5v5 faceoff-started PBP). See
    # edge/README.md. D/N/O Start% is NOT EDGE data (it's my own PBP faceoff
    # data) — it stays under Zone Impact only, not duplicated here, so nothing
    # under the EDGE pill implies an EDGE-API source it doesn't have.
    "EDGE": _EDGE_VALUE_DISP,
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
                         color_title: str = None) -> None:
    """Shared scatter renderer for the 5 team-scatter charts: tight (non-zero)
    axis domains so points aren't clustered in a corner, player-name labels
    shown directly ONLY when team_scoped (a small, readable point count) —
    otherwise names are hover-only (league-wide would be unreadable), and
    shown as LAST NAME only to keep the chart readable. Dots are always blue
    (uniform size), labels always orange. color_col (optional): a light
    (easy) -> dark (hard) color gradient by a 3rd metric — team-scoped views
    ONLY (league-wide always plain blue dots, since a color legend across
    hundreds of points isn't readable); silently falls back to plain dots if
    that column isn't available for the current scope."""
    import altair as alt
    d = df.dropna(subset=[x_col, y_col]).copy()
    if d.empty:
        return
    st.caption(caption)
    _xdom = _tight_domain(d[x_col], pad_frac=0.15, min_pad=1e-6)
    _ydom = _tight_domain(d[y_col], pad_frac=0.15, min_pad=1e-6)
    _use_color = (team_scoped and color_col is not None and color_col in d.columns
                  and d[color_col].notna().any())
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
                            legend=alt.Legend(orient="bottom")),
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
        labels = alt.Chart(d).mark_text(align="left", dx=6, dy=-6, fontSize=10,
                                        color=PALETTE["orange"]).encode(
            x=alt.X(f"{x_col}:Q", scale=alt.Scale(domain=_xdom, zero=False)),
            y=alt.Y(f"{y_col}:Q", scale=alt.Scale(domain=_ydom, zero=False)),
            text="_label:N",
        )
        chart = chart + labels
    _show_chart(chart, dl_name=dl_name)


def _pdo_xg_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "") -> None:
    import altair as alt
    if not {"PDO", "xGF/60", "xGA/60"}.issubset(df.columns):
        return
    d = df.copy()
    d["xG Diff/60"] = d["xGF/60"] - d["xGA/60"]
    rule100 = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(
        color=PALETTE["text_secondary"], strokeDash=[4, 4]).encode(y="y:Q")
    _scatter_with_labels(
        d, "xG Diff/60", "PDO", "xG Differential /60 (xGF − xGA)", "PDO",
        f"pdo-vs-xg-differential{dl_suffix}",
        "**Descriptive luck lens — not a ranking.** PDO (my 5v5 shot-events "
        "computation) against xG differential (MoneyPuck-derived). Above the dashed "
        "PDO=100 line = running hot; below = running cold. Color = **OZ Start%** "
        "(my PBP data), team-scoped views only — light = easier/more sheltered zone "
        "starts, dark = harder.",
        team_scoped, extra_layer=rule100, color_col="OZ Start%",
        color_title="OZ Start% (light = easier, dark = harder)")


def _zone_start_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "") -> None:
    if not {"DZ Start%", "OZ Start%"}.issubset(df.columns):
        return
    _scatter_with_labels(
        df, "DZ Start%", "OZ Start%", "DZ Start%", "OZ Start%",
        f"zone-start-scatter{dl_suffix}",
        "**Source: my PBP data** (faceoff-started 5v5 shifts, pooled) — D-zone vs "
        "O-zone faceoff-start share, one point per player.",
        team_scoped)


def _edge_zone_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "") -> None:
    if not {"EDGE DZ%", "EDGE OZ%"}.issubset(df.columns):
        return
    _scatter_with_labels(
        df, "EDGE DZ%", "EDGE OZ%", "EDGE DZ%", "EDGE OZ%",
        f"EDGE-zone-scatter{dl_suffix}",
        "**Source: NHL EDGE tracking** (not my PBP data) — D-zone vs O-zone time "
        "share, one point per player.",
        team_scoped)


def _edge_speed_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "") -> None:
    if not {"EDGE Top Speed", "EDGE Bursts 20+"}.issubset(df.columns):
        return
    _scatter_with_labels(
        df, "EDGE Top Speed", "EDGE Bursts 20+", "Top Speed (mph)",
        "Speed Bursts (20+ mph)", f"EDGE-speed-burst-vs-top-speed{dl_suffix}",
        "**Source: NHL EDGE tracking** (not my PBP data) — top skating speed vs "
        "20+ mph speed-burst count.",
        team_scoped)


def _edge_distance_xg_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "") -> None:
    if not {"EDGE Distance/min", "xGF/60", "xGA/60"}.issubset(df.columns):
        return
    d = df.copy()
    d["xG Diff/60"] = d["xGF/60"] - d["xGA/60"]
    _scatter_with_labels(
        d, "xG Diff/60", "EDGE Distance/min", "xG Differential /60 (xGF − xGA)",
        "Distance Skated (mi/min)", f"EDGE-distance-per-min-vs-xg{dl_suffix}",
        "**Source: NHL EDGE tracking** (distance/min) vs xG differential "
        "(MoneyPuck-derived), one point per player. EDGE distance is all-"
        "situations while minutes played is ES-only, so this rate is an "
        "approximation.",
        team_scoped)


def _edge_bursts_xg_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "") -> None:
    if not {"EDGE Bursts 20+", "xGF/60", "xGA/60"}.issubset(df.columns):
        return
    d = df.copy()
    d["xG Diff/60"] = d["xGF/60"] - d["xGA/60"]
    _scatter_with_labels(
        d, "xG Diff/60", "EDGE Bursts 20+", "xG Differential /60 (xGF − xGA)",
        "Speed Bursts (20+ mph)", f"EDGE-bursts-vs-xg{dl_suffix}",
        "**Source: NHL EDGE tracking** (speed bursts) vs xG differential (MoneyPuck-"
        "derived), one point per player.",
        team_scoped)


def _edge_topspeed_xg_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "") -> None:
    if not {"EDGE Top Speed", "xGF/60", "xGA/60"}.issubset(df.columns):
        return
    d = df.copy()
    d["xG Diff/60"] = d["xGF/60"] - d["xGA/60"]
    _scatter_with_labels(
        d, "xG Diff/60", "EDGE Top Speed", "xG Differential /60 (xGF − xGA)",
        "Top Speed (mph)", f"EDGE-topspeed-vs-xg{dl_suffix}",
        "**Source: NHL EDGE tracking** (top speed) vs xG differential (MoneyPuck-"
        "derived), one point per player.",
        team_scoped)


def _edge_distance_toi_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "") -> None:
    if not {"EDGE Distance/min", "TOI"}.issubset(df.columns):
        return
    _scatter_with_labels(
        df, "TOI", "EDGE Distance/min", "Minutes Played (TOI)", "Distance Skated (mi/min)",
        f"EDGE-distance-per-min-vs-toi{dl_suffix}",
        "**Source: NHL EDGE tracking** (distance/min) vs minutes played, one "
        "point per player. EDGE distance is all-situations while minutes "
        "played is ES-only, so this rate is an approximation.",
        team_scoped)


def _nfi_xg_scatter(df: pd.DataFrame, team_scoped: bool, dl_suffix: str = "") -> None:
    import altair as alt
    if not {"NFI%", "xGF/60", "xGA/60"}.issubset(df.columns):
        return
    d = df.copy()
    d["xG Diff/60"] = d["xGF/60"] - d["xGA/60"]
    _scatter_with_labels(
        d, "xG Diff/60", "NFI%", "xG Differential /60 (xGF − xGA)", "NFI%",
        f"nfi-vs-xg-differential{dl_suffix}",
        "Net-Front Impact share vs xG differential (MoneyPuck-derived), one point "
        "per player. Color = **OZ Start%** (my PBP data), team-scoped views only — "
        "light = easier/more sheltered zone starts, dark = harder.",
        team_scoped, color_col="OZ Start%",
        color_title="OZ Start% (light = easier, dark = harder)")


def render_players() -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Player List</h2>",
        unsafe_allow_html=True,
    )
    season_label, game_type = render_scoped_filters("players")
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

    # Fixed ES-TOI floor for RANKING: players below it are ranked "(UR)". The Min
    # ES TOI slider (default = this floor) FILTERS the list — by default it hides
    # the sub-floor players; slide it down to reveal them (shown as UR), up to
    # trim further. The slider never changes the ranking denominator.
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
        if playoffs:
            min_toi = st.slider("Min ES TOI (min)", 0, 1500, rank_floor, 25,
                                key="players_toi_playoffs")
        else:
            toi_key = "players_toi_pooled" if is_pooled else "players_toi_season"
            min_toi = st.slider("Min ES TOI (min)", 0, 7500, rank_floor, 50, key=toi_key)

    # Metric-family toggles first, then the Team filter. Families start with none
    # selected (only the identity columns show); click a family to display it.
    fcol, tcol = st.columns([2.8, 1.0])
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
        st.markdown(
            f"<div style='background:{PALETTE['panel']}; border:1px solid {PALETTE['border']}; "
            f"border-radius:8px; padding:0.6rem 0.9rem; margin-bottom:0.6rem;'>"
            f"<span style='color:{PALETTE['orange']}; font-weight:700;'>EDGE columns — "
            f"a DIFFERENT basis than my Zone metrics.</span> "
            f"<span style='color:{PALETTE['text']};'>NHL EDGE tracking data: measured by player "
            f"<b>position</b> (not puck position), across <b>all-situations / even-strength "
            f"TOI</b> (not strict 5v5 faceoff-started shifts). Do not read EDGE zone-time% as "
            f"the same metric as NZI/DZI/OZI or the D/N/O Start% columns (on Zone Impact) — "
            f"different data source, different definition. Each EDGE value shows a computed "
            f"(league / team) rank, same convention as every other column — not NHL's own "
            f"percentile. Pooled/2yr views are a games-played-weighted average across seasons "
            f"— regular season only.</span></div>",
            unsafe_allow_html=True,
        )
        st.session_state.setdefault("players_edge_scope", "All Situations")
        st.radio("EDGE OZ% scope", list(_EDGE_OZ_SCOPE_COL.keys()), horizontal=True,
                 key="players_edge_scope",
                 help="Scope for EDGE OZ% only — NZ%/DZ% always show their one "
                      "available (all-situations) number; NHL doesn't publish an "
                      "even-strength split for those two.")
        st.caption("The scope toggle applies only to **EDGE OZ%** — NZ%/DZ% have no "
                   "even-strength variant from NHL, so they're unaffected.")
    if "xG" in display_fams:
        st.session_state.setdefault("players_pdo_scope", "5v5")
        st.radio("PDO shot scope", list(PDO_SCOPE_FILE.keys()), horizontal=True,
                 key="players_pdo_scope",
                 help="Shot scope for PDO only — every other xG-group column stays "
                      "as-is regardless.")
        st.caption("The shot-scope toggle applies only to **PDO** — other xG columns "
                   "are unaffected.")
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
        "RelxG_pct": "RelxG%", "RelxG_F_pct": "RelxG-F%", "RelxG_A_pct": "RelxG-A%",
        **_EDGE_REN,
    }
    df = df.rename(columns=_ren)
    rank_cohort = rank_cohort.rename(columns=_ren)

    # Always show the full column set (Compact view removed; Qual GP dropped).
    # NFI-A/60 / NFI-S/60 are RAW per-60 rates; RelNFI-A% / RelNFI-S% are the
    # relative (vs own-team) versions — both coexist, placed side by side.
    cols = ["Player", "Pos", "Team", "GP", "TOI",
            "xG-QG%", "xG-QG-F%", "xG-QG-A%", "RelxG-QG%",
            "NFI-QG%", "NFI-QG-A%", "NFI-QG-S%", "RelNFI-QG%",
            "xGF/60", "xGA/60", "RelxG%", "RelxG-F%", "RelxG-A%", "PDO",
            "RelNFI%", "RelNFI-A%", "RelNFI-S%", "NFI%", "NFI-A/60", "NFI-S/60",
            "DZ Start%", "NZ Start%", "OZ Start%", "NZI", "DZI", "OZI",
            *_EDGE_VALUE_DISP]
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
    if "PDO" in disp.columns:
        fmt["PDO"] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    for c in ("NZI", "DZI", "OZI"):
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
    if "EDGE Distance/min" in disp.columns:
        fmt["EDGE Distance/min"] = lambda x: "—" if pd.isna(x) else f"{x:.3f} mi/min"
    if "TOI" in disp.columns:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    for c in ("GP",):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    _player_rank = ["NFI%", "RelNFI%", "RelNFI-A%", "RelNFI-S%", "NFI-A/60",
                    "NFI-S/60", "NZI", "DZI", "OZI", "DZ Start%", "NZ Start%", "OZ Start%",
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
    _pl_qual = pd.to_numeric(disp["TOI"], errors="coerce").fillna(0) >= rank_floor
    _apply_ranks(disp, fmt, rank_cohort, _player_rank, lower_better=_lower,
                 mark_unranked=True, qualified=_pl_qual, team_rank_idx=_team_rank_idx)
    _cohort_label = {"All": "all skaters (F + D)", "F": "forwards",
                     "D": "defense"}[pos]
    _team_txt = team_sel if team_sel != "All" else "their own team"
    st.caption(f"ℹ️ A **blank cell** anywhere on this page means that player or goalie "
               f"fell below the metric's qualifying **sample-size** minimum for that "
               f"scope — it's “not enough data”, not zero. Each metric shows "
               f"**(league rank / team rank)** — rank within **{_cohort_label}** "
               f"league-wide, then within **{_team_txt}**. Only players with "
               f"**≥ {rank_floor:,} ES minutes** are ranked; lower the Min ES TOI "
               f"slider to reveal the rest as **(UR)** = unranked (same meaning as a "
               f"blank cell — shown but unranked, not zero). NFI-S/60 (shots against): "
               f"lowest = #1.")
    _sort_hint()
    st.caption("Click a row to open that player's detail (collapses the list).")
    _gen = st.session_state.get("_pl_tbl_gen", 0)
    _event = _show_df(disp.style.format(fmt, na_rep="—"), hide_index=True,
                      on_select="rerun", selection_mode="single-row",
                      key=f"players_tbl_{_gen}")

    if playoffs:
        zone_note = " · NZI/DZI/OZI pooled across playoffs"
    elif SEASON_KEY.get(season_label) == "pooled_2yr":
        zone_note = (" · NZI/DZI/OZI pooled 2024-25 + 2025-26; "
                     "D/N/O Start% pooled across all seasons (no 2yr build yet)")
    elif is_pooled:
        zone_note = " · NZI/DZI/OZI and D/N/O Start% pooled across all seasons"
    else:
        zone_note = " · NZI/DZI/OZI for this season · D/N/O Start% not available for single seasons (pooled only)"
    st.caption(
        f"{len(disp):,} players (≥ {min_toi:,} ES min) · {scope_label} · sorted by "
        f"RelNFI% descending · ranked at ≥ {rank_floor:,} ES min (else UR){zone_note}"
    )

    import altair as alt
    _team_scoped = team_sel != "All"
    # Scatter/bar/distribution charts below use `df` (merged, renamed, but NOT
    # narrowed by the family-pill column filter that `disp` went through) so
    # they always show whenever their underlying data exists — regardless of
    # which metric-family pills are toggled. Only genuine data-availability
    # gates remain (e.g. is_pooled for Start%, since that data has no
    # per-season cut).
    if {"PDO", "xGF/60", "xGA/60"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>PDO vs xG Differential</h4>",
                    unsafe_allow_html=True)
        _pdo_xg_scatter(df, _team_scoped)

    if {"EDGE DZ%", "EDGE OZ%"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: D-Zone vs "
                    f"O-Zone Time%</h4>", unsafe_allow_html=True)
        _edge_zone_scatter(df, _team_scoped)

    if is_pooled and {"DZ Start%", "OZ Start%"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>Zone Starts: D-Zone vs "
                    f"O-Zone</h4>", unsafe_allow_html=True)
        _zone_start_scatter(df, _team_scoped)
    elif not is_pooled:
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>Zone Starts: D-Zone vs "
                    f"O-Zone</h4>", unsafe_allow_html=True)
        st.caption("⚠️ Not shown for a single-season view — D/N/O Start% has no "
                   "per-season cut (pooled faceoff data only). Switch the **Season** "
                   "filter above to **2yr / 3yr / 4yr** to see this chart.")

    if {"NFI%", "xGF/60", "xGA/60"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>NFI% vs xG Differential</h4>",
                    unsafe_allow_html=True)
        _nfi_xg_scatter(df, _team_scoped)

    if {"EDGE Distance/min", "xGF/60", "xGA/60"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: Distance/min vs "
                    f"xG Differential</h4>", unsafe_allow_html=True)
        _edge_distance_xg_scatter(df, _team_scoped)

    if {"EDGE Bursts 20+", "xGF/60", "xGA/60"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: Speed Bursts vs "
                    f"xG Differential</h4>", unsafe_allow_html=True)
        _edge_bursts_xg_scatter(df, _team_scoped)

    if {"EDGE Top Speed", "xGF/60", "xGA/60"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: Top Speed vs "
                    f"xG Differential</h4>", unsafe_allow_html=True)
        _edge_topspeed_xg_scatter(df, _team_scoped)

    if {"EDGE Distance/min", "TOI"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: Distance/min vs "
                    f"Minutes Played</h4>", unsafe_allow_html=True)
        _edge_distance_toi_scatter(df, _team_scoped)

    if {"EDGE Top Speed", "EDGE Bursts 20+"}.issubset(df.columns):
        st.markdown(f"<h4 style='color:{PALETTE['text']}; margin-top:1rem;'>EDGE: Speed Bursts vs "
                    f"Top Speed</h4>", unsafe_allow_html=True)
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
        ("NZI", _f("NZI", "rate")),
        ("DZI", _f("DZI", "rate")),
        ("OZI", _f("OZI", "rate")),
        ("ES TOI (min)", _f("toi_min", "toi")),
    ]
    st.caption(f"**{r['player_name']} ({r['position']})** · pooled playoffs "
               "(2022-23 → 2024-25).")
    # Horizontal layout: metrics as columns, a single value row (matches the rest
    # of the app), rather than a tall two-column Metric/Value table.
    hdf = pd.DataFrame([{m: v for m, v in items}])[[m for m, _ in items]]
    _show_df(hdf, width="stretch", hide_index=True)

    # Quality-Games diverging bar vs the 50% baseline — same chart as the regular
    # season (NFI% + the four QG %), built from this player's pooled playoff row.
    qg_vals = {m: (float(r[_P2YR_MAP[m]]) * 100
                   if _P2YR_MAP.get(m) in r.index and pd.notna(r.get(_P2YR_MAP[m]))
                   else np.nan)
               for m in _QG_BAR_METRICS}
    if any(pd.notna(v) for v in qg_vals.values()):
        _qg_bar_chart(qg_vals, "Playoffs (2022-25)")


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

    _team_rank = (["NFI%", "Attack events", "Suppress events"] + zcols
                  + ["xG-QG%", "xG-QG-F%", "xG-QG-A%", "NFI-QG%", "NFI-QG-A%", "NFI-QG-S%"])
    _apply_ranks(disp, fmt, disp, _team_rank, lower_better={"Suppress events"})
    st.caption("Each metric shows its **(rank)** across playoff teams. "
               "Suppress events (shots against): lowest = #1.")
    _sort_hint()
    _show_df(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
    st.caption(
        f"{len(disp)} teams · all playoffs (2022-2025 pooled) · sorted by NFI% "
        "(CNFI+MNFI share) descending · Zone Impact (NZI/DZI/OZI) is TOI-weighted."
    )


def render_teams() -> None:
    st.markdown(
        f"<h2 style='color:{PALETTE['text']}; margin-bottom:0.2rem;'>Teams</h2>",
        unsafe_allow_html=True,
    )
    season_label, game_type = render_scoped_filters("teams")
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

    _fa_disp = list(_TEAM_QG_FA.values())   # xG-QG-F%, xG-QG-A%, NFI-QG-A%, NFI-QG-S%
    for c in ["TOI", "xG-QG%", "NFI-QG%", "Attack events",
              "Suppress events"] + zcols + _fa_disp:
        if c not in team.columns:
            team[c] = np.nan

    team = team.rename(columns={"team": "Team"})
    team = team.sort_values("NFI%", ascending=False, na_position="last").reset_index(drop=True)
    cols = (["Team", "GP", "TOI", "NFI%", "Attack events", "Suppress events"]
            + zcols + ["xG-QG%", "xG-QG-F%", "xG-QG-A%",
                       "NFI-QG%", "NFI-QG-A%", "NFI-QG-S%"])
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
                row[c] = f"{txt} ({rk})" if rk is not None else txt
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


def _goalie_consistency_bar(row, qg_label: str, sv_baseline) -> None:
    """One-year diverging bar (like the player QG bar): QNFG%/QG%/sQS% above/below
    the 50% line, and NFI SV% above/below the season's league-average save% —
    two blue cut-off lines. Bar colour: blue above its line, orange below."""
    import altair as alt
    specs = [("QNFG%", 50.0, 1.0), ("QG%", 50.0, 1.0), (qg_label, 50.0, 1.0)]
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
    st.caption(f"**{row['Season']}** — QNFG% / QG% / sQS% vs the **50%** line; "
               "**NFI SV%** vs the season's **league-average save%** (both cut-offs "
               "in blue). Bar up = above the line.")
    bars = alt.Chart(d).mark_bar(size=40).encode(
        x=alt.X("Metric:N", sort=[r["Metric"] for r in rows],
                axis=alt.Axis(labelAngle=0, title=None, labelFontWeight="bold")),
        y=alt.Y("base:Q", scale=alt.Scale(domain=dom), title="%"), y2="value:Q",
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("value:Q", format=".1f")])
    cuts = pd.DataFrame({"y": sorted({50.0} | ({sv_base} if sv_base is not None else set()))})
    rule = alt.Chart(cuts).mark_rule(color=_CHART_THIRD, strokeDash=[4, 4]).encode(y="y:Q")
    _show_chart(bars + rule, dl_name=f"Goalie-consistency-{row['Season']}")


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
    st.caption(f"**{row['Season']}** — GSAx vs the season's **starter-tier average** "
               "(blue line = starter avg): total (left) and per-60 (right). Bar up = "
               "above the average starter. **NFI-GSAx** = net-front, **MP-GSAx** = "
               "all-shot (MoneyPuck).")
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
    _show_chart(alt.hconcat(*panels, spacing=110), dl_name=f"Goalie-GSAx-{row['Season']}")


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
            _svb = (load_nfi_sv_baseline().get(str(int(_row["season"])))
                    if pd.notna(_row.get("season")) else None)
            _goalie_consistency_bar(_row, qg_label, _svb)
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
        g = (b.groupby(["goalie_id", "goalie_name"])
               .agg(GP_qs=("GP", "sum"), _q=("quality_games", "sum"),
                    _maxgp=("GP", "max")).reset_index())
        g["QS_GSAx_pct"] = g["_q"] / g["GP_qs"] * 100
        g["QS_GSAx_lo"] = g.apply(lambda r: _wilson(r["_q"], r["GP_qs"], True) * 100, axis=1)
        g["qual_qs"] = g["_maxgp"] >= 25
        qs = g[["goalie_id", "goalie_name", "GP_qs", "QS_GSAx_pct", "QS_GSAx_lo", "qual_qs"]]
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
        qs = (s[["goalie_id", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo"]]
              .rename(columns={"GP": "GP_qs"}) if not s.empty else pd.DataFrame())
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
        qs = (s[[c for c in ["goalie_id", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo", "qualified"] if c in s.columns]]
              .rename(columns={"GP": "GP_qs", "qualified": "qual_qs"}) if not s.empty else pd.DataFrame())
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
        qs = (s0[s0["season"] == sk][[c for c in ["goalie_id", "goalie_name", "GP", "QS_GSAx_pct", "QS_GSAx_lo", "qualified"] if c in s0.columns]]
              .rename(columns={"GP": "GP_qs", "qualified": "qual_qs"}) if (not s0.empty and sk) else pd.DataFrame())
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
    # Per-metric qualification. PREFER the producer's `qualified` flag when the
    # data file carries it (correct, e.g. pooled requires a season with ≥25 GP,
    # not just accumulated games). Fall back to an in-app floor only when the
    # column is absent (older data file), so the tab never crashes. Playoffs have
    # no floor → rank everyone.
    _gsax_floor = 300 if is_pooled else (200 if is_2yr else 100)
    _fallback = {"qual_gsax": ("total_faced", _gsax_floor),
                 "qual_qn": ("GP_qn", 25), "qual_qs": ("GP_qs", 25),
                 "qual_qg": ("GP_qg", 25)}
    for _qc, (_col, _flr) in _fallback.items():
        if playoffs:
            base[_qc] = True
        elif _qc in base.columns:
            base[_qc] = base[_qc].fillna(False).astype(bool)
        else:
            base[_qc] = base.get(_col, pd.Series(np.nan, index=base.index)).fillna(0) >= _flr
    _gid_of = dict(zip(base["Goalie"], base["goalie_id"]))   # name → id for drill-in

    c1, c2, c3 = st.columns([2.0, 1.3, 1.0])
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

    cols = ["Goalie", "Team", "GP", "NFI-GSAx/60", "NFI SV%", "QNFG%", "QG%", _qg_label]
    disp = base[[c for c in cols if c in base.columns]].copy()

    fmt = {}
    if "NFI-GSAx/60" in disp:
        fmt["NFI-GSAx/60"] = lambda x: "—" if pd.isna(x) else f"{x:+.3f}"
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
    _metric_qual = {"NFI-GSAx/60": "qual_gsax", "NFI SV%": "qual_gsax",
                     "QNFG%": "qual_qn", "QG%": "qual_qs", _qg_label: "qual_qg"}
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
                   f"bar — **NFI-GSAx** ≥ {_shot_floor}, **QNFG / QG / {_qg_label}** ≥ 25 GP "
                   f"— else **(UR)** = unranked.")
    else:
        st.caption(f"Each metric shows its **(rank)**. Every goalie is listed; a metric "
                   f"is ranked only if the goalie clears its bar — **NFI-GSAx** "
                   f"≥ {_shot_floor}, **QNFG / QG / {_qg_label}** ≥ 25 GP — else "
                   f"**(UR)** = unranked.")
    st.caption("**QG** = goals-saved-above-expected, as a game rate (formerly GQG) — "
               "the share of a goalie's games where their all-shot GSAx ≥ 0 (beat "
               "expected on a danger/xG-weighted basis), not raw save%.")
    st.caption("**NFI SV%** = raw (unadjusted) save% on the net-front danger-zone shot "
               "set only (CNFI+MNFI shots faced) — a sanity-check stat, not shot-quality "
               "adjusted like NFI-GSAx.")
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
        f"{len(disp)} goalies (≥ {min_shots:,} shots faced) · {_goalie_scope} · sorted "
        "by NFI-GSAx/60 descending · (UR) = below that metric's ranking floor"
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

    items = [
        ("NFI-GSAx/60", fmt(pick(n, "GSAx_per60"), "gsax")),
        ("NFI SV%", fmt(pick(n, "NFI_save_pct"), "sv")),
        ("QNFG%", fmt(pick(q, "QNFS_pct"), "pct")),
        ("QG%", fmt(pick(s, "QS_GSAx_pct"), "pct")),
        (qg_label, fmt(pick(g, qg_col), "pct")),
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
    # filter's year; the secondary line image = QG % over time.
    _seasons_all = [SEASON_DISPLAY.get(s, s) for s in PROFILE_SEASONS] + ["2yr avg (24-26)"]
    _cmp_yr = _default_qg_year(season_label, _seasons_all)
    _trends, _pv = {}, {}
    for pid in sel:
        _nm = plabel.get(int(pid), str(pid))
        _tr = _player_trend(int(pid))
        if not _tr.empty:
            _trends[_nm] = _tr
            _pv[_nm] = _player_qg_vals(int(pid), _tr, _cmp_yr)
    if _pv:
        _set_dl_title(" vs ".join(_pv.keys()))
        _qg_bar_chart_compare(_pv, _cmp_yr)
    if _trends:
        _set_dl_title(" vs ".join(_trends.keys()))
        _qg_line_chart_compare(_trends)
        _zone_line_chart_compare(_trends)


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


def render_scoped_filters(scope: str) -> tuple[str, str]:
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

    c1, c2 = st.columns([1.2, 2.4])
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
    # Charts use the '⬇ Save PNG' button, so drop the chart fullscreen button
    # (scoped to charts — data tables keep their toolbar) and the Vega menu.
    _css = ("[data-testid='stElementContainer']:has([data-testid='stVegaLiteChart']) "
            "[data-testid='StyledFullScreenButton']{display:none !important;}")
    if _HAS_VLC:
        _css += ".vega-embed details,.vega-embed .vega-actions{display:none !important;}"
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
