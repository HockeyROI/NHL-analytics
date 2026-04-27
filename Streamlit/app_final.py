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
    "2025-26",
    "2024-25",
    "2023-24",
    "2022-23",
    "Pooled (2022–2026)",
]
SEASON_OPTIONS = REGULAR_SEASON_OPTIONS  # back-compat alias for TNZI filter code

# Default season we want selected on first paint (drop-downs render with this
# pre-selected even though it's no longer index 0).
DEFAULT_SEASON = "Pooled (2022–2026)"
DEFAULT_PLAYOFF_SEASON = "2025-26 Playoffs"

GAME_TYPE_OPTIONS = ["Regular Season", "Playoffs"]
# All-playoffs label is now "(2022–2026)" since the dropdown also shows
# 2025-26 even before any 2025-26 playoff data has been played.
PLAYOFF_POOLED_LABEL = "All Playoffs (2022–2026)"
PLAYOFF_SEASON_OPTIONS_BASE = [PLAYOFF_POOLED_LABEL]

SEASON_TO_DTNZI_COL = {
    "2025-26": "DTNZI_25_26",
    "2024-25": "DTNZI_24_25",
    "2023-24": "DTNZI_23_24",
}

FLAG_EMOJI = {"RISING": "🟢", "STABLE": "🟡", "DECLINING": "🔴"}

# Spec colors for TNZI / TOZI / TDZI tertile shading + DTNZI_flag coloring.
# Independent of the softer PALETTE['rising'/'stable'/'declining'] used elsewhere
# in the app — keeping those untouched preserves NFI tab styling.
TERTILE_COLORS = {
    "high": "#44AA66",   # top third  — green
    "mid":  "#FFB700",   # middle     — yellow
    "low":  "#CC3333",   # bottom     — red
}

TERTILE_METRICS = ["TNZI_L", "TNZI", "TOZI", "TDZI"]

DISPLAY_COLUMNS = [
    # Bio: name, team, pos, GP (TNZI source CSVs use 'pos' / 'GP', not the
    # NFI-side 'position' / 'toi_min').
    "player_name", "team", "pos", "GP",
    # FIX 2 / FIX 4 — TNZI_L first metric column. Only the six metric columns
    # the user wants displayed: TNZI_L, TNZI, TOZI, TDZI, DTNZI_recent,
    # DTNZI_flag. ZQoC / ZQoL removed.
    "TNZI_L", "TNZI", "TOZI", "TDZI",
    "DTNZI_recent", "DTNZI_flag",
]

NFI_SEASON_KEY = {
    "Pooled (2022–2026)": "pooled",
    "2025-26": "20252026",
    "2024-25": "20242025",
    "2023-24": "20232024",
    "2022-23": "20222023",
}

# NFI TOI thresholds per season context.
# Playoff TOI is dramatically smaller than regular season — top performers
# log roughly 50–600 ES min over a deep run, so the 2000-min regular-season
# threshold wipes the entire playoff table. Use playoff-specific defaults.
NFI_TOI_DEFAULT = {
    "pooled": 2000, "season": 500,
    "playoffs_pooled": 100, "playoffs_season": 25,
}
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
@st.cache_data(show_spinner=False, ttl=3600)
def load_last_updated(kind: str = "regular") -> str:
    fp = ZONES / f"last_updated_{kind}.txt"
    if not fp.exists():
        return "unknown"
    return fp.read_text().strip()


@st.cache_data(show_spinner=False, ttl=3600)
def load_adjusted(position: str) -> pd.DataFrame:
    fp = ADJ / f"tnzi_adjusted_{position}.csv"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_csv(fp)


@st.cache_data(show_spinner=False, ttl=3600)
def load_tozi_tdzi(metric: str, position: str) -> pd.DataFrame:
    """Read TOZI / TDZI CSVs from Zones/adjusted_rankings/.
    Returns a slim 4-column frame: player_name, team, pos, <metric>."""
    fp = ADJ / f"{metric.lower()}_adjusted_{position}.csv"
    if not fp.exists():
        return pd.DataFrame(columns=["player_name", "team", "pos", metric])
    df = pd.read_csv(fp)
    if metric not in df.columns:
        return pd.DataFrame(columns=["player_name", "team", "pos", metric])
    keep = ["player_name", "team", "pos", metric]
    return df[[c for c in keep if c in df.columns]].copy()


@st.cache_data(show_spinner=False, ttl=3600)
def load_combined_regular() -> pd.DataFrame:
    frames = []
    for pos_file in ("forwards", "defense"):
        d = load_adjusted(pos_file)
        if not d.empty:
            d = d.copy()
            d["_pos_group"] = pos_file
            # Merge TOZI / TDZI columns from their respective CSVs.
            for new_metric in ("TOZI", "TDZI"):
                add = load_tozi_tdzi(new_metric, pos_file)
                if add.empty:
                    d[new_metric] = pd.NA
                    continue
                d = d.merge(add, on=["player_name", "team", "pos"], how="left")
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False, ttl=3600)
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


@st.cache_data(show_spinner=False, ttl=3600)
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


@st.cache_data(show_spinner=False, ttl=3600)
def playoff_season_options(df: pd.DataFrame) -> list[str]:
    """Return playoff season dropdown options in display order:
    [2025-26 Playoffs, 2024-25 Playoffs, 2023-24 Playoffs, 2022-23 Playoffs,
     All Playoffs (2022–2026)]. Most recent first, pooled at the bottom.
    The 2025-26 bucket is always shown even before any games have played —
    the renderers display a coming-soon note in that case."""
    label_map = {
        "20252026": "2025-26 Playoffs",
        "20242025": "2024-25 Playoffs",
        "20232024": "2023-24 Playoffs",
        "20222023": "2022-23 Playoffs",
    }
    # Years that exist in the data
    have = set()
    if df is not None and not df.empty and "season" in df.columns:
        have = set(df["season"].astype(str).unique()) - {"playoffs", "all_playoffs", ""}
    # Build the per-year list: include any year present in data + always include
    # the current playoff bucket (so the coming-soon message is reachable).
    have.add("20252026")
    per_year = [label_map[y] for y in sorted(have, reverse=True) if y in label_map]
    return per_year + [PLAYOFF_POOLED_LABEL]


CURRENT_PLAYOFF_LABEL = "2025-26 Playoffs"
CURRENT_PLAYOFF_KEY = "20252026"


@st.cache_data(show_spinner=False, ttl=3600)
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


@st.cache_data(show_spinner=False, ttl=3600)
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


@st.cache_data(show_spinner=False, ttl=3600)
def tertile_cutoffs(metric: str) -> tuple[float, float] | None:
    df = load_combined_regular()
    if df.empty or metric not in df.columns:
        return None
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(values) < 3:
        return None
    low, high = np.percentile(values, [33.333, 66.666])
    return float(low), float(high)


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
# Goalie + Team Construction loaders (FIX 7, FIX 8)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_goalie_nfi() -> pd.DataFrame:
    """Goalie NFI-GSAx + team + ES TOI + tier (joined from publication file)."""
    base = REPO_ROOT / "NFI" / "output" / "goalie_nfi_gsax.csv"
    if not base.exists():
        return pd.DataFrame()
    df = pd.read_csv(base)

    # ES TOI
    toi_fp = REPO_ROOT / "NFI" / "output" / "player_toi.csv"
    if toi_fp.exists():
        toi = pd.read_csv(toi_fp)
        toi = toi[toi["position"] == "G"][["player_id", "toi_ES_sec"]].copy()
        toi["ES_TOI_min"] = (toi["toi_ES_sec"] / 60.0).round(2)
        df = df.merge(toi.rename(columns={"player_id": "goalie_id"})[
            ["goalie_id", "ES_TOI_min"]
        ], on="goalie_id", how="left")
    else:
        df["ES_TOI_min"] = np.nan

    # Per-60 (goalie_nfi_gsax doesn't carry it directly)
    df["NFI_GSAx_per60"] = np.where(
        df["ES_TOI_min"].fillna(0) > 0,
        df["NFI_GSAx_calibrated"] / df["ES_TOI_min"] * 60.0,
        np.nan,
    )
    df = df.rename(columns={"NFI_GSAx_calibrated": "NFI_GSAx_cumulative"})

    # Team — latest known
    team_fp = REPO_ROOT / "Goalies" / "Benchmarks Goalies" / "Data" / "goalie_team_lookup.csv"
    if team_fp.exists():
        teams = pd.read_csv(team_fp)
        teams_latest = (
            teams.sort_values("season").drop_duplicates("goalie_id", keep="last")[
                ["goalie_id", "goalie_team"]
            ].rename(columns={"goalie_team": "team"})
        )
        df = df.merge(teams_latest, on="goalie_id", how="left")
    else:
        df["team"] = ""

    # Tier from publication file
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
def load_team_construction(season_choice: str = "Current Season (2025-26)") -> pd.DataFrame:
    """Per-team forward RelNFI% (TOI-weighted) joined with the team's primary
    goalie GSAx /60. Uses only player_fully_adjusted CSVs and goalie_nfi_gsax —
    no shots_tagged dependency.

    'Current Season (2025-26)' loads current_season_player_fully_adjusted.csv.
    'Pooled (2022–2026)' loads the full multi-season player file and TOI-weights
    across all seasons.

    Starter goalie per team is identified from the lightweight
    Goalies/Benchmarks Goalies/Data/goalie_team_lookup.csv (games_played per
    goalie-team-season). The primary goalie = the one with the most
    games_played for that team (current season) or summed across seasons
    (pooled view).
    """
    # Resolve which file to load (FIX 9 — accept individual season options)
    cur_fp = NFI_ADJ / "current_season_player_fully_adjusted.csv"
    full_fp = NFI_ADJ / "player_fully_adjusted.csv"
    players = pd.DataFrame()
    season_to_key = {
        "Current Season (2025-26)": "20252026",
        "2025-26": "20252026",
        "2024-25": "20242025",
        "2023-24": "20232024",
        "2022-23": "20222023",
    }
    season_key = season_to_key.get(season_choice)  # None for "Pooled (2022–2026)"

    if season_choice == "Current Season (2025-26)":
        if cur_fp.exists():
            players = pd.read_csv(cur_fp)
            # The current-season export occasionally ships with all-NaN RelNFI
            # columns (mid-pipeline state). If that's the case, fall back to
            # the full file filtered to 20252026.
            if "RelNFI_pct" not in players.columns or players["RelNFI_pct"].isna().all():
                players = pd.DataFrame()
        if players.empty and full_fp.exists():
            full = pd.read_csv(full_fp)
            full["season"] = full["season"].astype(str)
            players = full[full["season"] == "20252026"].copy()
    elif season_key is not None:
        # Per-season slice from the full file
        if not full_fp.exists():
            return pd.DataFrame()
        full = pd.read_csv(full_fp)
        full["season"] = full["season"].astype(str)
        players = full[full["season"] == season_key].copy()
    else:
        # Pooled view across all seasons
        if not full_fp.exists():
            return pd.DataFrame()
        players = pd.read_csv(full_fp)

    if players.empty:
        return pd.DataFrame()

    # FIX 8 — X-axis is ALWAYS NFI_pct_ZA. RelNFI% can't aggregate cleanly
    # (it's defined as on-ice minus off-ice within each team and demeans to
    # ~zero per team), so NFI%_ZA is the correct team-level predictor — no
    # fallback.
    x_metric = "NFI_pct_ZA"
    x_label = "Forward NFI%_ZA"

    fwd = players[(players["position"] == "F")].dropna(
        subset=[x_metric, "team"]
    ).copy()
    if fwd.empty:
        return pd.DataFrame()

    def wmean(g: pd.DataFrame) -> float:
        toi = g["toi_min"].astype(float)
        if toi.sum() <= 0:
            return float("nan")
        return float(np.average(g[x_metric], weights=toi))

    fwd_team = (
        fwd.groupby("team").apply(wmean, include_groups=False)
           .rename("fwd_RelNFI_pct").reset_index()
    )
    fwd_team.attrs["x_metric"] = x_metric
    fwd_team.attrs["x_label"] = x_label

    # Goalie metric — pooled across seasons in goalie_nfi_gsax.csv
    g = load_goalie_nfi()
    if g.empty:
        return pd.DataFrame()

    # Starter per team — use goalie_team_lookup.csv (in-repo, lightweight).
    # If unavailable, fall back to assigning each goalie to the team listed
    # in the goalie loader (latest team).
    lookup_fp = REPO_ROOT / "Goalies" / "Benchmarks Goalies" / "Data" / "goalie_team_lookup.csv"
    if lookup_fp.exists():
        lk = pd.read_csv(lookup_fp)
        if season_key is not None:
            lk = lk[lk["season"].astype(str) == season_key]
        # Highest games_played per (team, goalie) -> primary goalie for the team
        starter = (
            lk.groupby(["goalie_team", "goalie_id"])["games_played"].sum()
              .rename("games").reset_index()
              .sort_values(["goalie_team", "games"], ascending=[True, False])
              .drop_duplicates("goalie_team", keep="first")
              .rename(columns={"goalie_team": "team"})
        )
    else:
        starter = (
            g.dropna(subset=["team"])
             .sort_values("ES_TOI_min", ascending=False)
             .drop_duplicates("team", keep="first")[["team", "goalie_id"]]
        )
    # FIX (EDM dedup) — guarantee one starter row per team
    starter = starter.drop_duplicates(subset=["team"], keep="first")

    starter = starter.merge(
        g[["goalie_id", "goalie_name", "NFI_GSAx_cumulative", "NFI_GSAx_per60"]],
        on="goalie_id", how="left",
    )

    out = fwd_team.merge(starter, on="team", how="inner")
    # FIX (EDM dedup) — final safety net on the merged frame
    out = out.drop_duplicates(subset=["team"], keep="first").reset_index(drop=True)
    out.attrs["x_metric"] = fwd_team.attrs.get("x_metric", "RelNFI_pct")
    out.attrs["x_label"] = fwd_team.attrs.get("x_label", "Forward RelNFI%")
    return out


# FIX 2 — explicit NFI pre-loader. Wraps the pooled NFI loader in a defensive
# try/except so any I/O hiccup at startup never crashes the page.
@st.cache_data(ttl=3600)
def _preload_nfi():
    """Pre-load the pooled NFI dataframe at startup."""
    try:
        return load_nfi_player()
    except Exception:
        return None


def _prefetch_all() -> None:
    """Warm every cached loader at startup so framework tabs render with
    pre-loaded dataframes instead of flashing empty on first paint."""
    try:
        load_combined_regular()
        load_combined_playoffs()
        load_nfi_player()
        load_nfi_playoffs()
        load_goalie_nfi()
        load_team_construction("Current Season (2025-26)")
        load_team_construction("Pooled (2022–2026)")
    except Exception:
        # Loaders are individually defensive; ignore prefetch errors so a
        # broken auxiliary file can never block app startup.
        pass


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
    if season in SEASON_TO_DTNZI_COL:
        col = SEASON_TO_DTNZI_COL[season]
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
    if flag != "All" and "DTNZI_flag" in f.columns:
        f = f[f["DTNZI_flag"] == flag]

    return f


def prepare_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "DTNZI_flag" in out.columns:
        def _fmt(v):
            if pd.isna(v) or v == "" or v is None:
                return ""
            return f"{FLAG_EMOJI.get(v, '')} {v}".strip()
        out["DTNZI_flag"] = out["DTNZI_flag"].apply(_fmt)
    cols = [c for c in DISPLAY_COLUMNS if c in out.columns]
    return out[cols]


def style_frame(display_df: pd.DataFrame, color_map: bool = True):
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
                out.append(f"background-color: {TERTILE_COLORS['high']}; color: white;")
            elif x >= low:
                out.append(f"background-color: {TERTILE_COLORS['mid']}; color: black;")
            else:
                out.append(f"background-color: {TERTILE_COLORS['low']}; color: white;")
        return out

    def color_flag(col):
        out = []
        for v in col:
            s = str(v) if v is not None else ""
            if "RISING" in s:
                out.append(f"background-color: {TERTILE_COLORS['high']}; color: white;")
            elif "STABLE" in s:
                out.append(f"background-color: {TERTILE_COLORS['mid']}; color: black;")
            elif "DECLINING" in s:
                out.append(f"background-color: {TERTILE_COLORS['low']}; color: white;")
            else:
                out.append("")
        return out

    # FIX 4 — TNZI_L is the primary individual metric. Use brand-orange
    # (strongest) heat map intensity, matching the RelNFI% treatment in NFI.
    PRIMARY_TOP = "#FF6B35"
    PRIMARY_MID = "#B07A14"
    PRIMARY_LOW = "#8C2A2A"

    def color_primary(col_name):
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
                out.append(
                    f"background-color: {PRIMARY_TOP}; color: white;"
                    " font-weight: 700;"
                )
            elif x >= low:
                out.append(
                    f"background-color: {PRIMARY_MID}; color: white;"
                    " font-weight: 700;"
                )
            else:
                out.append(
                    f"background-color: {PRIMARY_LOW}; color: white;"
                    " font-weight: 700;"
                )
        return out

    styler = display_df.style
    # FIX 4 — TNZI_L (primary metric) gets the strongest highlight always
    if "TNZI_L" in display_df.columns:
        styler = styler.apply(lambda _c: color_primary("TNZI_L"), subset=["TNZI_L"])

    # Standard tertile shading on the remaining metrics. color_map kept as a
    # pass-through arg for back-compat with the playoff branch.
    for m in TERTILE_METRICS:
        if m == "TNZI_L":
            continue  # already painted as primary
        if m in display_df.columns:
            styler = styler.apply(lambda _c, name=m: color_metric(name), subset=[m])
    if "DTNZI_flag" in display_df.columns:
        styler = styler.apply(color_flag, subset=["DTNZI_flag"])

    fmt = {}
    for m in TERTILE_METRICS:
        if m in display_df.columns:
            fmt[m] = "{:.1f}"
    for m in ("ZQoC", "ZQoL"):
        if m in display_df.columns:
            fmt[m] = "{:.3f}"
    if "DTNZI_recent" in display_df.columns:
        fmt["DTNZI_recent"] = "{:+.3f}"
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
        <strong>Primary Metric: TNZI_L</strong> — Zone impact adjusted for linemate
        quality. Measures how well a player drives zone possession after neutral zone
        faceoffs independent of their linemates. TNZI does not outperform Corsi or
        xG% as a team winning predictor (r=0.603 pooled). It excels at individual
        player context and same-team comparisons.<br><br>
        Already sorted by TNZI_L — the most informative individual metric.
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
            season_opts = [PLAYOFF_POOLED_LABEL]
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
    st.sidebar.selectbox("DTNZI flag", ["All", "RISING", "STABLE", "DECLINING"], key="f_flag")


def render_tnzi_table() -> None:
    game_type = st.session_state.get("game_type_tnzi", "Regular Season")
    season = st.session_state.get("f_season", REGULAR_SEASON_OPTIONS[0])
    if game_type == "Playoffs":
        # FIX 4 — 2025-26 playoffs not yet started
        if season == CURRENT_PLAYOFF_LABEL:
            st.info("2025-26 playoff data will populate automatically as games are played.")
            return
        data = load_combined_playoffs()
        if data.empty:
            st.markdown(
                '<p style="color:#F0F4F8;">🏒 TNZI playoff data coming soon — populates automatically as games are played</p>',
                unsafe_allow_html=True,
            )
            return
        # Per-year playoff filter: match the dropdown label to the season
        # column written by build_playoff_data.py ('all_playoffs' or
        # int year like 20242025).
        target_key = PLAYOFF_SEASON_LABEL_TO_KEY.get(season)
        if target_key is not None and "season" in data.columns:
            # Same defensive cast as the NFI playoff filter
            data = data[data["season"].astype("string").astype(str) == str(target_key)]
    else:
        data = load_combined_regular()
        if data.empty:
            st.error("Adjusted ranking files not found under Zones/adjusted_rankings/.")
            return
    filtered = apply_filters(data)
    if filtered.empty:
        st.markdown(
            '<p style="color:#F0F4F8;">No players match the current filters. '
            'Widen position, team, or GP filters.</p>',
            unsafe_allow_html=True,
        )
        return

    display_df = prepare_display(filtered)
    # FIX 3 — Default sort: TNZI_L descending. If a row is missing TNZI_L,
    # fall back to that row's TNZI (per-row fallback, not just per-column).
    # Build a synthetic "_sort" key that prefers TNZI_L and uses TNZI when
    # TNZI_L is NaN.
    if "TNZI_L" in display_df.columns:
        primary = pd.to_numeric(display_df["TNZI_L"], errors="coerce")
        fallback = (pd.to_numeric(display_df["TNZI"], errors="coerce")
                    if "TNZI" in display_df.columns else primary)
        display_df = display_df.assign(_sort=primary.fillna(fallback))
        display_df = display_df.sort_values(
            "_sort", ascending=False, na_position="last"
        ).drop(columns=["_sort"])
        default_sort = "TNZI_L"
    elif "TNZI" in display_df.columns:
        display_df = display_df.sort_values(
            "TNZI", ascending=False, na_position="last"
        )
        default_sort = "TNZI"
    else:
        default_sort = display_df.columns[0]
        display_df = display_df.sort_values(
            default_sort, ascending=False, na_position="last"
        )

    is_pooled_view = season == DEFAULT_SEASON
    st.dataframe(
        style_frame(display_df, color_map=is_pooled_view),
        width="stretch", hide_index=True,
    )
    st.caption(
        f"Showing {len(display_df):,} players — sorted by {default_sort} desc"
        + ("" if is_pooled_view else " · single-season view (heat map shown only on pooled)")
    )


def render_tnzi_explainers() -> None:
    W = "#F0F4F8"  # body text color — inline to bypass CSS specificity
    with st.expander("What each metric means"):
        st.markdown(EXPANDER_STYLE, unsafe_allow_html=True)
        st.markdown(
            f'<ul style="color:{W};">'
            f'<li style="color:{W};"><strong>TNZI</strong>: Net zone driving from neutral ice faceoffs. OZ event time% minus DZ event time% after NZ starts. Best single zone time predictor of team winning (r=0.490 pooled).</li>'
            f'<li style="color:{W};"><strong>TOZI</strong>: Net zone sustain after offensive zone faceoffs. OZ minus DZ event time% after OZ starts. Did you hold the zone or give it back?</li>'
            f'<li style="color:{W};"><strong>TDZI</strong>: Net zone transition after defensive zone faceoffs. OZ minus DZ event time% after DZ starts. Did you fully escape your own end and transition to attack? (r=0.448 pooled)</li>'
            f'<li style="color:{W};"><strong>DTNZI</strong>: Delta on TNZI — year over year change in zone driving score. RISING = improving. DECLINING = deteriorating. STABLE = consistent.</li>'
            f'</ul>',
            unsafe_allow_html=True,
        )

    with st.expander("Methodology"):
        st.markdown(EXPANDER_STYLE, unsafe_allow_html=True)
        st.markdown(
            f'<ul style="color:{W};">'
            f'<li style="color:{W};">Zone tracking from NHL API x/y coordinates</li>'
            f'<li style="color:{W};">All events with xCoord contribute — shots, hits, blocked shots, faceoffs, giveaways, takeaways</li>'
            f'<li style="color:{W};">Zone boundaries: OZ <code style="background-color:#1B3A5C; color:#4AB3E8; padding:2px 4px; border-radius:3px;">x &gt; 25</code>, NZ <code style="background-color:#1B3A5C; color:#4AB3E8; padding:2px 4px; border-radius:3px;">-25 to 25</code>, DZ <code style="background-color:#1B3A5C; color:#4AB3E8; padding:2px 4px; border-radius:3px;">x &lt; -25</code></li>'
            f'<li style="color:{W};">Faceoff shifts only — line changes excluded</li>'
            f'<li style="color:{W};">5v5 situations only via <code style="background-color:#1B3A5C; color:#4AB3E8; padding:2px 4px; border-radius:3px;">situationCode 1551</code></li>'
            f'<li style="color:{W};">Wilson CI at 95% confidence using faceoff shift count as <code style="background-color:#1B3A5C; color:#4AB3E8; padding:2px 4px; border-radius:3px;">n</code></li>'
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
        <strong>Primary Metric: RelNFI%</strong> — Two-way dangerous zone impact
        (generation + suppression). r=0.769 vs standings across 126 team-seasons.
        Beats xG% (r=0.731), HD Fenwick (r=0.732), and Corsi (r=0.650).<br><br>
        Sort by <strong>RelNFI%</strong> for most complete players.
        Sort by <strong>RelNFI_F%</strong> for pure generators.
        Sort by <strong>RelNFI_A%</strong> for pure suppressors.
        Zone adjustment applied using <strong>3.5pp conventional factor</strong>.
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
    st.sidebar.radio(
        "Position", ["All", "Forwards", "Defensemen", "Goalies"], key="nfi_position"
    )

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
            season_opts = [PLAYOFF_POOLED_LABEL]
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
        is_playoffs = game_type == "Playoffs"
        if is_playoffs:
            is_pooled_view = chosen_season == PLAYOFF_POOLED_LABEL
            default_toi = (
                NFI_TOI_DEFAULT["playoffs_pooled"]
                if is_pooled_view else NFI_TOI_DEFAULT["playoffs_season"]
            )
        else:
            is_pooled_view = chosen_season == "Pooled (2022–2026)"
            default_toi = (
                NFI_TOI_DEFAULT["pooled"]
                if is_pooled_view else NFI_TOI_DEFAULT["season"]
            )
        st.session_state["nfi_min_toi"] = default_toi
        st.session_state["nfi_last_season"] = chosen_season

    st.sidebar.multiselect("Teams", NHL_TEAMS, key="nfi_teams")
    st.sidebar.text_input("Player name contains", key="nfi_name")
    st.sidebar.slider(
        "Minimum ES TOI (min)",
        min_value=0, max_value=NFI_TOI_MAX, step=50, key="nfi_min_toi",
    )

    # FIX 10 — Corsi / Fenwick comparison toggle (skater views only)
    if st.session_state.get("nfi_position", "All") != "Goalies":
        st.sidebar.checkbox(
            "Show Corsi/Fenwick comparison columns",
            key="nfi_show_corsi_fenwick",
            value=st.session_state.get("nfi_show_corsi_fenwick", False),
            help="Adds CF%_ZA and FF%_ZA next to NFI%_ZA for direct comparison.",
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

        # FIX 9 — pooled MOM: use the most recent season's MOM (not blank)
        mom_latest = np.nan
        if "NFI_pct_3A_MOM" in g.columns:
            recent = g.sort_values("season").dropna(subset=["NFI_pct_3A_MOM"])
            if not recent.empty:
                mom_latest = float(recent["NFI_pct_3A_MOM"].iloc[-1])

        rows.append({
            "player_id": int(pid),
            "player_name": name,
            "position": pos,
            "team": g.sort_values("season")["team"].iloc[-1],
            "toi_min": toi_total,
            "NFI_pct":    tw("NFI_pct"),
            "NFI_pct_ZA": tw("NFI_pct_ZA"),
            "NFI_pct_3A": tw("NFI_pct_3A"),
            "RelNFI_F_pct": tw("RelNFI_F_pct"),
            "RelNFI_A_pct": tw("RelNFI_A_pct"),
            "RelNFI_pct":   tw("RelNFI_pct"),
            "NFQOC":        tw("NFQOC"),
            "NFQOL":        tw("NFQOL"),
            "NFI_pct_3A_MOM": mom_latest,
            "CF_pct_ZA":     tw("CF_pct_ZA"),
            "FF_pct_ZA":     tw("FF_pct_ZA"),
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
            "NFI_pct", "NFI_pct_ZA", "NFI_pct_3A",
            "RelNFI_F_pct", "RelNFI_A_pct", "RelNFI_pct",
            "NFQOC", "NFQOL",
            "NFI_pct_3A_MOM",
            "CF_pct_ZA", "FF_pct_ZA",
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


PLAYOFF_SEASON_LABEL_TO_KEY = {
    PLAYOFF_POOLED_LABEL: "all_playoffs",
    "2022-23 Playoffs": "20222023",
    "2023-24 Playoffs": "20232024",
    "2024-25 Playoffs": "20242025",
    "2025-26 Playoffs": "20252026",
}


def _filter_nfi_playoffs(df: pd.DataFrame, season_opt: str) -> pd.DataFrame:
    """Filter the NFI playoff table by sidebar widgets.
    The playoff CSV contains both per-season rows (season=20222023, etc.)
    and pooled rows (season='all_playoffs'). The dropdown maps to the
    matching season key so we never mix pooled and per-season rows."""
    if df.empty:
        return df
    out = df.copy()
    position = st.session_state.get("nfi_position", "All")
    if position == "Forwards":
        out = out[out["position"] == "F"]
    elif position == "Defensemen":
        out = out[out["position"] == "D"]
    if "season" in out.columns:
        target_key = PLAYOFF_SEASON_LABEL_TO_KEY.get(season_opt)
        if target_key is not None:
            # Defensive: coerce both sides to plain Python str so ArrowStringArray
            # vs str (or older numpy.int) mismatches never produce empty filters.
            out = out[out["season"].astype("string").astype(str) == str(target_key)]
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
        "NFI_pct": "NFI%",
        "NFI_pct_ZA": "NFI%_ZA", "NFI_pct_3A": "NFI%_3A",
        "RelNFI_F_pct": "RelNFI_F%", "RelNFI_A_pct": "RelNFI_A%", "RelNFI_pct": "RelNFI%",
        "NFQOC": "NFQOC", "NFQOL": "NFQOL",
        "NFI_pct_3A_MOM": "NFI%_3A_MOM",
        "CF_pct_ZA": "CF%_ZA", "FF_pct_ZA": "FF%_ZA",
    })
    show_cf_ff = bool(st.session_state.get("nfi_show_corsi_fenwick", False))
    # FIX 5 — display only the columns the spec requires:
    # Player, Team, Position, TOI, RelNFI%, RelNFI_F%, RelNFI_A%, NFI%, NFI%_ZA.
    # NFI%_3A, NFQOC, NFQOL, NFI%_3A_MOM dropped from the displayed table.
    cols = ["Player", "Pos", "Team", "TOI",
            "RelNFI%", "RelNFI_F%", "RelNFI_A%",
            "NFI%", "NFI%_ZA"]
    if show_cf_ff:
        # Insert CF/FF right next to NFI%_ZA for direct comparison
        idx = cols.index("NFI%_ZA")
        cols = cols[:idx] + ["CF%_ZA", "FF%_ZA"] + cols[idx:]
    cols = [c for c in cols if c in out.columns]
    return out[cols]


def _style_nfi(display_df: pd.DataFrame, color_map: bool = True):
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

    def color_pct_tertile(col):
        vals = pd.to_numeric(col, errors="coerce")
        clean = vals.dropna()
        if len(clean) < 6:
            return [""] * len(col)
        low, high = np.percentile(clean, [33.333, 66.666])
        out = []
        for v in vals:
            if pd.isna(v):
                out.append("")
            elif v >= high:
                out.append(f"background-color: {PALETTE['rising']}; color: white;")
            elif v >= low:
                out.append(f"background-color: {PALETTE['stable']}; color: black;")
            else:
                out.append(f"background-color: {PALETTE['declining']}; color: white;")
        return out

    # FIX 3 — RelNFI% is the primary metric. Top tertile gets the brand
    # orange (strongest possible visual emphasis), middle tertile a darker
    # gold, bottom tertile a darker red. Bold weight throughout.
    PRIMARY_TOP = "#FF6B35"   # brand orange
    PRIMARY_MID = "#B07A14"   # darker gold for mid tier
    PRIMARY_LOW = "#8C2A2A"   # darker red for bottom tier

    def color_primary(col):
        vals = pd.to_numeric(col, errors="coerce")
        clean = vals.dropna()
        if len(clean) < 6:
            return [""] * len(col)
        low, high = np.percentile(clean, [33.333, 66.666])
        out = []
        for v in vals:
            if pd.isna(v):
                out.append("")
            elif v >= high:
                out.append(
                    f"background-color: {PRIMARY_TOP}; color: white;"
                    " font-weight: 700;"
                )
            elif v >= low:
                out.append(
                    f"background-color: {PRIMARY_MID}; color: white;"
                    " font-weight: 700;"
                )
            else:
                out.append(
                    f"background-color: {PRIMARY_LOW}; color: white;"
                    " font-weight: 700;"
                )
        return out

    styler = display_df.style
    # FIX 3 — RelNFI% column highlighted with strongest intensity always
    if "RelNFI%" in display_df.columns:
        styler = styler.apply(color_primary, subset=["RelNFI%"])

    # FIX 7 — secondary heat map only when color_map flag is on (pooled view).
    if color_map:
        if "NFI%_3A_MOM" in display_df.columns:
            styler = styler.apply(color_mom, subset=["NFI%_3A_MOM"])
        for c in ("NFI%_ZA", "CF%_ZA", "FF%_ZA"):
            if c in display_df.columns:
                styler = styler.apply(color_pct_tertile, subset=[c])

    fmt = {}
    for c in ("NFI%", "NFI%_ZA", "NFI%_3A", "NFQOC", "NFQOL", "CF%_ZA", "FF%_ZA"):
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

    # FIX 6 — Goalies branch: pass current game_type + season so the goalie
    # table refreshes when either selector changes.
    if st.session_state.get("nfi_position", "All") == "Goalies":
        render_nfi_goalie_table(game_type=game_type, season=season)
        return

    if game_type == "Playoffs":
        # FIX 4 — 2025-26 playoffs not yet started
        if season == CURRENT_PLAYOFF_LABEL:
            st.info("2025-26 playoff data will populate automatically as games are played.")
            return
        if not nfi_playoffs_available():
            st.markdown(
                '<p style="color:#F0F4F8;">🏒 Playoff data coming soon — populates automatically as games are played</p>',
                unsafe_allow_html=True,
            )
            return
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
        st.markdown(
            '<p style="color:#F0F4F8;">No players match the current filters. '
            'Widen position, team, or TOI filters.</p>',
            unsafe_allow_html=True,
        )
        return

    # FIX 12 — default sort RelNFI% descending (fall back if missing)
    sort_col = next(
        (c for c in ("RelNFI_pct", "NFI_pct_ZA") if c in filtered.columns),
        filtered.columns[0],
    )
    filtered = filtered.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    filtered.insert(0, "Rank", np.arange(1, len(filtered) + 1))
    display_df = _nfi_display(filtered)
    display_df.insert(0, "#", filtered["Rank"].values)

    # FIX 9 — safe_avg: NaN guard so playoff data (RelNFI all NaN) renders
    # +0.00 instead of +nan in the three summary boxes.
    def safe_avg(series):
        if series is None:
            return 0.0
        val = series.mean()
        return 0.0 if pd.isna(val) else float(val)

    avg_f = safe_avg(display_df["RelNFI_F%"]) if "RelNFI_F%" in display_df.columns else 0.0
    avg_a = safe_avg(display_df["RelNFI_A%"]) if "RelNFI_A%" in display_df.columns else 0.0
    avg_rel = safe_avg(display_df["RelNFI%"]) if "RelNFI%" in display_df.columns else 0.0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Generation", f"{avg_f:+.2f}",
                  help="Average RelNFI_F% for filtered players")
    with col2:
        st.metric("Avg Suppression", f"{avg_a:+.2f}",
                  help="Average RelNFI_A% for filtered players")
    with col3:
        st.metric("Avg Two-Way", f"{avg_rel:+.2f}",
                  help="Average RelNFI% for filtered players")

    is_pooled_view = season == DEFAULT_SEASON
    st.dataframe(
        _style_nfi(display_df, color_map=is_pooled_view),
        width="stretch", hide_index=True,
    )
    sort_label = {"RelNFI_pct": "RelNFI%", "NFI_pct_ZA": "NFI%_ZA"}.get(sort_col, sort_col)
    caption = f"Showing {len(display_df):,} players — sorted by {sort_label} descending"
    if not is_pooled_view:
        caption += "  ·  single-season view (heat map shown only on pooled)"
    if "small_sample" in filtered.columns and filtered["small_sample"].any():
        n_small = int(filtered["small_sample"].sum())
        caption += f"  |  {n_small} small-sample players flagged with *"
    st.caption(caption)


# ---------------------------------------------------------------------------
# FIX 7 — Goalie NFI-GSAx table
# ---------------------------------------------------------------------------
def render_nfi_goalie_table(game_type: str | None = None,
                            season: str | None = None) -> None:
    """FIX 6 — reactive to game_type and season selectors.

    - Playoffs → "Goalie playoff data coming soon." (always, regardless of
      which playoff year is selected).
    - Regular Season → renders the goalie table; per-season views show pooled
      multi-season data with a caption noting per-season breakdown is upcoming.
    """
    # If the caller didn't pass them, read from session state. Reading state
    # here also makes Streamlit re-run this function whenever the toggles
    # change, so the table refreshes with each switch.
    if game_type is None:
        game_type = st.session_state.get("game_type_nfi", "Regular Season")
    if season is None:
        season = st.session_state.get("nfi_season", DEFAULT_SEASON)

    if game_type == "Playoffs":
        st.info("Goalie playoff data coming soon.")
        return

    st.markdown(
        f"<p style='color:{PALETTE['text']}; font-size:0.95rem; line-height:1.5;'>"
        "Goalie NFI-GSAx measures goals saved above expected from dangerous zones. "
        "Validated against MoneyPuck GSAx at r=0.858. "
        "Minimum 2000 ES minutes for qualification.</p>",
        unsafe_allow_html=True,
    )

    df = load_goalie_nfi()
    if df.empty:
        st.error("Goalie file not found at `NFI/output/goalie_nfi_gsax.csv`.")
        return

    # Per-season caption — the underlying CSV is pooled across seasons, so
    # per-season views show the pooled table with a heads-up caption.
    season_caption = None
    if season != DEFAULT_SEASON:
        season_caption = (
            f"Showing pooled multi-season goalie data — "
            f"per-season goalie breakdown for {season} coming soon."
        )

    # Filters — goalies use a fixed 2000-min qualification regardless of the
    # skater TOI slider; the slider exists only for skaters.
    teams = st.session_state.get("nfi_teams", [])
    if teams and "team" in df.columns:
        df = df[df["team"].isin(teams)]
    name_q = (st.session_state.get("nfi_name", "") or "").strip().lower()
    if name_q:
        df = df[df["goalie_name"].fillna("").str.lower().str.contains(name_q, na=False)]

    GOALIE_MIN_TOI = 2000
    if "ES_TOI_min" in df.columns and df["ES_TOI_min"].notna().any():
        df = df[df["ES_TOI_min"].fillna(0) >= GOALIE_MIN_TOI]

    if df.empty:
        st.markdown(
            '<p style="color:#F0F4F8;">No goalies match the current filters.</p>',
            unsafe_allow_html=True,
        )
        return

    df = df.sort_values("NFI_GSAx_cumulative", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "#", np.arange(1, len(df) + 1))

    show_cols = {
        "#": "#",
        "goalie_name": "Goalie",
        "team": "Team",
        "ES_TOI_min": "TOI",
        "NFI_GSAx_cumulative": "NFI_GSAx_cumulative",
        "NFI_GSAx_per60": "NFI_GSAx_per60",
        "Tier": "Tier",
    }
    if "CNFI_rebound_goal_rate" in df.columns:
        show_cols["CNFI_rebound_goal_rate"] = "CNFI_rebound_goal_rate"
        show_cols["rebound_z"] = "z_score"

    disp = df[[c for c in show_cols if c in df.columns]].rename(columns=show_cols)

    fmt = {}
    if "TOI" in disp.columns:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    if "NFI_GSAx_cumulative" in disp.columns:
        fmt["NFI_GSAx_cumulative"] = lambda x: "—" if pd.isna(x) else f"{x:+.2f}"
    if "NFI_GSAx_per60" in disp.columns:
        fmt["NFI_GSAx_per60"] = lambda x: "—" if pd.isna(x) else f"{x:+.3f}"
    if "CNFI_rebound_goal_rate" in disp.columns:
        fmt["CNFI_rebound_goal_rate"] = lambda x: "—" if pd.isna(x) else f"{x:.3f}"
    if "z_score" in disp.columns:
        fmt["z_score"] = lambda x: "—" if pd.isna(x) else f"{x:+.2f}"

    styler = disp.style.format(fmt, na_rep="—")
    st.dataframe(styler, width="stretch", hide_index=True)
    if season_caption:
        st.caption(season_caption)
    st.caption(f"Showing {len(disp):,} goalies — sorted by NFI_GSAx_cumulative descending")


def render_nfi_explainers() -> None:
    W = "#F0F4F8"
    with st.expander("What each NFI metric means"):
        st.markdown(EXPANDER_STYLE, unsafe_allow_html=True)
        st.markdown(
            f'<ul style="color:{W};">'
            f'<li style="color:{W};"><strong>NFI%</strong> — <strong>Net Front Impact %</strong>. Fenwick attempts (shots on goal + misses + goals, blocks excluded) filtered to CNFI (central net-front) and MNFI (mid net-front) zones. Team CNFI+MNFI for / (for + against) while the player is on ice. <strong>R² = 0.583 vs standings</strong> — beats xG% (0.538) and Corsi (0.397).</li>'
            f'<li style="color:{W};"><strong>NFI%_ZA</strong> — Zone-Adjusted using the <strong>3.5pp conventional zone adjustment factor (Tulsky 2013)</strong>.</li>'
            f'<li style="color:{W};"><strong>NFI%_3A</strong> — <strong>Three-Adjusted</strong>: zone adjustment + NFQOC + NFQOL.</li>'
            f'<li style="color:{W};"><strong>RelNFI_F%</strong> — on-ice minus off-ice team CNFI+MNFI For per 60. Positive = team generates more dangerous shots with player on ice.</li>'
            f'<li style="color:{W};"><strong>RelNFI_A%</strong> — off-ice minus on-ice CNFI+MNFI Against per 60. Positive = suppresses more.</li>'
            f'<li style="color:{W};"><strong>RelNFI%</strong> — net two-way dangerous-zone impact = RelNFI_F% + RelNFI_A%.</li>'
            f'<li style="color:{W};"><strong>NFQOC</strong> — <strong>Net Front Quality of Competition</strong> — shared-TOI weighted mean of opponents\' NFI%, computed linemate-without-me to avoid shared-event collinearity.</li>'
            f'<li style="color:{W};"><strong>NFQOL</strong> — <strong>Net Front Quality of Linemates</strong> — same approach for teammates.</li>'
            f'<li style="color:{W};"><strong>NFI%_3A_MOM</strong> — year-over-year change in NFI%_3A. Positive = ascending.</li>'
            f'<li style="color:#F0F4F8;"><strong>NFI%_3A_MOM vs DTNZI</strong>: NFI%_3A_MOM tracks year-over-year change in quality-adjusted dangerous zone performance. DTNZI in the Zone Impact tab tracks zone possession change. They measure analogous momentum concepts through different methodologies — check both tabs for the most complete player trajectory picture.</li>'
            f'</ul>',
            unsafe_allow_html=True,
        )

    with st.expander("Methodology"):
        st.markdown(EXPANDER_STYLE, unsafe_allow_html=True)
        st.markdown(
            f'<ul style="color:{W};">'
            f'<li style="color:{W};">NFI zones (CNFI, MNFI) derived from shot-density clustering of NHL API x/y coordinates</li>'
            f'<li style="color:{W};">Fenwick events only (shots on goal + missed + goals). Blocks excluded — their coordinates are recorded at the blocker\'s location, not the shooter\'s.</li>'
            f'<li style="color:{W};">5v5 ES regulation only (state = ES in <code style="background-color:#1B3A5C; color:#4AB3E8; padding:2px 4px; border-radius:3px;">shots_tagged.csv</code>)</li>'
            f'<li style="color:{W};">Per-player on-ice attribution: each ES shot event counts for all skaters on-ice</li>'
            f'<li style="color:{W};">Zone factor: <strong>3.5pp conventional zone adjustment (Tulsky 2013)</strong> applied to the OZ/DZ deployment ratio</li>'
            f'<li style="color:{W};">NFQOC / NFQOL use a <strong>linemate-without-me</strong> correction — teammate <code style="background-color:#1B3A5C; color:#4AB3E8; padding:2px 4px; border-radius:3px;">j</code>\'s rating is recomputed excluding events where both <code style="background-color:#1B3A5C; color:#4AB3E8; padding:2px 4px; border-radius:3px;">i</code> and <code style="background-color:#1B3A5C; color:#4AB3E8; padding:2px 4px; border-radius:3px;">j</code> were on ice, preventing β_QoL ≈ 1.0</li>'
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
# FIX 8 — Team Construction (Two Pillar) page
# ---------------------------------------------------------------------------
def render_team_construction_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer-box">
        <strong>Team Construction (Two Pillar)</strong> — Pairs each team's
        forward group RelNFI% (TOI-weighted) against its starter goalie's NFI-GSAx
        per 60. Quadrants reveal which teams are complete, which lean entirely on
        their goalie, which are exposed when the goalie struggles, and which are
        rebuilding.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_team_construction_sidebar() -> None:
    st.sidebar.header("Team Construction Filters")
    st.sidebar.markdown("### Season")
    st.session_state.setdefault("tc_season", "Current Season (2025-26)")
    # FIX 9 — two options only, current vs pooled
    st.sidebar.selectbox(
        "Season",
        ["Current Season (2025-26)", "Pooled (2022–2026)"],
        key="tc_season",
    )
    st.sidebar.caption(
        "Forward NFI%_ZA is TOI-weighted across the team. "
        "Starter goalie is the one with the most games played for the team."
    )


def render_team_construction() -> None:
    import matplotlib.pyplot as plt

    season_choice = st.session_state.get("tc_season", "Current Season (2025-26)")
    df = load_team_construction(season_choice)
    if df.empty:
        st.error(
            "Team construction data unavailable — required: "
            "`player_fully_adjusted.csv` and `goalie_nfi_gsax.csv`."
        )
        return

    # Carry through which forward metric is on the X-axis (RelNFI fallback
    # to NFI%_ZA when RelNFI is unavailable).
    x_label = df.attrs.get("x_label", "Forward RelNFI%")
    sub = df.dropna(subset=["fwd_RelNFI_pct", "NFI_GSAx_per60"]).copy()
    if sub.empty:
        st.markdown(
            '<p style="color:#F0F4F8;">No teams with both metrics available for this view.</p>',
            unsafe_allow_html=True,
        )
        return

    x = sub["fwd_RelNFI_pct"]
    y = sub["NFI_GSAx_per60"]
    x_avg = x.mean()
    y_avg = y.mean()

    sub["q"] = np.select(
        [
            (sub["fwd_RelNFI_pct"] >= x_avg) & (sub["NFI_GSAx_per60"] >= y_avg),
            (sub["fwd_RelNFI_pct"] < x_avg) & (sub["NFI_GSAx_per60"] < y_avg),
            (sub["fwd_RelNFI_pct"] < x_avg) & (sub["NFI_GSAx_per60"] >= y_avg),
            (sub["fwd_RelNFI_pct"] >= x_avg) & (sub["NFI_GSAx_per60"] < y_avg),
        ],
        ["Complete Teams", "Rebuilding", "Goalie Dependent", "Goalie Exposed"],
        default="?",
    )

    GREEN, RED, YELLOW, ORANGE, NAVY = "#5DAA7A", "#C05555", "#D4A843", "#FF6B35", "#0B1D2E"

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xmin, xmax = x.min() - 0.05, x.max() + 0.05
    ymin, ymax = y.min() - 0.05, y.max() + 0.05
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    x_frac = (x_avg - xmin) / (xmax - xmin)
    y_frac = (y_avg - ymin) / (ymax - ymin)
    ax.axhspan(y_avg, ymax, xmin=x_frac, xmax=1.0, facecolor=GREEN, alpha=0.13)
    ax.axhspan(ymin, y_avg, xmin=0.0, xmax=x_frac, facecolor=RED, alpha=0.13)
    ax.axhspan(y_avg, ymax, xmin=0.0, xmax=x_frac, facecolor=YELLOW, alpha=0.13)
    ax.axhspan(ymin, y_avg, xmin=x_frac, xmax=1.0, facecolor=YELLOW, alpha=0.13)

    ax.axvline(x_avg, color=NAVY, linestyle="--", linewidth=1)
    ax.axhline(y_avg, color=NAVY, linestyle="--", linewidth=1)

    ax.text(xmax, ymax, "  Complete Teams", ha="right", va="top",
            fontsize=11, color=GREEN, weight="bold")
    ax.text(xmin, ymin, "  Rebuilding", ha="left", va="bottom",
            fontsize=11, color=RED, weight="bold")
    ax.text(xmin, ymax, "  Goalie Dependent", ha="left", va="top",
            fontsize=11, color="#9B7E1E", weight="bold")
    ax.text(xmax, ymin, "  Goalie Exposed", ha="right", va="bottom",
            fontsize=11, color="#9B7E1E", weight="bold")

    qcol = {"Complete Teams": GREEN, "Rebuilding": RED,
            "Goalie Dependent": YELLOW, "Goalie Exposed": YELLOW}
    for q, c in qcol.items():
        sel = sub[sub["q"] == q]
        ax.scatter(sel["fwd_RelNFI_pct"], sel["NFI_GSAx_per60"],
                   s=140, color=c, edgecolor=NAVY, linewidth=1, zorder=3)

    # Guarantee a single highlight even if upstream data accidentally carries
    # two EDM rows (e.g. two goalies tied on games-played for the same team).
    edm = sub[sub["team"] == "EDM"].head(1)
    if not edm.empty:
        ax.scatter(edm["fwd_RelNFI_pct"], edm["NFI_GSAx_per60"],
                   s=300, facecolors="none", edgecolors=ORANGE,
                   linewidth=3, zorder=4, label="EDM")

    for _, r in sub.iterrows():
        ax.annotate(r["team"], (r["fwd_RelNFI_pct"], r["NFI_GSAx_per60"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, color=NAVY, weight="bold")

    ax.set_xlabel(f"Team Forwards {x_label} (TOI-weighted)", color=NAVY)
    ax.set_ylabel("Starter Goalie NFI-GSAx per 60", color=NAVY)
    ax.set_title("Team Construction — Forwards × Goalie", color=NAVY,
                 fontsize=14, weight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=NAVY)
    if not edm.empty:
        ax.legend(loc="lower right", frameon=False)

    st.pyplot(fig, clear_figure=True, use_container_width=True)

    # Sortable rank table
    rank_df = sub.copy()
    rank_df["fwd_rank"] = rank_df["fwd_RelNFI_pct"].rank(ascending=False, method="min").astype(int)
    rank_df["goalie_rank"] = rank_df["NFI_GSAx_per60"].rank(ascending=False, method="min").astype(int)
    rank_df["combined_rank"] = (rank_df["fwd_rank"] + rank_df["goalie_rank"]).rank(method="min").astype(int)
    rank_df = rank_df.sort_values("combined_rank")
    out = rank_df[["team", "fwd_rank", "goalie_rank", "combined_rank", "q",
                   "goalie_name", "fwd_RelNFI_pct", "NFI_GSAx_per60"]].rename(
        columns={
            "team": "Team",
            "fwd_rank": f"{x_label} rank",
            "goalie_rank": "Goalie GSAx rank",
            "combined_rank": "Combined rank",
            "q": "Quadrant",
            "goalie_name": "Starter",
            "fwd_RelNFI_pct": x_label,
            "NFI_GSAx_per60": "Goalie GSAx /60",
        }
    )
    st.dataframe(
        out.style.format({
            x_label: "{:+.3f}",
            "Goalie GSAx /60": "{:+.3f}",
        }),
        width="stretch", hide_index=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # FIX 11 — sidebar always starts expanded so desktop users see filters.
    st.set_page_config(
        page_title="HockeyROI — Zone Time + Net Front Impact",
        page_icon="🏒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # FIX 1 — session-state defaults set BEFORE any widget renders, so the
    # very first paint of the NFI tab has the right season / game type / pos.
    for key, val in {
        "framework":     "NFI",
        "nfi_season":    "Pooled (2022–2026)",
        "nfi_game_type": "Regular Season",
        "nfi_position":  "All",
        "game_type_nfi": "Regular Season",
    }.items():
        st.session_state.setdefault(key, val)
    # Per-tab widget defaults — also seeded before any widget runs.
    for key, val in {
        "nfi_teams": [], "nfi_name": "",
        "nfi_min_toi": NFI_TOI_DEFAULT["pooled"],
        "nfi_last_season": "Pooled (2022–2026)",
        "nfi_show_corsi_fenwick": False,
        "f_position": "All", "game_type_tnzi": "Regular Season",
        "f_season": REGULAR_SEASON_OPTIONS[0], "f_teams": [],
        "f_name": "", "f_min_gp": 0, "f_flag": "All",
        "tc_season": "Current Season (2025-26)",
    }.items():
        st.session_state.setdefault(key, val)

    # FIX 1 — load NFI pooled data unconditionally before framework selector.
    # The cache_data on _preload_nfi / load_nfi_player keeps this O(0) on
    # subsequent reruns.
    _nfi_preload = _preload_nfi()
    _prefetch_all()

    inject_css()
    render_header()

    # Small mobile-only filter hint
    st.markdown(
        '<p class="mobile-filter-hint">← Use the sidebar arrow to access filters</p>',
        unsafe_allow_html=True,
    )

    # FIX 11 — framework toggle buttons in main area (not sidebar)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        nfi_btn = st.button(
            "Net Front Impact (NFI)",
            use_container_width=True,
            type="primary" if st.session_state.get("framework") == "NFI" else "secondary",
            key="btn_nfi",
        )
    with col2:
        tnzi_btn = st.button(
            "Zone Impact (TNZI)",
            use_container_width=True,
            type="primary" if st.session_state.get("framework") == "TNZI" else "secondary",
            key="btn_tnzi",
        )
    with col3:
        team_btn = st.button(
            "Team Construction",
            use_container_width=True,
            type="primary" if st.session_state.get("framework") == "TEAM" else "secondary",
            key="btn_team",
        )
    if nfi_btn:
        st.session_state["framework"] = "NFI"
        st.rerun()
    if tnzi_btn:
        st.session_state["framework"] = "TNZI"
        st.rerun()
    if team_btn:
        st.session_state["framework"] = "TEAM"
        st.rerun()

    framework = st.session_state["framework"]
    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    if framework == "TNZI":
        render_tnzi_disclaimer()
        render_tnzi_sidebar()
        render_tnzi_table()
        render_tnzi_explainers()
    elif framework == "NFI":
        render_nfi_disclaimer()
        render_nfi_sidebar()
        render_nfi_table()
        render_nfi_explainers()
    elif framework == "TEAM":
        render_team_construction_disclaimer()
        render_team_construction_sidebar()
        render_team_construction()

    render_footer()


if __name__ == "__main__":
    main()
