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

# Goalie tab season selector — independent from the skater season selector
# because the goalie pipeline goes back to 2021-22 and uses its own pooled
# label (and the canonical pooled CSV uses a different denominator than the
# per-season file by design).
GOALIE_POOLED_LABEL = "Pooled (2021-22 → 2025-26)"
GOALIE_SEASON_OPTIONS = [
    GOALIE_POOLED_LABEL,
    "2025-26", "2024-25", "2023-24", "2022-23", "2021-22",
]
GOALIE_SEASON_TO_KEY = {
    "2025-26": 20252026,
    "2024-25": 20242025,
    "2023-24": 20232024,
    "2022-23": 20222023,
    "2021-22": 20212022,
}

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

    # FIX 8 — include ALL skaters (forwards + defensemen). Goalies are the
    # only thing we exclude; the player file uses 'position' values F / D / G
    # (some files also carry L / R / C which all map to forwards).
    skaters = players[players["position"] != "G"].dropna(
        subset=[x_metric, "team"]
    ).copy()
    if skaters.empty:
        return pd.DataFrame()
    fwd = skaters  # rename kept downstream for minimal diff

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

    # Goalie metric — pooled v2 (CNFI+MNFI, aggregated from per-season file)
    g = load_goalie_nfi()
    if g.empty:
        return pd.DataFrame()

    # Starter per team — use goalie_team_lookup.csv (in-repo, lightweight).
    # If unavailable, fall back to assigning each goalie to the team listed
    # in the goalie loader (latest team).
    lookup_fp = REPO_ROOT / "Goalies" / "Benchmarks Goalies" / "Data" / "goalie_team_lookup.csv"
    if lookup_fp.exists():
        lk = pd.read_csv(lookup_fp)
        # FIX 2 — Normalise legacy team codes (UTA == ARI for 2022-23/23-24).
        lk["goalie_team"] = lk["goalie_team"].replace({"ARI": "UTA"})

        def _starter_from(frame: pd.DataFrame) -> pd.DataFrame:
            return (
                frame.groupby(["goalie_team", "goalie_id"])["games_played"].sum()
                     .rename("games").reset_index()
                     .sort_values(["goalie_team", "games"], ascending=[True, False])
                     .drop_duplicates("goalie_team", keep="first")
            )

        if season_key is not None:
            lk_season = lk[lk["season"].astype(str) == season_key]
            starter_season = _starter_from(lk_season)
            # FIX 2 — Some historical seasons are missing from the lookup
            # (e.g. 2022-23 has 0 rows; 2023-24 lacks UTA). Backfill any
            # missing teams using the pooled lookup so the team panel
            # always returns a full 32-team frame.
            present = set(starter_season["goalie_team"].unique())
            missing = set(lk["goalie_team"].unique()) - present
            if missing:
                pool = lk[lk["goalie_team"].isin(missing)]
                starter = pd.concat(
                    [starter_season, _starter_from(pool)], ignore_index=True
                )
            else:
                starter = starter_season
        else:
            starter = _starter_from(lk)
        starter = starter.rename(columns={"goalie_team": "team"})
    else:
        starter = (
            g.dropna(subset=["team"])
             .sort_values("ES_TOI_min", ascending=False)
             .drop_duplicates("team", keep="first")[["team", "goalie_id"]]
        )
    # FIX 2 — guarantee one starter row per team after the season + fallback merge
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


# FIX 3 — display NaN guard. Every dataframe that goes into st.dataframe()
# / st.table() runs through this first so users never see bare 'nan' or
# 'None' in any cell. Numeric columns also get rounded to 2 decimals as a
# uniform default; column-specific formatters (e.g. percent / per-60 / TOI
# integer) still take precedence at the styler level.
def _prep_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0.0)
            # 2dp default rounding — column-specific formatters override
            try:
                if pd.api.types.is_float_dtype(out[col]):
                    out[col] = out[col].round(2)
            except (TypeError, ValueError):
                pass
        else:
            out[col] = out[col].fillna("—").astype(str).replace(
                {"nan": "—", "None": "—", "": "—"}
            )
    return out


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
        # FIX 1 — every playoff selection (including 2025-26) shows the
        # canonical context-only message and nothing else.
        st.info(
            "Playoff data is available but samples are too small for "
            "reliable individual rankings (median 7–28 games per player). "
            "Use the Team Construction tab to evaluate playoff performance "
            "at the team level."
        )
        return
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

    # FIX 6 — prepend an integer Rank column (no heat map). Reset index so
    # the rank lines up with the post-sort row order.
    display_df = display_df.reset_index(drop=True)
    display_df.insert(0, "Rank", np.arange(1, len(display_df) + 1))

    is_pooled_view = season == DEFAULT_SEASON
    # FIX 3 — fillna numeric→0.0, object→"—" + 2dp default before styling.
    display_df = _prep_for_display(display_df)
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
        Sort by <strong>RelNFI_A%</strong> for pure suppressors.<br><br>
        <em>Zone adjustment not applied to individual rankings — RelNFI% is
        zone-adjustment-invariant by construction.</em>
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

    # Goalie-specific season selector — only shown when Goalies position is
    # active. Independent from the skater nfi_season selector because the
    # goalie pipeline goes back to 2021-22 and has its own pooled label.
    if st.session_state.get("nfi_position", "All") == "Goalies":
        st.session_state.setdefault("goalie_season_view", GOALIE_POOLED_LABEL)
        st.sidebar.selectbox(
            "Goalie Season",
            GOALIE_SEASON_OPTIONS,
            key="goalie_season_view",
            help=("Pooled = goalie_nfi_gsax_pooled_v2.csv (min 300 dangerous-"
                  "zone shots over 2021-22 → 2025-26). Per-season = "
                  "goalie_nfi_gsax_by_season.csv (min 100 per season)."),
        )
        # Min Shots Faced — applies to both pooled and per-season views.
        # Defaults to 500 to filter out the small-sample tail (Greaves 102,
        # Aaron Dell 107, etc.) without hiding mid-volume backups.
        st.session_state.setdefault("goalie_min_shots", 500)
        st.sidebar.slider(
            "Min Shots Faced",
            min_value=100, max_value=3000, step=100,
            key="goalie_min_shots",
            help=("Filters the displayed goalies by total dangerous-zone "
                  "shots-faced. Higher = fewer small-sample anomalies."),
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
    # FIX 5 — drop NFI%_ZA from the player table; raw NFI% only.
    # RelNFI% is zone-adjustment-invariant so the ZA flavour adds no signal
    # for individual rankings.
    cols = ["Player", "Pos", "Team", "TOI",
            "RelNFI%", "RelNFI_F%", "RelNFI_A%",
            "NFI%"]
    if show_cf_ff:
        # CF/FF comparison still useful — append after NFI%
        cols += ["CF%_ZA", "FF%_ZA"]
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
        # FIX 1 — every playoff selection (including 2025-26) shows the
        # canonical context-only message and nothing else.
        st.info(
            "Playoff data is available but samples are too small for "
            "reliable individual rankings (median 7–28 games per player). "
            "Use the Team Construction tab to evaluate playoff performance "
            "at the team level."
        )
        return
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
    # FIX 3 — fillna before NFI styler too
    display_df = _prep_for_display(display_df)
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
    """Goalie tab — playoff short-circuit, then either pooled or per-season.

    - Playoffs (any year) → canonical playoff context message and nothing
      else (matches the player tabs).
    - Regular Season → toggles on the new sidebar selector
      `goalie_season_view` between Pooled and a single season:
        * Pooled  → NFI/output/goalie_nfi_gsax_pooled_v2.csv (min 300
                    shots-faced, CNFI+MNFI, aggregated from the per-season
                    file — single source of truth)
        * Season  → NFI/output/goalie_nfi_gsax_by_season.csv filtered to
                    the chosen season (min 100 shots-faced, CNFI+MNFI,
                    pooled-share TOI proxy)

    Both branches render the same column set (Rank, Goalie, Team, Games,
    Faced, GSAx, GSAx/60), sort descending by GSAx/60, and run through
    _prep_for_display() before the styler.
    """
    # Reading state here makes Streamlit re-run the function whenever any
    # toggle changes, so the table refreshes on every switch.
    if game_type is None:
        game_type = st.session_state.get("game_type_nfi", "Regular Season")
    if season is None:
        season = st.session_state.get("nfi_season", DEFAULT_SEASON)

    if game_type == "Playoffs":
        st.info(
            "Playoff data is available but samples are too small for "
            "reliable individual rankings (median 7–28 games per player). "
            "Use the Team Construction tab to evaluate playoff performance "
            "at the team level."
        )
        return

    goalie_view = st.session_state.get("goalie_season_view", GOALIE_POOLED_LABEL)
    is_pooled = goalie_view == GOALIE_POOLED_LABEL

    st.markdown(
        f"<p style='color:{PALETTE['text']}; font-size:0.95rem; line-height:1.5;'>"
        "Goalie NFI-GSAx measures goals saved above expected from dangerous zones. "
        "Validated against MoneyPuck GSAx at r=0.858.</p>",
        unsafe_allow_html=True,
    )

    # ---------------- Build the display frame for either branch ----------------
    if is_pooled:
        df = load_goalie_nfi()
        if df.empty:
            st.error("Goalie file not found at "
                     "`NFI/output/goalie_nfi_gsax_pooled_v2.csv`.")
            return
        # The v2 file already has games + total_faced + GSAx + GSAx_per60
        # natively — just rename total_faced for the shared display schema.
        df = df.rename(columns={"total_faced": "faced"})
        caption = (
            "Pooled across 2021-22 → 2025-26. "
            "Minimum 300 dangerous-zone shots-faced (CNFI+MNFI) to qualify."
        )
    else:
        season_key = GOALIE_SEASON_TO_KEY.get(goalie_view)
        df_all = load_goalie_nfi_by_season()
        if df_all.empty or season_key is None:
            st.error(
                "Per-season goalie file not found at "
                "`NFI/output/goalie_nfi_gsax_by_season.csv`."
            )
            return
        df = df_all[df_all["season"] == season_key].copy()
        # Already named: goalie_name, team, games, total_faced, GSAx, GSAx_per60
        df = df.rename(columns={"total_faced": "faced"})
        caption = (
            "Single season data. Minimum 100 dangerous-zone shots-faced to "
            "qualify. Per-season TOI estimated from pooled allocation."
        )

    # ---------------- Sidebar filters (team / name / min shots) ----------------
    teams = st.session_state.get("nfi_teams", [])
    if teams and "team" in df.columns:
        df = df[df["team"].isin(teams)]
    name_q = (st.session_state.get("nfi_name", "") or "").strip().lower()
    if name_q and "goalie_name" in df.columns:
        df = df[df["goalie_name"].fillna("").str.lower().str.contains(name_q, na=False)]

    # Min Shots Faced — filters the small-sample tail in both views
    min_shots = int(st.session_state.get("goalie_min_shots", 500))
    if "faced" in df.columns:
        df = df[df["faced"].fillna(0) >= min_shots]

    if df.empty:
        st.info("No goalie data available for this season.")
        return

    # ---------------- Sort + rank + project to display columns ----------------
    df = df.sort_values("GSAx_per60", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))

    disp = df[[
        "Rank", "goalie_name", "team", "games", "faced", "GSAx", "GSAx_per60",
    ]].rename(columns={
        "goalie_name": "Goalie",
        "team":        "Team",
        "games":       "Games",
        "faced":       "Faced",
        "GSAx_per60":  "GSAx/60",
    })

    # ---------------- Format + render ----------------
    fmt = {
        "Games":   lambda x: "—" if pd.isna(x) else f"{int(x):,}",
        "Faced":   lambda x: "—" if pd.isna(x) else f"{int(x):,}",
        "GSAx":    lambda x: "—" if pd.isna(x) else f"{x:+.2f}",
        "GSAx/60": lambda x: "—" if pd.isna(x) else f"{x:+.3f}",
    }
    disp = _prep_for_display(disp)
    styler = disp.style.format(fmt, na_rep="—")
    st.dataframe(styler, width="stretch", hide_index=True)
    st.caption(caption)
    st.caption(
        f"Showing {len(disp):,} goalies — "
        f"{'pooled view' if is_pooled else goalie_view} · "
        "sorted by GSAx/60 descending"
    )


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
        skater group NFI%_ZA (forwards + defensemen, TOI-weighted) against its starter goalie's NFI-GSAx
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
        # FIX 4 — five season options: current + three historical years + pooled
        ["Current Season (2025-26)", "2024-25", "2023-24", "2022-23",
         "Pooled (2022–2026)"],
        key="tc_season",
    )
    st.sidebar.caption(
        "NFI%_ZA is TOI-weighted across all skaters (forwards and defensemen). "
        "Starter goalie is the one with the most games played for the team."
    )


def render_team_construction() -> None:
    import matplotlib.pyplot as plt

    season_choice = st.session_state.get("tc_season", "Current Season (2025-26)")
    df = load_team_construction(season_choice)
    if df.empty:
        st.error(
            "Team construction data unavailable — required: "
            "`player_fully_adjusted.csv` and "
            "`goalie_nfi_gsax_pooled_v2.csv`."
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

    ax.set_xlabel("Team NFI%_ZA (Forwards + D, TOI-weighted)", color=NAVY)
    ax.set_ylabel("Starter Goalie NFI-GSAx per 60", color=NAVY)
    ax.set_title("Team Construction — Forwards × Goalie", color=NAVY,
                 fontsize=14, weight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=NAVY)
    if not edm.empty:
        ax.legend(loc="lower right", frameon=False)

    st.pyplot(fig, clear_figure=True, use_container_width=True)

    # FIX 7 — caveat caption: scatter shows construction quality, not standings
    st.caption(
        "Note: This scatter shows two-pillar construction quality — not "
        "current standings. A team can rank highly here while underperforming "
        "in standings due to goaltending variance, special teams, or PDO. "
        "Columbus (CBJ) ranks highly due to strong goalie GSAx combined with "
        "above-average NFI%_ZA."
    )

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
    # FIX 3 — fillna before TC rank table
    out = _prep_for_display(out)
    st.dataframe(
        out.style.format({
            x_label: "{:+.3f}",
            "Goalie GSAx /60": "{:+.3f}",
        }),
        width="stretch", hide_index=True,
    )


# ---------------------------------------------------------------------------
# Methodology tab + placeholders
# ---------------------------------------------------------------------------
GITHUB_METHODOLOGY_URL = (
    "https://github.com/HockeyROI/NHL-analytics/blob/main/docs/METHODOLOGY.md"
)
TAB_LABELS = ["Players", "Teams", "Goalies", "Referees", "Methodology"]


def render_coming_soon(title: str) -> None:
    st.markdown(
        f"""
        <div style="border:1px dashed {PALETTE['border']}; background:{PALETTE['panel']};
             border-radius:8px; padding:2.75rem 1.5rem; text-align:center; margin-top:1rem;">
          <div style="font-family:'Bebas Neue',Impact,sans-serif; font-size:1.7rem;
               color:{PALETTE['text']}; letter-spacing:1px;">{title}</div>
          <div style="color:{PALETTE['text_secondary']}; margin-top:0.4rem; font-size:0.95rem;">
            Coming in this build.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
            "<b>RelNFI%</b> is the two-way version (generation + suppression). Zone-adjusted with "
            "Tulsky's 3.5pp factor.",
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
    "Pooled (2022–2026)": "pooled",
    "2025-26": "20252026",
    "2024-25": "20242025",
    "2023-24": "20232024",
    "2022-23": "20222023",
}


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
    is_pooled = SEASON_KEY.get(season_label, "pooled") == "pooled"

    if is_pooled:
        base = _aggregate_nfi_pooled(nfi)
        if not qg.empty:
            base = base.merge(_qg_pooled(qg), on="player_id", how="left")
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
        st.info(
            "Individual player samples in playoffs are too small for meaningful "
            "analysis (median 7–28 games per player). Regular-season player "
            "metrics are available on this tab."
        )
        return

    frame, is_pooled = _build_players_frame(season_label)
    if frame.empty:
        st.error("Player data not found "
                 "(`NFI/output/fully_adjusted/player_fully_adjusted.csv`).")
        return

    c1, c2, c3 = st.columns([1.1, 1.7, 1.0])
    with c1:
        pos = st.radio("Position", ["All", "F", "D"], horizontal=True, key="players_pos")
    with c2:
        toi_key = "players_toi_pooled" if is_pooled else "players_toi_season"
        default_toi = 2000 if is_pooled else 200
        min_toi = st.slider("Min ES TOI (min)", 0, 7500, default_toi, 50, key=toi_key)
    with c3:
        view = st.radio("View", ["Compact", "Full"], horizontal=True, key="players_view")

    df = frame.copy()
    if pos in ("F", "D"):
        df = df[df["position"] == pos]
    else:
        df = df[df["position"].isin(["F", "D"])]
    df = df[df["toi_min"].fillna(0) >= min_toi]
    if df.empty:
        st.markdown(
            f"<p style='color:{PALETTE['text']};'>No players match the current filters. "
            "Widen Position or Min TOI, or change the Season in the sidebar.</p>",
            unsafe_allow_html=True,
        )
        return

    df = df.sort_values("RelNFI_pct", ascending=False, na_position="last").reset_index(drop=True)
    df = df.rename(columns={
        "player_name": "Player", "position": "Pos", "team": "Team", "toi_min": "TOI",
        "NFI_pct": "NFI%", "RelNFI_pct": "RelNFI%",
        "RelNFI_F_pct": "RelNFI_F%", "RelNFI_A_pct": "RelNFI_A%",
        "xG_QG_pct": "xG_QG%", "NFI_QG_pct": "NFI_QG%", "qualifying_GP": "Qual GP",
    })

    full_cols = ["Player", "Pos", "Team", "GP", "TOI", "NFI%", "RelNFI%", "RelNFI_F%",
                 "RelNFI_A%", "NZI", "DZI", "OZI", "xG_QG%", "NFI_QG%", "Qual GP"]
    compact_cols = ["Player", "Pos", "Team", "TOI", "NFI%", "RelNFI%", "NFI_QG%", "xG_QG%"]
    cols = full_cols if view == "Full" else compact_cols
    if not is_pooled:  # zone metrics are pooled-only — hide for single-season views
        cols = [c for c in cols if c not in ("NZI", "DZI", "OZI")]
    cols = [c for c in cols if c in df.columns]
    disp = df[cols].copy()

    fmt = {}
    for c in ("NFI%", "xG_QG%", "NFI_QG%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("RelNFI%", "RelNFI_F%", "RelNFI_A%"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:+.2f}"
    for c in ("NZI", "DZI", "OZI"):
        if c in disp.columns:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    if "TOI" in disp.columns:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    for c in ("GP", "Qual GP"):
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
        st.info("Team playoff metrics aren't available yet — the team pipeline "
                "currently covers regular season only.")
        return

    tl = load_team_level()
    if tl.empty:
        st.error("Team data not found (`NFI/output/team_level_all_metrics.csv`).")
        return
    qg = load_team_qg()
    is_pooled = SEASON_KEY.get(season_label, "pooled") == "pooled"

    def _wmean(g: pd.DataFrame, col: str) -> float:
        w = g["total_team_TOI_min"].astype(float)
        v = pd.to_numeric(g[col], errors="coerce")
        m = v.notna() & (w > 0)
        return float(np.average(v[m], weights=w[m])) if m.any() else np.nan

    if is_pooled:
        sub = tl[tl["season"].isin(POOLED_SEASONS)]
        agg = sub.groupby("team").agg(
            CNFI_FF=("CNFI_FF", "sum"), MNFI_FF=("MNFI_FF", "sum"),
            CNFI_FA=("CNFI_FA", "sum"), MNFI_FA=("MNFI_FA", "sum"),
            GP=("gp", "sum"),
        ).reset_index()
        agg["NFI%"] = _team_nfi_share(agg)
        team = agg[["team", "GP", "NFI%"]]
        if not qg.empty:
            q = qg[qg["season"].isin(POOLED_SEASONS)]
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

    # Attack / Suppress — 2025-26 snapshot only
    if season_label == "2025-26":
        a = load_team_attack_suppress().rename(
            columns={"attack_per_game": "Attack rate", "suppress_per_game": "Suppress rate"})
        team = team.merge(a, on="team", how="left")
    else:
        team["Attack rate"] = np.nan
        team["Suppress rate"] = np.nan

    for c in ("NZI", "DZI", "OZI"):   # deliberate team-level placeholders
        team[c] = np.nan
    for c in ("TOI", "xG_QG%", "NFI_QG%"):
        if c not in team.columns:
            team[c] = np.nan

    team = team.rename(columns={"team": "Team"})
    team = team.sort_values("NFI%", ascending=False, na_position="last").reset_index(drop=True)
    cols = ["Team", "GP", "TOI", "NFI%", "Attack rate", "Suppress rate",
            "NZI", "DZI", "OZI", "xG_QG%", "NFI_QG%"]
    disp = team[[c for c in cols if c in team.columns]].copy()

    fmt = {}
    for c in ("NFI%", "xG_QG%", "NFI_QG%"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    for c in ("Attack rate", "Suppress rate"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    for c in ("NZI", "DZI", "OZI"):
        if c in disp:
            fmt[c] = lambda x: "—" if pd.isna(x) else f"{x:.1f}"
    if "TOI" in disp:
        fmt["TOI"] = lambda x: "—" if pd.isna(x) else f"{x:,.0f}"
    if "GP" in disp:
        fmt["GP"] = lambda x: "—" if pd.isna(x) else f"{int(x):,}"

    st.dataframe(disp.style.format(fmt, na_rep="—"), width="stretch", hide_index=True)

    cap = f"{len(disp)} teams · {season_label} · sorted by NFI% (CNFI+MNFI share) descending"
    if season_label != "2025-26":
        cap += (" · Attack and Suppress rates are currently only computed for 2025-26. "
                "Per-season history requires a pipeline run not yet performed.")
    cap += " · NZI/DZI/OZI are team-level placeholders (not yet computed)."
    st.caption(cap)


# ---------------------------------------------------------------------------
# Global sidebar (Season + Game type — apply to every tab)
# ---------------------------------------------------------------------------
def render_global_sidebar() -> tuple[str, str]:
    st.sidebar.markdown(
        f"<div style='font-family:\"Bebas Neue\",Impact,sans-serif; font-size:1.4rem; "
        f"color:{PALETTE['text']}; letter-spacing:1px; margin-bottom:0.3rem;'>Filters</div>",
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("g_season", "Pooled (2022–2026)")
    st.session_state.setdefault("g_game_type", "Regular Season")
    season = st.sidebar.selectbox("Season", list(SEASON_KEY.keys()), key="g_season")
    game_type = st.sidebar.radio("Game type", ["Regular Season", "Playoffs"],
                                 key="g_game_type")
    st.sidebar.caption(
        "Season and game type apply across all tabs. Zone Impact (NZI/DZI/OZI) is "
        "pooled-only and appears in the Pooled season view."
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
        render_coming_soon("Goalies — NFI-GSAx + QNFS% + QS-GSAx")
    with refs_tab:
        render_coming_soon("Referees — penalty tendencies")
    with meth_tab:
        render_methodology()

    render_footer()


if __name__ == "__main__":
    main()
