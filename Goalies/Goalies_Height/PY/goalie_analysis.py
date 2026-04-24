"""
NHL Goalie Frame Size vs Save Performance Analysis
Scrapes Hockey Reference for height/weight data and downloads MoneyPuck GSAX data,
then analyzes the relationship between goalie frame size and save performance.
"""

import time
import re
import unicodedata
import io
import os

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

import argparse

OUTPUT_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
DATA_DIR   = os.path.join(OUTPUT_DIR, "Data")

# Intermediate cache files — reused across runs to skip re-scraping
HR_CACHE  = os.path.join(DATA_DIR, "cache_hockey_reference.csv")
MP_CACHE  = os.path.join(DATA_DIR, "cache_moneypuck.csv")

# Season start years (2010-11 through 2023-24)
SEASON_START_YEARS = list(range(2010, 2024))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.hockey-reference.com/",
}

# Defaults — can be overridden via CLI args
MIN_GAMES_STARTED     = 30
MIN_AGE               = 22
MAX_AGE               = 38
MIN_QUALIFYING_SEASONS = 4

SLEEP_BETWEEN_REQUESTS = 4  # seconds — be polite to Hockey Reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """
    Lowercase, strip accents, remove suffixes (Jr., Sr., II, III), strip
    punctuation, and collapse whitespace so names can be merged across sources.
    """
    # Decompose unicode characters and drop combining marks (accents)
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")

    # Remove common suffixes
    ascii_name = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", ascii_name, flags=re.IGNORECASE)

    # Keep only letters and spaces
    ascii_name = re.sub(r"[^a-zA-Z ]", " ", ascii_name)

    return " ".join(ascii_name.lower().split())


def height_to_inches(height_str: str) -> float | None:
    """Convert '6-2' or '6\'2\"' style strings to total inches."""
    if not isinstance(height_str, str):
        return None
    m = re.search(r"(\d+)['\-](\d+)", height_str)
    if m:
        feet = int(m.group(1))
        inches = int(m.group(2))
        return feet * 12 + inches
    return None


def extract_table_from_comment(html: str, table_id: str) -> BeautifulSoup | None:
    """
    Hockey Reference hides some tables inside HTML comments.
    Find the commented-out table with the given id and return its BeautifulSoup.
    """
    pattern = re.compile(r"<!--(.*?)-->", re.DOTALL)
    for comment in pattern.findall(html):
        if f'id="{table_id}"' in comment or f"id='{table_id}'" in comment:
            return BeautifulSoup(comment, "lxml")
    return None


# ---------------------------------------------------------------------------
# Step 1 — Scrape Hockey Reference
# ---------------------------------------------------------------------------

def scrape_hockey_reference_season(season_end_year: int) -> pd.DataFrame:
    """
    Scrape the goalie stats page for a single NHL season from Hockey Reference.
    season_end_year: e.g. 2023 for the 2022-23 season.
    Returns a DataFrame with columns: name, name_raw, season_end_year, age,
                                      games_started, player_href.
    """
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_end_year}_goalies.html"
    print(f"  Fetching: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: Failed to fetch {url}: {e}")
        return pd.DataFrame()

    html = resp.text
    soup = BeautifulSoup(html, "lxml")

    # Table ID is "goalie_stats" (live, not comment-hidden)
    table = soup.find("table", {"id": "goalie_stats"})

    if table is None:
        # Fallback: try comment-hidden
        comment_soup = extract_table_from_comment(html, "goalie_stats")
        if comment_soup:
            table = comment_soup.find("table", {"id": "goalie_stats"})

    if table is None:
        print(f"  WARNING: Could not find goalie_stats table for {season_end_year}")
        return pd.DataFrame()

    tbody = table.find("tbody")
    if tbody is None:
        return pd.DataFrame()

    rows = []
    for tr in tbody.find_all("tr"):
        cls = tr.get("class", [])
        if "thead" in cls or "partial_table" in cls:
            continue

        td_name = tr.find("td", {"data-stat": "name_display"})
        if td_name is None:
            continue

        name_raw = td_name.get_text(strip=True)
        if not name_raw or name_raw in ("Player", ""):
            continue

        def get_text(stat):
            td = tr.find(["td", "th"], {"data-stat": stat})
            return td.get_text(strip=True) if td else ""

        age_str = get_text("age")
        gs_str = get_text("goalie_starts")

        a_tag = td_name.find("a")
        player_href = a_tag["href"] if a_tag and a_tag.get("href") else None

        try:
            age = int(age_str) if age_str else None
        except ValueError:
            age = None

        try:
            games_started = int(gs_str) if gs_str else None
        except ValueError:
            games_started = None

        rows.append({
            "name_raw": name_raw,
            "name": normalize_name(name_raw),
            "season_end_year": season_end_year,
            "age": age,
            "games_started_hr": games_started,
            "player_href": player_href,
        })

    return pd.DataFrame(rows)


def fetch_player_bio(player_href: str, session: requests.Session) -> tuple[float | None, float | None]:
    """
    Fetch height (inches) and weight (lbs) from a player's Hockey Reference bio page.
    Returns (height_inches, weight_lbs) or (None, None) on failure.
    """
    if not player_href:
        return None, None

    url = f"https://www.hockey-reference.com{player_href}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    WARNING: Bio fetch failed for {url}: {e}")
        return None, None

    soup = BeautifulSoup(resp.text, "lxml")

    # The bio block is usually in a <div id="info"> section with <p> tags
    # containing text like "6-2, 185lb"
    height_in = None
    weight_lbs = None

    info_div = soup.find("div", {"id": "info"})
    if info_div is None:
        return None, None

    full_text = info_div.get_text(" ")

    # Height pattern: digits-digits (e.g. 6-2)
    h_match = re.search(r"\b(\d)-(\d{1,2})\b", full_text)
    if h_match:
        height_in = int(h_match.group(1)) * 12 + int(h_match.group(2))

    # Weight pattern: digits followed by lb
    w_match = re.search(r"(\d{2,3})\s*lb", full_text, re.IGNORECASE)
    if w_match:
        weight_lbs = float(w_match.group(1))

    return height_in, weight_lbs


def scrape_all_hockey_reference() -> pd.DataFrame:
    """
    Scrape Hockey Reference for all seasons and build a DataFrame with
    columns: name, season_end_year, age, height_in, weight_lbs.
    """
    print("\n=== Scraping Hockey Reference ===")
    all_season_frames = []

    for start_year in SEASON_START_YEARS:
        end_year = start_year + 1
        df_season = scrape_hockey_reference_season(end_year)
        if not df_season.empty:
            all_season_frames.append(df_season)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not all_season_frames:
        print("ERROR: No Hockey Reference data collected.")
        return pd.DataFrame()

    df_all = pd.concat(all_season_frames, ignore_index=True)

    # Deduplicate player hrefs to avoid redundant bio fetches
    print(f"\nFetched {len(df_all)} goalie-season rows. Now fetching player bios...")
    unique_hrefs = df_all[df_all["player_href"].notna()]["player_href"].unique()
    print(f"  Unique players to bio-fetch: {len(unique_hrefs)}")

    bio_map: dict[str, tuple[float | None, float | None]] = {}
    session = requests.Session()
    session.headers.update(HEADERS)

    for i, href in enumerate(unique_hrefs, 1):
        if href in bio_map:
            continue
        print(f"  [{i}/{len(unique_hrefs)}] Bio: {href}")
        ht, wt = fetch_player_bio(href, session)
        bio_map[href] = (ht, wt)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    df_all["height_in"] = df_all["player_href"].map(lambda h: bio_map.get(h, (None, None))[0])
    df_all["weight_lbs"] = df_all["player_href"].map(lambda h: bio_map.get(h, (None, None))[1])

    return df_all[["name", "name_raw", "season_end_year", "age", "height_in", "weight_lbs"]]


# ---------------------------------------------------------------------------
# Step 2 — Download MoneyPuck GSAX data
# ---------------------------------------------------------------------------

def download_moneypuck_season(year: int, session: requests.Session) -> pd.DataFrame:
    """
    Download MoneyPuck goalie CSV for the season starting in `year`.
    URL pattern: seasonSummary/{year+1}/regular/goalies.csv
    Returns DataFrame with columns: name, season_end_year, gsax, games_started.

    GSAx is computed as xGoals - goals (for situation='all'),
    meaning positive = goalie saved more goals than expected.
    """
    season_end_year = year + 1
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season_end_year}/regular/goalies.csv"
    print(f"  Downloading: {url}")

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: Failed to download {url}: {e}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f"  WARNING: Failed to parse CSV for {year}: {e}")
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]

    # Filter to situation='all'
    if "situation" in df.columns:
        df = df[df["situation"].astype(str).str.lower() == "all"]

    if df.empty:
        print(f"  WARNING: No 'all' situation rows for {year}")
        return pd.DataFrame()

    # Compute GSAx = xGoals - goals (expected goals against minus actual)
    if "xGoals" not in df.columns or "goals" not in df.columns:
        print(f"  WARNING: Missing xGoals/goals columns for {year}. Columns: {list(df.columns)[:20]}")
        return pd.DataFrame()

    df = df.copy()
    df["gsax"] = pd.to_numeric(df["xGoals"], errors="coerce") - pd.to_numeric(df["goals"], errors="coerce")

    # Name column
    name_col = next((c for c in ["name", "Name", "playerName"] if c in df.columns), None)
    if name_col is None:
        name_col = next((c for c in df.columns if "name" in c.lower()), None)
    if name_col is None:
        print(f"  WARNING: No name column for {year}")
        return pd.DataFrame()

    # Games played (proxy for games started — goalies rarely enter mid-game)
    gp_col = next((c for c in ["games_played", "gamesPlayed", "GP"] if c in df.columns), None)

    result_rows = []
    for _, row in df.iterrows():
        raw_name = str(row[name_col])
        gsax_val = row["gsax"]
        gp_val = pd.to_numeric(row[gp_col], errors="coerce") if gp_col else np.nan

        result_rows.append({
            "name": normalize_name(raw_name),
            "name_raw_mp": raw_name,
            "season_end_year": season_end_year,
            "gsax": gsax_val,
            "games_started": gp_val,
        })

    return pd.DataFrame(result_rows)


def download_all_moneypuck() -> pd.DataFrame:
    """Download MoneyPuck data for all seasons and return combined DataFrame."""
    print("\n=== Downloading MoneyPuck data ===")
    session = requests.Session()
    session.headers.update({
        "User-Agent": HEADERS["User-Agent"],
        "Referer": "https://moneypuck.com/",
    })

    frames = []
    for year in SEASON_START_YEARS:
        df = download_moneypuck_season(year, session)
        if not df.empty:
            frames.append(df)
        time.sleep(1)

    if not frames:
        print("ERROR: No MoneyPuck data collected.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 3 — Merge and Filter
# ---------------------------------------------------------------------------

def merge_and_filter(df_hr: pd.DataFrame, df_mp: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Hockey Reference and MoneyPuck data on (normalized name, season_end_year),
    apply filtering criteria, and compute per-goalie averages.
    """
    print("\n=== Merging datasets ===")
    print(f"Hockey Reference rows: {len(df_hr)}")
    print(f"MoneyPuck rows:        {len(df_mp)}")

    merged = pd.merge(
        df_hr,
        df_mp[["name", "season_end_year", "gsax", "games_started", "name_raw_mp"]],
        on=["name", "season_end_year"],
        how="inner",
    )
    print(f"Merged rows (before filters): {len(merged)}")

    # Apply per-season filters — use HR games_started (goalie_starts) for GS filter
    gs_col = "games_started_hr" if "games_started_hr" in merged.columns else "games_started"
    merged = merged[merged[gs_col].fillna(0) >= MIN_GAMES_STARTED]
    if "age" in merged.columns:
        merged = merged[merged["age"].between(MIN_AGE, MAX_AGE, inclusive="both")]

    # Drop rows without valid height/weight/gsax
    merged = merged.dropna(subset=["height_in", "weight_lbs", "gsax"])
    merged = merged[merged["height_in"] > 0]
    merged = merged[merged["weight_lbs"] > 0]

    print(f"Merged rows (after per-season filters): {len(merged)}")

    # Count qualifying seasons per goalie
    season_counts = merged.groupby("name")["season_end_year"].nunique()
    qualifying = season_counts[season_counts >= MIN_QUALIFYING_SEASONS].index
    merged = merged[merged["name"].isin(qualifying)]
    print(f"Goalies with >= {MIN_QUALIFYING_SEASONS} qualifying seasons: {len(qualifying)}")

    return merged


def compute_per_goalie(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Aggregate filtered data to one row per goalie."""
    agg = (
        df_filtered
        .groupby("name")
        .agg(
            display_name=("name_raw", "first"),
            height_in=("height_in", "first"),
            weight_lbs=("weight_lbs", "first"),
            avg_gsax=("gsax", "mean"),
            qualifying_seasons=("season_end_year", "nunique"),
            total_gsax=("gsax", "sum"),
        )
        .reset_index()
    )

    agg["weight_height_ratio"] = agg["weight_lbs"] / agg["height_in"]

    return agg.sort_values("avg_gsax", ascending=False)


# ---------------------------------------------------------------------------
# Step 4 — Output
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"\nSaved CSV: {path}")


def save_plot(df: pd.DataFrame, path: str) -> None:
    """Create scatter plot of weight/height ratio vs average GSAX."""
    x = df["weight_height_ratio"].values
    y = df["avg_gsax"].values
    names = df["display_name"].values

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(x, y, color="steelblue", alpha=0.75, s=70, zorder=3)

    # Label each point
    for xi, yi, name in zip(x, y, names):
        ax.annotate(
            name,
            (xi, yi),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
            alpha=0.85,
        )

    # Trend line
    ax.plot(x_line, y_line, color="crimson", linewidth=1.8, label=f"Trend (r={r_value:.3f}, p={p_value:.3f})")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Weight / Height Ratio  (lbs / inch)", fontsize=12)
    ax.set_ylabel("Average GSAx  (per qualifying season)", fontsize=12)
    ax.set_title("NHL Goalie Frame Size vs GSAX (2010-present)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global MIN_GAMES_STARTED, MIN_QUALIFYING_SEASONS

    parser = argparse.ArgumentParser(description="NHL Goalie Frame vs GSAx Analysis")
    parser.add_argument("--min-starts",   type=int, default=MIN_GAMES_STARTED,
                        help="Minimum games started per qualifying season")
    parser.add_argument("--min-seasons",  type=int, default=MIN_QUALIFYING_SEASONS,
                        help="Minimum number of qualifying seasons")
    parser.add_argument("--csv-out",      type=str,
                        default=os.path.join(OUTPUT_DIR, "goalie_frame_gsax.csv"),
                        help="Output CSV path")
    parser.add_argument("--png-out",      type=str,
                        default=os.path.join(OUTPUT_DIR, "goalie_frame_vs_gsax.png"),
                        help="Output scatter plot path")
    parser.add_argument("--no-cache",     action="store_true",
                        help="Force re-scrape even if cache exists")
    args = parser.parse_args()

    MIN_GAMES_STARTED      = args.min_starts
    MIN_QUALIFYING_SEASONS = args.min_seasons
    csv_out = args.csv_out
    png_out = args.png_out

    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Hockey Reference (cached) ---
    if not args.no_cache and os.path.exists(HR_CACHE):
        print(f"\n=== Loading Hockey Reference from cache: {HR_CACHE} ===")
        df_hr = pd.read_csv(HR_CACHE)
    else:
        df_hr = scrape_all_hockey_reference()
        if df_hr.empty:
            print("FATAL: No Hockey Reference data. Exiting.")
            return
        df_hr.to_csv(HR_CACHE, index=False)
        print(f"Cached HR data → {HR_CACHE}")

    # --- MoneyPuck (cached) ---
    if not args.no_cache and os.path.exists(MP_CACHE):
        print(f"\n=== Loading MoneyPuck from cache: {MP_CACHE} ===")
        df_mp = pd.read_csv(MP_CACHE)
    else:
        df_mp = download_all_moneypuck()
        if df_mp.empty:
            print("FATAL: No MoneyPuck data. Exiting.")
            return
        df_mp.to_csv(MP_CACHE, index=False)
        print(f"Cached MP data → {MP_CACHE}")

    # --- Merge & Filter ---
    df_merged = merge_and_filter(df_hr, df_mp)
    if df_merged.empty:
        print("FATAL: No data after merging and filtering.")
        return

    # --- Per-goalie aggregation ---
    df_goalies = compute_per_goalie(df_merged)

    print(f"\nFinal goalie count: {len(df_goalies)}")
    print("\nTop 10 by avg GSAX:")
    print(df_goalies[["display_name", "height_in", "weight_lbs", "weight_height_ratio", "avg_gsax", "qualifying_seasons"]].head(10).to_string(index=False))

    # --- Save outputs ---
    save_csv(df_goalies, csv_out)

    if len(df_goalies) >= 2:
        save_plot(df_goalies, png_out)
    else:
        print("Not enough data points to generate a plot (need >= 2 goalies).")

    print("\nDone.")


if __name__ == "__main__":
    main()
