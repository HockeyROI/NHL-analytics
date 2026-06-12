"""
Goalie frame-size vs save performance, restricted to 2022-23, 2023-24, 2024-25.

Two parallel pipelines, same three correlations (height, weight, weight/height
ratio) and the same summary-table format as the original frame analysis:

  MoneyPuck:  reload cache_moneypuck.csv (+ live 2024-25 download since the cache
              stops at 2023-24), filter to the 3 target seasons, apply min 3
              qualifying seasons / 25 starts per season / ages 22-38.
              x = avg GSAx per qualifying season.
              -> goalie_frame_gsax_2023now.csv

  NFI:        reload goalie_nfi_gsax_by_season.csv, filter to the 3 target
              seasons, require min 3 qualifying seasons, merge height/weight
              from the existing HR cache (live-scrape fallback for misses).
              x = avg GSAx_per60 per qualifying season.
              -> goalie_nfi_gsax_size_2023now.csv

Height/weight/age are resolved with clean (UTF-8) names: the existing HR cache
has accent-corrupted names ("markstram"), so we match what we can to the cache
and live-scrape the rest from Hockey Reference.
"""

import io
import os
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy import stats

from goalie_nfi_size_analysis import normalize_name, fetch_player_bio

OUT_DIR    = "/Users/ashgarg/Documents/HockeyROI/Goalies/sizes/output"
MP_CACHE   = "/Users/ashgarg/Documents/HockeyROI/Goalies/Benchmarks Goalies/moneypuck/cache_moneypuck.csv"
HR_CACHE   = "/Users/ashgarg/Documents/HockeyROI/Goalies/Goalies_Height/Data/cache_hockey_reference.csv"
NFI_SEASON = "/Users/ashgarg/Documents/HockeyROI/NFI/Output/goalie_nfi_gsax_by_season.csv"

MP_OUT  = os.path.join(OUT_DIR, "goalie_frame_gsax_2023now.csv")
NFI_OUT = os.path.join(OUT_DIR, "goalie_nfi_gsax_size_2023now.csv")

# season_end_year: 2023 = 2022-23, 2024 = 2023-24, 2025 = 2024-25
MP_SEASONS  = [2023, 2024, 2025]
NFI_SEASONS = [20222023, 20232024, 20242025]

MIN_SEASONS = 3
MIN_STARTS  = 25
MIN_AGE, MAX_AGE = 22, 38

HEADERS = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"),
           "Referer": "https://www.hockey-reference.com/"}
SLEEP = 3.5


# ---------------------------------------------------------------------------
# Hockey Reference scrapes (clean names via UTF-8)
# ---------------------------------------------------------------------------

def scrape_hr_seasons(years, session):
    """Return (href_map, birthyear_map) keyed by clean normalized name."""
    href_map, age_obs = {}, {}
    for yr in years:
        url = f"https://www.hockey-reference.com/leagues/NHL_{yr}_goalies.html"
        print(f"  HR season page {yr} ...")
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"    WARNING: {e}")
            continue
        r.encoding = "utf-8"
        table = BeautifulSoup(r.text, "lxml").find("table", {"id": "goalie_stats"})
        if table is None or table.find("tbody") is None:
            continue
        for tr in table.find("tbody").find_all("tr"):
            td = tr.find("td", {"data-stat": "name_display"})
            if td is None:
                continue
            nm = normalize_name(td.get_text(strip=True))
            a = td.find("a")
            if a and a.get("href"):
                href_map.setdefault(nm, a["href"])
            age_td = tr.find(["td", "th"], {"data-stat": "age"})
            try:
                age = int(age_td.get_text(strip=True))
                age_obs.setdefault(nm, []).append(yr - age)  # approx birth year
            except (ValueError, AttributeError):
                pass
        time.sleep(SLEEP)
    birthyear_map = {nm: int(np.median(v)) for nm, v in age_obs.items() if v}
    return href_map, birthyear_map


def cache_size_map():
    """clean normalized name -> (height_in, weight_lbs) from existing HR cache."""
    hr = pd.read_csv(HR_CACHE).dropna(subset=["height_in", "weight_lbs"])
    return {nm: (float(g.iloc[0].height_in), float(g.iloc[0].weight_lbs))
            for nm, g in hr.groupby("name")}


def resolve_size(names, cache_map, href_map, session):
    """names: iterable of normalized names. Returns nm -> (h, w, source)."""
    def fuzzy(n):
        p = n.split()
        return (p[-1], p[0][0]) if p else (n, "")
    fuzzy_href = {fuzzy(k): v for k, v in href_map.items()}

    out = {}
    for nm in names:
        if nm in cache_map:
            h, w = cache_map[nm]
            out[nm] = (h, w, "cache")
            continue
        href = href_map.get(nm) or fuzzy_href.get(fuzzy(nm))
        if not href:
            print(f"    NOTE: no HR href for {nm}")
            continue
        h, w = fetch_player_bio(href, session)
        if h and w:
            out[nm] = (h, w, "scraped")
            print(f"    scraped {nm}: {h}in {w}lb")
        else:
            print(f"    WARNING: incomplete bio for {nm}")
        time.sleep(SLEEP)
    return out


# ---------------------------------------------------------------------------
# Shared analysis / reporting (mirrors frame_correlations.py)
# ---------------------------------------------------------------------------

def analyze(df, x_col, y_col, x_label, low_label, high_label):
    r, p_corr = stats.pearsonr(df[x_col], df[y_col])
    thirds = pd.qcut(df[x_col], q=3, labels=["Low", "Mid", "High"])
    d = df.copy(); d["_grp"] = thirds
    low_df, high_df = d[d["_grp"] == "Low"], d[d["_grp"] == "High"]
    low, high = low_df[y_col], high_df[y_col]
    t, p_t = stats.ttest_ind(low, high)

    print(f"\n{'='*55}\n  {x_label} vs {y_col}\n{'='*55}")
    print(f"  Pearson r = {r:.4f},  p = {p_corr:.4f}  "
          f"({'sig *' if p_corr < 0.05 else 'n.s.'}),  n = {len(df)}")
    print(f"\n  {low_label}:  n = {len(low_df)},  "
          f"range = {low_df[x_col].min():.2f} - {low_df[x_col].max():.2f},  "
          f"mean {y_col} = {low.mean():.4f}")
    print(f"  {high_label}: n = {len(high_df)},  "
          f"range = {high_df[x_col].min():.2f} - {high_df[x_col].max():.2f},  "
          f"mean {y_col} = {high.mean():.4f}")
    print(f"\n  T-test (low vs high): t = {t:.4f},  p = {p_t:.4f}  "
          f"({'sig *' if p_t < 0.05 else 'n.s.'})")
    return {"metric": x_label, "r": r, "p_corr": p_corr,
            "mean_low": low.mean(), "mean_high": high.mean(),
            "n_low": len(low), "n_high": len(high), "p_ttest": p_t}


def report(df, y_col, name_col, extra_cols, results, label):
    cols = [name_col] + extra_cols + ["height_in", "weight_lbs", "weight_height_ratio", y_col]
    print(f"\n\n=== {label}: TOP 5 by {y_col} ===")
    print(df.nlargest(5, y_col)[cols].to_string(index=False))
    print(f"\n=== {label}: BOTTOM 5 by {y_col} ===")
    print(df.nsmallest(5, y_col)[cols].to_string(index=False))

    print(f"\n\n=== {label} SUMMARY TABLE ===")
    print(f"{'Metric':<30} {'r':>7} {'p(corr)':>9} {'Mean Low':>10} {'Mean High':>10} "
          f"{'n_low':>6} {'n_high':>7} {'p(t-test)':>10}")
    print("-" * 97)
    for row in results:
        sc = " *" if row["p_corr"] < 0.05 else "   "
        st = " *" if row["p_ttest"] < 0.05 else "   "
        print(f"{row['metric']:<30} {row['r']:>7.3f} {row['p_corr']:>7.3f}{sc} "
              f"{row['mean_low']:>10.4f} {row['mean_high']:>10.4f} "
              f"{row['n_low']:>6} {row['n_high']:>7} {row['p_ttest']:>8.3f}{st}")


def run_three(df, y_col, name_col, extra_cols, label):
    results = [
        analyze(df, "height_in",           y_col, "Height (inches)",              "Short", "Tall"),
        analyze(df, "weight_lbs",          y_col, "Weight (lbs)",                 "Light", "Heavy"),
        analyze(df, "weight_height_ratio", y_col, "Weight/Height Ratio (lbs/in)", "Lean",  "Wide"),
    ]
    report(df, y_col, name_col, extra_cols, results, label)


# ---------------------------------------------------------------------------
# MoneyPuck data load (cache 2023-24 + live 2024-25)
# ---------------------------------------------------------------------------

def download_mp_2025(session):
    url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies.csv"
    print(f"  Live download MoneyPuck 2024-25: {url}")
    r = session.get(url, headers={"User-Agent": HEADERS["User-Agent"],
                                  "Referer": "https://moneypuck.com/"}, timeout=30)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    d.columns = [c.strip() for c in d.columns]
    d = d[d["situation"].astype(str).str.lower() == "all"].copy()
    d["gsax"] = pd.to_numeric(d["xGoals"], errors="coerce") - pd.to_numeric(d["goals"], errors="coerce")
    gp = next(c for c in ["games_played", "gamesPlayed", "GP"] if c in d.columns)
    return pd.DataFrame({
        "name": d["name"].map(normalize_name),
        "season_end_year": 2025,
        "gsax": d["gsax"].values,
        "games_started": pd.to_numeric(d[gp], errors="coerce").values,
    })


def load_moneypuck(session):
    mp = pd.read_csv(MP_CACHE)
    mp = mp[mp["season_end_year"].isin([2023, 2024])][
        ["name", "season_end_year", "gsax", "games_started"]].copy()
    mp25 = download_mp_2025(session)
    mp = pd.concat([mp, mp25], ignore_index=True)
    print(f"  MoneyPuck season rows (2022-23..2024-25): {len(mp)} "
          f"(cache 2023-24: {len(mp)-len(mp25)}, live 2024-25: {len(mp25)})")
    return mp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    session = requests.Session()
    cache_map = cache_size_map()

    print("=== Scraping HR season pages for age + hrefs (clean names) ===")
    href_map, birthyear = scrape_hr_seasons([2022, 2023, 2024, 2025], session)

    # ======================= MoneyPuck pipeline =======================
    print("\n\n########## MONEYPUCK PIPELINE ##########")
    mp = load_moneypuck(session)

    # 25 starts per season
    mp = mp[mp["games_started"].fillna(0) >= MIN_STARTS].copy()
    # ages 22-38 (derive age from estimated birth year)
    mp["age"] = mp.apply(lambda r: r["season_end_year"] - birthyear.get(r["name"], np.nan), axis=1)
    mp = mp[mp["age"].between(MIN_AGE, MAX_AGE)].copy()
    # min 3 qualifying seasons
    counts = mp.groupby("name")["season_end_year"].nunique()
    keep = counts[counts >= MIN_SEASONS].index
    mp = mp[mp["name"].isin(keep)].copy()
    print(f"  Goalies with >= {MIN_SEASONS} qualifying seasons: {len(keep)}")

    size = resolve_size(sorted(keep), cache_map, href_map, session)
    mp_g = (mp.groupby("name")
              .agg(avg_gsax=("gsax", "mean"), qualifying_seasons=("season_end_year", "nunique"),
                   total_gsax=("gsax", "sum"))
              .reset_index())
    mp_g["height_in"]  = mp_g["name"].map(lambda n: size.get(n, (None, None, None))[0])
    mp_g["weight_lbs"] = mp_g["name"].map(lambda n: size.get(n, (None, None, None))[1])
    mp_g["size_source"] = mp_g["name"].map(lambda n: size.get(n, (None, None, None))[2])
    nb = len(mp_g)
    mp_g = mp_g.dropna(subset=["height_in", "weight_lbs"]).copy()
    print(f"  Resolved height/weight for {len(mp_g)}/{nb} goalies")
    mp_g["weight_height_ratio"] = mp_g["weight_lbs"] / mp_g["height_in"]
    mp_g["display_name"] = mp_g["name"].str.title()
    mp_g = mp_g.sort_values("avg_gsax", ascending=False).reset_index(drop=True)

    run_three(mp_g, "avg_gsax", "display_name", ["qualifying_seasons"], "MONEYPUCK (2022-23 to 2024-25)")
    mp_cols = ["display_name", "height_in", "weight_lbs", "weight_height_ratio",
               "avg_gsax", "total_gsax", "qualifying_seasons", "size_source"]
    mp_g[mp_cols].to_csv(MP_OUT, index=False)
    print(f"\n  Saved -> {MP_OUT}")

    # ========================== NFI pipeline ==========================
    print("\n\n########## NFI PIPELINE ##########")
    nfi = pd.read_csv(NFI_SEASON)
    nfi = nfi[nfi["season"].isin(NFI_SEASONS)].copy()
    nfi["nm"] = nfi["goalie_name"].map(normalize_name)
    counts = nfi.groupby("nm")["season"].nunique()
    keep = counts[counts >= MIN_SEASONS].index
    nfi = nfi[nfi["nm"].isin(keep)].copy()
    print(f"  Goalies with >= {MIN_SEASONS} qualifying seasons: {len(keep)}")

    nfi_g = (nfi.sort_values("season")
                .groupby("nm")
                .agg(goalie_name=("goalie_name", "first"), team=("team", "last"),
                     GSAx_per60=("GSAx_per60", "mean"), qualifying_seasons=("season", "nunique"),
                     total_GSAx=("GSAx", "sum"))
                .reset_index())

    size = resolve_size(sorted(keep), cache_map, href_map, session)
    nfi_g["height_in"]  = nfi_g["nm"].map(lambda n: size.get(n, (None, None, None))[0])
    nfi_g["weight_lbs"] = nfi_g["nm"].map(lambda n: size.get(n, (None, None, None))[1])
    nfi_g["size_source"] = nfi_g["nm"].map(lambda n: size.get(n, (None, None, None))[2])
    nb = len(nfi_g)
    nfi_g = nfi_g.dropna(subset=["height_in", "weight_lbs"]).copy()
    print(f"  Resolved height/weight for {len(nfi_g)}/{nb} goalies")
    nfi_g["weight_height_ratio"] = nfi_g["weight_lbs"] / nfi_g["height_in"]
    nfi_g = nfi_g.sort_values("GSAx_per60", ascending=False).reset_index(drop=True)

    run_three(nfi_g, "GSAx_per60", "goalie_name", ["team", "qualifying_seasons"], "NFI (2022-23 to 2024-25)")
    nfi_cols = ["goalie_name", "team", "qualifying_seasons", "GSAx_per60", "total_GSAx",
                "height_in", "weight_lbs", "weight_height_ratio", "size_source"]
    nfi_g[nfi_cols].to_csv(NFI_OUT, index=False)
    print(f"\n  Saved -> {NFI_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
