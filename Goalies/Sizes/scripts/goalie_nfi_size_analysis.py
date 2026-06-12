"""
NFI GSAx_per60 vs Goalie Frame Size Analysis.

Loads pooled NFI GSAx data, filters to goalies with >= 3 seasons, scrapes
height/weight from Hockey Reference (reusing the existing HR cache where the
name matches, scraping the rest live), computes weight/height ratio, and runs
three correlations against GSAx_per60. Output mirrors the MoneyPuck frame
analysis (frame_correlations.py) summary-table format.
"""

import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy import stats

NFI_CSV   = "/Users/ashgarg/Documents/HockeyROI/NFI/output/goalie_nfi_gsax_pooled_v2.csv"
HR_CACHE  = "/Users/ashgarg/Documents/HockeyROI/Goalies/Goalies_Height/Data/cache_hockey_reference.csv"
OUT_DIR   = "/Users/ashgarg/Documents/HockeyROI/Goalies/sizes/output"
OUT_CSV   = os.path.join(OUT_DIR, "goalie_nfi_gsax_size.csv")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.hockey-reference.com/",
}
SLEEP = 3.5


def normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", ascii_name, flags=re.IGNORECASE)
    ascii_name = re.sub(r"[^a-zA-Z ]", " ", ascii_name)
    return " ".join(ascii_name.lower().split())


def fetch_player_bio(player_href, session):
    """Return (height_in, weight_lbs) from a HR player bio page."""
    url = f"https://www.hockey-reference.com{player_href}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    WARNING: bio fetch failed {url}: {e}")
        return None, None
    resp.encoding = "utf-8"  # HR omits charset; default ISO-8859-1 mangles accents
    soup = BeautifulSoup(resp.text, "lxml")
    info = soup.find("div", {"id": "info"})
    if info is None:
        return None, None
    txt = info.get_text(" ")
    h = re.search(r"\b(\d)-(\d{1,2})\b", txt)
    w = re.search(r"(\d{2,3})\s*lb", txt, re.IGNORECASE)
    height_in = int(h.group(1)) * 12 + int(h.group(2)) if h else None
    weight_lbs = float(w.group(1)) if w else None
    return height_in, weight_lbs


def scrape_season_hrefs(season_end_year, session):
    """Return {normalized_name: player_href} for one HR goalie season page."""
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_end_year}_goalies.html"
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: season fetch failed {url}: {e}")
        return {}
    resp.encoding = "utf-8"  # HR omits charset; default ISO-8859-1 mangles accents
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"id": "goalie_stats"})
    if table is None or table.find("tbody") is None:
        return {}
    out = {}
    for tr in table.find("tbody").find_all("tr"):
        td = tr.find("td", {"data-stat": "name_display"})
        if td is None:
            continue
        a = td.find("a")
        if a and a.get("href"):
            out[normalize_name(td.get_text(strip=True))] = a["href"]
    return out


def build_size_table(names_raw):
    """names_raw: list of (display_name, normalized_name). Returns dict nm -> (h,w,source)."""
    # 1) reuse HR cache (first valid height/weight per normalized name)
    hr = pd.read_csv(HR_CACHE)
    hr = hr.dropna(subset=["height_in", "weight_lbs"])
    cache_map = {}
    for nm, grp in hr.groupby("name"):
        r = grp.iloc[0]
        cache_map[nm] = (float(r.height_in), float(r.weight_lbs))

    result = {}
    missing = []
    for disp, nm in names_raw:
        if nm in cache_map:
            h, w = cache_map[nm]
            result[nm] = (h, w, "cache")
        else:
            missing.append((disp, nm))

    print(f"From cache: {len(result)} goalies.  Need to scrape: {len(missing)}")
    if not missing:
        return result

    # 2) scrape live for the missing ones — gather hrefs from recent seasons
    session = requests.Session()
    href_map = {}
    for yr in (2025, 2024, 2023, 2022, 2021):
        print(f"  Scraping season page {yr} for hrefs...")
        for nm, href in scrape_season_hrefs(yr, session).items():
            href_map.setdefault(nm, href)
        time.sleep(SLEEP)

    def fuzzy_key(n):
        parts = n.split()
        return (parts[-1], parts[0][0]) if parts else (n, "")

    fuzzy_map = {fuzzy_key(k): v for k, v in href_map.items()}

    for disp, nm in missing:
        href = href_map.get(nm) or fuzzy_map.get(fuzzy_key(nm))
        if not href:
            print(f"  NOTE: no HR href found for {disp} ({nm})")
            continue
        print(f"  Bio: {disp} -> {href}")
        h, w = fetch_player_bio(href, session)
        if h and w:
            result[nm] = (h, w, "scraped")
        else:
            print(f"  WARNING: incomplete bio for {disp}: h={h} w={w}")
        time.sleep(SLEEP)

    return result


# ---------------------------------------------------------------------------
# Analysis (mirrors frame_correlations.py format, x vs GSAx_per60)
# ---------------------------------------------------------------------------

def analyze(df, x_col, x_label, low_label, high_label):
    r, p_corr = stats.pearsonr(df[x_col], df["GSAx_per60"])
    thirds = pd.qcut(df[x_col], q=3, labels=["Low", "Mid", "High"])
    d = df.copy()
    d["_grp"] = thirds
    low_df  = d[d["_grp"] == "Low"]
    high_df = d[d["_grp"] == "High"]
    low  = low_df["GSAx_per60"]
    high = high_df["GSAx_per60"]
    t, p_t = stats.ttest_ind(low, high)

    print(f"\n{'='*55}")
    print(f"  {x_label} vs GSAx_per60")
    print(f"{'='*55}")
    print(f"  Pearson r = {r:.4f},  p = {p_corr:.4f}  "
          f"({'sig *' if p_corr < 0.05 else 'n.s.'}),  n = {len(df)}")
    print(f"\n  {low_label}:")
    print(f"    n = {len(low_df)},  range = {low_df[x_col].min():.2f} - {low_df[x_col].max():.2f}")
    print(f"    mean GSAx_per60 = {low.mean():.4f}")
    print(f"\n  {high_label}:")
    print(f"    n = {len(high_df)},  range = {high_df[x_col].min():.2f} - {high_df[x_col].max():.2f}")
    print(f"    mean GSAx_per60 = {high.mean():.4f}")
    print(f"\n  T-test (low vs high): t = {t:.4f},  p = {p_t:.4f}  "
          f"({'sig *' if p_t < 0.05 else 'n.s.'})")

    return {"metric": x_label, "r": r, "p_corr": p_corr,
            "mean_low": low.mean(), "mean_high": high.mean(),
            "n_low": len(low), "n_high": len(high), "p_ttest": p_t}


def main():
    df = pd.read_csv(NFI_CSV)
    df = df[df["n_seasons"] >= 3].copy()
    print(f"Goalies with n_seasons >= 3: {len(df)}")

    df["nm"] = df["goalie_name"].map(normalize_name)
    size = build_size_table(list(zip(df["goalie_name"], df["nm"])))

    df["height_in"]  = df["nm"].map(lambda n: size.get(n, (None, None, None))[0])
    df["weight_lbs"] = df["nm"].map(lambda n: size.get(n, (None, None, None))[1])
    df["size_source"] = df["nm"].map(lambda n: size.get(n, (None, None, None))[2])

    n_before = len(df)
    df = df.dropna(subset=["height_in", "weight_lbs"]).copy()
    print(f"\nGoalies with height+weight resolved: {len(df)} (dropped {n_before - len(df)})")

    df["weight_height_ratio"] = df["weight_lbs"] / df["height_in"]
    df = df.sort_values("GSAx_per60", ascending=False).reset_index(drop=True)

    results = [
        analyze(df, "height_in", "Height (inches)", "Short", "Tall"),
        analyze(df, "weight_lbs", "Weight (lbs)", "Light", "Heavy"),
        analyze(df, "weight_height_ratio", "Weight/Height Ratio (lbs/in)", "Lean", "Wide"),
    ]

    # Top 5 / Bottom 5 by GSAx_per60
    cols = ["goalie_name", "team", "n_seasons", "GSAx_per60",
            "height_in", "weight_lbs", "weight_height_ratio"]
    print("\n\n=== TOP 5 GOALIES by GSAx_per60 ===")
    print(df[cols].head(5).to_string(index=False))
    print("\n=== BOTTOM 5 GOALIES by GSAx_per60 ===")
    print(df[cols].tail(5).to_string(index=False))

    # Summary table (same format as MoneyPuck frame analysis)
    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'Metric':<30} {'r':>7} {'p(corr)':>9} {'Mean Low':>10} {'Mean High':>10} "
          f"{'n_low':>6} {'n_high':>7} {'p(t-test)':>10}")
    print("-" * 97)
    for row in results:
        sig_c = " *" if row["p_corr"]  < 0.05 else "   "
        sig_t = " *" if row["p_ttest"] < 0.05 else "   "
        print(f"{row['metric']:<30} {row['r']:>7.3f} {row['p_corr']:>7.3f}{sig_c} "
              f"{row['mean_low']:>10.4f} {row['mean_high']:>10.4f} "
              f"{row['n_low']:>6} {row['n_high']:>7} {row['p_ttest']:>8.3f}{sig_t}")

    df[cols + ["size_source"]].to_csv(OUT_CSV, index=False)
    print(f"\nSaved final dataset -> {OUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
