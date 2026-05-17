"""
Scrape NHLDB rounds 2-3 forwards for 2016 and 2017 drafts.

Source: __NEXT_DATA__ JSON blob embedded in the SSR HTML.
No Selenium needed — page is server-rendered.
"""
import json
import os
import re
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup

URLS = {
    2016: "https://www.nhldb.com/drafts/2016",
    2017: "https://www.nhldb.com/drafts/2017",
}
OUT_PATH = os.path.expanduser("~/Documents/HockeyROI/Prospects/rounds23_raw.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (research/personal-analysis)"}

# Positions that count as forwards. Exclude D and G.
def is_forward(pos: str | None) -> bool:
    if not pos:
        return False
    # Strip and split combos like "C/LW", "LW/RW"
    parts = re.split(r"[/,\s]+", pos.strip())
    return any(p in {"C", "LW", "RW", "W", "F"} for p in parts) and not all(
        p in {"D", "G", "LD", "RD"} for p in parts
    )


def fetch_drafts(year: int) -> list[dict]:
    resp = requests.get(URLS[year], headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        raise RuntimeError(f"__NEXT_DATA__ not found for {year}")
    blob = json.loads(tag.string)
    return blob["props"]["pageProps"]["draftsData"]


def extract_row(draft_yr: int, p: dict) -> dict:
    jr = p.get("draftYearJuniors") or {}
    nhl = p.get("nhlCareer") or {}
    return {
        "draft_yr": draft_yr,
        "pick": p.get("overallPickNumber"),
        "player": p.get("playerName"),
        "pos": p.get("position"),
        "league": p.get("amateurLeague"),
        "jr_gp": jr.get("gp"),
        "jr_g": jr.get("g"),
        "jr_a": jr.get("a"),
        "jr_pts": jr.get("pts"),
        "nhl_gp": nhl.get("gp", 0),
        "nhl_pts": nhl.get("pts", 0),
    }


def main() -> None:
    all_rows: list[dict] = []
    for year, url in URLS.items():
        picks = fetch_drafts(year)
        forwards_23 = [
            p for p in picks
            if p.get("roundNumber") in (2, 3) and is_forward(p.get("position"))
        ]
        print(f"  {year}: {len(forwards_23)} round 2-3 forwards "
              f"(of {sum(1 for p in picks if p.get('roundNumber') in (2,3))} total in r2-3)")
        all_rows.extend(extract_row(year, p) for p in forwards_23)

    df = pd.DataFrame(all_rows).sort_values(["draft_yr", "pick"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n  Wrote {len(df)} rows → {OUT_PATH}\n")

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
