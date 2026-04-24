#!/usr/bin/env python3
"""Fetch 20252026 standings and append to standings_5seasons.csv,
producing standings_pool5.csv covering 20212022..20252026."""
import urllib.request, json, csv, os
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
IN = f"{ROOT}/NFI/Geometry_post/Data/standings_5seasons.csv"
OUT = f"{ROOT}/NFI/output/standings_pool5.csv"

UA = {"User-Agent": "Mozilla/5.0 (research)"}

def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)

# NHL standings endpoint: "https://api-web.nhle.com/v1/standings/now"
# Need season-final standings -> use 2026-04-16 (day after regular season end)
d = fetch("https://api-web.nhle.com/v1/standings/2026-04-16")
rows = []
for t in d.get("standings", []):
    rows.append({
        "season": "20252026",
        "team": t["teamAbbrev"]["default"],
        "gp": t["gamesPlayed"],
        "points": t["points"],
        "points_pct": round(t["pointPctg"], 4),
        "wins": t["wins"],
        "reg_wins": t.get("regulationWins", t["wins"]),
        "goal_diff": t["goalFor"] - t["goalAgainst"],
    })
df2526 = pd.DataFrame(rows)
print(f"Fetched {len(df2526)} 20252026 standings")
print(df2526.head().to_string(index=False))

base = pd.read_csv(IN)
# keep 20212022..20242025 and append 20252026
base = base[base["season"].astype(str).isin({"20212022","20222023","20232024","20242025"})]
pool = pd.concat([base, df2526], ignore_index=True)
pool.to_csv(OUT, index=False)
print(f"Wrote pool5 standings -> {OUT} ({len(pool)} rows)")
