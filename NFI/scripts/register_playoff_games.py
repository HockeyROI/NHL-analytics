#!/usr/bin/env python3
"""Discover finished playoff games from the NHL schedule API and register any
missing ones in Data/game_ids.csv (game_type=playoff).

Why this exists: build_shot_db.py / pull_all_games.py are resume-safe and DO
handle playoffs, but they only fetch game_ids that are already in game_ids.csv.
The incremental weekly updater (update_current_season.py) is regular-season only
(gameType==2, SEASON_END 2026-04-30), so a postseason silently never gets
registered — which is exactly how the entire 2025-26 playoffs went missing.

Run this (idempotent) to register the current season's playoff games, then run
build_shot_db.py + pull_all_games.py to fetch their shot events / PBP / shifts.

The NHL API 403s a bare urllib request, so a User-Agent header is REQUIRED.
"""
import csv
import json
import os
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))
GAME_IDS = ROOT / "Data" / "game_ids.csv"
BASE = "https://api-web.nhle.com/v1"
UA = {"User-Agent": "HockeyROI-Analysis/2.0"}   # REQUIRED — API 403s without it

# season code -> a seed date inside that season (the API returns the real
# playoffEndDate from any date in-window).
SEASON_SEEDS = {"20252026": "2026-04-12"}


def fetch(url):
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.load(r)
    except Exception as e:
        print(f"  fetch failed {url}: {e}")
        return None


def discover_playoff_games(season_code: str, seed: str) -> list[dict]:
    """Walk the schedule from the seed date to playoffEndDate, collecting every
    FINAL playoff (gameType==3) game."""
    data = fetch(f"{BASE}/schedule/{seed}")
    if not data:
        return []
    end = data.get("playoffEndDate") or data.get("regularSeasonEndDate")
    cur = seed
    out, seen = [], set()
    while cur <= end:
        sched = fetch(f"{BASE}/schedule/{cur}")
        if sched:
            for week in sched.get("gameWeek", []):
                for g in week.get("games", []):
                    if (g.get("gameType") == 3
                            and g.get("gameState") in ("OFF", "FINAL")
                            and g["id"] not in seen):
                        seen.add(g["id"])
                        out.append({
                            "game_id": g["id"],
                            "season": season_code,
                            "game_type": "playoff",
                            "game_date": g.get("gameDate", week["date"]),
                            "home_abbrev": g["homeTeam"]["abbrev"],
                            "away_abbrev": g["awayTeam"]["abbrev"],
                        })
            nxt = sched.get("nextStartDate")
            cur = nxt if (nxt and nxt > cur) else (
                date.fromisoformat(cur) + timedelta(days=7)).isoformat()
        else:
            cur = (date.fromisoformat(cur) + timedelta(days=7)).isoformat()
    return sorted(out, key=lambda r: r["game_id"])


def main() -> int:
    existing = set()
    fieldnames = ["game_id", "season", "game_type", "game_date",
                  "home_abbrev", "away_abbrev"]
    if GAME_IDS.exists():
        with open(GAME_IDS) as f:
            for r in csv.DictReader(f):
                existing.add(int(r["game_id"]))
    print(f"game_ids.csv: {len(existing):,} games already registered")

    new_rows = []
    for season, seed in SEASON_SEEDS.items():
        found = discover_playoff_games(season, seed)
        missing = [g for g in found if int(g["game_id"]) not in existing]
        print(f"  {season}: {len(found)} final playoff games in API, "
              f"{len(missing)} not yet registered")
        new_rows.extend(missing)

    if not new_rows:
        print("Nothing to add — every finished playoff game is already registered.")
        return 0

    with open(GAME_IDS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for r in new_rows:
            w.writerow(r)
    print(f"Appended {len(new_rows)} playoff games to {GAME_IDS}.")
    print("Next: run NF_PY/build_shot_db.py + Zones/scripts/pull_all_games.py "
          "to fetch their shot events / PBP / shifts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
