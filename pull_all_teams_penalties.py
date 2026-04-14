#!/usr/bin/env python3
"""
HockeyROI - All-Teams Penalty Pull
Pulls every regular-season penalty for all 32 NHL teams across 3 seasons.

Usage:
  cd ".../NHL analysis/Refs"
  python3 -u pull_all_teams_penalties.py

Output:
  Refs/all_teams_penalties_3seasons.csv

Columns:
  game_id, season, date, home_team, away_team, penalized_team, opponent,
  home_or_away, referee, penalty_type, duration_minutes, period
"""

import json, time, pickle, requests, sys
import pandas as pd
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────
BASE      = "https://api-web.nhle.com/v1"
REFS_DIR  = Path("/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Refs")
CACHE     = REFS_DIR / "cache"
OUT_CSV   = REFS_DIR / "all_teams_penalties_3seasons.csv"
SEASONS   = ["20232024", "20242025", "20252026"]
CACHE.mkdir(exist_ok=True)

ALL_TEAMS = [
    "ANA","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET",
    "EDM","FLA","LAK","MIN","MTL","NJD","NSH","NYI","NYR","OTT",
    "PHI","PIT","SEA","SJS","STL","TBL","TOR","UTA","VAN","VGK","WPG","WSH",
]

# Map team tricode → NHL team ID (used to match eventOwnerTeamId)
TEAM_ID_MAP = {
    "ANA":24,"BOS":6,"BUF":7,"CAR":12,"CBJ":29,"CGY":20,"CHI":16,"COL":21,
    "DAL":25,"DET":17,"EDM":22,"FLA":13,"LAK":26,"MIN":30,"MTL":8,"NJD":1,
    "NSH":18,"NYI":2,"NYR":3,"OTT":9,"PHI":4,"PIT":5,"SEA":55,"SJS":28,
    "STL":19,"TBL":14,"TOR":10,"UTA":59,"VAN":23,"VGK":54,"WPG":52,"WSH":15,
}
ID_TO_TEAM = {v: k for k, v in TEAM_ID_MAP.items()}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "HockeyROI-Analysis/2.0"

def fetch(url, retries=3):
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=12)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(0.4 * (i + 1))
    return None

# ─── Step 1: Collect all regular-season game IDs ──────────────────────────────
def get_all_game_ids():
    gf      = CACHE / "all_game_ids.pkl"
    gf_full = CACHE / "all_game_ids_full.pkl"   # our richer format

    if gf_full.exists():
        with open(gf_full, "rb") as f:
            ids = pickle.load(f)
        print(f"  Loaded {len(ids)} game records from cache")
        return ids

    print("  Fetching game IDs from team schedules...")
    seen = set()
    ids  = []
    for season in SEASONS:
        for team in ALL_TEAMS:
            d = fetch(f"{BASE}/club-schedule-season/{team}/{season}")
            if d and "games" in d:
                for g in d["games"]:
                    if g.get("gameType") == 2 and g["id"] not in seen:
                        seen.add(g["id"])
                        ids.append((
                            g["id"],
                            season,
                            g.get("awayTeam", {}).get("abbrev", ""),
                            g.get("homeTeam", {}).get("abbrev", ""),
                            g.get("gameDate", ""),
                        ))
            time.sleep(0.05)
        print(f"    {season}: schedule fetched")

    ids.sort(key=lambda x: x[0])
    with open(gf_full, "wb") as f:
        pickle.dump(ids, f)
    print(f"  {len(ids)} unique regular-season games across 3 seasons")
    return ids

# ─── Step 2: Fetch refs for a game (cached) ───────────────────────────────────
def get_refs(gid):
    cf = CACHE / f"refs_{gid}.json"
    if cf.exists():
        with open(cf) as f:
            return json.load(f)
    d = fetch(f"{BASE}/gamecenter/{gid}/right-rail")
    refs = []
    if d:
        for o in d.get("gameInfo", {}).get("referees", []):
            name = o.get("default", "").strip()
            if name:
                refs.append(name)
    with open(cf, "w") as f:
        json.dump(refs, f)
    return refs

# ─── Step 3: Fetch penalties for a game (cached) ──────────────────────────────
def get_penalties(gid):
    cf = CACHE / f"pbp_{gid}.json"
    if cf.exists():
        with open(cf) as f:
            return json.load(f)
    d = fetch(f"{BASE}/gamecenter/{gid}/play-by-play")
    penalties = []
    if d:
        plays = d.get("plays", [])
        for play in plays:
            if play.get("typeCode") != 509:   # 509 = penalty
                continue
            det = play.get("details", {})
            penalties.append({
                "period":            play.get("periodDescriptor", {}).get("number"),
                "penalty_type":      det.get("descKey", ""),
                "duration_minutes":  det.get("duration", 0),
                "penalized_team_id": det.get("eventOwnerTeamId"),   # lives in details, not top-level
            })
    with open(cf, "w") as f:
        json.dump(penalties, f)
    return penalties

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HOCKEYROI - ALL TEAMS PENALTY PULL")
    print("32 teams | 3 seasons | Regular season only")
    print("=" * 60)

    # If output already exists, offer to skip
    if OUT_CSV.exists():
        rows = sum(1 for _ in open(OUT_CSV)) - 1
        choice = input(f"\n{OUT_CSV.name} already exists ({rows:,} rows). Rebuild? (y/n): ").strip().lower()
        if choice != 'y':
            print("Keeping existing file.")
            return

    print("\nStep 1: Collecting game IDs...")
    games = get_all_game_ids()
    total = len(games)
    print(f"  {total} games to process\n")

    all_rows = []
    checkpoint_every = 500

    for i, (gid, season, away, home, date) in enumerate(games):
        refs      = get_refs(gid)
        penalties = get_penalties(gid)
        ref_str   = ", ".join(refs) if refs else "Unknown"

        home_id = TEAM_ID_MAP.get(home)
        away_id = TEAM_ID_MAP.get(away)

        for pen in penalties:
            penalized_id   = pen["penalized_team_id"]
            penalized_team = ID_TO_TEAM.get(penalized_id, str(penalized_id))

            if penalized_id == home_id:
                opponent     = away
                home_or_away = "home"
            elif penalized_id == away_id:
                opponent     = home
                home_or_away = "away"
            else:
                opponent     = ""
                home_or_away = ""

            all_rows.append({
                "game_id":          gid,
                "season":           season,
                "date":             date,
                "home_team":        home,
                "away_team":        away,
                "penalized_team":   penalized_team,
                "opponent":         opponent,
                "home_or_away":     home_or_away,
                "referee":          ref_str,
                "penalty_type":     str(pen["penalty_type"]).replace("-", " ").replace("_", " ").title(),
                "duration_minutes": pen["duration_minutes"],
                "period":           pen["period"],
            })

        if (i + 1) % checkpoint_every == 0:
            pct = round((i + 1) / total * 100, 1)
            print(f"  {i+1}/{total} games ({pct}%) | {len(all_rows):,} penalties so far")
            # Checkpoint save
            pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)

        time.sleep(0.05)

    # Final save
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Total games processed:  {total:,}")
    print(f"Total penalties saved:  {len(df):,}")
    print(f"Output: {OUT_CSV}")
    print(f"\nBreakdown by season:")
    print(df.groupby("season")["game_id"].nunique().to_string())
    print(f"\nTop 10 most penalized teams (all situations):")
    print(df.groupby("penalized_team").size().sort_values(ascending=False).head(10).to_string())

if __name__ == "__main__":
    main()
