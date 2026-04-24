#!/usr/bin/env python3
"""
HockeyROI - NHL Shot Events Database Builder
Save to: NHL analysis/build_shot_db.py

Usage:
  cd "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
  python3 build_shot_db.py

Builds master shot events table from NHL play-by-play API for seasons
20202021 through 20242025 (regular season + playoffs).

Outputs (all in Data/):
  game_ids.csv            — all game IDs with metadata
  nhl_shot_events.csv     — master shot events table
  failed_games.csv        — any games that couldn't be fetched

Resume-safe: re-running skips already-fetched games.
"""

import csv
import math
import os
import time
from datetime import date, timedelta

import requests

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://api-web.nhle.com/v1"
DATA_DIR   = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"
CHUNK_SIZE = 500      # write to CSV every N games
SLEEP_OK   = 0.1     # seconds between successful requests
SLEEP_RETRY = 2.0    # seconds before retry on failure
GAME_TYPES = {2: "regular", 3: "playoff"}

# Known season start dates (used to seed the schedule iterator)
# The API returns correct regularSeasonStartDate / regularSeasonEndDate / playoffEndDate
# from any date within that season's window.
SEASON_SEEDS = {
    "20202021": "2021-01-13",
    "20212022": "2021-10-12",
    "20222023": "2022-10-07",
    "20232024": "2023-10-10",
    "20242025": "2024-10-04",
}

EVENT_TYPES = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}

# Output paths
GAME_IDS_FILE   = os.path.join(DATA_DIR, "game_ids.csv")
SHOTS_FILE      = os.path.join(DATA_DIR, "nhl_shot_events.csv")
FAILED_FILE     = os.path.join(DATA_DIR, "failed_games.csv")

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "HockeyROI-Analysis/2.0"


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def fetch(url, timeout=15):
    """Fetch JSON from URL. Returns (data, ok). Retries once on failure."""
    try:
        r = SESSION.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json(), True
        if r.status_code == 404:
            return None, True   # legitimate miss, don't retry
    except Exception:
        pass
    # One retry
    time.sleep(SLEEP_RETRY)
    try:
        r = SESSION.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json(), True
    except Exception:
        pass
    return None, False


def time_to_secs(t):
    """Convert 'MM:SS' to integer seconds. Returns None on bad input."""
    try:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return None


def should_flip(shooting_team_is_home, home_defending_side):
    """
    True if we need to negate (x, y) to normalize coordinates so the
    shooting team always attacks toward positive x (left-to-right).

    homeTeamDefendingSide = 'left'  → home defends x≈-89, attacks right (+x)
    homeTeamDefendingSide = 'right' → home defends x≈+89, attacks left (-x)

    We flip when the shooting team's natural attack direction is negative x:
      - Home shooting AND defending 'right'  (attacks left)
      - Away shooting AND defending 'left'   (attacks left)
    """
    if shooting_team_is_home:
        return home_defending_side == "right"
    else:
        return home_defending_side == "left"


# ─── SHOT COLUMNS ──────────────────────────────────────────────────────────────
SHOT_COLS = [
    "game_id", "season", "game_type", "game_date",
    "home_team_id", "home_team_abbrev",
    "away_team_id", "away_team_abbrev",
    "event_id", "period", "period_type", "time_in_period", "time_secs",
    "situation_code", "event_type",
    "shooting_team_id", "shooting_team_abbrev",
    "shooter_player_id", "goalie_id",
    "shot_type", "x_coord", "y_coord",
    "x_coord_norm", "y_coord_norm",
    "zone_code", "is_goal",
    "home_team_defending_side",
    "blocker_player_id",   # blocked-shot only
    "miss_reason",         # missed-shot only
]


# ─── STEP 1 — SCHEDULE ─────────────────────────────────────────────────────────
def fetch_season_games(season_code, seed_date):
    """
    Iterate the schedule endpoint week by week, collecting all regular season
    and playoff game IDs for this season. Returns list of dicts.
    """
    # Get season date window from the seed date
    data, ok = fetch(f"{BASE_URL}/schedule/{seed_date}")
    if not data:
        print(f"  ERROR: Could not fetch schedule for {season_code} seed {seed_date}")
        return []

    reg_start  = data["regularSeasonStartDate"]
    reg_end    = data["regularSeasonEndDate"]
    playoff_end = data.get("playoffEndDate") or reg_end

    print(f"  {season_code}: {reg_start} → {playoff_end}")

    games = []
    current_date = reg_start

    while current_date <= playoff_end:
        sched, ok = fetch(f"{BASE_URL}/schedule/{current_date}")
        if not sched:
            # Advance one week and keep going
            d = date.fromisoformat(current_date)
            current_date = (d + timedelta(days=7)).isoformat()
            continue

        for week in sched.get("gameWeek", []):
            for g in week.get("games", []):
                gtype = g.get("gameType")
                if gtype not in GAME_TYPES:
                    continue
                games.append({
                    "game_id"   : g["id"],
                    "season"    : season_code,
                    "game_type" : GAME_TYPES[gtype],
                    "game_date" : g.get("gameDate", week["date"]),
                    "home_abbrev": g["homeTeam"]["abbrev"],
                    "away_abbrev": g["awayTeam"]["abbrev"],
                })

        next_date = sched.get("nextStartDate")
        if not next_date or next_date <= current_date:
            # Safety: advance manually to avoid infinite loop
            d = date.fromisoformat(current_date)
            current_date = (d + timedelta(days=7)).isoformat()
        else:
            current_date = next_date

        time.sleep(0.05)

    # Deduplicate (same game can appear in multiple week windows)
    seen = set()
    unique = []
    for g in games:
        if g["game_id"] not in seen:
            seen.add(g["game_id"])
            unique.append(g)

    return unique


# ─── STEP 2 — PLAY-BY-PLAY EVENT EXTRACTION ────────────────────────────────────
def extract_events(game_meta, pbp_data):
    """
    Given game metadata dict and full play-by-play JSON, return list of
    shot-event dicts. Returns empty list if data is malformed.
    """
    home   = pbp_data.get("homeTeam", {})
    away   = pbp_data.get("awayTeam", {})
    home_id    = home.get("id")
    home_abbrev = home.get("abbrev", "")
    away_id    = away.get("id")
    away_abbrev = away.get("abbrev", "")

    rows = []
    for play in pbp_data.get("plays", []):
        etype = play.get("typeDescKey")
        if etype not in EVENT_TYPES:
            continue

        details = play.get("details", {})
        period_desc = play.get("periodDescriptor", {})

        # ── Shooting team ─────────────────────────────────────────────────────
        owner_id = details.get("eventOwnerTeamId")
        if etype == "blocked-shot":
            # eventOwnerTeamId = blocker's team → shooter is the OTHER team
            if owner_id == home_id:
                shooting_team_id     = away_id
                shooting_team_abbrev = away_abbrev
                shooting_is_home     = False
            else:
                shooting_team_id     = home_id
                shooting_team_abbrev = home_abbrev
                shooting_is_home     = True
        else:
            if owner_id == home_id:
                shooting_team_id     = home_id
                shooting_team_abbrev = home_abbrev
                shooting_is_home     = True
            else:
                shooting_team_id     = away_id
                shooting_team_abbrev = away_abbrev
                shooting_is_home     = False

        # ── Shooter player ID ─────────────────────────────────────────────────
        if etype == "goal":
            shooter_id = details.get("scoringPlayerId")
        else:
            shooter_id = details.get("shootingPlayerId")

        # ── Coordinate normalization ──────────────────────────────────────────
        x_raw = details.get("xCoord")
        y_raw = details.get("yCoord")
        home_side = play.get("homeTeamDefendingSide", "")

        if x_raw is not None and y_raw is not None and home_side:
            flip = should_flip(shooting_is_home, home_side)
            x_norm = -x_raw if flip else x_raw
            y_norm = -y_raw if flip else y_raw
        else:
            x_norm = x_raw
            y_norm = y_raw

        rows.append({
            "game_id"              : game_meta["game_id"],
            "season"               : game_meta["season"],
            "game_type"            : game_meta["game_type"],
            "game_date"            : game_meta["game_date"],
            "home_team_id"         : home_id,
            "home_team_abbrev"     : home_abbrev,
            "away_team_id"         : away_id,
            "away_team_abbrev"     : away_abbrev,
            "event_id"             : play.get("eventId"),
            "period"               : period_desc.get("number"),
            "period_type"          : period_desc.get("periodType", ""),
            "time_in_period"       : play.get("timeInPeriod", ""),
            "time_secs"            : time_to_secs(play.get("timeInPeriod", "")),
            "situation_code"       : play.get("situationCode", ""),
            "event_type"           : etype,
            "shooting_team_id"     : shooting_team_id,
            "shooting_team_abbrev" : shooting_team_abbrev,
            "shooter_player_id"    : shooter_id,
            "goalie_id"            : details.get("goalieInNetId"),
            "shot_type"            : details.get("shotType", ""),
            "x_coord"              : x_raw,
            "y_coord"              : y_raw,
            "x_coord_norm"         : x_norm,
            "y_coord_norm"         : y_norm,
            "zone_code"            : details.get("zoneCode", ""),
            "is_goal"              : 1 if etype == "goal" else 0,
            "home_team_defending_side": home_side,
            "blocker_player_id"    : details.get("blockingPlayerId"),
            "miss_reason"          : details.get("reason", "") if etype == "missed-shot" else "",
        })

    return rows


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Step 1: Game IDs ──────────────────────────────────────────────────────
    if os.path.exists(GAME_IDS_FILE):
        print(f"game_ids.csv exists — loading cached game list...")
        with open(GAME_IDS_FILE, newline="") as f:
            all_games = list(csv.DictReader(f))
        # Restore int types
        for g in all_games:
            g["game_id"] = int(g["game_id"])
        print(f"  Loaded {len(all_games):,} games from cache")
    else:
        print("── Step 1: Fetching game IDs from schedule API ──")
        all_games = []
        for season, seed in SEASON_SEEDS.items():
            season_games = fetch_season_games(season, seed)
            all_games.extend(season_games)
            reg = sum(1 for g in season_games if g["game_type"] == "regular")
            play = sum(1 for g in season_games if g["game_type"] == "playoff")
            print(f"  {season}: {reg} regular season, {play} playoff games")

        with open(GAME_IDS_FILE, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["game_id","season","game_type","game_date","home_abbrev","away_abbrev"])
            w.writeheader()
            w.writerows(all_games)
        print(f"\n  Saved {len(all_games):,} games → {GAME_IDS_FILE}")

    # Print per-season summary
    print("\n  Game counts by season:")
    seasons = {}
    for g in all_games:
        key = (g["season"], g["game_type"])
        seasons[key] = seasons.get(key, 0) + 1
    for (s, t), cnt in sorted(seasons.items()):
        print(f"    {s}  {t:<10}  {cnt:>4} games")

    # ── Step 2: Play-by-play extraction ──────────────────────────────────────
    print(f"\n── Step 2: Pulling play-by-play ──")

    # Resume: find already-processed game IDs
    processed_ids = set()
    shots_file_exists = os.path.exists(SHOTS_FILE)
    if shots_file_exists:
        with open(SHOTS_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_ids.add(int(row["game_id"]))
        print(f"  Resume: {len(processed_ids):,} games already processed — skipping")

    # Load already-failed game IDs so we don't retry them
    failed_ids = set()
    if os.path.exists(FAILED_FILE):
        with open(FAILED_FILE, newline="") as f:
            for row in csv.DictReader(f):
                failed_ids.add(int(row["game_id"]))

    remaining = [g for g in all_games
                 if int(g["game_id"]) not in processed_ids
                 and int(g["game_id"]) not in failed_ids]
    print(f"  Games to process: {len(remaining):,}")

    if not remaining:
        print("  Nothing to do — all games already processed.")
    else:
        # Open output files in append mode
        shots_f  = open(SHOTS_FILE,  "a", newline="")
        failed_f = open(FAILED_FILE, "a", newline="")

        shots_writer  = csv.DictWriter(shots_f,  fieldnames=SHOT_COLS, extrasaction="ignore")
        failed_writer = csv.DictWriter(failed_f, fieldnames=["game_id","season","game_type","game_date","reason"])

        if not shots_file_exists:
            shots_writer.writeheader()
        if not os.path.exists(FAILED_FILE) or os.path.getsize(FAILED_FILE) == 0:
            failed_writer.writeheader()

        chunk_buf   = []
        new_failed  = 0
        new_events  = 0

        for i, game in enumerate(remaining):
            gid = int(game["game_id"])
            try:
                url = f"{BASE_URL}/gamecenter/{gid}/play-by-play"
                pbp, ok = fetch(url)

                if not ok or pbp is None:
                    failed_writer.writerow({
                        "game_id"  : gid,
                        "season"   : game["season"],
                        "game_type": game["game_type"],
                        "game_date": game["game_date"],
                        "reason"   : "fetch_failed" if not ok else "404",
                    })
                    failed_f.flush()
                    new_failed += 1
                else:
                    events = extract_events(game, pbp)
                    chunk_buf.extend(events)
                    new_events += len(events)

                # Flush chunk
                if len(chunk_buf) >= CHUNK_SIZE * 60:   # ~60 events/game avg → flush every ~500 games
                    shots_writer.writerows(chunk_buf)
                    shots_f.flush()
                    chunk_buf = []

                if (i + 1) % 500 == 0:
                    # Force flush at exactly every 500 games regardless
                    shots_writer.writerows(chunk_buf)
                    shots_f.flush()
                    chunk_buf = []
                    print(f"    {i+1:,}/{len(remaining):,} games processed  "
                          f"({new_events:,} events so far, {new_failed} failed)")

            except Exception as e:
                failed_writer.writerow({
                    "game_id"  : gid,
                    "season"   : game["season"],
                    "game_type": game["game_type"],
                    "game_date": game["game_date"],
                    "reason"   : f"exception: {type(e).__name__}: {e}",
                })
                failed_f.flush()
                new_failed += 1

            time.sleep(SLEEP_OK)

        # Final flush
        if chunk_buf:
            shots_writer.writerows(chunk_buf)
            shots_f.flush()

        shots_f.close()
        failed_f.close()

        print(f"\n  Done: {len(remaining):,} games processed")
        print(f"  New events extracted: {new_events:,}")
        print(f"  Failed games: {new_failed}")

    # ── Step 3: Summary ───────────────────────────────────────────────────────
    print(f"\n── Step 3: Summary ──")

    if not os.path.exists(SHOTS_FILE):
        print("  No shot events file found.")
        return

    # Count by season + game_type, total goals, file size
    totals  = {}
    goals   = 0
    total_rows = 0

    with open(SHOTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            key = (row["season"], row["game_type"])
            totals[key] = totals.get(key, 0) + 1
            if row.get("is_goal") == "1":
                goals += 1

    file_mb = os.path.getsize(SHOTS_FILE) / 1_048_576

    failed_count = 0
    if os.path.exists(FAILED_FILE):
        with open(FAILED_FILE, newline="") as f:
            failed_count = sum(1 for _ in csv.DictReader(f))

    print(f"\n  Shot events by season and game type:")
    print(f"  {'Season':<12}  {'Type':<10}  {'Events':>10}")
    print(f"  {'-'*36}")
    for (s, t), cnt in sorted(totals.items()):
        print(f"  {s:<12}  {t:<10}  {cnt:>10,}")
    print(f"  {'-'*36}")
    print(f"  {'TOTAL':<24}  {total_rows:>10,}")
    print(f"\n  Total goals captured : {goals:,}")
    print(f"  Failed game IDs      : {failed_count}")
    print(f"  Output file size     : {file_mb:.1f} MB")
    print(f"  Output file          : {SHOTS_FILE}")
    print()


if __name__ == "__main__":
    main()
