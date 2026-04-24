#!/usr/bin/env python3
"""
HockeyROI — Add 2025-26 season to Data/nhl_shot_events.csv
Appends only. Identical extraction logic to build_shot_db.py.

Step 1 — Fetch 20252026 schedule, collect completed game IDs (date ≤ today).
Step 2 — Pull play-by-play for each new game, append to nhl_shot_events.csv.
Step 3 — Append new game IDs to game_ids.csv.
Step 4 — Print summary.
"""

import csv
import math
import os
import time
from datetime import date, timedelta

import requests

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL    = "https://api-web.nhle.com/v1"
DATA_DIR    = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"
SLEEP_OK    = 0.12
SLEEP_RETRY = 2.0
GAME_TYPES  = {2: "regular", 3: "playoff"}
EVENT_TYPES = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}
CHUNK_SIZE  = 30_000    # flush every ~30k events

NEW_SEASON  = "20252026"
SEED_DATE   = "2025-10-07"     # approximate 2025-26 opening night
TODAY       = date.today().isoformat()

SHOTS_FILE  = os.path.join(DATA_DIR, "nhl_shot_events.csv")
GAME_IDS_FILE = os.path.join(DATA_DIR, "game_ids.csv")
FAILED_FILE = os.path.join(DATA_DIR, "failed_games.csv")

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
    "blocker_player_id",
    "miss_reason",
]

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "HockeyROI-Analysis/2.0"


# ── HELPERS (identical to build_shot_db.py) ───────────────────────────────────
def fetch(url, timeout=15):
    try:
        r = SESSION.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json(), True
        if r.status_code == 404:
            return None, True
    except Exception:
        pass
    time.sleep(SLEEP_RETRY)
    try:
        r = SESSION.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json(), True
    except Exception:
        pass
    return None, False


def time_to_secs(t):
    try:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return None


def should_flip(shooting_team_is_home, home_defending_side):
    if shooting_team_is_home:
        return home_defending_side == "right"
    else:
        return home_defending_side == "left"


def extract_events(game_meta, pbp_data):
    home        = pbp_data.get("homeTeam", {})
    away        = pbp_data.get("awayTeam", {})
    home_id     = home.get("id")
    home_abbrev = home.get("abbrev", "")
    away_id     = away.get("id")
    away_abbrev = away.get("abbrev", "")

    rows = []
    for play in pbp_data.get("plays", []):
        etype = play.get("typeDescKey")
        if etype not in EVENT_TYPES:
            continue

        details     = play.get("details", {})
        period_desc = play.get("periodDescriptor", {})

        owner_id = details.get("eventOwnerTeamId")
        if etype == "blocked-shot":
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

        if etype == "goal":
            shooter_id = details.get("scoringPlayerId")
        else:
            shooter_id = details.get("shootingPlayerId")

        x_raw     = details.get("xCoord")
        y_raw     = details.get("yCoord")
        home_side = play.get("homeTeamDefendingSide", "")

        if x_raw is not None and y_raw is not None and home_side:
            flip   = should_flip(shooting_is_home, home_side)
            x_norm = -x_raw if flip else x_raw
            y_norm = -y_raw if flip else y_raw
        else:
            x_norm = x_raw
            y_norm = y_raw

        rows.append({
            "game_id"                 : game_meta["game_id"],
            "season"                  : game_meta["season"],
            "game_type"               : game_meta["game_type"],
            "game_date"               : game_meta["game_date"],
            "home_team_id"            : home_id,
            "home_team_abbrev"        : home_abbrev,
            "away_team_id"            : away_id,
            "away_team_abbrev"        : away_abbrev,
            "event_id"                : play.get("eventId"),
            "period"                  : period_desc.get("number"),
            "period_type"             : period_desc.get("periodType", ""),
            "time_in_period"          : play.get("timeInPeriod", ""),
            "time_secs"               : time_to_secs(play.get("timeInPeriod", "")),
            "situation_code"          : play.get("situationCode", ""),
            "event_type"              : etype,
            "shooting_team_id"        : shooting_team_id,
            "shooting_team_abbrev"    : shooting_team_abbrev,
            "shooter_player_id"       : shooter_id,
            "goalie_id"               : details.get("goalieInNetId"),
            "shot_type"               : details.get("shotType", ""),
            "x_coord"                 : x_raw,
            "y_coord"                 : y_raw,
            "x_coord_norm"            : x_norm,
            "y_coord_norm"            : y_norm,
            "zone_code"               : details.get("zoneCode", ""),
            "is_goal"                 : 1 if etype == "goal" else 0,
            "home_team_defending_side": home_side,
            "blocker_player_id"       : details.get("blockingPlayerId"),
            "miss_reason"             : details.get("reason", "") if etype == "missed-shot" else "",
        })
    return rows


# ── STEP 1 — SCHEDULE: collect 20252026 game IDs ─────────────────────────────
print(f"── Step 1: Fetching 20252026 schedule (completed games ≤ {TODAY}) ──")

data, ok = fetch(f"{BASE_URL}/schedule/{SEED_DATE}")
if not data:
    raise SystemExit(f"ERROR: Could not fetch schedule seed for {SEED_DATE}")

reg_start   = data["regularSeasonStartDate"]
reg_end     = data["regularSeasonEndDate"]
playoff_end = data.get("playoffEndDate") or reg_end

print(f"  Season window: {reg_start} → {playoff_end}")
print(f"  Collecting games through: {TODAY}")

new_schedule_games = []
seen_ids = set()
current = reg_start

while current <= min(playoff_end, TODAY):
    sched, ok = fetch(f"{BASE_URL}/schedule/{current}")
    if not sched:
        d = date.fromisoformat(current)
        current = (d + timedelta(days=7)).isoformat()
        continue

    for week in sched.get("gameWeek", []):
        week_date = week["date"]
        for g in week.get("games", []):
            gtype = g.get("gameType")
            if gtype not in GAME_TYPES:
                continue
            gdate = g.get("gameDate", week_date)
            if gdate > TODAY:           # skip future games
                continue
            gid = g["id"]
            if gid in seen_ids:
                continue
            seen_ids.add(gid)
            new_schedule_games.append({
                "game_id"    : gid,
                "season"     : NEW_SEASON,
                "game_type"  : GAME_TYPES[gtype],
                "game_date"  : gdate,
                "home_abbrev": g["homeTeam"]["abbrev"],
                "away_abbrev": g["awayTeam"]["abbrev"],
            })

    nxt = sched.get("nextStartDate")
    if not nxt or nxt <= current:
        d = date.fromisoformat(current)
        current = (d + timedelta(days=7)).isoformat()
    else:
        current = nxt
    time.sleep(SLEEP_OK)

reg_ct  = sum(1 for g in new_schedule_games if g["game_type"] == "regular")
play_ct = sum(1 for g in new_schedule_games if g["game_type"] == "playoff")
dates   = sorted(g["game_date"] for g in new_schedule_games)
print(f"  Found: {reg_ct} regular-season games, {play_ct} playoff games")
print(f"  Date range: {dates[0] if dates else '—'} → {dates[-1] if dates else '—'}")


# ── Cross-check: which game IDs already in nhl_shot_events.csv? ──────────────
print(f"\n── Step 2: Checking already-processed games ──")
processed_ids = set()
if os.path.exists(SHOTS_FILE):
    with open(SHOTS_FILE, newline="") as f:
        for row in csv.DictReader(f):
            processed_ids.add(int(row["game_id"]))
    print(f"  nhl_shot_events.csv: {len(processed_ids):,} games already have events")

# Also load known failed IDs so we don't waste calls
failed_ids = set()
if os.path.exists(FAILED_FILE):
    with open(FAILED_FILE, newline="") as f:
        for row in csv.DictReader(f):
            failed_ids.add(int(row["game_id"]))

remaining = [
    g for g in new_schedule_games
    if int(g["game_id"]) not in processed_ids
    and int(g["game_id"]) not in failed_ids
]
already_done = len(new_schedule_games) - len(remaining)
print(f"  Already processed (from prior run): {already_done}")
print(f"  To fetch now: {len(remaining)}")


# ── Step 2 continued — pull PBP and append ───────────────────────────────────
new_events  = 0
new_failed  = 0
chunk_buf   = []
game_events = {}   # game_id → event count (for summary)

if remaining:
    shots_f  = open(SHOTS_FILE,  "a", newline="")
    failed_f = open(FAILED_FILE, "a", newline="")

    shots_writer  = csv.DictWriter(shots_f,  fieldnames=SHOT_COLS, extrasaction="ignore")
    failed_writer = csv.DictWriter(failed_f, fieldnames=["game_id","season","game_type","game_date","reason"])

    # Write header for failed_games if new
    if not os.path.exists(FAILED_FILE) or os.path.getsize(FAILED_FILE) == 0:
        failed_writer.writeheader()

    for i, game in enumerate(remaining):
        gid = int(game["game_id"])
        try:
            pbp, ok = fetch(f"{BASE_URL}/gamecenter/{gid}/play-by-play")
            if not ok or pbp is None:
                reason = "fetch_failed" if not ok else "404"
                failed_writer.writerow({
                    "game_id"  : gid, "season": game["season"],
                    "game_type": game["game_type"], "game_date": game["game_date"],
                    "reason"   : reason,
                })
                failed_f.flush()
                new_failed += 1
                game_events[gid] = 0
            else:
                events = extract_events(game, pbp)
                chunk_buf.extend(events)
                new_events += len(events)
                game_events[gid] = len(events)

            if len(chunk_buf) >= CHUNK_SIZE:
                shots_writer.writerows(chunk_buf)
                shots_f.flush()
                chunk_buf = []

        except Exception as e:
            failed_writer.writerow({
                "game_id"  : gid, "season": game["season"],
                "game_type": game["game_type"], "game_date": game["game_date"],
                "reason"   : f"exception:{type(e).__name__}:{e}",
            })
            failed_f.flush()
            new_failed += 1
            game_events[gid] = 0

        if (i + 1) % 100 == 0:
            shots_writer.writerows(chunk_buf); shots_f.flush(); chunk_buf = []
            print(f"  {i+1:>4}/{len(remaining)}  games  |  {new_events:,} events so far  "
                  f"|  {new_failed} failed", flush=True)

        time.sleep(SLEEP_OK)

    # Final flush
    if chunk_buf:
        shots_writer.writerows(chunk_buf)
        shots_f.flush()

    shots_f.close()
    failed_f.close()
    print(f"  Done: {len(remaining)} games processed, {new_events:,} events, {new_failed} failed")
else:
    print("  Nothing to fetch — all games already in nhl_shot_events.csv")


# ── Step 3 — Update game_ids.csv ─────────────────────────────────────────────
print(f"\n── Step 3: Updating game_ids.csv ──")

# Load existing game IDs
existing_gids = set()
if os.path.exists(GAME_IDS_FILE):
    with open(GAME_IDS_FILE, newline="") as f:
        for row in csv.DictReader(f):
            existing_gids.add(int(row["game_id"]))
    print(f"  Existing game_ids.csv: {len(existing_gids):,} entries")

truly_new = [g for g in new_schedule_games if int(g["game_id"]) not in existing_gids]
if truly_new:
    with open(GAME_IDS_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_id","season","game_type","game_date","home_abbrev","away_abbrev"])
        w.writerows(truly_new)
    print(f"  Appended {len(truly_new)} new 20252026 game IDs")
else:
    print(f"  game_ids.csv already up to date — no new rows needed")

total_gids = len(existing_gids) + len(truly_new)
print(f"  game_ids.csv total: {total_gids:,} entries")


# ── Step 4 — Summary ──────────────────────────────────────────────────────────
print(f"\n── Step 4: Summary ──")

# Count 2025-26 rows in nhl_shot_events.csv
s26_counts = {"regular": 0, "playoff": 0}
s26_goals  = 0
all_s26_games = set()

with open(SHOTS_FILE, newline="") as f:
    for row in csv.DictReader(f):
        if row["season"] != NEW_SEASON:
            continue
        gtype = row["game_type"]
        s26_counts[gtype] = s26_counts.get(gtype, 0) + 1
        all_s26_games.add(row["game_id"])
        if row["is_goal"] == "1":
            s26_goals += 1

# Date range of 2025-26 games we have
s26_dates = sorted(g["game_date"] for g in new_schedule_games
                   if int(g["game_id"]) in processed_ids | {int(g2["game_id"]) for g2 in remaining
                   if game_events.get(int(g2["game_id"]), 0) > 0})
# Simpler: use dates from schedule list for captured games
captured_games = [g for g in new_schedule_games
                  if int(g["game_id"]) not in failed_ids
                  and (int(g["game_id"]) in processed_ids or game_events.get(int(g["game_id"]),0) > 0)]
cap_dates = sorted(g["game_date"] for g in captured_games)

SEP = "═" * 72
print(f"\n{SEP}")
print("  HOCKEYROI — 2025-26 SEASON INGESTION SUMMARY")
print(SEP)
print(f"  Schedule collected:  {len(new_schedule_games):,} games  "
      f"({reg_ct} regular + {play_ct} playoff)")
print(f"  Date range:          {dates[0] if dates else '—'} → {dates[-1] if dates else '—'}")
print(f"  Already in DB:       {already_done} games (skipped)")
print(f"  Newly fetched:       {len(remaining) - new_failed} games successfully pulled")
print(f"  Failed:              {new_failed} games")
print(f"")
print(f"  20252026 events now in nhl_shot_events.csv:")
for gtype, cnt in sorted(s26_counts.items()):
    print(f"    {gtype:<12}  {cnt:>8,} shot events")
print(f"    {'TOTAL':<12}  {sum(s26_counts.values()):>8,} shot events")
print(f"    Goals captured:   {s26_goals:,}")
print(f"    Unique games:     {len(all_s26_games):,}")
print(f"")
print(f"  game_ids.csv:        {total_gids:,} total entries across all seasons")
print(f"  Run date:            {TODAY}")
print(SEP + "\n")
