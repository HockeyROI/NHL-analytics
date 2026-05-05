#!/usr/bin/env python3
"""Incremental ingest of new NHL games into the canonical CSVs.

Appends rows for the supplied game IDs to:
  - Data/nhl_shot_events.csv     (per-shot events, schema = SHOT_COLS)
  - Data/game_ids.csv            (game metadata, 6 cols)
  - NFI/Geometry_post/Data/shift_data.csv  (per-shift rows, 13 cols)

Sources (in priority order):
  - Cached PBP/shifts at Zones/raw/{pbp,shifts}/{gid}.json
  - Live NHL APIs as fallback (and to refresh empty cache placeholders)

Idempotent: a game already represented in any target CSV is skipped for that
file. If shifts data is unavailable from both cache and the live API
(NHL ingestion lag for very recent games), shots and game_ids are still
appended and the shifts append for that game is skipped with a warning.

Usage:
  python NFI/scripts/02_incremental_ingest.py --games 2025021307 2025021308 ...
  python NFI/scripts/02_incremental_ingest.py --games-file path/to/ids.txt

Honors HOCKEYROI_ROOT env var.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))

PBP_DIR     = ROOT / "Zones" / "raw" / "pbp"
SHIFT_DIR   = ROOT / "Zones" / "raw" / "shifts"
SHOTS_CSV   = ROOT / "Data" / "nhl_shot_events.csv"
GAMES_CSV   = ROOT / "Data" / "game_ids.csv"
SHIFTS_CSV  = ROOT / "NFI" / "Geometry_post" / "Data" / "shift_data.csv"

PBP_URL     = "https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
SHIFT_URL   = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}"

GAME_TYPES  = {2: "regular", 3: "playoff"}
EVENT_TYPES = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}

SHOT_COLS = [
    "game_id", "season", "game_type", "game_date",
    "home_team_id", "home_team_abbrev", "away_team_id", "away_team_abbrev",
    "event_id", "period", "period_type", "time_in_period", "time_secs",
    "situation_code", "event_type",
    "shooting_team_id", "shooting_team_abbrev",
    "shooter_player_id", "goalie_id",
    "shot_type", "x_coord", "y_coord", "x_coord_norm", "y_coord_norm",
    "zone_code", "is_goal", "home_team_defending_side",
    "blocker_player_id", "miss_reason",
]
GAME_COLS = ["game_id", "season", "game_type", "game_date", "home_abbrev", "away_abbrev"]
SHIFT_COLS = [
    "game_id", "player_id", "first_name", "last_name",
    "period", "team_abbrev", "start_time", "end_time",
    "start_secs", "end_secs", "abs_start_secs", "abs_end_secs",
    "type_code",
]


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def time_to_secs(t):
    if not t or ":" not in t:
        return None
    try:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return None


def should_flip(shooting_is_home, home_defending_side):
    if shooting_is_home:
        return home_defending_side == "right"
    return home_defending_side == "left"


def season_from_gid(gid: int) -> str:
    y = int(str(gid)[:4])
    return f"{y}{y+1}"


def gtype_from_gid(gid: int) -> str:
    code = int(str(gid)[4:6])
    return GAME_TYPES.get(code, "other")


def existing_game_ids(path: Path, col: str = "game_id") -> set[int]:
    if not path.exists():
        return set()
    out = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.add(int(r[col]))
            except (KeyError, ValueError, TypeError):
                continue
    return out


def curl_json(url: str, timeout: int = 25):
    try:
        out = subprocess.run(
            ["curl", "-s", "-m", str(timeout), url],
            capture_output=True, text=True, check=True,
        )
        return json.loads(out.stdout) if out.stdout else None
    except Exception:
        return None


def read_json_file(p: Path):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def load_pbp(gid: int):
    p = PBP_DIR / f"{gid}.json"
    js = read_json_file(p) if p.exists() else None
    if js and js.get("plays"):
        return js
    js = curl_json(PBP_URL.format(gid=gid))
    if js and js.get("plays"):
        PBP_DIR.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(js, f)
        return js
    return None


def load_shifts(gid: int):
    """Return parsed shifts JSON, refreshing the cache if it's an empty
    placeholder. Returns dict with possibly-empty 'data' list, or None on
    network failure with no cache."""
    p = SHIFT_DIR / f"{gid}.json"
    cached = read_json_file(p) if p.exists() else None
    if cached and cached.get("data"):
        return cached
    live = curl_json(SHIFT_URL.format(gid=gid))
    if live is not None:
        # If live has data and cache was empty, overwrite cache.
        if live.get("data") and (not cached or not cached.get("data")):
            SHIFT_DIR.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(live, f)
        return live
    return cached  # may be empty/None


# ─── PARSERS ───────────────────────────────────────────────────────────────────
def extract_shot_rows(gid: int, pbp: dict) -> list[dict]:
    home = pbp.get("homeTeam") or {}
    away = pbp.get("awayTeam") or {}
    home_id, home_ab = home.get("id"), home.get("abbrev", "")
    away_id, away_ab = away.get("id"), away.get("abbrev", "")
    season = season_from_gid(gid)
    gtype  = gtype_from_gid(gid)
    game_date = pbp.get("gameDate", "")

    rows = []
    for play in pbp.get("plays") or []:
        etype = play.get("typeDescKey")
        if etype not in EVENT_TYPES:
            continue
        details = play.get("details") or {}
        period_desc = play.get("periodDescriptor") or {}

        owner_id = details.get("eventOwnerTeamId")
        if etype == "blocked-shot":
            if owner_id == home_id:
                shooting_team_id, shooting_team_ab, shooting_is_home = away_id, away_ab, False
            else:
                shooting_team_id, shooting_team_ab, shooting_is_home = home_id, home_ab, True
        else:
            if owner_id == home_id:
                shooting_team_id, shooting_team_ab, shooting_is_home = home_id, home_ab, True
            else:
                shooting_team_id, shooting_team_ab, shooting_is_home = away_id, away_ab, False

        shooter_id = details.get("scoringPlayerId") if etype == "goal" else details.get("shootingPlayerId")

        x_raw = details.get("xCoord")
        y_raw = details.get("yCoord")
        home_side = play.get("homeTeamDefendingSide", "")
        if x_raw is not None and y_raw is not None and home_side:
            flip = should_flip(shooting_is_home, home_side)
            x_norm = -x_raw if flip else x_raw
            y_norm = -y_raw if flip else y_raw
        else:
            x_norm, y_norm = x_raw, y_raw

        rows.append({
            "game_id": gid,
            "season": season,
            "game_type": gtype,
            "game_date": game_date,
            "home_team_id": home_id, "home_team_abbrev": home_ab,
            "away_team_id": away_id, "away_team_abbrev": away_ab,
            "event_id": play.get("eventId"),
            "period": period_desc.get("number"),
            "period_type": period_desc.get("periodType", ""),
            "time_in_period": play.get("timeInPeriod", ""),
            "time_secs": time_to_secs(play.get("timeInPeriod", "")),
            "situation_code": play.get("situationCode", ""),
            "event_type": etype,
            "shooting_team_id": shooting_team_id,
            "shooting_team_abbrev": shooting_team_ab,
            "shooter_player_id": shooter_id,
            "goalie_id": details.get("goalieInNetId"),
            "shot_type": details.get("shotType", ""),
            "x_coord": x_raw, "y_coord": y_raw,
            "x_coord_norm": x_norm, "y_coord_norm": y_norm,
            "zone_code": details.get("zoneCode", ""),
            # Shootout "goals" are tagged event_type='goal' in the NHL PBP feed,
            # but they don't count toward a player's official goal total
            # (see HDB / NHL stat conventions). Exclude them here so downstream
            # `df[df.is_goal==1]` queries match official totals. SO rows are
            # still kept in the file for goalie shootout-save% analysis.
            "is_goal": 1 if (etype == "goal" and period_desc.get("periodType", "") != "SO") else 0,
            "home_team_defending_side": home_side,
            "blocker_player_id": details.get("blockingPlayerId"),
            "miss_reason": details.get("reason", "") if etype == "missed-shot" else "",
        })
    return rows


def extract_game_meta(gid: int, pbp: dict) -> dict:
    home = pbp.get("homeTeam") or {}
    away = pbp.get("awayTeam") or {}
    return {
        "game_id":   gid,
        "season":    season_from_gid(gid),
        "game_type": gtype_from_gid(gid),
        "game_date": pbp.get("gameDate", ""),
        "home_abbrev": home.get("abbrev", ""),
        "away_abbrev": away.get("abbrev", ""),
    }


def extract_shift_rows(gid: int, shifts_js: dict) -> list[dict]:
    rows = []
    for s in shifts_js.get("data") or []:
        pid = s.get("playerId")
        period = s.get("period")
        st = s.get("startTime") or ""
        et = s.get("endTime") or ""
        if not pid or not period or ":" not in st or ":" not in et:
            continue
        st_sec = time_to_secs(st)
        et_sec = time_to_secs(et)
        if st_sec is None or et_sec is None:
            continue
        offset = (int(period) - 1) * 1200
        rows.append({
            "game_id": gid,
            "player_id": int(pid),
            "first_name": s.get("firstName") or "",
            "last_name":  s.get("lastName") or "",
            "period": int(period),
            "team_abbrev": s.get("teamAbbrev") or "",
            "start_time": st,
            "end_time":   et,
            "start_secs": st_sec,
            "end_secs":   et_sec,
            "abs_start_secs": offset + st_sec,
            "abs_end_secs":   offset + et_sec,
            "type_code": s.get("typeCode") or "",
        })
    return rows


# ─── APPENDERS ─────────────────────────────────────────────────────────────────
def append_rows(path: Path, fields: list[str], rows: list[dict]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if need_header:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--games", nargs="+", type=int, help="Game IDs to ingest")
    g.add_argument("--games-file", type=str, help="Text file, one game ID per line")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.games:
        gids = list(args.games)
    else:
        with open(args.games_file) as f:
            gids = [int(line.strip()) for line in f if line.strip() and not line.strip().startswith("#")]

    print(f"[ingest] HOCKEYROI_ROOT = {ROOT}")
    print(f"[ingest] target games   : {gids}")

    # Hard guard: refuse playoff IDs (format YYYY03NNNNN)
    bad = [g for g in gids if gtype_from_gid(g) != "regular"]
    if bad:
        print(f"[abort] non-regular-season game_ids passed: {bad}")
        sys.exit(2)

    have_shots  = existing_game_ids(SHOTS_CSV)
    have_games  = existing_game_ids(GAMES_CSV)
    have_shifts = existing_game_ids(SHIFTS_CSV)

    shot_rows_added = game_rows_added = shift_rows_added = 0
    failures = []
    shift_skipped_unavailable = []
    games_processed = 0

    for gid in gids:
        print(f"[ingest] {gid}")
        pbp = load_pbp(gid)
        if not pbp:
            print(f"  [warn] no PBP available; skipping game entirely")
            failures.append((gid, "no_pbp"))
            continue

        # nhl_shot_events.csv
        if gid in have_shots:
            print(f"  shots: already present, skipping")
        else:
            shots = extract_shot_rows(gid, pbp)
            n = append_rows(SHOTS_CSV, SHOT_COLS, shots)
            shot_rows_added += n
            print(f"  shots: appended {n} rows")

        # game_ids.csv
        if gid in have_games:
            print(f"  game_ids: already present, skipping")
        else:
            meta = extract_game_meta(gid, pbp)
            n = append_rows(GAMES_CSV, GAME_COLS, [meta])
            game_rows_added += n
            print(f"  game_ids: appended {n} row")

        # shift_data.csv
        if gid in have_shifts:
            print(f"  shifts: already present, skipping")
        else:
            shifts_js = load_shifts(gid)
            if not shifts_js or not shifts_js.get("data"):
                print(f"  shifts: unavailable (NHL API returned 0 records); skipping append")
                shift_skipped_unavailable.append(gid)
            else:
                shifts = extract_shift_rows(gid, shifts_js)
                n = append_rows(SHIFTS_CSV, SHIFT_COLS, shifts)
                shift_rows_added += n
                print(f"  shifts: appended {n} rows")

        games_processed += 1
        time.sleep(0.05)

    print()
    print("─" * 60)
    print(f"games processed             : {games_processed}/{len(gids)}")
    print(f"shot rows appended          : {shot_rows_added}")
    print(f"game_ids rows appended      : {game_rows_added}")
    print(f"shift rows appended         : {shift_rows_added}")
    if shift_skipped_unavailable:
        print(f"shifts unavailable (skipped): {shift_skipped_unavailable}")
    if failures:
        print(f"failures                    : {failures}")
    print("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
