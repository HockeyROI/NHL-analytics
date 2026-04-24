"""
build_shift_data.py
Fetches per-shift data from the NHL shift chart API for every game in Data/game_ids.csv.
Saves results incrementally to Data/shift_data.csv; logs failures to Data/failed_shifts.csv.
"""

import csv
import os
import time
import requests

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "Data")
GAME_IDS_F  = os.path.join(DATA_DIR, "game_ids.csv")
SHIFT_OUT_F = os.path.join(DATA_DIR, "shift_data.csv")
FAILED_F    = os.path.join(DATA_DIR, "failed_shifts.csv")

# ── Config ───────────────────────────────────────────────────────────────────
SAVE_EVERY   = 500          # flush to disk every N games
RATE_LIMIT   = 0.1          # seconds between requests
API_TEMPLATE = (
    "https://api.nhle.com/stats/rest/en/shiftcharts"
    "?cayenneExp=gameId={game_id}"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

SHIFT_COLS = [
    "game_id", "player_id", "first_name", "last_name",
    "period", "team_abbrev",
    "start_time", "end_time",
    "start_secs", "end_secs",
    "abs_start_secs", "abs_end_secs",
    "type_code",
]

FAILED_COLS = ["game_id", "reason"]


# ── Helpers ──────────────────────────────────────────────────────────────────
def mmss_to_secs(mmss: str) -> int:
    """Convert 'MM:SS' string to total seconds."""
    parts = mmss.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def parse_shifts(game_id: int, raw_shifts: list) -> list:
    rows = []
    for s in raw_shifts:
        period     = int(s.get("period", 0))
        start_time = s.get("startTime", "00:00") or "00:00"
        end_time   = s.get("endTime",   "00:00") or "00:00"
        start_secs = mmss_to_secs(start_time)
        end_secs   = mmss_to_secs(end_time)
        rows.append({
            "game_id":       game_id,
            "player_id":     s.get("playerId"),
            "first_name":    s.get("firstName", ""),
            "last_name":     s.get("lastName", ""),
            "period":        period,
            "team_abbrev":   s.get("teamAbbrev", ""),
            "start_time":    start_time,
            "end_time":      end_time,
            "start_secs":    start_secs,
            "end_secs":      end_secs,
            "abs_start_secs": (period - 1) * 1200 + start_secs,
            "abs_end_secs":   (period - 1) * 1200 + end_secs,
            "type_code":     s.get("typeCode"),
        })
    return rows


def fetch_shifts(game_id: int, session: requests.Session) -> list | None:
    """Fetch shift data with one retry. Returns list of raw shifts or None on failure."""
    url = API_TEMPLATE.format(game_id=game_id)
    for attempt in range(2):
        try:
            resp = session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception as exc:
            if attempt == 0:
                time.sleep(1)           # brief pause before retry
            else:
                return None             # give up after second failure
    return None


def load_processed_ids(path: str) -> set:
    """Return set of game_ids already in shift_data.csv."""
    processed = set()
    if not os.path.exists(path):
        return processed
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(int(row["game_id"]))
    return processed


def load_game_ids(path: str) -> list:
    """Return list of game_ids from game_ids.csv."""
    ids = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["game_id"]))
    return ids


def open_writer(path: str, cols: list):
    """Open a CSV in append mode; write header only if file is new/empty."""
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=cols)
    if is_new:
        writer.writeheader()
    return fh, writer


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    all_ids   = load_game_ids(GAME_IDS_F)
    processed = load_processed_ids(SHIFT_OUT_F)

    todo = [gid for gid in all_ids if gid not in processed]
    total = len(todo)
    print(f"Games in game_ids.csv : {len(all_ids)}")
    print(f"Already processed     : {len(processed)}")
    print(f"To fetch              : {total}")

    if total == 0:
        print("Nothing to do. Exiting.")
        return

    shift_fh, shift_writer = open_writer(SHIFT_OUT_F, SHIFT_COLS)
    fail_fh,  fail_writer  = open_writer(FAILED_F,    FAILED_COLS)

    session     = requests.Session()
    buffer      = []        # accumulate rows between flushes
    fail_buffer = []
    n_done      = 0
    n_failed    = 0

    try:
        for i, game_id in enumerate(todo, start=1):
            raw = fetch_shifts(game_id, session)
            time.sleep(RATE_LIMIT)

            if raw is None:
                n_failed += 1
                fail_buffer.append({"game_id": game_id, "reason": "fetch_failed"})
            else:
                buffer.extend(parse_shifts(game_id, raw))

            n_done += 1

            # ── Incremental flush every SAVE_EVERY games ──────────────────
            if n_done % SAVE_EVERY == 0 or i == total:
                if buffer:
                    shift_writer.writerows(buffer)
                    shift_fh.flush()
                    buffer.clear()
                if fail_buffer:
                    fail_writer.writerows(fail_buffer)
                    fail_fh.flush()
                    fail_buffer.clear()
                print(
                    f"  [{n_done}/{total}] "
                    f"flushed — failures so far: {n_failed}"
                )

    finally:
        # Flush anything remaining in buffers even if interrupted
        if buffer:
            shift_writer.writerows(buffer)
        if fail_buffer:
            fail_writer.writerows(fail_buffer)
        shift_fh.close()
        fail_fh.close()

    print(f"\nDone. {n_done - n_failed} games fetched, {n_failed} failed.")
    print(f"Shift data  → {SHIFT_OUT_F}")
    print(f"Failures    → {FAILED_F}")


if __name__ == "__main__":
    main()
