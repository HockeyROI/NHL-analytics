"""repair_empty_shifts.py
---------------------------
Find shift chart JSONs under Data/qoc_qol/shifts/ that are empty or corrupted
(size < 100 bytes, unparseable, or `{"data": [], "total": 0}`) and re-pull
them from the NHL API. Populated responses overwrite the empty files. Games
that still return empty after retry are written to
Player_Ranking/empty_shifts_log.txt.

Prints:
  - empty files found
  - successfully repopulated
  - still empty after retry
  - Draisaitl (8477934) GP count after reprocessing (EDM 2025-26 regular)

The user originally specified endpoint
    https://api-web.nhle.com/v1/shiftcharts/{gameId}
which returns 404 for every game id (verified against known-populated
2025020001 / 2025020006). The working stats endpoint is used instead, which
is the same one 02_pull_all_shifts.py pulls from.
"""
import csv, json, os, sys, time, urllib.request

ROOT    = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
SHIFTS_DIR = os.path.join(ROOT, "raw", "shifts")
GAME_IDS   = os.path.join(PROJECT, "Data", "game_ids.csv")
LOG_PATH   = os.path.join(ROOT, "empty_shifts_log.txt")

PRIMARY_URL = "https://api-web.nhle.com/v1/shiftcharts/{gid}"          # user-specified (404)
FALLBACK_URL = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}"
UA = {"User-Agent": "Mozilla/5.0"}

DRAISAITL_ID = 8477934
EMPTY_SIZE_THRESHOLD = 100


def is_empty_shift_file(path):
    """Return True if the file is missing/small/parse-fails/empty data."""
    if not os.path.exists(path):
        return True
    try:
        if os.path.getsize(path) < EMPTY_SIZE_THRESHOLD:
            return True
        with open(path, "r") as f:
            d = json.load(f)
    except Exception:
        return True
    if isinstance(d, dict):
        data = d.get("data")
        if data is None or len(data) == 0:
            return True
        if d.get("total") == 0:
            return True
    elif isinstance(d, list):
        if len(d) == 0:
            return True
    return False


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_shifts(gid, use_primary=False):
    """Try the user-specified endpoint first, then fall back. Return parsed
    JSON dict with `data` + `total`, or None."""
    errs = []
    if use_primary:
        try:
            d = fetch(PRIMARY_URL.format(gid=gid))
            # api-web returns a different schema if it ever works; normalise
            if isinstance(d, dict) and "data" in d:
                return d
            if isinstance(d, list):
                return {"data": d, "total": len(d)}
        except Exception as e:
            errs.append(f"primary: {e}")
    try:
        d = fetch(FALLBACK_URL.format(gid=gid))
        if isinstance(d, dict):
            return d
    except Exception as e:
        errs.append(f"fallback: {e}")
    if errs:
        print(f"  {gid} fetch errors: {'; '.join(errs)}")
    return None


def count_draisaitl_gp():
    """Count EDM 2025-26 regular-season games in which Draisaitl appears
    in the (possibly repopulated) shift files."""
    edm_games = set()
    with open(GAME_IDS) as f:
        for row in csv.DictReader(f):
            if row["season"] == "20252026" and row["game_type"] == "regular":
                if row["home_abbrev"] == "EDM" or row["away_abbrev"] == "EDM":
                    edm_games.add(row["game_id"])
    gp = 0
    for gid in edm_games:
        path = os.path.join(SHIFTS_DIR, f"{gid}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        shifts = d.get("data", []) if isinstance(d, dict) else d
        if any(isinstance(s, dict) and s.get("playerId") == DRAISAITL_ID
               for s in shifts):
            gp += 1
    return gp, len(edm_games)


def main():
    # Scan every shift file in the directory (not just EDM)
    all_files = sorted(os.listdir(SHIFTS_DIR))
    empty = [fn for fn in all_files
             if fn.endswith(".json")
             and is_empty_shift_file(os.path.join(SHIFTS_DIR, fn))]
    print(f"[repair] scanned {len(all_files)} shift files; {len(empty)} empty/corrupted")

    # Try primary endpoint once on a known id to decide whether to attempt it
    try:
        _ = fetch(PRIMARY_URL.format(gid=empty[0].replace(".json", ""))) if empty else None
        use_primary = True
    except Exception as e:
        use_primary = False
        print(f"[repair] primary endpoint unavailable ({e}); using fallback only")

    repopulated = []
    still_empty = []
    for i, fn in enumerate(empty, 1):
        gid = fn.replace(".json", "")
        path = os.path.join(SHIFTS_DIR, fn)
        d = fetch_shifts(gid, use_primary=use_primary)
        if d is None:
            still_empty.append((gid, "fetch_failed"))
            continue
        data = d.get("data", [])
        total = d.get("total", len(data))
        if data and total:
            with open(path, "w") as f:
                json.dump(d, f)
            repopulated.append(gid)
            if len(repopulated) % 25 == 0:
                print(f"  ... {i}/{len(empty)} repopulated={len(repopulated)}")
        else:
            still_empty.append((gid, f"api_empty total={total}"))
        time.sleep(0.15)

    # Log still-empty games
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        f.write(f"# Games whose shift chart remained empty after retry\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Scanned: {len(all_files)} files | found empty: {len(empty)} | "
                f"repopulated: {len(repopulated)} | still empty: {len(still_empty)}\n\n")
        for gid, reason in still_empty:
            f.write(f"{gid}\t{reason}\n")

    drai_gp, edm_total = count_draisaitl_gp()

    print()
    print(f"=== repair summary ===")
    print(f"  empty files found:           {len(empty)}")
    print(f"  successfully repopulated:    {len(repopulated)}")
    print(f"  still empty after retry:     {len(still_empty)}")
    print(f"  log written to:              {LOG_PATH}")
    print(f"  Draisaitl GP (EDM reg 25-26): {drai_gp} of {edm_total} scheduled")


if __name__ == "__main__":
    main()
