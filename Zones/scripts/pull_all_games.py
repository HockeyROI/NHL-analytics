"""Pull shift charts + play-by-play for all regular and playoff games,
seasons 2022-23 through 2025-26. Resumable: skips files already on disk.

Shift charts: https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}
  (the api-web.nhle.com/v1/shiftcharts/{gid} path returns 404 - confirmed.)
Play-by-play: https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play

Parallel workers keep throughput up without being rude to the API.
"""
from __future__ import annotations
import csv, json, os, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT    = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
GAME_IDS = os.path.join(PROJECT, "Data", "game_ids.csv")
SHIFTS_DIR = os.path.join(ROOT, "raw", "shifts")
PBP_DIR    = os.path.join(ROOT, "raw", "pbp")
UA = {"User-Agent": "Mozilla/5.0 (NHL analytics research)"}
SEASONS = {"20222023", "20232024", "20242025", "20252026"}
GAME_TYPES = {"regular", "playoff"}
WORKERS = 6

os.makedirs(SHIFTS_DIR, exist_ok=True)
os.makedirs(PBP_DIR, exist_ok=True)

def fetch(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return None
            if i == tries - 1: raise
            time.sleep(1 + i)
        except Exception:
            if i == tries - 1: raise
            time.sleep(1 + i)

def pull_one(gid):
    sp = os.path.join(SHIFTS_DIR, f"{gid}.json")
    pp = os.path.join(PBP_DIR, f"{gid}.json")
    status = []
    if not os.path.exists(sp):
        d = fetch(f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}")
        if d is not None:
            with open(sp, "w") as f: json.dump(d, f)
            status.append("shifts")
        else:
            status.append("shifts-404")
    if not os.path.exists(pp):
        d = fetch(f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play")
        if d is not None:
            with open(pp, "w") as f: json.dump(d, f)
            status.append("pbp")
        else:
            status.append("pbp-404")
    return gid, status

def main():
    with open(GAME_IDS) as f:
        games = [r["game_id"] for r in csv.DictReader(f)
                 if r["season"] in SEASONS and r["game_type"] in GAME_TYPES]
    print(f"[pull] {len(games)} games across seasons {sorted(SEASONS)}")

    # Skip games where both files already exist (fast pre-filter, main guard is in pull_one)
    todo = []
    for gid in games:
        if not (os.path.exists(os.path.join(SHIFTS_DIR, f"{gid}.json"))
                and os.path.exists(os.path.join(PBP_DIR, f"{gid}.json"))):
            todo.append(gid)
    print(f"[pull] {len(todo)} games still need at least one file")

    done = 0
    fails = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(pull_one, gid): gid for gid in todo}
        for fu in as_completed(futs):
            gid = futs[fu]
            try:
                gid, st = fu.result()
                done += 1
                if any(s.endswith("404") for s in st):
                    fails.append((gid, st))
                if done % 100 == 0:
                    print(f"  ... {done}/{len(todo)} done  (fails so far: {len(fails)})")
            except Exception as e:
                print(f"  FAIL {gid}: {e}")
                fails.append((gid, str(e)))
    print(f"[pull] complete. processed={done}  fail/missing={len(fails)}")
    if fails[:10]:
        print("[pull] sample failures:", fails[:10])

if __name__ == "__main__":
    main()
