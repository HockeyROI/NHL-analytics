"""Pull shift charts for ALL 2025-26 regular season games.

Needed to compute every player's season-wide 5v5 TOI/game and CF%, which is the
basis of the quality score used for QOC/QOL.
"""
import csv, json, os, time, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(ROOT)
GAME_IDS_CSV = os.path.join(DATA_ROOT, "game_ids.csv")
SHIFTS_DIR = os.path.join(ROOT, "shifts")
UA = {"User-Agent": "Mozilla/5.0"}

def all_2025_26_regular():
    with open(GAME_IDS_CSV) as f:
        for row in csv.DictReader(f):
            if row["season"] == "20252026" and row["game_type"] == "regular":
                yield row

def fetch_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def main():
    games = list(all_2025_26_regular())
    print(f"Total 2025-26 regular games: {len(games)}")
    done = fail = skipped = 0
    for i, g in enumerate(games, 1):
        gid = g["game_id"]
        path = os.path.join(SHIFTS_DIR, f"{gid}.json")
        if os.path.exists(path):
            skipped += 1
            continue
        try:
            d = fetch_json(f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}")
            with open(path, "w") as f:
                json.dump(d, f)
            done += 1
            if done % 50 == 0:
                print(f"  ... {i}/{len(games)} pulled={done} skipped={skipped} fail={fail}")
            time.sleep(0.15)
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(games)}] {gid} FAIL: {e}")
    print(f"Done: pulled={done}, already-had={skipped}, failed={fail}")

if __name__ == "__main__":
    main()
