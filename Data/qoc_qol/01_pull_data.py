"""Pull shift charts + play-by-play for all EDM 2025-26 regular season games.

Shift charts (strength state must be inferred from skater counts):
  https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gameId}
Play-by-play (has rosterSpots with positions => identifies goalies):
  https://api-web.nhle.com/v1/gamecenter/{gameId}/play-by-play
"""
import csv, json, os, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(ROOT)
GAME_IDS_CSV = os.path.join(DATA_ROOT, "game_ids.csv")
SHIFTS_DIR = os.path.join(ROOT, "shifts")
PBP_DIR = os.path.join(ROOT, "pbp")
UA = {"User-Agent": "Mozilla/5.0"}

def edm_games_2025_26():
    games = []
    with open(GAME_IDS_CSV) as f:
        for row in csv.DictReader(f):
            if row["season"] == "20252026" and row["game_type"] == "regular" \
                    and ("EDM" in (row["home_abbrev"], row["away_abbrev"])):
                games.append(row)
    return games

def fetch_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)

def pull_game(gid):
    shift_path = os.path.join(SHIFTS_DIR, f"{gid}.json")
    pbp_path = os.path.join(PBP_DIR, f"{gid}.json")
    if not os.path.exists(shift_path):
        d = fetch_json(f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}")
        with open(shift_path, "w") as f:
            json.dump(d, f)
        time.sleep(0.25)
    if not os.path.exists(pbp_path):
        d = fetch_json(f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play")
        with open(pbp_path, "w") as f:
            json.dump(d, f)
        time.sleep(0.25)

def main():
    games = edm_games_2025_26()
    print(f"Found {len(games)} EDM 2025-26 regular season games")
    for i, g in enumerate(games, 1):
        gid = g["game_id"]
        try:
            pull_game(gid)
            print(f"[{i}/{len(games)}] {gid} {g['game_date']} {g['away_abbrev']}@{g['home_abbrev']}  OK")
        except Exception as e:
            print(f"[{i}/{len(games)}] {gid}  FAIL: {e}")

if __name__ == "__main__":
    main()
