"""update_daily.py
-------------------
Incremental updater invoked by .github/workflows/update.yml.

Steps:
  1. Ask the NHL schedule endpoint for games that finished between
     YYYY-MM-DD(from last run) and today.
  2. Fetch shift chart + PBP for any new games not already on disk.
  3. Re-run the full compute pipeline. Rebuilding is fast (~60s) because
     the per-game reads are local.
  4. Update the Player_Ranking/last_updated_{regular,playoffs}.txt files.
  5. GitHub Actions then commits + pushes any changes.

Detection of regular vs playoff uses the gameId prefix (02 = regular,
03 = playoff) as specified. Only the current season's games feed the
current-regular / current-playoffs outputs; all four seasons feed the
pooled outputs.
"""
from __future__ import annotations
import csv, datetime as dt, json, os, subprocess, sys, time, urllib.request

ROOT    = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
DATA_DIR = ROOT
SHIFTS_DIR = os.path.join(ROOT, "raw", "shifts")
PBP_DIR    = os.path.join(ROOT, "raw", "pbp")
GAME_IDS_CSV = os.path.join(PROJECT, "Data", "game_ids.csv")

UA = {"User-Agent": "Mozilla/5.0 (HockeyROI nhl-analytics updater)"}
CURRENT_SEASON = "20252026"

def fetch_json(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1: raise
            time.sleep(2 ** i)

def last_update_date():
    path = os.path.join(DATA_DIR, "last_updated_regular.txt")
    if not os.path.exists(path):
        return dt.date.today() - dt.timedelta(days=2)
    ts = open(path).read().strip()
    try:
        return dt.datetime.strptime(ts[:10], "%Y-%m-%d").date() - dt.timedelta(days=1)
    except Exception:
        return dt.date.today() - dt.timedelta(days=2)

def game_type_from_gid(gid: str) -> str:
    # gameId prefix: YYYY{02|03}...  -> first 4 chars season, next 2 type
    if len(gid) < 6: return "unknown"
    t = gid[4:6]
    return "regular" if t == "02" else ("playoff" if t == "03" else "other")

def season_from_gid(gid: str) -> str:
    # season = YYYY + (YYYY+1)
    yyyy = gid[:4]
    try:
        y = int(yyyy)
        return f"{y}{y+1}"
    except ValueError:
        return ""

def fetch_completed_games(from_date, to_date):
    """Walk schedule day-by-day, yielding (gameId, date, home, away, type) for finished games."""
    d = from_date
    out = []
    while d <= to_date:
        try:
            js = fetch_json(f"https://api-web.nhle.com/v1/schedule/{d.isoformat()}")
        except Exception as e:
            print(f"  schedule {d} failed: {e}")
            d += dt.timedelta(days=1); continue
        for gw in js.get("gameWeek", []):
            for g in gw.get("games", []):
                state = g.get("gameState", "")
                if state not in ("OFF", "FINAL"):
                    continue
                gid = str(g["id"])
                if game_type_from_gid(gid) not in ("regular", "playoff"):
                    continue
                out.append((gid, gw["date"],
                            g["homeTeam"]["abbrev"],
                            g["awayTeam"]["abbrev"],
                            game_type_from_gid(gid)))
        d += dt.timedelta(days=1)
    # dedupe keeping first occurrence
    seen, uniq = set(), []
    for row in out:
        if row[0] in seen: continue
        seen.add(row[0]); uniq.append(row)
    return uniq

def pull_game(gid):
    sp = os.path.join(SHIFTS_DIR, f"{gid}.json")
    pp = os.path.join(PBP_DIR, f"{gid}.json")
    pulled = []
    if not os.path.exists(sp):
        d = fetch_json(f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}")
        json.dump(d, open(sp, "w"))
        pulled.append("shifts")
    if not os.path.exists(pp):
        d = fetch_json(f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play")
        json.dump(d, open(pp, "w"))
        pulled.append("pbp")
    return pulled

def append_game_ids(new_games):
    """Append new rows to Data/game_ids.csv if not already present."""
    existing = set()
    if os.path.exists(GAME_IDS_CSV):
        with open(GAME_IDS_CSV) as f:
            for r in csv.DictReader(f):
                existing.add(r["game_id"])
    added = []
    # figure out if file has header
    fields = ["game_id", "season", "game_type", "game_date", "home_abbrev", "away_abbrev"]
    need_header = not os.path.exists(GAME_IDS_CSV) or os.path.getsize(GAME_IDS_CSV) == 0
    with open(GAME_IDS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if need_header: w.writeheader()
        for gid, date, home, away, gtype in new_games:
            if gid in existing: continue
            w.writerow({"game_id": gid, "season": season_from_gid(gid),
                        "game_type": gtype, "game_date": date,
                        "home_abbrev": home, "away_abbrev": away})
            added.append(gid)
    return added

def main():
    os.makedirs(SHIFTS_DIR, exist_ok=True)
    os.makedirs(PBP_DIR, exist_ok=True)
    today = dt.date.today()
    from_date = last_update_date()
    print(f"[update] scanning {from_date} ... {today}")

    games = fetch_completed_games(from_date, today)
    print(f"[update] {len(games)} completed games found in window")

    added = append_game_ids(games)
    print(f"[update] added {len(added)} new rows to game_ids.csv")

    new_pulls = 0
    for gid, *_ in games:
        try:
            pulled = pull_game(gid)
            if pulled: new_pulls += 1
        except Exception as e:
            print(f"  pull {gid} failed: {e}")
    print(f"[update] pulled raw files for {new_pulls} games")

    # Detect which scenarios changed so we only bump the matching last_updated
    regular_changed = any(game_type_from_gid(g[0]) == "regular"
                          and season_from_gid(g[0]) == CURRENT_SEASON
                          for g in games)
    playoff_changed = any(game_type_from_gid(g[0]) == "playoff"
                          and season_from_gid(g[0]) == CURRENT_SEASON
                          for g in games)

    # Re-run the full compute pipeline
    env = dict(os.environ)
    def run(cmd):
        print(f"[run] {cmd}")
        r = subprocess.run(cmd, shell=True, cwd=ROOT, env=env)
        if r.returncode != 0:
            sys.exit(r.returncode)

    run("python3 corsi_reference.py")
    run("python3 compute_zone_and_overlap.py")
    run("python3 compute_pqr_roc_rol.py")

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if regular_changed or not os.path.exists(os.path.join(DATA_DIR, "last_updated_regular.txt")):
        open(os.path.join(DATA_DIR, "last_updated_regular.txt"), "w").write(ts)
    if playoff_changed or not os.path.exists(os.path.join(DATA_DIR, "last_updated_playoffs.txt")):
        open(os.path.join(DATA_DIR, "last_updated_playoffs.txt"), "w").write(ts)
    print("[update] done")

if __name__ == "__main__":
    main()
