"""Pull NHL EDGE per-player, per-season tracking stats (descriptive only).

Endpoint (undocumented, discovered via live network inspection of
www.nhl.com/nhl-edge/skaters/{slug}, 2026-07-07 — the old edge.nhl.com
front-end 301-redirects there under "EDGE 2.0"):

    https://api-web.nhle.com/v1/edge/skater-detail/{playerId}/{season}/{gameTypeId}

Same api-web.nhle.com host the rest of this repo's NHL API pulls use
(NOT Sportradar — no such reference exists in the EDGE app's JS bundle).
gameTypeId: 2 = regular season, 3 = playoffs.

Season/player targets come from NFI/output/player_counts_by_state_zone_per_season.csv
(the existing NFI pipeline's per-player-per-season roster) rather than re-scraping
rosters — same (player_id, season) pairs, 2021-22 through 2025-26.

NOTE ON DATA QUALITY: the EDGE API exposes season TOTALS only (plus a single
"best game" highlight per stat) — there is no per-game log. A tracking-system
failure in an individual game (camera dropout, etc.) cannot be detected or
excluded from outside; season totals are taken as NHL computed them. See
edge/README.md.

Resumable: caches each (player, season, gameType) response as raw JSON under
edge/raw/ and skips ones already on disk on re-run.

Writes:
  edge/output/edge_skater_stats.csv           (game_type = regular)
  edge/output/edge_skater_stats_playoffs.csv  (game_type = playoffs)
"""
from __future__ import annotations
import csv, json, os, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(EDGE_DIR)

TARGET_FILE = os.path.join(REPO_ROOT, "NFI", "output",
                            "player_counts_by_state_zone_per_season.csv")
RAW_DIR = os.path.join(EDGE_DIR, "raw")
OUT_DIR = os.path.join(EDGE_DIR, "output")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (NHL analytics research)"}
WORKERS = 6
GAME_TYPES = {2: "regular", 3: "playoffs"}


def load_targets() -> list[tuple[str, str]]:
    with open(TARGET_FILE) as f:
        rows = csv.DictReader(f)
        pairs = sorted({(r["player_id"], r["season"]) for r in rows})
    return pairs


def fetch(url: str, tries: int = 3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return None
            if i == tries - 1:
                raise
            time.sleep(1 + i)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1 + i)


def pull_one(player_id: str, season: str, game_type: int) -> str:
    raw_path = os.path.join(RAW_DIR, f"{player_id}_{season}_{game_type}.json")
    if os.path.exists(raw_path):
        return "cached"
    url = f"https://api-web.nhle.com/v1/edge/skater-detail/{player_id}/{season}/{game_type}"
    data = fetch(url)
    if data is None:
        with open(raw_path, "w") as f:
            json.dump({"_missing": True}, f)
        return "missing"
    with open(raw_path, "w") as f:
        json.dump(data, f)
    return "pulled"


def _g(d: dict, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def row_from_raw(player_id: str, season: str, game_type: int, d: dict) -> dict | None:
    if not d or d.get("_missing"):
        return None
    zt = d.get("zoneTimeDetails") or None
    dist = d.get("totalDistanceSkated") or None
    speed = _g(d, "skatingSpeed", "speedMax") or None
    bursts = _g(d, "skatingSpeed", "burstsOver20") or None
    if zt is None and dist is None and speed is None:
        return None  # no EDGE data recorded for this player/season/game-type
    return {
        "player_id": player_id,
        "player_name": (_g(d, "player", "firstName", "default", default="") + " " +
                         _g(d, "player", "lastName", "default", default="")).strip(),
        "season": season,
        "game_type": GAME_TYPES[game_type],
        "position": _g(d, "player", "position"),
        "team": _g(d, "player", "team", "abbrev"),
        "games_played": _g(d, "player", "gamesPlayed"),
        "oz_time_pct": _g(zt, "offensiveZonePctg") if zt else None,
        "oz_time_pct_percentile": _g(zt, "offensiveZonePercentile") if zt else None,
        "oz_time_pct_league_avg": _g(zt, "offensiveZoneLeagueAvg") if zt else None,
        "oz_time_pct_ev": _g(zt, "offensiveZoneEvPctg") if zt else None,
        "oz_time_pct_ev_percentile": _g(zt, "offensiveZoneEvPercentile") if zt else None,
        "oz_time_pct_ev_league_avg": _g(zt, "offensiveZoneEvLeagueAvg") if zt else None,
        "nz_time_pct": _g(zt, "neutralZonePctg") if zt else None,
        "nz_time_pct_percentile": _g(zt, "neutralZonePercentile") if zt else None,
        "nz_time_pct_league_avg": _g(zt, "neutralZoneLeagueAvg") if zt else None,
        "dz_time_pct": _g(zt, "defensiveZonePctg") if zt else None,
        "dz_time_pct_percentile": _g(zt, "defensiveZonePercentile") if zt else None,
        "dz_time_pct_league_avg": _g(zt, "defensiveZoneLeagueAvg") if zt else None,
        "top_skating_speed_mph": _g(speed, "imperial") if speed else None,
        "top_skating_speed_percentile": _g(speed, "percentile") if speed else None,
        "top_skating_speed_league_avg_mph": _g(speed, "leagueAvg", "imperial") if speed else None,
        "speed_bursts_over_20mph": _g(bursts, "value") if bursts else None,
        "speed_bursts_over_20mph_percentile": _g(bursts, "percentile") if bursts else None,
        "speed_bursts_over_20mph_league_avg": _g(bursts, "leagueAvg", "value") if bursts else None,
        "distance_skated_miles": _g(dist, "imperial") if dist else None,
        "distance_skated_percentile": _g(dist, "percentile") if dist else None,
        "distance_skated_league_avg_miles": _g(dist, "leagueAvg", "imperial") if dist else None,
    }


FIELDS = [
    "player_id", "player_name", "season", "game_type", "position", "team", "games_played",
    "oz_time_pct", "oz_time_pct_percentile", "oz_time_pct_league_avg",
    "oz_time_pct_ev", "oz_time_pct_ev_percentile", "oz_time_pct_ev_league_avg",
    "nz_time_pct", "nz_time_pct_percentile", "nz_time_pct_league_avg",
    "dz_time_pct", "dz_time_pct_percentile", "dz_time_pct_league_avg",
    "top_skating_speed_mph", "top_skating_speed_percentile", "top_skating_speed_league_avg_mph",
    "speed_bursts_over_20mph", "speed_bursts_over_20mph_percentile", "speed_bursts_over_20mph_league_avg",
    "distance_skated_miles", "distance_skated_percentile", "distance_skated_league_avg_miles",
]


def main():
    pairs = load_targets()
    jobs = [(pid, season, gt) for pid, season in pairs for gt in GAME_TYPES]
    print(f"{len(pairs)} player-season pairs -> {len(jobs)} requests (regular + playoffs)")

    counts = {"cached": 0, "pulled": 0, "missing": 0}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(pull_one, pid, season, gt): (pid, season, gt) for pid, season, gt in jobs}
        done = 0
        for fut in as_completed(futs):
            status = fut.result()
            counts[status] += 1
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)}  cached={counts['cached']} pulled={counts['pulled']} missing={counts['missing']}")
    print(f"done: cached={counts['cached']} pulled={counts['pulled']} missing={counts['missing']}")

    rows_by_type: dict[int, list[dict]] = {2: [], 3: []}
    no_data = 0
    for pid, season, gt in jobs:
        raw_path = os.path.join(RAW_DIR, f"{pid}_{season}_{gt}.json")
        with open(raw_path) as f:
            d = json.load(f)
        row = row_from_raw(pid, season, gt, d)
        if row is None:
            no_data += 1
            continue
        rows_by_type[gt].append(row)

    out_regular = os.path.join(OUT_DIR, "edge_skater_stats.csv")
    out_playoffs = os.path.join(OUT_DIR, "edge_skater_stats_playoffs.csv")
    for path, gt in ((out_regular, 2), (out_playoffs, 3)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows_by_type[gt])
        print(f"wrote {path}: {len(rows_by_type[gt])} rows")
    print(f"no-EDGE-data (skipped, not written): {no_data}")


if __name__ == "__main__":
    main()
