"""build_ccg_events.py -- the committed, CI-portable event source for CCG.

CCG needs the "no crossing a whistle" rule, which requires faceoff/stoppage
events that the shot-events parquet lacks. Raw PBP has them but is gitignored
(absent on CI runners). So we distil the minimal per-event stream CCG needs into
a committed per-season parquet:

  Data/ccg_events_by_season/{season}.parquet
    columns: game_id, season, game_type, abs, sort, type, team_id, team_abbrev,
             shooter, situation   (situation = '<own>v<opp>' skaters, shots only)

INCREMENTAL: only games in the raw PBP cache that are NOT already in the parquet
are added (existing games are kept as-is). So in CI, after update_current_season
fetches the week's new games into Zones/raw/pbp, this appends just those; locally
it backfills the whole cache the first time. build_ccg_by_season.py then computes
CCG from these parquets in BOTH local and CI runs -- identical results.
"""
from __future__ import annotations
import glob, json, os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ZONES = os.path.dirname(ROOT)
PROJECT = os.path.dirname(ZONES)
PBP_DIR = os.path.join(ZONES, "raw", "pbp")
OUT_DIR = os.path.join(PROJECT, "Data", "ccg_events_by_season")
os.makedirs(OUT_DIR, exist_ok=True)

SHOTS = {"shot-on-goal", "missed-shot", "blocked-shot"}
KEEP_TYPES = SHOTS | {"goal", "faceoff", "stoppage", "period-start",
                      "period-end", "game-end", "penalty", "delayed-penalty"}
KEEP = {3, 4, 5, 6}


def mmss(s):
    m, x = s.split(":")
    return int(m) * 60 + int(x)


def sit_label(code, shooter_is_home):
    s = str(code).zfill(4)
    try:
        ask, hsk = int(s[1]), int(s[2])
    except (ValueError, IndexError):
        return "other"
    own, opp = (hsk, ask) if shooter_is_home else (ask, hsk)
    return f"{own}v{opp}" if own in KEEP and opp in KEEP else "other"


def extract(path):
    g = json.load(open(path))
    season = str(g.get("season"))
    gtype = "playoff" if g.get("gameType") == 3 else "regular"
    gid = int(g.get("id"))
    home = g.get("homeTeam", {}).get("id")
    ab = {g.get("homeTeam", {}).get("id"): g.get("homeTeam", {}).get("abbrev"),
          g.get("awayTeam", {}).get("id"): g.get("awayTeam", {}).get("abbrev")}
    rows = []
    for p in g.get("plays") or []:
        t = p.get("typeDescKey")
        if t not in KEEP_TYPES:
            continue
        per = (p.get("periodDescriptor") or {}).get("number")
        tip = p.get("timeInPeriod")
        if per is None or tip is None:
            continue
        d = p.get("details") or {}
        team = d.get("eventOwnerTeamId")
        is_shot = t in SHOTS
        sit = sit_label(p.get("situationCode"), team == home) if is_shot else ""
        rows.append({
            "game_id": gid, "season": season, "game_type": gtype,
            "abs": (per - 1) * 1200 + mmss(tip), "sort": p.get("sortOrder", 0),
            "type": t, "team_id": team,
            "team_abbrev": ab.get(team, "") if team is not None else "",
            "shooter": d.get("shootingPlayerId") or d.get("scoringPlayerId"),
            "situation": sit,
        })
    return season, rows


def main():
    files = sorted(glob.glob(os.path.join(PBP_DIR, "*.json")))
    if not files:
        print(f"[ccg_events] no raw PBP in {PBP_DIR} -- nothing to add "
              f"(kept committed parquets).", flush=True)
        return
    # existing game_ids per season (to skip)
    existing = {}
    for pq in glob.glob(os.path.join(OUT_DIR, "*.parquet")):
        s = os.path.basename(pq)[:-8]
        try:
            existing[s] = set(pd.read_parquet(pq, columns=["game_id"])["game_id"].unique())
        except Exception:
            existing[s] = set()
    new_by_season = {}
    added = 0
    for path in files:
        gid = int(os.path.basename(path)[:-5])
        s_guess = str(gid)[:4]
        season_key = None
        # cheap season key from filename prefix (YYYYSS...) -> season string
        yr = int(str(gid)[:4]); season_key = f"{yr}{yr+1}"
        if gid in existing.get(season_key, set()):
            continue
        try:
            season, rows = extract(path)
        except Exception as ex:
            print(f"  !! {os.path.basename(path)}: {ex}", flush=True)
            continue
        if gid in existing.get(season, set()):
            continue
        new_by_season.setdefault(season, []).extend(rows)
        added += 1
        if added % 1000 == 0:
            print(f"  extracted {added} new games", flush=True)
    if not new_by_season:
        print("[ccg_events] no new games to add.", flush=True)
        return
    for season, rows in new_by_season.items():
        pq = os.path.join(OUT_DIR, f"{season}.parquet")
        df_new = pd.DataFrame(rows)
        if os.path.exists(pq):
            old = pd.read_parquet(pq)
            df = pd.concat([old, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=["game_id", "sort"], keep="last")
        else:
            df = df_new
        df = df.sort_values(["game_id", "abs", "sort"]).reset_index(drop=True)
        df.to_parquet(pq, index=False)
        print(f"  wrote {season}: +{df_new['game_id'].nunique()} games "
              f"({len(df):,} events total) -> {pq}", flush=True)


if __name__ == "__main__":
    main()
