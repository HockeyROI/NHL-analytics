"""build_ccg_by_season.py -- Chaos Created Goals (CCG) per situation, for the app.

CCG = a player's shot (SOG/miss/block) followed by a goal from a TEAMMATE within
0-30s of game clock, same continuous play (no crossing a whistle). Rebounds
INCLUDED (0-30s, per the published definition). Counted per
(player_id, season, game_type, situation) with situation = "<own>v<opp>" skaters
from the shooter's team perspective -- matching player_situation_onice.csv labels
so the app's per-situation engine can roll it into buckets and per-60 by
ratio-of-sums exactly like iG/ixG.

Needs raw PBP (Zones/raw/pbp) because the "no crossing whistle" rule needs
faceoff/stoppage events, which the shot-events CSV lacks.

Outputs (Data/):
  player_ccg_by_season.csv : player_id, season, game_type, situation, ccg
  team_ccg_by_season.csv   : team_id, team, season, game_type, situation, ccg
"""
from __future__ import annotations
import csv, glob, json, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
ZONES = os.path.dirname(ROOT)
PROJECT = os.path.dirname(ZONES)
PBP_DIR = os.path.join(ZONES, "raw", "pbp")
OUT_DIR = os.path.join(PROJECT, "Data")

SHOTS = {"shot-on-goal", "missed-shot", "blocked-shot"}
STOP = {"faceoff", "stoppage", "period-start", "period-end", "game-end",
        "penalty", "delayed-penalty"}
WIN = 30                        # 0-30s window, rebounds INCLUDED
KEEP = {3, 4, 5, 6}


def mmss(s):
    m, x = s.split(":")
    return int(m) * 60 + int(x)


def sit_label(code, shooter_is_home):
    """situationCode = awayG awaySk homeSk homeG -> '<own>v<opp>' skaters from
    the shooter's team perspective (matches build_situation_onice.py)."""
    s = str(code).zfill(4)
    try:
        ask, hsk = int(s[1]), int(s[2])
    except (ValueError, IndexError):
        return "other"
    own, opp = (hsk, ask) if shooter_is_home else (ask, hsk)
    if own in KEEP and opp in KEEP:
        return f"{own}v{opp}"
    return "other"


def main():
    pl = defaultdict(int)      # (pid, season, gtype, sit) -> ccg
    tm = defaultdict(int)      # (team_id, abbrev, season, gtype, sit) -> ccg
    files = sorted(glob.glob(os.path.join(PBP_DIR, "*.json")))
    # Raw PBP is gitignored and absent on CI runners. Never overwrite the
    # committed CCG CSVs with empty output -- no PBP -> no-op (keep last build).
    if not files:
        print(f"[build_ccg] no raw PBP in {PBP_DIR} -- skipping (kept committed "
              f"CSVs). Run locally where the raw cache exists.", flush=True)
        return
    print(f"processing {len(files)} games", flush=True)
    for k, path in enumerate(files):
        g = json.load(open(path))
        season = str(g.get("season"))
        gtype = "playoff" if g.get("gameType") == 3 else "regular"
        home = g.get("homeTeam", {}).get("id")
        abbr = {g.get("homeTeam", {}).get("id"): g.get("homeTeam", {}).get("abbrev"),
                g.get("awayTeam", {}).get("id"): g.get("awayTeam", {}).get("abbrev")}
        ev = []
        for p in g.get("plays") or []:
            per = (p.get("periodDescriptor") or {}).get("number")
            tip = p.get("timeInPeriod")
            if per is None or tip is None:
                continue
            d = p.get("details") or {}
            ev.append({"t": p.get("typeDescKey"),
                       "abs": (per - 1) * 1200 + mmss(tip),
                       "team": d.get("eventOwnerTeamId"),
                       "sit": p.get("situationCode"),
                       "sh": d.get("shootingPlayerId") or d.get("scoringPlayerId"),
                       "stop": p.get("typeDescKey") in STOP})
        ev.sort(key=lambda e: e["abs"])
        n = len(ev)
        for i, e in enumerate(ev):
            if e["t"] not in SHOTS or e["team"] is None or e["sh"] is None:
                continue
            # teammate goal within (0, WIN], same continuous play
            gg = 0
            j = i + 1
            while j < n:
                f = ev[j]
                if f["stop"]:
                    break
                if f["abs"] - e["abs"] > WIN:
                    break
                if f["t"] == "goal":
                    if f["team"] == e["team"] and f["sh"] != e["sh"]:
                        gg = 1
                    break
                j += 1
            if not gg:
                continue
            sit = sit_label(e["sit"], e["team"] == home)
            pl[(e["sh"], season, gtype, sit)] += 1
            tm[(e["team"], abbr.get(e["team"], ""), season, gtype, sit)] += 1
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(files)}", flush=True)

    pf = os.path.join(OUT_DIR, "player_ccg_by_season.csv")
    with open(pf, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "season", "game_type", "situation", "ccg"])
        for (pid, s, gt, sit), c in sorted(pl.items()):
            w.writerow([pid, s, gt, sit, c])
    tf = os.path.join(OUT_DIR, "team_ccg_by_season.csv")
    with open(tf, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team_id", "team", "season", "game_type", "situation", "ccg"])
        for (tid, ab, s, gt, sit), c in sorted(tm.items()):
            w.writerow([tid, ab, s, gt, sit, c])
    print(f"wrote {len(pl):,} player rows -> {pf}")
    print(f"wrote {len(tm):,} team rows -> {tf}")


if __name__ == "__main__":
    main()
