"""build_ccg_by_season.py -- Chaos Created Goals (CCG) per situation, for the app.

CCG = a player's shot (SOG/miss/block) followed by a goal from a TEAMMATE within
0-30s of the same continuous play (never crossing a whistle/faceoff; rebounds
included). Counted per (player_id/team, season, game_type, situation), situation =
"<own>v<opp>" skaters from the shooter's perspective -- matching
player_situation_onice.csv labels so the app's per-situation engine rolls it into
buckets and per-60 by ratio-of-sums.

Reads the COMMITTED event parquets in Data/ccg_events_by_season/ (built from raw
PBP by build_ccg_events.py). This makes CCG rebuildable in CI from committed data
-- identical results locally and on the runner -- rather than needing the
gitignored raw PBP cache directly. If the parquets are absent it no-ops (keeps the
committed CSVs) instead of writing empty output.

Outputs (Data/):
  player_ccg_by_season.csv : player_id, season, game_type, situation, ccg
  team_ccg_by_season.csv   : team_id, team, season, game_type, situation, ccg
"""
from __future__ import annotations
import csv, glob, os
from collections import defaultdict
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ZONES = os.path.dirname(ROOT)
PROJECT = os.path.dirname(ZONES)
DATA_DIR = os.path.join(PROJECT, "Data")
EVENTS_DIR = os.path.join(DATA_DIR, "ccg_events_by_season")

SHOTS = {"shot-on-goal", "missed-shot", "blocked-shot"}
STOP = {"faceoff", "stoppage", "period-start", "period-end", "game-end",
        "penalty", "delayed-penalty"}
WIN = 30                        # 0-30s window, rebounds included


def main():
    files = sorted(glob.glob(os.path.join(EVENTS_DIR, "*.parquet")))
    if not files:
        print(f"[build_ccg] no event parquets in {EVENTS_DIR} -- skipping (kept "
              f"committed CSVs). Run build_ccg_events.py first (needs raw PBP).",
              flush=True)
        return
    ev = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ev = ev.sort_values(["game_id", "abs", "sort"]).reset_index(drop=True)

    pl = defaultdict(int)      # (pid, season, gtype, sit) -> ccg
    tm = defaultdict(int)      # (team_id, abbrev, season, gtype, sit) -> ccg
    ng = 0
    for gid, g in ev.groupby("game_id", sort=False):
        types = g["type"].to_numpy()
        absl = g["abs"].to_numpy()
        team = g["team_id"].to_numpy()
        shooter = g["shooter"].to_numpy()
        sitc = g["situation"].to_numpy()
        abbr = g["team_abbrev"].to_numpy()
        season = g["season"].iloc[0]
        gtype = g["game_type"].iloc[0]
        n = len(g)
        for i in range(n):
            if types[i] not in SHOTS or pd.isna(team[i]) or pd.isna(shooter[i]):
                continue
            t0, tm0, sh0 = absl[i], team[i], shooter[i]
            gg = 0
            j = i + 1
            while j < n:
                tj = types[j]
                if tj in STOP:
                    break
                if absl[j] - t0 > WIN:
                    break
                if tj == "goal":
                    if team[j] == tm0 and shooter[j] != sh0:
                        gg = 1
                    break
                j += 1
            if not gg:
                continue
            sit = sitc[i] if isinstance(sitc[i], str) and sitc[i] else "other"
            pl[(int(sh0), season, gtype, sit)] += 1
            tm[(int(tm0), abbr[i], season, gtype, sit)] += 1
        ng += 1

    pf = os.path.join(DATA_DIR, "player_ccg_by_season.csv")
    with open(pf, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "season", "game_type", "situation", "ccg"])
        for (pid, s, gt, sit), c in sorted(pl.items()):
            w.writerow([pid, s, gt, sit, c])
    tf = os.path.join(DATA_DIR, "team_ccg_by_season.csv")
    with open(tf, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team_id", "team", "season", "game_type", "situation", "ccg"])
        for (tid, ab, s, gt, sit), c in sorted(tm.items()):
            w.writerow([tid, ab, s, gt, sit, c])
    print(f"[build_ccg] {ng} games -> {len(pl):,} player rows, {len(tm):,} team "
          f"rows ({pf}, {tf})", flush=True)


if __name__ == "__main__":
    main()
