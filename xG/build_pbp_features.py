#!/usr/bin/env python3
"""Pre-shot play-by-play features for xG, from the cached FULL PBP.

The shot-events file only keeps shot rows, so "time/distance since last event"
could previously only look back to the last SHOT. The Zones raw PBP cache
(Zones/raw/pbp/*.json) keeps EVERY event (faceoff, hit, giveaway, takeaway,
stoppage, shots) with coordinates + time — so we can compute the true
last-event context and a rush flag, MoneyPuck's main edge over a shot-only model.

For each shot-type event we record, relative to the immediately preceding
event in the game:
  time_since_last  seconds since the previous play (any type)
  dist_last        distance between the two event coordinates (raw rink units)
  last_type        category of the previous event
  last_zone        zoneCode of the previous event (O/N/D)
  rush             quick shot (<=5s) that followed an event outside the O-zone
                   or a controlled-transition event (takeaway/giveaway/hit) —
                   a transition-driven chance

Output: xG/output/shot_pbp_features.csv  (game_id, event_id, ...features)
keyed (game_id, event_id) to merge onto the shot table in build_xg.py.
Only 2022-23..2025-26 are cached; older shots get neutral defaults downstream.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI"))
PBP_DIR = ROOT / "Zones" / "raw" / "pbp"
OUT_CSV = ROOT / "xG" / "output" / "shot_pbp_features.csv"

SHOT_TYPES = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}
LAST_CATS = {"faceoff", "hit", "giveaway", "takeaway", "blocked-shot",
             "missed-shot", "shot-on-goal", "goal", "stoppage"}


def _abs_time(play) -> float:
    pd_ = play.get("periodDescriptor", {}) or {}
    per = pd_.get("number", 1)
    t = play.get("timeInPeriod", "0:00") or "0:00"
    try:
        mm, ss = t.split(":")
        return (int(per) - 1) * 1200 + int(mm) * 60 + int(ss)
    except Exception:
        return (int(per) - 1) * 1200


def main() -> int:
    files = sorted(glob.glob(str(PBP_DIR / "*.json")))
    if not files:
        print(f"No PBP files in {PBP_DIR}")
        return 2
    print(f"parsing {len(files):,} cached PBP games...")
    rows = []
    for n, fp in enumerate(files, 1):
        if n % 1000 == 0:
            print(f"  {n:,} games")
        try:
            gid = int(Path(fp).stem)
            d = json.load(open(fp))
        except Exception:
            continue
        plays = d.get("plays", []) if isinstance(d, dict) else d
        plays = [p for p in plays if isinstance(p, dict)]
        plays.sort(key=lambda p: p.get("sortOrder", 0))
        prev = None
        for p in plays:
            et = p.get("typeDescKey")
            det = p.get("details", {}) or {}
            x, y = det.get("xCoord"), det.get("yCoord")
            t = _abs_time(p)
            if et in SHOT_TYPES and p.get("eventId") is not None:
                if prev is not None:
                    ptype, px, py, pz, pt = prev
                    tsl = max(0.0, t - pt)
                    if x is not None and y is not None and px is not None and py is not None:
                        dl = float(np.hypot(x - px, y - py))
                    else:
                        dl = np.nan
                    lz = pz if pz in ("O", "N", "D") else "U"
                    lc = ptype if ptype in LAST_CATS else "other"
                    # rush: quick chance out of a non-offensive-zone event, or a
                    # possession-change/hit transition
                    rush = int(tsl <= 5 and (lz in ("N", "D")
                                             or lc in ("takeaway", "giveaway", "hit")))
                    rows.append((gid, int(p["eventId"]), round(tsl, 1),
                                 round(dl, 1) if pd.notna(dl) else np.nan, lc, lz, rush))
                else:
                    rows.append((gid, int(p["eventId"]), 999.0, np.nan, "none", "U", 0))
            # update prev to THIS event (any type with a usable time)
            prev = (et, x, y, det.get("zoneCode"), t)

    out = pd.DataFrame(rows, columns=["game_id", "event_id", "time_since_last",
                                      "dist_last", "last_type", "last_zone", "rush"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}: {len(out):,} shot rows with PBP context")
    print("  rush share: %.3f" % out["rush"].mean())
    print("  last_type mix:\n", out["last_type"].value_counts(normalize=True).round(3).head(8).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
