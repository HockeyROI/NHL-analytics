#!/usr/bin/env python3
"""Repair corrupted 2024-25 hit zoneCodes in the raw PBP.

The NHL's 2024-25 play-by-play feed mis-codes hit events' `details.zoneCode`
(~67% disagree with their own coordinates, mostly dumped into 'N'), which
poisons the V1 zone-time walk in compute_zone_variations.py (hits are in the V1
event set). Every OTHER event type in 2024-25 is clean (0-1% conflict), and the
coordinates are intact (100% coverage), so we re-derive each 2024-25 hit's
zoneCode from `details.xCoord` + `homeTeamDefendingSide`.

Derivation (validated at 100.0% against 2023-24 and 2025-26 trusted zoneCodes):
  home attacks toward -x if homeTeamDefendingSide == 'right' else +x
  |xCoord| <= 24 -> N ; else O if sign(x) == home-attack-sign else D (home frame)
  zoneCode is owner-perspective -> flip O<->D when eventOwner is the away team

Only touches 2024-25 (season == 20242025) hit events; rewrites the raw JSON in
place (idempotent — re-deriving from coordinates is stable). Other seasons and
other event types are left untouched.

Run before build_zone_index100.py so the rebuilt index uses the corrected zones.
"""
import glob
import json
import os
import sys

# compute_zone_variations reads Zones/raw/pbp (ZONES = scripts-dir parent) — the
# build consumes these exact files, so repair them in place.
_ZONES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBP_DIR = os.path.join(_ZONES, "raw", "pbp")
SEASON = 20242025
NEUTRAL_HALF_WIDTH = 24        # |x| <= 24 -> neutral zone (blue line at 25)
FLIP = {"O": "D", "D": "O", "N": "N"}


def derive_owner_zone(x, home_defend_side, owner_id, home_id):
    """Owner-perspective zoneCode from coordinate + home defending side."""
    home_attack_sign = -1 if home_defend_side == "right" else 1
    if abs(x) <= NEUTRAL_HALF_WIDTH:
        ez_home = "N"
    elif (x > 0) == (home_attack_sign > 0):
        ez_home = "O"
    else:
        ez_home = "D"
    return ez_home if owner_id == home_id else FLIP[ez_home]


def repair_file(path):
    """Returns (n_hits, n_changed, n_skipped) for one game file; rewrites if changed."""
    try:
        d = json.load(open(path))
    except Exception:
        return (0, 0, 0)
    if int(d.get("season", 0)) != SEASON:
        return (0, 0, 0)
    home_id = (d.get("homeTeam") or {}).get("id")
    hits = changed = skipped = 0
    for p in d.get("plays", []):
        if p.get("typeDescKey") != "hit":
            continue
        det = p.get("details") or {}
        x = det.get("xCoord")
        owner = det.get("eventOwnerTeamId")
        hds = p.get("homeTeamDefendingSide")
        if x is None or owner is None or hds not in ("left", "right") or home_id is None:
            skipped += 1
            continue
        hits += 1
        new_z = derive_owner_zone(x, hds, owner, home_id)
        if det.get("zoneCode") != new_z:
            det["zoneCode"] = new_z
            changed += 1
    if changed:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, path)
    return (hits, changed, skipped)


def main() -> int:
    files = sorted(glob.glob(os.path.join(PBP_DIR, "*.json")))
    if not files:
        print(f"No PBP files under {PBP_DIR}")
        return 2
    games = hits = changed = skipped = 0
    touched = 0
    for fp in files:
        h, c, s = repair_file(fp)
        if h or s:
            games += 1
        hits += h
        changed += c
        skipped += s
        if c:
            touched += 1
    print(f"2024-25 games with hits: {games}")
    print(f"hits examined: {hits:,} | zoneCodes rewritten: {changed:,} "
          f"({changed/max(hits,1)*100:.1f}%) | skipped (missing fields): {skipped}")
    print(f"game files rewritten: {touched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
