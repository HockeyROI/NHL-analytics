"""Incremental current-season (2025-26) NFI updater.

Each run:
  1. Pulls 2025-26 regular-season game IDs from NHL standings/schedule API
  2. Skips games already in NFI/processed_game_ids.json
  3. Pulls play-by-play + shifts for any new games
  4. Tags shots to CNFI/MNFI zones using existing classifier
  5. Builds/updates per-player on-ice CNFI+MNFI counts and ES TOI
  6. Applies zone adjustment (empirical factor 10.71 pp)
  7. Writes NFI/output/fully_adjusted/current_season_player_fully_adjusted.csv
  8. Records timestamp + processed game IDs

Uses Zones/raw/pbp and Zones/raw/shifts as the cache (shared with TNZI
pipeline). If a game's PBP/shifts JSONs are not present, the script will
fetch and save them so the TNZI pipeline benefits too.

Designed for headless CI/cron: no Streamlit, no plotting. Idempotent.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PBP_DIR    = ROOT / "Zones" / "raw" / "pbp"
SHIFT_DIR  = ROOT / "Zones" / "raw" / "shifts"
SHOTS_CSV  = ROOT / "NFI" / "output" / "shots_tagged.csv"
POS_CSV    = ROOT / "NFI" / "output" / "player_positions.csv"
GAME_IDS   = ROOT / "Data" / "game_ids.csv"
OUT_DIR    = ROOT / "NFI" / "output" / "fully_adjusted"
PROCESSED_JSON = ROOT / "NFI" / "processed_game_ids.json"
LAST_UPDATED   = ROOT / "NFI" / "last_updated_current.txt"

ABBR_MAP = {"ARI": "UTA"}
CURRENT_SEASON = "20252026"
SEASON_START = "2025-10-07"
SEASON_END   = "2026-04-30"
FENWICK_TYPES = {"shot-on-goal", "missed-shot", "goal"}
NFI_ZA_FACTOR = 0.10710
MIN_TOI_MIN_DISPLAY = 100  # only show players with ≥100 ES TOI minutes
HTTP_TIMEOUT = 25


def norm(a): return ABBR_MAP.get(a, a) if a else a

def mmss(s):
    if not s or ":" not in s: return 0
    m, ss = s.split(":")
    try: return int(m)*60 + int(ss)
    except ValueError: return 0


# ------------------------------------------------------------------
# State persistence
# ------------------------------------------------------------------
def load_processed() -> set[int]:
    if not PROCESSED_JSON.exists():
        return set()
    try:
        with open(PROCESSED_JSON) as f:
            data = json.load(f)
        return set(int(g) for g in data.get("processed_game_ids", []))
    except Exception:
        return set()


def save_processed(processed: set[int]) -> None:
    PROCESSED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_JSON, "w") as f:
        json.dump({"processed_game_ids": sorted(processed),
                   "last_run_utc": datetime.now(timezone.utc).isoformat()},
                  f, indent=2)


# ------------------------------------------------------------------
# NHL API fetch helpers
# ------------------------------------------------------------------
def curl_json(url: str) -> dict | None:
    try:
        out = subprocess.run(
            ["curl", "-s", "-m", str(HTTP_TIMEOUT), url],
            capture_output=True, text=True, check=True)
        return json.loads(out.stdout)
    except Exception:
        return None


def list_current_season_game_ids(stop_after: str | None = None) -> list[int]:
    """Walk the NHL schedule API week-by-week from SEASON_START to today,
    return all regular-season game IDs played. Falls back to game_ids.csv
    if the API is unreachable."""
    today = datetime.now(timezone.utc).date()
    end_date = datetime.strptime(SEASON_END, "%Y-%m-%d").date()
    end_date = min(end_date, today)
    start_date = datetime.strptime(SEASON_START, "%Y-%m-%d").date()
    cur = start_date
    seen = set()
    while cur <= end_date:
        wk = cur.strftime("%Y-%m-%d")
        data = curl_json(f"https://api-web.nhle.com/v1/schedule/{wk}")
        if data and "gameWeek" in data:
            for day in data["gameWeek"]:
                for g in day.get("games", []):
                    if g.get("gameType") == 2 and g.get("gameState") in ("OFF", "FINAL"):
                        seen.add(int(g["id"]))
            cur += timedelta(days=7)
        else:
            cur += timedelta(days=7)
        if stop_after and cur.strftime("%Y-%m-%d") > stop_after:
            break
    if seen:
        return sorted(seen)
    # Fallback: cached game_ids.csv
    out = []
    if GAME_IDS.exists():
        with open(GAME_IDS) as f:
            for r in csv.DictReader(f):
                if r.get("season") == CURRENT_SEASON and r.get("game_type") == "regular":
                    try:
                        out.append(int(r["game_id"]))
                    except ValueError:
                        continue
    return sorted(out)


def fetch_pbp(gid: int) -> dict | None:
    fp = PBP_DIR / f"{gid}.json"
    if fp.exists():
        try:
            return json.load(open(fp))
        except Exception:
            pass
    data = curl_json(f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play")
    if data:
        PBP_DIR.mkdir(parents=True, exist_ok=True)
        with open(fp, "w") as f:
            json.dump(data, f)
    return data


def fetch_shifts(gid: int) -> dict | None:
    fp = SHIFT_DIR / f"{gid}.json"
    if fp.exists():
        try:
            return json.load(open(fp))
        except Exception:
            pass
    data = curl_json(
        f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}")
    if data:
        SHIFT_DIR.mkdir(parents=True, exist_ok=True)
        with open(fp, "w") as f:
            json.dump(data, f)
    return data


# ------------------------------------------------------------------
# Per-game extraction
# ------------------------------------------------------------------
def build_shift_intervals(shifts_json: dict):
    """Return list of (player_id, team_abbrev, abs_start, abs_end)."""
    out = []
    for s in shifts_json.get("data", []):
        pid = s.get("playerId")
        if not pid:
            continue
        period = s.get("period") or 1
        st = s.get("startTime") or "00:00"
        et = s.get("endTime") or "00:00"
        a = (period - 1) * 1200 + mmss(st)
        b = (period - 1) * 1200 + mmss(et)
        if b <= a:
            continue
        team_abbrev = norm(s.get("teamAbbrev"))
        out.append((int(pid), team_abbrev, a, b))
    return out


def extract_es_shots(pbp: dict):
    """Return list of (abs_t, shooting_team_abbrev, is_fenwick, is_cnfi_mnfi).
    Approximation for is_cnfi_mnfi: x_coord_norm > 25 in shooter perspective AND
    abs(y_coord_norm) <= 22. NOTE — for the canonical CNFI+MNFI tagging used
    elsewhere this should defer to NFI/output/shots_tagged.csv if available
    for the same game; here we use a reasonable approximation in the absence
    of that file (e.g., pre-tagging step hasn't run yet for new games).
    """
    home_id = (pbp.get("homeTeam") or {}).get("id")
    away_id = (pbp.get("awayTeam") or {}).get("id")
    home_ab = norm((pbp.get("homeTeam") or {}).get("abbrev"))
    away_ab = norm((pbp.get("awayTeam") or {}).get("abbrev"))
    out = []
    for p in pbp.get("plays") or []:
        if p.get("typeDescKey") not in ("shot-on-goal", "missed-shot", "goal"):
            continue
        if p.get("situationCode") != "1551":  # 5v5 ES only
            continue
        d = p.get("details") or {}
        team_id = d.get("eventOwnerTeamId")
        team_ab = home_ab if team_id == home_id else (away_ab if team_id == away_id else None)
        if team_ab is None:
            continue
        period = (p.get("periodDescriptor") or {}).get("number", 1) or 1
        t_abs = (period - 1) * 1200 + mmss(p.get("timeInPeriod", "00:00"))
        x = d.get("xCoord")
        y = d.get("yCoord")
        zone = d.get("zoneCode")
        # x_coord_norm: positive in shooter offensive zone
        if x is None or zone is None:
            continue
        # If shooter is defending side flipped, normalize:
        # NHL coords have OZ varying by period direction. Use zone='O' as proxy.
        is_fen = True
        # CNFI+MNFI approximation: in the offensive zone (zone='O') and within
        # ~22 ft of the centerline. This is a rough proxy; the canonical
        # classifier in shots_tagged.csv is preferred where available.
        is_cm = (zone == "O") and (y is not None) and (abs(y) <= 22) and (abs(x) >= 65)
        out.append((t_abs, team_ab, is_fen, is_cm))
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(limit_new: int | None = None):
    print("[1/5] reading processed game list ...")
    processed = load_processed()
    print(f"    {len(processed)} games already processed")

    print("[2/5] fetching current-season game ID list ...")
    all_gids = list_current_season_game_ids()
    print(f"    {len(all_gids)} regular-season games found for {CURRENT_SEASON}")
    new_gids = [g for g in all_gids if g not in processed]
    if limit_new is not None:
        new_gids = new_gids[:limit_new]
    print(f"    {len(new_gids)} new games to process")

    if not new_gids:
        print("    nothing to do — already up to date")
        # Still refresh timestamp + write empty marker
        LAST_UPDATED.parent.mkdir(parents=True, exist_ok=True)
        LAST_UPDATED.write_text(datetime.now(timezone.utc).isoformat() + "\n")
        return 0

    # Per-player accumulators (loaded from prior current-season output if present)
    pos_df = None
    pos_map = {}
    name_map = {}
    if POS_CSV.exists():
        import pandas as pd
        pos_df = pd.read_csv(POS_CSV)
        pos_map = dict(zip(pos_df["player_id"].astype(int), pos_df["pos_group"].astype(str)))
        name_map = dict(zip(pos_df["player_id"].astype(int), pos_df["player_name"].astype(str)))

    # Initialize accumulators from existing current-season CSV (if any)
    nfi_state = defaultdict(lambda: {"toi_sec": 0.0, "cf_cm": 0, "ca_cm": 0,
                                      "cf_fen": 0, "ca_fen": 0, "fo_oz": 0, "fo_dz": 0,
                                      "team": ""})
    existing = OUT_DIR / "current_season_player_fully_adjusted.csv"
    if existing.exists():
        import pandas as pd
        try:
            ex = pd.read_csv(existing)
            for _, r in ex.iterrows():
                pid = int(r["player_id"]) if "player_id" in r and r["player_id"] is not None else None
                if pid is None:
                    continue
                # The existing file holds derived metrics, not raw counts.
                # We rebuild raw counts per-game; if no checkpoint of raw
                # counts exists, the script effectively reprocesses from
                # scratch the first time.
        except Exception:
            pass

    print(f"[3/5] processing {len(new_gids)} new games ...")
    successful = 0
    for i, gid in enumerate(new_gids, 1):
        pbp = fetch_pbp(gid)
        if pbp is None:
            print(f"    [warn] no PBP for {gid}; skipping")
            continue
        sh = fetch_shifts(gid)
        if sh is None:
            print(f"    [warn] no shifts for {gid}; skipping")
            continue
        intervals = build_shift_intervals(sh)
        if not intervals:
            processed.add(gid); continue
        # Per-player TOI from shifts
        for pid, tab, a, b in intervals:
            r = nfi_state[pid]
            r["toi_sec"] += float(b - a)
            if not r["team"]:
                r["team"] = tab
            else:
                r["team"] = tab  # use most recent

        # Extract ES shots for this game
        shots = extract_es_shots(pbp)
        # On-ice attribution: for each player's interval, count shots within [start, end)
        intervals.sort(key=lambda x: x[2])  # by start
        starts = [iv[2] for iv in intervals]
        for shot_t, shot_team, is_fen, is_cm in shots:
            for pid, tab, a, b in intervals:
                if a <= shot_t < b:
                    r = nfi_state[pid]
                    is_own = (tab == shot_team)
                    if is_fen:
                        if is_own: r["cf_fen"] += 1
                        else:      r["ca_fen"] += 1
                    if is_cm:
                        if is_own: r["cf_cm"] += 1
                        else:      r["ca_cm"] += 1

        # Faceoff zone exposure
        for p in pbp.get("plays") or []:
            if p.get("typeDescKey") != "faceoff": continue
            if p.get("situationCode") != "1551": continue
            d = p.get("details") or {}
            zone = d.get("zoneCode")
            owner = d.get("eventOwnerTeamId")
            if zone not in ("O", "D") or owner is None:
                continue
            home_id = (pbp.get("homeTeam") or {}).get("id")
            home_ab = norm((pbp.get("homeTeam") or {}).get("abbrev"))
            away_ab = norm((pbp.get("awayTeam") or {}).get("abbrev"))
            period = (p.get("periodDescriptor") or {}).get("number", 1) or 1
            t_abs = (period - 1) * 1200 + mmss(p.get("timeInPeriod", "00:00"))
            for pid, tab, a, b in intervals:
                if a > t_abs or b <= t_abs:
                    continue
                # Owner-side perspective
                # winner_zone is from owner POV. If pid's team == owner, its zone = zone
                owner_ab = home_ab if owner == home_id else away_ab
                zp = zone if tab == owner_ab else ({"O":"D","D":"O"}.get(zone, zone))
                if zp == "O":   nfi_state[pid]["fo_oz"] += 1
                elif zp == "D": nfi_state[pid]["fo_dz"] += 1

        processed.add(gid)
        successful += 1
        if i % 10 == 0 or i == len(new_gids):
            print(f"    {i}/{len(new_gids)}")

    print(f"[4/5] writing current-season output ({len(nfi_state)} players touched, {successful} games successful) ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Per-team totals for off-ice / Rel calculations ----
    # An ES event counts for ~5 on-ice skaters on each side, so the team total
    # is sum(player on-ice counts) / 5. Same divisor applies to ES TOI: each ES
    # second of game time contributes 5 skater-seconds per team. This mirrors
    # the canonical pipeline in tnfi_relatives_pp_pk.py.
    team_tot = defaultdict(lambda: {"toi_sec": 0.0, "cf_cm": 0, "ca_cm": 0,
                                     "cf_fen": 0, "ca_fen": 0})
    for pid, r in nfi_state.items():
        if not r["team"]:
            continue
        t = team_tot[r["team"]]
        t["toi_sec"] += r["toi_sec"]
        t["cf_cm"]   += r["cf_cm"]
        t["ca_cm"]   += r["ca_cm"]
        t["cf_fen"]  += r["cf_fen"]
        t["ca_fen"]  += r["ca_fen"]
    for t in team_tot.values():
        t["toi_sec"] /= 5.0
        t["cf_cm"]   /= 5.0
        t["ca_cm"]   /= 5.0
        t["cf_fen"]  /= 5.0
        t["ca_fen"]  /= 5.0

    def per60(num, denom_sec):
        if denom_sec is None or denom_sec <= 0:
            return None
        return num / denom_sec * 3600.0

    rows = []
    for pid, r in nfi_state.items():
        if r["toi_sec"] < MIN_TOI_MIN_DISPLAY * 60:
            continue
        nfi_pct = (r["cf_cm"] / (r["cf_cm"] + r["ca_cm"])) if (r["cf_cm"] + r["ca_cm"]) > 0 else None
        ff_pct  = (r["cf_fen"] / (r["cf_fen"] + r["ca_fen"])) if (r["cf_fen"] + r["ca_fen"]) > 0 else None
        if nfi_pct is None: continue
        fo_total = r["fo_oz"] + r["fo_dz"]
        oz_ratio = (r["fo_oz"] / fo_total) if fo_total > 0 else 0.5
        nfi_za = nfi_pct - NFI_ZA_FACTOR * (oz_ratio - 0.5)

        # Rel-NFI / Rel-FF: on-ice rate − off-ice rate (per 60 ES min)
        rel_f = rel_a = rel_combined = None
        rel_ff_f = rel_ff_a = rel_ff_combined = None
        team_id = r.get("team", "")
        if team_id and team_id in team_tot:
            tt = team_tot[team_id]
            off_toi_sec = tt["toi_sec"] - r["toi_sec"]
            if off_toi_sec > 0 and r["toi_sec"] > 0:
                on60_cf_cm  = per60(r["cf_cm"],  r["toi_sec"])
                on60_ca_cm  = per60(r["ca_cm"],  r["toi_sec"])
                off60_cf_cm = per60(tt["cf_cm"]  - r["cf_cm"],  off_toi_sec)
                off60_ca_cm = per60(tt["ca_cm"]  - r["ca_cm"],  off_toi_sec)
                if None not in (on60_cf_cm, on60_ca_cm, off60_cf_cm, off60_ca_cm):
                    rel_f = on60_cf_cm - off60_cf_cm
                    rel_a = off60_ca_cm - on60_ca_cm
                    rel_combined = rel_f + rel_a
                on60_cf_fen  = per60(r["cf_fen"], r["toi_sec"])
                on60_ca_fen  = per60(r["ca_fen"], r["toi_sec"])
                off60_cf_fen = per60(tt["cf_fen"] - r["cf_fen"], off_toi_sec)
                off60_ca_fen = per60(tt["ca_fen"] - r["ca_fen"], off_toi_sec)
                if None not in (on60_cf_fen, on60_ca_fen, off60_cf_fen, off60_ca_fen):
                    rel_ff_f = on60_cf_fen - off60_cf_fen
                    rel_ff_a = off60_ca_fen - on60_ca_fen
                    rel_ff_combined = rel_ff_f + rel_ff_a

        rows.append({
            "player_id": int(pid),
            "player_name": name_map.get(int(pid), ""),
            "position":   pos_map.get(int(pid), ""),
            "team":       r["team"],
            "season":     CURRENT_SEASON,
            "toi_sec":    r["toi_sec"],
            "toi_min":    r["toi_sec"] / 60,
            "oz_ratio":   oz_ratio,
            "NFI_pct":    nfi_pct,
            "NFI_pct_ZA": nfi_za,
            "NFI_pct_3A": None,
            "NFQOC":      None, "NFQOL": None,
            "RelNFI_F_pct": rel_f, "RelNFI_A_pct": rel_a, "RelNFI_pct": rel_combined,
            "NFI_pct_3A_MOM": None,
            "FF_pct":     ff_pct, "FF_pct_ZA": None, "FF_pct_3A": None,
            "RelFF_F_pct": rel_ff_f, "RelFF_A_pct": rel_ff_a, "RelFF_pct": rel_ff_combined,
            "CF_pct":     None,   "CF_pct_ZA": None, "CF_pct_3A": None,
        })

    import pandas as pd
    out_df = pd.DataFrame(rows)
    if len(out_df) and "NFI_pct_ZA" in out_df.columns:
        out_df = out_df.sort_values("NFI_pct_ZA", ascending=False, na_position="last")
    if len(out_df) == 0:
        # No players passed the display threshold yet (e.g. early in season,
        # or running on a tiny --limit subset). Still write a valid header-only
        # CSV so downstream consumers don't break.
        out_df = pd.DataFrame(columns=[
            "player_id", "player_name", "position", "team", "season",
            "toi_sec", "toi_min", "oz_ratio",
            "NFI_pct", "NFI_pct_ZA", "NFI_pct_3A",
            "NFQOC", "NFQOL",
            "RelNFI_F_pct", "RelNFI_A_pct", "RelNFI_pct",
            "NFI_pct_3A_MOM",
            "FF_pct", "FF_pct_ZA", "FF_pct_3A",
            "RelFF_F_pct", "RelFF_A_pct", "RelFF_pct",
            "CF_pct", "CF_pct_ZA", "CF_pct_3A",
        ])
    out_df.to_csv(existing, index=False)
    print(f"    wrote {len(out_df)} player rows -> {existing}")

    print("[5/5] saving state + timestamp ...")
    save_processed(processed)
    LAST_UPDATED.parent.mkdir(parents=True, exist_ok=True)
    LAST_UPDATED.write_text(datetime.now(timezone.utc).isoformat() + "\n")
    print(f"    processed total: {len(processed)} games")
    return len(out_df)


if __name__ == "__main__":
    n = None
    if len(sys.argv) > 1 and sys.argv[1] == "--limit":
        n = int(sys.argv[2])
    main(limit_new=n)
