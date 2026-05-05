"""compute_single_game.py
-------------------------------
Single-game TNZI / TOZI / TDZI pipeline. Mirrors the zone-classification
and faceoff-start logic from compute_tozi_tdzi.py, but operates on one
game and counts EVENTS (not time) — single-game samples are too small for
time-share to be meaningful.

Filter (same as pooled scripts):
  - 5v5 only            : situationCode == "1551"
  - regulation          : periods 1-3 (OT/SO ignored)
  - V1 event set        : faceoff/hit/shot-on-goal/missed-shot/blocked-shot/
                          goal/giveaway/takeaway
  - faceoff-start shift : on-ice at a 5v5 OZ/DZ/NZ faceoff; the FO zone
                          (in player-team perspective via FLIP) tags the shift
  - context closes on   : next faceoff, sit !=1551, period-end, game-end

Outputs:
  Zones/output/single_game/{game_id}_team.csv
  Zones/output/single_game/{game_id}_players.csv

Usage:
  python compute_single_game.py <game_id>
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Reuse constants + helpers verbatim from the pooled TOZI/TDZI script.
from compute_tozi_tdzi import (
    V1_EVENTS, FLIP, POS_F, POS_D,
    mmss, play_t, build_intervals, on_ice, shift_end,
    RAW_PBP, RAW_SHIFTS,
)

HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parent
OUT_DIR = ROOT / "output" / "single_game"

REGULATION_PERIODS = {1, 2, 3}


# ---------------------------------------------------------------------------
def _new_bucket():
    return {"shifts": 0, "oz_ev": 0, "dz_ev": 0, "nz_ev": 0}


def process_game(gid: str, period_filter: set[int] | None = None):
    """Run the FO-context simulation on a single game.

    Returns:
      team_buckets   : {team_id: {"O"/"D"/"N": bucket}}
      player_buckets : {player_id: {"O"/"D"/"N": bucket}}
      meta           : {team_id: abbrev}, {player_id: {name, pos, team_id}}
    """
    pp = RAW_PBP / f"{gid}.json"
    sp = RAW_SHIFTS / f"{gid}.json"
    pbp = json.load(open(pp))
    sj  = json.load(open(sp))

    home = pbp["homeTeam"]["id"]; home_ab = pbp["homeTeam"]["abbrev"]
    away = pbp["awayTeam"]["id"]; away_ab = pbp["awayTeam"]["abbrev"]
    team_ab = {home: home_ab, away: away_ab}

    player_meta = {}
    for r in pbp.get("rosterSpots", []):
        pid = r["playerId"]
        player_meta[pid] = {
            "name": f"{r['firstName']['default']} {r['lastName']['default']}",
            "pos":  r["positionCode"],
            "team_id": r["teamId"],
        }

    intervals = build_intervals(sj)

    team_b   = {home: defaultdict(_new_bucket), away: defaultdict(_new_bucket)}
    player_b = defaultdict(lambda: defaultdict(_new_bucket))

    plays = pbp.get("plays") or []
    ctx = None  # current FO context

    def in_window(p):
        per = (p.get("periodDescriptor") or {}).get("number") or 1
        if period_filter is not None and per not in period_filter:
            return False
        return per in REGULATION_PERIODS

    def emit(ctx, close_t):
        if ctx is None:
            return
        # team-level: one count per team per FO
        for tid in (home, away):
            fz = ctx["fo_zone_home"] if tid == home else FLIP[ctx["fo_zone_home"]]
            tb = team_b[tid][fz]
            tb["shifts"] += 1
            for (t, ezh) in ctx["events"]:
                if not (ctx["fo_t"] <= t < close_t):
                    continue
                ezp = ezh if tid == home else FLIP[ezh]
                if   ezp == "O": tb["oz_ev"] += 1
                elif ezp == "D": tb["dz_ev"] += 1
                else:            tb["nz_ev"] += 1

        # player-level: one count per player on-ice at the FO, clipped to shift end
        for tid, pids in ctx["players_at_fo"].items():
            if tid not in (home, away): continue
            fz = ctx["fo_zone_home"] if tid == home else FLIP[ctx["fo_zone_home"]]
            for pid in pids:
                se = shift_end(intervals, pid, ctx["fo_t"])
                if se is None: continue
                eff = min(close_t, se)
                if eff <= ctx["fo_t"]: continue
                pb = player_b[pid][fz]
                pb["shifts"] += 1
                for (t, ezh) in ctx["events"]:
                    if not (ctx["fo_t"] <= t < eff):
                        continue
                    ezp = ezh if tid == home else FLIP[ezh]
                    if   ezp == "O": pb["oz_ev"] += 1
                    elif ezp == "D": pb["dz_ev"] += 1
                    else:            pb["nz_ev"] += 1

    for p in plays:
        if not in_window(p):
            # close any open context if we leave the window
            if ctx is not None:
                emit(ctx, play_t(p)); ctx = None
            continue

        typ = p.get("typeDescKey") or ""
        det = p.get("details") or {}
        ta  = play_t(p)
        sit = p.get("situationCode") or ""

        if typ == "faceoff":
            if ctx is not None:
                emit(ctx, ta); ctx = None
            if sit != "1551": continue
            zc = det.get("zoneCode")
            if zc not in ("O", "D", "N"): continue
            owner = det.get("eventOwnerTeamId")
            if owner not in (home, away): continue
            fh = zc if owner == home else FLIP[zc]
            oi = on_ice(intervals, ta)
            ctx = {
                "fo_t": ta,
                "fo_zone_home": fh,
                "events": [(ta, fh)],          # the faceoff itself counts as a zone-tag event
                "players_at_fo": {
                    home: set(oi.get(home, set())),
                    away: set(oi.get(away, set())),
                },
            }
            continue

        if ctx is None: continue

        if sit and sit != "1551":
            emit(ctx, ta); ctx = None; continue

        zc = det.get("zoneCode")
        if zc in ("O", "D", "N") and typ in V1_EVENTS:
            owner = det.get("eventOwnerTeamId")
            if owner in (home, away):
                ezh = zc if owner == home else FLIP[zc]
                ctx["events"].append((ta, ezh))

        if typ in ("period-end", "game-end"):
            emit(ctx, ta); ctx = None

    if ctx is not None:
        last = play_t(plays[-1]) if plays else ctx["fo_t"]
        emit(ctx, last)

    return team_b, player_b, team_ab, player_meta


# ---------------------------------------------------------------------------
def metrics_from_bucket(buckets_by_fz: dict) -> dict:
    """Collapse per-FZ buckets into the headline metrics + raw event counts."""
    o = buckets_by_fz.get("O") or _new_bucket()
    d = buckets_by_fz.get("D") or _new_bucket()
    n = buckets_by_fz.get("N") or _new_bucket()

    return {
        # raw counts (across all FO-start shifts of that type)
        "OZ_FO_shifts": o["shifts"], "DZ_FO_shifts": d["shifts"], "NZ_FO_shifts": n["shifts"],
        "OZ_FO_OZev": o["oz_ev"],   "OZ_FO_DZev": o["dz_ev"],   "OZ_FO_NZev": o["nz_ev"],
        "DZ_FO_OZev": d["oz_ev"],   "DZ_FO_DZev": d["dz_ev"],   "DZ_FO_NZev": d["nz_ev"],
        "NZ_FO_OZev": n["oz_ev"],   "NZ_FO_DZev": n["dz_ev"],   "NZ_FO_NZev": n["nz_ev"],
        # totals across all FO shifts
        "OZ_events_total": o["oz_ev"] + d["oz_ev"] + n["oz_ev"],
        "DZ_events_total": o["dz_ev"] + d["dz_ev"] + n["dz_ev"],
        "NZ_events_total": o["nz_ev"] + d["nz_ev"] + n["nz_ev"],
        # headline metrics (event-net, single-game flavor)
        "TOZI": o["oz_ev"] - o["dz_ev"],
        "TDZI": d["oz_ev"] - d["dz_ev"],
        "TNZI": n["oz_ev"] - n["dz_ev"],
    }


# ---------------------------------------------------------------------------
def write_outputs(gid: str, team_b, player_b, team_ab, player_meta):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # team CSV
    team_path = OUT_DIR / f"{gid}_team.csv"
    fields = ["team_id", "team", "OZ_FO_shifts", "DZ_FO_shifts", "NZ_FO_shifts",
              "OZ_events_total", "DZ_events_total", "NZ_events_total",
              "OZ_FO_OZev", "OZ_FO_DZev", "OZ_FO_NZev",
              "DZ_FO_OZev", "DZ_FO_DZev", "DZ_FO_NZev",
              "NZ_FO_OZev", "NZ_FO_DZev", "NZ_FO_NZev",
              "TOZI", "TDZI", "TNZI"]
    with open(team_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for tid, b in team_b.items():
            row = {"team_id": tid, "team": team_ab.get(tid, "")}
            row.update(metrics_from_bucket(b))
            w.writerow(row)

    # player CSV
    player_path = OUT_DIR / f"{gid}_players.csv"
    pfields = ["player_id", "player_name", "team", "pos",
               "OZ_FO_shifts", "DZ_FO_shifts", "NZ_FO_shifts",
               "OZ_events_total", "DZ_events_total", "NZ_events_total",
               "OZ_FO_OZev", "OZ_FO_DZev", "OZ_FO_NZev",
               "DZ_FO_OZev", "DZ_FO_DZev", "DZ_FO_NZev",
               "NZ_FO_OZev", "NZ_FO_DZev", "NZ_FO_NZev",
               "TOZI", "TDZI", "TNZI"]
    with open(player_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pfields); w.writeheader()
        rows = []
        for pid, b in player_b.items():
            m = player_meta.get(pid, {})
            if m.get("pos") == "G":
                continue
            row = {
                "player_id": pid,
                "player_name": m.get("name", ""),
                "team": team_ab.get(m.get("team_id"), ""),
                "pos":  m.get("pos", ""),
            }
            row.update(metrics_from_bucket(b))
            rows.append(row)
        rows.sort(key=lambda r: (r["team"], -r["TNZI"]))
        for r in rows: w.writerow(r)

    return team_path, player_path


# ---------------------------------------------------------------------------
def _print_team_table(team_b, team_ab, label):
    print(f"\n--- TEAM TABLE [{label}] ---")
    hdr = f"{'Team':<5} {'OZ_ev':>6} {'DZ_ev':>6} {'NZ_ev':>6} {'OZ_FO':>5} {'DZ_FO':>5} {'NZ_FO':>5} {'TOZI':>6} {'TDZI':>6} {'TNZI':>6}"
    print(hdr); print("-" * len(hdr))
    for tid, b in team_b.items():
        m = metrics_from_bucket(b)
        print(f"{team_ab.get(tid,''):<5} "
              f"{m['OZ_events_total']:>6} {m['DZ_events_total']:>6} {m['NZ_events_total']:>6} "
              f"{m['OZ_FO_shifts']:>5} {m['DZ_FO_shifts']:>5} {m['NZ_FO_shifts']:>5} "
              f"{m['TOZI']:>+6} {m['TDZI']:>+6} {m['TNZI']:>+6}")


def _print_top_forwards(player_b, team_ab, player_meta, team_id, metric, n=5):
    rows = []
    for pid, b in player_b.items():
        m = player_meta.get(pid, {})
        if m.get("team_id") != team_id: continue
        if m.get("pos") not in POS_F: continue
        mt = metrics_from_bucket(b)
        rows.append((m["name"], mt))
    rows.sort(key=lambda r: r[1][metric], reverse=True)
    abbr = team_ab.get(team_id, "")
    print(f"\n--- Top {n} {abbr} forwards by {metric} ---")
    print(f"{'#':>2} {'Player':<24} {'NZ_FO':>5} {'NZ_FO_OZ':>8} {'NZ_FO_DZ':>8} {metric:>6}")
    for i, (nm, mt) in enumerate(rows[:n], 1):
        print(f"{i:>2} {nm[:24]:<24} {mt['NZ_FO_shifts']:>5} {mt['NZ_FO_OZev']:>8} {mt['NZ_FO_DZev']:>8} {mt[metric]:>+6}")


# ---------------------------------------------------------------------------
def main(gid: str):
    # full-game pass
    team_b, player_b, team_ab, player_meta = process_game(gid)
    team_path, player_path = write_outputs(gid, team_b, player_b, team_ab, player_meta)
    print(f"[write] {team_path}")
    print(f"[write] {player_path}")

    _print_team_table(team_b, team_ab, "FULL GAME (5v5 ES, regulation)")

    home_id = next(t for t, ab in team_ab.items() if ab == "EDM") if "EDM" in team_ab.values() else None
    away_id = next(t for t, ab in team_ab.items() if ab == "ANA") if "ANA" in team_ab.values() else None
    if home_id and away_id:
        _print_top_forwards(player_b, team_ab, player_meta, home_id, "TNZI")
        _print_top_forwards(player_b, team_ab, player_meta, away_id, "TNZI")

        edm = metrics_from_bucket(team_b[home_id])
        print(f"\n--- EDM TDZI inspection ---")
        print(f"  EDM TDZI = {edm['TDZI']:+d}  (OZ events on DZ-FO shifts: {edm['DZ_FO_OZev']}, "
              f"DZ events on DZ-FO shifts: {edm['DZ_FO_DZev']}, DZ-FO shifts: {edm['DZ_FO_shifts']})")
        if edm["TDZI"] > 0:
            print("  -> positive: EDM net-pushed play out of their own zone after DZ FOs.")
        elif edm["TDZI"] < 0:
            print("  -> negative: EDM stayed pinned / sat back after DZ FOs.")
        else:
            print("  -> neutral.")

    # period-by-period split
    print("\n--- PERIOD SPLITS (team) ---")
    for per in (1, 2, 3):
        tb_p, _, _, _ = process_game(gid, period_filter={per})
        _print_team_table(tb_p, team_ab, f"Period {per}")

    print("\n" + "=" * 78)
    print("REMINDER: This is ONE game. Single-game zone-event metrics are")
    print("descriptive of what happened on the ice tonight, NOT predictive of")
    print("team identity. TNZI mean-reverts strongly year over year, and")
    print("system / matchup effects (e.g. Carolina-style forecheck) are real.")
    print("Frame these numbers as 'what happened in this game,' not 'what this")
    print("team is.'")
    print("=" * 78)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python compute_single_game.py <game_id>"); sys.exit(2)
    main(sys.argv[1])
