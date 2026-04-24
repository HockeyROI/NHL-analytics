"""Compute OZE / DZE / NZE / DNZE zone-time metrics for all NHL players.

Reads three intermediate files:
  Zones/_zone_time.json   (per player zone time seconds by faceoff type)
  Zones/_player_meta.json (player name, team, position)
  Zones/Data/game_ids.csv (game IDs by season/type — fallback: Data/game_ids.csv)

Writes 6 CSVs (forwards/defense x {regular, pooled, playoffs}).
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ZONE_TIME_PATH = HERE / "_zone_time.json"
PLAYER_META_PATH = HERE / "_player_meta.json"
CORSI_PATH = HERE / "corsi_reference.csv"

# game_ids.csv location — try Zones/Data first, then Data/
_candidate_game_paths = [HERE / "Data" / "game_ids.csv", ROOT / "Data" / "game_ids.csv"]
GAME_IDS_PATH = next((p for p in _candidate_game_paths if p.exists()), _candidate_game_paths[0])

OUTPUT_MAP = {
    "current_regular": ("oze_dze_nze_forwards_regular.csv", "oze_dze_nze_defense_regular.csv"),
    "pooled":          ("oze_dze_nze_forwards_pooled.csv",  "oze_dze_nze_defense_pooled.csv"),
    "current_playoffs":("oze_dze_nze_forwards_playoffs.csv","oze_dze_nze_defense_playoffs.csv"),
}

Z = 1.96
Z2 = Z * Z
MIN_SHIFTS = 50

FORWARD_POS = {"C", "L", "R"}
DEFENSE_POS = {"D"}

OUT_COLS = [
    "player_name", "team", "pos", "GP", "toi_per_game",
    "oz_faceoff_shifts", "dz_faceoff_shifts", "nz_faceoff_shifts",
    "OZE", "DZE", "NZE", "DNZE",
    "OZE_score", "DZE_score", "NZE_score", "DNZE_score",
    "CF_pct",
]


# ---------- Wilson confidence interval ----------

def wilson_lower(num: float, den: float, n_shifts: int) -> float | None:
    """Wilson lower bound. p = num/den (time ratio); n = faceoff shift count."""
    if den is None or den <= 0 or n_shifts is None or n_shifts <= 0:
        return None
    p = num / den
    n = float(n_shifts)
    denom = 1.0 + Z2 / n
    center = p + Z2 / (2.0 * n)
    margin = Z * math.sqrt(max(0.0, p * (1.0 - p) / n + Z2 / (4.0 * n * n)))
    return (center - margin) / denom


def wilson_upper(num: float, den: float, n_shifts: int) -> float | None:
    """Wilson upper bound. p = num/den (time ratio); n = faceoff shift count."""
    if den is None or den <= 0 or n_shifts is None or n_shifts <= 0:
        return None
    p = num / den
    n = float(n_shifts)
    denom = 1.0 + Z2 / n
    center = p + Z2 / (2.0 * n)
    margin = Z * math.sqrt(max(0.0, p * (1.0 - p) / n + Z2 / (4.0 * n * n)))
    return (center + margin) / denom


# ---------- Loaders ----------

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_corsi(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pid = str(row.get("player_id", "")).strip()
            try:
                cf = float(row.get("cfpct_5v5", "") or "nan")
            except ValueError:
                continue
            if pid and not math.isnan(cf):
                out[pid] = cf
    return out


def load_game_counts(path: Path) -> dict[tuple[str, str], int]:
    """Returns {(season, game_type): count}. Useful for context / sanity."""
    counts: dict[tuple[str, str], int] = {}
    if not path.exists():
        return counts
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            season = str(row.get("season", "")).strip()
            gtype = str(row.get("game_type", "")).strip()
            if not season or not gtype:
                continue
            counts[(season, gtype)] = counts.get((season, gtype), 0) + 1
    return counts


# ---------- Metric computation ----------

def compute_row(pid: str, stats: dict, meta: dict, cf_pct: float | None) -> dict | None:
    name = meta.get("name") or ""
    pos = (meta.get("position") or "").upper()
    if pos in ("G", ""):  # skip goalies / unknown
        return None

    oz_fo_shifts = int(stats.get("oz_fo_shifts", 0) or 0)
    dz_fo_shifts = int(stats.get("dz_fo_shifts", 0) or 0)
    nz_fo_shifts = int(stats.get("nz_fo_shifts", 0) or 0)

    oz_fo_total = float(stats.get("oz_fo_5v5_sec", 0) or 0)
    dz_fo_total = float(stats.get("dz_fo_5v5_sec", 0) or 0)
    nz_fo_total = float(stats.get("nz_fo_5v5_sec", 0) or 0)

    oz_fo_oz = float(stats.get("oz_fo_oz_sec", 0) or 0)
    dz_fo_oz = float(stats.get("dz_fo_oz_sec", 0) or 0)
    nz_fo_oz = float(stats.get("nz_fo_oz_sec", 0) or 0)
    nz_fo_dz = float(stats.get("nz_fo_dz_sec", 0) or 0)

    gp = int(stats.get("games_played", 0) or 0)
    toi_sec = float(stats.get("toi_sec", 0) or 0)
    toi_per_game_min = (toi_sec / gp / 60.0) if gp > 0 else 0.0

    # OZE — needs min 50 OZ fo shifts; n = OZ fo shift count
    oze = wilson_lower(oz_fo_oz, oz_fo_total, oz_fo_shifts) if oz_fo_shifts >= MIN_SHIFTS else None
    # DZE — needs min 50 DZ fo shifts; n = DZ fo shift count
    dze = wilson_lower(dz_fo_oz, dz_fo_total, dz_fo_shifts) if dz_fo_shifts >= MIN_SHIFTS else None
    # NZE — needs min 50 NZ fo shifts; n = NZ fo shift count
    nze = wilson_lower(nz_fo_oz, nz_fo_total, nz_fo_shifts) if nz_fo_shifts >= MIN_SHIFTS else None
    # DNZE — Wilson lower OZ% minus Wilson upper DZ% on NZ fo shifts; n = NZ fo shift count for both
    dnze = None
    if nz_fo_shifts >= MIN_SHIFTS:
        lo = wilson_lower(nz_fo_oz, nz_fo_total, nz_fo_shifts)
        hi = wilson_upper(nz_fo_dz, nz_fo_total, nz_fo_shifts)
        if lo is not None and hi is not None:
            dnze = lo - hi

    return {
        "player_id": pid,
        "player_name": name,
        "team": stats.get("team_abbrev") or meta.get("team_abbrev") or "",
        "pos": pos,
        "GP": gp,
        "toi_per_game": round(toi_per_game_min, 3),
        "oz_faceoff_shifts": oz_fo_shifts,
        "dz_faceoff_shifts": dz_fo_shifts,
        "nz_faceoff_shifts": nz_fo_shifts,
        "OZE": oze,
        "DZE": dze,
        "NZE": nze,
        "DNZE": dnze,
        "CF_pct": cf_pct if cf_pct is not None else "",
    }


def normalize_scores(rows: list[dict], metric: str) -> None:
    vals = [r[metric] for r in rows if r[metric] is not None]
    out_key = f"{metric}_score"
    if not vals:
        for r in rows:
            r[out_key] = ""
        return
    lo, hi = min(vals), max(vals)
    span = hi - lo
    for r in rows:
        v = r[metric]
        if v is None:
            r[out_key] = ""
        elif span == 0:
            r[out_key] = 5.0
        else:
            r[out_key] = round(((v - lo) / span) * 10.0, 1)


def fmt(v, digits=4):
    if v is None or v == "":
        return ""
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return v


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OUT_COLS)
        for r in rows:
            w.writerow([
                r["player_name"], r["team"], r["pos"], r["GP"], r["toi_per_game"],
                r["oz_faceoff_shifts"], r["dz_faceoff_shifts"], r["nz_faceoff_shifts"],
                fmt(r["OZE"]), fmt(r["DZE"]), fmt(r["NZE"]), fmt(r["DNZE"]),
                r["OZE_score"], r["DZE_score"], r["NZE_score"], r["DNZE_score"],
                fmt(r["CF_pct"], 4),
            ])


# ---------- Main ----------

def process_scenario(scenario: str, zone_time: dict, player_meta: dict, corsi: dict) -> tuple[list[dict], list[dict]]:
    players = zone_time.get(scenario, {}) or {}
    all_rows: list[dict] = []
    for pid, stats in players.items():
        meta = player_meta.get(str(pid), {})
        row = compute_row(str(pid), stats, meta, corsi.get(str(pid)))
        if row is None:
            continue
        all_rows.append(row)

    forwards = [r for r in all_rows if r["pos"] in FORWARD_POS]
    defense = [r for r in all_rows if r["pos"] in DEFENSE_POS]

    for group in (forwards, defense):
        for m in ("OZE", "DZE", "NZE", "DNZE"):
            normalize_scores(group, m)

    forwards.sort(key=lambda r: r["player_name"])
    defense.sort(key=lambda r: r["player_name"])
    return forwards, defense


def league_averages(rows: list[dict]) -> dict:
    out = {}
    for m in ("OZE", "DZE", "NZE"):
        vals = [r[m] for r in rows if r[m] is not None]
        out[m] = (sum(vals) / len(vals)) if vals else None
    return out


def print_top(rows: list[dict], metric: str, label: str, n: int = 20) -> None:
    ranked = [r for r in rows if r[metric] is not None]
    ranked.sort(key=lambda r: r[metric], reverse=True)
    print(f"\n=== Top {n} {label} by {metric} score ===")
    print(f"{'#':>3} {'Player':<26} {'Team':<4} {'Pos':<3} {'GP':>3} {'TOI/GP':>6} "
          f"{'OZE':>6} {'DZE':>6} {'NZE':>6} {'DNZE':>7} {'Score':>5}")
    score_key = metric.replace("_score", "") if metric.endswith("_score") else f"{metric}_score"
    # metric here is one of the raw metrics; we show its _score
    for i, r in enumerate(ranked[:n], 1):
        print(f"{i:>3} {r['player_name'][:26]:<26} {r['team']:<4} {r['pos']:<3} "
              f"{r['GP']:>3} {r['toi_per_game']:>6.2f} "
              f"{(fmt(r['OZE']) or 0):>6.4f} {(fmt(r['DZE']) or 0):>6.4f} "
              f"{(fmt(r['NZE']) or 0):>6.4f} {(fmt(r['DNZE']) or 0):>7.4f} "
              f"{r[score_key] if r[score_key] != '' else 0:>5}")


def print_row(label: str, rows_f: list[dict], rows_d: list[dict]) -> None:
    for r in rows_f + rows_d:
        if label.lower() in r["player_name"].lower():
            print(f"{r['player_name']:<24} {r['team']:<4} {r['pos']:<3} "
                  f"GP={r['GP']:>3} TOI/GP={r['toi_per_game']:>5.2f} "
                  f"OZfs={r['oz_faceoff_shifts']:>4} DZfs={r['dz_faceoff_shifts']:>4} "
                  f"NZfs={r['nz_faceoff_shifts']:>4} "
                  f"OZE={fmt(r['OZE'])} DZE={fmt(r['DZE'])} NZE={fmt(r['NZE'])} "
                  f"DNZE={fmt(r['DNZE'])} "
                  f"OZE_s={r['OZE_score']} DZE_s={r['DZE_score']} "
                  f"NZE_s={r['NZE_score']} DNZE_s={r['DNZE_score']} "
                  f"CF%={fmt(r['CF_pct'])}")


def main() -> None:
    zone_time = load_json(ZONE_TIME_PATH)
    player_meta = load_json(PLAYER_META_PATH)
    corsi = load_corsi(CORSI_PATH)
    game_counts = load_game_counts(GAME_IDS_PATH)

    print(f"[info] zone_time scenarios: {list(zone_time.keys())}")
    print(f"[info] player_meta players: {len(player_meta)}")
    print(f"[info] corsi entries: {len(corsi)}")
    print(f"[info] game_ids file: {GAME_IDS_PATH}")
    if game_counts:
        by_type: dict = {}
        for (season, gtype), n in game_counts.items():
            by_type.setdefault(gtype, 0)
            by_type[gtype] += n
        print(f"[info] total games by type: {by_type}")

    scenarios_output: dict[str, tuple[list[dict], list[dict]]] = {}
    for scenario, (fwd_name, def_name) in OUTPUT_MAP.items():
        forwards, defense = process_scenario(scenario, zone_time, player_meta, corsi)
        write_csv(HERE / fwd_name, forwards)
        write_csv(HERE / def_name, defense)
        scenarios_output[scenario] = (forwards, defense)
        print(f"[write] {fwd_name}: {len(forwards)} rows  |  {def_name}: {len(defense)} rows")

    # ===== Reporting: use POOLED scenario for headline outputs =====
    forwards_p, defense_p = scenarios_output["pooled"]

    avg_f = league_averages(forwards_p)
    avg_d = league_averages(defense_p)
    print("\n=== League averages (POOLED, Wilson lower bounds) ===")
    print(f"  Forwards  OZE avg = {avg_f['OZE']:.4f}  DZE avg = {avg_f['DZE']:.4f}  NZE avg = {avg_f['NZE']:.4f}")
    print(f"  Defense   OZE avg = {avg_d['OZE']:.4f}  DZE avg = {avg_d['DZE']:.4f}  NZE avg = {avg_d['NZE']:.4f}")

    print_top(forwards_p, "OZE", "forwards")
    print_top(forwards_p, "DZE", "forwards")
    print_top(forwards_p, "NZE", "forwards")
    print_top(forwards_p, "DNZE", "forwards")
    print_top(defense_p, "DNZE", "defensemen")

    targets = [
        "McDavid", "MacKinnon", "Draisaitl", "Makar", "Hughes", "Nurse",
        "Bouchard", "Ekholm", "Martinook", "Hayton", "Bedard", "Kopitar", "Henrique",
    ]
    print("\n=== Specific player rows (POOLED) ===")
    for t in targets:
        print_row(t, forwards_p, defense_p)


if __name__ == "__main__":
    main()
