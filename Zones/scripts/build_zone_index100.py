"""Build the 0-100 Zone-Impact index (50 = position-group average) for every
season AND the 4-season pool, for all four zone metrics: OZI, DZI, NZI, TZI.

TZI is the transition split (the metric internally called TNZI: OZ% minus DZ%
off neutral-zone faceoffs). Methodology is reused verbatim from
compute_zone_variations.py (same V1 event set, same faceoff-anchor context walk,
same MIN_SHIFTS / MIN_GP gates) — this script only RESCALES the raw per-shift
percentage onto a 0-100 line:

    raw_pct   = raw_metric * 100                      (raw_metric from compute_metric)
    avg_pct   = mean(raw_pct) over qualifying players in that (season, metric,
                                 position group)
    index     = clip(50 + (raw_pct - avg_pct), 0, 100)

The natural percentage-point spread of the underlying (bounded) stat becomes the
index spread — no artificial stretch — so 50 = league-average, above = good,
below = below-average, mirroring the Quality-Games 50% baseline. Position groups
(forwards C/L/R vs defense D) are normalised independently.

Writes (one row per player, columns OZI/DZI/NZI/TZI on the 0-100 scale):
  Zones/adjusted_rankings/zone_index100/{season}_{forwards|defense}.csv   (per season)
  Zones/adjusted_rankings/zone_index100/pooled_{forwards|defense}.csv     (4-yr pool)
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

# Reuse the published methodology verbatim (import, don't copy).
import compute_zone_variations as zv

METRICS = [("OZI", "OZI"), ("DZI", "DZI"), ("NZI", "NZI"), ("TZI", "TNZI")]
SEASONS = ["20222023", "20232024", "20242025", "20252026", "20262027"]
OUT_DIR = zv.ZONES / "adjusted_rankings" / "zone_index100"


def raw_for_player(data, metric_internal):
    """raw_metric for one player for a V1 metric, or None if not qualified."""
    vb = data["versions"].get("V1", {})
    raw, _adj = zv.compute_metric(vb.get("O"), vb.get("D"), vb.get("N"),
                                  metric_internal, "shifts")
    return raw


def build_index(bundle, player_meta, min_gp=None):
    """bundle -> ({pos_group -> {pid -> {metric_disp: index}}}, {pid -> (meta,pos,gp)}).
    min_gp gates on games played (default zv.MIN_GP=20); pass a lower floor for
    short-sample scopes like playoffs, where the 50-shift gate in compute_metric
    does the real qualifying instead."""
    if min_gp is None:
        min_gp = zv.MIN_GP
    # 1) collect raw_pct per (pos_group, metric_disp) for qualifying players
    raw = {g: defaultdict(dict) for g in ("forwards", "defense")}
    meta_of = {}
    for pid, data in bundle.items():
        meta = player_meta.get(pid, {})
        pos = (meta.get("position") or "").upper()
        if pos in zv.POS_FORWARD:
            grp = "forwards"
        elif pos in zv.POS_DEFENSE:
            grp = "defense"
        else:
            continue
        if data["gp"] < min_gp:
            continue
        for disp, internal in METRICS:
            r = raw_for_player(data, internal)
            if r is not None:
                raw[grp][disp][pid] = r * 100.0
                meta_of[pid] = (meta, pos, data["gp"])
    # 2) recentre each (grp, metric) on its own average -> 0-100 index
    out = {g: {} for g in ("forwards", "defense")}
    avgs = {}
    for grp in ("forwards", "defense"):
        for disp, _ in METRICS:
            pv = raw[grp][disp]
            if not pv:
                continue
            avg = sum(pv.values()) / len(pv)
            avgs[(grp, disp)] = avg
            for pid, rp in pv.items():
                rec = out[grp].setdefault(pid, {})
                rec[disp] = round(max(0.0, min(100.0, 50.0 + (rp - avg))), 1)
    # 3) everyone else who played (below the GP / per-metric shift minimums) is
    #    scored from whatever shifts they have against the SAME qualified average,
    #    so they're in the data without moving anyone's published index. The
    #    app's Min TOI / Min GP sliders decide who is shown and ranked.
    for pid, data in bundle.items():
        meta = player_meta.get(pid, {})
        pos = (meta.get("position") or "").upper()
        grp = ("forwards" if pos in zv.POS_FORWARD else
               "defense" if pos in zv.POS_DEFENSE else None)
        if grp is None or data["gp"] < 1:
            continue
        vb = data["versions"].get("V1", {})
        rec = out[grp].setdefault(pid, {})
        for disp, _ in METRICS:
            if disp in rec or (grp, disp) not in avgs:
                continue
            rp = ungated_raw(vb, disp)
            if rp is None:
                continue
            rec[disp] = round(max(0.0, min(100.0, 50.0 + (rp * 100.0 - avgs[(grp, disp)]))), 1)
        if rec:
            meta_of.setdefault(pid, (meta, pos, data["gp"]))
        else:
            out[grp].pop(pid, None)
    return out, meta_of


def ungated_raw(vb, disp):
    """Raw 0-1 share for one metric with no shift minimum (same definitions as
    compute_metric): OZI/DZI/NZI = OZ time share after O/D/N-zone faceoffs; TZI =
    (OZ - DZ) share after neutral-zone faceoffs."""
    fz = {"OZI": "O", "DZI": "D", "NZI": "N", "TZI": "N"}[disp]
    b = vb.get(fz)
    if not b or b["total_sec"] <= 0:
        return None
    if disp == "TZI":
        return (b["oz_sec"] - b["dz_sec"]) / b["total_sec"]
    return b["oz_sec"] / b["total_sec"]


def write_scope(scope_label, bundle, player_meta, min_gp=None):
    idx, meta_of = build_index(bundle, player_meta, min_gp=min_gp)
    for grp in ("forwards", "defense"):
        rows = []
        for pid, vals in idx[grp].items():
            meta, pos, gp = meta_of[pid]
            rows.append({
                "player_name": meta.get("name", ""),
                "team": zv.norm_team(meta.get("team_abbrev", "") or ""),
                "pos": pos, "GP": gp,
                "OZI": vals.get("OZI", ""), "DZI": vals.get("DZI", ""),
                "NZI": vals.get("NZI", ""), "TZI": vals.get("TZI", ""),
                "player_id": pid,
            })

        def sk(r):
            return (-(r["TZI"] if r["TZI"] != "" else -1),
                    -(r["OZI"] if r["OZI"] != "" else -1))
        rows.sort(key=sk)
        cols = ["player_name", "team", "pos", "GP",
                "OZI", "DZI", "NZI", "TZI", "player_id"]
        path = OUT_DIR / f"{scope_label}_{grp}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r[c] for c in cols])
        print(f"    wrote {path.name}  ({len(rows)} players)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] loading player meta ...")
    player_meta = {int(k): v for k, v in json.load(open(zv.PLAYER_META)).items()}

    print("[2/3] processing raw PBP + shifts per game (V1 methodology) ...")
    games = zv.load_game_ids()
    player_bucket = defaultdict(
        lambda: {"shifts": 0, "lost_shifts": 0, "total_sec": 0.0,
                 "oz_sec": 0.0, "dz_sec": 0.0, "nz_sec": 0.0})
    player_season_gp = defaultdict(int)
    total = len(games)
    for i, (gid, _season) in enumerate(games):
        zv.process_game(gid, player_season_gp, player_bucket)
        if (i + 1) % 1000 == 0 or i + 1 == total:
            print(f"    processed {i+1}/{total}")

    print("[3/3] building + writing per-season and pooled indices ...")
    for season in SEASONS:
        label = {"20222023": "2022-23", "20232024": "2023-24",
                 "20242025": "2024-25", "20252026": "2025-26",
                 "20262027": "2026-27"}[season]
        bundle = zv.build_scenario_rows(player_bucket, player_season_gp,
                                        player_meta, {season})
        write_scope(label, bundle, player_meta)
    bundle = zv.build_scenario_rows(player_bucket, player_season_gp,
                                    player_meta, {"20222023","20232024","20242025","20252026"})
    write_scope("pooled", bundle, player_meta)
    # 2-year pool (2024-25 + 2025-26) — mirrors the app's "2yr (2024–2026)" lens.
    bundle = zv.build_scenario_rows(player_bucket, player_season_gp,
                                    player_meta, {"20242025", "20252026"})
    write_scope("2yr", bundle, player_meta)
    print("done.")


if __name__ == "__main__":
    main()
