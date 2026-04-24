#!/usr/bin/env python3
"""
HockeyROI — Single-player season vs career profile
Player: Andrew Mangiapane  ID: 8478233
"""

import csv, math, os
from collections import defaultdict

DATA_DIR  = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"
PID       = "8478233"
NAME      = "Andrew Mangiapane"
TARGET_S  = "20252026"
ES_CODE   = "1551"
NF_X_MIN  = 74.0;  NF_Y_MIN = -8.0;  NF_Y_MAX = 8.0
SEASONS   = ["20202021","20212022","20222023","20232024","20242025","20252026"]

# ── PASS 1: shot events ───────────────────────────────────────────────────────
# Per-season: es_shots, nf_shots, nf_goals, nf_x_list, nf_y_list
season_data = defaultdict(lambda: {
    "es": 0, "nf": 0, "nf_goals": 0,
    "nf_x": [], "nf_y": [], "team": defaultdict(int)
})

with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["shooter_player_id"] != PID:
            continue
        if row["situation_code"] != ES_CODE:
            continue
        season = row["season"]
        try:
            x = float(row["x_coord_norm"])
            y = float(row["y_coord_norm"])
        except (ValueError, TypeError):
            continue

        is_goal = row["is_goal"] == "1"
        is_nf   = (x >= NF_X_MIN and NF_Y_MIN <= y <= NF_Y_MAX)
        team    = row["shooting_team_abbrev"]

        sd = season_data[season]
        sd["es"] += 1
        sd["team"][team] = sd["team"].get(team, 0) + 1
        if is_nf:
            sd["nf"] += 1
            sd["nf_x"].append(x)
            sd["nf_y"].append(y)
            if is_goal:
                sd["nf_goals"] += 1

# ── PASS 2: rebound sequences (time gap as rebound shooter) ──────────────────
# Per-season: list of time_gap_secs where reb_shooter_id = PID
season_gaps = defaultdict(list)

with open(os.path.join(DATA_DIR, "rebound_sequences.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["reb_shooter_id"] != PID:
            continue
        season = row["season"]
        try:
            gap = float(row["time_gap_secs"])
        except ValueError:
            continue
        season_gaps[season].append(gap)

# ── BUILD PER-SEASON TABLE ────────────────────────────────────────────────────
def safe_div(a, b):
    return a / b if b else 0.0

def fmt_diff(val, ref, pct=False, invert=False):
    """Format delta vs reference, with direction arrow."""
    if ref == 0:
        return "  n/a"
    diff = val - ref
    if invert:
        diff = -diff   # lower is better (time gap)
    arrow = "▲" if diff > 0 else "▼" if diff < 0 else "—"
    if pct:
        return f"  {arrow}{abs(diff)*100:4.1f}pp"
    return f"  {arrow}{abs(diff):.3f}"

rows = []
all_seasons_played = [s for s in SEASONS if season_data[s]["es"] > 0]

for season in SEASONS:
    sd   = season_data[season]
    gaps = season_gaps[season]
    if sd["es"] == 0 and not gaps:
        continue
    team = max(sd["team"], key=sd["team"].get) if sd["team"] else "—"
    nf_rate  = safe_div(sd["nf"],      sd["es"])
    nf_gr    = safe_div(sd["nf_goals"], sd["nf"])
    avg_x    = sum(sd["nf_x"]) / len(sd["nf_x"]) if sd["nf_x"] else float("nan")
    avg_y    = sum(sd["nf_y"]) / len(sd["nf_y"]) if sd["nf_y"] else float("nan")
    avg_gap  = sum(gaps) / len(gaps) if gaps else float("nan")
    rows.append({
        "season": season, "team": team,
        "es": sd["es"], "nf": sd["nf"], "nf_goals": sd["nf_goals"],
        "nf_rate": nf_rate, "nf_gr": nf_gr,
        "avg_x": avg_x, "avg_y": avg_y, "avg_gap": avg_gap,
        "n_reb": len(gaps),
    })

# ── CAREER TOTALS (all 6 seasons) ────────────────────────────────────────────
tot_es    = sum(r["es"]       for r in rows)
tot_nf    = sum(r["nf"]       for r in rows)
tot_nfg   = sum(r["nf_goals"] for r in rows)
all_x     = [x for s in SEASONS for x in season_data[s]["nf_x"]]
all_y     = [y for s in SEASONS for y in season_data[s]["nf_y"]]
all_gaps  = [g for s in SEASONS for g in season_gaps[s]]

car_nf_rate = safe_div(tot_nf, tot_es)
car_nf_gr   = safe_div(tot_nfg, tot_nf)
car_avg_x   = sum(all_x)    / len(all_x)    if all_x    else float("nan")
car_avg_y   = sum(all_y)    / len(all_y)    if all_y    else float("nan")
car_avg_gap = sum(all_gaps) / len(all_gaps) if all_gaps else float("nan")

# ── TARGET SEASON ROW ────────────────────────────────────────────────────────
tgt = next((r for r in rows if r["season"] == TARGET_S), None)

# ── PRINT ─────────────────────────────────────────────────────────────────────
SEP  = "═" * 90
SEP2 = "─" * 90

print(f"\n{SEP}")
print(f"  {NAME}  —  Player Profile  (ES, situation_code=1551)")
print(f"  Player ID: {PID}")
print(SEP)

# Season-by-season table
print(f"\n  {'Season':<10} {'Tm':<5} {'ES att':>7} {'NF att':>7} {'NF rate':>8} "
      f"{'NF goals':>9} {'NF GR':>7} {'Avg X':>7} {'Avg Y':>7} {'Avg Gap':>8} {'Reb shots':>10}")
print(f"  {SEP2}")
for r in rows:
    x_str   = f"{r['avg_x']:7.1f}" if not math.isnan(r["avg_x"])  else "    —  "
    y_str   = f"{r['avg_y']:7.1f}" if not math.isnan(r["avg_y"])  else "    —  "
    gap_str = f"{r['avg_gap']:7.2f}s" if not math.isnan(r["avg_gap"]) else "     —  "
    marker  = "  ◄" if r["season"] == TARGET_S else ""
    print(f"  {r['season']:<10} {r['team']:<5} {r['es']:>7,} {r['nf']:>7} "
          f"{r['nf_rate']:>8.4f} {r['nf_goals']:>9} {r['nf_gr']:>7.4f} "
          f"{x_str} {y_str} {gap_str} {r['n_reb']:>10}{marker}")

# Career row
cx = f"{car_avg_x:7.1f}" if not math.isnan(car_avg_x)  else "    —  "
cy = f"{car_avg_y:7.1f}" if not math.isnan(car_avg_y)  else "    —  "
cg = f"{car_avg_gap:7.2f}s" if not math.isnan(car_avg_gap) else "     —  "
print(f"  {SEP2}")
print(f"  {'CAREER (6 seasons)':<16} {tot_es:>7,} {tot_nf:>7} "
      f"{car_nf_rate:>8.4f} {tot_nfg:>9} {car_nf_gr:>7.4f} "
      f"{cx} {cy} {cg} {len(all_gaps):>10}")

# ── DETAILED 20252026 vs CAREER COMPARISON ────────────────────────────────────
print(f"\n{SEP}")
print(f"  {TARGET_S} vs CAREER AVERAGE — DETAILED COMPARISON")
print(SEP)

if tgt is None:
    print(f"  No data found for {TARGET_S}")
else:
    def row_fmt(label, cur, car, unit="", pct=False, invert=False, higher_better=True):
        if pct:
            cur_s = f"{cur*100:.2f}%"
            car_s = f"{car*100:.2f}%"
        elif unit == "s":
            cur_s = f"{cur:.2f}s"
            car_s = f"{car:.2f}s"
        elif unit == "coord":
            cur_s = f"{cur:.1f}"
            car_s = f"{car:.1f}"
        else:
            cur_s = f"{cur:,.0f}" if cur >= 10 else f"{cur:.4f}"
            car_s = f"{car:,.0f}" if car >= 10 else f"{car:.4f}"

        if car == 0:
            delta_s = "  n/a"
            verdict = ""
        else:
            diff_abs = cur - car
            diff_pct = diff_abs / car * 100
            arrow = "▲" if diff_abs > 0 else "▼" if diff_abs < 0 else "—"
            if pct:
                delta_s = f"  {arrow}{abs(diff_abs)*100:4.1f}pp  ({diff_pct:+.1f}%)"
            elif unit in ("s", "coord"):
                delta_s = f"  {arrow}{abs(diff_abs):.2f}  ({diff_pct:+.1f}%)"
            else:
                delta_s = f"  {arrow}{abs(diff_abs):.1f}  ({diff_pct:+.1f}%)"
            # verdict
            better = (diff_abs > 0) == higher_better
            if abs(diff_pct) < 5:
                verdict = "(≈ same)"
            elif better:
                verdict = "✓ better"
            else:
                verdict = "✗ worse"

        print(f"  {label:<32}  {cur_s:>12}  {car_s:>12}  {delta_s:<28}  {verdict}")

    print(f"\n  {'Metric':<32}  {'20252026':>12}  {'Career avg':>12}  {'Delta':^28}  Verdict")
    print(f"  {'-'*88}")

    row_fmt("ES shot attempts",        tgt["es"],       safe_div(tot_es, len(rows)),
            higher_better=True)
    row_fmt("NF attempts",             tgt["nf"],       safe_div(tot_nf, len(rows)),
            higher_better=True)
    row_fmt("NF attempt rate",         tgt["nf_rate"],  car_nf_rate,
            pct=True, higher_better=True)
    row_fmt("NF goals",                tgt["nf_goals"], safe_div(tot_nfg, len(rows)),
            higher_better=True)
    row_fmt("NF goal rate",            tgt["nf_gr"],    car_nf_gr,
            pct=True, higher_better=True)
    row_fmt("Avg x_coord (crease depth)", tgt["avg_x"] if not math.isnan(tgt["avg_x"]) else 0,
            car_avg_x, unit="coord", higher_better=True)
    row_fmt("Avg y_coord (lateral)",   tgt["avg_y"] if not math.isnan(tgt["avg_y"]) else 0,
            car_avg_y, unit="coord", higher_better=False)
    row_fmt("Avg time gap (secs)",     tgt["avg_gap"] if not math.isnan(tgt["avg_gap"]) else 0,
            car_avg_gap, unit="s", higher_better=False)

    # NF rate vs prior 5 seasons only
    prior_rows = [r for r in rows if r["season"] != TARGET_S]
    if prior_rows:
        prior_es  = sum(r["es"] for r in prior_rows)
        prior_nf  = sum(r["nf"] for r in prior_rows)
        prior_nfg = sum(r["nf_goals"] for r in prior_rows)
        prior_nf_rate = safe_div(prior_nf, prior_es)
        prior_nf_gr   = safe_div(prior_nfg, prior_nf)
        chg_rate = (tgt["nf_rate"] - prior_nf_rate) / prior_nf_rate * 100 if prior_nf_rate else 0
        chg_gr   = (tgt["nf_gr"]   - prior_nf_gr)   / prior_nf_gr   * 100 if prior_nf_gr   else 0
        print(f"\n  Prior-5-season baseline (20202021–20242025):")
        print(f"    NF attempt rate:  {prior_nf_rate:.4f}  →  {tgt['nf_rate']:.4f}  "
              f"({'▲' if chg_rate>0 else '▼'}{abs(chg_rate):.1f}%)")
        print(f"    NF goal rate:     {prior_nf_gr:.4f}  →  {tgt['nf_gr']:.4f}  "
              f"({'▲' if chg_gr>0 else '▼'}{abs(chg_gr):.1f}%)")

print(f"\n{SEP}\n")
