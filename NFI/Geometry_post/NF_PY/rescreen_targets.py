#!/usr/bin/env python3
"""
HockeyROI — Revised acquisition target screening
Fixes: GR removed from screening below 30 reb shots; loosened to ±35% above 30.
       Gap ±0.25s (was 0.20). Rate ±25% (was 20%).
       New tier logic: primary = gap + rate; GR secondary (30+ only).
       Adds sample_confidence: High/Medium/Low.
"""

import csv, json, math, os, time, urllib.request

DATA_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"
BASE_URL = "https://api-web.nhle.com/v1"

# ── New thresholds ──────────────────────────────────────────────────────────
THR_GR_MIN_SHOTS = 30        # GR flag only if reb_shots_2526 >= this
THR_GR   = 0.35              # ±35% of career GR
THR_GAP  = 0.25              # ±0.25s of career gap  (primary)
THR_RATE = 0.25              # ±25% of career rate   (primary)

SEP  = "═" * 112
SEP2 = "─" * 112

TREND_ICON = {"rising": "↑", "declining": "↓", "stable": "—"}

def sample_conf(n):
    if n >= 30:   return "High"
    if n >= 20:   return "Medium"
    if n >= 15:   return "Low"
    return "Insuf"

def fetch_age(pid):
    url = f"{BASE_URL}/player/{int(pid)}/landing"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read())
        bd = d.get("birthDate", "")
        return 2026 - int(bd[:4]) if bd else "?"
    except Exception:
        return "?"

# ── Load career metrics from existing acquisition_targets.csv ───────────────
print("Loading career metrics from acquisition_targets.csv...")
career = {}   # pid → {career_composite, career_gr, career_gap, career_rate,
              #         ci_confirmed, trend, player_name, age, seasons_in_db}

with open(os.path.join(DATA_DIR, "acquisition_targets.csv"), newline="") as f:
    for row in csv.DictReader(f):
        pid = str(row["player_id"])
        career[pid] = {
            "player_name"      : row["player_name"],
            "career_composite" : float(row["career_composite"]),
            "career_gr"        : float(row["career_gr"]) if row["career_gr"] else 0.0,
            "career_gap"       : float(row["career_gap"]) if row["career_gap"] else 0.0,
            "career_rate"      : float(row["career_rate"]) if row["career_rate"] else 0.0,
            "ci_confirmed"     : row["ci_confirmed"] == "True",
            "trend"            : row.get("trend", "stable"),
            "age"              : row.get("age", ""),
            "seasons_in_db"    : row.get("seasons_in_db", ""),
        }

print(f"  {len(career)} players with career metrics")

# ── Load 2025-26 metrics from netfront_2526_rankings.csv ────────────────────
print("Loading 2025-26 metrics from netfront_2526_rankings.csv...")
metrics_2526 = {}   # pid → {reb_shots, goal_rate, avg_gap, nf_rate, team, composite}

with open(os.path.join(DATA_DIR, "netfront_2526_rankings.csv"), newline="") as f:
    for row in csv.DictReader(f):
        pid = str(row["player_id"])
        metrics_2526[pid] = {
            "reb_shots"   : int(row["reb_shots"]),
            "goal_rate"   : float(row["goal_rate"]),
            "avg_gap"     : float(row["avg_gap"]),
            "nf_rate"     : float(row["nf_rate"]),
            "team"        : row["team"],
            "composite_2526": float(row["composite"]),
            "ci_2526"     : row["ci_confirmed"] == "True",
        }

print(f"  {len(metrics_2526)} players with 2025-26 metrics")

# ── Screen every player in career dict ──────────────────────────────────────
print("Applying revised screening logic...")

tier1, tier2, tier3, tier_insuf = [], [], [], []

for pid, c in career.items():
    m = metrics_2526.get(pid)

    # Insufficient sample
    if m is None or m["reb_shots"] < 15:
        tier_insuf.append({**c, "player_id": pid,
                           "reb_shots_2526": m["reb_shots"] if m else 0,
                           "team_2526": m["team"] if m else c.get("age","—"),
                           "sample_conf": "Insuf", "tier": "insufficient_sample",
                           "flags": "—", "n_primary_flags": 0})
        continue

    n_reb = m["reb_shots"]
    sc    = sample_conf(n_reb)

    # ── Compute flags ──────────────────────────────────────────────────────
    # Primary flags (gap + rate)
    gap_diff  = abs(m["avg_gap"] - c["career_gap"])
    gap_flag  = gap_diff > THR_GAP
    gap_dir   = "▲" if m["avg_gap"] > c["career_gap"] else "▼"
    gap_note  = f"gap {m['avg_gap']:.2f}s vs career {c['career_gap']:.2f}s ({gap_dir}{gap_diff:.2f}s)"

    if c["career_rate"] > 0:
        rate_pct  = abs(m["nf_rate"] - c["career_rate"]) / c["career_rate"]
        rate_flag = rate_pct > THR_RATE
        rate_dir  = "▲" if m["nf_rate"] > c["career_rate"] else "▼"
        rate_note = (f"rate {m['nf_rate']:.3f} vs career {c['career_rate']:.3f} "
                     f"({rate_dir}{rate_pct*100:.0f}%)")
    else:
        rate_flag = False
        rate_note = ""

    # Secondary flag (GR — only if 30+ shots)
    if n_reb >= THR_GR_MIN_SHOTS and c["career_gr"] > 0:
        gr_pct   = abs(m["goal_rate"] - c["career_gr"]) / c["career_gr"]
        gr_flag  = gr_pct > THR_GR
        gr_dir   = "▲" if m["goal_rate"] > c["career_gr"] else "▼"
        gr_note  = (f"GR {m['goal_rate']:.3f} vs career {c['career_gr']:.3f} "
                    f"({gr_dir}{gr_pct*100:.0f}%)")
    else:
        gr_flag  = False
        gr_note  = ""

    # ── Tier assignment ────────────────────────────────────────────────────
    n_primary = int(gap_flag) + int(rate_flag)
    gr_active = gr_flag and n_reb >= THR_GR_MIN_SHOTS   # only counts if high-sample

    # Collect flag descriptions
    flag_parts = []
    if gap_flag:  flag_parts.append(gap_note)
    if rate_flag: flag_parts.append(rate_note)
    if gr_active: flag_parts.append(gr_note + " [GR★]")
    flags_str = " | ".join(flag_parts) if flag_parts else "all clear"

    if n_primary == 0 and not gr_active:
        tier = "Tier1_healthy"
    elif n_primary == 1 and not gr_active:
        tier = "Tier2_watch"
    elif gr_active and n_primary == 0:
        # GR flag alone (30+) → Tier 3 per new rules
        tier = "Tier3_declining"
    else:
        # n_primary >= 2, OR (n_primary >= 1 AND gr_active)
        tier = "Tier3_declining"

    record = {
        "player_id"        : pid,
        "player_name"      : c["player_name"],
        "age"              : c["age"],
        "team_2526"        : m["team"],
        "seasons_in_db"    : c["seasons_in_db"],
        "career_composite" : c["career_composite"],
        "career_gr"        : c["career_gr"],
        "career_gap"       : c["career_gap"],
        "career_rate"      : c["career_rate"],
        "ci_confirmed"     : c["ci_confirmed"],
        "trend"            : c["trend"],
        "reb_shots_2526"   : n_reb,
        "gr_2526"          : m["goal_rate"],
        "gap_2526"         : m["avg_gap"],
        "rate_2526"        : m["nf_rate"],
        "composite_2526"   : m["composite_2526"],
        "sample_conf"      : sc,
        "gap_flag"         : gap_flag,
        "rate_flag"        : rate_flag,
        "gr_flag_active"   : gr_active,
        "n_primary_flags"  : n_primary,
        "flags"            : flags_str,
        "tier"             : tier,
    }

    if tier == "Tier1_healthy":     tier1.append(record)
    elif tier == "Tier2_watch":     tier2.append(record)
    else:                           tier3.append(record)

# Sort by career composite
for lst in [tier1, tier2, tier3, tier_insuf]:
    lst.sort(key=lambda x: -x["career_composite"])

print(f"  Tier 1 (buy):        {len(tier1)}")
print(f"  Tier 2 (watch):      {len(tier2)}")
print(f"  Tier 3 (declining):  {len(tier3)}")
print(f"  Insufficient:        {len(tier_insuf)}")

# ── Fetch age for Tier 1 players who are missing it ─────────────────────────
t1_need_age = [r for r in tier1 if not r["age"] or str(r["age"]) == ""]
if t1_need_age:
    print(f"\n  Fetching age for {len(t1_need_age)} Tier 1 players...")
    for r in t1_need_age:
        r["age"] = fetch_age(r["player_id"])
        time.sleep(0.12)

# ── PRINT TIER 1 ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  REVISED ACQUISITION TARGETS — New thresholds: GR ±{int(THR_GR*100)}% (30+ shots only) | "
      f"Gap ±{THR_GAP}s | Rate ±{int(THR_RATE*100)}%")
print(SEP)

print(f"\n  ┌─ TIER 1: BUY — gap + rate healthy, GR healthy where measurable  [{len(tier1)} players] ─┐")
print(f"\n  {'⚠' if any(r['sample_conf']=='Low' for r in tier1) else ' '}"
      f"  Players marked ⚠Low have fewer than 20 rebound shots — treat as provisional.\n")
print(f"  {'Player':<22} {'Age':>4} {'Tm':<5} {'Szns':>5} {'Conf':<7} "
      f"{'Car.Comp':>9}  {'CI':>3}  {'Trend':>10}  "
      f"{'Car.GR':>7} {'2526.GR':>8}  "
      f"{'Car.Gap':>8} {'2526.Gap':>9}  "
      f"{'Car.Rate':>9} {'2526.Rate':>10}  "
      f"{'RbSh':>5}")
print(f"  {SEP2}")

for r in tier1:
    ci  = "✓" if r["ci_confirmed"] else " "
    tr  = TREND_ICON.get(r["trend"],"—") + " " + r["trend"]
    low = "⚠" if r["sample_conf"] == "Low" else " "
    print(f"  {low} {r['player_name']:<22} {str(r['age']):>4} {r['team_2526']:<5} "
          f"{str(r['seasons_in_db']):>5} {r['sample_conf']:<7} "
          f"{r['career_composite']:>9.4f}  {ci:>3}  {tr:<12}  "
          f"{r['career_gr']:>7.4f} {r['gr_2526']:>8.4f}  "
          f"{r['career_gap']:>8.2f}s {r['gap_2526']:>8.2f}s  "
          f"{r['career_rate']:>9.4f} {r['rate_2526']:>10.4f}  "
          f"{r['reb_shots_2526']:>5}")

# ── PRINT TIER 2 — only actionable (Medium/High confidence OR both metrics close) ──
print(f"\n  ┌─ TIER 2: WATCH — one primary flag (gap or rate)  [{len(tier2)} players] ─┐")
print(f"  {'Showing all — sample confidence indicates weight to place on flag.'}")
print(f"\n  {'Player':<22} {'Tm':<5} {'Conf':<7} {'Car.Comp':>9}  {'CI':>3}  {'Trend':>10}  "
      f"{'RbSh':>5}  Flag detail")
print(f"  {SEP2}")

for r in tier2:
    ci  = "✓" if r["ci_confirmed"] else " "
    tr  = TREND_ICON.get(r["trend"],"—") + " " + r["trend"]
    # Highlight Medium/High confidence flags (most actionable)
    weight = "●" if r["sample_conf"] in ("High","Medium") else "○"
    print(f"  {weight} {r['player_name']:<22} {r['team_2526']:<5} {r['sample_conf']:<7} "
          f"{r['career_composite']:>9.4f}  {ci:>3}  {tr:<12}  {r['reb_shots_2526']:>5}  "
          f"{r['flags']}")

# ── PRINT TIER 3 summary ─────────────────────────────────────────────────────
print(f"\n  ┌─ TIER 3: AVOID / SELL — both primary flags or GR flagged at 30+  [{len(tier3)} players] ─┐")
print(f"  {'Player':<22} {'Tm':<5} {'Conf':<7} {'Car.Comp':>9}  {'CI':>3}  {'Trend':>10}  "
      f"{'RbSh':>5}  Flags")
print(f"  {SEP2}")
for r in tier3:
    ci  = "✓" if r["ci_confirmed"] else " "
    tr  = TREND_ICON.get(r["trend"],"—") + " " + r["trend"]
    print(f"  {r['player_name']:<22} {r['team_2526']:<5} {r['sample_conf']:<7} "
          f"{r['career_composite']:>9.4f}  {ci:>3}  {tr:<12}  {r['reb_shots_2526']:>5}  "
          f"{r['flags']}")

# ── SAVE ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, "acquisition_targets.csv")
save_cols = [
    "tier","player_name","player_id","age","team_2526","seasons_in_db",
    "career_composite","career_gr","career_gap","career_rate",
    "composite_2526","gr_2526","gap_2526","rate_2526",
    "reb_shots_2526","sample_conf",
    "ci_confirmed","trend",
    "gap_flag","rate_flag","gr_flag_active","n_primary_flags","flags",
]

all_rows = []
for r in tier1:    all_rows.append({**r, "tier": "Tier1_healthy"})
for r in tier2:    all_rows.append({**r, "tier": "Tier2_watch"})
for r in tier3:    all_rows.append({**r, "tier": "Tier3_declining"})
for r in tier_insuf:
    row = {k: r.get(k,"") for k in save_cols}
    row["tier"] = "insufficient_sample"
    all_rows.append(row)

with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=save_cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(all_rows)

print(f"\n{SEP}")
print(f"  SUMMARY — Revised thresholds")
print(f"  GR flag:   ±{int(THR_GR*100)}% of career, only when reb_shots ≥ {THR_GR_MIN_SHOTS} (was ±25% always)")
print(f"  Gap flag:  ±{THR_GAP}s of career avg (was ±0.20s)")
print(f"  Rate flag: ±{int(THR_RATE*100)}% of career avg (was ±20%)")
print(f"")
print(f"  {'Tier':<28}  {'N':>4}  Notes")
print(f"  {'-'*70}")
print(f"  {'Tier 1 — Buy':<28}  {len(tier1):>4}  "
      f"High/Med: {sum(1 for r in tier1 if r['sample_conf'] in ('High','Medium'))}  "
      f"Low: {sum(1 for r in tier1 if r['sample_conf']=='Low')}")
print(f"  {'Tier 2 — Watch':<28}  {len(tier2):>4}  "
      f"High/Med: {sum(1 for r in tier2 if r['sample_conf'] in ('High','Medium'))}  "
      f"Low: {sum(1 for r in tier2 if r['sample_conf']=='Low')}")
print(f"  {'Tier 3 — Avoid/Sell':<28}  {len(tier3):>4}  "
      f"High/Med: {sum(1 for r in tier3 if r['sample_conf'] in ('High','Medium'))}  "
      f"Low: {sum(1 for r in tier3 if r['sample_conf']=='Low')}")
print(f"  {'Insufficient sample':<28}  {len(tier_insuf):>4}  (<15 reb shots in 20252026)")
print(f"")
print(f"  ● Tier 2 high/medium-confidence flags = most actionable monitoring targets")
print(f"  ○ Tier 2 low-confidence flags = wait for more data")
print(f"  ⚠ Tier 1 low-confidence = provisional buy; re-screen at season end")
print(f"  Saved: {out_path}")
print(SEP + "\n")
