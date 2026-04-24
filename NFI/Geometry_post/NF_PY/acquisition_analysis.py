#!/usr/bin/env python3
"""
HockeyROI — Two analyses
  Analysis 1: 2025-26 season net-front rankings (min 15 reb shots)
  Analysis 2: Acquisition target screening against 6-season elite composite

Outputs:
  Data/netfront_2526_rankings.csv
  Data/acquisition_targets.csv
"""

import csv, json, math, os, time, urllib.request
from collections import defaultdict

DATA_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data"
BASE_URL = "https://api-web.nhle.com/v1"

ES_CODE   = "1551"
TARGET_S  = "20252026"
ALL_S     = {"20202021","20212022","20222023","20232024","20242025","20252026"}
NF_X_MIN  = 74.0;  NF_Y_MIN = -8.0;  NF_Y_MAX = 8.0
IDEAL_X   = 80.0;  IDEAL_Y  = 0.0
Z95       = 1.96
MIN_REB_2526    = 15
W_GR, W_GAP, W_POS = 0.40, 0.30, 0.30

# Acquisition health thresholds
THR_GR   = 0.25   # ±25% of career NF goal rate
THR_GAP  = 0.20   # ±0.20s of career avg time gap
THR_RATE = 0.20   # ±20% of career NF attempt rate

SEP  = "═" * 110
SEP2 = "─" * 110

def wilson_ci(k, n, z=Z95):
    if n == 0: return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return p, max(0.0, c - m), min(1.0, c + m)

def minmax_norm(vals, invert=False):
    mn, mx = min(vals), max(vals)
    rng = mx - mn
    if rng == 0: return [0.5]*len(vals)
    out = [(v-mn)/rng for v in vals]
    return [1-v for v in out] if invert else out

def fetch_player_api(pid):
    url = f"{BASE_URL}/player/{int(pid)}/landing"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read())
        fn  = d.get("firstName",{}).get("default","")
        ln  = d.get("lastName", {}).get("default","")
        bd  = d.get("birthDate","")
        age = 2026 - int(bd[:4]) if bd else "?"
        return {"name": f"{fn} {ln}".strip(), "age": age,
                "team": d.get("currentTeamAbbrev","—")}
    except Exception:
        return {"name": f"ID:{pid}", "age": "?", "team": "—"}


# ════════════════════════════════════════════════════════════════════════
#  PHASE 0 — Single pass through shot events
# ════════════════════════════════════════════════════════════════════════
print("Phase 0: Reading nhl_shot_events.csv...")

# Per (pid, season): es, nf, nf_goals, nf_x[], nf_y[]
es_shots  = defaultdict(int)
nf_shots  = defaultdict(int)
nf_goals  = defaultdict(int)
nf_x_acc  = defaultdict(list)
nf_y_acc  = defaultdict(list)
pid_team  = defaultdict(lambda: defaultdict(dict))   # pid → season → {team:count}
pid_name  = {}   # fallback names from prior positioning file

# Load cached names
pos_file = os.path.join(DATA_DIR, "player_rebound_positioning.csv")
if os.path.exists(pos_file):
    with open(pos_file, newline="") as f:
        for row in csv.DictReader(f):
            p = str(row["shooter_player_id"])
            n = row["player_name"]
            if n and not n.startswith("ID:"):
                pid_name[p] = n

with open(os.path.join(DATA_DIR, "nhl_shot_events.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["situation_code"] != ES_CODE:
            continue
        pid    = row["shooter_player_id"]
        season = row["season"]
        team   = row["shooting_team_abbrev"]
        if not pid or not team:
            continue
        try:
            x = float(row["x_coord_norm"])
            y = float(row["y_coord_norm"])
        except (ValueError, TypeError):
            continue
        is_goal = row["is_goal"] == "1"
        is_nf   = (x >= NF_X_MIN and NF_Y_MIN <= y <= NF_Y_MAX)
        key = (pid, season)
        es_shots[key] += 1
        pid_team[pid][season][team] = pid_team[pid][season].get(team, 0) + 1
        if is_nf:
            nf_shots[key] += 1
            nf_x_acc[key].append(x)
            nf_y_acc[key].append(y)
            if is_goal:
                nf_goals[key] += 1

print(f"  Shot events loaded.")

# ════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Rebound sequences: 2526 only + career per player
# ════════════════════════════════════════════════════════════════════════
print("Phase 1: Reading rebound_sequences.csv...")

gap_2526   = defaultdict(list)   # pid → [gap, ...]  for 2526
gap_career = defaultdict(list)   # pid → all gaps all seasons

with open(os.path.join(DATA_DIR, "rebound_sequences.csv"), newline="") as f:
    for row in csv.DictReader(f):
        pid = row["reb_shooter_id"]
        if not pid:
            continue
        try:
            gap = float(row["time_gap_secs"])
        except ValueError:
            continue
        season = row["season"]
        gap_career[pid].append(gap)
        if season == TARGET_S:
            gap_2526[pid].append(gap)

print(f"  Sequences loaded.")


# ════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1 — 2025-26 rankings
# ════════════════════════════════════════════════════════════════════════
print(f"\n── Analysis 1: Building {TARGET_S} rankings ──")

# All unique PIDs with any ES data in 2526
pids_2526 = {pid for (pid, s) in nf_shots if s == TARGET_S}

# Also include anyone with gap data in 2526 (might have reb shots but no NF shots recorded)
pids_2526 |= set(gap_2526.keys())

candidates = []
for pid in pids_2526:
    key   = (pid, TARGET_S)
    n_reb = len(gap_2526[pid])
    if n_reb < MIN_REB_2526:
        continue
    es    = es_shots.get(key, 0)
    nf    = nf_shots.get(key, 0)
    ng    = nf_goals.get(key, 0)
    xlist = nf_x_acc.get(key, [])
    ylist = nf_y_acc.get(key, [])
    gaps  = gap_2526[pid]

    nf_rate  = nf / es if es else 0.0
    gr, ci_lo, ci_hi = wilson_ci(ng, n_reb)
    avg_x = sum(xlist)/len(xlist) if xlist else float("nan")
    avg_y = sum(ylist)/len(ylist) if ylist else float("nan")
    avg_gap = sum(gaps)/len(gaps)
    pos_dist = math.sqrt((avg_x-IDEAL_X)**2 + (avg_y-IDEAL_Y)**2) if not math.isnan(avg_x) else float("nan")

    team_2526 = max(pid_team[pid].get(TARGET_S,{"—":0}), key=pid_team[pid].get(TARGET_S,{"—":0}).get)
    name = pid_name.get(pid, f"ID:{pid}")

    candidates.append({
        "player_id": pid, "player_name": name, "team": team_2526,
        "es": es, "nf": nf, "nf_rate": nf_rate,
        "nf_goals": ng, "reb_shots": n_reb,
        "goal_rate": gr, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "avg_x": avg_x, "avg_y": avg_y, "avg_gap": avg_gap,
        "pos_dist": pos_dist,
    })

# League avg rebound goal rate for 2526 (for CI-confirmed)
total_reb_2526 = sum(len(g) for g in gap_2526.values())
total_reb_goals_2526 = sum(
    nf_goals.get((pid, TARGET_S), 0) for pid in gap_2526
)
# Better: compute from rebound_sequences season=20252026 reb_is_goal
reb_goals_count = 0
reb_shots_count = 0
with open(os.path.join(DATA_DIR, "rebound_sequences.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["season"] == TARGET_S:
            reb_shots_count += 1
            if row["reb_is_goal"] == "1":
                reb_goals_count += 1
league_gr_2526 = reb_goals_count / reb_shots_count if reb_shots_count else 0.0
print(f"  2526 league rebound goal rate: {league_gr_2526:.4f}  "
      f"({reb_goals_count}/{reb_shots_count:,} sequences)")
print(f"  Players with ≥{MIN_REB_2526} reb shots in {TARGET_S}: {len(candidates)}")

# Drop rows with nan pos_dist before normalization
valid = [c for c in candidates if not math.isnan(c["pos_dist"])]
invalid = [c for c in candidates if math.isnan(c["pos_dist"])]

if valid:
    gr_n   = minmax_norm([c["goal_rate"] for c in valid], invert=False)
    gap_n  = minmax_norm([c["avg_gap"]   for c in valid], invert=True)
    dist_n = minmax_norm([c["pos_dist"]  for c in valid], invert=True)

    for i, c in enumerate(valid):
        c["norm_gr"]  = gr_n[i]
        c["norm_gap"] = gap_n[i]
        c["norm_pos"] = dist_n[i]
        c["composite"] = round(W_GR*gr_n[i] + W_GAP*gap_n[i] + W_POS*dist_n[i], 4)
        c["ci_confirmed"] = c["ci_lo"] > league_gr_2526

for c in invalid:
    c["norm_gr"] = c["norm_gap"] = c["norm_pos"] = c["composite"] = 0.0
    c["ci_confirmed"] = False

ranked_2526 = sorted(valid + invalid, key=lambda x: -x["composite"])
for i, c in enumerate(ranked_2526):
    c["rank_2526"] = i + 1

# Save
out1 = os.path.join(DATA_DIR, "netfront_2526_rankings.csv")
cols1 = ["rank_2526","player_name","player_id","team",
         "es","nf","nf_rate","nf_goals","reb_shots",
         "goal_rate","ci_lo","ci_hi","avg_x","avg_y","avg_gap","pos_dist",
         "norm_gr","norm_gap","norm_pos","composite","ci_confirmed"]
with open(out1, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols1, extrasaction="ignore")
    w.writeheader()
    w.writerows(ranked_2526)

# Print top 50
print(f"\n{SEP}")
print(f"  ANALYSIS 1 — {TARGET_S} NET-FRONT RANKINGS  (ES, min {MIN_REB_2526} reb shots, n={len(ranked_2526)})")
print(f"  League rebound GR: {league_gr_2526:.4f}   Composite: 40% GR + 30% timing + 30% position")
print(SEP)
print(f"  {'Rk':<4} {'Player':<22} {'Tm':<5} {'ES':>5} {'NF':>4} {'NF%':>6} "
      f"{'NF Gls':>7} {'RbSh':>5} {'GR':>7} {'CI lo':>6} {'CI hi':>6} "
      f"{'Avg X':>6} {'Avg Y':>6} {'Gap':>5} {'Score':>7}  CI")
print(f"  {SEP2}")
for c in ranked_2526[:50]:
    ci = "✓" if c["ci_confirmed"] else " "
    xf = f"{c['avg_x']:6.1f}" if not math.isnan(c["avg_x"]) else "   —  "
    yf = f"{c['avg_y']:6.1f}" if not math.isnan(c["avg_y"]) else "   —  "
    print(f"  {c['rank_2526']:<4} {c['player_name']:<22} {c['team']:<5} "
          f"{c['es']:>5} {c['nf']:>4} {c['nf_rate']*100:>5.1f}% "
          f"{c['nf_goals']:>7} {c['reb_shots']:>5} "
          f"{c['goal_rate']:>7.4f} {c['ci_lo']:>6.4f} {c['ci_hi']:>6.4f} "
          f"{xf} {yf} {c['avg_gap']:>4.2f}s "
          f"{c['composite']:>7.4f}  {ci}")
print(f"\n  Saved: {out1}  ({len(ranked_2526)} players)")


# ════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2 — Acquisition target screening
# ════════════════════════════════════════════════════════════════════════
print(f"\n── Analysis 2: Acquisition target screening ──")

# Build lookup: pid → 2526 metrics (from ranked_2526)
lookup_2526 = {c["player_id"]: c for c in ranked_2526}

# Career NF attempt rate per player (all 6 seasons pooled)
def career_nf_rate(pid):
    tot_es = sum(es_shots.get((pid,s),0) for s in ALL_S)
    tot_nf = sum(nf_shots.get((pid,s),0) for s in ALL_S)
    return tot_nf / tot_es if tot_es >= 20 else None

def seasons_in_db(pid):
    return sum(1 for s in ALL_S if es_shots.get((pid,s),0) >= 10)

# Load elite file
elite = []
with open(os.path.join(DATA_DIR, "elite_netfront_players.csv"), newline="") as f:
    for row in csv.DictReader(f):
        elite.append(row)

print(f"  Elite players (6-season file): {len(elite)}")

tier1, tier2, tier3, tier_insuf = [], [], [], []

for e in elite:
    pid          = str(e["player_id"])
    name         = e["player_name"]
    car_gr       = float(e["rebound_goal_rate"])
    car_gap      = float(e["avg_time_gap"])
    car_composite= float(e["composite"])
    ci_confirmed = e["ci_confirmed"] == "True"
    trend        = e.get("trend","stable")

    car_rate = career_nf_rate(pid)   # computed from shot data

    # 2526 data
    m = lookup_2526.get(pid)

    if m is None or m["reb_shots"] < MIN_REB_2526:
        tier_insuf.append({
            "player_id": pid, "player_name": name,
            "career_composite": car_composite,
            "ci_confirmed": ci_confirmed,
            "reb_shots_2526": m["reb_shots"] if m else 0,
            "tier": "insufficient_sample",
            "current_team": m["team"] if m else e["current_team"],
            "trend": trend,
        })
        continue

    # Health checks
    flags = []

    # 1. NF goal rate
    gr_lo = car_gr * (1 - THR_GR)
    gr_hi = car_gr * (1 + THR_GR)
    gr_ok = gr_lo <= m["goal_rate"] <= gr_hi
    if not gr_ok:
        flags.append(f"GR {m['goal_rate']:.3f} vs career {car_gr:.3f} "
                     f"({'▲' if m['goal_rate']>car_gr else '▼'}"
                     f"{abs(m['goal_rate']-car_gr)/car_gr*100:.0f}%)")

    # 2. Avg time gap
    gap_ok = abs(m["avg_gap"] - car_gap) <= THR_GAP
    if not gap_ok:
        flags.append(f"gap {m['avg_gap']:.2f}s vs career {car_gap:.2f}s "
                     f"({'▲' if m['avg_gap']>car_gap else '▼'}"
                     f"{abs(m['avg_gap']-car_gap):.2f}s)")

    # 3. NF attempt rate
    if car_rate:
        rate_lo = car_rate * (1 - THR_RATE)
        rate_hi = car_rate * (1 + THR_RATE)
        rate_ok = rate_lo <= m["nf_rate"] <= rate_hi
        if not rate_ok:
            flags.append(f"rate {m['nf_rate']:.3f} vs career {car_rate:.3f} "
                         f"({'▲' if m['nf_rate']>car_rate else '▼'}"
                         f"{abs(m['nf_rate']-car_rate)/car_rate*100:.0f}%)")
    else:
        rate_ok = True   # can't assess — treat as OK

    n_flags = len(flags)
    record = {
        "player_id"        : pid,
        "player_name"      : name,
        "career_composite" : car_composite,
        "career_gr"        : car_gr,
        "career_gap"       : car_gap,
        "career_rate"      : car_rate or 0.0,
        "ci_confirmed"     : ci_confirmed,
        "trend"            : trend,
        "current_team_2526": m["team"],
        "reb_shots_2526"   : m["reb_shots"],
        "gr_2526"          : m["goal_rate"],
        "gap_2526"         : m["avg_gap"],
        "rate_2526"        : m["nf_rate"],
        "composite_2526"   : m["composite"],
        "n_flags"          : n_flags,
        "flags"            : " | ".join(flags) if flags else "all clear",
        "seasons_in_db"    : seasons_in_db(pid),
    }

    if n_flags == 0:
        record["tier"] = "Tier1_healthy"
        tier1.append(record)
    elif n_flags == 1:
        record["tier"] = "Tier2_watch"
        tier2.append(record)
    else:
        record["tier"] = "Tier3_declining"
        tier3.append(record)

# Sort each tier by career composite desc
tier1.sort(key=lambda x: -x["career_composite"])
tier2.sort(key=lambda x: -x["career_composite"])
tier3.sort(key=lambda x: -x["career_composite"])
tier_insuf.sort(key=lambda x: -x["career_composite"])

print(f"  Tier 1 (healthy):             {len(tier1)}")
print(f"  Tier 2 (watch — 1 flag):      {len(tier2)}")
print(f"  Tier 3 (declining — 2+ flags): {len(tier3)}")
print(f"  Insufficient sample:           {len(tier_insuf)}")

# Fetch age for Tier 1 players from API
print(f"\n  Fetching age data for {len(tier1)} Tier 1 players from NHL API...")
for rec in tier1:
    api = fetch_player_api(rec["player_id"])
    rec["age"] = api.get("age","?")
    time.sleep(0.12)

# ── PRINT TIER 1 ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  ANALYSIS 2 — ACQUISITION TARGETS")
print(SEP)

TREND_ICON = {"rising":"↑","declining":"↓","stable":"—"}

print(f"\n  ┌─ TIER 1: BUY TARGETS — All metrics healthy in {TARGET_S} ({len(tier1)} players) ─┐")
print(f"  {'Player':<22} {'Age':>4} {'Tm':<5} {'Szns':>5}  "
      f"{'Car.Comp':>9}  {'Car.GR':>7}  {'2526.GR':>8}  "
      f"{'Car.Gap':>8}  {'2526.Gap':>9}  {'Car.Rate':>9}  {'2526.Rate':>10}  "
      f"{'CI':>3}  {'Trend':>8}")
print(f"  {SEP2}")
for r in tier1:
    ci  = "✓" if r["ci_confirmed"] else " "
    tr  = TREND_ICON.get(r["trend"],"—") + " " + r["trend"]
    print(f"  {r['player_name']:<22} {str(r['age']):>4} {r['current_team_2526']:<5} "
          f"{r['seasons_in_db']:>5}  "
          f"{r['career_composite']:>9.4f}  {r['career_gr']:>7.4f}  {r['gr_2526']:>8.4f}  "
          f"{r['career_gap']:>8.2f}s  {r['gap_2526']:>8.2f}s  "
          f"{r['career_rate']:>9.4f}  {r['rate_2526']:>10.4f}  "
          f"{ci:>3}  {tr}")

# ── PRINT TIER 2 ─────────────────────────────────────────────────────────────
print(f"\n  ┌─ TIER 2: WATCH — One metric slipping ({len(tier2)} players) ─┐")
print(f"  {'Player':<22} {'Tm':<5} {'Car.Comp':>9}  {'CI':>3}  {'Trend':>10}  Issue")
print(f"  {'-'*90}")
for r in tier2:
    ci = "✓" if r["ci_confirmed"] else " "
    tr = TREND_ICON.get(r["trend"],"—") + " " + r["trend"]
    print(f"  {r['player_name']:<22} {r['current_team_2526']:<5} "
          f"{r['career_composite']:>9.4f}  {ci:>3}  {tr:<12}  {r['flags']}")

# ── PRINT TIER 3 summary ─────────────────────────────────────────────────────
print(f"\n  ┌─ TIER 3: AVOID / SELL — 2+ metrics off ({len(tier3)} players) ─┐")
print(f"  {'Player':<22} {'Tm':<5} {'Car.Comp':>9}  {'CI':>3}  {'Trend':>10}  Issues")
print(f"  {'-'*90}")
for r in tier3:
    ci = "✓" if r["ci_confirmed"] else " "
    tr = TREND_ICON.get(r["trend"],"—") + " " + r["trend"]
    print(f"  {r['player_name']:<22} {r['current_team_2526']:<5} "
          f"{r['career_composite']:>9.4f}  {ci:>3}  {tr:<12}  {r['flags']}")

print(f"\n  ┌─ INSUFFICIENT SAMPLE (<{MIN_REB_2526} reb shots in {TARGET_S}) — {len(tier_insuf)} players ─┐")
print(f"  {'Player':<22} {'Tm':<5} {'Car.Comp':>9}  {'CI':>3}  {'Reb2526':>8}  Trend")
print(f"  {'-'*70}")
for r in sorted(tier_insuf, key=lambda x: -x["career_composite"]):
    ci = "✓" if r["ci_confirmed"] else " "
    tr = TREND_ICON.get(r.get("trend","—"),"—")
    print(f"  {r['player_name']:<22} {r['current_team']:<5} "
          f"{r['career_composite']:>9.4f}  {ci:>3}  {r['reb_shots_2526']:>8}  {tr} {r.get('trend','')}")

# ── SAVE acquisition targets ──────────────────────────────────────────────────
out2 = os.path.join(DATA_DIR, "acquisition_targets.csv")
all_records = []
for r in tier1:
    r2 = dict(r); r2["tier"] = "Tier1_healthy"; all_records.append(r2)
for r in tier2:
    r2 = dict(r); r2["age"] = ""; all_records.append(r2)
for r in tier3:
    r2 = dict(r); r2["age"] = ""; all_records.append(r2)
for r in tier_insuf:
    r2 = dict(r)
    r2["age"] = ""; r2["current_team_2526"] = r2.pop("current_team","—")
    r2.setdefault("gr_2526",""); r2.setdefault("gap_2526",""); r2.setdefault("rate_2526","")
    r2.setdefault("composite_2526",""); r2.setdefault("n_flags",""); r2.setdefault("seasons_in_db","")
    r2.setdefault("career_gr",""); r2.setdefault("career_gap",""); r2.setdefault("career_rate","")
    all_records.append(r2)

save_cols = [
    "tier","player_name","player_id","age","current_team_2526","seasons_in_db",
    "career_composite","career_gr","career_gap","career_rate",
    "composite_2526","gr_2526","gap_2526","rate_2526","reb_shots_2526",
    "ci_confirmed","trend","n_flags","flags"
]
with open(out2, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=save_cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(all_records)

print(f"\n{SEP}")
print(f"  Saved: {out1}")
print(f"  Saved: {out2}")
print(f"\n  Tier summary:")
print(f"    Tier 1 (buy):       {len(tier1):>3}  — all 3 metrics within bounds in {TARGET_S}")
print(f"    Tier 2 (watch):     {len(tier2):>3}  — exactly 1 metric outside bounds")
print(f"    Tier 3 (avoid):     {len(tier3):>3}  — 2+ metrics outside bounds")
print(f"    Insuf. sample:      {len(tier_insuf):>3}  — fewer than {MIN_REB_2526} reb shots in {TARGET_S}")
print(f"\n  Thresholds: GR ±{int(THR_GR*100)}% | gap ±{THR_GAP}s | rate ±{int(THR_RATE*100)}%")
print(SEP + "\n")
