#!/usr/bin/env python3
"""
HockeyROI - Add Shooting Team + Team Net-Front Rates
Save to: NHL analysis/Goalies/add_shooting_team.py

Usage:
  cd "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies"
  python3 add_shooting_team.py

Step 1: Build goalie_id + season → goalie_team lookup via NHL API player landing.
        Uses teamName.default → abbrev via standings endpoint.
        Handles traded goalies: picks team with most gamesPlayed.
        Saves: Goalies/Benchmarks/goalie_team_lookup.csv

Step 2: Build game_id → home_team, away_team via boxscore API.
        Shooting team = whichever team is NOT the goalie's team in that game.
        Saves: Goalies/Benchmarks/all_goalie_shots_with_teams.csv

Step 3: Per (shooting_team, season) and pooled — net-front attempt rate + Wilson CI.
        Net-front = is_rebound==True OR shot_type in {tip-in, deflected, bat}
        Saves: Goalies/Benchmarks/team_netfront_rates.csv
"""

import math
import os
import time

import numpy as np
import pandas as pd
import requests

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BENCH_DIR     = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies/Benchmarks"
DATA_FILE     = os.path.join(BENCH_DIR, "all_goalie_shots_3seasons.csv")
BASE_URL      = "https://api-web.nhle.com/v1"
NF_SHOT_TYPES = {"tip-in", "deflected", "bat"}
Z95           = 1.96

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "HockeyROI-Analysis/2.0"


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def fetch(url, timeout=15):
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(0.6 * (attempt + 1))
    return None


def wilson_ci(k, n, z=Z95):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading shot data...")
df = pd.read_csv(DATA_FILE)
for col in ("is_rebound", "is_rush"):
    df[col] = df[col].astype(str).str.strip().map({"True": True, "False": False}).fillna(False)
df["is_goal"] = df["is_goal"].astype(int)
print(f"  Loaded {len(df):,} rows")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build team name → abbrev map from standings
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Step 1: Build goalie-to-team lookup ──")
print("  Fetching team name → abbrev from standings...")

standings = fetch(f"{BASE_URL}/standings/now")
name_to_abbrev = {}
if standings:
    for t in standings.get("standings", []):
        full  = t.get("teamName", {}).get("default", "")
        abbr  = t.get("teamAbbrev", {}).get("default", "")
        place = t.get("placeName", {}).get("default", "")
        common = t.get("teamCommonName", {}).get("default", "")
        if full and abbr:
            name_to_abbrev[full]   = abbr
            name_to_abbrev[place]  = abbr   # e.g. "Winnipeg" → "WPG"
            name_to_abbrev[common] = abbr   # e.g. "Jets" → "WPG"

# Add known edge cases (relocated/renamed teams across 3 seasons)
# Arizona → Utah rename happened after 2023-24
name_to_abbrev["Arizona Coyotes"] = "ARI"
name_to_abbrev["Arizona"]         = "ARI"
name_to_abbrev["Coyotes"]         = "ARI"
name_to_abbrev["Utah Mammoth"]    = "UTA"
name_to_abbrev["Utah"]            = "UTA"
name_to_abbrev["Mammoth"]         = "UTA"

print(f"  {len(name_to_abbrev)} name variants mapped")

# ─── Per (goalie_id, season) → goalie_team ─────────────────────────────────
combos = df[["goalie_id", "goalie_name", "season"]].drop_duplicates()
unique_goalie_ids = combos["goalie_id"].unique()
print(f"  Fetching landing pages for {len(unique_goalie_ids)} unique goalies...")

# Cache full season history per goalie
goalie_season_cache = {}  # goalie_id → list of {season, teamName, gamesPlayed}

for i, gid in enumerate(unique_goalie_ids):
    data = fetch(f"{BASE_URL}/player/{int(gid)}/landing")
    if data:
        totals = data.get("seasonTotals", [])
        rows = [
            {
                "season"      : t["season"],
                "team_name"   : t.get("teamName", {}).get("default", ""),
                "games_played": t.get("gamesPlayed", 0),
            }
            for t in totals
            if t.get("gameTypeId") == 2 and t.get("leagueAbbrev", "") == "NHL"
        ]
        goalie_season_cache[gid] = rows
    else:
        goalie_season_cache[gid] = []

    if (i + 1) % 20 == 0:
        print(f"    {i+1}/{len(unique_goalie_ids)} done...")
    time.sleep(0.05)

print(f"  Done fetching landing pages.")

# Build lookup: (goalie_id, season) → team_abbrev
lookup_rows = []
unresolved  = []

for _, row in combos.iterrows():
    gid    = row["goalie_id"]
    season = row["season"]
    name   = row["goalie_name"]
    history = goalie_season_cache.get(gid, [])

    # Filter to this season's NHL regular-season records
    season_rows = [h for h in history if h["season"] == season]

    if not season_rows:
        unresolved.append((gid, name, season, "no_api_data"))
        lookup_rows.append({"goalie_id": gid, "goalie_name": name,
                            "season": season, "goalie_team": None})
        continue

    # If traded (multiple rows), take the one with most games played
    best = max(season_rows, key=lambda x: x["games_played"])
    team_full = best["team_name"]
    abbrev    = name_to_abbrev.get(team_full)

    # Fallback: try common name or place name from the string
    if not abbrev:
        for part in team_full.split():
            if part in name_to_abbrev:
                abbrev = name_to_abbrev[part]
                break

    if not abbrev:
        unresolved.append((gid, name, season, f"abbrev_not_found:{team_full}"))

    lookup_rows.append({
        "goalie_id"  : gid,
        "goalie_name": name,
        "season"     : season,
        "goalie_team": abbrev,
        "team_full"  : team_full,
        "games_played": best["games_played"],
    })

lookup_df = pd.DataFrame(lookup_rows)
out_lookup = os.path.join(BENCH_DIR, "goalie_team_lookup.csv")
lookup_df.to_csv(out_lookup, index=False)
print(f"  Saved: {out_lookup}  ({len(lookup_df)} rows)")

if unresolved:
    print(f"  WARNING — {len(unresolved)} (goalie, season) combos unresolved:")
    for u in unresolved:
        print(f"    {u}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Game ID → home/away team, then derive shooting_team
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Step 2: Fetch game home/away teams ──")

unique_games = df["game_id"].unique()
print(f"  Fetching boxscores for {len(unique_games):,} unique games...")

game_teams = {}   # game_id → {"home": abbrev, "away": abbrev}
failed_games = 0

for i, gid in enumerate(unique_games):
    data = fetch(f"{BASE_URL}/gamecenter/{int(gid)}/boxscore")
    if data:
        home = data.get("homeTeam", {}).get("abbrev", "")
        away = data.get("awayTeam", {}).get("abbrev", "")
        if home and away:
            game_teams[gid] = {"home": home, "away": away}
        else:
            failed_games += 1
    else:
        failed_games += 1

    if (i + 1) % 500 == 0:
        print(f"    {i+1:,}/{len(unique_games):,} games fetched...")
    time.sleep(0.04)

print(f"  Done. {len(game_teams):,} games resolved, {failed_games} failed.")

# ─── Join goalie team + game teams → shooting team ─────────────────────────
# Build (goalie_id, season) → goalie_team dict
goalie_team_map = {
    (int(r["goalie_id"]), int(r["season"])): r["goalie_team"]
    for _, r in lookup_df.iterrows()
    if pd.notna(r["goalie_team"])
}

def get_shooting_team(row):
    teams = game_teams.get(row["game_id"])
    if not teams:
        return None
    goalie_team = goalie_team_map.get((int(row["goalie_id"]), int(row["season"])))
    if not goalie_team:
        return None
    home, away = teams["home"], teams["away"]
    if goalie_team == home:
        return away
    elif goalie_team == away:
        return home
    else:
        # Goalie's team doesn't match either — flag it (rare edge case: traded mid-game)
        return None

print("  Deriving shooting_team column...")
df["shooting_team"] = df.apply(get_shooting_team, axis=1)

null_count = df["shooting_team"].isna().sum()
print(f"  shooting_team resolved: {(~df['shooting_team'].isna()).sum():,} rows")
print(f"  Could not resolve: {null_count:,} rows ({null_count/len(df)*100:.1f}%)")

out_enriched = os.path.join(BENCH_DIR, "all_goalie_shots_with_teams.csv")
df.to_csv(out_enriched, index=False)
print(f"  Saved: {out_enriched}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Team net-front rates
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Step 3: Team net-front rates ──")

es = df[(df["situation"] == "Even Strength") & df["shooting_team"].notna()].copy()
es["is_netfront"] = es["is_rebound"] | es["shot_type"].isin(NF_SHOT_TYPES)
print(f"  ES shots with shooting_team: {len(es):,}")

# League average
league_nf_rate = es["is_netfront"].sum() / len(es)
print(f"  League avg net-front rate: {league_nf_rate:.4f}")

team_rows = []

# Per season
for season, sgrp in es.groupby("season"):
    for team, tgrp in sgrp.groupby("shooting_team"):
        total = len(tgrp)
        nf    = tgrp["is_netfront"].sum()
        rate  = nf / total
        _, lo, hi = wilson_ci(nf, total)
        team_rows.append({
            "season"         : season,
            "shooting_team"  : team,
            "total_es_att"   : total,
            "netfront_att"   : int(nf),
            "nf_attempt_rate": round(rate, 4),
            "nf_rate_ci_lo"  : round(lo, 4),
            "nf_rate_ci_hi"  : round(hi, 4),
        })

# Pooled (all seasons combined)
for team, tgrp in es.groupby("shooting_team"):
    total = len(tgrp)
    nf    = tgrp["is_netfront"].sum()
    rate  = nf / total
    _, lo, hi = wilson_ci(nf, total)
    team_rows.append({
        "season"         : "POOLED",
        "shooting_team"  : team,
        "total_es_att"   : total,
        "netfront_att"   : int(nf),
        "nf_attempt_rate": round(rate, 4),
        "nf_rate_ci_lo"  : round(lo, 4),
        "nf_rate_ci_hi"  : round(hi, 4),
    })

team_df = pd.DataFrame(team_rows)
# Rank within each season group (1 = highest rate)
team_df["rank"] = team_df.groupby("season")["nf_attempt_rate"].rank(
    ascending=False, method="min").astype(int)
team_df = team_df.sort_values(["season", "rank"]).reset_index(drop=True)

out_team = os.path.join(BENCH_DIR, "team_netfront_rates.csv")
team_df.to_csv(out_team, index=False)
print(f"  Saved: {out_team}")


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
pooled = team_df[team_df["season"] == "POOLED"].sort_values("nf_attempt_rate", ascending=False).reset_index(drop=True)

print("\n" + "═" * 65)
print("  HOCKEYROI — TEAM NET-FRONT RATES (3 SEASONS POOLED)")
print("═" * 65)
print(f"\n  League avg net-front attempt rate: {league_nf_rate:.4f}")
print(f"  Net-front = is_rebound OR shot_type in {{tip-in, deflected, bat}}")
print(f"\n  {'Rk':<4}  {'Team':<6}  {'ES Att':>8}  {'NF Att':>7}  {'Rate':>7}  {'CI lo':>7}  {'CI hi':>7}")
print(f"  {'-'*52}")

def print_team_row(rank, row):
    print(f"  {rank:<4}  {row['shooting_team']:<6}  {row['total_es_att']:>8,}"
          f"  {row['netfront_att']:>7,}  {row['nf_attempt_rate']:>7.4f}"
          f"  {row['nf_rate_ci_lo']:>7.4f}  {row['nf_rate_ci_hi']:>7.4f}")

print(f"\n  TOP 5")
for i, row in pooled.head(5).iterrows():
    print_team_row(i + 1, row)

print(f"\n  BOTTOM 5")
bottom5 = pooled.tail(5)
for i, row in bottom5.iterrows():
    print_team_row(i + 1, row)

# FLA and EDM callouts
print(f"\n  FLORIDA & EDMONTON")
for team in ["FLA", "EDM"]:
    row = pooled[pooled["shooting_team"] == team]
    if not row.empty:
        row = row.iloc[0]
        rank = pooled.index[pooled["shooting_team"] == team][0] + 1
        print_team_row(rank, row)
    else:
        print(f"  {team}: not found in pooled data")

print("\n" + "═" * 65)
print("  Output files written to Goalies/Benchmarks/")
print("  - goalie_team_lookup.csv")
print("  - all_goalie_shots_with_teams.csv")
print("  - team_netfront_rates.csv")
print("═" * 65 + "\n")
