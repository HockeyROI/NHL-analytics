#!/usr/bin/env python3
"""
HockeyROI - Goalie Comparison Script
Save to: NHL analysis/Goalies/goalie_analysis_compare.py

Usage:
  cd "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies"
  python3 goalie_analysis_compare.py

What this does:
  - Prompts for a goalie name, finds their ID via NHL search API
  - Pulls all even strength shots across 3 seasons (2023-24, 2024-25, 2025-26)
  - Uses the exact same distance, danger zone, rebound, and rush logic as goalie_analysis.py
  - Compares goalie SV% against league benchmark CSVs for:
      shot type | lateral zone | danger zone | rebound | rush | backhand from center
  - Prints a clean summary table per category with goalie SV%, league SV%, differential
  - Flags anything >.020 above/below league average as STRENGTH or WEAKNESS
  - Saves all shots to Goalies/{GoalieName}/ for future reuse
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import sys
import math
import glob

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL    = "https://api-web.nhle.com/v1"
SEARCH_URL  = "https://search.d3.nhle.com/api/v1/search/player"
GOALIES_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies"
BENCH_DIR   = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies/Benchmarks"
SEASONS     = ["20252026", "20242025", "20232024"]

# Must match goalie_analysis.py exactly
NET_X = 89.0
NET_Y = 0.0
DIST_BINS   = list(range(0, 62, 2)) + [200]
DIST_LABELS = [f"{i}-{i+2}ft" for i in range(0, 60, 2)] + ["60ft+"]

# Threshold for flagging a strength or weakness
FLAG_THRESHOLD = 0.020
MIN_SHOTS      = 20   # minimum shots to report a category

# ─── HTTP ──────────────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers['User-Agent'] = 'HockeyROI-Analysis/2.0'

def fetch(url, timeout=12):
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(0.5 * (attempt + 1))
    return None

# ─── DISTANCE & ZONE (identical to goalie_analysis.py) ────────────────────────
def calc_distance(x, y):
    if pd.isna(x) or pd.isna(y):
        return None
    x_abs = abs(x)
    dist = math.sqrt((x_abs - NET_X)**2 + (y - NET_Y)**2)
    return round(dist, 1)

def calc_danger_zone(dist, y):
    if dist is None:
        return 'Unknown'
    if dist <= 20 and abs(y) <= 22:
        return 'High'
    elif dist <= 40:
        return 'Medium'
    else:
        return 'Low'

def parse_situation(code):
    if code is None:
        return 'Unknown'
    code = str(code)
    if len(code) != 4:
        return 'Unknown'
    away_g  = int(code[0])
    away_sk = int(code[1])
    home_sk = int(code[2])
    home_g  = int(code[3])
    if home_g == 0 or away_g == 0:
        if home_sk != away_sk:
            return 'Penalty Shot'
        return 'Empty Net'
    if home_sk == away_sk == 5:
        return 'Even Strength'
    if home_sk > away_sk:
        return 'Power Play'
    if away_sk > home_sk:
        return 'Shorthanded'
    return 'Other'

def time_to_secs(t):
    try:
        parts = str(t).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0

def detect_rebound(prev_event, period, time_secs):
    if prev_event is None:
        return False
    if prev_event['period'] != period:
        return False
    if prev_event['type_code'] not in [505, 506]:
        return False
    time_diff = time_secs - prev_event['time_secs']
    return 0 < time_diff <= 3

def detect_rush(all_events, shot_idx, period, time_secs, shot_x):
    if shot_x is None or abs(shot_x) < 50:
        return False
    window_start = time_secs - 12
    for k in range(shot_idx - 1, max(0, shot_idx - 20), -1):
        ev = all_events[k]
        if ev['period'] != period:
            break
        if ev['time_secs'] < window_start:
            break
        if ev['type_code'] in [502, 516, 520]:
            return False
        if ev['x'] is not None and abs(ev['x']) < 25:
            return True
    return False

# ─── STEP 1: FIND GOALIE ───────────────────────────────────────────────────────
def find_goalie(name_query):
    """Search NHL API for goalie by name. Returns (player_id, full_name)."""
    print(f"\nSearching NHL API for '{name_query}'...")
    params = {'culture': 'en-us', 'limit': 20, 'q': name_query, 'active': 'true'}
    try:
        r = SESSION.get(SEARCH_URL, params=params, timeout=10)
        results = r.json() if r.status_code == 200 else []
    except Exception:
        results = []

    goalies = [p for p in results if p.get('positionCode') == 'G']

    if not goalies:
        # Try without active filter in case goalie retired mid-search window
        try:
            params2 = {'culture': 'en-us', 'limit': 20, 'q': name_query}
            r2 = SESSION.get(SEARCH_URL, params=params2, timeout=10)
            all_results = r2.json() if r2.status_code == 200 else []
            goalies = [p for p in all_results if p.get('positionCode') == 'G']
        except Exception:
            pass

    if not goalies:
        print(f"  No goalies found matching '{name_query}'.")
        return None, None

    if len(goalies) == 1:
        g = goalies[0]
        pid  = g.get('playerId')
        name = f"{g.get('firstName', '')} {g.get('lastName', '')}".strip()
        print(f"  Found: {name} | ID: {pid}")
        return pid, name

    # Multiple matches — let user pick
    print(f"\n  Multiple matches:")
    for i, g in enumerate(goalies):
        name = f"{g.get('firstName', '')} {g.get('lastName', '')}".strip()
        team = g.get('teamAbbrev', '???')
        print(f"  [{i+1}] {name} ({team})")
    while True:
        choice = input(f"  Enter number [1-{len(goalies)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(goalies):
            g    = goalies[int(choice) - 1]
            pid  = g.get('playerId')
            name = f"{g.get('firstName', '')} {g.get('lastName', '')}".strip()
            print(f"  Selected: {name} | ID: {pid}")
            return pid, name

# ─── STEP 2: GET GAME LOGS ─────────────────────────────────────────────────────
def get_game_logs(player_id):
    all_games = []
    for season in SEASONS:
        data = fetch(f"{BASE_URL}/player/{player_id}/game-log/{season}/2")
        if data:
            games = data.get('gameLog', [])
            print(f"  Season {season}: {len(games)} games")
            all_games.extend(games)
    print(f"  Total games: {len(all_games)}")
    return all_games

# ─── STEP 3: PULL SHOT DATA ────────────────────────────────────────────────────
def pull_shots(games_list, goalie_id, goalie_name):
    all_shots = []
    total = len(games_list)
    mins  = round(total * 0.3 / 60, 1)
    print(f"\n  Processing {total} games (~{mins} min)...\n")

    for i, game in enumerate(games_list):
        game_id   = game.get('gameId')
        game_date = game.get('gameDate')
        data      = fetch(f"{BASE_URL}/gamecenter/{game_id}/play-by-play")
        if not data:
            continue

        plays = data.get('plays', [])

        all_events = []
        for play in plays:
            all_events.append({
                'event_idx':  play.get('eventId', 0),
                'type_code':  play.get('typeCode'),
                'type_desc':  play.get('typeDescKey', ''),
                'period':     play.get('periodDescriptor', {}).get('number'),
                'time_secs':  time_to_secs(play.get('timeInPeriod', '0:00')),
                'x':          play.get('details', {}).get('xCoord'),
                'y':          play.get('details', {}).get('yCoord'),
            })

        for j, play in enumerate(plays):
            tc = play.get('typeCode')
            if tc not in [505, 506]:
                continue
            det = play.get('details', {})
            if det.get('goalieInNetId') != goalie_id:
                continue

            shooter_id = det.get('scoringPlayerId') if tc == 505 else det.get('shootingPlayerId')
            period     = play.get('periodDescriptor', {}).get('number')
            time_str   = play.get('timeInPeriod', '0:00')
            time_secs  = time_to_secs(time_str)
            sit_code   = play.get('situationCode')
            x          = det.get('xCoord')
            y          = det.get('yCoord')

            prev_event = all_events[j-1] if j > 0 else None
            is_rebound = detect_rebound(prev_event, period, time_secs)
            is_rush    = detect_rush(all_events, j, period, time_secs, x)

            all_shots.append({
                'game_id':            game_id,
                'game_date':          game_date,
                'period':             period,
                'time':               time_str,
                'time_secs':          time_secs,
                'shot_type':          det.get('shotType'),
                'shooting_player_id': shooter_id,
                'x_coord':            x,
                'y_coord':            y,
                'zone':               det.get('zoneCode'),
                'is_goal':            1 if tc == 505 else 0,
                'situation_code':     sit_code,
                'is_rebound':         is_rebound,
                'is_rush':            is_rush,
            })

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{total} games | {len(all_shots)} shots")
        time.sleep(0.3)

    print(f"\n  Completed. Total shots: {len(all_shots)}")
    return all_shots

# ─── STEP 4: ADD COMPUTED FEATURES ────────────────────────────────────────────
def add_features(df):
    df = df.copy()

    df['distance_ft'] = df.apply(
        lambda r: calc_distance(r['x_coord'], r['y_coord']), axis=1
    )
    df['distance_band'] = pd.cut(
        df['distance_ft'], bins=DIST_BINS, labels=DIST_LABELS, right=False
    ).astype(str)
    df['danger_zone'] = df.apply(
        lambda r: calc_danger_zone(
            r['distance_ft'], r['y_coord'] if r['y_coord'] is not None else 0
        ), axis=1
    )
    df['lateral'] = pd.cut(
        df['y_coord'], bins=[-100, -15, 15, 100], labels=['Left', 'Center', 'Right']
    ).astype(str)
    df['situation'] = df['situation_code'].apply(parse_situation)

    def get_season(x):
        x = str(x)
        if x[:4] == '2023' or (x[:4] == '2024' and int(x[5:7]) < 9):
            return '2023-24'
        if x[:4] == '2024' or (x[:4] == '2025' and int(x[5:7]) < 9):
            return '2024-25'
        return '2025-26'

    df['season'] = df['game_date'].apply(get_season)
    return df

# ─── STEP 5: LOAD BENCHMARKS ──────────────────────────────────────────────────
def load_benchmarks():
    """
    Load all available benchmark CSVs from the Benchmarks folder.
    If multiple season files exist, aggregate shots+goals to get combined league SV%.
    Returns a dict: category -> DataFrame with league_sv_pct column.
    """
    season_tags = ["20252026", "20242025", "20232024"]

    categories = {
        'shot_type':   ('shot_type',   'benchmarks_shot_type_{}.csv'),
        'lateral':     ('lateral',     'benchmarks_lateral_{}.csv'),
        'danger_zone': ('danger_zone', 'benchmarks_danger_zone_{}.csv'),
        'rebound':     ('is_rebound',  'benchmarks_rebound_{}.csv'),
        'rush':        ('is_rush',     'benchmarks_rush_{}.csv'),
    }

    benchmarks = {}
    for cat_key, (key_col, fname_template) in categories.items():
        frames = []
        for tag in season_tags:
            path = os.path.join(BENCH_DIR, fname_template.format(tag))
            if os.path.exists(path):
                df = pd.read_csv(path)
                frames.append(df)

        if not frames:
            continue

        # Aggregate across all available seasons
        combined = pd.concat(frames, ignore_index=True)
        agg = combined.groupby(key_col, as_index=False).agg(
            total_shots=('total_shots', 'sum'),
            total_goals=('total_goals', 'sum'),
        )
        agg['league_sv_pct'] = round(1 - agg['total_goals'] / agg['total_shots'], 3)
        benchmarks[cat_key] = agg[[key_col, 'total_shots', 'total_goals', 'league_sv_pct']]

    # Backhand-from-center: derive from raw league shots files
    raw_frames = []
    for tag in season_tags:
        path = os.path.join(BENCH_DIR, f"all_goalie_shots_{tag}.csv")
        if os.path.exists(path):
            try:
                raw_frames.append(pd.read_csv(path))
            except Exception:
                pass

    if raw_frames:
        raw = pd.concat(raw_frames, ignore_index=True)
        # Even strength only (matches benchmark methodology)
        es_raw = raw[raw['situation'] == 'Even Strength'] if 'situation' in raw.columns else raw
        bfc = es_raw[(es_raw['shot_type'] == 'backhand') & (es_raw['lateral'] == 'Center')]
        if len(bfc) >= 50:
            league_sv = round(1 - bfc['is_goal'].sum() / len(bfc), 3)
            benchmarks['backhand_center'] = pd.DataFrame([{
                'category':      'Backhand from Center',
                'total_shots':   len(bfc),
                'total_goals':   int(bfc['is_goal'].sum()),
                'league_sv_pct': league_sv,
            }])

    return benchmarks

# ─── STEP 6: BUILD COMPARISON TABLES ──────────────────────────────────────────
def sv_group(df, col):
    """Compute SV% grouped by one column."""
    g = df.groupby(col, observed=True).agg(
        shots=('is_goal', 'count'),
        goals=('is_goal', 'sum'),
    ).reset_index()
    g['sv_pct'] = round(1 - g['goals'] / g['shots'], 3)
    return g[g['shots'] >= MIN_SHOTS].copy()

def flag(diff):
    if diff >= FLAG_THRESHOLD:
        return 'STRENGTH +'
    if diff <= -FLAG_THRESHOLD:
        return 'WEAKNESS -'
    return ''

def print_comparison_table(title, rows, key_col_label):
    """
    rows: list of dicts with keys: label, shots, goalie_sv, league_sv, diff, flag
    """
    print(f"\n{'─'*68}")
    print(f"  {title}")
    print(f"{'─'*68}")
    header = f"  {'Category':<22} {'Shots':>6}  {'Goalie SV%':>10}  {'League SV%':>10}  {'Diff':>7}  {'':>12}"
    print(header)
    print(f"  {'-'*64}")
    for r in rows:
        diff_str = f"{r['diff']:+.3f}"
        print(f"  {str(r['label']):<22} {r['shots']:>6}  {r['goalie_sv']:>10.3f}  {r['league_sv']:>10.3f}  {diff_str:>7}  {r['flag']:<12}")

# ─── STEP 7: RUN COMPARISON ────────────────────────────────────────────────────
def run_comparison(df, goalie_name, benchmarks):
    # Filter to even strength only
    es = df[df['situation'] == 'Even Strength'].copy()
    total_es = len(es)
    overall_sv = round(1 - es['is_goal'].mean(), 3)

    print(f"\n{'='*68}")
    print(f"  COMPARISON REPORT: {goalie_name.upper()}")
    print(f"{'='*68}")
    print(f"  Even strength shots (3 seasons): {total_es}")
    print(f"  Overall ES SV%: {overall_sv:.3f}")

    season_breakdown = es.groupby('season').agg(
        shots=('is_goal', 'count'), goals=('is_goal', 'sum')
    )
    season_breakdown['sv_pct'] = round(1 - season_breakdown['goals'] / season_breakdown['shots'], 3)
    print(f"\n  Season breakdown:")
    for season, row in season_breakdown.iterrows():
        print(f"    {season}: {int(row['shots'])} shots | SV% {row['sv_pct']:.3f}")

    all_flags = {'strengths': [], 'weaknesses': []}

    # ── Shot type ──────────────────────────────────────────────────────────────
    if 'shot_type' in benchmarks:
        g = sv_group(es, 'shot_type')
        bench = benchmarks['shot_type']
        merged = g.merge(bench[['shot_type', 'league_sv_pct', 'total_shots']],
                         on='shot_type', how='left')
        merged['diff'] = round(merged['sv_pct'] - merged['league_sv_pct'], 3)
        rows = []
        for _, r in merged.dropna(subset=['league_sv_pct']).sort_values('diff').iterrows():
            f = flag(r['diff'])
            rows.append({'label': r['shot_type'], 'shots': int(r['shots']),
                         'goalie_sv': r['sv_pct'], 'league_sv': r['league_sv_pct'],
                         'diff': r['diff'], 'flag': f})
            if f == 'STRENGTH +':
                all_flags['strengths'].append(f"Shot type — {r['shot_type']} ({r['diff']:+.3f})")
            elif f == 'WEAKNESS -':
                all_flags['weaknesses'].append(f"Shot type — {r['shot_type']} ({r['diff']:+.3f})")
        print_comparison_table("SHOT TYPE", rows, 'shot_type')

    # ── Lateral zone ──────────────────────────────────────────────────────────
    if 'lateral' in benchmarks:
        g = sv_group(es, 'lateral')
        bench = benchmarks['lateral']
        merged = g.merge(bench[['lateral', 'league_sv_pct']], on='lateral', how='left')
        merged['diff'] = round(merged['sv_pct'] - merged['league_sv_pct'], 3)
        rows = []
        for _, r in merged.dropna(subset=['league_sv_pct']).sort_values('diff').iterrows():
            f = flag(r['diff'])
            rows.append({'label': r['lateral'], 'shots': int(r['shots']),
                         'goalie_sv': r['sv_pct'], 'league_sv': r['league_sv_pct'],
                         'diff': r['diff'], 'flag': f})
            if f == 'STRENGTH +':
                all_flags['strengths'].append(f"Lateral — {r['lateral']} ({r['diff']:+.3f})")
            elif f == 'WEAKNESS -':
                all_flags['weaknesses'].append(f"Lateral — {r['lateral']} ({r['diff']:+.3f})")
        print_comparison_table("LATERAL ZONE", rows, 'lateral')

    # ── Danger zone ────────────────────────────────────────────────────────────
    if 'danger_zone' in benchmarks:
        g = sv_group(es, 'danger_zone')
        bench = benchmarks['danger_zone']
        merged = g.merge(bench[['danger_zone', 'league_sv_pct']], on='danger_zone', how='left')
        merged['diff'] = round(merged['sv_pct'] - merged['league_sv_pct'], 3)
        rows = []
        for _, r in merged.dropna(subset=['league_sv_pct']).sort_values('diff').iterrows():
            f = flag(r['diff'])
            rows.append({'label': r['danger_zone'], 'shots': int(r['shots']),
                         'goalie_sv': r['sv_pct'], 'league_sv': r['league_sv_pct'],
                         'diff': r['diff'], 'flag': f})
            if f == 'STRENGTH +':
                all_flags['strengths'].append(f"Danger zone — {r['danger_zone']} ({r['diff']:+.3f})")
            elif f == 'WEAKNESS -':
                all_flags['weaknesses'].append(f"Danger zone — {r['danger_zone']} ({r['diff']:+.3f})")
        print_comparison_table("DANGER ZONE", rows, 'danger_zone')

    # ── Rebound ────────────────────────────────────────────────────────────────
    if 'rebound' in benchmarks:
        g = sv_group(es, 'is_rebound')
        bench = benchmarks['rebound']
        # Normalize boolean column type for merge
        g['is_rebound'] = g['is_rebound'].astype(str)
        bench_copy = bench.copy()
        bench_copy['is_rebound'] = bench_copy['is_rebound'].astype(str)
        merged = g.merge(bench_copy[['is_rebound', 'league_sv_pct']], on='is_rebound', how='left')
        merged['diff'] = round(merged['sv_pct'] - merged['league_sv_pct'], 3)
        label_map = {'True': 'Rebound', 'False': 'Non-Rebound'}
        rows = []
        for _, r in merged.dropna(subset=['league_sv_pct']).sort_values('diff').iterrows():
            lbl = label_map.get(str(r['is_rebound']), str(r['is_rebound']))
            f = flag(r['diff'])
            rows.append({'label': lbl, 'shots': int(r['shots']),
                         'goalie_sv': r['sv_pct'], 'league_sv': r['league_sv_pct'],
                         'diff': r['diff'], 'flag': f})
            if f == 'STRENGTH +':
                all_flags['strengths'].append(f"Rebound — {lbl} ({r['diff']:+.3f})")
            elif f == 'WEAKNESS -':
                all_flags['weaknesses'].append(f"Rebound — {lbl} ({r['diff']:+.3f})")
        print_comparison_table("REBOUND", rows, 'is_rebound')

    # ── Rush ───────────────────────────────────────────────────────────────────
    if 'rush' in benchmarks:
        g = sv_group(es, 'is_rush')
        bench = benchmarks['rush']
        g['is_rush'] = g['is_rush'].astype(str)
        bench_copy = bench.copy()
        bench_copy['is_rush'] = bench_copy['is_rush'].astype(str)
        merged = g.merge(bench_copy[['is_rush', 'league_sv_pct']], on='is_rush', how='left')
        merged['diff'] = round(merged['sv_pct'] - merged['league_sv_pct'], 3)
        label_map = {'True': 'Rush Shot', 'False': 'Non-Rush'}
        rows = []
        for _, r in merged.dropna(subset=['league_sv_pct']).sort_values('diff').iterrows():
            lbl = label_map.get(str(r['is_rush']), str(r['is_rush']))
            f = flag(r['diff'])
            rows.append({'label': lbl, 'shots': int(r['shots']),
                         'goalie_sv': r['sv_pct'], 'league_sv': r['league_sv_pct'],
                         'diff': r['diff'], 'flag': f})
            if f == 'STRENGTH +':
                all_flags['strengths'].append(f"Rush — {lbl} ({r['diff']:+.3f})")
            elif f == 'WEAKNESS -':
                all_flags['weaknesses'].append(f"Rush — {lbl} ({r['diff']:+.3f})")
        print_comparison_table("RUSH SHOT", rows, 'is_rush')

    # ── Backhand from center ───────────────────────────────────────────────────
    bfc = es[(es['shot_type'] == 'backhand') & (es['lateral'] == 'Center')]
    if 'backhand_center' in benchmarks and len(bfc) >= MIN_SHOTS:
        bench_bfc = benchmarks['backhand_center'].iloc[0]
        g_sv  = round(1 - bfc['is_goal'].mean(), 3)
        l_sv  = bench_bfc['league_sv_pct']
        diff  = round(g_sv - l_sv, 3)
        f     = flag(diff)
        rows  = [{'label': 'Backhand/Center', 'shots': len(bfc),
                  'goalie_sv': g_sv, 'league_sv': l_sv, 'diff': diff, 'flag': f}]
        if f == 'STRENGTH +':
            all_flags['strengths'].append(f"Backhand from center ({diff:+.3f})")
        elif f == 'WEAKNESS -':
            all_flags['weaknesses'].append(f"Backhand from center ({diff:+.3f})")
        print_comparison_table(
            f"BACKHAND FROM CENTER  (league sample: {int(bench_bfc['total_shots'])} shots)",
            rows, 'backhand_center'
        )
    elif len(bfc) > 0:
        g_sv = round(1 - bfc['is_goal'].mean(), 3)
        print(f"\n  Backhand from center: {len(bfc)} shots | SV% {g_sv:.3f} (no benchmark — need raw league shots file)")

    # ── Overall summary ────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  OVERALL SUMMARY — {goalie_name.upper()}")
    print(f"{'='*68}")

    if all_flags['strengths']:
        print(f"\n  STRENGTHS (>{FLAG_THRESHOLD:.3f} above league avg):")
        for s in all_flags['strengths']:
            print(f"    + {s}")
    else:
        print(f"\n  No categories flagged as strengths (>{FLAG_THRESHOLD:.3f} above league avg).")

    if all_flags['weaknesses']:
        print(f"\n  WEAKNESSES (>{FLAG_THRESHOLD:.3f} below league avg):")
        for w in all_flags['weaknesses']:
            print(f"    - {w}")
    else:
        print(f"\n  No categories flagged as weaknesses (>{FLAG_THRESHOLD:.3f} below league avg).")

    print()

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  HOCKEYROI - GOALIE COMPARISON REPORT")
    print("  Even Strength | 3 Seasons | vs League Benchmarks")
    print("=" * 68)

    name_query = input("\nGoalie name (first, last, or both): ").strip()
    if not name_query:
        sys.exit("No name entered.")

    goalie_id, goalie_name = find_goalie(name_query)
    if not goalie_id:
        sys.exit(1)

    # Save directory: Goalies/{LastName}/
    last_name = goalie_name.split()[-1]
    save_dir  = os.path.join(GOALIES_DIR, last_name)
    os.makedirs(save_dir, exist_ok=True)
    csv_path  = os.path.join(save_dir, f"{last_name.lower()}_shots_all_seasons.csv")

    # Load or pull shots
    if os.path.exists(csv_path):
        print(f"\nExisting data found: {csv_path}")
        choice = input("Use existing data? (y) or re-pull from API? (n): ").strip().lower()
        if choice == 'y':
            df = pd.read_csv(csv_path)
            print(f"  Loaded {len(df)} shots.")
            if 'distance_ft' not in df.columns or 'lateral' not in df.columns:
                print("  Updating features...")
                df = add_features(df)
                df.to_csv(csv_path, index=False)
        else:
            df = _pull_and_save(goalie_id, goalie_name, csv_path)
    else:
        df = _pull_and_save(goalie_id, goalie_name, csv_path)

    if df is None or len(df) == 0:
        sys.exit("No shot data found.")

    # Add features if needed
    if 'situation' not in df.columns:
        df = add_features(df)
        df.to_csv(csv_path, index=False)

    # Load league benchmarks
    print("\nLoading league benchmarks...")
    benchmarks = load_benchmarks()
    loaded = list(benchmarks.keys())
    if loaded:
        print(f"  Loaded benchmarks: {', '.join(loaded)}")
    else:
        print("  No benchmark files found. Run league_benchmarks.py first.")
        sys.exit(1)

    # Run comparison
    run_comparison(df, goalie_name, benchmarks)

    print(f"  Data saved: {csv_path}")
    print(f"\n  @HockeyROI\n")


def _pull_and_save(goalie_id, goalie_name, csv_path):
    print(f"\nPulling game logs...")
    games = get_game_logs(goalie_id)
    if not games:
        print("  No games found.")
        return None
    shots = pull_shots(games, goalie_id, goalie_name)
    if not shots:
        print("  No shots recorded.")
        return None
    df = pd.DataFrame(shots)
    df = add_features(df)
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} shots to {csv_path}")
    return df


if __name__ == "__main__":
    main()
