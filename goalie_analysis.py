#!/usr/bin/env python3
"""
HockeyROI - Goalie Analysis Script v2
Save to: NHL analysis/Goalies/goalie_analysis.py

Usage:
  cd "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies"
  python3 goalie_analysis.py

What this does:
  - Pulls all shots faced by a goalie across 3 seasons from NHL API
  - Adds shooter handedness, distance from net, danger zone,
    situation type, rebound flag, rush shot flag
  - Runs full analysis broken down by shot type, distance band,
    location, handedness, and situation
  - Produces 3 professional dark themed charts
  - If league benchmarks exist, automatically compares goalie to league
  - All raw data saved so you can reanalyze anytime

All data is STORED. Nothing is thrown away. Filters only apply to analysis.
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import math

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL    = "https://api-web.nhle.com/v1"
GOALIES_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies"
BENCH_DIR   = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies/Benchmarks"
SEASONS     = ["20252026", "20242025", "20232024"]
LEAGUE_AVG  = 0.910
WATERMARK   = "@HockeyROI"

# NHL rink: net is at x=89, center of net at y=0
NET_X = 89.0
NET_Y = 0.0

# Distance bands in 2 foot increments up to 60ft, then 60+
DIST_BINS   = list(range(0, 62, 2)) + [200]
DIST_LABELS = [f"{i}-{i+2}ft" for i in range(0, 60, 2)] + ["60ft+"]

# Minimum shots to show a finding (individual goalie)
MIN_SHOTS_INDIVIDUAL = 20
# Minimum shots to show a finding (league benchmarks)
MIN_SHOTS_LEAGUE = 50

# ─── THEME ─────────────────────────────────────────────────────────────────────
BG     = '#1a1a2e'
RED    = '#ff4444'
ORANGE = '#ffaa00'
GREEN  = '#44ff44'
BLUE   = '#4488ff'
WHITE  = '#ffffff'
GREY   = '#8b949e'
YELLOW = '#ffff00'

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": WHITE, "axes.labelcolor": WHITE,
    "xtick.color": WHITE, "ytick.color": WHITE,
    "axes.edgecolor": GREY,
})

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

# ─── DISTANCE & ZONE ──────────────────────────────────────────────────────────
def calc_distance(x, y):
    """Calculate shot distance from net in feet"""
    if pd.isna(x) or pd.isna(y):
        return None
    # Shots always toward positive x net
    x_abs = abs(x)
    dist = math.sqrt((x_abs - NET_X)**2 + (y - NET_Y)**2)
    return round(dist, 1)

def calc_danger_zone(dist, y):
    """
    Classify shot danger zone based on distance and lateral position.
    High:   within 20ft AND between faceoff dots (roughly y -22 to 22)
    Medium: within 40ft but outside high danger
    Low:    beyond 40ft or very wide angle
    """
    if dist is None:
        return 'Unknown'
    if dist <= 20 and abs(y) <= 22:
        return 'High'
    elif dist <= 40:
        return 'Medium'
    else:
        return 'Low'

def parse_situation(code):
    """
    Situation code is 4 digits: home_goalie | away_goalie | home_skaters | away_skaters
    1551 = even strength both goalies
    1541 = away PP (home shorthanded)
    1451 = home PP (away shorthanded)
    0651 or 0641 = penalty shot (one goalie pulled)
    1461 or 1641 = pulled goalie situation
    """
    if code is None:
        return 'Unknown'
    code = str(code)
    if len(code) != 4:
        return 'Unknown'
    away_g  = int(code[0])
    away_sk = int(code[1])
    home_sk = int(code[2])
    home_g  = int(code[3])

    # Empty net
    if home_g == 0 or away_g == 0:
        if home_sk != away_sk:
            return 'Penalty Shot'
        return 'Empty Net'

    # Even strength
    if home_sk == away_sk == 5:
        return 'Even Strength'

    # Power play
    if home_sk > away_sk:
        return 'Power Play'
    if away_sk > home_sk:
        return 'Shorthanded'

    return 'Other'

# ─── STEP 1: FIND GOALIE ───────────────────────────────────────────────────────
def find_goalie(last_name, team_abbrev):
    print(f"\nSearching for {last_name} on {team_abbrev} roster...")
    data = fetch(f"{BASE_URL}/roster/{team_abbrev.upper()}/20252026")
    if not data:
        print(f"  Could not fetch roster for {team_abbrev}")
        return None, None
    for g in data.get('goalies', []):
        last  = g.get('lastName',  {}).get('default', '')
        first = g.get('firstName', {}).get('default', '')
        if last_name.lower() in last.lower():
            pid = g.get('id')
            print(f"  Found: {first} {last} | ID: {pid}")
            return pid, f"{first} {last}"
    print(f"  Could not find {last_name} on {team_abbrev} roster")
    return None, None

# ─── STEP 2: GET GAME LOGS ─────────────────────────────────────────────────────
def get_game_logs(player_id, seasons):
    all_games = []
    for season in seasons:
        data = fetch(f"{BASE_URL}/player/{player_id}/game-log/{season}/2")
        if data:
            games = data.get('gameLog', [])
            print(f"  Season {season}: {len(games)} games")
            all_games.extend(games)
    print(f"  Total games: {len(all_games)}")
    return all_games

# ─── STEP 3: PULL SHOT DATA ────────────────────────────────────────────────────
def pull_shots(games_list, goalie_id, goalie_name):
    """
    Pull all shots and goals. For each shot record:
    - Basic info: game, date, period, time, shot type, coordinates
    - Situation code (used later to classify PP/SH/ES/etc)
    - Previous event info (used for rebound and rush detection)
    """
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

        # Build a lookup of all events in this game for context
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

            # Correct shooter ID (goals vs shots use different fields)
            shooter_id = det.get('scoringPlayerId') if tc == 505 else det.get('shootingPlayerId')

            period    = play.get('periodDescriptor', {}).get('number')
            time_str  = play.get('timeInPeriod', '0:00')
            time_secs = time_to_secs(time_str)
            sit_code  = play.get('situationCode')
            x         = det.get('xCoord')
            y         = det.get('yCoord')

            # Previous event analysis (for rebound and rush detection)
            prev_event     = all_events[j-1] if j > 0 else None
            is_rebound     = detect_rebound(prev_event, period, time_secs)
            is_rush        = detect_rush(all_events, j, period, time_secs, x)

            all_shots.append({
                'game_id':           game_id,
                'game_date':         game_date,
                'period':            period,
                'time':              time_str,
                'time_secs':         time_secs,
                'shot_type':         det.get('shotType'),
                'shooting_player_id': shooter_id,
                'x_coord':           x,
                'y_coord':           y,
                'zone':              det.get('zoneCode'),
                'is_goal':           1 if tc == 505 else 0,
                'situation_code':    sit_code,
                'is_rebound':        is_rebound,
                'is_rush':           is_rush,
            })

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{total} games | {len(all_shots)} shots")

        time.sleep(0.3)

    print(f"\n  Completed. Total shots: {len(all_shots)}")
    return all_shots

def time_to_secs(t):
    """Convert MM:SS string to total seconds"""
    try:
        parts = str(t).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0

def detect_rebound(prev_event, period, time_secs):
    """
    Flag as rebound if previous event was a shot or goal
    in the same period within 3 seconds
    """
    if prev_event is None:
        return False
    if prev_event['period'] != period:
        return False
    if prev_event['type_code'] not in [505, 506]:
        return False
    time_diff = time_secs - prev_event['time_secs']
    return 0 < time_diff <= 3

def detect_rush(all_events, shot_idx, period, time_secs, shot_x):
    """
    Flag as rush shot if:
    1. No stoppage (faceoff) in last 12 seconds
    2. Previous puck event was outside the offensive zone (x < 25 abs)
    3. Shot is in offensive zone (x > 50 abs)
    Rush shots include breakaways, 2-on-1s, and odd man rushes.
    Does NOT flag standalone player in front after offensive zone turnover.
    """
    if shot_x is None or abs(shot_x) < 50:
        return False

    window_start = time_secs - 12

    for k in range(shot_idx - 1, max(0, shot_idx - 20), -1):
        ev = all_events[k]
        if ev['period'] != period:
            break
        if ev['time_secs'] < window_start:
            break
        # Stoppage = not a rush
        if ev['type_code'] in [502, 516, 520]:  # faceoff, stoppage codes
            return False
        # Previous puck event was outside offensive zone = rush
        if ev['x'] is not None and abs(ev['x']) < 25:
            return True

    return False

# ─── STEP 4: ADD HANDEDNESS ────────────────────────────────────────────────────
def add_handedness(df):
    unique = df['shooting_player_id'].dropna().unique()
    print(f"\n  Looking up handedness for {len(unique)} shooters...")
    hand = {}
    for i, pid in enumerate(unique):
        data = fetch(f"{BASE_URL}/player/{int(pid)}/landing")
        hand[pid] = data.get('shootsCatches') if data else None
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(unique)} done")
        time.sleep(0.15)
    df['shoots'] = df['shooting_player_id'].map(hand)
    missing = df['shoots'].isna().sum()
    print(f"  Done. Missing handedness: {missing} shots ({round(missing/len(df)*100,1)}%)")
    return df

# ─── STEP 5: ADD COMPUTED FEATURES ────────────────────────────────────────────
def add_features(df):
    """Add distance, distance band, danger zone, situation type, season"""

    print("\n  Adding computed features...")

    # Distance from net
    df['distance_ft'] = df.apply(
        lambda r: calc_distance(r['x_coord'], r['y_coord']), axis=1
    )

    # 2 foot distance bands
    df['distance_band'] = pd.cut(
        df['distance_ft'],
        bins=DIST_BINS,
        labels=DIST_LABELS,
        right=False
    ).astype(str)

    # Danger zone
    df['danger_zone'] = df.apply(
        lambda r: calc_danger_zone(r['distance_ft'], r['y_coord']
                                   if r['y_coord'] is not None else 0),
        axis=1
    )

    # Left / Center / Right lateral position
    df['lateral'] = pd.cut(
        df['y_coord'],
        bins=[-100, -15, 15, 100],
        labels=['Left', 'Center', 'Right']
    ).astype(str)

    # Situation type from situation code
    df['situation'] = df['situation_code'].apply(parse_situation)

    # Season label
    def get_season(x):
        x = str(x)
        if x[:4] == '2023' or (x[:4] == '2024' and int(x[5:7]) < 9):
            return '2023-24'
        if x[:4] == '2024' or (x[:4] == '2025' and int(x[5:7]) < 9):
            return '2024-25'
        return '2025-26'

    df['season'] = df['game_date'].apply(get_season)

    print(f"  Distance range: {df['distance_ft'].min():.1f} to {df['distance_ft'].max():.1f} ft")
    print(f"  Situation breakdown:")
    print(df['situation'].value_counts().to_string())
    print(f"  Danger zone breakdown:")
    print(df['danger_zone'].value_counts().to_string())
    print(f"  Rebounds: {df['is_rebound'].sum()} ({round(df['is_rebound'].mean()*100,1)}%)")
    print(f"  Rush shots: {df['is_rush'].sum()} ({round(df['is_rush'].mean()*100,1)}%)")

    return df

# ─── STEP 6: RUN ANALYSIS ──────────────────────────────────────────────────────
def run_analysis(df, goalie_name):
    """
    Full analysis. Primary analysis = even strength, non empty net.
    Rebounds, rush shots, PP, SH reported separately.
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {goalie_name.upper()}")
    print(f"{'='*60}")

    # Overall
    overall_sv = round(1 - df['is_goal'].mean(), 3)
    print(f"All situations: {len(df)} shots | SV% {overall_sv}")

    # Primary: even strength, both goalies in net
    es = df[df['situation'] == 'Even Strength']
    es_sv = round(1 - es['is_goal'].mean(), 3)
    print(f"Even strength only: {len(es)} shots | SV% {es_sv}")

    # Empty net excluded from all further analysis
    main = es.copy()

    print(f"\nLeague average for reference: {LEAGUE_AVG}")
    diff = round((es_sv - LEAGUE_AVG) * 1000, 1)
    direction = 'BELOW' if es_sv < LEAGUE_AVG else 'ABOVE'
    print(f"ES save pct is {abs(diff)} points {direction} league average\n")

    results = {}

    # ── By shot type ──
    shot_a = _sv_pct(main, 'shot_type', MIN_SHOTS_INDIVIDUAL)
    print("EVEN STRENGTH SAVE % BY SHOT TYPE:")
    print(shot_a.to_string(index=False))
    results['shot_type'] = shot_a

    # ── By distance band ──
    print(f"\nEVEN STRENGTH SAVE % BY DISTANCE (2ft bands, min {MIN_SHOTS_INDIVIDUAL} shots):")
    dist_a = _sv_pct(main, 'distance_band', MIN_SHOTS_INDIVIDUAL)
    print(dist_a.to_string(index=False))
    results['distance'] = dist_a

    # ── Shot type by distance (key table) ──
    print(f"\nSHOT TYPE x DISTANCE (high danger only, within 30ft):")
    close = main[main['distance_ft'] <= 30]
    combo_dist = close.groupby(['shot_type', 'distance_band'], observed=True).agg(
        shots=('is_goal','count'), goals=('is_goal','sum')
    ).reset_index()
    combo_dist['sv_pct'] = round(1 - combo_dist['goals'] / combo_dist['shots'], 3)
    combo_dist = combo_dist[combo_dist['shots'] >= 15].sort_values('sv_pct')
    print(combo_dist.to_string(index=False))
    results['shot_type_x_distance'] = combo_dist

    # ── By danger zone ──
    print(f"\nEVEN STRENGTH SAVE % BY DANGER ZONE:")
    zone_a = _sv_pct(main, 'danger_zone', MIN_SHOTS_INDIVIDUAL)
    print(zone_a.to_string(index=False))
    results['danger_zone'] = zone_a

    # ── By lateral position ──
    print(f"\nEVEN STRENGTH SAVE % BY LATERAL POSITION:")
    lat_a = _sv_pct(main, 'lateral', MIN_SHOTS_INDIVIDUAL)
    print(lat_a.to_string(index=False))
    results['lateral'] = lat_a

    # ── By handedness ──
    print(f"\nEVEN STRENGTH SAVE % BY SHOOTER HANDEDNESS:")
    hand_a = _sv_pct(main, 'shoots', MIN_SHOTS_INDIVIDUAL)
    print(hand_a.to_string(index=False))
    results['handedness'] = hand_a

    # ── Shot type x handedness ──
    print(f"\nSHOT TYPE x HANDEDNESS (even strength):")
    hand_shot = main.groupby(['shoots', 'shot_type'], observed=True).agg(
        shots=('is_goal','count'), goals=('is_goal','sum')
    ).reset_index()
    hand_shot['sv_pct'] = round(1 - hand_shot['goals'] / hand_shot['shots'], 3)
    hand_shot = hand_shot[hand_shot['shots'] >= MIN_SHOTS_INDIVIDUAL].sort_values('sv_pct')
    print(hand_shot.to_string(index=False))
    results['shot_type_x_handedness'] = hand_shot

    # ── Rebound analysis ──
    print(f"\nREBOUND vs NON-REBOUND (even strength):")
    reb = main.groupby('is_rebound').agg(
        shots=('is_goal','count'), goals=('is_goal','sum')
    ).reset_index()
    reb['sv_pct'] = round(1 - reb['goals'] / reb['shots'], 3)
    reb['type'] = reb['is_rebound'].map({True: 'Rebound', False: 'Non-Rebound'})
    print(reb[['type','shots','goals','sv_pct']].to_string(index=False))
    results['rebound'] = reb

    # ── Rush shot analysis ──
    print(f"\nRUSH SHOT vs NON-RUSH (even strength):")
    rush = main.groupby('is_rush').agg(
        shots=('is_goal','count'), goals=('is_goal','sum')
    ).reset_index()
    rush['sv_pct'] = round(1 - rush['goals'] / rush['shots'], 3)
    rush['type'] = rush['is_rush'].map({True: 'Rush Shot', False: 'Non-Rush'})
    print(rush[['type','shots','goals','sv_pct']].to_string(index=False))
    results['rush'] = rush

    # ── Power play and shorthanded for reference ──
    for sit in ['Power Play', 'Shorthanded', 'Penalty Shot']:
        subset = df[df['situation'] == sit]
        if len(subset) >= 20:
            sv = round(1 - subset['is_goal'].mean(), 3)
            print(f"\n{sit.upper()}: {len(subset)} shots | SV% {sv}")

    # ── Season trends for weakest shot types ──
    weakest = shot_a.head(3)['shot_type'].tolist() if len(shot_a) >= 3 else shot_a['shot_type'].tolist()
    print(f"\nSEASON TRENDS FOR WEAKEST SHOT TYPES (even strength):")
    for shot in weakest:
        subset = main[main['shot_type'] == shot]
        by_s = subset.groupby('season').agg(
            shots=('is_goal','count'), goals=('is_goal','sum')
        )
        by_s['sv_pct'] = round(1 - by_s['goals'] / by_s['shots'], 3)
        print(f"\n{shot.upper()}:")
        print(by_s[['shots','goals','sv_pct']])

    results['overall_sv'] = es_sv
    results['main_df'] = main
    return results

def _sv_pct(df, group_col, min_shots):
    """Helper to calculate save pct grouped by one column"""
    a = df.groupby(group_col, observed=True).agg(
        shots=('is_goal','count'), goals=('is_goal','sum')
    ).reset_index()
    a['sv_pct'] = round(1 - a['goals'] / a['shots'], 3)
    a = a[a['shots'] >= min_shots].sort_values('sv_pct')
    return a

# ─── STEP 7: COMPARE TO LEAGUE BENCHMARKS ─────────────────────────────────────
def compare_to_benchmarks(goalie_name, results):
    """If league benchmarks exist, compare goalie to league averages"""

    shot_b_path  = os.path.join(BENCH_DIR, "benchmarks_shot_type_3seasons.csv")
    dist_b_path  = os.path.join(BENCH_DIR, "benchmarks_distance_3seasons.csv")
    combo_b_path = os.path.join(BENCH_DIR, "benchmarks_shot_x_distance_3seasons.csv")
    zone_b_path  = os.path.join(BENCH_DIR, "benchmarks_danger_zone_3seasons.csv")

    if not os.path.exists(shot_b_path):
        print(f"\nNo league benchmarks found. Run league_benchmarks.py first.")
        print("Then re-run this script to see how this goalie compares to the league.")
        return None

    print(f"\n{'='*60}")
    print(f"{goalie_name.upper()} vs LEAGUE BENCHMARKS")
    print(f"{'='*60}")

    shot_b  = pd.read_csv(shot_b_path)
    dist_b  = pd.read_csv(dist_b_path)
    combo_b = pd.read_csv(combo_b_path)
    zone_b  = pd.read_csv(zone_b_path)

    comparisons = {}

    # Shot type comparison
    g_shot = results['shot_type'].copy()
    comp_shot = g_shot.merge(shot_b[['shot_type','league_sv_pct']], on='shot_type', how='left')
    comp_shot['vs_league'] = round(comp_shot['sv_pct'] - comp_shot['league_sv_pct'], 3)
    comp_shot['verdict'] = comp_shot['vs_league'].apply(
        lambda x: 'WEAK' if x < -0.010 else 'STRONG' if x > 0.010 else 'AVERAGE'
    )
    print("\nSHOT TYPE vs LEAGUE (negative = worse than league for that shot type):")
    print(comp_shot[['shot_type','shots','sv_pct','league_sv_pct','vs_league','verdict']].to_string(index=False))
    comparisons['shot_type'] = comp_shot

    # Distance comparison
    g_dist = results['distance'].copy()
    comp_dist = g_dist.merge(dist_b[['distance_band','league_sv_pct']], on='distance_band', how='left')
    comp_dist['vs_league'] = round(comp_dist['sv_pct'] - comp_dist['league_sv_pct'], 3)
    comp_dist['verdict'] = comp_dist['vs_league'].apply(
        lambda x: 'WEAK' if x < -0.010 else 'STRONG' if x > 0.010 else 'AVERAGE'
    )
    print("\nDISTANCE BAND vs LEAGUE:")
    print(comp_dist[['distance_band','shots','sv_pct','league_sv_pct','vs_league','verdict']].to_string(index=False))
    comparisons['distance'] = comp_dist

    # Danger zone comparison
    g_zone = results['danger_zone'].copy()
    comp_zone = g_zone.merge(zone_b[['danger_zone','league_sv_pct']], on='danger_zone', how='left')
    comp_zone['vs_league'] = round(comp_zone['sv_pct'] - comp_zone['league_sv_pct'], 3)
    comp_zone['verdict'] = comp_zone['vs_league'].apply(
        lambda x: 'WEAK' if x < -0.010 else 'STRONG' if x > 0.010 else 'AVERAGE'
    )
    print("\nDANGER ZONE vs LEAGUE:")
    print(comp_zone[['danger_zone','shots','sv_pct','league_sv_pct','vs_league','verdict']].to_string(index=False))
    comparisons['danger_zone'] = comp_zone

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: WHAT HOLDS UP vs LEAGUE?")
    print(f"{'='*60}")

    weak_shots  = comp_shot[comp_shot['verdict'] == 'WEAK']['shot_type'].tolist()
    strong_shots = comp_shot[comp_shot['verdict'] == 'STRONG']['shot_type'].tolist()
    weak_zones  = comp_zone[comp_zone['verdict'] == 'WEAK']['danger_zone'].tolist()

    if weak_shots:
        print(f"\nGENUINE SHOT TYPE WEAKNESSES (worse than league average):")
        for _, r in comp_shot[comp_shot['verdict'] == 'WEAK'].iterrows():
            print(f"  {r['shot_type']}: {r['sv_pct']:.3f} vs league {r['league_sv_pct']:.3f} ({r['vs_league']:+.3f})")
    else:
        print(f"\nNo shot type weaknesses survive league comparison.")
        print(f"Your friend was right - these reflect shot quality not goalie weakness.")

    if strong_shots:
        print(f"\nGENUINE STRENGTHS (better than league average):")
        for _, r in comp_shot[comp_shot['verdict'] == 'STRONG'].iterrows():
            print(f"  {r['shot_type']}: {r['sv_pct']:.3f} vs league {r['league_sv_pct']:.3f} ({r['vs_league']:+.3f})")

    if weak_zones:
        print(f"\nDANGER ZONE WEAKNESSES:")
        for _, r in comp_zone[comp_zone['verdict'] == 'WEAK'].iterrows():
            print(f"  {r['danger_zone']} danger: {r['sv_pct']:.3f} vs league {r['league_sv_pct']:.3f} ({r['vs_league']:+.3f})")

    return comparisons

# ─── STEP 8: BUILD CHARTS ──────────────────────────────────────────────────────
def build_charts(goalie_name, game_info, save_dir, results, comparisons=None):
    paths = []
    last = goalie_name.split()[-1].lower()
    overall_sv = results['overall_sv']

    def wm(fig):
        fig.text(0.99, 0.01, WATERMARK, ha='right', va='bottom',
                 color=GREY, fontsize=9, alpha=0.75, style='italic')

    def style(ax):
        for sp in ['top','right']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom','left']:
            ax.spines[sp].set_color(GREY)
        ax.set_facecolor(BG)

    # ── Chart 1: Shot type (with league comparison if available) ──
    shot_data = results['shot_type']
    if comparisons and 'shot_type' in comparisons:
        shot_data = comparisons['shot_type']
        has_bench = True
    else:
        shot_data = shot_data.copy()
        shot_data['league_sv_pct'] = LEAGUE_AVG
        shot_data['vs_league'] = round(shot_data['sv_pct'] - LEAGUE_AVG, 3)
        has_bench = False

    shot_data = shot_data.dropna(subset=['sv_pct']).sort_values('vs_league')

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)
    x = np.arange(len(shot_data))
    w = 0.35

    colors_g = [RED if v < -0.010 else ORANGE if v < 0 else GREEN
                for v in shot_data['vs_league']]

    b1 = ax.bar(x - w/2, shot_data['sv_pct'], w, label=goalie_name,
                color=colors_g, alpha=0.9, edgecolor=BG)
    b2 = ax.bar(x + w/2, shot_data['league_sv_pct'], w,
                label='League Avg' if has_bench else 'League Est.',
                color=BLUE, alpha=0.55, edgecolor=BG)

    ax.set_ylim(0.76, 1.00)
    ax.set_xticks(x)
    ax.set_xticklabels(shot_data['shot_type'].tolist(), color=WHITE, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Save Percentage', fontsize=11)
    ax.legend(facecolor=BG, labelcolor=WHITE, fontsize=9)
    title_suffix = "vs League Average" if has_bench else "vs League Estimate"
    ax.set_title(f'{goalie_name} — Even Strength Save % by Shot Type\n{title_suffix} | Red = Worse | Green = Better',
                 fontsize=12, fontweight='bold', pad=12)

    for bar, val, vs in zip(b1, shot_data['sv_pct'], shot_data['vs_league']):
        sign = '+' if vs > 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.002,
                f'{val:.3f}\n({sign}{vs:.3f})',
                ha='center', va='bottom', color=WHITE, fontsize=7.5, fontweight='bold')

    style(ax)
    wm(fig)
    plt.tight_layout()
    p = os.path.join(save_dir, f"{last}_shot_type.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    paths.append(p)
    print(f"  Chart 1 saved: {os.path.basename(p)}")

    # ── Chart 2: Distance bands for key shot types ──
    main_df = results['main_df']
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG)

    weakest_shots = results['shot_type'].head(2)['shot_type'].tolist()

    for ax, shot in zip(axes, weakest_shots[:2]):
        subset = main_df[
            (main_df['shot_type'] == shot) &
            (main_df['distance_ft'] <= 40)
        ]
        dist_g = subset.groupby('distance_band', observed=True).agg(
            shots=('is_goal','count'), goals=('is_goal','sum')
        ).reset_index()
        dist_g['sv_pct'] = round(1 - dist_g['goals'] / dist_g['shots'], 3)
        dist_g = dist_g[dist_g['shots'] >= 10]

        if len(dist_g) == 0:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {shot}',
                    ha='center', va='center', color=WHITE, transform=ax.transAxes)
            style(ax)
            continue

        colors = [RED if v < overall_sv - 0.01 else ORANGE if v < overall_sv else GREEN
                  for v in dist_g['sv_pct']]
        bars = ax.bar(dist_g['distance_band'], dist_g['sv_pct'],
                      color=colors, alpha=0.85, edgecolor=BG)
        ax.set_ylim(0.70, 1.02)
        ax.axhline(overall_sv, color=WHITE, linestyle='--', alpha=0.6,
                   label=f'His avg {overall_sv}')
        ax.axhline(LEAGUE_AVG, color=YELLOW, linestyle='--', alpha=0.6,
                   label='League avg')
        ax.set_xticklabels(dist_g['distance_band'], rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{shot.title()} Shots — Save % by Distance',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('Save %', fontsize=10)
        ax.legend(facecolor=BG, labelcolor=WHITE, fontsize=8)

        for bar, val, shots in zip(bars, dist_g['sv_pct'], dist_g['shots']):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.005,
                    f'{val:.3f}\nn={shots}',
                    ha='center', va='bottom', color=WHITE, fontsize=7)

        style(ax)

    plt.suptitle(f'{goalie_name} — Save % by Distance | Weakest Shot Types | {game_info}',
                 color=WHITE, fontsize=13, fontweight='bold', y=1.02)
    wm(fig)
    plt.tight_layout()
    p = os.path.join(save_dir, f"{last}_distance.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    paths.append(p)
    print(f"  Chart 2 saved: {os.path.basename(p)}")

    # ── Chart 3: Handedness comparison ──
    hand_shot = results['shot_type_x_handedness']
    shot_types_h = hand_shot['shot_type'].unique().tolist()
    rpcts, lpcts = [], []
    for shot in shot_types_h:
        r = hand_shot[(hand_shot['shoots']=='R') & (hand_shot['shot_type']==shot)]['sv_pct'].values
        l = hand_shot[(hand_shot['shoots']=='L') & (hand_shot['shot_type']==shot)]['sv_pct'].values
        rpcts.append(float(r[0]) if len(r) > 0 else None)
        lpcts.append(float(l[0]) if len(l) > 0 else None)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG)
    x = np.arange(len(shot_types_h))
    w = 0.35

    r_vals = [v if v is not None else 0 for v in rpcts]
    l_vals = [v if v is not None else 0 for v in lpcts]

    b1 = ax.bar(x - w/2, r_vals, w, label='Right Handed', color=RED,   alpha=0.85, edgecolor=BG)
    b2 = ax.bar(x + w/2, l_vals, w, label='Left Handed',  color=GREEN, alpha=0.85, edgecolor=BG)

    ax.set_ylim(0.76, 1.00)
    ax.axhline(overall_sv, color=WHITE, linestyle='--', alpha=0.7, label=f'His avg {overall_sv}')
    ax.axhline(LEAGUE_AVG, color=YELLOW, linestyle='--', alpha=0.7, label='League avg')
    ax.set_xticks(x)
    ax.set_xticklabels(shot_types_h, color=WHITE, fontsize=11)
    ax.set_ylabel('Save Percentage', fontsize=12)
    ax.legend(facecolor=BG, labelcolor=WHITE, fontsize=10)
    ax.set_title(f'{goalie_name} — Even Strength Save % by Shooter Handedness\nRight = Red | Left = Green',
                 fontsize=13, fontweight='bold', pad=15)

    for bars, vals in [(b1, r_vals), (b2, l_vals)]:
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2., v + 0.002,
                        f'{v:.3f}', ha='center', va='bottom',
                        color=WHITE, fontsize=9, fontweight='bold')

    style(ax)
    wm(fig)
    plt.tight_layout()
    p = os.path.join(save_dir, f"{last}_handedness.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    paths.append(p)
    print(f"  Chart 3 saved: {os.path.basename(p)}")

    return paths

# ─── STEP 9: OPPONENT ROSTER ───────────────────────────────────────────────────
def get_opponent_handedness(team_abbrev):
    print(f"\nFetching {team_abbrev} roster...")
    data = fetch(f"{BASE_URL}/roster/{team_abbrev.upper()}/20252026")
    if not data:
        print("  Could not fetch roster")
        return
    print(f"\n{team_abbrev.upper()} ROSTER HANDEDNESS:")
    print("FORWARDS:")
    for p in data.get('forwards', []):
        name = f"{p.get('firstName',{}).get('default','')} {p.get('lastName',{}).get('default','')}"
        print(f"  {name}: shoots {p.get('shootsCatches','?')}")
    print("DEFENSEMEN:")
    for p in data.get('defensemen', []):
        name = f"{p.get('firstName',{}).get('default','')} {p.get('lastName',{}).get('default','')}"
        print(f"  {name}: shoots {p.get('shootsCatches','?')}")

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("HOCKEYROI - GOALIE ANALYSIS v2")
    print("Even strength | Distance bands | Rebounds | Rush shots")
    print("="*60)

    goalie_last   = input("\nGoalie last name: ").strip()
    team_abbrev   = input("Goalie team (e.g. LAK): ").strip().upper()
    opponent_team = input("Opponent team (e.g. EDM): ").strip().upper()
    game_info     = input("Game description for charts (e.g. 'Oilers vs Kings | Sat Apr 12'): ").strip()

    goalie_id, goalie_name = find_goalie(goalie_last, team_abbrev)
    if not goalie_id:
        sys.exit(1)

    save_dir = os.path.join(GOALIES_DIR, goalie_name.split()[-1])
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{goalie_name.split()[-1].lower()}_shots_all_seasons.csv")

    if os.path.exists(csv_path):
        print(f"\nExisting data found: {csv_path}")
        choice = input("Use existing data? (y) or re-pull from API? (n): ").strip().lower()
        if choice == 'y':
            df = pd.read_csv(csv_path)
            print(f"  Loaded {len(df)} shots")
            # Check if new columns exist, if not re-add features
            if 'distance_ft' not in df.columns:
                print("  Old data format detected. Adding new features...")
                df = add_features(df)
                df.to_csv(csv_path, index=False)
        else:
            games = get_game_logs(goalie_id, SEASONS)
            shots = pull_shots(games, goalie_id, goalie_name)
            df = pd.DataFrame(shots)
            df = add_handedness(df)
            df = add_features(df)
            df.to_csv(csv_path, index=False)
            print(f"  Saved {len(df)} shots to {csv_path}")
    else:
        games = get_game_logs(goalie_id, SEASONS)
        if not games:
            print("No games found.")
            sys.exit(1)
        shots = pull_shots(games, goalie_id, goalie_name)
        df = pd.DataFrame(shots)
        df = add_handedness(df)
        df = add_features(df)
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved {len(df)} shots to {csv_path}")

    # Verify handedness
    if 'shoots' not in df.columns or df['shoots'].isna().sum() > len(df) * 0.1:
        print("  Re-adding handedness...")
        df = add_handedness(df)
        df.to_csv(csv_path, index=False)

    # Run analysis
    results = run_analysis(df, goalie_name)

    # Compare to league benchmarks if available
    comparisons = compare_to_benchmarks(goalie_name, results)

    # Build charts
    print("\nBuilding charts...")
    chart_paths = build_charts(goalie_name, game_info, save_dir, results, comparisons)

    # Opponent roster
    if input(f"\nShow {opponent_team} roster handedness? (y/n): ").strip().lower() == 'y':
        get_opponent_handedness(opponent_team)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"Goalie:     {goalie_name}")
    print(f"Data saved: {csv_path}")
    print(f"Charts saved in: {save_dir}")
    for p in chart_paths:
        print(f"  {os.path.basename(p)}")
    print(f"\nGo Oilers. Post strong. @HockeyROI")

if __name__ == "__main__":
    main()
