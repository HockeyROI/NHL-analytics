#!/usr/bin/env python3
"""
Pull all shots faced by Lukas Dostal (ID: 8480843) across 3 seasons.
Uses the same pipeline as league_benchmarks.py — same feature columns,
same helper functions, same handedness lookup.

Output: Goalies/Ducks/dostal_3seasons.csv
"""

import requests
import pandas as pd
import numpy as np
import math
import time
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL  = "https://api-web.nhle.com/v1"
DOSTAL_ID = 8480843
SEASONS   = ["20232024", "20242025", "20252026"]
OUT_CSV   = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies/Ducks/dostal_3seasons.csv"
NET_X     = 89.0
NET_Y     = 0.0

DIST_BINS   = list(range(0, 62, 2)) + [200]
DIST_LABELS = [f"{i}-{i+2}ft" for i in range(0, 60, 2)] + ["60ft+"]

# ─── HTTP ──────────────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers['User-Agent'] = 'HockeyROI-Dostal/1.0'

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

# ─── FEATURE HELPERS (identical to league_benchmarks.py) ─────────────────────
def calc_distance(x, y):
    if pd.isna(x) or pd.isna(y):
        return None
    return round(math.sqrt((abs(x) - NET_X)**2 + (y - NET_Y)**2), 1)

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
    return 0 < (time_secs - prev_event['time_secs']) <= 3

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

def add_features(df):
    df['distance_ft']   = df.apply(lambda r: calc_distance(r['x_coord'], r['y_coord']), axis=1)
    df['distance_band'] = pd.cut(df['distance_ft'], bins=DIST_BINS,
                                  labels=DIST_LABELS, right=False).astype(str)
    df['danger_zone']   = df.apply(
        lambda r: calc_danger_zone(r['distance_ft'],
                                   r['y_coord'] if r['y_coord'] is not None else 0), axis=1)
    df['lateral']       = pd.cut(df['y_coord'], bins=[-100, -15, 15, 100],
                                  labels=['Left', 'Center', 'Right']).astype(str)
    df['situation']     = df['situation_code'].apply(parse_situation)
    return df

# ─── PULL SHOTS FOR ONE GAME ──────────────────────────────────────────────────
def pull_shots_from_game(game_id, game_date, season, goalie_id):
    data = fetch(f"{BASE_URL}/gamecenter/{game_id}/play-by-play")
    if not data:
        return []

    plays      = data.get('plays', [])
    all_events = []
    for play in plays:
        all_events.append({
            'event_idx': play.get('eventId', 0),
            'type_code': play.get('typeCode'),
            'period':    play.get('periodDescriptor', {}).get('number'),
            'time_secs': time_to_secs(play.get('timeInPeriod', '0:00')),
            'x':         play.get('details', {}).get('xCoord'),
            'y':         play.get('details', {}).get('yCoord'),
        })

    shots = []
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

        prev_event = all_events[j - 1] if j > 0 else None
        is_rebound = detect_rebound(prev_event, period, time_secs)
        is_rush    = detect_rush(all_events, j, period, time_secs, x)

        shots.append({
            'goalie_id':          goalie_id,
            'goalie_name':        'Lukas Dostal',
            'season':             season,
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
    return shots

# ─── HANDEDNESS LOOKUP ────────────────────────────────────────────────────────
def add_handedness(df):
    df = df.copy()
    unique_ids = df['shooting_player_id'].dropna().unique()
    print(f"\nLooking up handedness for {len(unique_ids)} unique shooters...")
    hand = {}
    for i, pid in enumerate(unique_ids):
        data = fetch(f"{BASE_URL}/player/{int(pid)}/landing")
        hand[pid] = data.get('shootsCatches') if data else None
        if (i + 1) % 50 == 0:
            pct = round((i + 1) / len(unique_ids) * 100, 1)
            print(f"  {i+1}/{len(unique_ids)} ({pct}%)")
        time.sleep(0.12)
    df['shoots'] = df['shooting_player_id'].map(hand)
    missing = int(df['shoots'].isna().sum())
    print(f"Handedness done. Missing: {missing}/{len(df)} ({round(missing/len(df)*100,1)}%)")
    return df

# ─── SANITY CHECK ─────────────────────────────────────────────────────────────
def sanity_check(df):
    print("\n" + "="*55)
    print("SANITY CHECK — Lukas Dostal Shot Data")
    print("="*55)
    print(f"Total shots across all seasons: {len(df)}")
    print()
    for season in SEASONS:
        s = df[df['season'] == season]
        if len(s) == 0:
            print(f"  {season}: no data")
            continue
        es = s[s['situation'] == 'Even Strength']
        sv_pct = round(1 - es['is_goal'].mean(), 3) if len(es) > 0 else float('nan')
        print(f"  {season}:")
        print(f"    Total shots : {len(s)}")
        print(f"    ES shots    : {len(es)}")
        print(f"    ES SV%      : {sv_pct}")
    print("="*55)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    all_shots = []

    for season in SEASONS:
        print(f"\n{'='*50}")
        print(f"Season: {season}")
        print(f"{'='*50}")

        log = fetch(f"{BASE_URL}/player/{DOSTAL_ID}/game-log/{season}/2")
        games = log.get('gameLog', []) if log else []

        if not games:
            print(f"  No game log found for {season} — skipping.")
            continue

        # Filter to games where Dostal had a decision (started)
        started = [g for g in games if g.get('decision') in ['W', 'L', 'O']]
        print(f"  Games started: {len(started)}")

        season_shots = []
        for i, game in enumerate(started):
            game_id   = game.get('gameId')
            game_date = game.get('gameDate', '')
            shots = pull_shots_from_game(game_id, game_date, season, DOSTAL_ID)
            season_shots.extend(shots)
            print(f"  [{i+1}/{len(started)}] {game_date} game {game_id} → {len(shots)} shots")
            time.sleep(0.25)

        print(f"  Season total: {len(season_shots)} shots")
        all_shots.extend(season_shots)

    if not all_shots:
        print("\nNo shots collected. Check API connectivity and player ID.")
        return

    print(f"\nBuilding DataFrame ({len(all_shots)} rows)...")
    df = pd.DataFrame(all_shots)

    print("Adding features (distance, danger zone, situation, etc.)...")
    df = add_features(df)

    print("Adding shooter handedness...")
    df = add_handedness(df)

    print(f"\nSaving to: {OUT_CSV}")
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved. Columns: {list(df.columns)}")

    sanity_check(df)

if __name__ == "__main__":
    main()
