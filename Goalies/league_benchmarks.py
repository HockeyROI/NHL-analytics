#!/usr/bin/env python3
"""
HockeyROI - League Benchmarks Script v2
Save to: NHL analysis/Goalies/league_benchmarks.py

Run once to build league average benchmarks across 3 seasons.
Then goalie_analysis_compare.py uses them for per-goalie comparisons.

Usage:
  cd "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Goalies"
  python3 league_benchmarks.py

Benchmarks saved to: NHL analysis/Goalies/Benchmarks/

What this builds:
  - All even strength shots for qualifying starters across 3 seasons
    (2023-24, 2024-25, 2025-26) with shooter handedness on every row
  - League average save % by shot type
  - League average save % by distance band (2ft increments)
  - League average save % by shot type x distance
  - League average save % by danger zone
  - League average save % by lateral zone
  - League average save % by shooter handedness
  - League average save % by rebound / rush
  - Per-goalie rankings across all 3 seasons

All raw data saved to all_goalie_shots_3seasons.csv so you can
reanalyze with different filters anytime without re-pulling.
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
MIN_STARTS  = 20
MIN_SHOTS   = 50   # minimum shots for a benchmark cell to be reported
WATERMARK   = "@HockeyROI"

NET_X = 89.0
NET_Y = 0.0

DIST_BINS   = list(range(0, 62, 2)) + [200]
DIST_LABELS = [f"{i}-{i+2}ft" for i in range(0, 60, 2)] + ["60ft+"]

ALL_TEAMS = [
    "ANA","BOS","BUF","CAR","CBJ","CGY","CHI","COL",
    "DAL","DET","EDM","FLA","LAK","MIN","MTL","NJD",
    "NSH","NYI","NYR","OTT","PHI","PIT","SEA","SJS",
    "STL","TBL","TOR","UTA","VAN","VGK","WPG","WSH"
]

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
})

# ─── HTTP ──────────────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers['User-Agent'] = 'HockeyROI-Benchmarks/2.0'

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

# ─── FEATURE HELPERS (same as goalie_analysis.py) ────────────────────────────
def calc_distance(x, y):
    if pd.isna(x) or pd.isna(y):
        return None
    dist = math.sqrt((abs(x) - NET_X)**2 + (y - NET_Y)**2)
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
        lambda r: calc_danger_zone(r['distance_ft'], r['y_coord'] if r['y_coord'] is not None else 0), axis=1)
    df['lateral']       = pd.cut(df['y_coord'], bins=[-100,-15,15,100],
                                  labels=['Left','Center','Right']).astype(str)
    df['situation']     = df['situation_code'].apply(parse_situation)
    return df

# ─── STEP 1: GET ALL STARTING GOALIES ─────────────────────────────────────────
def get_all_starting_goalies(season):
    """
    Return unique qualifying goalies for a season.
    Deduplicates by player_id so traded goalies are only counted once.
    """
    print(f"\nFinding qualifying starters ({MIN_STARTS}+ starts) across all 32 teams for {season}...")
    seen_ids   = set()
    qualifying = []
    for team in ALL_TEAMS:
        data = fetch(f"{BASE_URL}/roster/{team}/{season}")
        if not data:
            print(f"  Could not fetch {team}")
            continue
        for g in data.get('goalies', []):
            pid = g.get('id')
            if pid in seen_ids:
                continue   # already found this goalie via another team
            first = g.get('firstName', {}).get('default', '')
            last  = g.get('lastName',  {}).get('default', '')
            log   = fetch(f"{BASE_URL}/player/{pid}/game-log/{season}/2")
            if log:
                games  = log.get('gameLog', [])
                starts = len([x for x in games if x.get('decision') in ['W','L','O']])
                if starts >= MIN_STARTS:
                    seen_ids.add(pid)
                    qualifying.append({
                        'player_id': pid,
                        'name':      f"{first} {last}",
                        'team':      team,
                        'season':    season,
                        'starts':    starts
                    })
                    print(f"  Qualified: {first} {last} ({team}) — {starts} starts")
        time.sleep(0.2)
    print(f"Total qualifying goalies for {season}: {len(qualifying)}")
    return qualifying

# ─── STEP 2: PULL SHOTS FOR ONE GOALIE ────────────────────────────────────────
def pull_goalie_shots(games_list, goalie_id, goalie_name):
    all_shots = []
    for game in games_list:
        game_id   = game.get('gameId')
        game_date = game.get('gameDate')
        data      = fetch(f"{BASE_URL}/gamecenter/{game_id}/play-by-play")
        if not data:
            continue

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
                'goalie_id':          goalie_id,
                'goalie_name':        goalie_name,
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
        time.sleep(0.25)
    return all_shots

# ─── STEP 2b: ADD SHOOTER HANDEDNESS ──────────────────────────────────────────
def add_handedness(df):
    """
    Look up shoots/catches for every unique shooter in the dataset and add a
    'shoots' column.  Identical logic to goalie_analysis.py so the column
    means the same thing in both raw-shot files.
    """
    df = df.copy()
    unique_ids = df['shooting_player_id'].dropna().unique()
    print(f"\n  Looking up handedness for {len(unique_ids)} unique shooters...")
    hand = {}
    for i, pid in enumerate(unique_ids):
        data = fetch(f"{BASE_URL}/player/{int(pid)}/landing")
        hand[pid] = data.get('shootsCatches') if data else None
        if (i + 1) % 50 == 0:
            pct = round((i + 1) / len(unique_ids) * 100, 1)
            print(f"    {i+1}/{len(unique_ids)} ({pct}%)")
        time.sleep(0.12)
    df['shoots'] = df['shooting_player_id'].map(hand)
    missing = int(df['shoots'].isna().sum())
    total   = len(df)
    print(f"  Handedness complete. Missing: {missing} of {total} shots ({round(missing/total*100,1)}%)")
    return df

# ─── STEP 3: CALCULATE BENCHMARKS ─────────────────────────────────────────────
def calculate_benchmarks(all_df):
    """Calculate league averages from all goalie shots. Even strength only."""

    print("\nCalculating league benchmarks (even strength only)...")
    es = all_df[all_df['situation'] == 'Even Strength'].copy()
    print(f"Even strength shots: {len(es)} from {es['goalie_id'].nunique()} goalies")

    overall = round(1 - es['is_goal'].mean(), 3)
    print(f"League ES save percentage: {overall}")

    def bench(df, col):
        b = df.groupby(col, observed=True).agg(
            total_shots=('is_goal','count'),
            total_goals=('is_goal','sum'),
            num_goalies=('goalie_id','nunique')
        ).reset_index()
        b['league_sv_pct'] = round(1 - b['total_goals'] / b['total_shots'], 3)
        return b[b['total_shots'] >= MIN_SHOTS]

    shot_b  = bench(es, 'shot_type').sort_values('league_sv_pct')
    dist_b  = bench(es, 'distance_band').sort_values('league_sv_pct')
    zone_b  = bench(es, 'danger_zone').sort_values('league_sv_pct')
    lat_b   = bench(es, 'lateral').sort_values('league_sv_pct')
    hand_b  = bench(es[es['shoots'].notna()], 'shoots').sort_values('league_sv_pct')

    # Shot type x distance combo
    combo_b = es.groupby(['shot_type','distance_band'], observed=True).agg(
        total_shots=('is_goal','count'),
        total_goals=('is_goal','sum'),
        num_goalies=('goalie_id','nunique')
    ).reset_index()
    combo_b['league_sv_pct'] = round(1 - combo_b['total_goals'] / combo_b['total_shots'], 3)
    combo_b = combo_b[combo_b['total_shots'] >= MIN_SHOTS].sort_values('league_sv_pct')

    # Rebound benchmark
    reb_b = es.groupby('is_rebound').agg(
        total_shots=('is_goal','count'),
        total_goals=('is_goal','sum')
    ).reset_index()
    reb_b['league_sv_pct'] = round(1 - reb_b['total_goals'] / reb_b['total_shots'], 3)

    # Rush benchmark
    rush_b = es.groupby('is_rush').agg(
        total_shots=('is_goal','count'),
        total_goals=('is_goal','sum')
    ).reset_index()
    rush_b['league_sv_pct'] = round(1 - rush_b['total_goals'] / rush_b['total_shots'], 3)

    # Per goalie rankings
    per_goalie = es.groupby(['goalie_id','goalie_name']).agg(
        shots=('is_goal','count'),
        goals=('is_goal','sum')
    ).reset_index()
    per_goalie['sv_pct'] = round(1 - per_goalie['goals'] / per_goalie['shots'], 3)
    per_goalie = per_goalie.sort_values('sv_pct', ascending=False)

    print(f"\nLeague ES save % by shot type:")
    print(shot_b[['shot_type','total_shots','league_sv_pct']].to_string(index=False))
    print(f"\nLeague ES save % by danger zone:")
    print(zone_b[['danger_zone','total_shots','league_sv_pct']].to_string(index=False))
    print(f"\nGoalie ES save % rankings:")
    print(per_goalie[['goalie_name','shots','sv_pct']].head(10).to_string(index=False))

    return {
        'shot_type': shot_b,
        'distance':  dist_b,
        'combo':     combo_b,
        'zone':      zone_b,
        'lateral':   lat_b,
        'handedness': hand_b,
        'rebound':   reb_b,
        'rush':      rush_b,
        'per_goalie': per_goalie,
        'overall':   overall,
    }

# ─── STEP 4: COMPARE GOALIE TO LEAGUE ─────────────────────────────────────────
def compare_goalie(goalie_name, goalie_df, benchmarks):
    """Full comparison of one goalie to league benchmarks"""

    print(f"\n{'='*60}")
    print(f"{goalie_name.upper()} vs LEAGUE BENCHMARKS")
    print(f"{'='*60}")

    es = goalie_df[goalie_df['situation'] == 'Even Strength'].copy()
    goalie_overall = round(1 - es['is_goal'].mean(), 3)
    league_overall = benchmarks['overall']
    print(f"\n{goalie_name} ES SV%: {goalie_overall}")
    print(f"League ES SV%:  {league_overall}")
    print(f"vs League:      {goalie_overall - league_overall:+.3f}")

    results = {}

    def compare_col(goalie_df, col, bench_df):
        g = goalie_df.groupby(col, observed=True).agg(
            shots=('is_goal','count'), goals=('is_goal','sum')
        ).reset_index()
        g['sv_pct'] = round(1 - g['goals'] / g['shots'], 3)
        g = g[g['shots'] >= 20]
        merged = g.merge(bench_df[[col,'league_sv_pct']], on=col, how='left')
        merged['vs_league'] = round(merged['sv_pct'] - merged['league_sv_pct'], 3)
        merged['verdict'] = merged['vs_league'].apply(
            lambda x: 'WEAK' if x < -0.010 else 'STRONG' if x > 0.010 else 'AVERAGE'
        )
        return merged.sort_values('vs_league')

    # Shot type
    comp_shot = compare_col(es, 'shot_type', benchmarks['shot_type'])
    print(f"\nSHOT TYPE vs LEAGUE:")
    print(comp_shot[['shot_type','shots','sv_pct','league_sv_pct','vs_league','verdict']].to_string(index=False))
    results['shot_type'] = comp_shot

    # Distance band
    comp_dist = compare_col(es, 'distance_band', benchmarks['distance'])
    print(f"\nDISTANCE BAND vs LEAGUE:")
    print(comp_dist[['distance_band','shots','sv_pct','league_sv_pct','vs_league','verdict']].to_string(index=False))
    results['distance'] = comp_dist

    # Danger zone
    comp_zone = compare_col(es, 'danger_zone', benchmarks['zone'])
    print(f"\nDANGER ZONE vs LEAGUE:")
    print(comp_zone[['danger_zone','shots','sv_pct','league_sv_pct','vs_league','verdict']].to_string(index=False))
    results['danger_zone'] = comp_zone

    # Lateral
    comp_hand = compare_col(es, 'lateral', benchmarks['handedness'])
    print(f"\nLATERAL vs LEAGUE:")
    print(comp_hand[['lateral','shots','sv_pct','league_sv_pct','vs_league','verdict']].to_string(index=False))
    results['handedness'] = comp_hand

    # Rebound
    g_reb = es.groupby('is_rebound').agg(shots=('is_goal','count'), goals=('is_goal','sum')).reset_index()
    g_reb['sv_pct'] = round(1 - g_reb['goals'] / g_reb['shots'], 3)
    g_reb = g_reb.merge(benchmarks['rebound'][['is_rebound','league_sv_pct']], on='is_rebound', how='left')
    g_reb['vs_league'] = round(g_reb['sv_pct'] - g_reb['league_sv_pct'], 3)
    g_reb['type'] = g_reb['is_rebound'].map({True:'Rebound', False:'Non-Rebound'})
    print(f"\nREBOUND vs LEAGUE:")
    print(g_reb[['type','shots','sv_pct','league_sv_pct','vs_league']].to_string(index=False))
    results['rebound'] = g_reb

    # Rush
    g_rush = es.groupby('is_rush').agg(shots=('is_goal','count'), goals=('is_goal','sum')).reset_index()
    g_rush['sv_pct'] = round(1 - g_rush['goals'] / g_rush['shots'], 3)
    g_rush = g_rush.merge(benchmarks['rush'][['is_rush','league_sv_pct']], on='is_rush', how='left')
    g_rush['vs_league'] = round(g_rush['sv_pct'] - g_rush['league_sv_pct'], 3)
    g_rush['type'] = g_rush['is_rush'].map({True:'Rush Shot', False:'Non-Rush'})
    print(f"\nRUSH SHOTS vs LEAGUE:")
    print(g_rush[['type','shots','sv_pct','league_sv_pct','vs_league']].to_string(index=False))
    results['rush'] = g_rush

    # Goalie ranking
    rank_row = benchmarks['per_goalie'][benchmarks['per_goalie']['goalie_name'] == goalie_name]
    if len(rank_row) > 0:
        total_goalies = len(benchmarks['per_goalie'])
        rank = benchmarks['per_goalie'].reset_index(drop=True).index[
            benchmarks['per_goalie']['goalie_name'] == goalie_name
        ].tolist()
        if rank:
            print(f"\nOVERALL ES RANKING: {rank[0]+1} of {total_goalies} qualifying goalies")
    print(f"\nTop 10 ES SV% rankings this season:")
    print(benchmarks['per_goalie'][['goalie_name','shots','sv_pct']].head(10).to_string(index=False))

    # Final verdict
    print(f"\n{'='*60}")
    print("WHAT HOLDS UP vs LEAGUE? (answers your friend's question)")
    print(f"{'='*60}")

    weak  = comp_shot[comp_shot['verdict']=='WEAK']
    strong = comp_shot[comp_shot['verdict']=='STRONG']
    weak_z = comp_zone[comp_zone['verdict']=='WEAK']

    if len(weak) > 0:
        print(f"\nGENUINE SHOT TYPE WEAKNESSES (worse than league for same shot type):")
        for _, r in weak.iterrows():
            print(f"  {r['shot_type']}: {r['sv_pct']:.3f} vs league {r['league_sv_pct']:.3f} ({r['vs_league']:+.3f})")
        print(f"\n  These weaknesses SURVIVE shot quality adjustment.")
        print(f"  Your Substack post findings HOLD UP.")
    else:
        print(f"\n  No shot type weaknesses survive league comparison.")
        print(f"  Your friend was RIGHT about shot quality effects.")
        print(f"  The raw numbers reflect universal difficulty, not a goalie specific problem.")
        print(f"  You should update your Substack post to reflect this.")

    if len(weak_z) > 0:
        print(f"\nGENUINE DANGER ZONE WEAKNESSES:")
        for _, r in weak_z.iterrows():
            print(f"  {r['danger_zone']} danger: {r['sv_pct']:.3f} vs league {r['league_sv_pct']:.3f} ({r['vs_league']:+.3f})")

    if len(strong) > 0:
        print(f"\nGENUINE STRENGTHS (better than league for same shot type):")
        for _, r in strong.iterrows():
            print(f"  {r['shot_type']}: {r['sv_pct']:.3f} vs league {r['league_sv_pct']:.3f} ({r['vs_league']:+.3f})")

    return results

# ─── STEP 5: BUILD COMPARISON CHARTS ──────────────────────────────────────────
def build_comparison_charts(goalie_name, game_info, save_dir, comp_results):
    paths = []
    last = goalie_name.split()[-1].lower()

    def wm(fig):
        fig.text(0.99, 0.01, WATERMARK, ha='right', va='bottom',
                 color=GREY, fontsize=9, alpha=0.75, style='italic')

    def style(ax):
        for sp in ['top','right']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom','left']:
            ax.spines[sp].set_color(GREY)
        ax.set_facecolor(BG)

    # Chart 1: Shot type vs league
    data = comp_results['shot_type'].dropna(subset=['league_sv_pct']).sort_values('vs_league')
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG)
    x = np.arange(len(data))
    w = 0.35
    colors_g = [RED if v < -0.010 else ORANGE if v < 0 else GREEN for v in data['vs_league']]
    ax.bar(x - w/2, data['sv_pct'], w, label=goalie_name, color=colors_g, alpha=0.9, edgecolor=BG)
    ax.bar(x + w/2, data['league_sv_pct'], w, label='League Average', color=BLUE, alpha=0.55, edgecolor=BG)
    ax.set_ylim(0.76, 1.00)
    ax.set_xticks(x)
    ax.set_xticklabels(data['shot_type'].tolist(), color=WHITE, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Save Percentage', fontsize=11)
    ax.legend(facecolor=BG, labelcolor=WHITE, fontsize=9)
    ax.set_title(f'{goalie_name} vs League Average — Even Strength Save % by Shot Type\n'
                 f'Red = Genuinely Worse | Green = Genuinely Better | {game_info}',
                 fontsize=12, fontweight='bold', pad=12)
    for bar, val, vs in zip(ax.patches[:len(data)], data['sv_pct'], data['vs_league']):
        sign = '+' if vs > 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.002,
                f'{val:.3f}\n({sign}{vs:.3f})',
                ha='center', va='bottom', color=WHITE, fontsize=7.5, fontweight='bold')
    style(ax)
    wm(fig)
    plt.tight_layout()
    p = os.path.join(save_dir, f"{last}_vs_league_shot_type.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    paths.append(p)
    print(f"  Chart saved: {os.path.basename(p)}")

    # Chart 2: Danger zone + distance band vs league
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG)

    zone_data = comp_results['danger_zone'].dropna(subset=['league_sv_pct'])
    x = np.arange(len(zone_data))
    ax1.bar(x - w/2, zone_data['sv_pct'], w, label=goalie_name,
            color=[RED if v < -0.010 else ORANGE if v < 0 else GREEN for v in zone_data['vs_league']],
            alpha=0.9, edgecolor=BG)
    ax1.bar(x + w/2, zone_data['league_sv_pct'], w, label='League Avg', color=BLUE, alpha=0.55, edgecolor=BG)
    ax1.set_ylim(0.76, 1.00)
    ax1.set_xticks(x)
    ax1.set_xticklabels(zone_data['danger_zone'].tolist(), color=WHITE, fontsize=11)
    ax1.set_title('Save % by Danger Zone vs League', fontsize=11, fontweight='bold')
    ax1.legend(facecolor=BG, labelcolor=WHITE, fontsize=9)
    for bar, val, vs in zip(ax1.patches[:len(zone_data)], zone_data['sv_pct'], zone_data['vs_league']):
        sign = '+' if vs > 0 else ''
        ax1.text(bar.get_x() + bar.get_width()/2., val + 0.003,
                 f'{val:.3f}\n({sign}{vs:.3f})', ha='center', va='bottom',
                 color=WHITE, fontsize=9, fontweight='bold')
    style(ax1)

    dist_data = comp_results['distance'].dropna(subset=['league_sv_pct'])
    dist_data = dist_data[~dist_data['distance_band'].str.contains('60ft')].head(15)
    x2 = np.arange(len(dist_data))
    ax2.bar(x2 - w/2, dist_data['sv_pct'], w, label=goalie_name,
            color=[RED if v < -0.010 else ORANGE if v < 0 else GREEN for v in dist_data['vs_league']],
            alpha=0.9, edgecolor=BG)
    ax2.bar(x2 + w/2, dist_data['league_sv_pct'], w, label='League Avg', color=BLUE, alpha=0.55, edgecolor=BG)
    ax2.set_ylim(0.60, 1.02)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(dist_data['distance_band'].tolist(), rotation=45, ha='right', fontsize=8)
    ax2.set_title('Save % by Distance Band vs League\n(0-30ft shown)', fontsize=11, fontweight='bold')
    ax2.legend(facecolor=BG, labelcolor=WHITE, fontsize=9)
    style(ax2)

    plt.suptitle(f'{goalie_name} — Genuine Weaknesses vs League Average | {game_info}',
                 color=WHITE, fontsize=13, fontweight='bold', y=1.02)
    wm(fig)
    plt.tight_layout()
    p = os.path.join(save_dir, f"{last}_vs_league_zones.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    paths.append(p)
    print(f"  Chart saved: {os.path.basename(p)}")

    return paths

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("HOCKEYROI - LEAGUE BENCHMARKS v2")
    print("Even strength | 2ft distance bands | Rebounds | Rush shots")
    print("="*60)
    print(f"Seasons: {', '.join(SEASONS)} | Min starts per season: {MIN_STARTS}")

    os.makedirs(BENCH_DIR, exist_ok=True)

    shots_csv   = os.path.join(BENCH_DIR, "all_goalie_shots_3seasons.csv")
    shot_b_csv  = os.path.join(BENCH_DIR, "benchmarks_shot_type_3seasons.csv")
    dist_b_csv  = os.path.join(BENCH_DIR, "benchmarks_distance_3seasons.csv")
    combo_b_csv = os.path.join(BENCH_DIR, "benchmarks_shot_x_distance_3seasons.csv")
    zone_b_csv  = os.path.join(BENCH_DIR, "benchmarks_danger_zone_3seasons.csv")
    lat_b_csv   = os.path.join(BENCH_DIR, "benchmarks_lateral_3seasons.csv")
    hand_b_csv  = os.path.join(BENCH_DIR, "benchmarks_handedness_3seasons.csv")
    reb_b_csv   = os.path.join(BENCH_DIR, "benchmarks_rebound_3seasons.csv")
    rush_b_csv  = os.path.join(BENCH_DIR, "benchmarks_rush_3seasons.csv")
    rank_csv    = os.path.join(BENCH_DIR, "goalie_rankings_3seasons.csv")

    # Load or rebuild benchmarks
    bench_csvs_exist = all(os.path.exists(p) for p in [
        shot_b_csv, dist_b_csv, combo_b_csv, zone_b_csv,
        lat_b_csv, hand_b_csv, reb_b_csv, rush_b_csv, rank_csv
    ])
    if os.path.exists(shots_csv):
        if bench_csvs_exist:
            print(f"\nExisting 3-season data found.")
            choice = input("Use existing? (y) or rebuild from scratch? (n): ").strip().lower()
        else:
            print(f"\nRaw shots found but benchmarks are missing — recalculating.")
            choice = 'r'
        if choice == 'y':
            print("Loading...")
            all_df  = pd.read_csv(shots_csv)
            # Back-fill handedness if this file was built before the column was added
            if 'shoots' not in all_df.columns or all_df['shoots'].isna().mean() > 0.5:
                print("  'shoots' column missing or sparse — fetching handedness now...")
                all_df = add_handedness(all_df)
                all_df.to_csv(shots_csv, index=False)
                print("  Updated raw shots file with handedness.")
            shot_b  = pd.read_csv(shot_b_csv)
            dist_b  = pd.read_csv(dist_b_csv)
            combo_b = pd.read_csv(combo_b_csv)
            zone_b  = pd.read_csv(zone_b_csv)
            lat_b   = pd.read_csv(lat_b_csv)
            hand_b  = pd.read_csv(hand_b_csv)
            reb_b   = pd.read_csv(reb_b_csv)
            rush_b  = pd.read_csv(rush_b_csv)
            rank_df = pd.read_csv(rank_csv)
            benchmarks = {
                'shot_type': shot_b, 'distance': dist_b, 'combo': combo_b,
                'zone': zone_b, 'lateral': lat_b, 'handedness': hand_b,
                'rebound': reb_b, 'rush': rush_b, 'per_goalie': rank_df,
                'overall': round(1 - all_df[all_df['situation']=='Even Strength']['is_goal'].mean(), 3)
            }
            print(f"Loaded {len(all_df)} shots from {all_df['goalie_id'].nunique()} goalies")
        elif choice == 'r':
            print("Loading raw shots and recalculating benchmarks...")
            all_df = pd.read_csv(shots_csv)
            all_df = add_features(all_df)
            if 'shoots' not in all_df.columns or all_df['shoots'].isna().mean() > 0.5:
                print("Adding shooter handedness...")
                all_df = add_handedness(all_df)
            all_df.to_csv(shots_csv, index=False)
            print(f"Loaded {len(all_df)} shots from {all_df['goalie_id'].nunique()} goalies")
            benchmarks = calculate_benchmarks(all_df)
            benchmarks['shot_type'].to_csv(shot_b_csv,  index=False)
            benchmarks['distance'].to_csv(dist_b_csv,   index=False)
            benchmarks['combo'].to_csv(combo_b_csv,     index=False)
            benchmarks['zone'].to_csv(zone_b_csv,       index=False)
            benchmarks['lateral'].to_csv(lat_b_csv,     index=False)
            benchmarks['handedness'].to_csv(hand_b_csv, index=False)
            benchmarks['rebound'].to_csv(reb_b_csv,     index=False)
            benchmarks['rush'].to_csv(rush_b_csv,       index=False)
            benchmarks['per_goalie'].to_csv(rank_csv,   index=False)
            print(f"\nAll benchmarks saved to {BENCH_DIR}")
        else:
            choice = 'n'
    else:
        choice = 'n'

    if choice == 'n':
        all_shots = []
        for season in SEASONS:
            goalies = get_all_starting_goalies(season)
            if not goalies:
                print(f"No qualifying goalies found for {season}, skipping.")
                continue

            print(f"\nPulling shot data for {len(goalies)} goalies in {season}...")
            print("This takes approximately 30-45 minutes per season. Perfect time for a coffee.\n")

            for i, goalie in enumerate(goalies):
                print(f"\n[{i+1}/{len(goalies)}] {goalie['name']} ({goalie['team']}) {season}...")
                log = fetch(f"{BASE_URL}/player/{goalie['player_id']}/game-log/{season}/2")
                games = log.get('gameLog', []) if log else []
                if games:
                    shots = pull_goalie_shots(games, goalie['player_id'], goalie['name'])
                    for s in shots:
                        s['season'] = season
                    all_shots.extend(shots)
                    print(f"  {len(shots)} shots collected")
                else:
                    print(f"  No games found")

            # Save after every season so a crash doesn't lose everything
            if all_shots:
                _checkpoint = pd.DataFrame(all_shots)
                _checkpoint.to_csv(shots_csv, index=False)
                print(f"\n  Checkpoint saved: {len(_checkpoint)} shots → {shots_csv}")

        if not all_shots:
            print("No shots collected. Exiting.")
            return

        # Step 1: save raw shots immediately
        all_df = pd.DataFrame(all_shots)
        all_df.to_csv(shots_csv, index=False)
        print(f"\nRaw shots saved ({len(all_df)} rows): {shots_csv}")

        # Step 2: add distance/zone/situation features and re-save
        all_df = add_features(all_df)
        all_df.to_csv(shots_csv, index=False)
        print(f"Features added and saved.")

        # Step 3: add shooter handedness and re-save
        print("\nAdding shooter handedness (this takes a few minutes)...")
        all_df = add_handedness(all_df)
        all_df.to_csv(shots_csv, index=False)
        print(f"\nFinal raw data saved: {shots_csv}")
        print(f"Total: {len(all_df)} shots from {all_df['goalie_id'].nunique()} unique goalies across {', '.join(SEASONS)}")

        benchmarks = calculate_benchmarks(all_df)

        # Save all benchmark files
        benchmarks['shot_type'].to_csv(shot_b_csv,  index=False)
        benchmarks['distance'].to_csv(dist_b_csv,   index=False)
        benchmarks['combo'].to_csv(combo_b_csv,     index=False)
        benchmarks['zone'].to_csv(zone_b_csv,       index=False)
        benchmarks['lateral'].to_csv(lat_b_csv,     index=False)
        benchmarks['handedness'].to_csv(hand_b_csv, index=False)
        benchmarks['rebound'].to_csv(reb_b_csv,     index=False)
        benchmarks['rush'].to_csv(rush_b_csv,       index=False)
        benchmarks['per_goalie'].to_csv(rank_csv,   index=False)
        print(f"\nAll benchmarks saved to {BENCH_DIR}")

    # Compare a specific goalie
    print("\n" + "="*60)
    if input("\nCompare a goalie to league benchmarks? (y/n): ").strip().lower() != 'y':
        print("Done. Run again anytime to compare a goalie.")
        return

    goalie_name = input("Enter goalie full name exactly as saved (e.g. Darcy Kuemper): ").strip()
    game_info   = input("Game description for charts (e.g. 'Oilers vs Kings | Sat Apr 12'): ").strip()

    last_name  = goalie_name.split()[-1]
    goalie_csv = os.path.join(GOALIES_DIR, last_name,
                              f"{last_name.lower()}_shots_all_seasons.csv")

    if not os.path.exists(goalie_csv):
        print(f"\nCould not find goalie data at {goalie_csv}")
        print("Run goalie_analysis.py first to pull their shot data.")
        return

    goalie_df = pd.read_csv(goalie_csv)
    if 'situation' not in goalie_df.columns:
        goalie_df = add_features(goalie_df)
    print(f"Loaded {len(goalie_df)} shots for {goalie_name}")

    comp_results = compare_goalie(goalie_name, goalie_df, benchmarks)

    save_dir = os.path.join(GOALIES_DIR, last_name)
    os.makedirs(save_dir, exist_ok=True)

    print("\nBuilding comparison charts...")
    chart_paths = build_comparison_charts(goalie_name, game_info, save_dir, comp_results)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Charts saved in: {save_dir}")
    for p in chart_paths:
        print(f"  {os.path.basename(p)}")
    print(f"\nGo Oilers. Post strong. @HockeyROI")

if __name__ == "__main__":
    main()
