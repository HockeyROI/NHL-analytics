import requests
from datetime import datetime, date

# All 2026 playoff teams and their IDs
PLAYOFF_TEAMS = {
    'EDM': 22, 'ANA': 24, 'LAK': 26, 'VGK': 54,
    'DAL': 25, 'MIN': 30, 'COL': 21, 'UTA': 59,
    'WSH': 15, 'MTL': 8,  'TBL': 14, 'CAR': 12,
    'NJD': 1,  'PHI': 4,  'PIT': 5,  'BOS': 6,
    'BUF': 7,  'OTT': 9,
}

TODAY = date.today()

def calc_age(birth_date_str):
    bd = datetime.strptime(birth_date_str, '%Y-%m-%d').date()
    age = (TODAY - bd).days / 365.25
    return round(age, 1)

def get_roster_with_toi(team_abbrev, team_id):
    """Get top 6 F and top 4 D by regular season ice time with ages"""

    # Get roster
    url = f'https://api-web.nhle.com/v1/roster/{team_abbrev}/20252026'
    r = requests.get(url)
    roster = r.json()

    # Get skater stats for sorting by TOI
    stats_url = f'https://api-web.nhle.com/v1/club-stats/{team_abbrev}/20252026/2'
    r2 = requests.get(stats_url)
    stats = r2.json()

    # Build TOI lookup from skater stats (avgTimeOnIcePerGame is seconds)
    toi_lookup = {}
    for skater in stats.get('skaters', []):
        pid = skater['playerId']
        toi_sec = skater.get('avgTimeOnIcePerGame', 0) or 0
        toi_lookup[pid] = toi_sec / 60.0

    # Build player list with positions and ages
    forwards = []
    defensemen = []

    for pos_group in ['forwards', 'defensemen']:
        for p in roster.get(pos_group, []):
            pid = p['id']
            name = f"{p['firstName']['default']} {p['lastName']['default']}"
            birth = p.get('birthDate', None)
            age = calc_age(birth) if birth else None
            toi = toi_lookup.get(pid, 0)
            pos = p.get('positionCode', '')

            player_data = {
                'name': name,
                'age': age,
                'toi': toi,
                'pos': pos,
                'pid': pid
            }

            if pos_group == 'forwards':
                forwards.append(player_data)
            else:
                defensemen.append(player_data)

    # Sort by TOI descending, take top 6 F and top 4 D
    forwards_sorted = sorted([p for p in forwards if p['age']],
                              key=lambda x: x['toi'], reverse=True)[:6]
    defense_sorted = sorted([p for p in defensemen if p['age']],
                             key=lambda x: x['toi'], reverse=True)[:4]

    f_ages = [p['age'] for p in forwards_sorted]
    d_ages = [p['age'] for p in defense_sorted]

    avg_f = round(sum(f_ages) / len(f_ages), 1) if f_ages else None
    avg_d = round(sum(d_ages) / len(d_ages), 1) if d_ages else None

    return avg_f, avg_d, forwards_sorted, defense_sorted

# Run for all playoff teams
print(f"{'Team':<6} {'Top6F Avg':>10} {'Top4D Avg':>10}")
print("-" * 30)

results = []
for abbrev, team_id in PLAYOFF_TEAMS.items():
    try:
        avg_f, avg_d, fwds, dmen = get_roster_with_toi(abbrev, team_id)
        results.append({
            'team': abbrev,
            'avg_f': avg_f,
            'avg_d': avg_d,
            'forwards': fwds,
            'defense': dmen
        })
    except Exception as e:
        print(f"{abbrev}: Error - {e}")

# Sort by forward age
results_sorted = sorted(results, key=lambda x: x['avg_f'] or 99)

print(f"\n{'Team':<6} {'Top6F Avg':>10} {'Top4D Avg':>10}  {'Combined':>10}")
print("-" * 42)
for r in results_sorted:
    combined = round((r['avg_f'] + r['avg_d']) / 2, 1) if r['avg_f'] and r['avg_d'] else None
    print(f"{r['team']:<6} {r['avg_f']:>10} {r['avg_d']:>10}  {combined:>10}")

print("\n--- EDMONTON DETAIL ---")
edm = next((r for r in results if r['team'] == 'EDM'), None)
if edm:
    print("\nTop 6 Forwards by TOI:")
    for p in edm['forwards']:
        print(f"  {p['name']:<25} Age: {p['age']}  TOI: {p['toi']:.2f} min/gm")
    print("\nTop 4 Defense by TOI:")
    for p in edm['defense']:
        print(f"  {p['name']:<25} Age: {p['age']}  TOI: {p['toi']:.2f} min/gm")
