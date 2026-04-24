import requests
import pandas as pd
import numpy as np
import io, urllib.request, json
from scipy import stats

UA = {'User-Agent': 'Mozilla/5.0'}

def read_mp_csv(url):
    """MoneyPuck blocks pandas default UA -- fetch with urllib + Mozilla UA."""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return pd.read_csv(io.BytesIO(r.read()))

# -- CONFIG ------------------------------------------------------------------
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]  # MoneyPuck start-year convention
MIN_REG_GP = 20   # minimum regular season GP to qualify as starter
MIN_PO_GP  = 4    # minimum playoff GP to count as the playoff starter

# -- STEP 1: Pull regular season goalie data for each season ----------------
def get_reg_season(start_year):
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{start_year}/regular/goalies.csv"
    df = read_mp_csv(url)
    df_all = df[df['situation'] == 'all'].copy()
    df_all['season'] = start_year

    # Overall SV% using SOG
    df_all['sv_pct'] = 1 - (df_all['goals'] / df_all['ongoal'])

    # Danger tier SV% (Fenwick denominator)
    df_all['hd_sv_pct'] = 1 - (df_all['highDangerGoals'] / df_all['highDangerShots'])
    df_all['md_sv_pct'] = 1 - (df_all['mediumDangerGoals'] / df_all['mediumDangerShots'])
    df_all['ld_sv_pct'] = 1 - (df_all['lowDangerGoals'] / df_all['lowDangerShots'])

    # GSAx
    df_all['gsax'] = df_all['xGoals'] - df_all['goals']

    # 5v5 splits from same file
    df_5v5 = df[df['situation'] == '5on5'].copy()
    df_5v5['hd_sv_pct_5v5'] = 1 - (df_5v5['highDangerGoals'] / df_5v5['highDangerShots'])
    df_5v5['md_sv_pct_5v5'] = 1 - (df_5v5['mediumDangerGoals'] / df_5v5['mediumDangerShots'])
    df_5v5['ld_sv_pct_5v5'] = 1 - (df_5v5['lowDangerGoals'] / df_5v5['lowDangerShots'])
    df_5v5 = df_5v5[['name', 'team', 'hd_sv_pct_5v5', 'md_sv_pct_5v5', 'ld_sv_pct_5v5']]

    df_all = df_all.merge(df_5v5, on=['name', 'team'], how='left')

    return df_all[df_all['games_played'] >= MIN_REG_GP]

# -- STEP 2: Pull playoff goalie data for each season -----------------------
def get_playoffs(start_year):
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{start_year}/playoffs/goalies.csv"
    df = read_mp_csv(url)
    df = df[df['situation'] == 'all'].copy()
    df['season'] = start_year

    df['po_sv_pct'] = 1 - (df['goals'] / df['ongoal'])
    df['po_hd_sv_pct'] = 1 - (df['highDangerGoals'] / df['highDangerShots'])
    df['po_md_sv_pct'] = 1 - (df['mediumDangerGoals'] / df['mediumDangerShots'])
    df['po_gsax'] = df['xGoals'] - df['goals']

    return df[df['games_played'] >= MIN_PO_GP][
        ['name', 'team', 'season', 'games_played',
         'po_sv_pct', 'po_hd_sv_pct', 'po_md_sv_pct', 'po_gsax']
    ].rename(columns={'games_played': 'po_gp'})

# -- STEP 3: Pull playoff series results from NHL API -----------------------
SEASON_MAP = {
    2019: 20192020,
    2020: 20202021,
    2021: 20212022,
    2022: 20222023,
    2023: 20232024,
    2024: 20242025,
}

def get_series_results(start_year):
    nhl_season = SEASON_MAP[start_year]
    url = f"https://api-web.nhle.com/v1/playoff-series/carousel/{nhl_season}"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"  Could not fetch series for {start_year}: {e}")
        return pd.DataFrame()

    rows = []
    for round_data in data.get('rounds', []):
        for series in round_data.get('series', []):
            top = series.get('topSeed', {})
            bot = series.get('bottomSeed', {})
            winning_id = series.get('winningTeamId')
            losing_id  = series.get('losingTeamId')
            if winning_id is None or losing_id is None:
                continue  # series not complete
            id_to_abbrev = {top.get('id'): top.get('abbrev',''),
                            bot.get('id'): bot.get('abbrev','')}
            winner = id_to_abbrev.get(winning_id, '')
            loser  = id_to_abbrev.get(losing_id, '')
            if winner and loser:
                rows.append({'season': start_year, 'winner': winner, 'loser': loser})

    return pd.DataFrame(rows)

# -- MAIN --------------------------------------------------------------------
print("Pulling data for seasons:", SEASONS)
print("="*60)

all_reg = []
all_po  = []
all_series = []

for yr in SEASONS:
    print(f"\nSeason {yr}-{yr+1}...")
    try:
        reg = get_reg_season(yr)
        print(f"  Regular season: {len(reg)} qualifying goalies")
        all_reg.append(reg)
    except Exception as e:
        print(f"  Regular season failed: {e}")

    try:
        po = get_playoffs(yr)
        print(f"  Playoffs: {len(po)} qualifying goalies")
        all_po.append(po)
    except Exception as e:
        print(f"  Playoffs failed: {e}")

    try:
        series = get_series_results(yr)
        print(f"  Series results: {len(series)} completed series")
        all_series.append(series)
    except Exception as e:
        print(f"  Series results failed: {e}")

df_reg    = pd.concat(all_reg, ignore_index=True)
df_po     = pd.concat(all_po,  ignore_index=True)
df_series = pd.concat(all_series, ignore_index=True)

print(f"\n{'='*60}")
print(f"Combined: {len(df_reg)} reg-season goalie-seasons")
print(f"Combined: {len(df_po)} playoff goalie-seasons")
print(f"Combined: {len(df_series)} completed series")

# -- STEP 4: Link regular season metrics to playoff series outcomes ----------
df_winner = df_po.merge(
    df_series[['season','winner']].rename(columns={'winner':'team'}),
    on=['season','team']
).assign(won_series=1)

df_loser = df_po.merge(
    df_series[['season','loser']].rename(columns={'loser':'team'}),
    on=['season','team']
).assign(won_series=0)

df_po_outcome = pd.concat([df_winner, df_loser], ignore_index=True)

df_reg_slim = df_reg[[
    'name', 'season', 'games_played', 'sv_pct',
    'hd_sv_pct', 'md_sv_pct', 'ld_sv_pct', 'gsax',
    'hd_sv_pct_5v5', 'md_sv_pct_5v5', 'ld_sv_pct_5v5'
]].rename(columns={'games_played': 'reg_gp'})

df_merged = df_po_outcome.merge(df_reg_slim, on=['name','season'], how='inner')

print(f"\nMerged goalie-series observations: {len(df_merged)}")
print(f"Won series: {df_merged['won_series'].sum()}")
print(f"Lost series: {(df_merged['won_series']==0).sum()}")

# -- STEP 5: Correlations ----------------------------------------------------
print(f"\n{'='*60}")
print("CORRELATIONS: Regular season metric vs Won Series (1=win, 0=loss)")
print("(Point-biserial correlation -- higher = better predictor)")
print(f"{'='*60}")

metrics = [
    ('Overall SV%',        'sv_pct'),
    ('HD SV% (all sit)',   'hd_sv_pct'),
    ('MD SV% (all sit)',   'md_sv_pct'),
    ('LD SV% (all sit)',   'ld_sv_pct'),
    ('GSAx',               'gsax'),
    ('HD SV% (5v5)',       'hd_sv_pct_5v5'),
    ('MD SV% (5v5)',       'md_sv_pct_5v5'),
    ('LD SV% (5v5)',       'ld_sv_pct_5v5'),
]

for label, col in metrics:
    if col not in df_merged.columns:
        print(f"{label:<25} -- column missing")
        continue
    valid = df_merged[[col, 'won_series']].dropna()
    if len(valid) < 10:
        print(f"{label:<25} -- insufficient data ({len(valid)} obs)")
        continue
    corr, pval = stats.pointbiserialr(valid['won_series'], valid[col])
    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    print(f"{label:<25} r={corr:+.3f}  p={pval:.3f}  {sig}")

# -- STEP 6: Mean comparison -------------------------------------------------
print(f"\n{'='*60}")
print("MEAN COMPARISON: Series winners vs losers (regular season metrics)")
print(f"{'='*60}")
print(f"{'Metric':<25} {'Winners':>10} {'Losers':>10} {'Gap':>8}")
print(f"-"*55)

for label, col in metrics:
    if col not in df_merged.columns:
        continue
    winners = df_merged[df_merged['won_series']==1][col].dropna()
    losers  = df_merged[df_merged['won_series']==0][col].dropna()
    if len(winners) < 5 or len(losers) < 5:
        continue
    gap = winners.mean() - losers.mean()
    print(f"{label:<25} {winners.mean():>10.4f} {losers.mean():>10.4f} {gap:>+8.4f}")

# -- STEP 7: Save ------------------------------------------------------------
out = '/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data/goalie_md_playoff_test.csv'
df_merged.to_csv(out, index=False)
print(f"\nFull dataset saved to: {out}")
print("\nDone.")
