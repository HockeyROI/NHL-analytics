"""
v2 refinements:
  1. Single starter per team per series (highest playoff GP per season-team)
  2. First round only (roundNumber == 1) -- removes non-independence
     from same goalie appearing in multiple rounds
"""
import pandas as pd
import numpy as np
import io, urllib.request, json
from scipy import stats

UA = {'User-Agent': 'Mozilla/5.0'}

def read_mp_csv(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return pd.read_csv(io.BytesIO(r.read()))

def fetch_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
MIN_REG_GP = 20
MIN_PO_GP  = 4

SEASON_MAP = {
    2019: 20192020, 2020: 20202021, 2021: 20212022,
    2022: 20222023, 2023: 20232024, 2024: 20242025,
}

# -- Regular season data (unchanged) -----------------------------------------
def get_reg_season(start_year):
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{start_year}/regular/goalies.csv"
    df = read_mp_csv(url)
    df_all = df[df['situation'] == 'all'].copy()
    df_all['season'] = start_year
    df_all['sv_pct'] = 1 - (df_all['goals'] / df_all['ongoal'])
    df_all['hd_sv_pct'] = 1 - (df_all['highDangerGoals'] / df_all['highDangerShots'])
    df_all['md_sv_pct'] = 1 - (df_all['mediumDangerGoals'] / df_all['mediumDangerShots'])
    df_all['ld_sv_pct'] = 1 - (df_all['lowDangerGoals'] / df_all['lowDangerShots'])
    df_all['gsax'] = df_all['xGoals'] - df_all['goals']

    df_5v5 = df[df['situation'] == '5on5'].copy()
    df_5v5['hd_sv_pct_5v5'] = 1 - (df_5v5['highDangerGoals'] / df_5v5['highDangerShots'])
    df_5v5['md_sv_pct_5v5'] = 1 - (df_5v5['mediumDangerGoals'] / df_5v5['mediumDangerShots'])
    df_5v5['ld_sv_pct_5v5'] = 1 - (df_5v5['lowDangerGoals'] / df_5v5['lowDangerShots'])
    df_5v5 = df_5v5[['name','team','hd_sv_pct_5v5','md_sv_pct_5v5','ld_sv_pct_5v5']]
    df_all = df_all.merge(df_5v5, on=['name','team'], how='left')
    return df_all[df_all['games_played'] >= MIN_REG_GP]

# -- Playoff goalie data -----------------------------------------------------
def get_playoffs(start_year):
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{start_year}/playoffs/goalies.csv"
    df = read_mp_csv(url)
    df = df[df['situation'] == 'all'].copy()
    df['season'] = start_year
    return df[df['games_played'] >= MIN_PO_GP][
        ['name','team','season','games_played']
    ].rename(columns={'games_played':'po_gp'})

# -- Series results -- NOW WITH roundNumber ----------------------------------
def get_series_results(start_year):
    nhl_season = SEASON_MAP[start_year]
    url = f"https://api-web.nhle.com/v1/playoff-series/carousel/{nhl_season}"
    try:
        data = fetch_json(url)
    except Exception as e:
        print(f"  Could not fetch series for {start_year}: {e}")
        return pd.DataFrame()

    rows = []
    for round_data in data.get('rounds', []):
        round_num = round_data.get('roundNumber')
        for series in round_data.get('series', []):
            top = series.get('topSeed', {}) or {}
            bot = series.get('bottomSeed', {}) or {}
            winning_id = series.get('winningTeamId')
            losing_id  = series.get('losingTeamId')
            if winning_id is None or losing_id is None:
                continue
            id_to_abbrev = {top.get('id'): top.get('abbrev',''),
                            bot.get('id'): bot.get('abbrev','')}
            winner = id_to_abbrev.get(winning_id, '')
            loser  = id_to_abbrev.get(losing_id, '')
            if winner and loser:
                rows.append({
                    'season': start_year,
                    'round': round_num,
                    'winner': winner,
                    'loser': loser,
                })
    return pd.DataFrame(rows)

# -- MAIN --------------------------------------------------------------------
print("Pulling data...")
all_reg, all_po, all_series = [], [], []
for yr in SEASONS:
    print(f"  {yr}-{yr+1}...")
    all_reg.append(get_reg_season(yr))
    all_po.append(get_playoffs(yr))
    all_series.append(get_series_results(yr))

df_reg    = pd.concat(all_reg, ignore_index=True)
df_po     = pd.concat(all_po, ignore_index=True)
df_series = pd.concat(all_series, ignore_index=True)

print(f"\nTotal series pulled: {len(df_series)}")
print(f"Series by round:\n{df_series.groupby(['season','round']).size().unstack(fill_value=0)}")

# -- REFINEMENT 1: one goalie per (season, team) = the one with most playoff GP
df_po_top = df_po.sort_values('po_gp', ascending=False).drop_duplicates(subset=['season','team'])
print(f"\nAfter top-GP-per-team dedup: {len(df_po_top)} goalie-teams  (was {len(df_po)})")

# -- REFINEMENT 2: first round only ------------------------------------------
df_series_r1 = df_series[df_series['round'] == 1].copy()
print(f"First-round series: {len(df_series_r1)}  (total was {len(df_series)})")

# Build the goalie-series panel restricted to round 1
df_winner = df_po_top.merge(
    df_series_r1[['season','winner']].rename(columns={'winner':'team'}),
    on=['season','team']).assign(won_series=1)
df_loser = df_po_top.merge(
    df_series_r1[['season','loser']].rename(columns={'loser':'team'}),
    on=['season','team']).assign(won_series=0)
df_po_out = pd.concat([df_winner, df_loser], ignore_index=True)

# Merge in regular-season metrics
df_reg_slim = df_reg[[
    'name','season','games_played','sv_pct',
    'hd_sv_pct','md_sv_pct','ld_sv_pct','gsax',
    'hd_sv_pct_5v5','md_sv_pct_5v5','ld_sv_pct_5v5'
]].rename(columns={'games_played':'reg_gp'})

df_merged = df_po_out.merge(df_reg_slim, on=['name','season'], how='inner')

print(f"\n=== FINAL CLEAN DATASET ===")
print(f"Goalie-series observations: {len(df_merged)}")
print(f"Unique (season, team): {df_merged[['season','team']].drop_duplicates().shape[0]}")
print(f"Wins: {df_merged['won_series'].sum()}  |  Losses: {(df_merged['won_series']==0).sum()}")

# -- Correlations ------------------------------------------------------------
metrics = [
    ('Overall SV%',      'sv_pct'),
    ('HD SV% (all sit)', 'hd_sv_pct'),
    ('MD SV% (all sit)', 'md_sv_pct'),
    ('LD SV% (all sit)', 'ld_sv_pct'),
    ('GSAx',             'gsax'),
    ('HD SV% (5v5)',     'hd_sv_pct_5v5'),
    ('MD SV% (5v5)',     'md_sv_pct_5v5'),
    ('LD SV% (5v5)',     'ld_sv_pct_5v5'),
]

print(f"\n{'='*68}")
print("POINT-BISERIAL CORRELATIONS: reg-season metric vs won first-round series")
print(f"{'='*68}")
print(f"{'Metric':<25} {'r':>8}  {'p':>8}   sig")
for label, col in metrics:
    v = df_merged[[col,'won_series']].dropna()
    if len(v) < 10:
        print(f"{label:<25} insufficient data")
        continue
    corr, pval = stats.pointbiserialr(v['won_series'], v[col])
    sig = '***' if pval<0.01 else '**' if pval<0.05 else '*' if pval<0.1 else ''
    print(f"{label:<25} {corr:>+.3f}  {pval:>.3f}   {sig}")

print(f"\n{'='*68}")
print("MEAN COMPARISON (first-round series only, one goalie per team)")
print(f"{'='*68}")
print(f"{'Metric':<25} {'Winners':>10} {'Losers':>10} {'Gap':>9}   t-test p")
for label, col in metrics:
    w = df_merged[df_merged['won_series']==1][col].dropna()
    l = df_merged[df_merged['won_series']==0][col].dropna()
    if len(w) < 5 or len(l) < 5: continue
    t, p = stats.ttest_ind(w, l, equal_var=False)
    gap = w.mean() - l.mean()
    print(f"{label:<25} {w.mean():>10.4f} {l.mean():>10.4f} {gap:>+9.4f}   {p:.3f}")

# -- Logistic check on key ones (effect size in odds terms) ------------------
print(f"\n{'='*68}")
print("LOGISTIC REGRESSION: P(win series) ~ metric  (odds-ratio per +1 SD)")
print(f"{'='*68}")
try:
    from sklearn.linear_model import LogisticRegression
    for label, col in metrics:
        v = df_merged[[col,'won_series']].dropna()
        if len(v) < 20: continue
        x = ((v[col] - v[col].mean()) / v[col].std()).values.reshape(-1,1)
        y = v['won_series'].values
        m = LogisticRegression().fit(x, y)
        coef = m.coef_[0][0]
        or_per_sd = np.exp(coef)
        print(f"{label:<25} OR per +1 SD = {or_per_sd:.3f}")
except ImportError:
    print("sklearn not installed -- skipping logistic")

out = '/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data/goalie_md_playoff_test_v2.csv'
df_merged.to_csv(out, index=False)
print(f"\nSaved: {out}")
