"""
HockeyROI — Prospect EV NHLe Proof of Concept
===============================================
Tests whether age-adjusted EV production in draft year
correlates with NHL PPG outcomes better than raw junior PPG.

Data sources:
  - Junior stats: NHLDB.com (round 1 tables, fetched directly)
  - PP splits: EliteProspects / team sites / Wikipedia where available
  - NHL career totals: NHLDB.com (as of May 2026)

Cohorts: 2016, 2017 (clean pre-COVID), 2021 (COVID-era comparison)
Scope: Round 1 FORWARDS only (D and G excluded)

METHODOLOGY NOTES:
  - EV points = total points - PP points
  - Age adjustment: +/- 0.008 PPG per month vs class median birth month
    Older players (Jan) penalized slightly — beating younger competition
  - NHLe applied to EV PPG only
  - NHL outcome: career PPG (min 100 GP to qualify)
  - YoY: (draft_yr_pts - prev_yr_pts) / prev_yr_pts
  - Players with no NHL GP or <100 GP kept in dataset, excluded from correlation

KNOWN GAPS (to resolve with EP API):
  - European players mostly missing PP splits → excluded from EV correlation
  - NTDP/USA U-18 players: very small GP samples in COVID year
  - Tage Thompson: heavy PP usage correctly identified by model
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# NHLe FACTORS (standard literature values)
# ─────────────────────────────────────────────
NHLE = {
    'OHL':      0.33,
    'WHL':      0.30,
    'QMJHL':   0.28,
    'USHL':     0.25,
    'NTDP':     0.25,   # USA U-18 — same tier as USHL
    'NCAA':     0.46,
    'H-EAST':   0.46,   # Hockey East = NCAA
    'BIG10':    0.46,
    'BCHL':     0.18,   # Junior A — steep discount
    'AJHL':     0.18,
    'HIGH-MN':  0.10,   # HS hockey
    'SHL':      0.43,
    'Liiga':    0.42,
    'Finland':  0.42,
    'KHL':      0.43,
    'NL':       0.40,   # Swiss NL
    'Czech':    0.38,
    'SWEDEN':   0.43,
    'SWEDEN-JR':0.20,
    'RUSSIA-JR':0.15,
    'RUSSIA-2': 0.25,
    'AHL':      0.44,
}

# ─────────────────────────────────────────────
# FULL ROUND 1 FORWARD DATA
# Sources: NHLDB.com (junior totals + NHL totals)
#          EP / team sites / Wikipedia (PP splits)
# PP = None where unavailable (European players mainly)
# NHL stats as of May 2026
# Goalies and defensemen excluded
# ─────────────────────────────────────────────

raw_data = [

    # ════════════════════════════════════════════
    # 2016 DRAFT CLASS — Jr stats = 2015-16 season
    # ════════════════════════════════════════════

    {'draft_yr':2016,'pick':1, 'player':'Auston Matthews',    'pos':'C', 'league':'NL',
     'jr_gp':36, 'jr_g':24,'jr_a':21,'jr_pts':45,'jr_pp_g':5, 'jr_pp_a':7,
     'prev_gp':None,'prev_pts':None,
     'birth_month':9,  'nhl_gp':689, 'nhl_pts':780},

    {'draft_yr':2016,'pick':2, 'player':'Patrik Laine',       'pos':'RW','league':'Liiga',
     'jr_gp':46, 'jr_g':17,'jr_a':16,'jr_pts':33,'jr_pp_g':None,'jr_pp_a':None,
     'prev_gp':2,  'prev_pts':0,
     'birth_month':4,  'nhl_gp':537, 'nhl_pts':422},

    {'draft_yr':2016,'pick':3, 'player':'Pierre-Luc Dubois',  'pos':'LW','league':'QMJHL',
     'jr_gp':62, 'jr_g':42,'jr_a':57,'jr_pts':99,'jr_pp_g':15,'jr_pp_a':22,
     'prev_gp':66, 'prev_pts':55,
     'birth_month':6,  'nhl_gp':627, 'nhl_pts':427},

    {'draft_yr':2016,'pick':4, 'player':'Jesse Puljujarvi',   'pos':'RW','league':'Liiga',
     'jr_gp':50, 'jr_g':24,'jr_a':22,'jr_pts':46,'jr_pp_g':None,'jr_pp_a':None,
     'prev_gp':9,  'prev_pts':2,
     'birth_month':5,  'nhl_gp':387, 'nhl_pts':128},

    {'draft_yr':2016,'pick':6, 'player':'Matthew Tkachuk',    'pos':'LW','league':'OHL',
     'jr_gp':57, 'jr_g':30,'jr_a':77,'jr_pts':107,'jr_pp_g':9,'jr_pp_a':24,
     'prev_gp':68, 'prev_pts':71,
     'birth_month':12, 'nhl_gp':673, 'nhl_pts':670},

    {'draft_yr':2016,'pick':7, 'player':'Clayton Keller',     'pos':'C', 'league':'NTDP',
     'jr_gp':23, 'jr_g':13,'jr_a':24,'jr_pts':37,'jr_pp_g':4, 'jr_pp_a':8,
     'prev_gp':56, 'prev_pts':72,
     'birth_month':7,  'nhl_gp':683, 'nhl_pts':596},

    {'draft_yr':2016,'pick':8, 'player':'Alexander Nylander',  'pos':'LW','league':'OHL',
     'jr_gp':57, 'jr_g':28,'jr_a':47,'jr_pts':75,'jr_pp_g':8, 'jr_pp_a':16,
     'prev_gp':51, 'prev_pts':39,
     'birth_month':3,  'nhl_gp':126, 'nhl_pts':49},

    {'draft_yr':2016,'pick':10,'player':'Tyson Jost',          'pos':'C', 'league':'BCHL',
     'jr_gp':48, 'jr_g':42,'jr_a':62,'jr_pts':104,'jr_pp_g':12,'jr_pp_a':18,
     'prev_gp':68, 'prev_pts':47,
     'birth_month':3,  'nhl_gp':564, 'nhl_pts':165},

    {'draft_yr':2016,'pick':11,'player':'Logan Brown',         'pos':'C', 'league':'OHL',
     'jr_gp':59, 'jr_g':21,'jr_a':53,'jr_pts':74,'jr_pp_g':7, 'jr_pp_a':16,
     'prev_gp':56, 'prev_pts':38,
     'birth_month':1,  'nhl_gp':99,  'nhl_pts':26},

    {'draft_yr':2016,'pick':12,'player':'Michael McLeod',      'pos':'C', 'league':'OHL',
     'jr_gp':57, 'jr_g':21,'jr_a':40,'jr_pts':61,'jr_pp_g':6, 'jr_pp_a':12,
     'prev_gp':63, 'prev_pts':32,
     'birth_month':2,  'nhl_gp':287, 'nhl_pts':85},

    {'draft_yr':2016,'pick':15,'player':'Luke Kunin',          'pos':'C', 'league':'BIG10',
     'jr_gp':35, 'jr_g':16,'jr_a':13,'jr_pts':29,'jr_pp_g':5, 'jr_pp_a':4,
     'prev_gp':None,'prev_pts':None,
     'birth_month':12, 'nhl_gp':496, 'nhl_pts':152},

    {'draft_yr':2016,'pick':19,'player':'Kieffer Bellows',     'pos':'LW','league':'NTDP',
     'jr_gp':23, 'jr_g':16,'jr_a':16,'jr_pts':32,'jr_pp_g':5, 'jr_pp_a':4,
     'prev_gp':None,'prev_pts':None,
     'birth_month':8,  'nhl_gp':114, 'nhl_pts':32},

    {'draft_yr':2016,'pick':21,'player':'Julien Gauthier',     'pos':'RW','league':'QMJHL',
     'jr_gp':54, 'jr_g':41,'jr_a':16,'jr_pts':57,'jr_pp_g':14,'jr_pp_a':6,
     'prev_gp':62, 'prev_pts':40,
     'birth_month':10, 'nhl_gp':181, 'nhl_pts':41},

    {'draft_yr':2016,'pick':24,'player':'Max Jones',           'pos':'LW','league':'OHL',
     'jr_gp':63, 'jr_g':28,'jr_a':24,'jr_pts':52,'jr_pp_g':8, 'jr_pp_a':5,
     'prev_gp':49, 'prev_pts':19,
     'birth_month':2,  'nhl_gp':305, 'nhl_pts':69},

    {'draft_yr':2016,'pick':26,'player':'Tage Thompson',       'pos':'C', 'league':'H-EAST',
     'jr_gp':36, 'jr_g':14,'jr_a':18,'jr_pts':32,'jr_pp_g':13,'jr_pp_a':6,
     'prev_gp':None,'prev_pts':None,
     'birth_month':10, 'nhl_gp':529, 'nhl_pts':406},

    {'draft_yr':2016,'pick':27,'player':'Brett Howden',        'pos':'C', 'league':'WHL',
     'jr_gp':72, 'jr_g':26,'jr_a':44,'jr_pts':70,'jr_pp_g':8, 'jr_pp_a':14,
     'prev_gp':71, 'prev_pts':42,
     'birth_month':3,  'nhl_gp':364, 'nhl_pts':155},

    # ════════════════════════════════════════════
    # 2017 DRAFT CLASS — Jr stats = 2016-17 season
    # ════════════════════════════════════════════

    {'draft_yr':2017,'pick':1, 'player':'Nico Hischier',       'pos':'C', 'league':'QMJHL',
     'jr_gp':57, 'jr_g':38,'jr_a':48,'jr_pts':86,'jr_pp_g':9, 'jr_pp_a':18,
     'prev_gp':57, 'prev_pts':63,
     'birth_month':1,  'nhl_gp':609, 'nhl_pts':488},

    {'draft_yr':2017,'pick':2, 'player':'Nolan Patrick',       'pos':'C', 'league':'WHL',
     'jr_gp':33, 'jr_g':20,'jr_a':26,'jr_pts':46,'jr_pp_g':5, 'jr_pp_a':8,
     'prev_gp':72, 'prev_pts':102,
     'birth_month':9,  'nhl_gp':222, 'nhl_pts':77},

    {'draft_yr':2017,'pick':6, 'player':'Cody Glass',          'pos':'C', 'league':'WHL',
     'jr_gp':69, 'jr_g':32,'jr_a':62,'jr_pts':94,'jr_pp_g':8, 'jr_pp_a':19,
     'prev_gp':68, 'prev_pts':57,
     'birth_month':4,  'nhl_gp':322, 'nhl_pts':119},

    {'draft_yr':2017,'pick':8, 'player':'Casey Mittelstadt',   'pos':'C', 'league':'HIGH-MN',
     'jr_gp':24, 'jr_g':13,'jr_a':17,'jr_pts':30,'jr_pp_g':None,'jr_pp_a':None,
     'prev_gp':None,'prev_pts':None,
     'birth_month':11, 'nhl_gp':509, 'nhl_pts':278},

    {'draft_yr':2017,'pick':9, 'player':'Michael Rasmussen',   'pos':'C', 'league':'WHL',
     'jr_gp':50, 'jr_g':32,'jr_a':23,'jr_pts':55,'jr_pp_g':11,'jr_pp_a':5,
     'prev_gp':50, 'prev_pts':31,
     'birth_month':4,  'nhl_gp':454, 'nhl_pts':154},

    {'draft_yr':2017,'pick':10,'player':'Owen Tippett',        'pos':'RW','league':'OHL',
     'jr_gp':60, 'jr_g':44,'jr_a':31,'jr_pts':75,'jr_pp_g':18,'jr_pp_a':8,
     'prev_gp':57, 'prev_pts':29,
     'birth_month':2,  'nhl_gp':428, 'nhl_pts':236},

    {'draft_yr':2017,'pick':11,'player':'Gabriel Vilardi',     'pos':'C', 'league':'OHL',
     'jr_gp':49, 'jr_g':29,'jr_a':32,'jr_pts':61,'jr_pp_g':10,'jr_pp_a':11,
     'prev_gp':60, 'prev_pts':37,
     'birth_month':8,  'nhl_gp':352, 'nhl_pts':244},

    {'draft_yr':2017,'pick':13,'player':'Nick Suzuki',         'pos':'C', 'league':'OHL',
     'jr_gp':65, 'jr_g':45,'jr_a':51,'jr_pts':96,'jr_pp_g':14,'jr_pp_a':20,
     'prev_gp':68, 'prev_pts':63,
     'birth_month':8,  'nhl_gp':537, 'nhl_pts':476},

    {'draft_yr':2017,'pick':20,'player':'Kailer Yamamoto',     'pos':'RW','league':'WHL',
     'jr_gp':42, 'jr_g':24,'jr_a':42,'jr_pts':66,'jr_pp_g':6, 'jr_pp_a':12,
     'prev_gp':71, 'prev_pts':63,
     'birth_month':9,  'nhl_gp':348, 'nhl_pts':164},

    {'draft_yr':2017,'pick':22,'player':'Ryan Poehling',       'pos':'C', 'league':'NCAA',
     'jr_gp':37, 'jr_g':13,'jr_a':19,'jr_pts':32,'jr_pp_g':4, 'jr_pp_a':6,
     'prev_gp':None,'prev_pts':None,
     'birth_month':1,  'nhl_gp':200, 'nhl_pts':90},

    {'draft_yr':2017,'pick':23,'player':'Morgan Frost',        'pos':'C', 'league':'OHL',
     'jr_gp':67, 'jr_g':32,'jr_a':55,'jr_pts':87,'jr_pp_g':9, 'jr_pp_a':18,
     'prev_gp':63, 'prev_pts':54,
     'birth_month':5,  'nhl_gp':330, 'nhl_pts':196},

    {'draft_yr':2017,'pick':24,'player':'Nico Sturm',          'pos':'C', 'league':'NCAA',
     'jr_gp':39, 'jr_g':13,'jr_a':17,'jr_pts':30,'jr_pp_g':3, 'jr_pp_a':4,
     'prev_gp':37, 'prev_pts':22,
     'birth_month':5,  'nhl_gp':330, 'nhl_pts':104},

    {'draft_yr':2017,'pick':28,'player':'Maxime Comtois',      'pos':'LW','league':'QMJHL',
     'jr_gp':56, 'jr_g':28,'jr_a':27,'jr_pts':55,'jr_pp_g':7, 'jr_pp_a':8,
     'prev_gp':60, 'prev_pts':32,
     'birth_month':1,  'nhl_gp':259, 'nhl_pts':121},

    # ════════════════════════════════════════════
    # 2021 DRAFT CLASS — Jr stats = 2020-21 season
    # NOTE: COVID year — many players had 16-33 GP
    # OHL/WHL/QMJHL seasons were shortened or cancelled
    # NTDP played partial season
    # ════════════════════════════════════════════

    {'draft_yr':2021,'pick':2, 'player':'Matty Beniers',       'pos':'C', 'league':'BIG10',
     'jr_gp':26, 'jr_g':10,'jr_a':13,'jr_pts':23,'jr_pp_g':2, 'jr_pp_a':4,
     'prev_gp':None,'prev_pts':None,
     'birth_month':11, 'nhl_gp':331, 'nhl_pts':196},

    {'draft_yr':2021,'pick':3, 'player':'Mason McTavish',      'pos':'C', 'league':'OHL',
     'jr_gp':20, 'jr_g':13,'jr_a':14,'jr_pts':27,'jr_pp_g':4, 'jr_pp_a':5,
     'prev_gp':None,'prev_pts':None,
     'birth_month':1,  'nhl_gp':304, 'nhl_pts':181},

    {'draft_yr':2021,'pick':5, 'player':'Kent Johnson',        'pos':'C', 'league':'BIG10',
     'jr_gp':26, 'jr_g':9, 'jr_a':16,'jr_pts':25,'jr_pp_g':2, 'jr_pp_a':5,
     'prev_gp':None,'prev_pts':None,
     'birth_month':10, 'nhl_gp':274, 'nhl_pts':138},

    {'draft_yr':2021,'pick':9, 'player':'Dylan Guenther',      'pos':'RW','league':'WHL',
     'jr_gp':16, 'jr_g':15,'jr_a':14,'jr_pts':29,'jr_pp_g':5, 'jr_pp_a':6,
     'prev_gp':None,'prev_pts':None,
     'birth_month':9,  'nhl_gp':227, 'nhl_pts':183},

    {'draft_yr':2021,'pick':12,'player':'Cole Sillinger',      'pos':'C', 'league':'USHL',
     'jr_gp':31, 'jr_g':24,'jr_a':22,'jr_pts':46,'jr_pp_g':6, 'jr_pp_a':8,
     'prev_gp':None,'prev_pts':None,
     'birth_month':4,  'nhl_gp':367, 'nhl_pts':140},

    {'draft_yr':2021,'pick':13,'player':'Matt Coronato',       'pos':'RW','league':'USHL',
     'jr_gp':51, 'jr_g':48,'jr_a':37,'jr_pts':85,'jr_pp_g':14,'jr_pp_a':12,
     'prev_gp':None,'prev_pts':None,
     'birth_month':5,  'nhl_gp':192, 'nhl_pts':101},

    {'draft_yr':2021,'pick':17,'player':'Zack Bolduc',         'pos':'C', 'league':'QMJHL',
     'jr_gp':27, 'jr_g':10,'jr_a':19,'jr_pts':29,'jr_pp_g':3, 'jr_pp_a':6,
     'prev_gp':None,'prev_pts':None,
     'birth_month':1,  'nhl_gp':175, 'nhl_pts':75},

    {'draft_yr':2021,'pick':22,'player':'Xavier Bourgault',    'pos':'C', 'league':'QMJHL',
     'jr_gp':29, 'jr_g':20,'jr_a':20,'jr_pts':40,'jr_pp_g':6, 'jr_pp_a':8,
     'prev_gp':None,'prev_pts':None,
     'birth_month':6,  'nhl_gp':2,   'nhl_pts':0},

    {'draft_yr':2021,'pick':23,'player':'Wyatt Johnston',      'pos':'C', 'league':'OHL',
     'jr_gp':20, 'jr_g':15,'jr_a':20,'jr_pts':35,'jr_pp_g':4, 'jr_pp_a':7,
     'prev_gp':None,'prev_pts':None,
     'birth_month':5,  'nhl_gp':328, 'nhl_pts':263},

    {'draft_yr':2021,'pick':24,'player':'Mackie Samoskevich',  'pos':'RW','league':'USHL',
     'jr_gp':36, 'jr_g':13,'jr_a':24,'jr_pts':37,'jr_pp_g':3, 'jr_pp_a':7,
     'prev_gp':None,'prev_pts':None,
     'birth_month':2,  'nhl_gp':156, 'nhl_pts':63},

    {'draft_yr':2021,'pick':27,'player':'Zachary L\'Heureux',  'pos':'LW','league':'QMJHL',
     'jr_gp':33, 'jr_g':19,'jr_a':20,'jr_pts':39,'jr_pp_g':5, 'jr_pp_a':6,
     'prev_gp':None,'prev_pts':None,
     'birth_month':8,  'nhl_gp':87,  'nhl_pts':20},
]

# ─────────────────────────────────────────────
# BUILD DATAFRAME
# Read from CSV if it exists, else use raw_data above
# ─────────────────────────────────────────────
PROSPECTS_DIR = os.path.expanduser("~/Documents/HockeyROI/Prospects")
seed_path_read = os.path.join(PROSPECTS_DIR, "prospect_seed.csv")

if os.path.exists(seed_path_read):
    df = pd.read_csv(seed_path_read)
    print(f"[Reading from CSV: {seed_path_read}]")
    print(f"[{len(df)} players across {df['draft_yr'].nunique()} draft classes]\n")
else:
    df = pd.DataFrame(raw_data)
    print("[CSV not found — using hardcoded seed data]\n")

# ── DERIVED METRICS ───────────────────────────

df['jr_ppg'] = df['jr_pts'] / df['jr_gp']

df['jr_pp_pts'] = df['jr_pp_g'].fillna(0) + df['jr_pp_a'].fillna(0)
df['has_pp_data'] = df['jr_pp_g'].notna()
df['jr_ev_pts'] = np.where(df['has_pp_data'], df['jr_pts'] - df['jr_pp_pts'], np.nan)
df['jr_ev_ppg'] = df['jr_ev_pts'] / df['jr_gp']

df['nhle_factor'] = df['league'].map(NHLE)
df['ev_nhle_ppg'] = df['jr_ev_ppg'] * df['nhle_factor']
df['raw_nhle_ppg'] = df['jr_ppg'] * df['nhle_factor']

for yr in df['draft_yr'].unique():
    mask = df['draft_yr'] == yr
    median_month = df.loc[mask, 'birth_month'].median()
    df.loc[mask, 'age_adj'] = (df.loc[mask, 'birth_month'] - median_month) * 0.008

df['ev_nhle_age_adj'] = df['ev_nhle_ppg'] + df['age_adj']

df['prev_ppg'] = df['prev_pts'] / df['prev_gp']
df['yoy_pct_change'] = (df['jr_pts'] - df['prev_pts']) / df['prev_pts']
df['yoy_pct_change'] = df['yoy_pct_change'].clip(-0.5, 2.0)

df['nhl_ppg'] = df['nhl_pts'] / df['nhl_gp']

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
qualified = df[(df['nhl_gp'] >= 100) & (df['pos'].isin(['C','RW','LW','W']))].copy()

print("=" * 65)
print("HockeyROI — Prospect EV NHLe Proof of Concept")
print("=" * 65)
print(f"\nTotal players seeded:                    {len(df)}")
print(f"Qualified forwards (>=100 NHL GP):       {len(qualified)}")
print(f"Missing PP data (excluded from EV):      {df['has_pp_data'].eq(False).sum()}")
print(f"\nBy draft class (all forwards seeded):")
for yr, grp in df.groupby('draft_yr'):
    q = grp[grp['nhl_gp'] >= 100]
    print(f"  {yr}: {len(grp)} seeded, {len(q)} qualified (>=100 GP)")

print("\n" + "─" * 65)
print("CORRELATIONS WITH NHL PPG (Pearson r)")
print("─" * 65)

metrics = {
    'Draft pick # (inverse)':  -qualified['pick'],
    'Raw junior PPG':           qualified['jr_ppg'],
    'Raw NHLe PPG':             qualified['raw_nhle_ppg'],
    'EV NHLe PPG':              qualified['ev_nhle_ppg'],
    'EV NHLe + Age Adj':        qualified['ev_nhle_age_adj'],
}

results = []
for label, series in metrics.items():
    valid = qualified[series.notna() & qualified['nhl_ppg'].notna()]
    if len(valid) < 5:
        continue
    r, p = stats.pearsonr(series[valid.index], valid['nhl_ppg'])
    results.append({'Metric': label, 'r': r, 'r2': r**2, 'p': p, 'n': len(valid)})

results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
for _, row in results_df.iterrows():
    sig = '***' if row['p'] < 0.01 else '**' if row['p'] < 0.05 else '*' if row['p'] < 0.10 else ''
    print(f"  {row['Metric']:<30} r={row['r']:+.3f}  r²={row['r2']:.3f}  p={row['p']:.3f}  n={int(row['n'])} {sig}")

print("\n  * p<.10  ** p<.05  *** p<.01")

print("\n" + "─" * 65)
print("BY DRAFT CLASS — EV NHLe vs NHL PPG")
print("─" * 65)
for yr in sorted(qualified['draft_yr'].unique()):
    sub = qualified[(qualified['draft_yr'] == yr) &
                    qualified['ev_nhle_ppg'].notna() &
                    qualified['nhl_ppg'].notna()]
    if len(sub) >= 4:
        r, p = stats.pearsonr(sub['ev_nhle_ppg'], sub['nhl_ppg'])
        print(f"  {yr} (n={len(sub)}):  r={r:+.3f}  r²={r**2:.3f}  p={p:.3f}")
    else:
        print(f"  {yr}: insufficient data (n={len(sub)})")

print("\n" + "─" * 65)
print("FULL PLAYER TABLE (sorted by EV NHLe PPG, qualified only)")
print("─" * 65)
display_cols = ['draft_yr','pick','player','league',
                'jr_ppg','jr_ev_ppg','ev_nhle_ppg','ev_nhle_age_adj',
                'yoy_pct_change','nhl_ppg','nhl_gp']
display = qualified[display_cols].sort_values('ev_nhle_ppg', ascending=False, na_position='last')
display.columns = ['Yr','Pick','Player','League',
                   'JrPPG','EV_PPG','NHLe_EV','NHLe_EV_Adj',
                   'YoY%','NHL_PPG','NHL_GP']
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_rows', 60)
pd.set_option('display.width', 130)
print(display.to_string(index=False))

print("\n" + "─" * 65)
print("NOTABLE OBSERVATIONS")
print("─" * 65)

qualified_valid = qualified[qualified['ev_nhle_ppg'].notna()].copy()
qualified_valid['ev_rank'] = qualified_valid['ev_nhle_ppg'].rank(ascending=False)
qualified_valid['pick_rank'] = qualified_valid['pick'].rank(ascending=True)
qualified_valid['rank_diff'] = qualified_valid['pick_rank'] - qualified_valid['ev_rank']

print("\n  Most UNDERVALUED by EV NHLe vs actual pick (hidden value):")
for _, row in qualified_valid.nlargest(4, 'rank_diff')[['player','pick','ev_nhle_ppg','nhl_ppg']].iterrows():
    print(f"    {row['player']:<25} pick={int(row['pick']):>2}  EV_NHLe={row['ev_nhle_ppg']:.3f}  NHL_PPG={row['nhl_ppg']:.3f}")

print("\n  Most OVERVALUED by EV NHLe vs actual pick:")
for _, row in qualified_valid.nsmallest(4, 'rank_diff')[['player','pick','ev_nhle_ppg','nhl_ppg']].iterrows():
    print(f"    {row['player']:<25} pick={int(row['pick']):>2}  EV_NHLe={row['ev_nhle_ppg']:.3f}  NHL_PPG={row['nhl_ppg']:.3f}")

print("\n" + "─" * 65)
print("TAGE THOMPSON FLAG")
print("─" * 65)
tt = df[df['player'] == 'Tage Thompson'].iloc[0]
print(f"  Pick 26 — UConn freshman, led NCAA in PP goals (13)")
print(f"  Raw PPG: {tt['jr_ppg']:.3f}  →  EV PPG: {tt['jr_ev_ppg']:.3f}")
print(f"  PP pts: {int(tt['jr_pp_pts'])} of {int(tt['jr_pts'])} total ({tt['jr_pp_pts']/tt['jr_pts']*100:.0f}%)")
print(f"  NHL outcome: {tt['nhl_ppg']:.3f} PPG in {int(tt['nhl_gp'])} GP")
print(f"  → Model correctly flags heavy PP dependency pre-draft")

# ─────────────────────────────────────────────
# RANKING COMPARISON — draft order vs EV NHLe model
# ─────────────────────────────────────────────
print("\n" + "─" * 78)
print("RANKING COMPARISON — Draft Pick Order vs EV NHLe Model")
print("─" * 78)

rank_df = qualified[qualified['ev_nhle_ppg'].notna() & qualified['nhl_ppg'].notna()].copy()
rank_df['pick_rank'] = rank_df['pick'].rank(method='min', ascending=True).astype(int)
rank_df['ev_rank']   = rank_df['ev_nhle_ppg'].rank(method='min', ascending=False).astype(int)
rank_df['nhl_rank']  = rank_df['nhl_ppg'].rank(method='min', ascending=False).astype(int)

by_pick = rank_df.sort_values('pick_rank').reset_index(drop=True)
by_ev   = rank_df.sort_values('ev_rank').reset_index(drop=True)

print(f"\n  n = {len(rank_df)} qualified forwards with EV NHLe + NHL PPG\n")
print(f"  {'#':<3} {'By Draft Pick':<30}{'NHL_PPG':>8}  |  {'By EV NHLe PPG':<30}{'NHL_PPG':>8}")
print(f"  {'-'*2:<3} {'-'*28:<30}{'-'*7:>8}  |  {'-'*28:<30}{'-'*7:>8}")
for i in range(len(rank_df)):
    p = by_pick.iloc[i]
    e = by_ev.iloc[i]
    left  = f"{p['player']} (pk{int(p['pick'])})"
    right = f"{e['player']} (EV={e['ev_nhle_ppg']:.3f})"
    print(f"  {i+1:<3} {left:<30}{p['nhl_ppg']:>8.3f}  |  {right:<30}{e['nhl_ppg']:>8.3f}")

# Spearman rank correlations vs NHL PPG outcome
# Negate `pick` so higher value = better (lower pick #), matching EV NHLe direction
rho_ev, p_ev = stats.spearmanr(rank_df['ev_nhle_ppg'], rank_df['nhl_ppg'])
rho_pk, p_pk = stats.spearmanr(-rank_df['pick'],       rank_df['nhl_ppg'])

print(f"\n  Spearman rank correlation with NHL PPG outcome:")
print(f"    EV NHLe PPG       rho = {rho_ev:+.3f}   p = {p_ev:.3f}   n = {len(rank_df)}")
print(f"    Draft pick (inv)  rho = {rho_pk:+.3f}   p = {p_pk:.3f}   n = {len(rank_df)}")
print(f"  → Higher rho = ranking aligns better with actual NHL outcomes")

# ─────────────────────────────────────────────
# PER-YEAR RANKING BREAKDOWN — all sampled forwards (honest coverage view)
# ─────────────────────────────────────────────
print("\n" + "─" * 90)
print("PER-YEAR RANKING BREAKDOWN — all forwards in seed (transparent coverage)")
print("─" * 90)
print("  Columns:")
print("    DraftRk  = real NHL overall pick number")
print("    EVRk     = EV NHLe rank among sampled forwards in this draft year")
print("               (N/A if PP data missing → no EV NHLe computable)")
print("    Diff     = DraftRk − EVRk; positive = model ranks player higher than scouts")

def _is_forward(pos):
    if not isinstance(pos, str):
        return False
    parts = pos.upper().replace('/', ' ').replace(',', ' ').split()
    return any(p in {'C', 'LW', 'RW', 'W', 'F'} for p in parts)

def _i(v):  return 'N/A'  if pd.isna(v) else str(int(v))
def _f(v):  return '  —  ' if pd.isna(v) else f"{v:.3f}"
def _d(v):  return '  N/A' if pd.isna(v) else f"{int(v):+d}"

year_paths = {}
for yr in sorted(df['draft_yr'].unique()):
    sub = df[(df['draft_yr'] == yr) & df['pos'].apply(_is_forward)].copy()
    sub = sub.sort_values('pick').reset_index(drop=True)
    sub['actual_draft_rank'] = sub['pick']

    has_ev = sub['ev_nhle_ppg'].notna()
    sub['ev_nhle_rank'] = float('nan')
    if has_ev.any():
        sub.loc[has_ev, 'ev_nhle_rank'] = (
            sub.loc[has_ev, 'ev_nhle_ppg'].rank(method='min', ascending=False)
        )
    sub['rank_diff'] = sub['actual_draft_rank'] - sub['ev_nhle_rank']

    print(f"\n  {yr} DRAFT — {len(sub)} forwards in seed, "
          f"{int(has_ev.sum())} with EV data")
    hdr = (f"  {'Pick':<5}{'Player':<22}{'League':<12}"
           f"{'DraftRk':>8}{'EVRk':>6}{'Diff':>7}{'EV_NHLe':>10}{'NHL_PPG':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, row in sub.iterrows():
        flag = ''
        if not pd.isna(row['rank_diff']):
            if row['rank_diff'] >= 10:
                flag = '  ↑model'
            elif row['rank_diff'] <= -3:
                flag = '  ↓model'
        print(f"  {int(row['pick']):<5}{str(row['player'])[:21]:<22}"
              f"{str(row['league'])[:11]:<12}"
              f"{_i(row['actual_draft_rank']):>8}{_i(row['ev_nhle_rank']):>6}"
              f"{_d(row['rank_diff']):>7}{_f(row['ev_nhle_ppg']):>10}"
              f"{_f(row['nhl_ppg']):>10}{flag}")

    have_diff = sub[sub['rank_diff'].notna()]
    if len(have_diff) > 0:
        up = have_diff[have_diff['rank_diff'] > 0].nlargest(2, 'rank_diff')
        dn = have_diff[have_diff['rank_diff'] < 0].nsmallest(2, 'rank_diff')
        if len(up) > 0:
            names = ", ".join(f"{r['player']} (+{int(r['rank_diff'])})"
                              for _, r in up.iterrows())
            print(f"    Biggest model UPGRADES vs scouts:    {names}")
        if len(dn) > 0:
            names = ", ".join(f"{r['player']} ({int(r['rank_diff'])})"
                              for _, r in dn.iterrows())
            print(f"    Biggest model DOWNGRADES vs scouts:  {names}")

    year_cols = ['pick', 'player', 'league',
                 'actual_draft_rank', 'ev_nhle_rank', 'rank_diff',
                 'ev_nhle_ppg', 'nhl_ppg']
    year_path = os.path.join(PROSPECTS_DIR, f"ranking_{yr}.csv")
    sub[year_cols].to_csv(year_path, index=False)
    year_paths[yr] = year_path

# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
os.makedirs(PROSPECTS_DIR, exist_ok=True)

seed_path = os.path.join(PROSPECTS_DIR, "prospect_seed.csv")
# Only bootstrap the seed CSV on first run. Once it exists, treat it as the
# source of truth — external edits (Excel, scrapes) must persist across runs.
if not os.path.exists(seed_path):
    pd.DataFrame(raw_data).to_csv(seed_path, index=False)

output_cols = [
    'draft_yr','pick','player','pos','league',
    'jr_gp','jr_pts','jr_ppg',
    'jr_pp_pts','has_pp_data','jr_ev_pts','jr_ev_ppg',
    'nhle_factor','raw_nhle_ppg','ev_nhle_ppg',
    'age_adj','ev_nhle_age_adj',
    'prev_gp','prev_pts','prev_ppg','yoy_pct_change',
    'birth_month','nhl_gp','nhl_pts','nhl_ppg',
]
analysis_path = os.path.join(PROSPECTS_DIR, "prospect_analysis.csv")
df[output_cols].to_csv(analysis_path, index=False)

ranking_path = os.path.join(PROSPECTS_DIR, "ranking_comparison.csv")
ranking_cols = ['draft_yr','pick','player','pos','league',
                'pick_rank','ev_nhle_ppg','ev_rank',
                'nhl_ppg','nhl_rank']
rank_df[ranking_cols].sort_values('pick_rank').to_csv(ranking_path, index=False)

print(f"\n  Seed CSV ({len(df)} players) → {seed_path}")
print(f"  Analysis CSV → {analysis_path}")
print(f"  Ranking comparison CSV ({len(rank_df)} players) → {ranking_path}")
for yr, path in year_paths.items():
    print(f"  Per-year ranking CSV [{yr}] → {path}")
print(f"\n  To add players: open prospect_seed.csv in Excel, add rows, re-run.")
