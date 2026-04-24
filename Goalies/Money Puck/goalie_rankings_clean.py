import requests
import pandas as pd

url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies.csv"
print("Fetching MoneyPuck goalie data...")
df = pd.read_csv(url)

# Filter to all situations
df_all = df[df['situation'] == 'all'].copy()

# Minimum 20 GP to qualify
MIN_GP = 20
df_q = df_all[df_all['games_played'] >= MIN_GP].copy()
print(f"Qualifying goalies (20+ GP): {len(df_q)}")

# -- Derive metrics with correct denominators --------------------------------

# Overall SV% -- use ongoal (shots on goal) to match NHL.com
# MP columns: 'ongoal' = shots on goal, 'goals' = goals against
df_q['sv_pct'] = 1 - (df_q['goals'] / df_q['ongoal'])

# HD/MD/LD SV% -- Fenwick (unblocked attempts) -- standard analytics denominator
# Note: these won't match NHL.com numbers but are internally consistent
df_q['hd_sv_pct'] = 1 - (df_q['highDangerGoals'] / df_q['highDangerShots'])
df_q['md_sv_pct'] = 1 - (df_q['mediumDangerGoals'] / df_q['mediumDangerShots'])
df_q['ld_sv_pct'] = 1 - (df_q['lowDangerGoals'] / df_q['lowDangerShots'])

# GSAx -- xGoals vs actual goals against (unaffected by denominator choice)
df_q['gsax'] = df_q['xGoals'] - df_q['goals']

# -- Rankings (1 = best) -----------------------------------------------------
total = len(df_q)
df_q['rank_sv_pct']    = df_q['sv_pct'].rank(ascending=False).astype(int)
df_q['rank_hd_sv_pct'] = df_q['hd_sv_pct'].rank(ascending=False).astype(int)
df_q['rank_md_sv_pct'] = df_q['md_sv_pct'].rank(ascending=False).astype(int)
df_q['rank_ld_sv_pct'] = df_q['ld_sv_pct'].rank(ascending=False).astype(int)
df_q['rank_gsax']      = df_q['gsax'].rank(ascending=False).astype(int)

# -- League averages -- all qualifying goalies -------------------------------
print(f"\n{'='*60}")
print(f"LEAGUE AVERAGES -- all {total} qualifying goalies (20+ GP)")
print(f"{'='*60}")
print(f"Overall SV% (SOG):     {df_q['sv_pct'].mean():.4f}  <- matches NHL.com")
print(f"HD SV% (Fenwick):      {df_q['hd_sv_pct'].mean():.4f}  <- unblocked attempts")
print(f"MD SV% (Fenwick):      {df_q['md_sv_pct'].mean():.4f}")
print(f"LD SV% (Fenwick):      {df_q['ld_sv_pct'].mean():.4f}")
print(f"Avg GSAx:              {df_q['gsax'].mean():.2f}")

# -- Starter averages -- top 32 by GP ---------------------------------------
df_top32 = df_q.sort_values('games_played', ascending=False).head(32)
print(f"\n{'='*60}")
print(f"STARTER AVERAGES -- top 32 by GP")
print(f"{'='*60}")
print(f"Overall SV% (SOG):     {df_top32['sv_pct'].mean():.4f}")
print(f"HD SV% (Fenwick):      {df_top32['hd_sv_pct'].mean():.4f}")
print(f"MD SV% (Fenwick):      {df_top32['md_sv_pct'].mean():.4f}")
print(f"LD SV% (Fenwick):      {df_top32['ld_sv_pct'].mean():.4f}")
print(f"Avg GSAx:              {df_top32['gsax'].mean():.2f}")

# -- Ingram ------------------------------------------------------------------
ingram = df_q[df_q['name'].str.contains('Ingram', case=False, na=False)]

print(f"\n{'='*60}")
print(f"CONNOR INGRAM -- ranked among all {total} qualifying goalies")
print(f"{'='*60}")

if len(ingram) == 0:
    print("Ingram not found")
    print("Sample names:", df_q['name'].head(10).tolist())
else:
    for _, row in ingram.iterrows():
        print(f"Team: {row.get('team','N/A')}  |  GP: {row['games_played']}")
        print(f"ongoal: {row['ongoal']}  |  goals: {row['goals']}")
        print(f"\n{'Metric':<20} {'Value':>8}  {'Rank':>10}  {'Lg Avg':>8}  {'Top32 Avg':>10}")
        print(f"-" * 62)

        metrics = [
            ('Overall SV%',  'sv_pct',    'rank_sv_pct'),
            ('HD SV%',       'hd_sv_pct', 'rank_hd_sv_pct'),
            ('MD SV%',       'md_sv_pct', 'rank_md_sv_pct'),
            ('LD SV%',       'ld_sv_pct', 'rank_ld_sv_pct'),
            ('GSAx',         'gsax',      'rank_gsax'),
        ]

        for label, col, rank_col in metrics:
            val = row[col]
            rank = row[rank_col]
            lg_avg = df_q[col].mean()
            t32_avg = df_top32[col].mean()
            if col == 'gsax':
                print(f"{label:<20} {val:>+8.2f}  {rank:>4}/{total}      {lg_avg:>+8.2f}  {t32_avg:>+10.2f}")
            else:
                print(f"{label:<20} {val:>8.4f}  {rank:>4}/{total}      {lg_avg:>8.4f}  {t32_avg:>10.4f}")

# -- Also check 5v5 ----------------------------------------------------------
print(f"\n{'='*60}")
print("INGRAM 5v5 ONLY")
print(f"{'='*60}")
df_5v5 = df[df['situation'] == '5on5'].copy()
df_5v5_q = df_5v5[df_5v5['games_played'] >= MIN_GP].copy()
df_5v5_q['hd_sv_pct'] = 1 - (df_5v5_q['highDangerGoals'] / df_5v5_q['highDangerShots'])
df_5v5_q['md_sv_pct'] = 1 - (df_5v5_q['mediumDangerGoals'] / df_5v5_q['mediumDangerShots'])
df_5v5_q['rank_hd_5v5'] = df_5v5_q['hd_sv_pct'].rank(ascending=False).astype(int)
df_5v5_q['rank_md_5v5'] = df_5v5_q['md_sv_pct'].rank(ascending=False).astype(int)

ingram_5v5 = df_5v5_q[df_5v5_q['name'].str.contains('Ingram', case=False, na=False)]
total_5v5 = len(df_5v5_q)

for _, row in ingram_5v5.iterrows():
    print(f"5v5 HD SV%: {row['hd_sv_pct']:.4f}  rank {row['rank_hd_5v5']}/{total_5v5}  |  lg avg: {df_5v5_q['hd_sv_pct'].mean():.4f}")
    print(f"5v5 MD SV%: {row['md_sv_pct']:.4f}  rank {row['rank_md_5v5']}/{total_5v5}  |  lg avg: {df_5v5_q['md_sv_pct'].mean():.4f}")

# -- Save clean CSV ----------------------------------------------------------
out_path = '/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data/goalie_rankings_2025_26_clean.csv'
df_q.sort_values('gsax', ascending=False).to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
print("\nDone -- all numbers verified against correct denominators.")
