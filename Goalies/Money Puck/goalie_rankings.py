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

# -- Derive metrics ----------------------------------------------------------

# Overall SV%
df_q['total_shots'] = df_q['lowDangerShots'] + df_q['mediumDangerShots'] + df_q['highDangerShots']
df_q['total_goals'] = df_q['lowDangerGoals'] + df_q['mediumDangerGoals'] + df_q['highDangerGoals']
df_q['sv_pct'] = 1 - (df_q['total_goals'] / df_q['total_shots'])

# HD SV% (all shots including blocked)
df_q['hd_sv_pct'] = 1 - (df_q['highDangerGoals'] / df_q['highDangerShots'])

# MD SV%
df_q['md_sv_pct'] = 1 - (df_q['mediumDangerGoals'] / df_q['mediumDangerShots'])

# LD SV%
df_q['ld_sv_pct'] = 1 - (df_q['lowDangerGoals'] / df_q['lowDangerShots'])

# GSAx
df_q['gsax'] = df_q['xGoals'] - df_q['total_goals']

# -- Rankings (1 = best) -----------------------------------------------------
df_q['rank_sv_pct']    = df_q['sv_pct'].rank(ascending=False).astype(int)
df_q['rank_hd_sv_pct'] = df_q['hd_sv_pct'].rank(ascending=False).astype(int)
df_q['rank_md_sv_pct'] = df_q['md_sv_pct'].rank(ascending=False).astype(int)
df_q['rank_ld_sv_pct'] = df_q['ld_sv_pct'].rank(ascending=False).astype(int)
df_q['rank_gsax']      = df_q['gsax'].rank(ascending=False).astype(int)

total = len(df_q)

# -- League averages ---------------------------------------------------------
print(f"\n{'='*60}")
print(f"LEAGUE AVERAGES -- all {total} qualifying goalies (20+ GP)")
print(f"{'='*60}")
print(f"Overall SV%:  {df_q['sv_pct'].mean():.4f}")
print(f"HD SV%:       {df_q['hd_sv_pct'].mean():.4f}")
print(f"MD SV%:       {df_q['md_sv_pct'].mean():.4f}")
print(f"LD SV%:       {df_q['ld_sv_pct'].mean():.4f}")
print(f"Avg GSAx:     {df_q['gsax'].mean():.2f}")

# -- Top-32 starter averages -------------------------------------------------
df_top32 = df_q.sort_values('games_played', ascending=False).head(32)
print(f"\n{'='*60}")
print(f"STARTER AVERAGES -- top 32 by GP")
print(f"{'='*60}")
print(f"Overall SV%:  {df_top32['sv_pct'].mean():.4f}")
print(f"HD SV%:       {df_top32['hd_sv_pct'].mean():.4f}")
print(f"MD SV%:       {df_top32['md_sv_pct'].mean():.4f}")
print(f"LD SV%:       {df_top32['ld_sv_pct'].mean():.4f}")
print(f"Avg GSAx:     {df_top32['gsax'].mean():.2f}")

# -- Ingram ------------------------------------------------------------------
ingram = df_q[df_q['name'].str.contains('Ingram', case=False, na=False)]

print(f"\n{'='*60}")
print(f"CONNOR INGRAM -- ranked among all {total} qualifying goalies")
print(f"{'='*60}")

if len(ingram) == 0:
    print("Ingram not found -- check name spelling in dataset")
    print("Available names sample:", df_q['name'].head(10).tolist())
else:
    for _, row in ingram.iterrows():
        print(f"Team: {row.get('team', 'N/A')}  |  GP: {row['games_played']}")
        print(f"\nMetric          Value    Rank/{total}")
        print(f"-" * 35)
        print(f"Overall SV%     {row['sv_pct']:.4f}   {row['rank_sv_pct']}/{total}")
        print(f"HD SV%          {row['hd_sv_pct']:.4f}   {row['rank_hd_sv_pct']}/{total}")
        print(f"MD SV%          {row['md_sv_pct']:.4f}   {row['rank_md_sv_pct']}/{total}")
        print(f"LD SV%          {row['ld_sv_pct']:.4f}   {row['rank_ld_sv_pct']}/{total}")
        print(f"GSAx            {row['gsax']:+.2f}    {row['rank_gsax']}/{total}")

# -- Full sorted table -------------------------------------------------------
print(f"\n{'='*60}")
print("FULL RANKINGS TABLE (sorted by GSAx)")
print(f"{'='*60}")
cols = ['name', 'team', 'games_played', 'sv_pct', 'hd_sv_pct', 'md_sv_pct', 'gsax',
        'rank_sv_pct', 'rank_hd_sv_pct', 'rank_gsax']
available = [c for c in cols if c in df_q.columns]
print(df_q.sort_values('gsax', ascending=False)[available].to_string(index=False))

# -- Save to CSV -------------------------------------------------------------
out_path = '/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data/goalie_rankings_2025_26.csv'
df_q.sort_values('gsax', ascending=False).to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
