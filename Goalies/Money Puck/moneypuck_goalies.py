import requests
import pandas as pd

# MoneyPuck has a public API endpoint for goalie stats
url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2026/regular/goalies.csv"

print("Fetching MoneyPuck goalie data...")
df = pd.read_csv(url)

print(f"Total goalies in dataset: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Filter to relevant columns - MoneyPuck uses specific column names
# Common ones: name, team, situation, games_played, saves, shots, goals,
# highDangerSaves, highDangerShots, xGoals, goals (for GSAx = xGoals - goals)

# Filter to all situations or 5v5
# MoneyPuck splits by situation so filter to 'all' or '5on5'
situations = df['situation'].unique() if 'situation' in df.columns else ['unknown']
print(f"Situations available: {situations}")

# Use all situations for overall SV%
df_all = df[df['situation'] == 'all'] if 'situation' in df.columns else df

# Sort by games played descending, take top 32
if 'games_played' in df_all.columns:
    gp_col = 'games_played'
elif 'gamesPlayed' in df_all.columns:
    gp_col = 'gamesPlayed'
else:
    gp_col = [c for c in df_all.columns if 'game' in c.lower()][0]

print(f"\nUsing games column: {gp_col}")

df_starters = df_all.sort_values(gp_col, ascending=False).head(32)

print(f"\nTop 32 goalies by games played:")

# Find save % columns
save_cols = [c for c in df_all.columns if 'save' in c.lower() or 'sv' in c.lower()]
xgoal_cols = [c for c in df_all.columns if 'xgoal' in c.lower() or 'xg' in c.lower()]
hd_cols = [c for c in df_all.columns if 'danger' in c.lower() or 'hd' in c.lower()]
goal_cols = [c for c in df_all.columns if 'goal' in c.lower()]

print(f"Save columns: {save_cols}")
print(f"xGoal columns: {xgoal_cols}")
print(f"HD columns: {hd_cols}")
print(f"Goal columns: {goal_cols}")

# Print top 32 with key stats
name_col = 'name' if 'name' in df_starters.columns else 'playerId'

for _, row in df_starters.iterrows():
    print(f"  {row.get(name_col, 'Unknown'):<25} GP: {row.get(gp_col, 'N/A')}")

# Calculate averages for whatever columns exist
print("\n--- CALCULATING AVERAGES ---")
for col in save_cols + hd_cols:
    if col in df_starters.columns:
        avg = df_starters[col].mean()
        print(f"Avg {col}: {avg:.4f}")

# GSAx = goals saved above expected
# MoneyPuck: xGoals - goalsAgainst = GSAx
if 'xGoals' in df_starters.columns and 'goals' in df_starters.columns:
    df_starters['gsax'] = df_starters['xGoals'] - df_starters['goals']
    print(f"\nAvg GSAx (top 32): {df_starters['gsax'].mean():.2f}")
    print(f"Ingram GSAx: 1.7 (from MoneyPuck)")
    print(f"Ingram rank among top 32: {(df_starters['gsax'] > 1.7).sum() + 1}")
