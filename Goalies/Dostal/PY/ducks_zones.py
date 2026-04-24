import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests

df = pd.read_csv('/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Data/nhl_shot_events.csv')
df = df[df['x_coord_norm'] >= 25].copy()
df = df[(df['is_goal'] == 1) & (df['season'] == 20252026)].copy()

dmen = {'Jacob Trouba', 'John Carlson', 'Jackson LaCombe', 'Olen Zellweger',
        'Pavel Mintyukov', 'Drew Helleson', 'Radko Gudas'}

roster = requests.get('https://api-web.nhle.com/v1/roster/ANA/current').json()
players = []
for group in ['forwards', 'defensemen', 'goalies']:
    for p in roster.get(group, []):
        players.append({'player_id': p['id'], 'name': f"{p['firstName']['default']} {p['lastName']['default']}"})
roster_df = pd.DataFrame(players)

df = df[df['shooter_player_id'].isin(roster_df['player_id'])].copy()
df = df.merge(roster_df, left_on='shooter_player_id', right_on='player_id', how='left')

counts = df['name'].value_counts()
forwards = [n for n, c in counts.items() if n not in dmen and c >= 10]
df = df[df['name'].isin(forwards)].copy()

def get_zone(x, y):
    if x >= 82 and abs(y) <= 10:
        return 'Crease'
    elif x >= 69 and abs(y) <= 22:
        return 'Slot'
    elif 55 <= x < 82 and y > 10:
        return 'Left Circle'
    elif 55 <= x < 82 and y < -10:
        return 'Right Circle'
    elif 55 <= x <= 82 and abs(y) <= 10:
        return 'High Slot'
    else:
        return 'Point/Blue Line'

df['zone'] = df.apply(lambda r: get_zone(r['x_coord_norm'], r['y_coord_norm']), axis=1)

pivot = df.pivot_table(index='name', columns='zone', values='x_coord_norm', aggfunc='count', fill_value=0)
pivot['Total'] = pivot.sum(axis=1)
pivot = pivot.sort_values('Total', ascending=False)
zone_cols = [c for c in pivot.columns if c != 'Total' and pivot[c].max() >= 3]
pivot = pivot[zone_cols]

ZONE_COLORS = {
    'Slot':            '#2E7DC4',
    'Crease':          '#FF6B35',
    'High Slot':       '#4AB3E8',
    'Right Circle':    '#44AA66',
    'Left Circle':     '#FFB700',
    'Point/Blue Line': '#888888',
}

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#1B3A5C')
ax.set_facecolor('#0B1D2E')

x = np.arange(len(pivot.index))
bar_width = 0.13
offsets = np.linspace(-(len(zone_cols)-1)/2, (len(zone_cols)-1)/2, len(zone_cols)) * bar_width

for i, zone in enumerate(zone_cols):
    vals = pivot[zone].values
    bars = ax.bar(x + offsets[i], vals, width=bar_width,
                  color=ZONE_COLORS.get(zone, '#888888'), label=zone, zorder=3)
    for bar, val in zip(bars, vals):
        if val >= 3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    str(int(val)), ha='center', va='bottom',
                    fontsize=7.5, color='#F0F4F8', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([p.split()[-1] for p in pivot.index], color='#F0F4F8', fontsize=11)
ax.set_ylabel('Goals', color='#F0F4F8', fontsize=12)
ax.set_title('ANAHEIM DUCKS — FORWARD GOALS BY ZONE  |  2025·26',
             color='#F0F4F8', fontsize=14, fontweight='bold', pad=16, loc='left')
ax.tick_params(colors='#F0F4F8')
ax.spines['bottom'].set_color('#2E7DC4')
ax.spines['left'].set_color('#2E7DC4')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_tick_params(labelcolor='#F0F4F8')
ax.set_ylim(0, pivot.max().max() + 3)
ax.grid(axis='y', color='#2E7DC4', alpha=0.2, zorder=0)
ax.legend(loc='upper right', framealpha=0.2, labelcolor='#F0F4F8',
          facecolor='#1B3A5C', edgecolor='#2E7DC4', fontsize=10)
ax.text(0.0, -0.08, 'Cutoff: zones with 3+ goals only  |  Forwards with 10+ goals  |  HockeyROI.substack.com',
        transform=ax.transAxes, color='#4AB3E8', fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('/Users/ashgarg/Desktop/ducks_forward_zones.png', dpi=150,
            bbox_inches='tight', facecolor='#1B3A5C')
print("Saved to Desktop: ducks_forward_zones.png")
