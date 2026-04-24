import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Charts/frederic_mangiapane_post"
os.makedirs(output_dir, exist_ok=True)

seasons = ['20-21', '21-22', '22-23', '23-24\n(WSH)', '24-25\n(EDM)']
composite = [0.721,   0.771,   0.716,   0.527,         0.497]
goal_rate =  [0.429,   0.333,   0.133,   0.118,         0.091]
teams =      ['CGY',   'CGY',   'CGY',   'WSH',         'EDM']

fig, ax1 = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0B1D2E')
ax1.set_facecolor('#0B1D2E')

x = np.arange(len(seasons))

bars = ax1.bar(x, composite, color='#2E7DC4', alpha=0.4,
               width=0.5, zorder=2, label='NF composite score')

ax2 = ax1.twinx()
ax2.set_facecolor('#0B1D2E')
ax2.plot(x, goal_rate, color='#FF6B35', linewidth=2.5,
         marker='o', markersize=10, zorder=5, label='NF goal rate')

for xi, gr, comp in zip(x, goal_rate, composite):
    ax2.annotate(f'{gr:.1%}',
                 xy=(xi, gr), xytext=(0, 14),
                 textcoords='offset points',
                 ha='center', color='#FF6B35',
                 fontsize=11, fontweight='bold', fontfamily='Arial')

for xi, team in zip(x, teams):
    ax1.text(xi, 0.02, team, ha='center', va='bottom',
             color='#4AB3E8', fontsize=10, fontfamily='Arial',
             transform=ax1.get_xaxis_transform())

ax1.set_xticks(x)
ax1.set_xticklabels(seasons, color='#F0F4F8', fontsize=12)
ax1.set_ylabel('NF composite score', color='#2E7DC4', fontsize=12, labelpad=10)
ax1.set_ylim(0, 1.0)
ax1.tick_params(axis='y', colors='#2E7DC4')
ax1.tick_params(axis='x', colors='#F0F4F8')

ax2.set_ylabel('NF goal rate', color='#FF6B35', fontsize=12, labelpad=10)
ax2.set_ylim(0, 0.55)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax2.tick_params(axis='y', colors='#FF6B35')

for spine in ax1.spines.values():
    spine.set_visible(False)
for spine in ax2.spines.values():
    spine.set_visible(False)

ax1.grid(axis='y', color='#2E7DC4', alpha=0.15, zorder=0)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           facecolor='#1B3A5C', labelcolor='#F0F4F8',
           framealpha=0.8, fontsize=11, loc='upper right')

fig.text(0.5, 0.97, 'Andrew Mangiapane — Net-Front Trajectory',
         ha='center', va='top', color='#F0F4F8',
         fontsize=17, fontweight='bold', fontfamily='Arial')
fig.text(0.5, 0.91, 'Composite score and goal rate by season  |  Even strength 5v5',
         ha='center', va='top', color='#4AB3E8', fontsize=10, fontfamily='Arial')

plt.subplots_adjust(top=0.85)

fig.text(0.98, 0.02, 'HockeyROI', ha='right', va='bottom',
         color='#4AB3E8', fontsize=9, fontfamily='Arial')

plt.savefig(f'{output_dir}/mangiapane_trajectory.png',
            dpi=150, bbox_inches='tight', facecolor='#0B1D2E')
plt.close()
print("Saved mangiapane_trajectory.png")
