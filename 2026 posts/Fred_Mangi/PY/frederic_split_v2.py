import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Charts/frederic_teaser"
os.makedirs(output_dir, exist_ok=True)

labels = ['Games 1-60', 'Games 61-81']
values = [0.0, 0.20]
colors = ['#CC3333', '#FF6B35']
annotations = ['32 net-front attempts\n0 goals', '10 net-front attempts\n2 goals']

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0B1D2E')
ax.set_facecolor('#0B1D2E')

x = np.arange(len(labels))
bars = ax.bar(x, values, color=colors, width=0.5, zorder=3)

for i, (bar, val, ann) in enumerate(zip(bars, values, annotations)):
    if val == 0:
        ax.text(bar.get_x() + bar.get_width()/2,
                0.015,
                '0%',
                ha='center', va='bottom',
                color='#F0F4F8', fontsize=28,
                fontweight='bold', fontfamily='Arial')
        ax.text(bar.get_x() + bar.get_width()/2,
                0.075,
                ann,
                ha='center', va='bottom',
                color='#F0F4F8', fontsize=12,
                fontweight='bold', fontfamily='Arial',
                linespacing=1.5)
    else:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.008,
                f'{val:.0%}',
                ha='center', va='bottom',
                color='#F0F4F8', fontsize=28,
                fontweight='bold', fontfamily='Arial')
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height()/2,
                ann,
                ha='center', va='center',
                color='#F0F4F8', fontsize=12,
                fontweight='bold', fontfamily='Arial',
                linespacing=1.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, color='#F0F4F8', fontsize=14, fontweight='bold')
ax.set_ylabel('NF goal rate', color='#F0F4F8', fontsize=12, labelpad=10)
ax.set_ylim(0, 0.32)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax.tick_params(colors='#F0F4F8')

for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(axis='y', color='#2E7DC4', alpha=0.15, zorder=0)

fig.text(0.5, 0.97, 'Frederic 2025-26: Before and After',
         ha='center', va='top', color='#F0F4F8',
         fontsize=17, fontweight='bold', fontfamily='Arial')
fig.text(0.5, 0.91,
         'Net-front goal rate  |  Even strength 5v5',
         ha='center', va='top', color='#4AB3E8',
         fontsize=10, fontfamily='Arial')

plt.subplots_adjust(top=0.85)

fig.text(0.98, 0.02, 'HockeyROI', ha='right', va='bottom',
         color='#4AB3E8', fontsize=9, fontfamily='Arial')

plt.savefig(f'{output_dir}/frederic_split.png',
            dpi=150, bbox_inches='tight', facecolor='#0B1D2E')
plt.close()
print("Saved frederic_split.png")
