import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Charts/mcdavid_post"
os.makedirs(output_dir, exist_ok=True)

seasons = ['20-21', '21-22', '22-23', '23-24', '24-25', '25-26']
mcdavid = [0.400,   0.050,   0.000,   0.100,   0.000,   0.000]
league =  [0.187,   0.170,   0.141,   0.112,   0.117,   0.100]
attempts = [5,       20,      8,       10,      6,       11]

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0B1D2E')
ax.set_facecolor('#0B1D2E')

x = np.arange(len(seasons))

ax.plot(x, league, color='#4AB3E8', linewidth=2,
        linestyle='--', marker='o', markersize=6,
        label='League average', zorder=4)
ax.plot(x, mcdavid, color='#FF6B35', linewidth=2.5,
        marker='o', markersize=8,
        label='McDavid', zorder=5)

ax.fill_between(x, mcdavid, league,
                where=[m < l for m, l in zip(mcdavid, league)],
                alpha=0.15, color='#CC3333', zorder=2,
                label='Below league average')
ax.fill_between(x, mcdavid, league,
                where=[m >= l for m, l in zip(mcdavid, league)],
                alpha=0.15, color='#44AA66', zorder=2,
                label='Above league average')

for xi, m, a in zip(x, mcdavid, attempts):
    ax.annotate(f'{m:.1%}\n({a} att)',
                xy=(xi, m), xytext=(0, 16),
                textcoords='offset points',
                ha='center', color='#FF6B35',
                fontsize=9, fontfamily='Arial')

ax.set_xticks(x)
ax.set_xticklabels(seasons, color='#F0F4F8', fontsize=12)
ax.set_ylabel('Wrist rebound conversion rate', color='#4AB3E8',
              fontsize=12, labelpad=10)
ax.set_ylim(-0.02, 0.52)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax.tick_params(colors='#F0F4F8')

for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(axis='y', color='#2E7DC4', alpha=0.15, zorder=0)

fig.text(0.5, 0.97, 'McDavid Wrist Rebound Conversion vs League Average',
         ha='center', va='top', color='#F0F4F8',
         fontsize=17, fontweight='bold', fontfamily='Arial')
fig.text(0.5, 0.91, 'Season by season  |  Even strength 5v5  |  Wrist shot rebounds only',
         ha='center', va='top', color='#4AB3E8', fontsize=10, fontfamily='Arial')

plt.subplots_adjust(top=0.85)

ax.legend(facecolor='#1B3A5C', labelcolor='#F0F4F8',
          framealpha=0.8, fontsize=11, loc='upper right')

fig.text(0.98, 0.02, 'HockeyROI', ha='right', va='bottom',
         color='#4AB3E8', fontsize=9, fontfamily='Arial')

plt.savefig(f'{output_dir}/mcdavid_wrist_trend_chart.png',
            dpi=150, bbox_inches='tight', facecolor='#0B1D2E')
plt.close()
print("Saved mcdavid_wrist_trend_chart.png")
