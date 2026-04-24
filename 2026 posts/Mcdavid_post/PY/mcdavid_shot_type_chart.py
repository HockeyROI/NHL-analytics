import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Charts/mcdavid_post"
os.makedirs(output_dir, exist_ok=True)

shot_types = ['Snap', 'Backhand', 'Wrist', 'Tip-in', 'Deflected']
attempts =   [11,     16,         60,      5,         3]
rates =      [0.182,  0.125,      0.067,   0.000,     0.000]
league_avg = 0.1357

colors = []
for r, a in zip(rates, attempts):
    if a < 10:
        colors.append('#2E7DC4')
    elif r >= league_avg:
        colors.append('#00CC44')
    else:
        colors.append('#FF2222')

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0B1D2E')
ax.set_facecolor('#0B1D2E')

y = np.arange(len(shot_types))
bars = ax.barh(y, rates, color=colors, height=0.55, zorder=3)

ax.axvline(league_avg, color='#4AB3E8', linestyle='--',
           linewidth=1.5, zorder=4, label=f'League avg {league_avg:.1%}')

for bar, rate, att in zip(bars, rates, attempts):
    sample_note = " (small sample)" if att < 10 else ""
    label = f"{rate:.1%}  |  {att} att{sample_note}"
    ax.text(rate + 0.004, bar.get_y() + bar.get_height()/2,
            label, va='center', ha='left', color='#F0F4F8',
            fontsize=12, fontfamily='Arial')

ax.set_yticks(y)
ax.set_yticklabels(shot_types, color='#F0F4F8', fontsize=13, fontweight='bold')
ax.set_xlabel('Net-front conversion rate', color='#4AB3E8', fontsize=12, labelpad=10)
ax.tick_params(colors='#F0F4F8')
ax.set_xlim(0, 0.32)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
ax.set_xticklabels(['0%','5%','10%','15%','20%','25%','30%'],
                   color='#F0F4F8', fontsize=11)
ax.grid(axis='x', color='#2E7DC4', alpha=0.15, zorder=0)

fig.text(0.5, 0.97, 'McDavid Net-Front Conversion by Shot Type',
         ha='center', va='top', color='#F0F4F8',
         fontsize=17, fontweight='bold', fontfamily='Arial')
fig.text(0.5, 0.91, 'Even strength 5v5  |  6 seasons  |  League avg = 13.57%  |  Blue = small sample (<10 att)',
         ha='center', va='top', color='#4AB3E8', fontsize=10, fontfamily='Arial')

plt.subplots_adjust(top=0.85)

ax.legend(facecolor='#1B3A5C', labelcolor='#F0F4F8',
          framealpha=0.8, fontsize=11, loc='lower right')

fig.text(0.98, 0.02, 'HockeyROI', ha='right', va='bottom',
         color='#4AB3E8', fontsize=9, fontfamily='Arial')

plt.savefig(f'{output_dir}/mcdavid_shot_type_chart.png',
            dpi=150, bbox_inches='tight', facecolor='#0B1D2E')
plt.close()
print("Saved mcdavid_shot_type_chart.png")
