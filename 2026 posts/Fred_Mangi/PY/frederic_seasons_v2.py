import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Charts/frederic_teaser"
os.makedirs(output_dir, exist_ok=True)

seasons = ['20-21', '21-22', '22-23', '23-24', '24-25', '25-26']
attempt_rates = [0.111, 0.186, 0.249, 0.248, 0.225, 0.316]
timing_scores = [0.50,  0.78,  0.23,  0.72,  0.55,  0.42]
career_avg_attempt = 0.233
career_avg_timing = 0.53

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0B1D2E')
ax.set_facecolor('#0B1D2E')

x = np.arange(len(seasons))
width = 0.35

bars1 = ax.bar(x - width/2, attempt_rates, width,
               color='#2E7DC4', label='NF attempt rate', zorder=3)
bars2 = ax.bar(x + width/2, timing_scores, width,
               color='#FF6B35', label='Timing score (0-1)', zorder=3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.008,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom',
            color='#2E7DC4', fontsize=9,
            fontweight='bold', fontfamily='Arial')

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.008,
            f'{bar.get_height():.2f}',
            ha='center', va='bottom',
            color='#FF6B35', fontsize=9,
            fontweight='bold', fontfamily='Arial')

ax.axhline(career_avg_attempt, color='#2E7DC4',
           linestyle='--', alpha=0.6, linewidth=1.2,
           label=f'Career avg attempt ({career_avg_attempt:.3f})')
ax.axhline(career_avg_timing, color='#FF6B35',
           linestyle='--', alpha=0.6, linewidth=1.2,
           label=f'Career avg timing ({career_avg_timing:.2f})')

ax.annotate('Career high\nattempt rate',
            xy=(5 - width/2, 0.316),
            xytext=(3.8, 0.38),
            color='#F0F4F8', fontsize=9, fontfamily='Arial',
            arrowprops=dict(arrowstyle='->', color='#F0F4F8', lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels(seasons, color='#F0F4F8', fontsize=12)
ax.tick_params(colors='#F0F4F8')
ax.set_ylim(0, 0.50)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(axis='y', color='#2E7DC4', alpha=0.15, zorder=0)

ax.set_ylabel('Rate / Score', color='#F0F4F8', fontsize=12, labelpad=10)

ax.legend(facecolor='#1B3A5C', labelcolor='#F0F4F8',
          framealpha=0.8, fontsize=10, loc='upper left')

fig.text(0.5, 0.97, 'Trent Frederic — Net-Front Profile by Season',
         ha='center', va='top', color='#F0F4F8',
         fontsize=17, fontweight='bold', fontfamily='Arial')
fig.text(0.5, 0.91,
         'Attempt rate and timing score  |  Even strength 5v5',
         ha='center', va='top', color='#4AB3E8',
         fontsize=10, fontfamily='Arial')

plt.subplots_adjust(top=0.85)

fig.text(0.98, 0.02, 'HockeyROI', ha='right', va='bottom',
         color='#4AB3E8', fontsize=9, fontfamily='Arial')

plt.savefig(f'{output_dir}/frederic_seasons.png',
            dpi=150, bbox_inches='tight', facecolor='#0B1D2E')
plt.close()
print("Saved frederic_seasons.png")
