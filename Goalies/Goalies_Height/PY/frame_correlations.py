"""
Three-way frame correlation analysis: weight, height, and weight/height ratio vs avg GSAx.
Usage: python3 frame_correlations.py [--csv goalie_frame_gsax_expanded.csv] [--suffix expanded]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"


def thirds_ttest(df, col):
    thirds = pd.qcut(df[col], q=3, labels=["Low", "Mid", "High"])
    df = df.copy()
    df["_grp"] = thirds
    low  = df[df["_grp"] == "Low"]["avg_gsax"]
    high = df[df["_grp"] == "High"]["avg_gsax"]
    t, p = stats.ttest_ind(low, high)
    return low, high, t, p


def make_plots(df, x_col, x_label, low_label, high_label, suffix, out_prefix):
    r, p_corr = stats.pearsonr(df[x_col], df["avg_gsax"])
    slope, intercept, _, _, _ = stats.linregress(df[x_col], df["avg_gsax"])
    low, high, t_stat, p_ttest = thirds_ttest(df, x_col)

    thirds = pd.qcut(df[x_col], q=3, labels=["Low", "Mid", "High"])
    df = df.copy()
    df["_grp"] = thirds
    low_df  = df[df["_grp"] == "Low"]
    high_df = df[df["_grp"] == "High"]

    print(f"\n{'='*55}")
    print(f"  {x_label} vs Avg GSAx")
    print(f"{'='*55}")
    print(f"  Pearson r = {r:.4f},  p = {p_corr:.4f}  "
          f"({'sig *' if p_corr < 0.05 else 'n.s.'}),  n = {len(df)}")
    print(f"\n  {low_label}:")
    print(f"    n = {len(low_df)},  range = {low_df[x_col].min():.2f} – {low_df[x_col].max():.2f}")
    print(f"    mean GSAx = {low.mean():.3f}")
    print(f"\n  {high_label}:")
    print(f"    n = {len(high_df)},  range = {high_df[x_col].min():.2f} – {high_df[x_col].max():.2f}")
    print(f"    mean GSAx = {high.mean():.3f}")
    print(f"\n  T-test: t = {t_stat:.4f},  p = {p_ttest:.4f}  "
          f"({'sig *' if p_ttest < 0.05 else 'n.s.'})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"NHL Goalie {x_label} vs Save Performance (2010–present)\n"
                 f"min 3 qualifying seasons, 25 starts",
                 fontsize=13, fontweight="bold", y=1.01)

    # --- Scatter ---
    x = df[x_col].values
    y = df["avg_gsax"].values
    x_line = np.linspace(x.min() - (x.max()-x.min())*0.05,
                         x.max() + (x.max()-x.min())*0.05, 200)
    y_line = slope * x_line + intercept

    ax1.scatter(x, y, color="steelblue", s=65, alpha=0.8, zorder=3,
                edgecolors="white", linewidths=0.5)
    ax1.plot(x_line, y_line, color="crimson", linewidth=2,
             label=f"r = {r:.3f},  p = {p_corr:.3f}"
                   f"{' *' if p_corr < 0.05 else ' (n.s.)'}", zorder=4)

    for xi, yi, name in zip(x, y, df["display_name"].values):
        ax1.annotate(name, (xi, yi), textcoords="offset points",
                     xytext=(5, 3), fontsize=6.5, alpha=0.85, color="#222")

    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel("Average GSAx per qualifying season", fontsize=11)
    ax1.set_title(f"{x_label} vs Avg GSAx — Scatter", fontsize=12)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.25)

    # --- Bar chart ---
    groups_data  = [low, high]
    group_labels = [
        f"{low_label}\n{low_df[x_col].min():.2f}–{low_df[x_col].max():.2f}",
        f"{high_label}\n{high_df[x_col].min():.2f}–{high_df[x_col].max():.2f}",
    ]
    means      = [g.mean() for g in groups_data]
    bar_colors = ["#4C9BE8", "#E8834C"]
    dot_colors = ["#1a5fa3", "#a34a1a"]
    x_pos = [0, 1]

    ax2.bar(x_pos, means, width=0.5, color=bar_colors, alpha=0.75,
            zorder=2, edgecolor="white", linewidth=1.2)

    rng = np.random.default_rng(42)
    for i, (gdata, dcol) in enumerate(zip(groups_data, dot_colors)):
        jitter = rng.uniform(-0.12, 0.12, size=len(gdata))
        ax2.scatter(np.full(len(gdata), x_pos[i]) + jitter, gdata.values,
                    color=dcol, s=60, alpha=0.85, zorder=4,
                    edgecolors="white", linewidths=0.5)

    for xi, m in zip(x_pos, means):
        ax2.hlines(m, xi - 0.25, xi + 0.25, colors="white", linewidths=2.5, zorder=5)

    for xi, gdata in zip(x_pos, groups_data):
        ax2.text(xi, ax2.get_ylim()[0] + 0.3 if ax2.get_ylim()[0] > 0 else 0.3,
                 f"n={len(gdata)}", ha="center", va="bottom", fontsize=10,
                 color="white", fontweight="bold", zorder=6)

    y_top = max(low.max(), high.max()) + 1.5
    ax2.plot([0, 0, 1, 1], [y_top - 0.5, y_top, y_top, y_top - 0.5],
             lw=1.2, color="#333")
    sig_str = f"p = {p_ttest:.3f}" + (" *" if p_ttest < 0.05 else " (n.s.)")
    ax2.text(0.5, y_top + 0.3, sig_str, ha="center", va="bottom",
             fontsize=11, color="#333")

    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(group_labels, fontsize=10)
    ax2.set_ylabel("Average GSAx per qualifying season", fontsize=11)
    ax2.set_title(f"{low_label} vs {high_label}: Avg GSAx", fontsize=12)
    ax2.grid(axis="y", alpha=0.25, zorder=1)
    ax2.set_xlim(-0.5, 1.5)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{out_prefix}_{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    return {"metric": x_label, "r": r, "p_corr": p_corr,
            "mean_low": low.mean(), "mean_high": high.mean(),
            "n_low": len(low), "n_high": len(high),
            "t": t_stat, "p_ttest": p_ttest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default="goalie_frame_gsax_expanded.csv")
    parser.add_argument("--suffix", default="expanded")
    args = parser.parse_args()

    csv_path = os.path.join(OUTPUT_DIR, args.csv)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} goalies from {csv_path}")

    results = []

    results.append(make_plots(
        df, x_col="weight_lbs",
        x_label="Weight (lbs)",
        low_label="Light", high_label="Heavy",
        suffix=args.suffix, out_prefix="weight_vs_gsax"
    ))

    results.append(make_plots(
        df, x_col="height_in",
        x_label="Height (inches)",
        low_label="Short", high_label="Tall",
        suffix=args.suffix, out_prefix="height_vs_gsax"
    ))

    results.append(make_plots(
        df, x_col="weight_height_ratio",
        x_label="Weight/Height Ratio (lbs/in)",
        low_label="Lean", high_label="Wide",
        suffix=args.suffix, out_prefix="ratio_vs_gsax"
    ))

    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'Metric':<28} {'r':>7} {'p(corr)':>9} {'Mean Low':>10} {'Mean High':>10} "
          f"{'n_low':>6} {'n_high':>7} {'p(t-test)':>10}")
    print("-" * 95)
    for row in results:
        sig_c = " *" if row["p_corr"]  < 0.05 else "   "
        sig_t = " *" if row["p_ttest"] < 0.05 else "   "
        print(f"{row['metric']:<28} {row['r']:>7.3f} {row['p_corr']:>7.3f}{sig_c} "
              f"{row['mean_low']:>10.3f} {row['mean_high']:>10.3f} "
              f"{row['n_low']:>6} {row['n_high']:>7} {row['p_ttest']:>8.3f}{sig_t}")

    print("\nDone.")


if __name__ == "__main__":
    main()
