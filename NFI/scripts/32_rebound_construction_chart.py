#!/usr/bin/env python3
"""Chart: team starter CNFI rebound rate vs team forward RelNFI_A% (TOI-wt)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
OUT = f"{ROOT}/NFI/output"
CHART_DIR = os.environ.get(
    "HOCKEYROI_CHARTS",
    "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL Analysis/2026 posts/Charts",
)
os.makedirs(CHART_DIR, exist_ok=True)

REB = pd.read_csv(f"{OUT}/goalie_rebound_control.csv")
PLAYERS = pd.read_csv(f"{OUT}/fully_adjusted/current_season_player_fully_adjusted.csv")
SHOTS = pd.read_csv(f"{OUT}/shots_tagged.csv")

CURRENT_SEASON = 20252026

# ---------- Team forward RelNFI_A%, TOI-weighted ----------
fwd = PLAYERS[(PLAYERS["position"] == "F") & (PLAYERS["season"] == CURRENT_SEASON)].copy()
fwd = fwd.dropna(subset=["RelNFI_A_pct", "toi_min", "team"])
team_fwd = (
    fwd.groupby("team").apply(
        lambda g: np.average(g["RelNFI_A_pct"], weights=g["toi_min"]),
        include_groups=False,
    ).rename("team_fwd_RelNFI_A").reset_index()
)

# ---------- Team starter goalie this season (most ES shots faced) ----------
st = SHOTS[
    (SHOTS["season"] == CURRENT_SEASON)
    & (SHOTS["state"] == "ES")
    & (SHOTS["period"].between(1, 3))
    & (SHOTS["goalie_id"].notna())
].copy()
st["goalie_id"] = st["goalie_id"].astype("int64")
st["defending_team"] = np.where(
    st["shooting_team_abbrev"] == st["home_team_abbrev"],
    st["away_team_abbrev"], st["home_team_abbrev"],
)
starter = (
    st.groupby(["defending_team", "goalie_id"]).size().rename("faced").reset_index()
      .sort_values(["defending_team", "faced"], ascending=[True, False])
      .drop_duplicates("defending_team", keep="first")
      .rename(columns={"defending_team": "team"})
)

# Attach starter's pooled CNFI rebound rate
df = team_fwd.merge(starter, on="team", how="left")
df = df.merge(
    REB[["goalie_id", "goalie_name", "CNFI_rebound_goal_rate", "CNFI_saves"]],
    on="goalie_id", how="left",
)

# Some teams' current-season starter may not have ≥500 pooled CNFI saves
# (rookies, low usage). For those, fall back to second-most-faced goalie
# with rebound data, else drop.
fallback_ranked = (
    st.groupby(["defending_team", "goalie_id"]).size().rename("faced").reset_index()
      .merge(REB[["goalie_id", "CNFI_rebound_goal_rate", "goalie_name", "CNFI_saves"]],
             on="goalie_id", how="inner")
      .sort_values(["defending_team", "faced"], ascending=[True, False])
)
fallback = fallback_ranked.drop_duplicates("defending_team", keep="first").rename(
    columns={"defending_team": "team"}
)[["team", "goalie_id", "goalie_name", "CNFI_rebound_goal_rate", "CNFI_saves", "faced"]]

# Merge fallback to fill missing
df = team_fwd.merge(fallback, on="team", how="left")
print(f"Teams: {len(df)}    With rebound rate: {df['CNFI_rebound_goal_rate'].notna().sum()}")
missing = df[df["CNFI_rebound_goal_rate"].isna()]
if len(missing):
    print("Teams missing a qualified starter (no goalie on the roster met the "
          "≥500-save pooled threshold):")
    print(missing[["team"]].to_string(index=False))

df = df.dropna(subset=["CNFI_rebound_goal_rate"]).reset_index(drop=True)

x_avg = df["CNFI_rebound_goal_rate"].mean()
y_avg = df["team_fwd_RelNFI_A"].mean()

# Quadrant assignment + correlation
df["q"] = np.select(
    [
        (df["CNFI_rebound_goal_rate"] >= x_avg) & (df["team_fwd_RelNFI_A"] >= y_avg),
        (df["CNFI_rebound_goal_rate"] >= x_avg) & (df["team_fwd_RelNFI_A"] < y_avg),
        (df["CNFI_rebound_goal_rate"] < x_avg) & (df["team_fwd_RelNFI_A"] < y_avg),
        (df["CNFI_rebound_goal_rate"] < x_avg) & (df["team_fwd_RelNFI_A"] >= y_avg),
    ],
    ["Matched", "Exposed", "Compensated", "Optimal"],
    default="?"
)

r, pval = stats.pearsonr(df["CNFI_rebound_goal_rate"], df["team_fwd_RelNFI_A"])
print(f"\nLeague avg X (rebound rate): {x_avg:.3f}")
print(f"League avg Y (RelNFI_A%):    {y_avg:.4f}")
print(f"Pearson r = {r:+.3f}   p = {pval:.4f}   n = {len(df)}\n")

print("Edmonton row:")
print(df[df["team"] == "EDM"].to_string(index=False))

# ---------- Chart ----------
HOCKEY_NAVY = "#0B2545"
HOCKEY_ORANGE = "#F26B21"
GREEN = "#5DAA7A"
RED = "#C05555"
YELLOW = "#D4A843"
BLUE = "#2E7DC4"
QCOL = {"Matched": GREEN, "Exposed": RED, "Compensated": YELLOW, "Optimal": BLUE}

plt.rcParams.update({
    "font.family": "Arial",
    "axes.edgecolor": HOCKEY_NAVY,
    "axes.labelcolor": HOCKEY_NAVY,
    "xtick.color": HOCKEY_NAVY,
    "ytick.color": HOCKEY_NAVY,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor("white")

xmin, xmax = df["CNFI_rebound_goal_rate"].min() - 0.10, df["CNFI_rebound_goal_rate"].max() + 0.15
ymin, ymax = df["team_fwd_RelNFI_A"].min() - 0.005, df["team_fwd_RelNFI_A"].max() + 0.005
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Quadrant background tints
x_frac = (x_avg - xmin) / (xmax - xmin)
y_frac = (y_avg - ymin) / (ymax - ymin)
ax.axhspan(y_avg, ymax, xmin=x_frac, xmax=1.0, facecolor=GREEN, alpha=0.10)   # top-right Matched
ax.axhspan(ymin, y_avg, xmin=x_frac, xmax=1.0, facecolor=RED, alpha=0.10)     # bot-right Exposed
ax.axhspan(ymin, y_avg, xmin=0.0, xmax=x_frac, facecolor=YELLOW, alpha=0.10)  # bot-left Compensated
ax.axhspan(y_avg, ymax, xmin=0.0, xmax=x_frac, facecolor=BLUE, alpha=0.10)    # top-left Optimal

# crosshair
ax.axvline(x_avg, color=HOCKEY_NAVY, linestyle="--", linewidth=1)
ax.axhline(y_avg, color=HOCKEY_NAVY, linestyle="--", linewidth=1)
ax.text(x_avg, ymax, f"  league avg {x_avg:.2f}", va="top", ha="left",
        fontsize=8, color=HOCKEY_NAVY, alpha=0.8)
ax.text(xmax, y_avg, f"  league avg {y_avg:+.3f}", va="bottom", ha="right",
        fontsize=8, color=HOCKEY_NAVY, alpha=0.8)

# Quadrant labels
ax.text(xmax - 0.02, ymax - 0.001, "Matched\nPoor control, strong suppression",
        ha="right", va="top", fontsize=10, color=GREEN, weight="bold")
ax.text(xmax - 0.02, ymin + 0.001, "Exposed\nPoor control, weak suppression",
        ha="right", va="bottom", fontsize=10, color=RED, weight="bold")
ax.text(xmin + 0.02, ymin + 0.001, "Compensated\nGood control, weaker suppression OK",
        ha="left", va="bottom", fontsize=10, color="#9B7E1E", weight="bold")
ax.text(xmin + 0.02, ymax - 0.001, "Optimal\nGood control + strong suppression",
        ha="left", va="top", fontsize=10, color=BLUE, weight="bold")

# Scatter
for q in QCOL:
    sel = df[df["q"] == q]
    ax.scatter(sel["CNFI_rebound_goal_rate"], sel["team_fwd_RelNFI_A"],
               s=160, color=QCOL[q], edgecolor=HOCKEY_NAVY, linewidth=1, zorder=3)

# Highlight EDM
edm = df[df["team"] == "EDM"]
ax.scatter(edm["CNFI_rebound_goal_rate"], edm["team_fwd_RelNFI_A"],
           s=320, facecolors="none", edgecolors=HOCKEY_ORANGE, linewidth=3, zorder=4,
           label="EDM")

# Labels
for _, r_ in df.iterrows():
    dx, dy = 6, 6
    ax.annotate(r_["team"], (r_["CNFI_rebound_goal_rate"], r_["team_fwd_RelNFI_A"]),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=9, color=HOCKEY_NAVY, weight="bold")

ax.set_xlabel("Starter Goalie CNFI Rebound Goal Rate / 60 saves  (right = worse control)",
              fontsize=11)
ax.set_ylabel("Team Forwards RelNFI_A% (TOI-weighted)  (up = stronger suppression)",
              fontsize=11)
ax.set_title("Goalie Rebound Profile vs Forward Suppression — Are Teams Matched?",
             fontsize=15, color=HOCKEY_NAVY, weight="bold", pad=22)
ax.text(0.5, 1.012,
        "Teams with poor rebound control goalies need higher RelNFI_A% forwards to compensate",
        transform=ax.transAxes, ha="center", fontsize=10.5, color=HOCKEY_NAVY)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", frameon=False)

fig.text(0.99, 0.01, "@HockeyROI", ha="right", fontsize=10,
         color=HOCKEY_ORANGE, weight="bold")
plt.tight_layout()
out = f"{CHART_DIR}/chart_rebound_construction.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved {out}")
