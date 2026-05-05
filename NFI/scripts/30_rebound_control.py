#!/usr/bin/env python3
"""Build CNFI goalie rebound control metric and two charts."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
OUT_DIR = f"{ROOT}/NFI/output"
CHART_DIR = os.environ.get(
    "HOCKEYROI_CHARTS",
    "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL Analysis/2026 posts/Charts",
)
os.makedirs(CHART_DIR, exist_ok=True)

REB = pd.read_csv(f"{OUT_DIR}/rebound_sequences.csv")
SHOTS = pd.read_csv(f"{OUT_DIR}/shots_tagged.csv")
# Pooled v2 goalie GSAx (CNFI+MNFI; built by 22_pool_goalie_gsax.py).
# Native columns: goalie_id, goalie_name, team, n_seasons, games,
# total_faced, es_toi_min, GSAx, GSAx_per60.
NFI = pd.read_csv(f"{OUT_DIR}/goalie_nfi_gsax_pooled_v2.csv")
TOI = pd.read_csv(f"{OUT_DIR}/player_toi.csv")
TEAMS = pd.read_csv(f"{ROOT}/Goalies/Benchmarks Goalies/Data/goalie_team_lookup.csv")

ES_CODES = {1551, 1441, 1331}

# ---------- 1. Rebound sequences: <=2s, first shot in CNFI, saved ----------
reb = REB[
    (REB["time_gap_secs"] <= 2.0)
    & (REB["orig_event_type"] == "shot-on-goal")
    & REB["orig_x"].between(74, 89)
    & REB["orig_y"].abs().le(9)
    & REB["situation_code"].isin(ES_CODES)
].copy()

# ---------- 2. Determine goalie per (game_id, period) via shots_tagged ----------
st = SHOTS.dropna(subset=["goalie_id"]).copy()
st["goalie_id"] = st["goalie_id"].astype("int64")
st["defending_team_abbrev"] = np.where(
    st["shooting_team_abbrev"] == st["home_team_abbrev"],
    st["away_team_abbrev"], st["home_team_abbrev"],
)
gp_goalie = (
    st.groupby(["game_id", "period", "defending_team_abbrev"])["goalie_id"]
      .agg(lambda s: s.value_counts().idxmax())
      .reset_index()
      .rename(columns={"defending_team_abbrev": "defending_team", "goalie_id": "goalie_id_lookup"})
)

team_pair = SHOTS.groupby("game_id")[["home_team_abbrev", "away_team_abbrev"]].first().reset_index()
reb = reb.merge(team_pair, on="game_id", how="left")
reb["defending_team"] = np.where(
    reb["orig_team"] == reb["home_team_abbrev"],
    reb["away_team_abbrev"], reb["home_team_abbrev"],
)
reb = reb.merge(gp_goalie, on=["game_id", "period", "defending_team"], how="left")
reb["goalie_id"] = reb["goalie_id_lookup"].fillna(reb["orig_goalie_id"])
reb = reb.dropna(subset=["goalie_id"]).copy()
reb["goalie_id"] = reb["goalie_id"].astype("int64")

# ---------- 3. Rebound attempts/goals per goalie ----------
reb_agg = reb.groupby("goalie_id").agg(
    CNFI_rebound_attempts=("reb_event_type", "size"),
    CNFI_rebound_goals=("reb_is_goal", "sum"),
).reset_index()

# ---------- 4. Total CNFI saves per goalie from shots_tagged ----------
cnfi_shots = st[
    (st["state"] == "ES")
    & st["period"].between(1, 3)
    & st["x_coord_norm"].between(74, 89)
    & st["y_coord_norm"].abs().le(9)
    & (st["event_type"] == "shot-on-goal")
    & (st["is_goal_i"] == 0)
].copy()
saves_per = cnfi_shots.groupby("goalie_id").size().rename("CNFI_saves").reset_index()

agg = saves_per.merge(reb_agg, on="goalie_id", how="left").fillna({"CNFI_rebound_attempts": 0, "CNFI_rebound_goals": 0})
agg["CNFI_rebound_attempts"] = agg["CNFI_rebound_attempts"].astype(int)
agg["CNFI_rebound_goals"] = agg["CNFI_rebound_goals"].astype(int)
agg["CNFI_rebound_goal_rate"] = agg["CNFI_rebound_goals"] / agg["CNFI_saves"] * 60

# ---------- 5. Min 500 CNFI saves ----------
agg = agg[agg["CNFI_saves"] >= 500].copy()

# ---------- Names + team + ES TOI ----------
agg = agg.merge(NFI[["goalie_id", "goalie_name"]], on="goalie_id", how="left")

TEAMS_latest = (
    TEAMS.sort_values("season").drop_duplicates("goalie_id", keep="last")[["goalie_id", "goalie_team"]]
        .rename(columns={"goalie_team": "team"})
)
agg = agg.merge(TEAMS_latest, on="goalie_id", how="left")

TOI_g = TOI[TOI["position"] == "G"][["player_id", "toi_ES_sec"]].copy()
TOI_g["es_toi"] = (TOI_g["toi_ES_sec"] / 60.0).round(2)
agg = agg.merge(
    TOI_g.rename(columns={"player_id": "goalie_id"})[["goalie_id", "es_toi"]],
    on="goalie_id", how="left",
)

# ---------- 6. League avg + z-score ----------
total_g = agg["CNFI_rebound_goals"].sum()
total_s = agg["CNFI_saves"].sum()
league_avg_rate = total_g / total_s * 60
agg["league_avg_rate"] = round(league_avg_rate, 4)
mu = agg["CNFI_rebound_goal_rate"].mean()
sd = agg["CNFI_rebound_goal_rate"].std(ddof=0)
agg["z_score"] = ((agg["CNFI_rebound_goal_rate"] - mu) / sd).round(3)

cols = [
    "goalie_id", "goalie_name", "team",
    "CNFI_saves", "CNFI_rebound_attempts", "CNFI_rebound_goals",
    "CNFI_rebound_goal_rate", "league_avg_rate", "z_score", "es_toi",
]
agg = agg[cols].sort_values("CNFI_rebound_goal_rate").reset_index(drop=True)
agg["CNFI_rebound_goal_rate"] = agg["CNFI_rebound_goal_rate"].round(4)
agg.to_csv(f"{OUT_DIR}/goalie_rebound_control.csv", index=False)

# ---------- Report ----------
print(f"\nQualified goalies (>=500 CNFI saves): {len(agg)}")
print(f"League weighted-average CNFI rebound goal rate /60: {league_avg_rate:.4f}")
print(f"Mean of goalie rates: {mu:.4f}  | SD: {sd:.4f}\n")

print("=== TOP 5 (best — lowest rate) ===")
print(agg.head(5).to_string(index=False))
print("\n=== BOTTOM 5 (worst — highest rate) ===")
print(agg.tail(5).iloc[::-1].to_string(index=False))

print("\n=== Tristan Jarry ===")
jarry = agg[agg["goalie_name"].str.contains("Jarry", case=False, na=False)]
print(jarry.to_string(index=False) if len(jarry) else "Jarry not found / below threshold")

# Correlation w/ NFI-GSAx /60 — read GSAx_per60 directly from v2 file
# (no manual per-60 computation needed; the pooled v2 file pre-computes it
# from cumulative GSAx and pooled ES TOI minutes).
NFI_per60 = NFI[["goalie_id", "GSAx_per60"]].rename(
    columns={"GSAx_per60": "NFI_GSAx_per60"}
)
corr_df = agg.merge(NFI_per60, on="goalie_id", how="left").dropna(subset=["NFI_GSAx_per60"])
corr = corr_df["CNFI_rebound_goal_rate"].corr(corr_df["NFI_GSAx_per60"])
print(f"\nCorrelation (CNFI rebound goal rate vs NFI-GSAx /60): r = {corr:.3f}  n = {len(corr_df)}")

# ---------- Charts ----------
HOCKEY_NAVY = "#0B2545"
HOCKEY_ORANGE = "#F26B21"
GREEN = "#5DAA7A"
RED = "#C05555"
YELLOW = "#E6C457"
plt.rcParams.update({
    "font.family": "Arial",
    "axes.edgecolor": HOCKEY_NAVY,
    "axes.labelcolor": HOCKEY_NAVY,
    "xtick.color": HOCKEY_NAVY,
    "ytick.color": HOCKEY_NAVY,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

# ---- CHART 1: best/worst bar chart ----
top10 = agg.head(10).iloc[::-1]
bot10 = agg.tail(10)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor("white")

def bar_panel(ax, df, color, title):
    y = np.arange(len(df))
    ax.barh(y, df["CNFI_rebound_goal_rate"], color=color, edgecolor=HOCKEY_NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels(df["goalie_name"].values, fontsize=10)
    ax.set_xlabel("CNFI Rebound Goal Rate / 60 saves", fontsize=11)
    ax.axvline(league_avg_rate, color=HOCKEY_NAVY, linestyle="--", linewidth=1.2,
               label=f"League avg {league_avg_rate:.2f}")
    xmax = df["CNFI_rebound_goal_rate"].max()
    pad = xmax * 0.04 + 0.01
    for i, (rate, z) in enumerate(zip(df["CNFI_rebound_goal_rate"].values, df["z_score"].values)):
        ax.text(rate + pad, i, f"z={z:+.2f}", va="center", fontsize=9, color=HOCKEY_NAVY)
    ax.set_xlim(0, xmax * 1.25)
    ax.set_title(title, fontsize=13, color=HOCKEY_NAVY, weight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

bar_panel(axes[0], top10, GREEN, "Best Rebound Control (Top 10)")
bar_panel(axes[1], bot10, RED, "Worst Rebound Control (Bottom 10)")

fig.suptitle("Goalie Rebound Control — CNFI Zone Goals Allowed on Rebounds",
             fontsize=16, color=HOCKEY_NAVY, weight="bold", y=0.99)
fig.text(0.5, 0.94, "2-Second Window  •  ES Regulation  •  Pooled 2022-23 through 2025-26",
         ha="center", fontsize=11, color=HOCKEY_NAVY)
fig.text(0.99, 0.01, "@HockeyROI", ha="right", fontsize=10,
         color=HOCKEY_ORANGE, weight="bold")
plt.tight_layout(rect=[0, 0.02, 1, 0.93])
out1 = f"{CHART_DIR}/chart_rebound_control.png"
plt.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved {out1}")

# ---- CHART 2: scatter (NFI-GSAx /60 vs inverted rebound rate) ----
sc = corr_df.copy()
fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor("white")

x_med = sc["NFI_GSAx_per60"].median()
y_med = sc["CNFI_rebound_goal_rate"].median()
xmin, xmax = sc["NFI_GSAx_per60"].min() - 0.05, sc["NFI_GSAx_per60"].max() + 0.05
ymin = sc["CNFI_rebound_goal_rate"].min() - 0.05
ymax = sc["CNFI_rebound_goal_rate"].max() + 0.05

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymax, ymin)  # inverted: lower y (better) at top

x_frac = (x_med - xmin) / (xmax - xmin)
ax.axhspan(ymin, y_med, xmin=x_frac, xmax=1.0, facecolor=GREEN, alpha=0.14)   # top-right Elite
ax.axhspan(y_med, ymax, xmin=0.0, xmax=x_frac, facecolor=RED, alpha=0.14)     # bottom-left Avoid
ax.axhspan(ymin, y_med, xmin=0.0, xmax=x_frac, facecolor=YELLOW, alpha=0.14)  # top-left Hidden Risk
ax.axhspan(y_med, ymax, xmin=x_frac, xmax=1.0, facecolor=YELLOW, alpha=0.14)  # bottom-right Undervalued

ax.axvline(x_med, color=HOCKEY_NAVY, linestyle=":", linewidth=1)
ax.axhline(y_med, color=HOCKEY_NAVY, linestyle=":", linewidth=1)

ax.text(xmax, ymin + 0.02, "Elite", ha="right", va="top",
        fontsize=12, color=GREEN, weight="bold")
ax.text(xmin, ymax - 0.02, "Avoid", ha="left", va="bottom",
        fontsize=12, color=RED, weight="bold")
ax.text(xmin, ymin + 0.02, "Hidden Risk", ha="left", va="top",
        fontsize=12, color="#9B7E1E", weight="bold")
ax.text(xmax, ymax - 0.02, "Undervalued", ha="right", va="bottom",
        fontsize=12, color="#9B7E1E", weight="bold")

is_jarry = sc["goalie_name"].str.contains("Jarry", case=False, na=False)
ax.scatter(sc.loc[~is_jarry, "NFI_GSAx_per60"], sc.loc[~is_jarry, "CNFI_rebound_goal_rate"],
           s=80, color=HOCKEY_NAVY, edgecolor="white", linewidth=1, zorder=3)
ax.scatter(sc.loc[is_jarry, "NFI_GSAx_per60"], sc.loc[is_jarry, "CNFI_rebound_goal_rate"],
           s=200, color=HOCKEY_ORANGE, edgecolor=HOCKEY_NAVY, linewidth=1.5, zorder=4, label="Tristan Jarry")

for _, r in sc.iterrows():
    ax.annotate(r["goalie_name"], (r["NFI_GSAx_per60"], r["CNFI_rebound_goal_rate"]),
                xytext=(4, 4), textcoords="offset points", fontsize=8, color=HOCKEY_NAVY)

ax.set_xlabel("NFI-GSAx per 60 (higher = better goalie)", fontsize=11)
ax.set_ylabel("CNFI Rebound Goal Rate / 60 saves  (inverted: top = better)", fontsize=11)
ax.set_title("Goalie Quality vs Rebound Control — Are They the Same Thing?",
             fontsize=15, color=HOCKEY_NAVY, weight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", frameon=False)

fig.text(0.99, 0.01, "@HockeyROI", ha="right", fontsize=10,
         color=HOCKEY_ORANGE, weight="bold")
plt.tight_layout()
out2 = f"{CHART_DIR}/chart_rebound_scatter.png"
plt.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out2}")
