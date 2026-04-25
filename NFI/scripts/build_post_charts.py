"""Build 6 charts for HockeyROI Posts 6-10.

Skips Chart 6 (UFA value scatter) — that requires manual AAV / contract data
which is not in any project file.

Outputs to NFI/output/charts/:
  chart_team_rankings.png         (Post 6)
  chart_rebound_shot_type.png     (Post 7)
  chart_rebound_arrival.png       (Post 7)
  chart_oilers_nfi.png            (Post 8)
  chart_oilers_forwards.png       (Post 8)
  chart_scenery_divergence.png    (Post 10)
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path("/Users/ashgarg/Documents/HockeyROI")
OUT = ROOT / "NFI/output/charts"
OUT.mkdir(parents=True, exist_ok=True)

# Brand colors
BLUE       = "#2E7DC4"
LIGHT_BLUE = "#4AB3E8"
ORANGE     = "#FF6B35"
GREEN      = "#5DAA7A"
YELLOW     = "#D4A843"
RED        = "#C05555"
GREY       = "#888888"
DARK_TEXT  = "#1B3A5C"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
})

def watermark(ax, text="@HockeyROI"):
    ax.text(0.99, 0.01, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=GREY, alpha=0.6, style="italic")

def save(fig, name):
    fp = OUT / name
    fig.savefig(fp, dpi=180, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return fp

# =====================================================================
# CHART 1 — Team Rankings (Post 6)
# =====================================================================
def chart_team_rankings():
    df = pd.read_csv(ROOT / "NFI/output/fully_adjusted/current_season_team_fully_adjusted.csv")
    df = df.sort_values("NFI_pct_ZA", ascending=False).reset_index(drop=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(10, 11), facecolor="white")
    ax.set_facecolor("white")
    y = np.arange(n)[::-1]  # rank 1 at top
    colors = []
    for i in range(n):
        if   i < 8:        colors.append(GREEN)
        elif i < 16:       colors.append(BLUE)
        elif i < 24:       colors.append(YELLOW)
        else:              colors.append(RED)
    bars = ax.barh(y, df["NFI_pct_ZA"], color=colors, height=0.7,
                   edgecolor="#222", linewidth=0.5)
    # EDM outline override
    for i, t in enumerate(df["team"]):
        if t == "EDM":
            bars[i].set_edgecolor(ORANGE)
            bars[i].set_linewidth(2.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["team"], fontsize=10)
    ax.set_xlim(0.40, 0.62)
    ax.axvline(0.500, color=GREY, linestyle="--", linewidth=1.4, zorder=1)
    ax.text(0.500, n + 0.5, " league avg 0.500", color=GREY, fontsize=8.5,
            va="bottom", ha="left")
    for bar, v, t in zip(bars, df["NFI_pct_ZA"], df["team"]):
        ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                f"{v*100:.1f}%", va="center", fontsize=8.5, color=DARK_TEXT)
    ax.set_xlabel("NFI%_ZA (deployment-adjusted)", fontsize=11)
    ax.set_title("NHL Team Rankings — Net Front Impact 2025-26",
                 fontsize=14, color=DARK_TEXT, loc="left", pad=14, weight="bold")
    ax.text(0, 1.015, "NFI%_ZA: Fenwick filtered to dangerous zones, deployment adjusted",
            transform=ax.transAxes, fontsize=10, color="#555555", ha="left")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)
    watermark(ax)
    fig.tight_layout()
    return save(fig, "chart_team_rankings.png")

# =====================================================================
# CHART 2 — Rebound Shot Type (Post 7)
# =====================================================================
def chart_rebound_shot_type():
    rs = pd.read_csv(ROOT / "NFI/output/rebound_sequences.csv")
    # Filter: rebound within 3 sec, CNFI zone box on rebound coordinates
    f = rs[(rs["time_gap_secs"].between(0, 3)) &
           (rs["reb_x"].between(74, 89)) &
           (rs["reb_y"].abs() <= 9)].copy()
    grp = f.groupby("reb_shot_type").agg(attempts=("reb_is_goal","size"),
                                          goals=("reb_is_goal","sum")).reset_index()
    # Keep types with reasonable sample
    grp = grp[grp["attempts"] >= 30]
    grp["conv"] = grp["goals"] / grp["attempts"]
    grp = grp.sort_values("conv", ascending=False).reset_index(drop=True)
    league_avg = f["reb_is_goal"].mean()

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.set_facecolor("white")
    y = np.arange(len(grp))[::-1]
    colors = [ORANGE if i == 0 else BLUE for i in range(len(grp))]
    bars = ax.barh(y, grp["conv"], color=colors, height=0.65,
                   edgecolor="#222", linewidth=0.6)
    for bar, name, v, n_, g in zip(bars, grp["reb_shot_type"], grp["conv"],
                                     grp["attempts"], grp["goals"]):
        ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                f"{v*100:.1f}%   ({int(g)} of {int(n_)})", va="center",
                fontsize=10, color=DARK_TEXT)
    ax.axvline(league_avg, color=GREY, linestyle="--", linewidth=1.4, zorder=1)
    ax.text(league_avg, len(grp) + 0.15, f" CNFI rebound avg {league_avg*100:.1f}%",
            color=GREY, fontsize=9, va="bottom", ha="left")
    ax.set_yticks(y); ax.set_yticklabels(grp["reb_shot_type"].str.replace("-"," "), fontsize=11)
    ax.set_xlabel("Rebound conversion rate (goal / attempts)", fontsize=11)
    ax.set_xlim(0, max(grp["conv"]) * 1.45)
    ax.set_title("Not All Rebound Attempts Are Equal",
                 fontsize=15, color=DARK_TEXT, loc="left", pad=14, weight="bold")
    ax.text(0, 1.015, "Conversion Rate by Shot Type — CNFI Zone Rebounds, "
            f"{f['season'].nunique()} Seasons (N={len(f):,} attempts)",
            transform=ax.transAxes, fontsize=10, color="#555555", ha="left")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)
    watermark(ax)
    fig.tight_layout()
    return save(fig, "chart_rebound_shot_type.png")

# =====================================================================
# CHART 3 — Rebound Arrival Leaders (Post 7)
# Compute per-player rebound arrival rate per 60 ES from raw data.
# =====================================================================
def chart_rebound_arrival():
    print("    [chart 3] computing rebound-arrival per-player rate ...")
    rs = pd.read_csv(ROOT / "NFI/output/rebound_sequences.csv")
    # Filter to CNFI rebounds within 3 seconds
    rs = rs[(rs["time_gap_secs"].between(0, 3)) &
            (rs["reb_x"].between(74, 89)) &
            (rs["reb_y"].abs() <= 9)].copy()
    # Need rebound time absolute. rebound_sequences doesn't have time, so we need
    # to map (game_id, period, reb_shooter) → time. Use shots_tagged.csv to find
    # the matching shot event.
    st = pd.read_csv(ROOT / "NFI/output/shots_tagged.csv",
                     usecols=["game_id","period","abs_time","event_type",
                              "shooting_team_abbrev","x_coord_norm","y_coord_norm",
                              "shooter_player_id","state"])
    # Filter to ES + Fenwick events that are likely rebounds (in CNFI box)
    st_es = st[(st["state"] == "ES") &
               (st["event_type"].isin(["shot-on-goal","missed-shot","goal"])) &
               (st["x_coord_norm"].between(74, 89)) &
               (st["y_coord_norm"].abs() <= 9)].copy()
    # Match rebound sequences on (game_id, period, reb_shooter_id)
    rs["key"] = (rs["game_id"].astype(str) + "_" +
                 rs["period"].astype(str) + "_" +
                 rs["reb_shooter_id"].fillna(-1).astype(int).astype(str))
    st_es["key"] = (st_es["game_id"].astype(str) + "_" +
                    st_es["period"].astype(str) + "_" +
                    st_es["shooter_player_id"].fillna(-1).astype(int).astype(str))
    # For each rebound event, find an ES shot event with same key — keep one
    keep = st_es.drop_duplicates("key", keep="first")
    matched = rs.merge(keep[["key","abs_time","game_id"]],
                        on=["key","game_id"], how="inner",
                        suffixes=("","_st"))
    matched = matched.dropna(subset=["abs_time"])
    matched["abs_time"] = matched["abs_time"].astype(int)
    print(f"      matched rebound events with abs_time: {len(matched):,}")

    # Load shifts for these games
    sd = pd.read_csv(ROOT / "NFI/Geometry_post/Data/shift_data.csv",
        usecols=["game_id","player_id","abs_start_secs","abs_end_secs"])
    sd = sd.dropna(subset=["player_id","abs_start_secs","abs_end_secs"])
    sd["game_id"] = sd["game_id"].astype("int64")
    sd = sd[sd["game_id"].isin(matched["game_id"].unique())]
    sd["player_id"] = sd["player_id"].astype("int64")
    sd["abs_start_secs"] = sd["abs_start_secs"].astype("int32")
    sd["abs_end_secs"]   = sd["abs_end_secs"].astype("int32")
    sd = sd.sort_values(["game_id","abs_start_secs"]).reset_index(drop=True)
    shifts_by_game = {gid: (g["player_id"].to_numpy(),
                            g["abs_start_secs"].to_numpy(),
                            g["abs_end_secs"].to_numpy())
                      for gid, g in sd.groupby("game_id", sort=False)}

    # For each rebound event, find on-ice players at that abs_time
    counts = defaultdict(int)
    for r in matched.itertuples():
        gid = int(r.game_id); t = int(r.abs_time)
        if gid not in shifts_by_game: continue
        pids, starts, ends = shifts_by_game[gid]
        hi = bisect_left(starts, t + 1)  # all shifts that started <= t
        for i in range(hi):
            if ends[i] > t:
                counts[int(pids[i])] += 1

    # Use player_fully_adjusted.csv for total ES TOI per player (pooled)
    pp = pd.read_csv(ROOT / "NFI/output/fully_adjusted/player_fully_adjusted.csv")
    pp["toi_min"] = pd.to_numeric(pp.get("toi_min", pp.get("toi_sec", 0)/60), errors="coerce")
    grp = pp.groupby(["player_id","player_name","position"]).agg(
        toi_total=("toi_min","sum"),
        team_recent=("team","last")).reset_index()
    grp["arrivals"] = grp["player_id"].map(counts).fillna(0).astype(int)
    grp = grp[(grp["position"] == "F") & (grp["toi_total"] >= 2000)]
    grp["per60"] = grp["arrivals"] / grp["toi_total"] * 60.0
    # Wilson 95% CI on rate (treat as Poisson-ish on count over TOI)
    Z = 1.96
    n_eff = grp["arrivals"]
    # Wilson interval on rate per 60 — approximate with CI on (arrivals / TOI_60)
    grp["per60_lo"] = (grp["arrivals"] - Z*np.sqrt(grp["arrivals"])) / grp["toi_total"] * 60.0
    grp["per60_hi"] = (grp["arrivals"] + Z*np.sqrt(grp["arrivals"])) / grp["toi_total"] * 60.0
    grp = grp.sort_values("per60", ascending=False).head(20).reset_index(drop=True)
    median = grp["per60"].median()

    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor="white")
    ax.set_facecolor("white")
    y = np.arange(len(grp))[::-1]
    colors = []
    for _, r in grp.iterrows():
        if r["arrivals"] < 10:           colors.append(GREY)
        elif r["per60_lo"] >= median:    colors.append(GREEN)
        else:                             colors.append(BLUE)
    err = [grp["per60"] - grp["per60_lo"], grp["per60_hi"] - grp["per60"]]
    bars = ax.barh(y, grp["per60"], color=colors, height=0.65,
                   xerr=err, error_kw={"ecolor":"#444","lw":1.0,"capsize":3},
                   edgecolor="#222", linewidth=0.5)
    HIGHLIGHT = {"Anders Lee", "Zach Hyman"}
    for i, (bar, name, v, t) in enumerate(zip(bars, grp["player_name"],
                                               grp["per60"], grp["team_recent"])):
        is_hl = name in HIGHLIGHT
        ax.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                f"{v:.2f} /60   ({t})",
                va="center", fontsize=10,
                color=ORANGE if is_hl else DARK_TEXT,
                fontweight="bold" if is_hl else "normal")
    ax.set_yticks(y)
    labels = [f"{r['player_name']}" + ("  ●" if r["player_name"] in HIGHLIGHT else "")
              for _, r in grp.iterrows()]
    ax.set_yticklabels(labels, fontsize=10)
    for tick, name in zip(ax.get_yticklabels(), grp["player_name"]):
        if name in HIGHLIGHT:
            tick.set_color(ORANGE); tick.set_fontweight("bold")
    ax.set_xlabel("Rebound arrivals per 60 ES min (CNFI zone, ≤3 sec)", fontsize=11)
    ax.set_title("Rebound Arrival Rate per 60",
                 fontsize=15, color=DARK_TEXT, loc="left", pad=14, weight="bold")
    ax.text(0, 1.015, "Top 20 Forwards — ES Regulation Pooled 2022-23 through 2025-26 "
                       f"(min 2000 ES min)",
            transform=ax.transAxes, fontsize=10, color="#555555", ha="left")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)
    watermark(ax)
    fig.tight_layout()
    return save(fig, "chart_rebound_arrival.png")

# =====================================================================
# CHART 4 — Oilers NFI history (Post 8)
# =====================================================================
def chart_oilers_nfi():
    """Two-panel: line of EDM CNFI+MNFI shots-per-game vs league avg, +
    bar of EDM rank by season."""
    tm = pd.read_csv(ROOT / "NFI/output/team_level_all_metrics.csv")
    tm["cnfi_per_game"] = (tm["CNFI_CF"] + tm["MNFI_CF"]) / tm["gp"]
    season_labels = {20212022:"21-22", 20222023:"22-23", 20232024:"23-24",
                     20242025:"24-25", 20252026:"25-26"}
    tm["season_lbl"] = tm["season"].map(season_labels)

    edm = tm[tm["team"] == "EDM"].sort_values("season").copy()
    league = tm.groupby("season").agg(avg=("cnfi_per_game","mean")).reset_index()
    league["season_lbl"] = league["season"].map(season_labels)

    # EDM rank per season (lower rate = worse rank)
    tm["rank"] = tm.groupby("season")["cnfi_per_game"].rank(ascending=False, method="min").astype(int)
    edm_rank = tm[tm["team"] == "EDM"].sort_values("season").copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")

    # Left: line chart
    ax = axes[0]; ax.set_facecolor("white")
    seasons = edm["season_lbl"].tolist()
    ax.plot(seasons, edm["cnfi_per_game"], color=ORANGE, linewidth=2.5,
            marker="o", markersize=8, label="EDM")
    ax.plot(seasons, league.set_index("season_lbl").reindex(seasons)["avg"],
            color=GREY, linewidth=1.6, linestyle="--", marker="s",
            markersize=6, label="League avg")
    for s, v in zip(seasons, edm["cnfi_per_game"]):
        ax.text(s, v + 0.4, f"{v:.1f}", ha="center", fontsize=9, color=ORANGE,
                fontweight="bold")
    ax.set_ylabel("CNFI+MNFI shots per game (5v5 ES)", fontsize=11)
    ax.set_xlabel("Season", fontsize=11)
    ax.set_title("CNFI+MNFI Generation — EDM vs League",
                 fontsize=12, color=DARK_TEXT, loc="left", pad=10, weight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)

    # Right: rank bar (inverted Y so rank 1 at top)
    ax2 = axes[1]; ax2.set_facecolor("white")
    seasons_r = edm_rank["season_lbl"].tolist()
    ranks = edm_rank["rank"].tolist()
    rcolors = []
    for r in ranks:
        if   r >= 25: rcolors.append(RED)
        elif r >= 17: rcolors.append(YELLOW)
        else:         rcolors.append(BLUE)
    bars = ax2.bar(seasons_r, ranks, color=rcolors, edgecolor="#222", linewidth=0.6)
    for b, r in zip(bars, ranks):
        ax2.text(b.get_x() + b.get_width()/2, r + 0.5, f"#{r}",
                 ha="center", fontsize=11, color=DARK_TEXT, fontweight="bold")
    ax2.set_ylim(32.5, 0.5)  # inverted
    ax2.set_yticks([1, 8, 16, 24, 32])
    ax2.set_ylabel("EDM rank (lower = better)", fontsize=11)
    ax2.set_xlabel("Season", fontsize=11)
    ax2.set_title("EDM Rank Among 32 Teams",
                  fontsize=12, color=DARK_TEXT, loc="left", pad=10, weight="bold")
    ax2.axhline(8, color=GREY, linestyle=":", alpha=0.5)
    ax2.axhline(16, color=GREY, linestyle=":", alpha=0.5)
    ax2.axhline(24, color=GREY, linestyle=":", alpha=0.5)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--", zorder=0)

    fig.suptitle("The Oilers Net-Front Problem — 5 Seasons",
                 fontsize=15, color=DARK_TEXT, x=0.07, ha="left", weight="bold")
    fig.text(0.07, 0.93, "CNFI+MNFI Attempt Rate vs League Average and Team Rank",
             fontsize=10, color="#555555", ha="left")
    watermark(ax2)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return save(fig, "chart_oilers_nfi.png")

# =====================================================================
# CHART 5 — Oilers forwards scatter (Post 8)
# =====================================================================
def chart_oilers_forwards():
    pp = pd.read_csv(ROOT / "NFI/output/fully_adjusted/current_season_player_fully_adjusted.csv")
    pp = pp[(pp["team"] == "EDM") & (pp["position"] == "F") &
            (pp["toi_min"] >= 300)].copy()

    fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
    ax.set_facecolor("white")
    HIGHLIGHT = {"Connor McDavid", "Leon Draisaitl", "Zach Hyman", "Ryan Nugent-Hopkins"}
    sizes = (pp["toi_min"] / pp["toi_min"].max() * 380 + 60)
    for _, r in pp.iterrows():
        is_hl = r["player_name"] in HIGHLIGHT
        ax.scatter(r["NFI_pct_ZA"], r["RelNFI_F_pct"],
                   s=sizes.loc[r.name],
                   color=ORANGE if is_hl else BLUE, alpha=0.85 if is_hl else 0.55,
                   edgecolor="#222", linewidth=0.6 if not is_hl else 1.2,
                   zorder=3 if is_hl else 2)
        last_name = r["player_name"].split()[-1]
        ax.annotate(last_name, (r["NFI_pct_ZA"], r["RelNFI_F_pct"]),
                    xytext=(7, 4), textcoords="offset points",
                    fontsize=10 if is_hl else 9,
                    color=ORANGE if is_hl else DARK_TEXT,
                    fontweight="bold" if is_hl else "normal", zorder=4)
    ax.axvline(0.500, color=GREY, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(0.0, color=GREY, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("NFI%_ZA (deployment-adjusted dangerous-zone share)", fontsize=11)
    ax.set_ylabel("RelNFI_F% (offensive generation impact /60)", fontsize=11)
    ax.set_title("Oilers Forwards — Dangerous Zone Profile 2025-26",
                 fontsize=15, color=DARK_TEXT, loc="left", pad=14, weight="bold")
    ax.text(0, 1.015, f"NFI%_ZA vs Offensive Generation (RelNFI_F%) — N={len(pp)} EDM forwards (≥300 ES min)",
            transform=ax.transAxes, fontsize=10, color="#555555", ha="left")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--", zorder=0)
    watermark(ax)
    fig.tight_layout()
    return save(fig, "chart_oilers_forwards.png")

# =====================================================================
# CHART 7 — Change of scenery / divergence (Post 10)
# Computes CF%_ZA pooled rank, compares to NFI%_3A pooled rank.
# =====================================================================
def chart_scenery_divergence():
    pp = pd.read_csv(ROOT / "NFI/output/fully_adjusted/player_fully_adjusted.csv")
    if "toi_min" not in pp.columns:
        pp["toi_min"] = pp.get("toi_sec", 0) / 60
    # Career TOI-weighted CF%_ZA
    def tw(g, col):
        v = pd.to_numeric(g[col], errors="coerce")
        w = pd.to_numeric(g["toi_min"], errors="coerce")
        m = v.notna() & (w > 0)
        return float(np.average(v[m], weights=w[m])) if m.any() else np.nan

    rows = []
    for (pid, name, pos), g in pp.groupby(["player_id","player_name","position"]):
        toi = float(pd.to_numeric(g["toi_min"], errors="coerce").fillna(0).sum())
        if toi < 2000: continue
        rows.append({
            "player_id": int(pid), "player_name": name, "position": pos,
            "team_recent": g.sort_values("season")["team"].iloc[-1],
            "toi_total":   toi,
            "CF_pct_ZA":   tw(g, "CF_pct_ZA"),
            "NFI_pct_3A":  tw(g, "NFI_pct_3A"),
        })
    car = pd.DataFrame(rows).dropna(subset=["CF_pct_ZA","NFI_pct_3A"])
    car["rank_CF_ZA"]   = car["CF_pct_ZA"].rank(ascending=False, method="min").astype(int)
    car["rank_NFI_3A"]  = car["NFI_pct_3A"].rank(ascending=False, method="min").astype(int)
    car["divergence"]   = car["rank_CF_ZA"] - car["rank_NFI_3A"]  # positive = better on NFI
    top = car.sort_values("divergence", ascending=False).head(20).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 8), facecolor="white")
    ax.set_facecolor("white")
    y = np.arange(len(top))[::-1]
    bars = ax.barh(y, top["divergence"], color=GREEN, height=0.65,
                   edgecolor="#222", linewidth=0.5)
    for bar, r in zip(bars, top.itertuples()):
        ax.text(r.divergence + 4, bar.get_y() + bar.get_height()/2,
                f"+{r.divergence}   ({r.team_recent})  CF#{r.rank_CF_ZA} → NFI#{r.rank_NFI_3A}",
                va="center", fontsize=9.5, color=DARK_TEXT)
    labels = [f"{r['player_name']} ({r['position']})" for _, r in top.iterrows()]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Rank improvement: CF%_ZA → NFI%_3A   (positive = looks better on NFI)",
                  fontsize=11)
    ax.set_title("Suppressed by Corsi — Players Who Look Better on NFI%_3A",
                 fontsize=15, color=DARK_TEXT, loc="left", pad=14, weight="bold")
    ax.text(0, 1.015, "Rank Improvement from CF%_ZA to NFI%_3A — "
                       f"Pooled 2022-23 through 2025-26 (N={len(car)} players ≥ 2000 ES min)",
            transform=ax.transAxes, fontsize=10, color="#555555", ha="left")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)
    watermark(ax)
    fig.tight_layout()
    return save(fig, "chart_scenery_divergence.png")

# =====================================================================
if __name__ == "__main__":
    results = []
    builders = [
        ("1 chart_team_rankings",      chart_team_rankings),
        ("2 chart_rebound_shot_type",  chart_rebound_shot_type),
        ("3 chart_rebound_arrival",    chart_rebound_arrival),
        ("4 chart_oilers_nfi",         chart_oilers_nfi),
        ("5 chart_oilers_forwards",    chart_oilers_forwards),
        ("7 chart_scenery_divergence", chart_scenery_divergence),
    ]
    for name, fn in builders:
        try:
            fp = fn()
            size = fp.stat().st_size
            results.append((name, "OK", fp.name, size))
        except Exception as e:
            import traceback
            tb = traceback.format_exc().splitlines()[-3:]
            results.append((name, f"FAIL: {type(e).__name__}: {e}", None, 0))
            print(f"  {name} FAILED:")
            for l in tb: print(f"    {l}")
    print()
    print(f"{'chart':<32} {'status':<10} {'file':<36} {'bytes':>10}")
    for name, status, fn, size in results:
        print(f"{name:<32} {status:<10} {fn or '—':<36} {size:>10,}")
    print()
    print("[NOTE] Chart 6 (UFA value scatter) skipped — requires manual AAV / "
          "contract_status data not present in the project.")
