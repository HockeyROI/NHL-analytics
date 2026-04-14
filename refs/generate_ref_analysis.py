#!/usr/bin/env python3
"""
NHL Referee Penalty Analysis — Edmonton Oilers @HockeyROI
Outputs 1-6: see inline comments.
"""

import json, time, pickle, requests, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────────────────
BASE     = "https://api-web.nhle.com/v1"
REFS_DIR = Path("/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/Refs")
CSV_PATH = REFS_DIR / "oilers_penalties_3seasons.csv"
CACHE    = REFS_DIR / "cache"
CACHE.mkdir(exist_ok=True)

SEASONS   = ["20232024", "20242025", "20252026"]
EDM_ID    = 22
MIN_GAMES = 5
WORKERS   = 20
WM        = "@HockeyROI"

ALL_TEAMS = [
    "ANA","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET",
    "EDM","FLA","LAK","MIN","MTL","NJD","NSH","NYI","NYR","OTT",
    "PHI","PIT","SEA","SJS","STL","TBL","TOR","UTA","VAN","VGK","WPG","WSH",
]

# ─── Theme ─────────────────────────────────────────────────────────────────────
BG     = "#0d1117"
SURF   = "#161b22"
BORD   = "#30363d"
GRID   = "#21262d"
TXT    = "#e6edf3"
GREY   = "#8b949e"
ORANGE = "#FF4C00"
BLUE   = "#003DA5"
BLUE_L = "#4F7DC9"
GREEN  = "#3FB950"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   SURF,
    "axes.edgecolor":   BORD,
    "text.color":       TXT,
    "axes.labelcolor":  TXT,
    "xtick.color":      TXT,
    "ytick.color":      TXT,
    "grid.color":       GRID,
    "grid.alpha":       1.0,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})

# ─── HTTP ──────────────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "HockeyROI-Analysis/1.0"

def fetch(url, retries=3):
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=12)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(0.3 * (i + 1))
    return None

# ─── Chart helpers ─────────────────────────────────────────────────────────────
def style_ax(ax, title=None, xlabel=None, ylabel=None):
    for sp in ax.spines.values():
        sp.set_edgecolor(BORD)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    if title:  ax.set_title(title, color=TXT, fontsize=12, pad=10, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, color=GREY, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=GREY, fontsize=10)

def style_ax_h(ax, title=None, xlabel=None, ylabel=None):
    """Horizontal variant — grid on x axis."""
    for sp in ax.spines.values():
        sp.set_edgecolor(BORD)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    if title:  ax.set_title(title, color=TXT, fontsize=12, pad=10, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, color=GREY, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=GREY, fontsize=10)

def wm(fig):
    fig.text(0.99, 0.01, WM, ha="right", va="bottom",
             color=GREY, fontsize=9, alpha=0.75, style="italic")

def subtitle(fig, text):
    fig.text(0.5, 0.995, text, ha="center", va="top", color=GREY, fontsize=10)

def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  ✓ {path.name}")

def bar_labels_v(ax, bars, vals, fmt="{}", offset=0.15):
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    fmt.format(v), ha="center", va="bottom", color=TXT, fontsize=8)

def bar_labels_h(ax, bars, vals, offset=0.2):
    for bar, v in zip(bars, vals):
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", ha="left", color=TXT, fontsize=9)

def safe_name(ref):
    return ref.replace(" ", "_").replace(".", "")

def norm_type(s):
    return str(s).replace("-", " ").replace("_", " ").title()

# ─── Phase 1: Load Oilers data ─────────────────────────────────────────────────
print("─── Loading Oilers data ───")
oilers = pd.read_csv(CSV_PATH)
oilers["called_on_oilers"] = oilers["called_on_oilers"].astype(bool)
oilers["oilers_home"]      = oilers["oilers_home"].astype(bool)
oilers["penalty_type"]     = oilers["penalty_type"].apply(norm_type)

games_per_ref = oilers.groupby("referee")["game_id"].nunique()
target_refs   = sorted(games_per_ref[games_per_ref >= MIN_GAMES].index.tolist())
print(f"  {len(target_refs)} refs qualify (≥{MIN_GAMES} EDM games)")

# ─── Phase 2: Collect all game IDs ─────────────────────────────────────────────
def get_all_game_ids():
    gf = CACHE / "all_game_ids.pkl"
    if gf.exists():
        with open(gf, "rb") as f:
            ids = pickle.load(f)
        print(f"  Loaded {len(ids)} game IDs from cache")
        return ids
    print("  Fetching game IDs from team schedules...")
    ids = set()
    for season in SEASONS:
        for team in ALL_TEAMS:
            d = fetch(f"{BASE}/club-schedule-season/{team}/{season}")
            if d and "games" in d:
                for g in d["games"]:
                    if g.get("gameType") == 2:
                        ids.add(g["id"])
            time.sleep(0.05)
    ids = sorted(ids)
    with open(gf, "wb") as f:
        pickle.dump(ids, f)
    print(f"  {len(ids)} regular-season games")
    return ids

# ─── Phase 3: Fetch right-rail refs ───────────────────────────────────────────
def _fetch_refs(gid):
    cf = CACHE / f"refs_{gid}.json"
    if cf.exists():
        with open(cf) as f:
            return gid, json.load(f)
    d = fetch(f"{BASE}/gamecenter/{gid}/right-rail")
    refs = []
    if d:
        for o in d.get("gameInfo", {}).get("referees", []):
            name = o.get("default", "").strip()
            if name:
                refs.append(name)
    with open(cf, "w") as f:
        json.dump(refs, f)
    return gid, refs

def get_all_game_refs(game_ids):
    rf = CACHE / "game_refs.pkl"
    if rf.exists():
        with open(rf, "rb") as f:
            gr = pickle.load(f)
        print(f"  Loaded ref assignments for {len(gr)} games from cache")
        return gr
    print(f"  Fetching refs for {len(game_ids)} games...")
    gr = {}
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_fetch_refs, g): g for g in game_ids}
        for fut in as_completed(futs):
            gid, refs = fut.result()
            gr[gid] = refs
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{len(game_ids)}")
    with open(rf, "wb") as f:
        pickle.dump(gr, f)
    return gr

# ─── Phase 4: Fetch play-by-play for target-ref games ─────────────────────────
def _fetch_penalties(gid):
    cf = CACHE / f"pen_{gid}.pkl"
    if cf.exists():
        with open(cf, "rb") as f:
            return gid, pickle.load(f)
    d = fetch(f"{BASE}/gamecenter/{gid}/play-by-play")
    rows = []
    if d:
        home_id  = d.get("homeTeam", {}).get("id")
        away_id  = d.get("awayTeam", {}).get("id")
        season   = d.get("season", 0)
        gdate    = d.get("gameDate", "")
        for play in d.get("plays", []):
            if play.get("typeDescKey") != "penalty":
                continue
            det   = play.get("details", {})
            tid   = det.get("eventOwnerTeamId")
            ptype = norm_type(det.get("descKey", "Unknown"))
            dur   = det.get("duration", 2)
            per   = play.get("periodDescriptor", {}).get("number", 0)
            rows.append({
                "game_id":      gid,
                "season":       season,
                "date":         gdate,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "team_id":      tid,
                "penalty_type": ptype,
                "duration":     dur,
                "period":       per,
            })
    with open(cf, "wb") as f:
        pickle.dump(rows, f)
    return gid, rows

def build_league_df(game_ids, game_refs):
    lf = CACHE / "league_penalties.pkl"
    if lf.exists():
        with open(lf, "rb") as f:
            df = pickle.load(f)
        print(f"  Loaded league data from cache: {len(df):,} rows")
        return df

    target_set   = set(target_refs)
    target_games = [g for g, refs in game_refs.items()
                    if any(r in target_set for r in refs)]
    print(f"  Fetching play-by-play for {len(target_games)} games with target refs...")

    all_rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_fetch_penalties, g): g for g in target_games}
        for fut in as_completed(futs):
            gid, rows = fut.result()
            all_rows.extend(rows)
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{len(target_games)}")

    # Expand: one row per (penalty × ref in that game)
    tg_set = set(target_games)
    expanded = []
    for row in all_rows:
        for ref in game_refs.get(row["game_id"], []):
            if ref in target_set:
                expanded.append({**row, "referee": ref})

    df = pd.DataFrame(expanded)
    with open(lf, "wb") as f:
        pickle.dump(df, f)
    print(f"  League dataset: {len(df):,} rows")
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT 1 — Total penalties AGAINST Oilers by ref, HOME games only
# OUTPUT 2 — Total penalties AGAINST Oilers by ref, AWAY games only
# ═══════════════════════════════════════════════════════════════════════════════
def chart_against_edm(df, refs, num, home: bool):
    tag  = "Home" if home else "Away"
    fname = f"output{num}_penalties_against_edm_{tag.lower()}.png"

    sub    = df[(df["oilers_home"] == home) & (df["called_on_oilers"]) &
                (df["referee"].isin(refs))]
    counts = sub.groupby("referee").size().sort_values(ascending=True)
    refs_shown = counts.index.tolist()

    fig, ax = plt.subplots(figsize=(14, max(7, len(refs_shown) * 0.48)))
    fig.patch.set_facecolor(BG)

    bars = ax.barh(refs_shown, counts.values,
                   color=ORANGE, edgecolor=BG, linewidth=0.4, height=0.68)
    bar_labels_h(ax, bars, counts.values.tolist())

    # Per-game rate annotation (lighter)
    games_by_ref = (df[df["oilers_home"] == home]
                    .groupby("referee")["game_id"].nunique())
    for bar, ref in zip(bars, refs_shown):
        g = games_by_ref.get(ref, 1)
        rate = counts[ref] / g
        ax.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{rate:.2f}/game", va="center", ha="left",
                color=BG, fontsize=7.5, fontweight="bold")

    style_ax_h(ax,
               title=f"Penalties Called AGAINST Oilers — {tag} Games Only",
               xlabel=f"Total Penalties Against EDM ({tag})")
    ax.set_xlim(0, counts.max() * 1.18)
    subtitle(fig, f"3-Season Analysis 2023–26 · Refs with {MIN_GAMES}+ EDM Games")
    wm(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save(fig, REFS_DIR / fname)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT 3 — Per ref: penalty types against Oilers, home vs away
# ═══════════════════════════════════════════════════════════════════════════════
def chart_penalty_types_by_ref(df, refs):
    edm_pen = df[df["called_on_oilers"] & df["referee"].isin(refs)].copy()

    for ref in refs:
        sub = edm_pen[edm_pen["referee"] == ref]
        if sub.empty:
            continue

        home_c = sub[sub["oilers_home"]].groupby("penalty_type").size()
        away_c = sub[~sub["oilers_home"]].groupby("penalty_type").size()
        types  = sorted(set(home_c.index) | set(away_c.index))
        if not types:
            continue

        x  = np.arange(len(types))
        w  = 0.37
        hv = [int(home_c.get(t, 0)) for t in types]
        av = [int(away_c.get(t, 0)) for t in types]

        fig, ax = plt.subplots(figsize=(max(10, len(types) * 1.0), 6))
        fig.patch.set_facecolor(BG)

        b1 = ax.bar(x - w/2, hv, width=w, label="Home", color=BLUE,   edgecolor=BG, linewidth=0.4)
        b2 = ax.bar(x + w/2, av, width=w, label="Away", color=ORANGE, edgecolor=BG, linewidth=0.4)
        bar_labels_v(ax, b1, hv)
        bar_labels_v(ax, b2, av)

        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=38, ha="right")
        style_ax(ax,
                 title=f"{ref} — Penalty Types Called Against EDM",
                 ylabel="Count")
        ax.legend(facecolor=SURF, edgecolor=BORD, labelcolor=TXT, fontsize=10)
        subtitle(fig, "Home vs Away Split · 2023–26")
        wm(fig)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        save(fig, REFS_DIR / f"output3_{safe_name(ref)}_penalty_types_home_away.png")

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT 4 — Per ref: penalty types called league-wide (all 32 teams)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_leaguewide_types(league_df, refs):
    if league_df.empty:
        print("  No league data — skipping Output 4")
        return

    for ref in refs:
        sub = league_df[league_df["referee"] == ref]
        if sub.empty:
            continue

        counts = (sub.groupby("penalty_type").size()
                    .sort_values(ascending=False)
                    .head(20))
        types = counts.index.tolist()
        x = np.arange(len(types))

        fig, ax = plt.subplots(figsize=(max(10, len(types) * 0.95), 6))
        fig.patch.set_facecolor(BG)

        bars = ax.bar(x, counts.values, color=BLUE, edgecolor=BG, linewidth=0.4)
        bar_labels_v(ax, bars, counts.values.tolist())

        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=38, ha="right")
        style_ax(ax,
                 title=f"{ref} — Penalty Types Called League-Wide (All 32 Teams)",
                 ylabel="Total Penalties Called")
        g = league_df[league_df["referee"] == ref]["game_id"].nunique()
        subtitle(fig, f"3-Season Analysis 2023–26 · {g} Games Officiated")
        wm(fig)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        save(fig, REFS_DIR / f"output4_{safe_name(ref)}_leaguewide_types.png")

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT 5 — Net penalty differential per ref, home vs away
#   Net = (called against EDM) − (called against opponent in EDM games)
#   Positive = biased against EDM
# ═══════════════════════════════════════════════════════════════════════════════
def chart_net_differential(df, refs):
    sub = df[df["referee"].isin(refs)].copy()
    # +1 = against EDM, -1 = against opponent
    sub["dir"] = sub["called_on_oilers"].map({True: 1, False: -1})

    agg = (sub.groupby(["referee", "oilers_home"])["dir"].sum()
              .unstack(fill_value=0)
              .rename(columns={True: "Home", False: "Away"}))
    # Ensure both columns exist
    for col in ["Home", "Away"]:
        if col not in agg.columns:
            agg[col] = 0

    agg["Total"] = agg["Home"] + agg["Away"]
    agg = agg.sort_values("Total", ascending=True).drop(columns="Total")

    n = len(agg)
    y = np.arange(n)
    h = 0.36

    fig, ax = plt.subplots(figsize=(14, max(7, n * 0.55)))
    fig.patch.set_facecolor(BG)

    home_v = agg["Home"].values
    away_v = agg["Away"].values

    def bar_color(v, bright, dim): return bright if v >= 0 else dim

    hc = [bar_color(v, ORANGE, GREEN) for v in home_v]
    ac = [bar_color(v, "#FF7033", "#2EA043") for v in away_v]  # slightly different shades

    ax.barh(y - h/2, home_v, height=h, color=hc, edgecolor=BG, linewidth=0.4, label="Home")
    ax.barh(y + h/2, away_v, height=h, color=ac, edgecolor=BG, linewidth=0.4, label="Away",
            alpha=0.80)

    # Value labels
    for i, (hv, av) in enumerate(zip(home_v, away_v)):
        offset = 0.15
        ax.text(hv + (offset if hv >= 0 else -offset),
                y[i] - h/2, str(hv),
                va="center", ha="left" if hv >= 0 else "right",
                color=TXT, fontsize=8)
        ax.text(av + (offset if av >= 0 else -offset),
                y[i] + h/2, str(av),
                va="center", ha="left" if av >= 0 else "right",
                color=TXT, fontsize=8)

    ax.axvline(0, color=GREY, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(agg.index, fontsize=9)
    style_ax_h(ax,
               title="Net Penalty Differential — Positive = More Calls Against EDM",
               xlabel="Net (Against EDM − Against Opponent) in EDM Games")

    patches = [
        mpatches.Patch(color=ORANGE, label="Home — against EDM"),
        mpatches.Patch(color=GREEN,  label="Home — favors EDM"),
        mpatches.Patch(color="#FF7033", label="Away — against EDM", alpha=0.8),
        mpatches.Patch(color="#2EA043", label="Away — favors EDM", alpha=0.8),
    ]
    ax.legend(handles=patches, facecolor=SURF, edgecolor=BORD,
              labelcolor=TXT, fontsize=9, loc="lower right")

    subtitle(fig, f"3-Season Analysis 2023–26 · {MIN_GAMES}+ EDM Games")
    wm(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save(fig, REFS_DIR / "output5_net_differential_home_away.png")

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT 6 — Per ref: EDM penalty rate vs league-wide average rate
#   EDM rate  = penalties called on EDM per EDM game
#   League rate = total penalties per game across all games this ref worked,
#                 divided by 2 (per-team equivalent)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_edm_vs_league_rate(oilers_df, league_df, refs):
    if league_df.empty:
        print("  No league data — skipping Output 6")
        return

    edm_games_map = oilers_df.groupby("referee")["game_id"].nunique()
    edm_pen_map   = oilers_df[oilers_df["called_on_oilers"]].groupby("referee").size()
    lg_games_map  = league_df.groupby("referee")["game_id"].nunique()
    lg_pen_map    = league_df.groupby("referee").size()

    for ref in refs:
        eg = edm_games_map.get(ref, 0)
        ep = edm_pen_map.get(ref, 0)
        lg = lg_games_map.get(ref, 0)
        lp = lg_pen_map.get(ref, 0)

        if eg == 0 or lg == 0:
            continue

        edm_rate = ep / eg
        # League average per team per game: total_penalties / (games × 2 teams)
        lg_rate  = lp / (lg * 2)

        diff = edm_rate - lg_rate
        pct  = (diff / lg_rate * 100) if lg_rate > 0 else 0
        sign = "+" if diff > 0 else ""

        if diff > 0.15:
            bias_label = "BIASED AGAINST EDM"
            bias_color = ORANGE
        elif diff < -0.15:
            bias_label = "FAVORS EDM"
            bias_color = GREEN
        else:
            bias_label = "NEUTRAL"
            bias_color = GREY

        fig, ax = plt.subplots(figsize=(7, 5.5))
        fig.patch.set_facecolor(BG)

        labels = ["EDM Rate\n(penalties/game)", "League Avg Rate\n(per team/game)"]
        vals   = [edm_rate, lg_rate]
        colors = [ORANGE, BLUE]
        bars   = ax.bar(labels, vals, color=colors, edgecolor=BG,
                        linewidth=0.4, width=0.45)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + max(vals) * 0.02,
                    f"{v:.2f}", ha="center", va="bottom",
                    color=TXT, fontsize=13, fontweight="bold")

        # Bias callout
        ax.text(0.5, 0.91,
                f"{sign}{pct:.1f}% vs league avg  ·  {bias_label}",
                ha="center", transform=ax.transAxes,
                color=bias_color, fontsize=11, fontweight="bold")

        style_ax(ax, title=f"{ref} — EDM Penalty Rate vs League Average",
                 ylabel="Penalties Per Game")
        ax.set_ylim(0, max(vals) * 1.25)

        fig.text(0.5, 0.995,
                 f"EDM: {ep} penalties / {eg} games    ·    League: {lp} penalties / {lg} games",
                 ha="center", va="top", color=GREY, fontsize=9)
        wm(fig)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        save(fig, REFS_DIR / f"output6_{safe_name(ref)}_edm_vs_league_rate.png")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n─── Output 1: Home penalties against EDM ───")
    chart_against_edm(oilers, target_refs, 1, home=True)

    print("\n─── Output 2: Away penalties against EDM ───")
    chart_against_edm(oilers, target_refs, 2, home=False)

    print("\n─── Output 3: Penalty types per ref (home vs away) ───")
    chart_penalty_types_by_ref(oilers, target_refs)

    print("\n─── Output 5: Net differential (home vs away) ───")
    chart_net_differential(oilers, target_refs)

    # League-wide data fetch (cached after first run)
    print("\n─── Collecting league-wide data ───")
    game_ids  = get_all_game_ids()
    game_refs = get_all_game_refs(game_ids)
    league_df = build_league_df(game_ids, game_refs)

    print("\n─── Output 4: League-wide penalty types per ref ───")
    chart_leaguewide_types(league_df, target_refs)

    print("\n─── Output 6: EDM rate vs league rate per ref ───")
    chart_edm_vs_league_rate(oilers, league_df, target_refs)

    total = len(list(REFS_DIR.glob("output*.png")))
    print(f"\n✓ Done — {total} chart files in Refs/")

if __name__ == "__main__":
    main()
