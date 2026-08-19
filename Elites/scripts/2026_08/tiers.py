"""tiers.py -- Step 1 of the Elites pipeline.

One tier label (Elite / Middle / Poor) per player per season, computed ONLY from
the PRIOR TWO seasons -- never same-season data. This anti-circularity rule is
the whole point: a player's tier in season S is a fact known BEFORE S is played,
so downstream "who did you play against / with" metrics can't be contaminated by
the very season they describe.

Because a tier needs two prior seasons in the DB, tiers are produced for seasons
2022-2023 .. 2025-2026 (each uses the two seasons before it). 2020-21 and 2021-22
exist only as prior windows.

Pools: forwards and defensemen tiered separately. A player needs >= 500 EV(5v5)
minutes across the two-season window to be tiered; below that -> Middle by default.

Scores (percentiles are within-pool, among qualified players, for that season).
Position-specific by design: each position leans on the metrics its data actually
supports (validated via reliability = window-to-window repeatability across two
DISJOINT prior windows):
  Forwards:  0.40*pct(EV TOI/GP) + 0.30*pct(5v5 ixG/60) + 0.30*pct(5v5 prim pts/60)
  Defense:   0.60*pct(EV TOI/GP) + 0.40*pct(5v5 primary points/60)

  Why different: a forward's value is individually measurable (ixG r=.66, primary
  points r~.62), so weight those. A defenseman's value -- especially defense -- is
  hard to measure individually, so deployment (TOI, the most reliable signal r=.75)
  is the best proxy and carries more weight; it also recovers two-way credit that
  offense-only points miss. cxG (r=.49) and rel xGF% (r~.30) were tested and left
  OUT of the score for being too noisy; speed is orthogonal to quality, also out.

  ixG/60  = sum of xG on the player's OWN 5v5 shots, per 60 EV min.
  primary points/60 = (5v5 goals + 5v5 primary assists) per 60 EV min.
  cxG/60  = chaos xG: sum of xG on TEAMMATE shots within 30s after one of the
            player's own shots, same continuous play (never crossing a whistle) --
            the CCG logic (Zones/scripts/build_ccg_by_season.py) but summing shot
            xG. Kept as a REFERENCE column only (not in the score).

Cuts within each position pool (on the primary tier_score):
  Elite = top 15%, Middle = next 55%, Poor = bottom 30%.

SAME-SEASON tier (extra columns *_cur / tier_current / tier_score_current): the
identical position formula fed the CURRENT season's own stats instead of the prior
window. Qualify bar is lower (MIN_CURRENT_MIN, ~15 GP) so it's usable mid-season,
and it can be recomputed game-by-game as games accrue (only stable past ~mid-season).
This is for leaderboards / "who is elite RIGHT NOW" -- it is NOT anti-circular, so
elite_exposure / elite_support must keep using the prior-window `tier`, never this.

Also emitted for sensitivity/comparison:
  tier_score_alt : Forwards swap ixG -> cxG (chaos-weighted); Defense use the prior
    50/50 TOI/points weighting.  tier_alt = default cuts on that alt score.
  tier_poor15 / tier_poor20 / tier_poor30 : Elite fixed at top 15%, Poor cutoff
    swept at bottom 15% / 20% / 30%.  `tier` == tier_poor30 (the locked default).

Inputs:
  Elites/Output/ev_shifts.csv            (EV TOI, GP, team)         [step 0]
  Data/shot_events_by_season/*.parquet   (own shots)     + xG/output/shot_xg_per_event.csv
  Data/ccg_events_by_season/*.parquet    (chaos stream for cxG)
  Zones/raw/pbp/*.json                   (5v5 goals + primary assists)
  NFI/output/player_positions.csv        (F/D pool, names)

Intermediate cache (Elites/Output/, '_' prefix, safe to delete -- rebuilt on demand):
  _pbp_5v5_scoring.csv   : per player-season 5v5 goals + primary assists
Output:
  Elites/Output/tiers.csv

Prints the Edmonton roster (tiers, most recent season) for eyeballing, plus
row counts and the top/bottom 15 by tier_score.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))
DATA = os.path.join(PROJECT, "Data")
OUTDIR = os.path.join(PROJECT, "Elites", "Output")
EV_SHIFTS = os.path.join(OUTDIR, "ev_shifts.csv")
SHOT_DIR = os.path.join(DATA, "shot_events_by_season")
CCG_DIR = os.path.join(DATA, "ccg_events_by_season")
XG_CSV = os.path.join(PROJECT, "xG", "output", "shot_xg_per_event.csv")
PBP_DIR = os.path.join(PROJECT, "Zones", "raw", "pbp")
POS_CSV = os.path.join(PROJECT, "NFI", "output", "player_positions.csv")
PBP_CACHE = os.path.join(OUTDIR, "_pbp_5v5_scoring.csv")
SCORING_DIR = os.path.join(DATA, "elites_scoring_by_season")   # committed, CI-portable
OUT = os.path.join(OUTDIR, "tiers.csv")

ALL_SEASONS = ["20202021", "20212022", "20222023", "20232024", "20242025", "20252026"]
TIER_SEASONS = ALL_SEASONS[2:]           # need two priors -> 2022-23 .. 2025-26

FEN = {"shot-on-goal", "missed-shot", "goal"}
CCG_SHOTS = {"shot-on-goal", "missed-shot", "blocked-shot"}   # seed types (mirror CCG)
STOP = {"faceoff", "stoppage", "period-start", "period-end", "game-end",
        "penalty", "delayed-penalty"}
WIN = 30
MIN_WINDOW_MIN = 500        # qualify bar for the prior-two-season window
MIN_CURRENT_MIN = 200       # qualify bar for the same-season tier (~15 GP of 5v5;
                            # lower than the window so it's usable mid-season)


# ---------------------------------------------------------------------------
# Per-season building blocks (computed once per season, then windowed)
# ---------------------------------------------------------------------------
def ev_toi_gp_team():
    """From ev_shifts -> per (player_id, season): ev_toi_sec, GP, primary team."""
    toi = defaultdict(float)
    games = defaultdict(set)
    team_toi = defaultdict(float)               # (pid, season, team) -> sec
    df = pd.read_csv(EV_SHIFTS,
                     usecols=["game_id", "season", "duration",
                              "home_team", "away_team", "home_skaters", "away_skaters"],
                     dtype={"season": str})
    for gid, season, dur, hteam, ateam, hsk, ask in zip(
            df["game_id"], df["season"], df["duration"],
            df["home_team"], df["away_team"], df["home_skaters"], df["away_skaters"]):
        for ids, team in ((hsk, hteam), (ask, ateam)):
            for p in ids.split(";"):
                pid = int(p)
                toi[(pid, season)] += dur
                games[(pid, season)].add(gid)
                team_toi[(pid, season, team)] += dur
    # primary team per (pid, season)
    prim_team = {}
    best = defaultdict(float)
    for (pid, season, team), sec in team_toi.items():
        if sec > best[(pid, season)]:
            best[(pid, season)] = sec
            prim_team[(pid, season)] = team
    rows = []
    for (pid, season), sec in toi.items():
        rows.append({"player_id": pid, "season": season,
                     "ev_toi_sec": sec, "gp": len(games[(pid, season)]),
                     "team": prim_team.get((pid, season), "")})
    return pd.DataFrame(rows)


def ixg_by_season():
    """Per (player_id, season): sum xG on player's own 5v5 (strict 1551) shots."""
    xg = pd.read_csv(XG_CSV)
    rows = []
    for pq in sorted(glob.glob(os.path.join(SHOT_DIR, "*.parquet"))):
        e = pd.read_parquet(pq, columns=["game_id", "season", "event_id",
                                         "situation_code", "event_type",
                                         "shooter_player_id"])
        e = e[(e["situation_code"].astype(str) == "1551") & e["event_type"].isin(FEN)]
        e = e.dropna(subset=["shooter_player_id"])
        e = e.merge(xg, on=["game_id", "event_id"], how="left")
        e["shooter_player_id"] = e["shooter_player_id"].astype(int)
        e["season"] = e["season"].astype(str)
        g = e.groupby(["shooter_player_id", "season"])["xg"].sum().reset_index()
        g = g.rename(columns={"shooter_player_id": "player_id", "xg": "ixg"})
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def cxg_by_season():
    """Per (player_id, season): chaos xG created.

    For each of the player's own 5v5 shots (SOG/miss/block), sum the xG of every
    TEAMMATE fenwick shot that follows within 30s of the SAME continuous play
    (never crossing a whistle). Mirrors build_ccg_by_season's walk, summing xG.
    """
    xg = pd.read_csv(XG_CSV)
    # (game_id, abs, shooter, occ) -> xg, built from shot_events (has event_id)
    lut = {}
    for pq in sorted(glob.glob(os.path.join(SHOT_DIR, "*.parquet"))):
        e = pd.read_parquet(pq, columns=["game_id", "event_id", "period",
                                         "time_secs", "event_type", "shooter_player_id"])
        e = e[e["event_type"].isin(FEN)].dropna(subset=["shooter_player_id"]).copy()
        e["abs"] = (e["period"] - 1) * 1200 + e["time_secs"]
        e["shooter_player_id"] = e["shooter_player_id"].astype(int)
        e = e.merge(xg, on=["game_id", "event_id"], how="left")
        e = e.sort_values(["game_id", "abs", "event_id"])
        e["occ"] = e.groupby(["game_id", "abs", "shooter_player_id"]).cumcount()
        for gid, ab, sh, oc, x in zip(e["game_id"], e["abs"], e["shooter_player_id"],
                                      e["occ"], e["xg"]):
            lut[(int(gid), int(ab), int(sh), int(oc))] = 0.0 if pd.isna(x) else float(x)

    cxg = defaultdict(float)
    for pq in sorted(glob.glob(os.path.join(CCG_DIR, "*.parquet"))):
        c = pd.read_parquet(pq)
        c = c.sort_values(["game_id", "abs", "sort"]).reset_index(drop=True)
        # attach xg to fenwick shot rows via occurrence index
        is_fen = c["type"].isin(FEN)
        cf = c[is_fen].dropna(subset=["shooter"]).copy()
        cf["shooter"] = cf["shooter"].astype(int)
        cf["occ"] = cf.groupby(["game_id", "abs", "shooter"]).cumcount()
        xg_series = pd.Series(
            [lut.get((int(g), int(a), int(s), int(o)), 0.0)
             for g, a, s, o in zip(cf["game_id"], cf["abs"], cf["shooter"], cf["occ"])],
            index=cf.index)
        c["xg"] = 0.0
        c.loc[cf.index, "xg"] = xg_series.values

        for gid, g in c.groupby("game_id", sort=False):
            season = str(g["season"].iloc[0])
            types = g["type"].to_numpy()
            absl = g["abs"].to_numpy()
            team = g["team_id"].to_numpy()
            shooter = g["shooter"].to_numpy()
            sit = g["situation"].to_numpy()
            xgv = g["xg"].to_numpy()
            n = len(g)
            for i in range(n):
                if types[i] not in CCG_SHOTS or pd.isna(team[i]) or pd.isna(shooter[i]):
                    continue
                if not (isinstance(sit[i], str) and sit[i] == "5v5"):
                    continue
                t0, tm0, sh0 = absl[i], team[i], int(shooter[i])
                acc = 0.0
                j = i + 1
                while j < n:
                    tj = types[j]
                    if tj in STOP:
                        break
                    if absl[j] - t0 > WIN:
                        break
                    if (tj in FEN and team[j] == tm0 and not pd.isna(shooter[j])
                            and int(shooter[j]) != sh0
                            and isinstance(sit[j], str) and sit[j] == "5v5"):
                        acc += xgv[j]
                    j += 1
                if acc:
                    cxg[(sh0, season)] += acc
    return pd.DataFrame([{"player_id": k[0], "season": k[1], "cxg": v}
                         for k, v in cxg.items()])


def pbp_5v5_scoring():
    """Per (player_id, season): 5v5 goals + 5v5 primary assists.

    PREFERRED (CI-portable): aggregate from the committed per-season scoring
    parquets Data/elites_scoring_by_season/ (built by build_elites_scoring.py).
    FALLBACK: parse the raw PBP directly (local first build, or if the parquets
    are absent), cached to _pbp_5v5_scoring.csv.
    """
    pqs = sorted(glob.glob(os.path.join(SCORING_DIR, "*.parquet")))
    if pqs:
        ev = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
        ev["season"] = ev["season"].astype(str)
        gg = (ev.dropna(subset=["scorer"]).astype({"scorer": "int64"})
              .groupby(["scorer", "season"]).size().rename("g5v5").reset_index()
              .rename(columns={"scorer": "player_id"}))
        aa = (ev.dropna(subset=["assist1"]).astype({"assist1": "int64"})
              .groupby(["assist1", "season"]).size().rename("a1_5v5").reset_index()
              .rename(columns={"assist1": "player_id"}))
        df = gg.merge(aa, on=["player_id", "season"], how="outer")
        df[["g5v5", "a1_5v5"]] = df[["g5v5", "a1_5v5"]].fillna(0).astype(int)
        df["player_id"] = df["player_id"].astype(int)
        return df
    if os.path.exists(PBP_CACHE):
        df = pd.read_csv(PBP_CACHE)
        df["season"] = df["season"].astype(str)
        df["player_id"] = df["player_id"].astype(int)
        return df
    g = defaultdict(int)     # (pid, season) -> 5v5 goals
    a1 = defaultdict(int)    # (pid, season) -> 5v5 primary assists
    files = sorted(glob.glob(os.path.join(PBP_DIR, "*.json")))
    for k, path in enumerate(files):
        try:
            gm = json.load(open(path))
        except Exception:
            continue
        if gm.get("gameType") != 2:                    # regular season only
            continue
        season = str(gm.get("season"))
        for p in gm.get("plays") or []:
            if p.get("typeDescKey") != "goal":
                continue
            if str(p.get("situationCode")) != "1551":  # strict 5v5, goalies in
                continue
            d = p.get("details") or {}
            sc = d.get("scoringPlayerId")
            if sc:
                g[(int(sc), season)] += 1
            a = d.get("assist1PlayerId")
            if a:
                a1[(int(a), season)] += 1
        if (k + 1) % 1500 == 0:
            print(f"    [pbp] parsed {k+1}/{len(files)} games", flush=True)
    keys = set(g) | set(a1)
    df = pd.DataFrame([{"player_id": int(pid), "season": str(s),
                        "g5v5": g.get((pid, s), 0), "a1_5v5": a1.get((pid, s), 0)}
                       for (pid, s) in keys])
    df.to_csv(PBP_CACHE, index=False)
    print(f"    [pbp] wrote cache {PBP_CACHE} ({len(df)} rows)", flush=True)
    return df


# ---------------------------------------------------------------------------
def pct_rank(s):
    """Percentile rank in [0,1] among non-null values (average ties)."""
    return s.rank(pct=True, method="average")


def position_score(pos, pct_toi, pct_ixg, pct_pts):
    """The locked position-specific tier score, from within-pool percentiles.
    Forwards:   0.40 TOI + 0.30 ixG + 0.30 primary points
    Defensemen: 0.60 TOI + 0.40 primary points  (TOI-heavier; see main() notes).
    Single source of truth -- used for BOTH the prior-window and same-season tiers."""
    if pos == "F":
        return 0.40 * pct_toi + 0.30 * pct_ixg + 0.30 * pct_pts
    return 0.60 * pct_toi + 0.40 * pct_pts


def label_cuts(score, elite_top=0.15, poor_bottom=0.30):
    """Elite = top `elite_top`, Poor = bottom `poor_bottom`, else Middle.
    score: Series of tier_scores (qualified players only). Returns label Series."""
    hi = score.quantile(1 - elite_top)
    lo = score.quantile(poor_bottom)
    out = pd.Series("Middle", index=score.index)
    out[score >= hi] = "Elite"
    out[score <= lo] = "Poor"
    # guard: if hi==lo edge, elite wins
    out[score >= hi] = "Elite"
    return out


def main():
    print("[1/5] EV TOI / GP / team from ev_shifts ...", flush=True)
    toi = ev_toi_gp_team()
    print("[2/5] ixG per season ...", flush=True)
    ixg = ixg_by_season()
    print("[3/5] cxG per season (chaos walk) ...", flush=True)
    cxg = cxg_by_season()
    print("[4/5] 5v5 goals + primary assists from raw PBP ...", flush=True)
    scor = pbp_5v5_scoring()

    pos = pd.read_csv(POS_CSV)
    pos_map = dict(zip(pos["player_id"].astype(int), pos["pos_group"].astype(str)))
    name_map = dict(zip(pos["player_id"].astype(int), pos["player_name"].astype(str)))

    # merge per-season components into one frame keyed (player_id, season)
    per = toi.merge(ixg, on=["player_id", "season"], how="left") \
             .merge(cxg, on=["player_id", "season"], how="left") \
             .merge(scor, on=["player_id", "season"], how="left")
    for c in ["ixg", "cxg", "g5v5", "a1_5v5"]:
        per[c] = per[c].fillna(0.0)
    per = per.set_index(["player_id", "season"])

    def get(pid, season, col):
        try:
            return float(per.loc[(pid, season), col])
        except KeyError:
            return 0.0

    print("[5/5] windowing over prior two seasons + pooling/cuts ...", flush=True)
    out_rows = []
    for season in TIER_SEASONS:
        p1, p2 = ALL_SEASONS[ALL_SEASONS.index(season) - 2], \
                 ALL_SEASONS[ALL_SEASONS.index(season) - 1]
        # universe = players who appear (skate 5v5) in the CURRENT season
        universe = toi[toi["season"] == season][["player_id"]].drop_duplicates()
        recs = []
        for pid in universe["player_id"].astype(int):
            posg = pos_map.get(pid)
            if posg not in ("F", "D"):
                continue
            # ---- SAME-SEASON (current) components: season S itself ----
            toi_c = get(pid, season, "ev_toi_sec")
            gp_c = get(pid, season, "gp")
            cur_min = toi_c / 60.0
            rec = {
                "player_id": pid, "position": posg,
                "cur_ev_min": cur_min, "gp_current": int(gp_c),
                "ev_toi_gp_cur": (toi_c / 60.0) / gp_c if gp_c > 0 else np.nan,
                "ixg60_cur": get(pid, season, "ixg") / cur_min * 60.0 if cur_min > 0 else np.nan,
                "cxg60_cur": get(pid, season, "cxg") / cur_min * 60.0 if cur_min > 0 else np.nan,
                "prim_pts60_cur": (get(pid, season, "g5v5") + get(pid, season, "a1_5v5"))
                                  / cur_min * 60.0 if cur_min > 0 else np.nan,
                "qualified_cur": cur_min >= MIN_CURRENT_MIN,
            }
            # ---- PRIOR-TWO-SEASON (window) components: seasons p1 + p2 ----
            toi_sec = get(pid, p1, "ev_toi_sec") + get(pid, p2, "ev_toi_sec")
            gp = get(pid, p1, "gp") + get(pid, p2, "gp")
            win_min = toi_sec / 60.0
            if win_min <= 0 or gp <= 0:
                # no prior data at all -> window tier Middle by default
                rec.update({"ev_toi_gp": np.nan, "window_ev_min": 0.0, "gp_window": 0,
                            "ixg60": np.nan, "cxg60": np.nan,
                            "prim_pts60": np.nan, "prim_ast60": np.nan, "qualified": False})
            else:
                ixg_w = get(pid, p1, "ixg") + get(pid, p2, "ixg")
                cxg_w = get(pid, p1, "cxg") + get(pid, p2, "cxg")
                g_w = get(pid, p1, "g5v5") + get(pid, p2, "g5v5")
                a1_w = get(pid, p1, "a1_5v5") + get(pid, p2, "a1_5v5")
                rec.update({
                    "ev_toi_gp": (toi_sec / 60.0) / gp,
                    "window_ev_min": win_min, "gp_window": int(gp),
                    "ixg60": ixg_w / win_min * 60.0,
                    "cxg60": cxg_w / win_min * 60.0,
                    "prim_pts60": (g_w + a1_w) / win_min * 60.0,
                    "prim_ast60": a1_w / win_min * 60.0,
                    "qualified": win_min >= MIN_WINDOW_MIN,
                })
            recs.append(rec)
        sdf = pd.DataFrame(recs)
        if sdf.empty:
            continue
        sdf["season"] = season
        sdf["player_name"] = sdf["player_id"].map(name_map)
        cur_team = {pid: t for (pid, s), t in
                    per["team"].items() if s == season}
        sdf["team"] = sdf["player_id"].map(cur_team).fillna("")

        # percentiles + scores + cuts within each position pool (qualified only)
        for col in ["tier_score", "tier_score_alt", "tier",
                    "tier_alt", "tier_poor15", "tier_poor20", "tier_poor30",
                    "pct_ev_toi_gp", "pct_ixg60", "pct_cxg60",
                    "pct_prim_pts60", "pct_prim_ast60"]:
            sdf[col] = np.nan if col.startswith("pct") or "score" in col else "Middle"

        for posg in ("F", "D"):
            q = sdf[(sdf["position"] == posg) & (sdf["qualified"])].copy()
            if len(q) < 5:
                continue
            q["pct_ev_toi_gp"] = pct_rank(q["ev_toi_gp"])
            q["pct_ixg60"] = pct_rank(q["ixg60"])
            q["pct_cxg60"] = pct_rank(q["cxg60"])
            q["pct_prim_pts60"] = pct_rank(q["prim_pts60"])
            q["pct_prim_ast60"] = pct_rank(q["prim_ast60"])
            # Forwards: deployment + two RELIABLE individual signals (ixG r=.66,
            # primary pts r~.62); cxG (r=.49) retired to a reference column.
            # Defensemen: TOI-heavier -- a D's (esp. defensive) value is hard to
            # measure individually, so deployment (most reliable, r=.75) is the best
            # proxy and recovers two-way credit points miss; weak D ixG left out.
            q["tier_score"] = position_score(posg, q["pct_ev_toi_gp"],
                                             q["pct_ixg60"], q["pct_prim_pts60"])
            if posg == "F":
                # alt: swap ixG -> cxG (chaos-weighted) to sanity-check the choice
                q["tier_score_alt"] = (0.40 * q["pct_ev_toi_gp"]
                                       + 0.30 * q["pct_cxg60"] + 0.30 * q["pct_prim_pts60"])
            else:
                # alt: the prior 50/50 TOI/points weighting, for comparison
                q["tier_score_alt"] = (0.50 * q["pct_ev_toi_gp"] + 0.50 * q["pct_prim_pts60"])
            q["tier_poor15"] = label_cuts(q["tier_score"], 0.15, 0.15)
            q["tier_poor20"] = label_cuts(q["tier_score"], 0.15, 0.20)
            q["tier_poor30"] = label_cuts(q["tier_score"], 0.15, 0.30)
            q["tier"] = q["tier_poor30"]
            q["tier_alt"] = label_cuts(q["tier_score_alt"], 0.15, 0.30)
            for c in ["pct_ev_toi_gp", "pct_ixg60", "pct_cxg60", "pct_prim_pts60",
                      "pct_prim_ast60", "tier_score", "tier_score_alt",
                      "tier", "tier_alt", "tier_poor15", "tier_poor20", "tier_poor30"]:
                sdf.loc[q.index, c] = q[c]

        # ---- SAME-SEASON tier: identical formula, fed the CURRENT season's stats.
        # NOT anti-circular -- do NOT feed this into elite_exposure/support (those
        # must use the prior-window `tier`). This is for leaderboards / "who is elite
        # right now", and it can be recomputed game-by-game as the season fills in
        # (only stable once players clear the ~200-min bar, ~mid-season).
        for col in ["tier_score_current", "pct_ev_toi_gp_cur",
                    "pct_ixg60_cur", "pct_prim_pts60_cur"]:
            sdf[col] = np.nan
        sdf["tier_current"] = "Middle"
        for posg in ("F", "D"):
            qc = sdf[(sdf["position"] == posg) & (sdf["qualified_cur"])].copy()
            if len(qc) < 5:
                continue
            qc["pct_ev_toi_gp_cur"] = pct_rank(qc["ev_toi_gp_cur"])
            qc["pct_ixg60_cur"] = pct_rank(qc["ixg60_cur"])
            qc["pct_prim_pts60_cur"] = pct_rank(qc["prim_pts60_cur"])
            qc["tier_score_current"] = position_score(
                posg, qc["pct_ev_toi_gp_cur"], qc["pct_ixg60_cur"], qc["pct_prim_pts60_cur"])
            qc["tier_current"] = label_cuts(qc["tier_score_current"], 0.15, 0.30)
            for c in ["pct_ev_toi_gp_cur", "pct_ixg60_cur", "pct_prim_pts60_cur",
                      "tier_score_current", "tier_current"]:
                sdf.loc[qc.index, c] = qc[c]
        out_rows.append(sdf)

    res = pd.concat(out_rows, ignore_index=True)
    cols = ["player_id", "player_name", "season", "team", "position",
            # --- prior-two-season tier (anti-circular; feeds exposure/support) ---
            "qualified", "window_ev_min", "gp_window", "ev_toi_gp",
            "ixg60", "cxg60", "prim_pts60", "prim_ast60",
            "pct_ev_toi_gp", "pct_ixg60", "pct_cxg60", "pct_prim_pts60", "pct_prim_ast60",
            "tier_score", "tier_score_alt",
            "tier", "tier_alt", "tier_poor15", "tier_poor20", "tier_poor30",
            # --- same-season tier (current; for leaderboards, NOT the pipeline) ---
            "qualified_cur", "cur_ev_min", "gp_current",
            "ev_toi_gp_cur", "ixg60_cur", "cxg60_cur", "prim_pts60_cur",
            "pct_ev_toi_gp_cur", "pct_ixg60_cur", "pct_prim_pts60_cur",
            "tier_score_current", "tier_current"]
    res = res[cols].sort_values(["season", "position", "tier_score"],
                                ascending=[True, True, False]).reset_index(drop=True)
    res.to_csv(OUT, index=False)

    # ---- reporting ----
    print("\n" + "=" * 78)
    print(f"tiers.csv: {len(res)} player-season rows -> {OUT}")
    print("Tier distribution (qualified players), by season/position:")
    q = res[res["qualified"]]
    print(pd.crosstab([q["season"], q["position"]], q["tier"]).to_string())

    print("\nSame-season (current) tier distribution, qualified >= "
          f"{MIN_CURRENT_MIN} EV min:")
    qc = res[res["qualified_cur"]]
    print(pd.crosstab([qc["season"], qc["position"]], qc["tier_current"]).to_string())

    recent = "20252026"
    edm = res[(res["season"] == recent) & (res["team"] == "EDM")] \
        .sort_values(["position", "tier_score"], ascending=[True, False])
    print("\n" + "=" * 78)
    print(f"EDMONTON (EDM) roster -- {recent}:  PRIOR-2yr tier (from "
          f"{ALL_SEASONS[-3]}+{ALL_SEASONS[-2]}) vs SAME-SEASON tier")
    print("=" * 78)
    show = ["player_name", "position", "tier_score", "tier",
            "tier_score_current", "tier_current", "cur_ev_min"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(edm[show].round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("TOP 15 by tier_score (qualified, most recent season):")
    top = q[q["season"] == recent].sort_values("tier_score", ascending=False).head(15)
    print(top[["player_name", "position", "team", "tier_score", "tier"]].round(3).to_string(index=False))
    print("\nBOTTOM 15 by tier_score (qualified, most recent season):")
    bot = q[q["season"] == recent].sort_values("tier_score").head(15)
    print(bot[["player_name", "position", "team", "tier_score", "tier"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
