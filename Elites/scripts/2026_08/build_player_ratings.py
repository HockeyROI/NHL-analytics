"""build_player_ratings.py -- Elites "Player Ratings" for the Streamlit app.

Publishes the EV (5v5) Player Rating as a 0-100 index (Rating = tier_score x 100,
so ~50 = position-group average, matching the app's Zone Impact / Quality Games
convention) plus the Elite/Middle/Poor Tier, for every scope the app offers:

  single seasons : 2022-23, 2023-24, 2024-25, 2025-26   (same-season basis)
  2yr pool       : 2024-25 + 2025-26
  4yr pool ("pooled") : 2022-23 .. 2025-26

This is the SAME-SEASON / pooled rating (describes the window itself) -- NOT the
prior-two-season anti-circular tier that feeds elite_exposure/elite_support. cxG
is not needed (it's not in the locked formula), so the slow chaos walk is skipped.

Formula (single source of truth = tiers.position_score):
  Forwards:   0.40 pct(EV TOI/GP) + 0.30 pct(ixG/60) + 0.30 pct(primary pts/60)
  Defensemen: 0.60 pct(EV TOI/GP) + 0.40 pct(primary pts/60)
Percentiles are within position group, among qualified players, per scope.

Output (mirrors Zones/adjusted_rankings/zone_index100/ layout so the app loads it
the same way): Elites/Output/player_ratings/{scope}_{forwards|defense}.csv
  cols: player_name, team, pos, GP, Rating, Tier, Dep, Shot, Prod, player_id
  (Dep/Shot/Prod = the component sub-indices, each pct x 100; Shot is blank for D,
   whose formula omits ixG.)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiers import (ev_toi_gp_team, ixg_by_season, pbp_5v5_scoring,   # noqa: E402
                   position_score, pct_rank, label_cuts, POS_CSV)

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))
OUTDIR = os.path.join(PROJECT, "Elites", "Output", "player_ratings")
os.makedirs(OUTDIR, exist_ok=True)

# Live-season blending. While 2026-27 is in progress, a same-season-only rating
# is too noisy (tiny sample early), so the live scope blends the PREVIOUS season
# with the current one (ratio-of-sums pooling: as 2026-27 games accrue, the live
# season's weight grows automatically). Once the live season is COMPLETE (team
# games-played reaches the full slate), the scope auto-reverts to current-season-
# only, so a finished 2026-27 behaves exactly like every other completed season.
LIVE_SEASON = "20262027"
PREV_SEASON = "20252026"          # the "previous season" blended in while live
FULL_SEASON_TEAM_GP = 84          # 2026-27 is an 84-game season; "done" at/after this

# scope label -> (seasons in scope, min EV minutes to qualify). The live "2026-27"
# entry is a placeholder; main() overrides its seasons+floor based on whether the
# live season is complete (see _live_scope).
SCOPES = {
    "2022-23": (["20222023"], 200),
    "2023-24": (["20232024"], 200),
    "2024-25": (["20242025"], 200),
    "2025-26": (["20252026"], 200),
    "2026-27": (["20262027"], 10),
    "2yr":     (["20242025", "20252026"], 400),
    "pooled":  (["20222023", "20232024", "20242025", "20252026"], 700),
}


def _live_scope(per: pd.DataFrame) -> tuple[list[str], float]:
    """Seasons + qualify floor for the live 2026-27 Rating scope.

    In progress  -> blend [previous, current], low floor so players show early.
    Complete (>= FULL_SEASON_TEAM_GP team games) -> current-season only, standard
    200-min floor, i.e. identical treatment to a finished season."""
    live_gp = per.loc[per["season"] == LIVE_SEASON, "gp"]
    max_gp = int(live_gp.max()) if len(live_gp) and pd.notna(live_gp.max()) else 0
    if max_gp >= FULL_SEASON_TEAM_GP:
        return [LIVE_SEASON], 200.0
    return [PREV_SEASON, LIVE_SEASON], 10.0


def main():
    print("[1/3] per-season components (TOI/GP, ixG, primary points) ...", flush=True)
    toi = ev_toi_gp_team()
    ixg = ixg_by_season()
    scor = pbp_5v5_scoring()
    per = toi.merge(ixg, on=["player_id", "season"], how="left") \
             .merge(scor, on=["player_id", "season"], how="left")
    for c in ["ixg", "g5v5", "a1_5v5"]:
        per[c] = per[c].fillna(0.0)

    # Live-season rating scope: blend previous+current while 2026-27 is in
    # progress, auto-revert to current-only once it's a full 84-game season.
    SCOPES["2026-27"] = _live_scope(per)
    _lv_seasons, _lv_floor = SCOPES["2026-27"]
    print(f"[live] 2026-27 Rating scope = {_lv_seasons} (floor {_lv_floor} EV min) "
          f"{'[blended: season in progress]' if len(_lv_seasons) > 1 else '[current-only: season complete]'}",
          flush=True)

    pos = pd.read_csv(POS_CSV)
    pos_grp = dict(zip(pos["player_id"].astype(int), pos["pos_group"].astype(str)))
    pos_det = dict(zip(pos["player_id"].astype(int), pos["position"].astype(str)))
    name = dict(zip(pos["player_id"].astype(int), pos["player_name"].astype(str)))

    # Live-season games-played, for display. The blended live scope ranks on the
    # pooled (prev+current) window, but the GP shown in the app should be the
    # CURRENT season's games only -- otherwise a 2026-27 view reads a two-season
    # total (e.g. 86) that looks like more than a full 84-game slate.
    _live_gp_map = (per.loc[per["season"] == LIVE_SEASON]
                    .set_index("player_id")["gp"].to_dict())
    _blended_live = len(SCOPES["2026-27"][0]) > 1
    # Players actually active in the live season (>=1 game). The blended scope
    # ranks only these -- each using their pooled prev+current stats -- so a
    # 2025-26-only player (not on a 2026-27 roster) never shows in the 2026-27
    # view, while current rookies with no prior season still qualify on their
    # current-season sample.
    _live_participants = {int(pid) for pid, gp in _live_gp_map.items()
                          if pd.notna(gp) and gp >= 1}

    print("[2/3] pooling per scope + scoring ...", flush=True)
    for scope, (seasons, min_min) in SCOPES.items():
        sub = per[per["season"].isin(seasons)]
        # pool components across the scope's seasons, per player
        agg = sub.groupby("player_id").agg(
            ev_toi_sec=("ev_toi_sec", "sum"),
            gp=("gp", "sum"),
            ixg=("ixg", "sum"),
            g5v5=("g5v5", "sum"),
            a1_5v5=("a1_5v5", "sum"),
        ).reset_index()
        # primary team over the scope = most EV TOI
        team = (sub.groupby(["player_id", "team"])["ev_toi_sec"].sum()
                .reset_index().sort_values("ev_toi_sec")
                .drop_duplicates("player_id", keep="last")
                .set_index("player_id")["team"])
        agg["team"] = agg["player_id"].map(team).fillna("")
        agg["pos_group"] = agg["player_id"].map(pos_grp)
        agg["pos"] = agg["player_id"].map(pos_det)
        agg["player_name"] = agg["player_id"].map(name)
        agg = agg[agg["pos_group"].isin(["F", "D"])].copy()
        # Blended live scope ranks only current-season participants (pooled
        # prev+current stats); drop players with no 2026-27 games.
        if scope == "2026-27" and _blended_live:
            agg = agg[agg["player_id"].astype(int).isin(_live_participants)].copy()
        agg["ev_min"] = agg["ev_toi_sec"] / 60.0
        agg = agg[agg["ev_min"] >= min_min].copy()          # qualify
        agg["ev_toi_gp"] = agg["ev_min"] / agg["gp"]
        agg["ixg60"] = agg["ixg"] / agg["ev_min"] * 60.0
        agg["prim_pts60"] = (agg["g5v5"] + agg["a1_5v5"]) / agg["ev_min"] * 60.0
        agg["GP"] = agg["gp"].astype(int)
        # Blended live scope: show CURRENT-season GP (not the pooled 2-season sum).
        # The Rating/ev_toi_gp still use the pooled window; only the display GP
        # is reset so "2026-27" never shows more than its games-so-far.
        if scope == "2026-27" and _blended_live:
            agg["GP"] = (agg["player_id"].map(_live_gp_map)
                         .fillna(0).astype(int))

        for posg, posfile in (("F", "forwards"), ("D", "defense")):
            q = agg[agg["pos_group"] == posg].copy()
            if len(q) < 5:
                continue
            pt = pct_rank(q["ev_toi_gp"])
            pi = pct_rank(q["ixg60"])
            pp = pct_rank(q["prim_pts60"])
            q["Rating"] = (position_score(posg, pt, pi, pp) * 100).round(1)
            q["Tier"] = label_cuts(q["Rating"], 0.15, 0.30)
            q["Dep"] = (pt * 100).round(1)
            q["Prod"] = (pp * 100).round(1)
            q["Shot"] = (pi * 100).round(1) if posg == "F" else np.nan
            out = q[["player_name", "team", "pos", "GP",
                     "Rating", "Tier", "Dep", "Shot", "Prod", "player_id"]] \
                .sort_values("Rating", ascending=False).reset_index(drop=True)
            fp = os.path.join(OUTDIR, f"{scope}_{posfile}.csv")
            out.to_csv(fp, index=False)

    print("[3/3] done. Files in", OUTDIR, flush=True)
    # quick sanity: top 10 forwards, pooled
    top = pd.read_csv(os.path.join(OUTDIR, "pooled_forwards.csv")).head(10)
    print("\nPOOLED (2022-2026) top 10 forwards:")
    print(top[["player_name", "team", "pos", "GP", "Rating", "Tier"]].to_string(index=False))
    print("\nFiles written:")
    for f in sorted(os.listdir(OUTDIR)):
        n = len(pd.read_csv(os.path.join(OUTDIR, f)))
        print(f"  {f}: {n} rows")


if __name__ == "__main__":
    main()
