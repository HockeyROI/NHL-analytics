"""build_elite_context.py -- "Elite Context" family for the Streamlit app.

Same-season quality-of-competition / support metrics per player, for the app's
scopes (single seasons 2022-23..2025-26 + 2yr(2024-26) + pooled(2022-26)).
Opponents / teammates are classified by the SAME-SEASON rating tier
(tiers.csv tier_current: Elite = top 15%, Poor = bottom 30% of the position).

Per player, over each scope -- three exposure buckets (a shift can be in more
than one), each = share of 5v5 ice vs a unit with >=1 Elite AND zero Poor:
  vsEliteF     : opposing FORWARDS only (>=1 elite fwd, 0 poor fwd).
  vsEliteD     : opposing DEFENSEMEN only (>=1 elite D, 0 poor D).
  vsElite      : ALL five opponents (full-strength: >=1 elite, 0 poor).
Plus:
  EliteSupport : % of 5v5 ice with >=1 Elite TEAMMATE on the ice (excludes self).
  xGFvsElite   : on-ice xGF% during the vsElite (full-strength) shifts.
  ixG60        : individual xG per 60 (from tiers.csv, same-season) -- the scatter
                 x-axis; darker bubble = less EliteSupport.

Output (mirrors player_ratings / zone_index100 layout so the app loads it the
same way): Elites/Output/elite_context/{scope}_{forwards|defense}.csv
  cols: player_name, team, pos, GP, vsEliteF, vsEliteD, vsElite, EliteSupport,
        xGFvsElite, ixG60, player_id
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))
DATA = os.path.join(PROJECT, "Data")
OUTBASE = os.path.join(PROJECT, "Elites", "Output")
EV_SHIFTS = os.path.join(OUTBASE, "ev_shifts.csv")
TIERS = os.path.join(OUTBASE, "tiers.csv")
SHOT_DIR = os.path.join(DATA, "shot_events_by_season")
XG_CSV = os.path.join(PROJECT, "xG", "output", "shot_xg_per_event.csv")
OUTDIR = os.path.join(OUTBASE, "elite_context")
os.makedirs(OUTDIR, exist_ok=True)

FEN = {"shot-on-goal", "missed-shot", "goal"}
SEASONS = ["20222023", "20232024", "20242025", "20252026"]
SEASON_LABEL = {"20222023": "2022-23", "20232024": "2023-24",
                "20242025": "2024-25", "20252026": "2025-26"}
SCOPES = {  # scope label -> (seasons, min EV min to qualify)
    "2022-23": (["20222023"], 200), "2023-24": (["20232024"], 200),
    "2024-25": (["20242025"], 200), "2025-26": (["20252026"], 200),
    "2yr": (["20242025", "20252026"], 400),
    "pooled": (SEASONS, 700),
}


def shots_by_game():
    xg = pd.read_csv(XG_CSV)
    out = {}
    for pq in sorted(glob.glob(os.path.join(SHOT_DIR, "*.parquet"))):
        e = pd.read_parquet(pq, columns=["game_id", "event_id", "period", "time_secs",
                                         "situation_code", "event_type",
                                         "shooting_team_id", "home_team_id"])
        e = e[(e["situation_code"].astype(str) == "1551") & e["event_type"].isin(FEN)].copy()
        e = e.merge(xg, on=["game_id", "event_id"], how="left").dropna(subset=["xg"])
        e["abs"] = (e["period"] - 1) * 1200 + e["time_secs"]
        e["home_shot"] = e["shooting_team_id"] == e["home_team_id"]
        e = e.sort_values(["game_id", "abs"])
        for gid, g in e.groupby("game_id", sort=False):
            out[int(gid)] = {"abs": g["abs"].to_numpy(), "home": g["home_shot"].to_numpy(),
                             "xg": g["xg"].to_numpy()}
    return out


def main():
    t = pd.read_csv(TIERS)
    t = t[t["season"].isin([int(s) for s in SEASONS])].copy()
    t["season"] = t["season"].astype(str)
    t["player_id"] = t["player_id"].astype(int)
    tier = {(int(r.player_id), r.season): (r.tier_current if isinstance(r.tier_current, str) else "Middle")
            for r in t.itertuples()}
    posg = {(int(r.player_id), r.season): r.position for r in t.itertuples()}
    meta = t.set_index(["player_id", "season"])

    print("[1/3] loading 5v5 shots ...", flush=True)
    shots = shots_by_game()

    print("[2/3] walking shifts (same-season tier classification) ...", flush=True)
    # per (pid, season): totals
    ev = defaultdict(float); sup = defaultdict(float)
    fse = defaultdict(float); fse_f = defaultdict(float); fse_d = defaultdict(float)
    fxgf = defaultdict(float); fxga = defaultdict(float)

    def bucket(labels, positions):
        """(full, fwd, blue) full-strength flags for one opposing five."""
        f = [l for l, pp in zip(labels, positions) if pp == "F"]
        d = [l for l, pp in zip(labels, positions) if pp == "D"]
        full = ("Elite" in labels) and ("Poor" not in labels)
        fwd = bool(f) and ("Elite" in f) and ("Poor" not in f)
        blue = bool(d) and ("Elite" in d) and ("Poor" not in d)
        return full, fwd, blue

    df = pd.read_csv(EV_SHIFTS, dtype={"season": str})
    df = df[df["season"].isin(SEASONS)]
    for gid, g in df.groupby("game_id", sort=False):
        sg = shots.get(int(gid)); s_abs = sg["abs"] if sg is not None else None
        for season, start, end, hsk, ask in zip(g["season"], g["start_time"], g["end_time"],
                                                 g["home_skaters"], g["away_skaters"]):
            home = [int(x) for x in hsk.split(";")]; away = [int(x) for x in ask.split(";")]
            dur = end - start
            hlab = [tier.get((p, season), "Middle") for p in home]
            alab = [tier.get((p, season), "Middle") for p in away]
            hpos = [posg.get((p, season), "F") for p in home]
            apos = [posg.get((p, season), "F") for p in away]
            # each side's exposure is to the OTHER side's five
            h_full, h_fwd, h_blue = bucket(alab, apos)
            a_full, a_fwd, a_blue = bucket(hlab, hpos)
            n_elite_home = sum(l == "Elite" for l in hlab)
            n_elite_away = sum(l == "Elite" for l in alab)
            hxg = axg = 0.0
            if sg is not None:
                lo = np.searchsorted(s_abs, start, side="right"); hi = np.searchsorted(s_abs, end, side="right")
                if hi > lo:
                    xs = sg["xg"][lo:hi]; hs = sg["home"][lo:hi]
                    hxg = xs[hs].sum(); axg = xs[~hs].sum()
            for k, p in enumerate(home):
                key = (p, season); ev[key] += dur
                if h_full:
                    fse[key] += dur; fxgf[key] += hxg; fxga[key] += axg
                if h_fwd: fse_f[key] += dur
                if h_blue: fse_d[key] += dur
                if n_elite_home - (hlab[k] == "Elite") > 0:   # >=1 elite OTHER teammate
                    sup[key] += dur
            for k, p in enumerate(away):
                key = (p, season); ev[key] += dur
                if a_full:
                    fse[key] += dur; fxgf[key] += axg; fxga[key] += hxg
                if a_fwd: fse_f[key] += dur
                if a_blue: fse_d[key] += dur
                if n_elite_away - (alab[k] == "Elite") > 0:
                    sup[key] += dur

    print("[3/3] pooling per scope + writing ...", flush=True)
    rows = []
    for (pid, season), evsec in ev.items():
        if (pid, season) not in meta.index:
            continue
        m = meta.loc[(pid, season)]
        ixg60 = float(m["ixg60_cur"]) if pd.notna(m["ixg60_cur"]) else np.nan
        rows.append({
            "player_id": pid, "season": season,
            "player_name": m["player_name"], "team": m["team"],
            "pos": m["position"], "grp": m["position"],
            "gp": float(m["gp_current"]) if pd.notna(m["gp_current"]) else 0.0,
            "ev": evsec, "fse": fse[(pid, season)],
            "fse_f": fse_f[(pid, season)], "fse_d": fse_d[(pid, season)],
            "sup": sup[(pid, season)],
            "fxgf": fxgf[(pid, season)], "fxga": fxga[(pid, season)],
            # raw individual xG (back out from same-season rate) for clean pooling
            "ixg": (ixg60 * evsec / 3600.0) if ixg60 == ixg60 else 0.0,
        })
    per = pd.DataFrame(rows)

    for scope, (seasons, minmin) in SCOPES.items():
        sub = per[per["season"].isin(seasons)]
        agg = sub.groupby("player_id").agg(
            ev=("ev", "sum"), fse=("fse", "sum"), fse_f=("fse_f", "sum"),
            fse_d=("fse_d", "sum"), sup=("sup", "sum"),
            fxgf=("fxgf", "sum"), fxga=("fxga", "sum"), ixg=("ixg", "sum"),
            gp=("gp", "sum")).reset_index()
        # name/team/pos = the row with most EV in scope
        top = (sub.sort_values("ev").drop_duplicates("player_id", keep="last")
               .set_index("player_id"))
        agg["player_name"] = agg["player_id"].map(top["player_name"])
        agg["team"] = agg["player_id"].map(top["team"])
        agg["pos"] = agg["player_id"].map(top["pos"])
        agg["grp"] = agg["player_id"].map(top["grp"])
        agg["ev_min"] = agg["ev"] / 60.0
        agg = agg[agg["ev_min"] >= minmin].copy()
        agg["vsEliteF"] = (agg["fse_f"] / agg["ev"] * 100).round(1)
        agg["vsEliteD"] = (agg["fse_d"] / agg["ev"] * 100).round(1)
        agg["vsElite"] = (agg["fse"] / agg["ev"] * 100).round(1)
        agg["EliteSupport"] = (agg["sup"] / agg["ev"] * 100).round(1)
        denom = agg["fxgf"] + agg["fxga"]
        agg["xGFvsElite"] = np.where(denom > 0, agg["fxgf"] / denom * 100, np.nan).round(1)
        agg["ixG60"] = (agg["ixg"] / agg["ev"] * 3600.0).round(2)
        agg["GP"] = agg["gp"].astype(int)
        for grp, posfile in (("F", "forwards"), ("D", "defense")):
            q = agg[agg["grp"] == grp].sort_values("vsElite", ascending=False)
            if len(q) < 5:
                continue
            out = q[["player_name", "team", "pos", "GP", "vsEliteF", "vsEliteD",
                     "vsElite", "EliteSupport", "xGFvsElite", "ixG60",
                     "player_id"]].reset_index(drop=True)
            out.to_csv(os.path.join(OUTDIR, f"{scope}_{posfile}.csv"), index=False)

    print("done ->", OUTDIR, flush=True)
    top = pd.read_csv(os.path.join(OUTDIR, "2025-26_forwards.csv")).head(8)
    print("\n2025-26 top forwards by vsElite:")
    print(top[["player_name", "team", "vsEliteF", "vsEliteD", "vsElite",
               "EliteSupport", "xGFvsElite", "ixG60"]].to_string(index=False))
    for f in sorted(os.listdir(OUTDIR)):
        print("  ", f, len(pd.read_csv(os.path.join(OUTDIR, f))), "rows")


if __name__ == "__main__":
    main()
