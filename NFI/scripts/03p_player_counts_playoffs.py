#!/usr/bin/env python3
"""Playoff player on-ice counts — playoff companion to script 03's per-season
player_counts_by_state_zone_per_season.csv (the NFI-A/60 / NFI-S/60 source).

Replicates script 03's shift<->shot on-ice attribution (state intervals + shift
intersection) but on PLAYOFF games (shots_tagged.csv game_id digits == "03").
Emits per playoff season + an `all_playoffs` pooled block. Same schema as the
regular per-season file so the app loader just appends `_playoffs`.

NO floors. Output: NFI/output/player_counts_by_state_zone_playoffs.csv
Does NOT modify any regular output.
"""
import os
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
OUT = f"{ROOT}/NFI/output"
SHOTS_FP = f"{OUT}/shots_tagged.csv"
SHIFT_FP = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
POS_FP = f"{OUT}/player_positions.csv"
OUT_FP = f"{OUT}/player_counts_by_state_zone_playoffs.csv"

STATES = ["ES", "PP", "PK", "4v4", "3v3"]
ZONES_ALL = ["CNFI", "MNFI", "FNFI", "Wide", "lane_other", "unk"]

print("Loading positions + playoff shots_tagged ...")
pos = pd.read_csv(POS_FP, dtype={"player_id": int})
pos_map = dict(zip(pos["player_id"], pos["pos_group"]))

sh = pd.read_csv(SHOTS_FP)
sh = sh[sh["game_id"].astype(str).str[4:6] == "03"].copy()   # playoff games only
sh["season"] = sh["season"].astype(str)
sh["abs_time"] = sh["abs_time"].astype(int)
print(f"  playoff tagged shots: {len(sh):,} | seasons: {sorted(sh['season'].unique())}")
shots_by_game = dict(tuple(sh.groupby("game_id")))
valid_gids = set(shots_by_game.keys())
game_season = sh.drop_duplicates("game_id").set_index("game_id")["season"].to_dict()
team_ab = sh.groupby("game_id").agg(home=("home_team_abbrev", "first"),
                                    away=("away_team_abbrev", "first")).to_dict("index")

print("Loading playoff shifts (streaming filter) ...")
shift_cols = ["game_id", "player_id", "period", "team_abbrev", "abs_start_secs", "abs_end_secs"]
parts = []
for ch in pd.read_csv(SHIFT_FP, usecols=shift_cols, chunksize=500000):
    ch = ch.dropna(subset=["game_id", "player_id", "period", "abs_start_secs", "abs_end_secs"])
    ch["game_id"] = ch["game_id"].astype(int)
    ch = ch[ch["game_id"].isin(valid_gids) & ch["period"].astype(int).between(1, 3)]
    if len(ch):
        for c in ("player_id", "abs_start_secs", "abs_end_secs"):
            ch[c] = ch[c].astype(int)
        parts.append(ch)
shifts = pd.concat(parts, ignore_index=True)
shifts_by_game = dict(tuple(shifts.groupby("game_id")))
print(f"  playoff shift rows: {len(shifts):,} | games with shifts: {len(shifts_by_game)}")

# Accumulators keyed by (pid, season)
ind_att = defaultdict(lambda: defaultdict(int))
ind_gl = defaultdict(lambda: defaultdict(int))
for_att = defaultdict(lambda: defaultdict(int))
for_gl = defaultdict(lambda: defaultdict(int))
ag_att = defaultdict(lambda: defaultdict(int))
ag_gl = defaultdict(lambda: defaultdict(int))
# Fenwick (unblocked) on-ice for/against — same keying as the Corsi _att dicts
# but excluding blocked-shot events. Add-only: feed two new trailing columns in
# player_counts_by_state_zone_playoffs.csv and nothing else.
for_fen = defaultdict(lambda: defaultdict(int))
ag_fen = defaultdict(lambda: defaultdict(int))
toi = defaultdict(lambda: defaultdict(float))   # (pid, season) -> {state: sec}

print("Per-game shift-shot join (playoffs) ...")
for gid, gs in shots_by_game.items():
    if gid not in shifts_by_game:
        continue
    season = str(game_season[gid])
    home_ab, away_ab = team_ab[gid]["home"], team_ab[gid]["away"]
    ev = gs[["abs_time", "state"]].sort_values("abs_time").reset_index(drop=True)
    intervals, prev_t, prev_state = [], 0, "ES"
    for _, r in ev.iterrows():
        t = int(r["abs_time"])
        if t > prev_t:
            intervals.append((prev_t, t, prev_state))
        prev_t, prev_state = t, r["state"]
    if prev_t < 3600:
        intervals.append((prev_t, 3600, prev_state))
    iv_s = np.array([i[0] for i in intervals]); iv_e = np.array([i[1] for i in intervals])
    iv_st = np.array([i[2] for i in intervals])

    shifts_by_team = {}
    for tab, tsh in shifts_by_game[gid].groupby("team_abbrev"):
        shifts_by_team[tab] = (tsh["abs_start_secs"].values, tsh["abs_end_secs"].values,
                               tsh["player_id"].values)

    for tab, (st, en, pids) in shifts_by_team.items():
        for i in range(len(pids)):
            pid, s, e = int(pids[i]), int(st[i]), int(en[i])
            if e <= s:
                continue
            lo = np.searchsorted(iv_e, s, side="right")
            for j in range(lo, len(iv_s)):
                if iv_s[j] >= e:
                    break
                ovl = min(e, iv_e[j]) - max(s, iv_s[j])
                if ovl > 0:
                    toi[(pid, season)][iv_st[j]] += ovl

    for _, x in gs.iterrows():
        t = int(x["abs_time"]); state = x["state"]; zone = x["zone"]
        is_goal = int(x["is_goal_i"]); shoot_ab = x["shooting_team_abbrev"]
        is_fen = x["event_type"] != "blocked-shot"   # Fenwick = unblocked attempts
        def_ab = away_ab if shoot_ab == home_ab else home_ab
        shooter = x["shooter_player_id"]
        os_ = shifts_by_team.get(shoot_ab); od_ = shifts_by_team.get(def_ab)
        # shifts_by_team stores (starts, ends, pids); zip yields (start, end, pid).
        # Asymmetric (a, b] + dedup — see NFI/scripts/03_onice_attribution_
        # pillars.py for the full writeup: inclusive end recovers the
        # outgoing player who was truly on for the event; exclusive start
        # keeps out the incoming player whose shift starts at that same
        # instant; dedup guards a real shift + zero-length marker row both
        # matching the same player at once.
        onice_shoot = (list(dict.fromkeys(
                        int(pid) for a, b, pid in zip(*os_)
                        if a < t <= b and pos_map.get(int(pid)) != "G")) if os_ else [])
        onice_def = (list(dict.fromkeys(
                      int(pid) for a, b, pid in zip(*od_)
                      if a < t <= b and pos_map.get(int(pid)) != "G")) if od_ else [])
        if not pd.isna(shooter):
            ind_att[(int(shooter), season)][(state, zone)] += 1
            if is_goal:
                ind_gl[(int(shooter), season)][(state, zone)] += 1
        for p in onice_shoot:
            for_att[(p, season)][(state, zone)] += 1
            if is_fen:
                for_fen[(p, season)][(state, zone)] += 1
            if is_goal:
                for_gl[(p, season)][(state, zone)] += 1
        for p in onice_def:
            ag_att[(p, season)][(state, zone)] += 1
            if is_fen:
                ag_fen[(p, season)][(state, zone)] += 1
            if is_goal:
                ag_gl[(p, season)][(state, zone)] += 1

# Pooled "all_playoffs": sum each player's counts/toi across playoff seasons.
def pool(d):
    out = defaultdict(lambda: defaultdict(float))
    for (pid, _s), inner in d.items():
        for k, v in inner.items():
            out[(pid, "all_playoffs")][k] += v
    return out

seasons_present = sorted({s for (_p, s) in toi.keys()})
all_keys = set(toi) | set(for_att) | set(ag_att) | set(ind_att)
all_keys |= set(pool(toi)) | set(pool(for_att)) | set(pool(ag_att)) | set(pool(ind_att))
pooled = {"ind_att": pool(ind_att), "ind_gl": pool(ind_gl), "for_att": pool(for_att),
          "for_gl": pool(for_gl), "ag_att": pool(ag_att), "ag_gl": pool(ag_gl),
          "for_fen": pool(for_fen), "ag_fen": pool(ag_fen), "toi": pool(toi)}

def get(name, key, sz):
    src = {"ind_att": ind_att, "ind_gl": ind_gl, "for_att": for_att, "for_gl": for_gl,
           "ag_att": ag_att, "ag_gl": ag_gl, "for_fen": for_fen, "ag_fen": ag_fen, "toi": toi}
    if sz == "all_playoffs":
        return pooled[name].get(key, {})
    return src[name].get(key, {})

rows = []
keyset = set(toi) | set(for_att) | set(ag_att) | set(ind_att)
keyset |= {(p, "all_playoffs") for (p, _s) in keyset}
for (pid, sz) in sorted(keyset, key=lambda k: (str(k[1]), k[0])):
    to = get("toi", (pid, sz), sz)
    for state in STATES:
        mins = to.get(state, 0) / 60.0
        for zone in ZONES_ALL:
            rows.append({
                "player_id": pid, "season": sz, "position": pos_map.get(pid, ""),
                "state": state, "zone": zone, "toi_min": round(mins, 3),
                "ind_att": get("ind_att", (pid, sz), sz).get((state, zone), 0),
                "ind_gl": get("ind_gl", (pid, sz), sz).get((state, zone), 0),
                "onice_for_att": get("for_att", (pid, sz), sz).get((state, zone), 0),
                "onice_for_gl": get("for_gl", (pid, sz), sz).get((state, zone), 0),
                "onice_ag_att": get("ag_att", (pid, sz), sz).get((state, zone), 0),
                "onice_ag_gl": get("ag_gl", (pid, sz), sz).get((state, zone), 0),
                # Fenwick (no-blocks) siblings — appended last so existing columns
                # stay byte-identical; the file just gains these two trailing cols.
                "onice_for_fen": get("for_fen", (pid, sz), sz).get((state, zone), 0),
                "onice_ag_fen": get("ag_fen", (pid, sz), sz).get((state, zone), 0),
            })
out = pd.DataFrame(rows)
out.to_csv(OUT_FP, index=False)
print(f"\nWrote {OUT_FP} — {len(out)} rows, "
      f"{out['player_id'].nunique()} players, seasons {sorted(out['season'].unique())}")
# sanity: McDavid all_playoffs ES CNFI+MNFI A/S per 60
mc = out[(out.player_id == 8478402) & (out.season == "all_playoffs") & (out.state == "ES")
         & (out.zone.isin(["CNFI", "MNFI"]))]
if len(mc):
    es_toi = mc["toi_min"].iloc[0]
    a = mc["onice_for_att"].sum(); s = mc["onice_ag_att"].sum()
    print(f"  McDavid all-playoffs ES: toi_min={es_toi:.1f}  "
          f"NFI-A/60={a/es_toi*60:.1f}  NFI-S/60={s/es_toi*60:.1f}")
