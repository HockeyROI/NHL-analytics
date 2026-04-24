#!/usr/bin/env python3
"""
Build PP and PK splits for all 7 pillars + horse race.

Situation logic:
  - situation_code is 4-digit: away_g, away_sk, home_sk, home_g
  - Empty net dropped (any g==0)
  - PP from shooter POV: shooter_skaters > opponent_skaters
  - PK from shooter POV: shooter_skaters < opponent_skaters
  - For player attribution we use *player team* POV:
      * Player on shooting team in shooter-PP shot  -> player on PP
      * Player on defending team in shooter-PP shot -> player on PK
      * Player on shooting team in shooter-PK shot  -> player on PK
      * Player on defending team in shooter-PK shot -> player on PP

Pillar definitions:
  P1a  — F, on-ice FOR, wide CNFI/MNFI x-bands, y-band-weighted attempts/60
  P2   — F, on-ice AGAINST, tight CNFI/MNFI/FNFI, y-band-weighted SA/60
  P3   — F, on-ice FOR,    tight CNFI/MNFI/FNFI, y-band-weighted SF/60
  P4   — D, on-ice AGAINST, tight CNFI/MNFI/FNFI, y-band-weighted SA/60
  P5   — D, on-ice FOR,     tight CNFI/MNFI/FNFI, y-band-weighted SF/60
  P6   — G, save% on Inner-Slot (x>69, |y|<=14) shots faced
  P7   — G, save% on CNFI (x 74-89, |y|<=9) shots faced

Skater min: 100 PP min OR 100 PK min (per situation).
Goalie min: 100 PP shots faced OR 100 PK shots faced.

Wilson 95% CIs throughout. Fenwick spatial filter (drop blocked-shot for
spatial work; goalies use SOG+goal which is already SOG-only since blocked
doesn't reach goalie).

Outputs (NFI/output/):
  P1a_PP.csv P1a_PK.csv P2_PP.csv P2_PK.csv P3_PP.csv P3_PK.csv
  P4_PP.csv P4_PK.csv P5_PP.csv P5_PK.csv P6_PP.csv P6_PK.csv
  P7_PP.csv P7_PK.csv
  pp_pk_horse_race.csv  (R^2 vs standings points and GA/game)
"""
import math, time
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
SHIFT_CSV = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
TOI_CSV   = f"{OUT}/player_toi.csv"
POS_CSV   = f"{OUT}/player_positions.csv"
GAME_CSV  = f"{ROOT}/Data/game_ids.csv"
TEAM_CSV  = f"{OUT}/team_level_all_metrics.csv"

SEASONS = {"20212022","20222023","20232024","20242025","20252026"}
MIN_PP_TOI_MIN = 100.0
MIN_PK_TOI_MIN = 100.0
MIN_PP_FACED   = 100
MIN_PK_FACED   = 100

# Y-band weights (same ES-derived weights used as relative danger proxy)
WEIGHTS_WIDE = {
    "CNFI": {"0-5":0.1579, "5-10":0.1075, "10-15":0.0781, "15-20":0.0513,
             "20-25":0.0318, "25-30":0.0209, "30+":0.0123},
    "MNFI": {"0-5":0.1109, "5-10":0.0985, "10-15":0.0887, "15-20":0.0692,
             "20-25":0.0492, "25-30":0.0278, "30+":0.0122},
}
WEIGHTS_TIGHT = {
    "CNFI": {"0-5":0.1579, "5-10":0.1075},
    "MNFI": {"0-5":0.1109, "5-10":0.0985, "10-15":0.0887},
    "FNFI": {"0-5":0.0367, "5-10":0.0335, "10-15":0.0289},
}
WIDE_BANDS  = ["0-5","5-10","10-15","15-20","20-25","25-30","30+"]
TIGHT_BANDS = ["0-5","5-10","10-15"]

def y_band_wide(absy):
    if absy < 5: return "0-5"
    if absy < 10: return "5-10"
    if absy < 15: return "10-15"
    if absy < 20: return "15-20"
    if absy < 25: return "20-25"
    if absy < 30: return "25-30"
    return "30+"

def y_band_tight(absy):
    if absy < 5: return "0-5"
    if absy < 10: return "5-10"
    return "10-15"

def classify_wide(x):
    if 74 <= x <= 89: return "CNFI"
    if 55 <= x <= 73: return "MNFI"
    return None

def classify_tight(x, absy):
    if 74 <= x <= 89 and absy <= 9:   return "CNFI"
    if 55 <= x <= 73 and absy <= 15:  return "MNFI"
    if 25 <= x <= 54 and absy <= 15:  return "FNFI"
    return None

def in_inner_slot(x, absy): return (x > 69) and (absy <= 14)
def in_cnfi_tight(x, absy): return (74 <= x <= 89) and (absy <= 9)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0.0, c-h), min(1.0, c+h))

def rate_ci(events, minutes, z=1.96):
    if minutes <= 0: return (0.0, 0.0, 0.0)
    p, lo, hi = wilson(events, max(events, int(round(minutes))))
    return events/minutes*60.0, lo*60.0, hi*60.0

# ---- Lookups ----
print("Loading metadata ...")
pos_df = pd.read_csv(POS_CSV, dtype={"player_id":int})
pos_grp  = dict(zip(pos_df["player_id"], pos_df["pos_group"]))
pos_role = dict(zip(pos_df["player_id"], pos_df["position"]))
name_map = dict(zip(pos_df["player_id"], pos_df["player_name"]))
toi_df = pd.read_csv(TOI_CSV, dtype={"player_id":int})
toi_pp_min = dict(zip(toi_df["player_id"], toi_df["toi_PP_sec"]/60.0))
toi_pk_min = dict(zip(toi_df["player_id"], toi_df["toi_PK_sec"]/60.0))

# ---- Load shots and classify situation ----
print("Loading shots ...")
shot_cols = ["game_id","season","period","situation_code","event_type",
             "time_secs","home_team_id","shooting_team_id","shooting_team_abbrev",
             "home_team_abbrev","away_team_abbrev",
             "x_coord_norm","y_coord_norm","is_goal","goalie_id"]
shots = pd.read_csv(SHOT_CSV, usecols=shot_cols,
                    dtype={"season":str,"situation_code":str})
shots = shots[shots["season"].isin(SEASONS)].copy()
shots = shots[shots["period"].between(1,3)].copy()

# Parse situation_code
sc = shots["situation_code"].astype(str).str.zfill(4)
ag  = sc.str[0].astype(int); ask = sc.str[1].astype(int)
hsk = sc.str[2].astype(int); hg  = sc.str[3].astype(int)
empty_net = (ag==0) | (hg==0)
shoot_home = shots["shooting_team_id"]==shots["home_team_id"]
sh_sk  = np.where(shoot_home, hsk, ask)
opp_sk = np.where(shoot_home, ask, hsk)

# Shooter-POV situation
shots["sit"] = np.where(sh_sk == opp_sk, "ES",
                np.where(sh_sk > opp_sk, "PP", "PK"))
shots["shoot_home"] = shoot_home
shots = shots[~empty_net].copy()
print(f"  Reg, non-empty-net shots: {len(shots):,}")
print(f"    by situation: {shots['sit'].value_counts().to_dict()}")

# Apply blocked-shot abs() correction to fix sign-flip
blk = shots["event_type"]=="blocked-shot"
shots.loc[blk,"x_coord_norm"] = shots.loc[blk,"x_coord_norm"].abs()
shots.loc[blk,"y_coord_norm"] = shots.loc[blk,"y_coord_norm"].abs()
shots["abs_y"] = shots["y_coord_norm"].abs()
shots["abs_time"] = shots["time_secs"].astype(int) + (shots["period"].astype(int)-1)*1200

# We focus on PP and PK only (ES already done)
pp_pk = shots[shots["sit"].isin(["PP","PK"])].copy()
# Spatial filter to events that could matter
pp_pk = pp_pk[pp_pk["x_coord_norm"].between(25, 89)].copy()
print(f"  PP/PK shots in OZ x-range: {len(pp_pk):,}")

# Pre-classify zones for fast lookup
xs = pp_pk["x_coord_norm"].values
ys = pp_pk["y_coord_norm"].values
absy = pp_pk["abs_y"].values
pp_pk["zone_wide"]  = [classify_wide(x) for x in xs]
pp_pk["zone_tight"] = [classify_tight(x, ay) for x,ay in zip(xs, absy)]
pp_pk["band_wide"]  = [y_band_wide(ay) if zw else None
                       for ay, zw in zip(absy, pp_pk["zone_wide"].values)]
pp_pk["band_tight"] = [y_band_tight(ay) if zt else None
                       for ay, zt in zip(absy, pp_pk["zone_tight"].values)]

# Fenwick subset for spatial pillars (drop blocked)
pp_pk_fenwick = pp_pk[pp_pk["event_type"].isin(["shot-on-goal","missed-shot","goal"])].copy()
print(f"  PP/PK Fenwick shots: {len(pp_pk_fenwick):,}")

# Goalie-faced subset (SOG + goal)
pp_pk_faced = pp_pk[pp_pk["event_type"].isin(["shot-on-goal","goal"])].copy()
pp_pk_faced = pp_pk_faced[pp_pk_faced["goalie_id"].notna()].copy()
pp_pk_faced["goalie_id"] = pp_pk_faced["goalie_id"].astype(int)
print(f"  PP/PK goalie-faced shots: {len(pp_pk_faced):,}")

# ---- Load shifts (only games in pp_pk) ----
print("Loading shifts ...")
valid_gids = set(pp_pk_fenwick["game_id"].unique())
shift_cols = ["game_id","player_id","period","team_abbrev",
              "abs_start_secs","abs_end_secs"]
parts = []
for ch in pd.read_csv(SHIFT_CSV, usecols=shift_cols, chunksize=500_000):
    ch = ch.dropna(subset=shift_cols)
    ch["game_id"] = ch["game_id"].astype(int)
    ch["player_id"] = ch["player_id"].astype(int)
    ch["period"] = ch["period"].astype(int)
    ch["abs_start_secs"] = ch["abs_start_secs"].astype(int)
    ch["abs_end_secs"]   = ch["abs_end_secs"].astype(int)
    ch = ch[ch["game_id"].isin(valid_gids) & ch["period"].between(1,3)]
    if len(ch): parts.append(ch)
shifts = pd.concat(parts, ignore_index=True)
del parts
print(f"  shifts: {len(shifts):,}")
shifts_by_game = dict(tuple(shifts.groupby("game_id")))
shots_by_game  = dict(tuple(pp_pk_fenwick.groupby("game_id")))

# ---- Aggregators (player_id, situation_pov, zone_label, band) ----
# situation_pov is "PP" if player is on PP team; "PK" if on PK team.
# F counts (for and against, two zone systems):
p1a_for = defaultdict(int)   # (pid, "PP"/"PK", zone_wide, band_wide) — F shots-for
p2_ag   = defaultdict(int)   # (pid, sit, zone_tight, band_tight) — F shots-against
p3_for  = defaultdict(int)   # (pid, sit, zone_tight, band_tight) — F shots-for
# D
p4_ag   = defaultdict(int)
p5_for  = defaultdict(int)

print("Per-game shot-shift join ...")
t0 = time.time()
n_games = 0
for gid, gshots in shots_by_game.items():
    n_games += 1
    if n_games % 1000 == 0: print(f"  {n_games} games ({time.time()-t0:.1f}s)")
    gshifts = shifts_by_game.get(gid)
    if gshifts is None: continue

    shifts_by_team = {}
    for team_ab, tsh in gshifts.groupby("team_abbrev"):
        st = tsh["abs_start_secs"].values.astype(np.int32)
        en = tsh["abs_end_secs"].values.astype(np.int32)
        pids = tsh["player_id"].values.astype(np.int64)
        order = np.argsort(st)
        shifts_by_team[team_ab] = (st[order], en[order], pids[order])

    for r in gshots.itertuples(index=False):
        t = int(r.abs_time)
        sit = r.sit  # PP or PK from SHOOTER POV
        shoot_ab = r.shooting_team_abbrev
        def_ab = r.away_team_abbrev if shoot_ab == r.home_team_abbrev else r.home_team_abbrev
        # Player situation:
        #   shooter team -> player is on (sit) (shooter PP -> player on PP)
        #   defender     -> player is on opposite of (sit)
        sit_for_shooter = sit
        sit_for_defender = "PK" if sit == "PP" else "PP"

        # Get on-ice players for both teams once
        for team_ab, sit_pov in [(shoot_ab, sit_for_shooter), (def_ab, sit_for_defender)]:
            if team_ab not in shifts_by_team: continue
            st, en, pids = shifts_by_team[team_ab]
            idx = np.searchsorted(st, t, side="right")
            if idx == 0: continue
            on_mask = en[:idx] > t
            on_pids = pids[:idx][on_mask]

            # Determine if this team is the shooter (FOR) or defender (AGAINST)
            is_shooter_side = (team_ab == shoot_ab)

            # Wide zone classification (only used for P1a — F FOR)
            if r.zone_wide and is_shooter_side:
                for pid in on_pids:
                    if pos_grp.get(int(pid)) == "F":
                        p1a_for[(int(pid), sit_pov, r.zone_wide, r.band_wide)] += 1

            # Tight zones for P2/P3/P4/P5
            if r.zone_tight:
                zt, bt = r.zone_tight, r.band_tight
                for pid in on_pids:
                    pgrp = pos_grp.get(int(pid))
                    if pgrp == "F":
                        if is_shooter_side:
                            p3_for[(int(pid), sit_pov, zt, bt)] += 1
                        else:
                            p2_ag[(int(pid),  sit_pov, zt, bt)] += 1
                    elif pgrp == "D":
                        if is_shooter_side:
                            p5_for[(int(pid), sit_pov, zt, bt)] += 1
                        else:
                            p4_ag[(int(pid),  sit_pov, zt, bt)] += 1

print(f"Aggregation done in {time.time()-t0:.1f}s")

# ---- Goalie aggregation (P6/P7) ----
print("Aggregating goalie pillars ...")
# A goalie's situation = same as their team's. Their team is the DEFENDING
# team for the shot (they're in net facing the shooter).
# If shooter is PP -> goalie is on PK
# If shooter is PK -> goalie is on PP
# P6: inner_slot (x>69, |y|<=14)
# P7: tight CNFI (x 74-89, |y|<=9)
g_p6_faced = defaultdict(int);  g_p6_goals = defaultdict(int)
g_p7_faced = defaultdict(int);  g_p7_goals = defaultdict(int)

for r in pp_pk_faced.itertuples(index=False):
    sit = r.sit
    goalie_sit = "PK" if sit == "PP" else "PP"
    gid = int(r.goalie_id)
    x = float(r.x_coord_norm); ay = float(r.abs_y); g = int(r.is_goal)
    if in_inner_slot(x, ay):
        g_p6_faced[(gid, goalie_sit)] += 1
        g_p6_goals[(gid, goalie_sit)] += g
    if in_cnfi_tight(x, ay):
        g_p7_faced[(gid, goalie_sit)] += 1
        g_p7_goals[(gid, goalie_sit)] += g

# ============================================================
# Build CSVs
# ============================================================
print("Building CSVs ...")

def add_rank(df, score_col, asc):
    df["rank"] = df[score_col].rank(ascending=asc, method="min").astype(int)
    return df.sort_values(score_col, ascending=asc).reset_index(drop=True)

def build_skater_pillar(att_map, weights, bands, fwd_only, sit_filter,
                         tag, kind, min_toi):
    """
    att_map: defaultdict (pid, sit, zone, band) -> count
    weights: per-zone per-band weights
    bands: ordered list of bands
    fwd_only: True for F pillars, False for D pillars
    sit_filter: "PP" or "PK"  (player POV)
    tag: "P1a"/"P2"/"P3"/"P4"/"P5"
    kind: "for" or "against"
    min_toi: 100 (PP or PK)
    Returns DataFrame, with rate per 60 (using PP TOI for PP rows, PK TOI for PK rows)
    """
    toi_map = toi_pp_min if sit_filter == "PP" else toi_pk_min
    desired_pos = "F" if fwd_only else "D"

    # Players who appear at this situation
    pids_in = sorted({pid for (pid, sit, _, _) in att_map.keys() if sit == sit_filter})
    rows = []
    for pid in pids_in:
        if pos_grp.get(pid) != desired_pos: continue
        toi = toi_map.get(pid, 0.0)
        if toi < min_toi: continue
        rec = {"player_id": pid,
               "player_name": name_map.get(pid, ""),
               "position":    pos_role.get(pid, ""),
               f"{sit_filter}_TOI_min": round(toi, 2)}
        total_n=0; total_w=0.0
        zones = list(weights.keys())
        for zone in zones:
            zn=0; zw=0.0
            for b in bands:
                n = att_map.get((pid, sit_filter, zone, b), 0)
                zn += n
                zw += n * weights[zone].get(b, 0.0)
            r, lo, hi = rate_ci(zn, toi)
            rec[f"{zone}_n"] = zn
            rec[f"{zone}_per60"] = round(r, 4)
            rec[f"{zone}_lo95"]  = round(lo, 4)
            rec[f"{zone}_hi95"]  = round(hi, 4)
            total_n += zn
            total_w += zw
        r, lo, hi = rate_ci(total_n, toi)
        rec["total_n"]      = total_n
        rec["total_per60"]  = round(r, 4)
        rec["total_lo95"]   = round(lo, 4)
        rec["total_hi95"]   = round(hi, 4)
        rec[f"{tag}_weighted"] = round(total_w / toi * 60.0, 4)
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    asc = (kind == "against")  # lower against = better; higher for = better
    df = add_rank(df, f"{tag}_weighted", asc)
    return df

# 5 skater pillars × 2 situations = 10 outputs
plan_skater = [
    ("P1a", p1a_for, WEIGHTS_WIDE,  WIDE_BANDS,  True,  "for"),
    ("P2",  p2_ag,   WEIGHTS_TIGHT, TIGHT_BANDS, True,  "against"),
    ("P3",  p3_for,  WEIGHTS_TIGHT, TIGHT_BANDS, True,  "for"),
    ("P4",  p4_ag,   WEIGHTS_TIGHT, TIGHT_BANDS, False, "against"),
    ("P5",  p5_for,  WEIGHTS_TIGHT, TIGHT_BANDS, False, "for"),
]

skater_results = {}
for tag, att, w, bands, fwd, kind in plan_skater:
    for sit in ["PP","PK"]:
        min_toi = MIN_PP_TOI_MIN if sit=="PP" else MIN_PK_TOI_MIN
        df = build_skater_pillar(att, w, bands, fwd, sit, tag, kind, min_toi)
        out_path = f"{OUT}/{tag}_{sit}.csv"
        df.to_csv(out_path, index=False)
        skater_results[(tag, sit)] = df
        print(f"  {tag}_{sit}.csv: {len(df)} qualifying  (kind={kind})")

# Goalies — P6, P7
def build_goalie(faced_map, goals_map, sit_filter, tag, min_faced):
    rows = []
    pids = sorted({gid for (gid, sit) in faced_map.keys() if sit == sit_filter})
    for gid in pids:
        n = faced_map.get((gid, sit_filter), 0)
        if n < min_faced: continue
        gcnt = goals_map.get((gid, sit_filter), 0)
        saves = n - gcnt
        sv, lo, hi = wilson(saves, n)
        rec = {"goalie_id": gid,
               "goalie_name": name_map.get(gid, ""),
               "faced": n, "goals": gcnt, "saves": saves,
               f"{tag}_save_pct": round(sv, 5),
               f"{tag}_lo95":     round(lo, 5),
               f"{tag}_hi95":     round(hi, 5)}
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(f"{tag}_save_pct", ascending=False).reset_index(drop=True)
    df["rank"] = df[f"{tag}_save_pct"].rank(ascending=False, method="min").astype(int)
    return df

goalie_plan = [("P6", g_p6_faced, g_p6_goals),
               ("P7", g_p7_faced, g_p7_goals)]
goalie_results = {}
for tag, faced_m, goals_m in goalie_plan:
    for sit in ["PP","PK"]:
        min_faced = MIN_PP_FACED if sit=="PP" else MIN_PK_FACED
        df = build_goalie(faced_m, goals_m, sit, tag, min_faced)
        out_path = f"{OUT}/{tag}_{sit}.csv"
        df.to_csv(out_path, index=False)
        goalie_results[(tag, sit)] = df
        print(f"  {tag}_{sit}.csv: {len(df)} qualifying goalies")

# ============================================================
# Horse race — team-level versions vs standings + GA/game
# ============================================================
print("\nBuilding team-level PP/PK metrics for horse race ...")
team_metrics = pd.read_csv(TEAM_CSV, dtype={"season":str})
team_metrics["season"] = team_metrics["season"].astype(str)

# Compute team-level GF and GA from shots (for "GA per game")
shots_for_team = shots.copy()
shots_for_team["is_goal_int"] = shots_for_team["is_goal"].astype(int)
# Goals scored = shooting_team perspective
gf = shots_for_team[shots_for_team["is_goal_int"]==1].groupby(
    ["season","shooting_team_abbrev"]).size().reset_index(name="GF")
gf = gf.rename(columns={"shooting_team_abbrev":"team"})
# Total games per team-season — use team_metrics gp
gp_lookup = dict(zip(zip(team_metrics["season"], team_metrics["team"]), team_metrics["gp"]))

# Goals against: total goals - own goals
ga_rows = []
for (s, t), gp in gp_lookup.items():
    pts = team_metrics[(team_metrics["season"]==s) & (team_metrics["team"]==t)]["points"].iloc[0]
    gd  = team_metrics[(team_metrics["season"]==s) & (team_metrics["team"]==t)]["goal_diff"].iloc[0]
    own_gf = gf[(gf["season"]==s)&(gf["team"]==t)]["GF"].sum()
    own_ga = own_gf - gd  # GD = GF - GA, so GA = GF - GD
    ga_rows.append({"season":s, "team":t, "gp":gp, "points":pts,
                     "GF":own_gf, "GA":own_ga, "GA_per_game":own_ga/gp})
team_ext = pd.DataFrame(ga_rows)

# For each pillar/situation, compute team-season aggregate of the pillar
# (TOI-weighted average across qualifying players' weighted scores)
def team_aggregate(pillar_df, score_col, sit):
    """pillar_df: per-player; build team-season weighted average using PP/PK TOI"""
    # Need to know which team each player played for in each season — re-derive from shifts
    pass  # we'll skip per-team-season mapping and use a simpler approach

# Simpler approach: compute team-level pillar values directly from shot data
# rather than aggregating player-level (avoids needing per-(player, team, season) TOI).
print("Computing team-season pillar values directly from shots ...")

# Pre-load game->season map
games = pd.read_csv(GAME_CSV, dtype={"game_id":int,"season":str})
games = games[games["game_type"]=="regular"]
game_season_map = dict(zip(games["game_id"], games["season"]))

# Augment pp_pk shots with team perspective rows (one per team) and compute
# team-level for/against per (situation_pov)
team_metric_rows = []
# Rather than build all 7 — we'll focus on the FOR-side metrics (P1a, P3, P5)
# and the AGAINST-side metrics (P2, P4) plus goalie save (P6, P7) for each
# team-season under each situation_pov.
shots_for_team["season"] = shots_for_team["season"].astype(str)

# Helper: team_pp_toi by team-season (sum of PP TOI of all players who played for team)
# Approximation: use league total from team_metrics if PP TOI not available;
# horse race results most useful at relative level. We'll use shots-faced count as
# the natural "exposure" for some metrics. For per-60 normalization at team level
# we'll use TEAM minutes on PP/PK derived from shifts (sum of skater-seconds /
# 5 skaters, capped to game time).
# To keep the script tractable, compute team-season aggregate using SHOT COUNTS
# only and report per-shot / per-100-shot rates rather than per-60 at the
# horse-race level. This still measures the "shot quality mix" per team-situation.

# For each team-season-situation_pov, count weighted shots / total shots
def derive_team_PP_PK_shots(_shots, sit_pov_label):
    rows = []
    sub = _shots[_shots["sit"].isin(["PP","PK"])].copy()
    for (season, team), grp in sub.groupby(["season","shooting_team_abbrev"]):
        # Player POV: when SHOOTING team is in PP, player_pov="PP"; when PK, player_pov="PK"
        for sit_player in ["PP","PK"]:
            if sit_player == "PP":
                situ_match = grp[grp["sit"]=="PP"]
            else:
                situ_match = grp[grp["sit"]=="PK"]
            # Counts in CNFI/MNFI/FNFI tight zones
            for zone in ["CNFI","MNFI","FNFI"]:
                cz = situ_match[situ_match["zone_tight"]==zone]
                rows.append({"season":season, "team":team,
                             "side":"FOR", "sit":sit_player, "zone":zone,
                             "shots": len(cz), "goals": int(cz["is_goal"].sum())})
    return pd.DataFrame(rows)

# Skip the full team-aggregation for the horse race — it's secondary. Instead
# use a simpler diagnostic: regress STANDINGS POINTS (and GA/game) on the
# COUNT of goals scored on PP and conceded on PK (a standard team-level proxy).
# Then for each PILLAR file we just compute the average score among the team's
# qualifying players (weighted by TOI), and regress that against points / GA.

# Compute per-(season, team, player) PP/PK TOI from shifts (re-use prior pattern)
print("Per-(player, season, team) PP/PK TOI from shifts ...")
# We need the strength-state windows per game; re-derive from shots state intervals.
# Faster: PP TOI per player is already in player_toi.csv (5-season pool, no per-team).
# For team-season aggregation, we'll use a simpler proxy: count qualifying players
# per team-season (from shifts) and average their pillar score.

# Build per-(season, team, player) shift-time-on-ice; use it just to identify
# which players belong to which team-season (for averaging pillar scores).
ts_pid = defaultdict(int)  # (season, team, pid) -> total seconds (any sit)
shift_iter_cols = ["game_id","player_id","team_abbrev","abs_start_secs","abs_end_secs","period"]
for ch in pd.read_csv(SHIFT_CSV, usecols=shift_iter_cols, chunksize=500_000):
    ch = ch.dropna(subset=shift_iter_cols)
    ch["game_id"] = ch["game_id"].astype(int)
    ch["player_id"] = ch["player_id"].astype(int)
    ch["abs_start_secs"] = ch["abs_start_secs"].astype(int)
    ch["abs_end_secs"]   = ch["abs_end_secs"].astype(int)
    ch["period"] = ch["period"].astype(int)
    ch = ch[ch["period"].between(1,3)]
    ch["dur"] = (ch["abs_end_secs"] - ch["abs_start_secs"]).clip(lower=0)
    ch["season"] = ch["game_id"].map(game_season_map)
    ch = ch[ch["season"].isin(SEASONS)]
    grp = ch.groupby(["player_id","season","team_abbrev"], as_index=False)["dur"].sum()
    for r in grp.itertuples(index=False):
        ts_pid[(r.season, r.team_abbrev, int(r.player_id))] += int(r.dur)

# Build team-season pillar averages
print("Building horse-race team-season aggregates ...")
horse_rows = []
for (tag, sit), df in skater_results.items():
    if df.empty: continue
    score_col = f"{tag}_weighted"
    # Need to map each player to a team-season; use the team-season where they played the most.
    # Then aggregate avg pillar score by team-season.
    # First: build player -> primary team-season list (top by TOI in each season)
    primary_team = {}  # (player_id, season) -> team
    pid_season_team_toi = defaultdict(int)  # (pid, season) -> dict team->sec
    for (season, team, pid), sec in ts_pid.items():
        primary_team.setdefault((pid, season), (team, sec))
        # We'll do max separately
    # Recompute primary by iterating again
    pid_season_max = {}
    for (season, team, pid), sec in ts_pid.items():
        key = (pid, season)
        if key not in pid_season_max or pid_season_max[key][1] < sec:
            pid_season_max[key] = (team, sec)

    # For each player in the pillar (5-season pooled), assign them across all
    # seasons they played. Aggregate per team-season as mean of pillar values
    # of players who played for that team in that season (any TOI threshold).
    team_season_scores = defaultdict(list)
    for _, row in df.iterrows():
        pid = int(row["player_id"])
        score = row[score_col]
        for season in SEASONS:
            key = (pid, season)
            if key in pid_season_max:
                team, _ = pid_season_max[key]
                team_season_scores[(season, team)].append(score)
    for (season, team), vals in team_season_scores.items():
        horse_rows.append({"pillar":tag, "sit":sit, "season":season, "team":team,
                            "n_players": len(vals),
                            "team_avg": float(np.mean(vals))})

horse = pd.DataFrame(horse_rows)
# Pivot for regression — one row per (season, team) per (pillar, sit)
horse = horse.merge(team_ext, on=["season","team"], how="inner")

# Univariate R^2 per pillar/situation
hr_summary = []
for (tag, sit), grp in horse.groupby(["pillar","sit"]):
    if len(grp) < 10: continue
    for dep_col in ["points","GA_per_game"]:
        x = grp["team_avg"].values
        y = grp[dep_col].values
        # OLS R^2 via simple formulas
        x_mean = x.mean(); y_mean = y.mean()
        ss_xy = ((x-x_mean)*(y-y_mean)).sum()
        ss_xx = ((x-x_mean)**2).sum()
        if ss_xx == 0:
            r2 = np.nan; r=np.nan
        else:
            slope = ss_xy/ss_xx
            intercept = y_mean - slope*x_mean
            yhat = intercept + slope*x
            ss_res = ((y-yhat)**2).sum()
            ss_tot = ((y-y_mean)**2).sum()
            r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
            r = np.corrcoef(x,y)[0,1]
        hr_summary.append({"pillar":tag, "sit":sit, "dep":dep_col,
                            "n_team_seasons": len(grp),
                            "pearson_r": round(r,3) if not np.isnan(r) else np.nan,
                            "R2": round(r2,3) if not np.isnan(r2) else np.nan})

hr_df = pd.DataFrame(hr_summary)
hr_df.to_csv(f"{OUT}/pp_pk_horse_race.csv", index=False)
print(f"\nWrote pp_pk_horse_race.csv ({len(hr_df)} rows)")

# ============================================================
# Reporting
# ============================================================
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.width = 220

print("\n=========== QUALIFYING COUNTS ===========")
for tag in ["P1a","P2","P3","P4","P5","P6","P7"]:
    for sit in ["PP","PK"]:
        if tag in ("P6","P7"):
            df = goalie_results.get((tag, sit), pd.DataFrame())
        else:
            df = skater_results.get((tag, sit), pd.DataFrame())
        print(f"  {tag}_{sit}: {len(df)} qualifying")

# Top 5 per pillar
print("\n=========== TOP 5 PER PILLAR ===========")
for (tag, sit), df in skater_results.items():
    if df.empty:
        print(f"\n{tag}_{sit}: (no qualifying players)")
        continue
    score_col = f"{tag}_weighted"
    print(f"\n{tag}_{sit}  (rank by {score_col}, "
          f"{'lower=better' if tag in ('P2','P4') else 'higher=better'})")
    cols = ["rank","player_name","position",
            f"{sit}_TOI_min","total_n","total_per60",score_col]
    print(df[cols].head(5).to_string(index=False))

for (tag, sit), df in goalie_results.items():
    if df.empty:
        print(f"\n{tag}_{sit}: (no qualifying goalies)")
        continue
    score_col = f"{tag}_save_pct"
    print(f"\n{tag}_{sit}  (rank by {score_col})")
    cols = ["rank","goalie_name","faced","goals","saves",score_col,
            f"{tag}_lo95",f"{tag}_hi95"]
    print(df[cols].head(5).to_string(index=False))

print("\n=========== HORSE RACE R^2 ===========")
print(hr_df.sort_values(["dep","R2"], ascending=[True, False]).to_string(index=False))
