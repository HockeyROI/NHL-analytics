#!/usr/bin/env python3
"""
Step 4 - On-ice attribution (shift <-> shot join).
Step 5 - Compute Pillars 1-7. Wilson 95% CIs on proportions (save%);
        exact Poisson 95% CIs on per-60 rates (chi-square / Garwood).

Seasons: 20222023, 20232024, 20242025 (3 pooled).
Regulation periods only (1-3).
Drop empty net situations.
Min 500 minutes TOI per player (regulation total).

Outputs:
  shots_tagged.csv (per-shot with zone, state, abs_time, on-ice player IDs)  [intermediate, large]
  player_toi.csv  (player_id -> TOI total and per-state)
  pillar_1_netfront_F.csv
  pillar_2_defensive_F.csv
  pillar_3_offensive_F.csv
  pillar_4_defensive_D.csv
  pillar_5_offensive_D.csv
  pillar_6_goalie_FNFI_MNFI.csv
  pillar_7_goalie_CNFI.csv
"""
import os, csv, math, json
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.stats import chi2

ROOT = os.environ.get("HOCKEYROI_ROOT", "/Users/ashgarg/Documents/HockeyROI")
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
SHIFT_CSV = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
POS_CSV = f"{ROOT}/NFI/output/player_positions.csv"
GAME_CSV = f"{ROOT}/Data/game_ids.csv"
OUT_DIR = f"{ROOT}/NFI/output"

SEASONS = {"20212022", "20222023", "20232024", "20242025", "20252026"}
INFL1 = 55  # MNFI/FNFI boundary
BLUE  = 25
MIN_TOI_MIN = 500.0  # minutes

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    halfw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (p, max(0.0, center-halfw), min(1.0, center+halfw))

def rate_ci(events, minutes, z=1.96):
    """Per-60 rate with exact Poisson 95% CI on the event count.

    Uses the chi-square (Garwood) exact Poisson interval on the count k, then
    scales to per-60 by dividing by the time exposure (minutes/60). Replaces
    a prior Wilson-on-rate misuse: Wilson is a binomial-proportion interval
    in [0, 1]; per-60 rates are unbounded Poisson rates over time exposure.
    Signature preserved so callers don't change; the z arg is kept for API
    stability but the level is fixed at 95% via the chi-square method.
    """
    if minutes <= 0:
        return (0.0, 0.0, 0.0)
    k = int(round(events))
    rate = events / minutes * 60.0
    if k <= 0:
        hi_count = chi2.ppf(0.975, 2) / 2.0
        return (rate, 0.0, hi_count * 60.0 / minutes)
    lo_count = chi2.ppf(0.025, 2 * k) / 2.0
    hi_count = chi2.ppf(0.975, 2 * (k + 1)) / 2.0
    return (rate, lo_count * 60.0 / minutes, hi_count * 60.0 / minutes)

def classify_zone(x, y):
    if pd.isna(x) or pd.isna(y):
        return "unk"
    if 74 <= x <= 89 and -9 <= y <= 9:
        return "CNFI"
    if -15 <= y <= 15:
        if INFL1 <= x < 74:
            return "MNFI"
        if BLUE <= x < INFL1:
            return "FNFI"
        return "lane_other"
    return "Wide"

def state_from_code(sc, shoot_home):
    """5v5 ('ES') / 4v4 / 3v3 / PP / PK from 4-digit code: away_g, away_sk, home_sk, home_g.
    None if empty net. (Bug fix May 2026: 'ES' previously conflated 5v5 + 4v4 + 3v3.)"""
    if pd.isna(sc):
        return None
    s = str(int(sc)).zfill(4)
    if len(s) != 4:
        return None
    ag, ask, hsk, hg = int(s[0]), int(s[1]), int(s[2]), int(s[3])
    if ag == 0 or hg == 0:
        return None
    sh, op = (hsk, ask) if shoot_home else (ask, hsk)
    if sh == op:
        if sh == 5: return "ES"     # strict 5v5
        if sh == 4: return "4v4"
        if sh == 3: return "3v3"
        return "ES_other"
    if sh > op:  return "PP"
    return "PK"

# ----------------- Load position lookup -----------------
print("Loading positions...")
pos_df = pd.read_csv(POS_CSV, dtype={"player_id":int})
pos_map = dict(zip(pos_df["player_id"], pos_df["pos_group"]))
print(f"  positions: {len(pos_map)}")

# ----------------- Load game -> season map -----------------
g_df = pd.read_csv(GAME_CSV, dtype={"game_id":int,"season":str})
g_df = g_df[g_df["game_type"]=="regular"]
game_season = dict(zip(g_df["game_id"], g_df["season"]))

# ----------------- Load shots -----------------
print("Loading shots...")
cols = ["game_id","season","period","event_id","event_type","situation_code","time_secs",
        "home_team_id","shooting_team_id","home_team_abbrev","away_team_abbrev","shooting_team_abbrev",
        "shooter_player_id","goalie_id","x_coord_norm","y_coord_norm","is_goal"]
shots = pd.read_csv(SHOT_CSV, usecols=cols, dtype={"season":str,"situation_code":str})
shots["season"] = shots["season"].astype(str)
shots = shots[shots["season"].isin(SEASONS) & shots["period"].between(1,3)].copy()
shots = shots[shots["event_type"].isin(["shot-on-goal","missed-shot","blocked-shot","goal"])].copy()

# abs time (time_secs resets per period)
shots["abs_time"] = shots["time_secs"].astype(int) + (shots["period"].astype(int)-1) * 1200

# state derivation
shots["_shoot_home"] = shots["shooting_team_id"] == shots["home_team_id"]
# vectorize state
sc_str = shots["situation_code"].astype(str).str.zfill(4)
ag = sc_str.str[0].astype(int); ask = sc_str.str[1].astype(int)
hsk = sc_str.str[2].astype(int); hg = sc_str.str[3].astype(int)
empty_net = (ag==0) | (hg==0)
sh = np.where(shots["_shoot_home"], hsk, ask)
op = np.where(shots["_shoot_home"], ask, hsk)
# Bug fix May 2026: "ES" now means strict 5v5; 4v4 and 3v3 are separate labels.
state = np.where((sh==op) & (sh==5), "ES",
         np.where((sh==op) & (sh==4), "4v4",
         np.where((sh==op) & (sh==3), "3v3",
         np.where(sh>op, "PP", "PK"))))
shots["state"] = state
shots = shots[~empty_net].copy()

# zone
shots["zone"] = [classify_zone(x,y) for x,y in zip(shots["x_coord_norm"].values, shots["y_coord_norm"].values)]

# Unblocked / attempt classification (all 4 event types = Corsi attempts)
shots["is_attempt"] = 1
shots["is_goal_i"] = shots["is_goal"].astype(int)

# index by game
print(f"  tagged shots: {len(shots):,}")
shots_by_game = dict(tuple(shots.groupby("game_id")))
valid_gids = set(shots_by_game.keys())
print(f"  games with shots: {len(valid_gids)}")

# ----------------- Load shifts (filter to needed games) -----------------
print("Loading shifts (streaming filter)...")
shift_cols = ["game_id","player_id","period","team_abbrev","abs_start_secs","abs_end_secs"]
shift_iter = pd.read_csv(SHIFT_CSV, usecols=shift_cols, chunksize=500000)
shift_list = []
for i, ch in enumerate(shift_iter):
    ch = ch.dropna(subset=["game_id","player_id","period","abs_start_secs","abs_end_secs"])
    ch["game_id"] = ch["game_id"].astype(int)
    ch["player_id"] = ch["player_id"].astype(int)
    ch["period"] = ch["period"].astype(int)
    ch["abs_start_secs"] = ch["abs_start_secs"].astype(int)
    ch["abs_end_secs"] = ch["abs_end_secs"].astype(int)
    ch = ch[ch["game_id"].isin(valid_gids) & ch["period"].between(1,3)]
    if len(ch):
        shift_list.append(ch)
shifts = pd.concat(shift_list, ignore_index=True)
del shift_list
print(f"  shifts loaded: {len(shifts):,}")

shifts_by_game = dict(tuple(shifts.groupby("game_id")))

# ----------------- Team abbrev per game -----------------
# We'll get from shot data: home/away abbrevs
team_abbrevs = shots.groupby("game_id").agg(home_abbrev=("home_team_abbrev","first"),
                                             away_abbrev=("away_team_abbrev","first")).to_dict(orient="index")

# ----------------- Per-game processing -----------------
# Build state intervals from shot events; 1st interval starts at 0 = ES (default).
# For each shift, intersect -> TOI per state per player.
# For each shot, find on-ice players from shooting team and opposing team.

# Aggregates
# player level: for each (player, state), counts of attempts and goals in each zone,
# both individual (as shooter) and on-ice (for/against).
plr_ind_att = defaultdict(lambda: defaultdict(int))     # (pid,state,zone) -> attempts as shooter
plr_ind_gl  = defaultdict(lambda: defaultdict(int))     # as shooter goals
plr_onice_for_att  = defaultdict(lambda: defaultdict(int))  # (pid,state,zone) -> on-ice for
plr_onice_for_gl   = defaultdict(lambda: defaultdict(int))
plr_onice_ag_att   = defaultdict(lambda: defaultdict(int))
plr_onice_ag_gl    = defaultdict(lambda: defaultdict(int))

# TOI
plr_toi = defaultdict(lambda: defaultdict(float))   # (pid, state) -> seconds

# Season-scoped player buckets — mirror the per-(pid) buckets above but keyed
# by (pid, season_str), paralleling the season-keyed team buckets below. These
# feed ONLY the new player_counts_by_state_zone_per_season.csv writer; the
# existing per-(pid) buckets and their pooled output are untouched (add-only).
plr_ind_att_s = defaultdict(lambda: defaultdict(int))       # (pid,season) -> {(state,zone): attempts as shooter}
plr_ind_gl_s  = defaultdict(lambda: defaultdict(int))       # as shooter goals
plr_onice_for_att_s = defaultdict(lambda: defaultdict(int)) # (pid,season) -> on-ice for
plr_onice_for_gl_s  = defaultdict(lambda: defaultdict(int))
plr_onice_ag_att_s  = defaultdict(lambda: defaultdict(int))
plr_onice_ag_gl_s   = defaultdict(lambda: defaultdict(int))
# Fenwick (unblocked) on-ice for/against — same keying as the Corsi _att_s
# buckets but excluding blocked-shot events. Add-only: these feed two new
# trailing columns in player_counts_by_state_zone_per_season.csv and nothing
# else. The Corsi _att_s buckets above are untouched.
plr_onice_for_fen_s = defaultdict(lambda: defaultdict(int)) # (pid,season) -> on-ice for, Fenwick
plr_onice_ag_fen_s  = defaultdict(lambda: defaultdict(int)) # (pid,season) -> on-ice against, Fenwick
plr_toi_s = defaultdict(lambda: defaultdict(float))         # (pid, season) -> {state: seconds}

# Goalie-level (faced)
gk_faced = defaultdict(lambda: defaultdict(int))    # (gid, state, zone) -> shots faced (SOG+goals)
gk_goals = defaultdict(lambda: defaultdict(int))

# Team-level (game aggregates for later aggregation)
team_for = defaultdict(lambda: defaultdict(int))   # (season, team, state, zone) -> attempts for
team_ag  = defaultdict(lambda: defaultdict(int))
team_goals_for = defaultdict(lambda: defaultdict(int))
team_goals_ag  = defaultdict(lambda: defaultdict(int))
team_toi = defaultdict(float)  # (season, team, state) -> sec on-ice (skater-seconds/5)
                               # We'll compute team TOI differently: sum of game-regulation-seconds

# For score-adjustment (Step 6/7), we need score state when shot occurs.
# We don't have running score in shots. Approximate by computing score from prior goals in the shots dataframe.

# Precompute running score per game
print("Computing score state per shot...")
goal_rows = shots[shots["event_type"]=="goal"].sort_values(["game_id","abs_time"])
# for each shot, compute home_goals_at_time and away_goals_at_time BEFORE the event
shots = shots.sort_values(["game_id","abs_time","event_id"]).reset_index(drop=True)
shots["home_goal"] = ((shots["event_type"]=="goal") & (shots["shooting_team_id"]==shots["home_team_id"])).astype(int)
shots["away_goal"] = ((shots["event_type"]=="goal") & (shots["shooting_team_id"]!=shots["home_team_id"])).astype(int)
shots["home_score"] = shots.groupby("game_id")["home_goal"].cumsum() - shots["home_goal"]
shots["away_score"] = shots.groupby("game_id")["away_goal"].cumsum() - shots["away_goal"]
# shooting team score diff at time of event
shots["shoot_diff"] = np.where(shots["_shoot_home"],
                                shots["home_score"]-shots["away_score"],
                                shots["away_score"]-shots["home_score"])
# score state buckets: trailing >=2, trailing 1, tied, leading 1, leading >=2
def score_bucket(d):
    if d <= -2: return "trail2plus"
    if d == -1: return "trail1"
    if d == 0:  return "tied"
    if d == 1:  return "lead1"
    return "lead2plus"
shots["score_bucket"] = shots["shoot_diff"].apply(score_bucket)

# offensive/defensive zone at shot - we don't have explicit zone faceoff info; use coord zone proxy
# but zone adjustment uses WHERE shift STARTS. Without face-off data, approximate with:
# shot is "in O zone" => counts as Ozone-attempt for shooter team; use as proxy.
# For zone adjustment, we'll assume all shots are in attacking zone (since we're tracking shot attempts).
# We'll fall back to a simpler zone-adjust: score-only adjustment.

# Save tagged shots intermediate
print("Writing tagged shots CSV...")
shots[["game_id","season","period","event_id","abs_time","event_type","x_coord_norm","y_coord_norm",
       "shooting_team_id","shooting_team_abbrev","home_team_abbrev","away_team_abbrev",
       "shooter_player_id","goalie_id","state","zone","is_goal_i","score_bucket","shoot_diff","_shoot_home"]]\
       .rename(columns={"_shoot_home":"shoot_home"})\
       .to_csv(f"{OUT_DIR}/shots_tagged.csv", index=False)
print("  shots_tagged.csv written")

# ----------------- Game loop -----------------
print("Per-game shift-shot join...")
n_games = 0
for gid, gshots in shots_by_game.items():
    n_games += 1
    if n_games % 500 == 0:
        print(f"  processed {n_games} games")
    if gid not in shifts_by_game:
        continue
    gshifts = shifts_by_game[gid]
    season = game_season.get(gid, str(gid)[:4])
    season_str = f"{season[:4]}{season[4:]}" if len(season)==8 else season
    if season_str not in SEASONS and season not in SEASONS:
        # Only process target seasons
        if season_str in SEASONS:
            pass
        else:
            continue
    home_ab = team_abbrevs[gid]["home_abbrev"]
    away_ab = team_abbrevs[gid]["away_abbrev"]

    # -- Build state intervals per game --
    # events sorted by abs_time
    ev = gshots[["abs_time","state","_shoot_home","shooting_team_id","home_team_id"]].sort_values("abs_time").reset_index(drop=True)
    # first interval [0, first_event_time) assume ES
    intervals = []  # (start, end, state)
    prev_t = 0
    prev_state = "ES"
    for _, r in ev.iterrows():
        t = int(r["abs_time"])
        if t > prev_t:
            intervals.append((prev_t, t, prev_state))
        prev_t = t
        prev_state = r["state"]
    # final interval to end of regulation (3600)
    if prev_t < 3600:
        intervals.append((prev_t, 3600, prev_state))

    # Convert to arrays for fast intersect
    iv_starts = np.array([iv[0] for iv in intervals])
    iv_ends   = np.array([iv[1] for iv in intervals])
    iv_states = np.array([iv[2] for iv in intervals])

    # -- Build per-team shift lookup (list of (start,end,player_id)) --
    shifts_by_team = {}
    for team_ab, tsh in gshifts.groupby("team_abbrev"):
        starts = tsh["abs_start_secs"].values.astype(int)
        ends = tsh["abs_end_secs"].values.astype(int)
        pids = tsh["player_id"].values.astype(int)
        shifts_by_team[team_ab] = (starts, ends, pids)

    # -- TOI per player per state --
    for team_ab, (st, en, pids) in shifts_by_team.items():
        for i in range(len(pids)):
            pid = int(pids[i]); s=int(st[i]); e=int(en[i])
            if e <= s: continue
            # intersect with state intervals
            lo = np.searchsorted(iv_ends, s, side="right")
            for j in range(lo, len(iv_starts)):
                if iv_starts[j] >= e: break
                ovl = min(e, iv_ends[j]) - max(s, iv_starts[j])
                if ovl > 0:
                    plr_toi[pid][iv_states[j]] += ovl
                    plr_toi_s[(pid, season_str)][iv_states[j]] += ovl

    # -- Shot loop: identify on-ice players per team --
    for _, sh in gshots.iterrows():
        t = int(sh["abs_time"])
        state = sh["state"]; zone = sh["zone"]
        is_goal = int(sh["is_goal_i"])
        is_fen = sh["event_type"] != "blocked-shot"   # Fenwick = unblocked attempts
        is_attempt = 1
        shoot_ab = sh["shooting_team_abbrev"]
        def_ab = away_ab if shoot_ab == home_ab else home_ab
        shooter = sh["shooter_player_id"]
        goalie = sh["goalie_id"]

        # shooting team on-ice
        # Inclusive end boundary: shift-end timestamps are logged in the SAME
        # second as the stoppage that ends them, and a goal is exactly the
        # kind of event that causes that stoppage — so a strict "<" here
        # systematically dropped the scorer's own shift (and others ending
        # at that same second) from on-ice attribution. Verified directly:
        # 37/45 (82%) of a player's own goals failed the on-ice test under
        # the old strict bound; 3/45 (all explained by separate data gaps,
        # not this boundary) under the inclusive one.
        # np.unique() dedupes: a player can have a real shift AND a separate
        # zero-length "marker" row (abs_start==abs_end) both ending at the
        # same instant a goal is scored — the inclusive bound above lets
        # BOTH rows independently satisfy the on-ice test, double-counting
        # that player for that one event unless collapsed to one appearance.
        if shoot_ab in shifts_by_team:
            st_s, en_s, pids_s = shifts_by_team[shoot_ab]
            mask = (st_s < t) & (t <= en_s)
            onice_shoot = np.unique(pids_s[mask])
        else:
            onice_shoot = np.array([], dtype=int)
        if def_ab in shifts_by_team:
            st_d, en_d, pids_d = shifts_by_team[def_ab]
            mask = (st_d < t) & (t <= en_d)
            onice_def = np.unique(pids_d[mask])
        else:
            onice_def = np.array([], dtype=int)

        # exclude goalies from skater on-ice lists
        onice_shoot = [int(p) for p in onice_shoot if pos_map.get(int(p)) != "G"]
        onice_def   = [int(p) for p in onice_def   if pos_map.get(int(p)) != "G"]

        # individual shot (shooter)
        if not pd.isna(shooter):
            plr_ind_att[int(shooter)][(state, zone)] += 1
            plr_ind_att_s[(int(shooter), season_str)][(state, zone)] += 1
            if is_goal:
                plr_ind_gl[int(shooter)][(state, zone)] += 1
                plr_ind_gl_s[(int(shooter), season_str)][(state, zone)] += 1

        # on-ice for (shooting team)
        for p in onice_shoot:
            plr_onice_for_att[p][(state, zone)] += 1
            plr_onice_for_att_s[(p, season_str)][(state, zone)] += 1
            if is_fen:
                plr_onice_for_fen_s[(p, season_str)][(state, zone)] += 1
            if is_goal:
                plr_onice_for_gl[p][(state, zone)] += 1
                plr_onice_for_gl_s[(p, season_str)][(state, zone)] += 1
        # on-ice against (defending team)
        for p in onice_def:
            plr_onice_ag_att[p][(state, zone)] += 1
            plr_onice_ag_att_s[(p, season_str)][(state, zone)] += 1
            if is_fen:
                plr_onice_ag_fen_s[(p, season_str)][(state, zone)] += 1
            if is_goal:
                plr_onice_ag_gl[p][(state, zone)] += 1
                plr_onice_ag_gl_s[(p, season_str)][(state, zone)] += 1

        # Goalie pillar counting and team-level for/against counting moved
        # out of this loop (see vectorized post-loop pass below). Bug fix
        # May 2026 — they don't depend on shift data, so they shouldn't be
        # gated by the shift-availability check at the top of this loop.

print(f"Processed {n_games} games.")

# ---------------------------------------------------------------------
# Vectorized goalie + team aggregation (no shift-data dependency).
# Bug fix May 2026: previously gk_faced/gk_goals and team_for/team_ag/
# team_goals_for/team_goals_ag were populated inside the per-game
# shift-attribution loop above. When shift data was missing for a game,
# the entire game was skipped, causing those counters to under-count
# events by up to ~5% in seasons where shift_data.csv was incomplete
# (24-25 was missing 57 of 1312 regular-season games). Goalie save% and
# team for/against totals don't depend on skater shift data, so we now
# populate them in a vectorized post-loop pass over the shots DataFrame,
# filtered to regular-season games (game_id digits 4-5 == "02"). Skater
# on-ice attribution above remains correctly gated by shift availability.
# ---------------------------------------------------------------------
reg_shots = shots[shots["game_id"].astype(str).str[4:6] == "02"].copy()

# Goalies — faced events (SOG + goal) only, where goalie_id is present
_gk = reg_shots[
    reg_shots["event_type"].isin(["shot-on-goal","goal"])
    & reg_shots["goalie_id"].notna()
].copy()
_gk["goalie_id"] = _gk["goalie_id"].astype(int)
_gk_agg = _gk.groupby(["goalie_id","state","zone"]).agg(
    faced=("event_type","size"),
    goals=("is_goal_i","sum"),
).reset_index()
for _, r in _gk_agg.iterrows():
    gk_faced[int(r["goalie_id"])][(r["state"], r["zone"])] = int(r["faced"])
    gk_goals[int(r["goalie_id"])][(r["state"], r["zone"])] = int(r["goals"])

# Team-level — counts all Corsi attempts (event_type already filtered at
# line 119 to SOG/missed/blocked/goal)
reg_shots["_def_team"] = np.where(
    reg_shots["_shoot_home"],
    reg_shots["away_team_abbrev"],
    reg_shots["home_team_abbrev"])
_for_agg = reg_shots.groupby(
    ["season","shooting_team_abbrev","state","zone"]).agg(
    att=("event_type","size"),
    gl=("is_goal_i","sum"),
).reset_index()
for _, r in _for_agg.iterrows():
    key = (str(r["season"]), r["shooting_team_abbrev"])
    team_for[key][(r["state"], r["zone"])] = int(r["att"])
    team_goals_for[key][(r["state"], r["zone"])] = int(r["gl"])
_ag_agg = reg_shots.groupby(
    ["season","_def_team","state","zone"]).agg(
    att=("event_type","size"),
    gl=("is_goal_i","sum"),
).reset_index()
for _, r in _ag_agg.iterrows():
    key = (str(r["season"]), r["_def_team"])
    team_ag[key][(r["state"], r["zone"])] = int(r["att"])
    team_goals_ag[key][(r["state"], r["zone"])] = int(r["gl"])

print(f"  vectorized post-loop aggregation: "
      f"{len(_gk_agg)} goalie rows, {len(_for_agg)} team-for rows, "
      f"{len(_ag_agg)} team-ag rows")

# ----------------- Write TOI CSV -----------------
print("Writing TOI...")
toi_rows = []
all_pids = set(plr_toi) | set(plr_ind_att) | set(plr_onice_for_att) | set(plr_onice_ag_att)
for pid in sorted(all_pids):
    toi = plr_toi[pid]
    tot = sum(toi.values())
    toi_rows.append({
        "player_id": pid,
        "position": pos_map.get(pid,""),
        "toi_total_sec": round(tot,1),
        "toi_ES_sec": round(toi.get("ES",0),1),
        "toi_PP_sec": round(toi.get("PP",0),1),
        "toi_PK_sec": round(toi.get("PK",0),1),
        "toi_total_min": round(tot/60,2)
    })
toi_df = pd.DataFrame(toi_rows)
toi_df.to_csv(f"{OUT_DIR}/player_toi.csv", index=False)
print(f"  {len(toi_df)} players, {(toi_df['toi_total_min']>=MIN_TOI_MIN).sum()} >=500 min")

# Qualification map
qual = set(toi_df[toi_df["toi_total_min"]>=MIN_TOI_MIN]["player_id"].tolist())

# ----------------- Pillars -----------------
STATES = ["ES","PP","PK"]
ZONES_ALL = ["CNFI","MNFI","FNFI","Wide","lane_other","unk"]
ZONES_NFI = ["CNFI","MNFI","FNFI"]

def tnfi(counts):
    return sum(counts.get((s,z),0) for s in STATES for z in ZONES_NFI)

def agg_zone(dct, state, zone):
    """attempts for given zone/state. zone may be 'TNFI' = CNFI+MNFI+FNFI"""
    if zone == "TNFI":
        return sum(dct.get((state,z),0) for z in ZONES_NFI)
    return dct.get((state,zone),0)

# Helper: build pillar rows with exact Poisson 95% CIs on per-60 rates
def build_pillar(pid_dict, gl_dict, label, allowed_positions, onice=False):
    """pid_dict[pid][(state,zone)] = attempts. gl_dict same for goals.
       If onice: rate measured vs on-ice TOI. Else: individual rate also vs TOI.
       Per-60 CIs use the chi-square (Garwood) exact Poisson interval on the
       count, scaled to per-60 via rate_ci()."""
    rows = []
    for pid in sorted(set(pid_dict.keys()) | set(gl_dict.keys())):
        if pos_map.get(pid,"") not in allowed_positions:
            continue
        if pid not in qual:
            continue
        toi = plr_toi[pid]
        for state in STATES:
            mins = toi.get(state,0)/60.0
            if mins <= 0: continue
            for zone in ZONES_NFI + ["TNFI"]:
                att = agg_zone(pid_dict[pid], state, zone)
                gl  = agg_zone(gl_dict[pid], state, zone)
                r, lo, hi = rate_ci(att, mins)
                rows.append({
                    "player_id": pid,
                    "position": pos_map.get(pid,""),
                    "state": state,
                    "zone": zone,
                    "attempts": att,
                    "goals": gl,
                    "toi_min": round(mins,2),
                    "per60": round(r,3),
                    "per60_lo": round(lo,3),
                    "per60_hi": round(hi,3),
                })
    return pd.DataFrame(rows)

print("Building pillars...")
# Pillar 1: net-front forward (F) individual CNFI attempts / 60
rows = []
for pid, cnt in plr_ind_att.items():
    if pos_map.get(pid,"") != "F": continue
    if pid not in qual: continue
    toi = plr_toi[pid]
    for state in STATES:
        mins = toi.get(state,0)/60.0
        if mins <= 0: continue
        att = cnt.get((state,"CNFI"),0)
        gl  = plr_ind_gl[pid].get((state,"CNFI"),0)
        r,lo,hi = rate_ci(att, mins)
        rows.append({"player_id":pid,"position":"F","state":state,"zone":"CNFI",
                     "attempts":att,"goals":gl,"toi_min":round(mins,2),
                     "per60":round(r,3),"per60_lo":round(lo,3),"per60_hi":round(hi,3)})
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/pillar_1_netfront_F.csv", index=False)
print(f"  pillar 1 rows: {len(rows)}")

# Pillar 2: defensive F - CNFI/MNFI/FNFI/TNFI shots against per 60 while on ice (F)
pd2 = build_pillar(plr_onice_ag_att, plr_onice_ag_gl, "p2_defF", allowed_positions={"F"})
pd2.to_csv(f"{OUT_DIR}/pillar_2_defensive_F.csv", index=False)
print(f"  pillar 2 rows: {len(pd2)}")

# Pillar 3: offensive F individual MNFI/FNFI per 60
rows = []
for pid, cnt in plr_ind_att.items():
    if pos_map.get(pid,"") != "F": continue
    if pid not in qual: continue
    toi = plr_toi[pid]
    for state in STATES:
        mins = toi.get(state,0)/60.0
        if mins <= 0: continue
        for zone in ["MNFI","FNFI"]:
            att = cnt.get((state,zone),0)
            gl  = plr_ind_gl[pid].get((state,zone),0)
            r,lo,hi = rate_ci(att, mins)
            rows.append({"player_id":pid,"position":"F","state":state,"zone":zone,
                         "attempts":att,"goals":gl,"toi_min":round(mins,2),
                         "per60":round(r,3),"per60_lo":round(lo,3),"per60_hi":round(hi,3)})
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/pillar_3_offensive_F.csv", index=False)
print(f"  pillar 3 rows: {len(rows)}")

# Pillar 4: defensive D - all zones against on-ice
pd4 = build_pillar(plr_onice_ag_att, plr_onice_ag_gl, "p4_defD", allowed_positions={"D"})
# Also add Wide/lane_other/all reference zones
rows_extra = []
for pid, cnt in plr_onice_ag_att.items():
    if pos_map.get(pid,"") != "D": continue
    if pid not in qual: continue
    toi = plr_toi[pid]
    for state in STATES:
        mins = toi.get(state,0)/60.0
        if mins <= 0: continue
        for zone in ["Wide","lane_other"]:
            att = cnt.get((state,zone),0)
            gl  = plr_onice_ag_gl[pid].get((state,zone),0)
            r,lo,hi = rate_ci(att, mins)
            rows_extra.append({"player_id":pid,"position":"D","state":state,"zone":zone,
                               "attempts":att,"goals":gl,"toi_min":round(mins,2),
                               "per60":round(r,3),"per60_lo":round(lo,3),"per60_hi":round(hi,3)})
pd4 = pd.concat([pd4, pd.DataFrame(rows_extra)], ignore_index=True)
pd4.to_csv(f"{OUT_DIR}/pillar_4_defensive_D.csv", index=False)
print(f"  pillar 4 rows: {len(pd4)}")

# Pillar 5: offensive D - MNFI/FNFI/TNFI on-ice for + CNFI ref
pd5 = build_pillar(plr_onice_for_att, plr_onice_for_gl, "p5_offD", allowed_positions={"D"})
pd5.to_csv(f"{OUT_DIR}/pillar_5_offensive_D.csv", index=False)
print(f"  pillar 5 rows: {len(pd5)}")

# Pillar 6: Goalie FNFI + MNFI save% (min 500 shots faced)
rows = []
for gid, cnt in gk_faced.items():
    for state in STATES:
        for zone in ["MNFI","FNFI"]:
            faced = cnt.get((state,zone),0)
            goals = gk_goals[gid].get((state,zone),0)
            if faced < 500: continue
            sv = 1 - goals/faced if faced else 0
            p, lo, hi = wilson(faced-goals, faced)
            rows.append({"goalie_id":gid,"state":state,"zone":zone,"faced":faced,
                         "goals":goals,"save_pct":round(p,4),"sv_lo":round(lo,4),"sv_hi":round(hi,4)})
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/pillar_6_goalie_FNFI_MNFI.csv", index=False)
print(f"  pillar 6 rows: {len(rows)}")

# Pillar 7: Goalie CNFI save% (min 500 shots faced)
rows = []
for gid, cnt in gk_faced.items():
    for state in STATES:
        faced = cnt.get((state,"CNFI"),0)
        goals = gk_goals[gid].get((state,"CNFI"),0)
        if faced < 500: continue
        p, lo, hi = wilson(faced-goals, faced)
        rows.append({"goalie_id":gid,"state":state,"zone":"CNFI","faced":faced,"goals":goals,
                     "save_pct":round(p,4),"sv_lo":round(lo,4),"sv_hi":round(hi,4)})
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/pillar_7_goalie_CNFI.csv", index=False)
print(f"  pillar 7 rows: {len(rows)}")

# ----------------- Write aggregates for downstream steps -----------------
print("Saving aggregates for steps 6-9...")
# Player-level raw counts (all zones) for later
rows = []
pids_all = set(plr_ind_att)|set(plr_onice_for_att)|set(plr_onice_ag_att)|set(plr_toi)
for pid in pids_all:
    toi = plr_toi[pid]
    for state in STATES:
        mins = toi.get(state,0)/60.0
        for zone in ZONES_ALL:
            rec = {
                "player_id": pid, "position": pos_map.get(pid,""), "state": state, "zone": zone,
                "toi_min": round(mins,3),
                "ind_att": plr_ind_att[pid].get((state,zone),0),
                "ind_gl": plr_ind_gl[pid].get((state,zone),0),
                "onice_for_att": plr_onice_for_att[pid].get((state,zone),0),
                "onice_for_gl": plr_onice_for_gl[pid].get((state,zone),0),
                "onice_ag_att": plr_onice_ag_att[pid].get((state,zone),0),
                "onice_ag_gl": plr_onice_ag_gl[pid].get((state,zone),0),
            }
            rows.append(rec)
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/player_counts_by_state_zone.csv", index=False)
print("  wrote player_counts_by_state_zone.csv")

# Player-level raw counts, season-scoped (NEW, add-only). Same schema as the
# pooled player_counts_by_state_zone.csv above, plus a `season` column. Mirrors
# that writer's row structure (all STATES x ZONES_ALL per key) but iterates over
# the season-keyed buckets, emitting every processed season present. Downstream
# (Streamlit) scopes to the 4 reported seasons; the producer stays faithful.
rows = []
keys_all = (set(plr_ind_att_s) | set(plr_onice_for_att_s)
            | set(plr_onice_ag_att_s) | set(plr_toi_s))
for (pid, sn) in keys_all:
    toi = plr_toi_s[(pid, sn)]
    for state in STATES:
        mins = toi.get(state, 0) / 60.0
        for zone in ZONES_ALL:
            rec = {
                "player_id": pid, "season": sn, "position": pos_map.get(pid, ""),
                "state": state, "zone": zone,
                "toi_min": round(mins, 3),
                "ind_att": plr_ind_att_s[(pid, sn)].get((state, zone), 0),
                "ind_gl": plr_ind_gl_s[(pid, sn)].get((state, zone), 0),
                "onice_for_att": plr_onice_for_att_s[(pid, sn)].get((state, zone), 0),
                "onice_for_gl": plr_onice_for_gl_s[(pid, sn)].get((state, zone), 0),
                "onice_ag_att": plr_onice_ag_att_s[(pid, sn)].get((state, zone), 0),
                "onice_ag_gl": plr_onice_ag_gl_s[(pid, sn)].get((state, zone), 0),
                # Fenwick (no-blocks) siblings — appended last so existing columns
                # stay byte-identical; the file just gains these two trailing cols.
                "onice_for_fen": plr_onice_for_fen_s[(pid, sn)].get((state, zone), 0),
                "onice_ag_fen": plr_onice_ag_fen_s[(pid, sn)].get((state, zone), 0),
            }
            rows.append(rec)
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/player_counts_by_state_zone_per_season.csv", index=False)
print("  wrote player_counts_by_state_zone_per_season.csv")

# Team-level
rows = []
teams = set(team_for)|set(team_ag)
for (sn,t) in teams:
    for state in STATES:
        for zone in ZONES_ALL:
            rows.append({
                "season": sn, "team": t, "state": state, "zone": zone,
                "for_att": team_for[(sn,t)].get((state,zone),0),
                "for_gl": team_goals_for[(sn,t)].get((state,zone),0),
                "ag_att": team_ag[(sn,t)].get((state,zone),0),
                "ag_gl": team_goals_ag[(sn,t)].get((state,zone),0),
            })
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/team_counts_by_state_zone.csv", index=False)
print("  wrote team_counts_by_state_zone.csv")

# Done
print("Done.")
