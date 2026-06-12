#!/usr/bin/env python3
"""
Quality_Games / Step 1 — per-player-game attribution.

Lifts the on-ice attribution algorithm from
NFI/scripts/03_onice_attribution_pillars.py (lines 254-360) into a new
build that produces per-(player_id, season, game_id, team_abbrev)
records carrying:
  - 5v5 ES regulation TOI on ice
  - Fenwick attempts for/against while on ice
  - MoneyPuck xGoal sums for/against while on ice
  - Counts of attempts whose NFI zone is in CNFI ∪ MNFI

Scope: 4 seasons (2022-23 → 2025-26). 5v5 even-strength. Regulation
(periods 1-3). Fenwick only (SHOT/MISS/GOAL on MP; excludes blocked).
Empty-net rows dropped (homeEmptyNet==0 AND awayEmptyNet==0).

NFI ZONE SOURCE OF TRUTH (Path X):
  HR's NFI/Output/shots_tagged.csv `zone` column is canonical.
  - Inner-join MP↔HR on (nhl_gid, period, abs_time, shooter_player_id).
    HR's zone wins on matched rows.
  - For the small remainder where MP doesn't join cleanly to HR, fall
    back to classify_zone() recomputed on MP's xCordAdjusted/yCordAdjusted.
  - MP-internal collisions on the join key: deduplicate by sorting on
    shotID ascending and keeping the first row. ~250 rows/season dropped.
  - Sanity gate: per-season HR-match rate must be >=99%. RuntimeError if not.
  - Informational only: HR vs MP-recompute zone agreement on matched rows,
    printed but not a gate (expected 97.4%-99.7% per the May 2026 diagnostic).

Methodology rules:
  - xG% denom (downstream): all 5v5 reg Fenwick attempts on ice (no zone).
  - NFI% denom (downstream): CNFI ∪ MNFI attempts only.
  - Position F/D: per (player_id, season) from player_fully_adjusted.csv.
  - Team grain: per-game team_abbrev from shift_data (handles mid-season trades).
  - TOI denom: 5v5 ES state intervals built from shots_tagged.csv per game.

Outputs:
  Quality_Games/output/per_player_game.csv
  Quality_Games/output/diagnostic_log.txt
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROOT = "/Users/ashgarg/Documents/HockeyROI"
QG_DIR = f"{ROOT}/Quality_Games"
MP_DIR = f"{QG_DIR}/Data/Money_puck"
OUT_DIR = f"{QG_DIR}/output"

MP_SEASONS = [2022, 2023, 2024, 2025]
HR_SEASON = {2022: 20222023, 2023: 20232024, 2024: 20242025, 2025: 20252026}
HR_SEASONS_SET = set(HR_SEASON.values())

# Lifted from 03_onice_attribution_pillars.py lines 36-38
INFL1 = 55   # MNFI/FNFI boundary
BLUE = 25

HR_MATCH_FLOOR = 0.99   # sanity gate threshold for MP→HR join match rate

NFI_ZONES = {"CNFI", "MNFI"}


def classify_zone(x, y):
    """Lifted verbatim from 03_onice_attribution_pillars.py lines 71-82."""
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


# -----------------------------------------------------------------------------
# Diagnostic logger
# -----------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = f"{OUT_DIR}/diagnostic_log.txt"
_log_fh = open(LOG_PATH, "w")


def log(msg=""):
    s = str(msg)
    print(s, flush=True)
    _log_fh.write(s + "\n")
    _log_fh.flush()


log("=" * 78)
log("Quality_Games — 01_build_per_player_game.py  (Path X: HR-zone canonical)")
log("=" * 78)
log()
log("Scope:    4 seasons (2022-23, 2023-24, 2024-25, 2025-26)")
log("Strength: 5v5 even-strength regulation (periods 1-3)")
log("Filter:   Fenwick (SHOT/MISS/GOAL), no empty net")
log("Zone:     HR shots_tagged.csv.zone (canonical), MP-recompute fallback")
log(f"Output:   {OUT_DIR}")
log()

# -----------------------------------------------------------------------------
# STEP 1 — Load MoneyPuck shots, filter, MP-recompute zone (for fallback)
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 1 — Load MoneyPuck shots (filter + MP-recompute fallback zone)")
log("-" * 78)

mp_keep_cols = [
    'shotID',
    'season', 'game_id', 'isPlayoffGame', 'period', 'time',
    'event', 'xGoal', 'shooterPlayerId',
    'homeSkatersOnIce', 'awaySkatersOnIce',
    'homeEmptyNet', 'awayEmptyNet',
    'isHomeTeam', 'homeTeamCode', 'awayTeamCode',
    'xCordAdjusted', 'yCordAdjusted',
]

mp_list = []
for sy in MP_SEASONS:
    path = f"{MP_DIR}/shots_{sy}.csv"
    df = pd.read_csv(path, usecols=mp_keep_cols)
    n0 = len(df)
    mask = (
        (df.isPlayoffGame == 0)
        & df.event.isin(['SHOT', 'MISS', 'GOAL'])
        & df.period.between(1, 3)
        & (df.homeSkatersOnIce == 5) & (df.awaySkatersOnIce == 5)
        & (df.homeEmptyNet == 0) & (df.awayEmptyNet == 0)
    )
    df = df[mask].copy()
    df['nhl_gid'] = (df.season * 1_000_000 + df.game_id).astype(int)
    df['hr_season'] = HR_SEASON[sy]
    df['mp_recompute_zone'] = [classify_zone(x, y)
                                for x, y in zip(df.xCordAdjusted, df.yCordAdjusted)]
    n1 = len(df)
    log(f"  shots_{sy}.csv → {HR_SEASON[sy]}: {n0:>7,} rows → {n1:>7,} after filter ({n1/n0:.1%})")
    mp_list.append(df)

mp = pd.concat(mp_list, ignore_index=True)
log(f"  TOTAL MP 5v5 reg Fenwick rows (pre-dedupe, 4 seasons): {len(mp):,}")
log()

# Dedupe MP on (nhl_gid, period, time, shooterPlayerId): keep first row by shotID
log("  MP-internal dedupe on (nhl_gid, period, time, shooterPlayerId):")
mp = mp.sort_values(['nhl_gid', 'period', 'time', 'shooterPlayerId', 'shotID'],
                    kind='stable')
pre = len(mp)
mp_dedupe_drop_per_season = (
    mp.assign(_dup=mp.duplicated(subset=['nhl_gid', 'period', 'time', 'shooterPlayerId'],
                                  keep='first'))
      .groupby('hr_season')['_dup'].sum())
mp = mp.drop_duplicates(subset=['nhl_gid', 'period', 'time', 'shooterPlayerId'],
                        keep='first')
log(f"    dropped {pre - len(mp):,} duplicate rows  "
    f"(per season: {mp_dedupe_drop_per_season.to_dict()})")
log(f"  TOTAL MP rows after dedupe: {len(mp):,}")
log()

# -----------------------------------------------------------------------------
# STEP 2 — Load HR shots_tagged.csv (zone source + state intervals)
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 2 — Load HR shots_tagged.csv")
log("-" * 78)

hr_cols = ['game_id', 'season', 'period', 'abs_time',
           'event_type', 'shooter_player_id', 'state', 'zone']
hr_all = pd.read_csv(f"{ROOT}/NFI/Output/shots_tagged.csv", usecols=hr_cols)
n0 = len(hr_all)
hr = hr_all[
    hr_all.season.isin(HR_SEASONS_SET)
    & hr_all.period.between(1, 3)
    & (hr_all.game_id.astype(str).str[4:6] == '02')
].copy()
hr['game_id'] = hr.game_id.astype(int)
log(f"  shots_tagged.csv: {n0:,} → {len(hr):,} after 4-season+reg+period filter")
log(f"  per-season HR event counts: {hr.groupby('season').size().to_dict()}")
log()

# -----------------------------------------------------------------------------
# STEP 3 — Build HR zone-lookup table; join MP ← HR; sanity gate
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 3 — MP ← HR zone join (HR canonical, MP-recompute fallback)")
log("-" * 78)

# HR zone lookup limited to 5v5 ES Fenwick events (where shots_tagged carries
# meaningful zone tags for our scope)
hr_fen_es = hr[hr.event_type.isin(['shot-on-goal', 'missed-shot', 'goal'])
               & (hr.state == 'ES')].copy()

# Dedupe HR on same join key (consecutive HR rows at the same key typically
# share the same zone; keep first for a deterministic 1:1 lookup table)
hr_lookup = hr_fen_es.sort_values(['game_id', 'period', 'abs_time',
                                    'shooter_player_id'], kind='stable')
pre_hr = len(hr_lookup)
hr_lookup = hr_lookup.drop_duplicates(
    subset=['game_id', 'period', 'abs_time', 'shooter_player_id'], keep='first')
log(f"  HR lookup table: {pre_hr:,} → {len(hr_lookup):,} after key dedupe "
    f"(dropped {pre_hr - len(hr_lookup):,})")

# Left-join MP ← HR (per season for diagnostics)
mp_keys = mp[['nhl_gid', 'period', 'time', 'shooterPlayerId']].rename(
    columns={'nhl_gid': 'game_id', 'time': 'abs_time',
              'shooterPlayerId': 'shooter_player_id'})
mp_keys.index = mp.index
joined_zone = pd.merge(
    mp_keys,
    hr_lookup[['game_id', 'period', 'abs_time', 'shooter_player_id', 'zone']],
    on=['game_id', 'period', 'abs_time', 'shooter_player_id'],
    how='left',
).set_index(mp.index)

mp['hr_zone'] = joined_zone['zone'].values
mp['zone_source'] = np.where(mp['hr_zone'].notna(), 'HR', 'MP_recompute')
mp['nfi_zone'] = np.where(mp['hr_zone'].notna(),
                           mp['hr_zone'],
                           mp['mp_recompute_zone'])
mp['in_nfi'] = mp['nfi_zone'].isin(NFI_ZONES)

# Per-season match-rate sanity gate
log("  Per-season MP → HR match diagnostic:")
log(f"  {'season':<10} {'MP rows':>10} {'HR-matched':>12} {'match%':>9} "
    f"{'fallback':>10} {'fb_rate':>9}")
fail_seasons = []
for sn in sorted(HR_SEASONS_SET):
    sub = mp[mp.hr_season == sn]
    n = len(sub)
    n_hr = (sub.zone_source == 'HR').sum()
    n_fb = n - n_hr
    rate = n_hr / n if n > 0 else 0.0
    fb_rate = n_fb / n if n > 0 else 0.0
    log(f"  {sn:<10} {n:>10,} {n_hr:>12,} {rate:>8.3%} {n_fb:>10,} {fb_rate:>8.3%}")
    if rate < HR_MATCH_FLOOR:
        fail_seasons.append((sn, rate))

if fail_seasons:
    for sn, rate in fail_seasons:
        log(f"  ⚠  Season {sn} match rate {rate:.3%} < floor {HR_MATCH_FLOOR:.0%}")
    raise RuntimeError(
        f"MP→HR match rate below {HR_MATCH_FLOOR:.0%} for: "
        f"{[f'{sn}={r:.2%}' for sn,r in fail_seasons]}")
log("  ✓ all seasons pass HR-match-rate gate")
log()

# Informational: HR-zone vs MP-recompute agreement on matched rows
log("  Informational — HR vs MP-recompute zone agreement on matched rows:")
log("    (not a gate; documented for transparency. See May 2026 diagnostic for context.)")
log(f"  {'season':<10} {'matched':>10} {'5-way agree':>14} {'in_nfi agree':>14}")
for sn in sorted(HR_SEASONS_SET):
    sub = mp[(mp.hr_season == sn) & (mp.zone_source == 'HR')]
    if len(sub) == 0:
        log(f"  {sn:<10} {0:>10,}  (no matched rows)"); continue
    full = (sub.hr_zone == sub.mp_recompute_zone).mean()
    flag_match = (sub.hr_zone.isin(NFI_ZONES) ==
                  sub.mp_recompute_zone.isin(NFI_ZONES)).mean()
    log(f"  {sn:<10} {len(sub):>10,} {full:>13.3%} {flag_match:>13.3%}")
log()

# Cleanup helper columns no longer needed
mp = mp.drop(columns=['hr_zone', 'mp_recompute_zone'])

# -----------------------------------------------------------------------------
# STEP 4 — Build 5v5 ES state intervals per game (from HR shots_tagged)
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 4 — Build 5v5 ES state intervals per game")
log("-" * 78)

hr_iv = hr.sort_values(['game_id', 'abs_time']).reset_index(drop=True)
es_intervals_by_game = {}
for gid, ev in hr_iv.groupby('game_id', sort=False):
    intervals = []
    prev_t = 0
    prev_state = 'ES'
    abs_times = ev.abs_time.to_numpy()
    states = ev.state.to_numpy()
    for i in range(len(abs_times)):
        t = int(abs_times[i])
        if t > prev_t:
            intervals.append((prev_t, t, prev_state))
        prev_t = t
        prev_state = states[i]
    if prev_t < 3600:
        intervals.append((prev_t, 3600, prev_state))
    es_intervals_by_game[gid] = sorted((s, e) for (s, e, st) in intervals if st == 'ES')

n_games_int = len(es_intervals_by_game)
total_es_sec = sum(e - s for ivs in es_intervals_by_game.values() for s, e in ivs)
log(f"  Built ES intervals for {n_games_int:,} games")
log(f"  Total ES seconds: {total_es_sec:,}  (~{total_es_sec/n_games_int/60:.1f} min/game avg)")
log()

# -----------------------------------------------------------------------------
# STEP 5 — Load shift_data.csv (streaming filter)
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 5 — Load shift_data.csv")
log("-" * 78)

target_gids = set(mp.nhl_gid.unique().astype(int).tolist())
log(f"  Target game_ids (MP 5v5 reg universe): {len(target_gids):,}")

shift_cols = ['game_id', 'player_id', 'period', 'team_abbrev',
              'abs_start_secs', 'abs_end_secs']
shift_chunks = []
for chunk in pd.read_csv(f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv",
                          usecols=shift_cols, chunksize=500_000):
    chunk = chunk.dropna(subset=['game_id', 'player_id', 'period',
                                  'abs_start_secs', 'abs_end_secs'])
    chunk = chunk.astype({'game_id': int, 'player_id': int, 'period': int,
                           'abs_start_secs': int, 'abs_end_secs': int})
    chunk = chunk[chunk.game_id.isin(target_gids) & chunk.period.between(1, 3)]
    if len(chunk):
        shift_chunks.append(chunk)
shifts = pd.concat(shift_chunks, ignore_index=True)
del shift_chunks
log(f"  Loaded {len(shifts):,} shift rows for {shifts.game_id.nunique():,} games "
    f"({shifts.player_id.nunique():,} unique player_ids)")

gids_with_shifts = set(shifts.game_id.unique().astype(int).tolist())
gids_missing = target_gids - gids_with_shifts
if gids_missing:
    log(f"  ⚠  {len(gids_missing)} target gids have NO shift rows — skipped in attribution.")
log()

# -----------------------------------------------------------------------------
# STEP 6 — Position lookups
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 6 — Position lookups (goalie filter + per-season F/D)")
log("-" * 78)

pp = pd.read_csv(f"{ROOT}/NFI/Output/player_positions.csv")
pos_filter = dict(zip(pp.player_id.astype(int), pp.pos_group))
log(f"  player_positions.csv: {len(pp):,} rows  ({pp.pos_group.value_counts().to_dict()})")

fa = pd.read_csv(f"{ROOT}/NFI/Output/fully_adjusted/player_fully_adjusted.csv",
                  usecols=['player_id', 'season', 'position'])
pos_by_ps = {(int(r.player_id), int(r.season)): r.position
              for r in fa.itertuples(index=False)}
log(f"  player_fully_adjusted.csv: {len(fa):,} rows "
    f"({fa.position.value_counts().to_dict()})")
log()

# -----------------------------------------------------------------------------
# STEP 7 — Per-game attribution
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 7 — Per-game attribution (shift ↔ shot)")
log("-" * 78)

mp_by_game = dict(tuple(mp.groupby('nhl_gid', sort=False)))
shifts_by_game = dict(tuple(shifts.groupby('game_id', sort=False)))

toi_on = defaultdict(int)
team_in_game = {}
hr_season_for_gid = {}

att_for = defaultdict(int)
att_ag = defaultdict(int)
xg_for = defaultdict(float)
xg_ag = defaultdict(float)
nfi_for = defaultdict(int)
nfi_ag = defaultdict(int)

team_anomalies = 0
n_processed = 0
n_skipped_no_shifts = 0
n_skipped_no_intervals = 0
shoot_def_missing = 0
total_games = len(mp_by_game)

for gid, gshots in mp_by_game.items():
    n_processed += 1
    if n_processed % 1000 == 0:
        log(f"  processed {n_processed:,}/{total_games:,} games")
    if gid not in shifts_by_game:
        n_skipped_no_shifts += 1
        continue
    if gid not in es_intervals_by_game:
        n_skipped_no_intervals += 1
        continue

    gshifts = shifts_by_game[gid]
    hr_sn = int(gshots['hr_season'].iloc[0])
    hr_season_for_gid[gid] = hr_sn
    es_intervals = es_intervals_by_game[gid]

    shifts_by_team = {}
    player_seen_teams = defaultdict(set)
    for team_ab, tsh in gshifts.groupby('team_abbrev', sort=False):
        starts = tsh.abs_start_secs.to_numpy(dtype=np.int64)
        ends = tsh.abs_end_secs.to_numpy(dtype=np.int64)
        pids = tsh.player_id.to_numpy(dtype=np.int64)
        shifts_by_team[team_ab] = (starts, ends, pids)
        for p in np.unique(pids):
            player_seen_teams[int(p)].add(team_ab)
    for pid, teams in player_seen_teams.items():
        if len(teams) > 1:
            team_anomalies += 1
        team_in_game[(pid, gid)] = sorted(teams)[0]

    # TOI on-ice intersection with ES intervals
    for team_ab, (starts, ends, pids) in shifts_by_team.items():
        for i in range(len(pids)):
            s = int(starts[i]); e = int(ends[i])
            if e <= s:
                continue
            pid = int(pids[i])
            for es_s, es_e in es_intervals:
                if es_e <= s:
                    continue
                if es_s >= e:
                    break
                ovl = min(e, es_e) - max(s, es_s)
                if ovl > 0:
                    toi_on[(pid, gid)] += ovl

    # Shot-level attribution
    home_ab = gshots.homeTeamCode.iloc[0]
    away_ab = gshots.awayTeamCode.iloc[0]

    times = gshots.time.to_numpy(dtype=np.int64)
    is_home = gshots.isHomeTeam.to_numpy(dtype=np.int64)
    xgoals = gshots.xGoal.fillna(0.0).to_numpy(dtype=np.float64)
    in_nfi_arr = gshots.in_nfi.to_numpy(dtype=bool)

    for i in range(len(times)):
        t = int(times[i])
        if is_home[i] == 1:
            shoot_ab, def_ab = home_ab, away_ab
        else:
            shoot_ab, def_ab = away_ab, home_ab

        if shoot_ab not in shifts_by_team or def_ab not in shifts_by_team:
            shoot_def_missing += 1
            continue

        st_s, en_s, pids_s = shifts_by_team[shoot_ab]
        onice_s = pids_s[(st_s <= t) & (t < en_s)]
        st_d, en_d, pids_d = shifts_by_team[def_ab]
        onice_d = pids_d[(st_d <= t) & (t < en_d)]

        xg = float(xgoals[i])
        in_nfi = bool(in_nfi_arr[i])

        for p in onice_s:
            pid = int(p)
            if pos_filter.get(pid) == 'G':
                continue
            att_for[(pid, gid)] += 1
            xg_for[(pid, gid)] += xg
            if in_nfi:
                nfi_for[(pid, gid)] += 1
        for p in onice_d:
            pid = int(p)
            if pos_filter.get(pid) == 'G':
                continue
            att_ag[(pid, gid)] += 1
            xg_ag[(pid, gid)] += xg
            if in_nfi:
                nfi_ag[(pid, gid)] += 1

log(f"  processed {n_processed:,} games  "
    f"(skipped {n_skipped_no_shifts} no shifts, {n_skipped_no_intervals} no ES intervals)")
log(f"  team-assignment anomalies (>1 team_abbrev per player per game): {team_anomalies}")
log(f"  shots dropped — shoot/def team missing from shifts: {shoot_def_missing}")
log()

# -----------------------------------------------------------------------------
# STEP 8 — Emit per_player_game.csv
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 8 — Emit per_player_game.csv")
log("-" * 78)

all_keys = set(toi_on) | set(att_for) | set(att_ag)
log(f"  Distinct (player_id, game_id) records: {len(all_keys):,}")

rows = []
n_missing_team = 0
n_missing_pos = 0
n_goalie_dropped = 0
for (pid, gid) in all_keys:
    hr_sn = hr_season_for_gid.get(gid)
    if hr_sn is None:
        continue
    if pos_filter.get(pid) == 'G':
        n_goalie_dropped += 1
        continue
    team = team_in_game.get((pid, gid))
    if team is None:
        n_missing_team += 1
        continue
    position = pos_by_ps.get((pid, hr_sn), '')
    if position == '':
        n_missing_pos += 1
    rows.append({
        'season': hr_sn,
        'game_id': gid,
        'player_id': pid,
        'team_abbrev': team,
        'position': position,
        'TOI_on_sec': int(toi_on.get((pid, gid), 0)),
        'attempts_for': int(att_for.get((pid, gid), 0)),
        'attempts_ag': int(att_ag.get((pid, gid), 0)),
        'xG_for': round(float(xg_for.get((pid, gid), 0.0)), 6),
        'xG_ag': round(float(xg_ag.get((pid, gid), 0.0)), 6),
        'NFI_for': int(nfi_for.get((pid, gid), 0)),
        'NFI_ag': int(nfi_ag.get((pid, gid), 0)),
    })

out_df = pd.DataFrame(rows).sort_values(['season', 'game_id', 'player_id']) \
                            .reset_index(drop=True)
out_path = f"{OUT_DIR}/per_player_game.csv"
out_df.to_csv(out_path, index=False)
log(f"  Wrote: {out_path}  ({len(out_df):,} rows)")
log(f"  Goalies dropped at emit: {n_goalie_dropped}")
log(f"  Missing team_abbrev (dropped): {n_missing_team}")
log(f"  Missing position F/D (kept, position=''): {n_missing_pos}")
log()

# -----------------------------------------------------------------------------
# STEP 9 — Verification
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 9 — Verification")
log("-" * 78)

log(f"  Total rows:                              {len(out_df):,}")
log(f"  Unique (player_id, season):              "
    f"{out_df.groupby(['player_id','season']).ngroups:,}")
log(f"  Unique (player_id, season, team_abbrev): "
    f"{out_df.groupby(['player_id','season','team_abbrev']).ngroups:,}")
log(f"  Unique game_ids:                         {out_df.game_id.nunique():,}")
log(f"  Position distribution: {out_df.position.value_counts(dropna=False).to_dict()}")
log()
log("  Per-season sums:")
agg = out_df.groupby('season').agg(
    rows=('player_id', 'size'),
    games=('game_id', 'nunique'),
    TOI_min=('TOI_on_sec', lambda x: x.sum() / 60),
    att_for=('attempts_for', 'sum'),
    att_ag=('attempts_ag', 'sum'),
    xG_for=('xG_for', 'sum'),
    xG_ag=('xG_ag', 'sum'),
    NFI_for=('NFI_for', 'sum'),
    NFI_ag=('NFI_ag', 'sum'),
)
for sn, r in agg.iterrows():
    log(f"   {sn}: rows={int(r.rows):>7,}  games={int(r.games):,}  "
        f"TOI_min={r.TOI_min:>10,.0f}  "
        f"att_for={int(r.att_for):>7,} att_ag={int(r.att_ag):>7,}  "
        f"xG_for={r.xG_for:>9.1f} xG_ag={r.xG_ag:>9.1f}  "
        f"NFI_for={int(r.NFI_for):>6,} NFI_ag={int(r.NFI_ag):>6,}")
log()

# Spot-check: McDavid 2024-25 (player_id=8478402)
mcd_id = 8478402
mcd = out_df[(out_df.player_id == mcd_id) & (out_df.season == 20242025)]
log("  Spot-check: Connor McDavid 2024-25 (player_id=8478402)")
log(f"    per-game records:        {len(mcd)}")
if len(mcd):
    toi_min = mcd.TOI_on_sec.sum() / 60
    a_for = int(mcd.attempts_for.sum())
    a_ag = int(mcd.attempts_ag.sum())
    x_for = float(mcd.xG_for.sum())
    x_ag = float(mcd.xG_ag.sum())
    n_for = int(mcd.NFI_for.sum())
    n_ag = int(mcd.NFI_ag.sum())
    log(f"    TOI total:               {int(mcd.TOI_on_sec.sum()):,} sec ({toi_min:.1f} min)")
    log(f"    attempts_for sum:        {a_for:,}")
    log(f"    attempts_ag sum:         {a_ag:,}")
    log(f"    xG_for sum:              {x_for:.3f}")
    log(f"    xG_ag sum:               {x_ag:.3f}")
    log(f"    NFI_for sum:             {n_for:,}")
    log(f"    NFI_ag sum:              {n_ag:,}")
    if (x_for + x_ag) > 0:
        log(f"    Implied season xG%:      {x_for/(x_for+x_ag):.4f}")
    if (n_for + n_ag) > 0:
        log(f"    Implied season NFI%:     {n_for/(n_for+n_ag):.4f}")
    log(f"    Position:                {mcd.position.iloc[0]}")
    log(f"    Team(s) in season:       {sorted(mcd.team_abbrev.unique().tolist())}")
log()

log("Done.")
_log_fh.close()
