#!/usr/bin/env python3
"""
Quality_Games / Step 2 — Quality Game aggregation.

Reads:    Quality_Games/output/per_player_game.csv
Produces:
  Quality_Games/output/position_medians.csv
  Quality_Games/output/per_player_season_team.csv  (intermediate)
  Quality_Games/output/per_player_season.csv
  Quality_Games/output/per_team_season.csv
  appends to Quality_Games/output/diagnostic_log.txt

Methodology (locked):
  Per-game qualifying floor:
    TOI_on_sec >= 480 (8 min)  AND  (attempts_for + attempts_ag) >= 5

  Per-game rates (computed on qualifying games only):
    xG_pct_game  = xG_for / (xG_for + xG_ag)
    NFI_pct_game = NFI_for / (NFI_for + NFI_ag)   (NaN if denominator 0)

  Position-median thresholds (empirical, league-wide, 4 seasons pooled):
    F median xG%,  D median xG%,  F median NFI%,  D median NFI%
    Each computed across qualifying player-games of that position.

  Quality Game flag:
    is_xG_QG  = 1.0  if xG_pct_game  >= position median xG%   (else 0.0)
    is_NFI_QG = 1.0  if NFI_pct_game >  position median NFI%
              = 0.5  if NFI_pct_game == position median NFI%  (half-credit tie)
              = 0.0  if NFI_pct_game <  position median NFI%

  Tie handling (June 2026 revision): NFI per-game ratios are discrete (1/2,
  2/4, 3/6 → exactly 0.500) because NFI denominators per game are small —
  14.24% of all qualifying games land at exactly 0.5000 on NFI. xG per-game
  ratios are continuous (xGoal weight sums) — only ~0.002% land within
  ±1e-5 of 0.5000 — so the tie issue does not apply.
    - Prior strict-`>` rule on NFI counted ties as losses (asymmetric, flag
      rate fell to 0.432 vs xG's 0.500).
    - Prior `>=` rule counted them as wins (flag rate spiked to 0.574).
    - Current half-credit rule (0.5 for ties) handles indeterminate outcomes
      symmetrically. NFI flag rate now ≈ 0.500 by construction.
    - xG-QG retains `>=` because ties are vanishingly rare under a
      continuous distribution; asymmetry across metrics reflects the
      different underlying ratio shapes, not arbitrary choice.

  Player-season rates (across both teams in trade cases):
    xG_QG_pct  = sum(is_xG_QG)  / qualifying_GP_with_valid_xG
    NFI_QG_pct = sum(is_NFI_QG) / qualifying_GP_with_valid_NFI

  Team aggregation:
    Eligibility: 20+ GP with that team in that season
    Weight:      TOI_total_sec with that team
    team_xG_QG_pct  = TOI-weighted mean of player xG_QG_pct
    team_NFI_QG_pct = TOI-weighted mean of player NFI_QG_pct
    (rates are PLAYER-SEASON rates, not per-team rates)
"""
import os
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Documents/HockeyROI"
QG_DIR = f"{ROOT}/Quality_Games"
OUT_DIR = f"{QG_DIR}/output"

MIN_TOI_SEC = 480     # 8 min
MIN_ATTEMPTS = 5      # for + against, both included
TEAM_GP_FLOOR = 20    # 20+ GP with that team in that season

LOG_PATH = f"{OUT_DIR}/diagnostic_log.txt"
_log_fh = open(LOG_PATH, "a")


def log(msg=""):
    s = str(msg)
    print(s, flush=True)
    _log_fh.write(s + "\n")
    _log_fh.flush()


log()
log("=" * 78)
log("Quality_Games — 02_quality_game_aggregation.py")
log("=" * 78)
log()
log(f"Qualifying floor: TOI_on_sec >= {MIN_TOI_SEC}  AND  attempts_total >= {MIN_ATTEMPTS}")
log(f"Team eligibility: {TEAM_GP_FLOOR}+ GP with that team in that season")
log(f"Input:  {OUT_DIR}/per_player_game.csv")
log(f"Output: {OUT_DIR}/")
log()

# -----------------------------------------------------------------------------
# STEP 1 — Load per_player_game and apply qualifying floor
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 1 — Load per_player_game.csv + apply per-game qualifying floor")
log("-" * 78)

df = pd.read_csv(f"{OUT_DIR}/per_player_game.csv")
log(f"  Loaded {len(df):,} rows  ({df.groupby(['player_id','season']).ngroups:,} player-seasons, "
    f"{df.groupby(['player_id','season','team_abbrev']).ngroups:,} player-season-teams)")

df['attempts_total'] = df.attempts_for + df.attempts_ag
df['qualifies'] = (df.TOI_on_sec >= MIN_TOI_SEC) & (df.attempts_total >= MIN_ATTEMPTS)

n_qual = int(df.qualifies.sum())
log(f"  Per-game qualifying floor: {n_qual:,} / {len(df):,} rows pass "
    f"({n_qual/len(df):.1%})")

qual = df[df.qualifies].copy()
log(f"  Per-season qualifying counts:")
log(qual.groupby('season').size().to_string())
log()

# Check blank-position rows surviving the qualifying floor
blank_qual = qual[~qual.position.isin(['F', 'D'])]
log(f"  Blank-position rows surviving the qualifying floor: {len(blank_qual):,}")
if len(blank_qual):
    log(f"    distinct player_ids: {blank_qual.player_id.nunique()}")
    log(f"    distinct (player_id, season): "
        f"{blank_qual.groupby(['player_id','season']).ngroups}")
    sample = blank_qual.head(5)[['season','game_id','player_id','team_abbrev',
                                  'TOI_on_sec','attempts_for','attempts_ag']]
    log("    sample rows:")
    log(sample.to_string(index=False))
log()

# -----------------------------------------------------------------------------
# STEP 2 — Compute per-game rates
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 2 — Per-game xG% and NFI% on qualifying games")
log("-" * 78)

qual['xG_total_game'] = qual.xG_for + qual.xG_ag
qual['nfi_total_game'] = qual.NFI_for + qual.NFI_ag

# xG denominator is xG_for + xG_ag — only zero if no attempts, but attempts_total >= 5 guarantees
# at least some xG mass. Treat exact zero as NaN to be safe.
qual['xG_pct_game'] = np.where(qual.xG_total_game > 0,
                                qual.xG_for / qual.xG_total_game, np.nan)
qual['NFI_pct_game'] = np.where(qual.nfi_total_game > 0,
                                 qual.NFI_for / qual.nfi_total_game, np.nan)

log(f"  Qualifying games with valid xG%:  {qual.xG_pct_game.notna().sum():,}")
log(f"  Qualifying games with valid NFI%: {qual.NFI_pct_game.notna().sum():,}  "
    f"(NaN when NFI_for+NFI_ag == 0)")
log(f"  Qualifying games with NFI denom == 0: {qual.NFI_pct_game.isna().sum():,} "
    f"(player had no NFI-zone shots either for or against)")
log()

# -----------------------------------------------------------------------------
# STEP 3 — Empirical position-median thresholds
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 3 — Empirical position-median thresholds (4-season pooled)")
log("-" * 78)

# Use known-position rows for threshold derivation
qual_known = qual[qual.position.isin(['F', 'D'])].copy()
log(f"  Known-position qualifying rows: {len(qual_known):,} "
    f"({qual_known.position.value_counts().to_dict()})")

medians = {}
for pos in ['F', 'D']:
    sub = qual_known[qual_known.position == pos]
    xg_vals = sub.xG_pct_game.dropna()
    nfi_vals = sub.NFI_pct_game.dropna()
    medians[(pos, 'xG')] = float(xg_vals.median())
    medians[(pos, 'NFI')] = float(nfi_vals.median())
    log(f"  {pos}: xG median={medians[(pos,'xG')]:.4f}  (n={len(xg_vals):,})  "
        f"NFI median={medians[(pos,'NFI')]:.4f}  (n={len(nfi_vals):,})")

pm_df = pd.DataFrame([
    {'position': 'F', 'metric': 'xG',  'median': medians[('F', 'xG')],  'n_qualifying_games': int(qual_known[qual_known.position=='F'].xG_pct_game.notna().sum())},
    {'position': 'D', 'metric': 'xG',  'median': medians[('D', 'xG')],  'n_qualifying_games': int(qual_known[qual_known.position=='D'].xG_pct_game.notna().sum())},
    {'position': 'F', 'metric': 'NFI', 'median': medians[('F', 'NFI')], 'n_qualifying_games': int(qual_known[qual_known.position=='F'].NFI_pct_game.notna().sum())},
    {'position': 'D', 'metric': 'NFI', 'median': medians[('D', 'NFI')], 'n_qualifying_games': int(qual_known[qual_known.position=='D'].NFI_pct_game.notna().sum())},
])
pm_path = f"{OUT_DIR}/position_medians.csv"
pm_df.to_csv(pm_path, index=False)
log(f"  Wrote: {pm_path}")
log()

# -----------------------------------------------------------------------------
# STEP 4 — Flag Quality Games per qualifying player-game
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 4 — Quality Game flags per qualifying player-game")
log("-" * 78)

xg_thr = qual_known.position.map({'F': medians[('F','xG')],  'D': medians[('D','xG')]})
nfi_thr = qual_known.position.map({'F': medians[('F','NFI')], 'D': medians[('D','NFI')]})

qual_known['is_xG_QG'] = np.where(qual_known.xG_pct_game.isna(),
                                    np.nan,
                                    (qual_known.xG_pct_game >= xg_thr).astype(float))
qual_known['is_NFI_QG'] = np.where(
    qual_known.NFI_pct_game.isna(),                  np.nan,
    np.where(qual_known.NFI_pct_game >  nfi_thr,     1.0,
    np.where(qual_known.NFI_pct_game == nfi_thr,     0.5,
                                                     0.0)))

log(f"  is_xG_QG  flagged: {int(qual_known.is_xG_QG.sum()):,} / "
    f"{int(qual_known.is_xG_QG.notna().sum()):,} valid "
    f"({qual_known.is_xG_QG.mean():.3f})")
log(f"  is_NFI_QG flagged: {int(qual_known.is_NFI_QG.sum()):,} / "
    f"{int(qual_known.is_NFI_QG.notna().sum()):,} valid "
    f"({qual_known.is_NFI_QG.mean():.3f})")
log()

# -----------------------------------------------------------------------------
# STEP 5 — Per (player, season, team) aggregation
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 5 — Per (player_id, season, team_abbrev) aggregation")
log("-" * 78)

# GP and TOI come from the FULL df (qualifying or not — GP is just "showed up").
pst_gp = df.groupby(['player_id', 'season', 'team_abbrev']).agg(
    GP=('game_id', 'size'),
    TOI_total_sec=('TOI_on_sec', 'sum'),
    position=('position', lambda x: x.mode().iloc[0] if not x.mode().empty else ''),
).reset_index()

# Qualifying counts come from qual_known (where position was valid)
pst_qual = qual_known.groupby(['player_id', 'season', 'team_abbrev']).agg(
    qualifying_GP=('game_id', 'size'),
    xG_QG_count=('is_xG_QG', 'sum'),
    xG_qual_GP=('is_xG_QG', lambda x: x.notna().sum()),
    NFI_QG_count=('is_NFI_QG', 'sum'),
    NFI_qual_GP=('is_NFI_QG', lambda x: x.notna().sum()),
).reset_index()

pst = pst_gp.merge(pst_qual,
                    on=['player_id', 'season', 'team_abbrev'],
                    how='left').fillna({
    'qualifying_GP': 0, 'xG_QG_count': 0, 'xG_qual_GP': 0,
    'NFI_QG_count': 0, 'NFI_qual_GP': 0,
})
# Cast integer-valued count columns; keep NFI_QG_count as float because the
# half-credit-tie rule produces 0.5 fractional contributions per tie game.
# Truncating to int would silently lose 0.5 of QG credit per odd-tie season,
# producing ~0.003-0.008 systematic underestimate of NFI_QG_pct.
for c in ['qualifying_GP','xG_QG_count','xG_qual_GP','NFI_qual_GP']:
    pst[c] = pst[c].astype(int)
pst['NFI_QG_count'] = pst['NFI_QG_count'].astype(float)

pst_path = f"{OUT_DIR}/per_player_season_team.csv"
pst.to_csv(pst_path, index=False)
log(f"  Wrote: {pst_path}  ({len(pst):,} rows)")
log(f"  Distinct (player_id, season) reached via these rows: "
    f"{pst.groupby(['player_id','season']).ngroups:,}")
log()

# -----------------------------------------------------------------------------
# STEP 6 — Per (player, season) aggregation + rates
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 6 — Per (player_id, season) aggregation")
log("-" * 78)

ps = pst.groupby(['player_id', 'season']).agg(
    position=('position', lambda x: x.mode().iloc[0] if not x.mode().empty else ''),
    GP=('GP', 'sum'),
    TOI_total_sec=('TOI_total_sec', 'sum'),
    qualifying_GP=('qualifying_GP', 'sum'),
    xG_QG_count=('xG_QG_count', 'sum'),
    xG_qual_GP=('xG_qual_GP', 'sum'),
    NFI_QG_count=('NFI_QG_count', 'sum'),
    NFI_qual_GP=('NFI_qual_GP', 'sum'),
    teams_in_season=('team_abbrev', lambda x: ','.join(sorted(set(x)))),
).reset_index()

ps['xG_QG_pct']  = np.where(ps.xG_qual_GP  > 0, ps.xG_QG_count  / ps.xG_qual_GP,  np.nan)
ps['NFI_QG_pct'] = np.where(ps.NFI_qual_GP > 0, ps.NFI_QG_count / ps.NFI_qual_GP, np.nan)

# Verify: blank-position rows surviving into player-season output
ps_blank = ps[~ps.position.isin(['F', 'D'])]
log(f"  Blank-position rows surviving into per_player_season: {len(ps_blank):,}")
if len(ps_blank) > 0:
    log(f"  ⚠  Blank-position survivors at season level. Showing first 5:")
    log(ps_blank.head(5)[['player_id','season','GP','qualifying_GP','TOI_total_sec']].to_string(index=False))
    log("  Dropping these from per_player_season output.")
ps = ps[ps.position.isin(['F','D'])].copy()

ps_path = f"{OUT_DIR}/per_player_season.csv"
ps_out = ps[['player_id','season','position','teams_in_season',
              'GP','qualifying_GP','TOI_total_sec',
              'xG_QG_count','xG_qual_GP','xG_QG_pct',
              'NFI_QG_count','NFI_qual_GP','NFI_QG_pct']]
ps_out.to_csv(ps_path, index=False)
log(f"  Wrote: {ps_path}  ({len(ps_out):,} rows)")
log()

# -----------------------------------------------------------------------------
# STEP 7 — Per (team, season) aggregation
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 7 — Per (team_abbrev, season) team aggregation")
log("-" * 78)

# Attach player-season rates to per-(player, season, team) rows
pst_with_rates = pst.merge(
    ps[['player_id', 'season', 'xG_QG_pct', 'NFI_QG_pct']],
    on=['player_id', 'season'], how='inner')   # 'inner' drops blank-position rows

# Eligibility: 20+ GP with that team in that season
elig = pst_with_rates[pst_with_rates.GP >= TEAM_GP_FLOOR].copy()
log(f"  per-(player, season, team) rows: total={len(pst_with_rates):,}  "
    f"eligible (GP>={TEAM_GP_FLOOR}): {len(elig):,}")

# Per-team TOI-weighted aggregation
def toi_weighted_mean(group, rate_col):
    valid = group.dropna(subset=[rate_col])
    if len(valid) == 0:
        return np.nan
    w = valid['TOI_total_sec'].values.astype(float)
    if w.sum() == 0:
        return np.nan
    return float(np.average(valid[rate_col].values, weights=w))

team_rows = []
for (team, season), grp in elig.groupby(['team_abbrev', 'season']):
    txg = toi_weighted_mean(grp, 'xG_QG_pct')
    tnfi = toi_weighted_mean(grp, 'NFI_QG_pct')
    team_rows.append({
        'team_abbrev': team,
        'season': int(season),
        'team_xG_QG_pct': txg,
        'team_NFI_QG_pct': tnfi,
        'n_eligible_players': int(len(grp)),
        'n_with_valid_xG':  int(grp.xG_QG_pct.notna().sum()),
        'n_with_valid_NFI': int(grp.NFI_QG_pct.notna().sum()),
        'total_team_TOI_sec': int(grp.TOI_total_sec.sum()),
        'total_team_TOI_min': round(grp.TOI_total_sec.sum() / 60, 1),
    })
team_df = pd.DataFrame(team_rows).sort_values(['season', 'team_abbrev']).reset_index(drop=True)

ts_path = f"{OUT_DIR}/per_team_season.csv"
team_df.to_csv(ts_path, index=False)
log(f"  Wrote: {ts_path}  ({len(team_df):,} rows)")
log(f"  Per-season team counts: {team_df.groupby('season').size().to_dict()}")
log()

# -----------------------------------------------------------------------------
# STEP 8 — Verification prints
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 8 — Verification")
log("-" * 78)

# (a) position_medians.csv content
log("  position_medians.csv:")
log(pm_df.to_string(index=False))
log()

## DISPLAY FLOOR for leaderboards — does NOT filter per_player_season.csv.
##   Source column: TOI_total_sec / 60 (per_player_season.csv TOI_total_min,
##   = sum of per-game on-ice 5v5 ES seconds across the season).
DISPLAY_TOI_MIN_FLOOR = 400

# (b) Top 20 by xG_QG_pct in 2025-26 (with display floor)
log("-" * 78)
log(f"  Top 20 players by xG_QG_pct for 2025-26  (descending)  "
    f"[display floor: TOI_total_min >= {DISPLAY_TOI_MIN_FLOOR}]")
log("  (tie-broken by qualifying_GP descending)")
log("-" * 78)
ps_25 = ps[ps.season == 20252026].copy()
ps_25['TOI_total_min'] = ps_25.TOI_total_sec / 60
ps_25_disp = ps_25[ps_25.TOI_total_min >= DISPLAY_TOI_MIN_FLOOR]
log(f"  Players passing display floor for 2025-26: {len(ps_25_disp):,} "
    f"(of {len(ps_25):,} total)")

top_xg = ps_25_disp.dropna(subset=['xG_QG_pct']).sort_values(
    ['xG_QG_pct', 'qualifying_GP'], ascending=[False, False]).head(20)

# Add player_name from player_positions for readability
pp = pd.read_csv(f"{ROOT}/NFI/Output/player_positions.csv",
                 usecols=['player_id', 'player_name'])
pp = pp.drop_duplicates('player_id')
top_xg = top_xg.merge(pp, on='player_id', how='left')
log(top_xg[['player_id','player_name','position','teams_in_season',
             'GP','qualifying_GP','TOI_total_min',
             'xG_QG_count','xG_qual_GP','xG_QG_pct']].round({'TOI_total_min':1, 'xG_QG_pct':4}).to_string(index=False))
log()

# (c) Top 20 by NFI_QG_pct in 2025-26 (with display floor + half-credit-tie rule)
log("-" * 78)
log(f"  Top 20 players by NFI_QG_pct for 2025-26  (descending)  "
    f"[display floor: TOI_total_min >= {DISPLAY_TOI_MIN_FLOOR}]")
log("  (half-credit-tie rule on NFI, tie-broken by qualifying_GP descending)")
log("-" * 78)
top_nfi = ps_25_disp.dropna(subset=['NFI_QG_pct']).sort_values(
    ['NFI_QG_pct', 'qualifying_GP'], ascending=[False, False]).head(20)
top_nfi = top_nfi.merge(pp, on='player_id', how='left')
log(top_nfi[['player_id','player_name','position','teams_in_season',
              'GP','qualifying_GP','TOI_total_min',
              'NFI_QG_count','NFI_qual_GP','NFI_QG_pct']].round({'TOI_total_min':1, 'NFI_QG_pct':4}).to_string(index=False))
log()

# (d) 32 team table for 2025-26, sorted by team_xG_QG_pct desc
log("-" * 78)
log("  32 teams for 2025-26, ranked by team_xG_QG_pct (descending)")
log("-" * 78)
t25 = team_df[team_df.season == 20252026].sort_values('team_xG_QG_pct', ascending=False)
log(t25[['team_abbrev','n_eligible_players','total_team_TOI_min',
          'team_xG_QG_pct','team_NFI_QG_pct']].to_string(index=False))
log()

# (e) 4-season team trajectories (per_team_season.csv resorted)
log("-" * 78)
log("  4-season team trajectories — all 128 rows, sorted by team_abbrev ASC, season ASC")
log("-" * 78)
traj = team_df.sort_values(['team_abbrev', 'season']).reset_index(drop=True)
traj_path = f"{OUT_DIR}/team_trajectories_4season.csv"
traj.to_csv(traj_path, index=False)
log(f"  Saved: {traj_path}")
log()
log(traj[['team_abbrev','season','n_eligible_players','total_team_TOI_min',
          'team_xG_QG_pct','team_NFI_QG_pct']]
        .round({'team_xG_QG_pct':4, 'team_NFI_QG_pct':4})
        .to_string(index=False))
log()

# (f) Player spot-checks for build-correctness review
log("-" * 78)
log("  Player spot-checks")
log("-" * 78)


def report_player(pid, season, label):
    log()
    log(f"  {label} (player_id={pid}, season={season}):")
    r = ps[(ps.player_id == pid) & (ps.season == season)]
    if len(r) != 1:
        log(f"    ⚠  expected 1 row, found {len(r)}")
        return
    r = r.iloc[0]
    log(f"    position:        {r.position}")
    log(f"    teams_in_season: {r.teams_in_season}")
    log(f"    GP:              {r.GP}")
    log(f"    qualifying_GP:   {r.qualifying_GP}")
    log(f"    TOI_total_min:   {r.TOI_total_sec/60:.1f}")
    log(f"    xG_QG_count:     {r.xG_QG_count}  (of {r.xG_qual_GP} qual GP w/ valid xG)")
    log(f"    xG_QG_pct:       {r.xG_QG_pct:.4f}")
    log(f"    NFI_QG_count:    {r.NFI_QG_count}  (of {r.NFI_qual_GP} qual GP w/ valid NFI)")
    log(f"    NFI_QG_pct:      {r.NFI_QG_pct:.4f}")


report_player(8478402, 20242025, "Connor McDavid 2024-25")
report_player(8479318, 20252026, "Auston Matthews 2025-26 (TOR, F)")
report_player(8479542, 20252026, "Brandon Hagel 2025-26 (TBL, F, full-season-no-trade)")
report_player(8475218, 20252026, "Mattias Ekholm 2025-26 (EDM, D, top-D anchor)")
log()

log("Done.")
_log_fh.close()
