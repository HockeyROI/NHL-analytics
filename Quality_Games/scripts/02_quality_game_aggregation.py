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

# Playoff scope — mirror of script 01's QG_SCOPE. Reads per_player_game_playoffs,
# writes *_playoffs outputs (per playoff season + an `all_playoffs` pooled row),
# and drops the team-GP exclusion floor (playoff runs are short; the Streamlit
# slider does the thresholding). The per-game qualifying floor (TOI>=480 &
# attempts>=5) is the QG metric's own definition and is retained.
SCOPE = os.environ.get("QG_SCOPE", "regular")
_IS_PLAYOFF = SCOPE == "playoff"
_SUF = "_playoffs" if _IS_PLAYOFF else ""
if _IS_PLAYOFF:
    TEAM_GP_FLOOR = 1     # no exclusion floor — any team with playoff games

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

df = pd.read_csv(f"{OUT_DIR}/per_player_game{_SUF}.csv")
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

# Team-game totals (for Rel-QG metrics).
# At 5v5 ES regulation each shot is attributed to all 5 on-ice skaters per
# side. So summing per-player counters across team T's player rows in game G
# gives 5× the team's actual total. Divide by 5 to recover team totals.
# Sum is over `df` (all on-ice records, not just qualifying ones) so that
# sub-qualifying linemates' contributions aren't lost from the team rollup.
team_game = df.groupby(['game_id', 'season', 'team_abbrev']).agg(
    team_NFI_for=('NFI_for', lambda x: x.sum() / 5.0),
    team_NFI_ag =('NFI_ag',  lambda x: x.sum() / 5.0),
    team_xG_for =('xG_for',  lambda x: x.sum() / 5.0),
    team_xG_ag  =('xG_ag',   lambda x: x.sum() / 5.0),
    team_TOI_sec=('TOI_on_sec', lambda x: x.sum() / 5.0),
).reset_index()
log(f"  Team-game totals derived: {len(team_game):,} (team, game) rows")
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
pm_path = f"{OUT_DIR}/position_medians{_SUF}.csv"
pm_df.to_csv(pm_path, index=False)
log(f"  Wrote: {pm_path}")
log()

# -----------------------------------------------------------------------------
# STEP 3b — Per-game Rel shares (team-without-player baseline)
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 3b — Per-game Rel-NFI and Rel-xG shares (team-without-player)")
log("-" * 78)

# Merge team-game totals onto qual_known
qual_known = qual_known.merge(
    team_game, on=['game_id', 'season', 'team_abbrev'], how='left')

# "Without me" team totals = team total minus player's own on-ice piece.
# Same shot, when player is on ice, contributes to BOTH his on-ice counter
# AND the team's total (via the 5x rollup). Subtracting gives the team's
# total when this player was OFF the ice.
qual_known['team_NFI_for_wo'] = qual_known['team_NFI_for'] - qual_known['NFI_for']
qual_known['team_NFI_ag_wo']  = qual_known['team_NFI_ag']  - qual_known['NFI_ag']
qual_known['team_xG_for_wo']  = qual_known['team_xG_for']  - qual_known['xG_for']
qual_known['team_xG_ag_wo']   = qual_known['team_xG_ag']   - qual_known['xG_ag']

# Team-without-player share (NaN if no off-ice activity that game)
_denom_nfi_wo = qual_known.team_NFI_for_wo + qual_known.team_NFI_ag_wo
qual_known['team_NFI_pct_wo'] = np.where(_denom_nfi_wo > 0,
                                           qual_known.team_NFI_for_wo / _denom_nfi_wo,
                                           np.nan)
_denom_xg_wo = qual_known.team_xG_for_wo + qual_known.team_xG_ag_wo
qual_known['team_xG_pct_wo'] = np.where(_denom_xg_wo > 0,
                                          qual_known.team_xG_for_wo / _denom_xg_wo,
                                          np.nan)

# Per-game Rel shares (player on-ice share minus team-without-player share)
qual_known['RelNFI_pct_game'] = qual_known.NFI_pct_game - qual_known.team_NFI_pct_wo
qual_known['RelxG_pct_game']  = qual_known.xG_pct_game  - qual_known.team_xG_pct_wo

log(f"  Qualifying games with valid Rel-NFI share: {qual_known.RelNFI_pct_game.notna().sum():,}")
log(f"  Qualifying games with valid Rel-xG  share: {qual_known.RelxG_pct_game.notna().sum():,}")
log()

# -----------------------------------------------------------------------------
# STEP 4 — Flag Quality Games per qualifying player-game
# -----------------------------------------------------------------------------
log("-" * 78)
log("STEP 4 — Quality Game flags per qualifying player-game")
log("-" * 78)

xg_thr = qual_known.position.map({'F': medians[('F','xG')],  'D': medians[('D','xG')]})
nfi_thr = qual_known.position.map({'F': medians[('F','NFI')], 'D': medians[('D','NFI')]})

# Absolute QG flags (unchanged)
qual_known['is_xG_QG'] = np.where(qual_known.xG_pct_game.isna(),
                                    np.nan,
                                    (qual_known.xG_pct_game >= xg_thr).astype(float))
qual_known['is_NFI_QG'] = np.where(
    qual_known.NFI_pct_game.isna(),                  np.nan,
    np.where(qual_known.NFI_pct_game >  nfi_thr,     1.0,
    np.where(qual_known.NFI_pct_game == nfi_thr,     0.5,
                                                     0.0)))

# Rel-NFI-QG (half-credit, mirrors absolute NFI rule).
# Threshold is 0 (player share exceeds team-without-player share).
qual_known['is_RelNFI_QG'] = np.where(
    qual_known.RelNFI_pct_game.isna(),               np.nan,
    np.where(qual_known.RelNFI_pct_game >  0,        1.0,
    np.where(qual_known.RelNFI_pct_game == 0,        0.5,
                                                     0.0)))

# Rel-xG-QG (strict >, mirrors absolute xG rule — continuous distribution,
# exact-zero ties essentially never occur).
qual_known['is_RelxG_QG'] = np.where(qual_known.RelxG_pct_game.isna(),
                                       np.nan,
                                       (qual_known.RelxG_pct_game > 0).astype(float))

log(f"  is_xG_QG     flagged: {int(qual_known.is_xG_QG.sum()):,} / "
    f"{int(qual_known.is_xG_QG.notna().sum()):,} valid "
    f"({qual_known.is_xG_QG.mean():.3f})")
log(f"  is_NFI_QG    flagged: {int(qual_known.is_NFI_QG.sum()):,} / "
    f"{int(qual_known.is_NFI_QG.notna().sum()):,} valid "
    f"({qual_known.is_NFI_QG.mean():.3f})")
log(f"  is_RelxG_QG  flagged: {int(qual_known.is_RelxG_QG.sum()):,} / "
    f"{int(qual_known.is_RelxG_QG.notna().sum()):,} valid "
    f"({qual_known.is_RelxG_QG.mean():.3f})")
log(f"  is_RelNFI_QG flagged: {qual_known.is_RelNFI_QG.sum():.1f} / "
    f"{int(qual_known.is_RelNFI_QG.notna().sum()):,} valid "
    f"({qual_known.is_RelNFI_QG.mean():.3f})")
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
    RelxG_QG_count=('is_RelxG_QG', 'sum'),
    RelxG_qual_GP=('is_RelxG_QG', lambda x: x.notna().sum()),
    RelNFI_QG_count=('is_RelNFI_QG', 'sum'),
    RelNFI_qual_GP=('is_RelNFI_QG', lambda x: x.notna().sum()),
).reset_index()

pst = pst_gp.merge(pst_qual,
                    on=['player_id', 'season', 'team_abbrev'],
                    how='left').fillna({
    'qualifying_GP': 0, 'xG_QG_count': 0, 'xG_qual_GP': 0,
    'NFI_QG_count': 0, 'NFI_qual_GP': 0,
    'RelxG_QG_count': 0, 'RelxG_qual_GP': 0,
    'RelNFI_QG_count': 0, 'RelNFI_qual_GP': 0,
})
# Cast integer-valued count columns; keep NFI_QG_count and RelNFI_QG_count as
# float because the half-credit-tie rule produces 0.5 fractional contributions
# per tie game. Truncating to int would silently lose 0.5 of QG credit per
# odd-tie season, producing a systematic underestimate of the *_QG_pct rates.
# RelxG_QG_count is binary 0/1 so int is safe.
for c in ['qualifying_GP','xG_QG_count','xG_qual_GP','NFI_qual_GP',
           'RelxG_QG_count','RelxG_qual_GP','RelNFI_qual_GP']:
    pst[c] = pst[c].astype(int)
pst['NFI_QG_count']    = pst['NFI_QG_count'].astype(float)
pst['RelNFI_QG_count'] = pst['RelNFI_QG_count'].astype(float)

# -----------------------------------------------------------------------------
# STEP 5b — Stint-level RelxG per-60 rate differentials
# -----------------------------------------------------------------------------
# Methodology mirror of NFI/scripts/build_playoff_data.py _rel():
#   on60_F  = player xG_for / player TOI * 3600
#   off60_F = (team xG_for - player xG_for) / (team TOI - player TOI) * 3600
#   RelxG_F_pct = on60_F - off60_F
#   RelxG_A_pct = off60_A - on60_A   (sign flipped: positive = suppression lift)
#   RelxG_pct   = RelxG_F_pct + RelxG_A_pct
# Per-stint: player counters and team baseline both scoped to the stint's
# (team_abbrev, season) — team baseline is that team's full season (all games).
# Emits NaN when off_TOI <= 0 or player TOI <= 0 (no minute floor).
log("-" * 78)
log("STEP 5b — Stint-level RelxG per-60 rate differentials")
log("-" * 78)

# Per-stint player xG counters (across all games the player played with that team)
pst_xg = df.groupby(['player_id', 'season', 'team_abbrev']).agg(
    xG_for_sum=('xG_for', 'sum'),
    xG_ag_sum =('xG_ag',  'sum'),
).reset_index()

# Team-season counters: sum team_game across all games of that team in that season
team_season = team_game.groupby(['team_abbrev', 'season']).agg(
    team_xG_for_season=('team_xG_for',  'sum'),
    team_xG_ag_season =('team_xG_ag',   'sum'),
    team_TOI_sec_season=('team_TOI_sec', 'sum'),
).reset_index()

pst = pst.merge(pst_xg,      on=['player_id','season','team_abbrev'], how='left')
pst = pst.merge(team_season, on=['team_abbrev','season'],             how='left')

pst['RelxG_F_pct'] = np.nan
pst['RelxG_A_pct'] = np.nan
pst['RelxG_pct']   = np.nan

_p_toi   = pst['TOI_total_sec'].astype(float).values
_t_toi   = pst['team_TOI_sec_season'].astype(float).values
_off_toi = _t_toi - _p_toi
_mask    = (_p_toi > 0) & (_off_toi > 0)
if _mask.any():
    _p_xF = pst['xG_for_sum'].astype(float).values[_mask]
    _p_xA = pst['xG_ag_sum'].astype(float).values[_mask]
    _t_xF = pst['team_xG_for_season'].astype(float).values[_mask]
    _t_xA = pst['team_xG_ag_season'].astype(float).values[_mask]
    _pt   = _p_toi[_mask]
    _ot   = _off_toi[_mask]
    _on60_F  = _p_xF / _pt * 3600.0
    _on60_A  = _p_xA / _pt * 3600.0
    _off60_F = (_t_xF - _p_xF) / _ot * 3600.0
    _off60_A = (_t_xA - _p_xA) / _ot * 3600.0
    pst.loc[_mask, 'RelxG_F_pct'] = _on60_F  - _off60_F
    pst.loc[_mask, 'RelxG_A_pct'] = _off60_A - _on60_A
    pst.loc[_mask, 'RelxG_pct']   = (_on60_F - _off60_F) + (_off60_A - _on60_A)

log(f"  Stints with valid RelxG: {int(pst['RelxG_pct'].notna().sum()):,} / {len(pst):,}")
log(f"  RelxG_pct distribution: min={pst['RelxG_pct'].min():.3f} "
    f"median={pst['RelxG_pct'].median():.3f} max={pst['RelxG_pct'].max():.3f}")
log()

pst_path = f"{OUT_DIR}/per_player_season_team{_SUF}.csv"
pst_out_cols = ['player_id','season','team_abbrev','GP','TOI_total_sec','position',
                'qualifying_GP',
                'xG_QG_count','xG_qual_GP',
                'NFI_QG_count','NFI_qual_GP',
                'RelxG_QG_count','RelxG_qual_GP',
                'RelNFI_QG_count','RelNFI_qual_GP',
                'RelxG_F_pct','RelxG_A_pct','RelxG_pct']
pst[pst_out_cols].to_csv(pst_path, index=False)
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

def _player_agg(keys, season_label=None):
    g = pst.groupby(keys).agg(
        position=('position', lambda x: x.mode().iloc[0] if not x.mode().empty else ''),
        GP=('GP', 'sum'),
        TOI_total_sec=('TOI_total_sec', 'sum'),
        qualifying_GP=('qualifying_GP', 'sum'),
        xG_QG_count=('xG_QG_count', 'sum'),
        xG_qual_GP=('xG_qual_GP', 'sum'),
        NFI_QG_count=('NFI_QG_count', 'sum'),
        NFI_qual_GP=('NFI_qual_GP', 'sum'),
        RelxG_QG_count=('RelxG_QG_count', 'sum'),
        RelxG_qual_GP=('RelxG_qual_GP', 'sum'),
        RelNFI_QG_count=('RelNFI_QG_count', 'sum'),
        RelNFI_qual_GP=('RelNFI_qual_GP', 'sum'),
        # Pooled counters for season-level RelxG per-60 differential.
        # xG_for_pool / xG_ag_pool sum across all stints → player's full
        # (season or career) on-ice xG totals. team_xG_*_pool sum across
        # the same stints → each stint contributes ONE team's full-season
        # totals; for a player with N team stints in a season we get the
        # combined N-team baseline (single team for non-traded players,
        # A+B for traded ones). TOI_pool mirrors player TOI; team_TOI_pool
        # mirrors team baseline TOI.
        xG_for_pool=('xG_for_sum', 'sum'),
        xG_ag_pool =('xG_ag_sum',  'sum'),
        team_xG_for_pool =('team_xG_for_season',  'sum'),
        team_xG_ag_pool  =('team_xG_ag_season',   'sum'),
        team_TOI_sec_pool=('team_TOI_sec_season', 'sum'),
        teams_in_season=('team_abbrev', lambda x: ','.join(sorted(set(x)))),
    ).reset_index()
    if season_label is not None:
        g['season'] = season_label
    return g


ps = _player_agg(['player_id', 'season'])
if _IS_PLAYOFF:
    # Pooled all_playoffs row per player: sum counts across playoff seasons.
    ps = pd.concat([ps, _player_agg(['player_id'], 'all_playoffs')], ignore_index=True)

ps['xG_QG_pct']     = np.where(ps.xG_qual_GP     > 0, ps.xG_QG_count     / ps.xG_qual_GP,     np.nan)
ps['NFI_QG_pct']    = np.where(ps.NFI_qual_GP    > 0, ps.NFI_QG_count    / ps.NFI_qual_GP,    np.nan)
ps['RelxG_QG_pct']  = np.where(ps.RelxG_qual_GP  > 0, ps.RelxG_QG_count  / ps.RelxG_qual_GP,  np.nan)
ps['RelNFI_QG_pct'] = np.where(ps.RelNFI_qual_GP > 0, ps.RelNFI_QG_count / ps.RelNFI_qual_GP, np.nan)

# Season-level RelxG per-60 differential from pooled counters.
ps['RelxG_F_pct'] = np.nan
ps['RelxG_A_pct'] = np.nan
ps['RelxG_pct']   = np.nan
_p_toi   = ps['TOI_total_sec'].astype(float).values
_t_toi   = ps['team_TOI_sec_pool'].astype(float).values
_off_toi = _t_toi - _p_toi
_mask    = (_p_toi > 0) & (_off_toi > 0)
if _mask.any():
    _p_xF = ps['xG_for_pool'].astype(float).values[_mask]
    _p_xA = ps['xG_ag_pool'].astype(float).values[_mask]
    _t_xF = ps['team_xG_for_pool'].astype(float).values[_mask]
    _t_xA = ps['team_xG_ag_pool'].astype(float).values[_mask]
    _pt   = _p_toi[_mask]
    _ot   = _off_toi[_mask]
    _on60_F  = _p_xF / _pt * 3600.0
    _on60_A  = _p_xA / _pt * 3600.0
    _off60_F = (_t_xF - _p_xF) / _ot * 3600.0
    _off60_A = (_t_xA - _p_xA) / _ot * 3600.0
    ps.loc[_mask, 'RelxG_F_pct'] = _on60_F  - _off60_F
    ps.loc[_mask, 'RelxG_A_pct'] = _off60_A - _on60_A
    ps.loc[_mask, 'RelxG_pct']   = (_on60_F - _off60_F) + (_off60_A - _on60_A)

log(f"  Season rows with valid RelxG: {int(ps['RelxG_pct'].notna().sum()):,} / {len(ps):,}")
log(f"  RelxG_pct distribution: min={ps['RelxG_pct'].min():.3f} "
    f"median={ps['RelxG_pct'].median():.3f} max={ps['RelxG_pct'].max():.3f}")

# Verify: blank-position rows surviving into player-season output
ps_blank = ps[~ps.position.isin(['F', 'D'])]
log(f"  Blank-position rows surviving into per_player_season: {len(ps_blank):,}")
if len(ps_blank) > 0:
    log(f"  ⚠  Blank-position survivors at season level. Showing first 5:")
    log(ps_blank.head(5)[['player_id','season','GP','qualifying_GP','TOI_total_sec']].to_string(index=False))
    log("  Dropping these from per_player_season output.")
ps = ps[ps.position.isin(['F','D'])].copy()

ps_path = f"{OUT_DIR}/per_player_season{_SUF}.csv"
ps_out = ps[['player_id','season','position','teams_in_season',
              'GP','qualifying_GP','TOI_total_sec',
              'xG_QG_count','xG_qual_GP','xG_QG_pct',
              'NFI_QG_count','NFI_qual_GP','NFI_QG_pct',
              'RelxG_QG_count','RelxG_qual_GP','RelxG_QG_pct',
              'RelNFI_QG_count','RelNFI_qual_GP','RelNFI_QG_pct',
              'RelxG_F_pct','RelxG_A_pct','RelxG_pct']]
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
if _IS_PLAYOFF:
    # Pooled all_playoffs team rows: sum each player's GP/TOI-with-team across
    # playoff seasons, weight by the player's all_playoffs rate.
    pst_pool = pst.groupby(['player_id', 'team_abbrev']).agg(
        GP=('GP', 'sum'), TOI_total_sec=('TOI_total_sec', 'sum')).reset_index()
    pool_rates = ps[ps.season == 'all_playoffs'][['player_id', 'xG_QG_pct', 'NFI_QG_pct']]
    pst_pool = pst_pool.merge(pool_rates, on='player_id', how='inner')
    elig_pool = pst_pool[pst_pool.GP >= TEAM_GP_FLOOR].copy()
    log(f"  pooled all_playoffs (player, team) rows: total={len(pst_pool):,}  "
        f"eligible (GP>={TEAM_GP_FLOOR}): {len(elig_pool):,}")
    for team, grp in elig_pool.groupby('team_abbrev'):
        team_rows.append({
            'team_abbrev': team,
            'season': 'all_playoffs',
            'team_xG_QG_pct': toi_weighted_mean(grp, 'xG_QG_pct'),
            'team_NFI_QG_pct': toi_weighted_mean(grp, 'NFI_QG_pct'),
            'n_eligible_players': int(len(grp)),
            'n_with_valid_xG':  int(grp.xG_QG_pct.notna().sum()),
            'n_with_valid_NFI': int(grp.NFI_QG_pct.notna().sum()),
            'total_team_TOI_sec': int(grp.TOI_total_sec.sum()),
            'total_team_TOI_min': round(grp.TOI_total_sec.sum() / 60, 1),
        })

team_df = pd.DataFrame(team_rows)
if _IS_PLAYOFF:
    team_df['season'] = team_df['season'].astype(str)  # mixed int + 'all_playoffs'
team_df = team_df.sort_values(['season', 'team_abbrev']).reset_index(drop=True)

ts_path = f"{OUT_DIR}/per_team_season{_SUF}.csv"
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
traj_path = f"{OUT_DIR}/team_trajectories_4season{_SUF}.csv"
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

# -----------------------------------------------------------------------------
# Rel-QG locked spot-checks (June 2026 sensitivity-test values, regular-season
# only — playoff scope produces different career means by design and shouldn't
# be validated against these regular-season anchors).
# -----------------------------------------------------------------------------
_SKIP_SPOTCHECKS = _IS_PLAYOFF
if _SKIP_SPOTCHECKS:
    log("-" * 78)
    log("Rel-QG locked spot-checks — skipped under playoff scope")
    log("-" * 78)
else:
    log("-" * 78)
    log("Rel-QG locked spot-checks — 4-season career averages (TOI>=400 each season)")
    log("-" * 78)

# Expected values from the sensitivity test (June 2026). ±0.01 is the pass band.
RELQG_EXPECTED = [
    # (player_id, name, cNFI_QG, cRelNFI_QG, cxG_QG, cRelxG_QG)
    (8478402, "Connor McDavid",  0.6861, 0.625, 0.6793, 0.651),
    (8475786, "Zach Hyman",      0.6890, 0.626, 0.6552, 0.605),
    (8476958, "Jaccob Slavin",   0.6515, 0.537, 0.7005, 0.521),
    (8480803, "Evan Bouchard",   0.6860, 0.616, 0.6987, 0.643),
    (8479542, "Brandon Hagel",   0.6839, 0.674, 0.6025, 0.649),
    (8480069, "Cale Makar",      0.6165, 0.540, 0.5963, 0.538),
    (8476453, "Nikita Kucherov", 0.6331, 0.575, 0.5938, 0.579),
    (8470613, "Brent Burns",     0.6235, 0.485, 0.6656, 0.494),
    (8477956, "David Pastrnak",  0.5036, 0.511, 0.5039, 0.543),
]

if not _SKIP_SPOTCHECKS:
    log(f"  {'Player':<20}  {'cNFI_QG':>8} {'cRelNFI':>9}  {'cxG_QG':>8} {'cRelxG':>8}  pass?")
    log("  " + "-" * 74)
    ps_qual = ps[ps.position.isin(['F', 'D'])].copy()
    ps_qual['TOI_total_min'] = ps_qual.TOI_total_sec / 60
    n_pass = 0; n_fail = 0
    for pid, name, exp_nfi, exp_rel_nfi, exp_xg, exp_rel_xg in RELQG_EXPECTED:
        rows = ps_qual[ps_qual.player_id == pid]
        qual_rows = rows[rows.TOI_total_min >= 400]
        if len(qual_rows) == 0:
            log(f"  {name:<20}  no qualifying seasons"); n_fail += 1; continue
        cnfi    = qual_rows.NFI_QG_pct.mean()
        cnfirel = qual_rows.RelNFI_QG_pct.mean()
        cxg     = qual_rows.xG_QG_pct.mean()
        cxgrel  = qual_rows.RelxG_QG_pct.mean()
        # ±0.01 tolerance per spec
        pass_nfi    = abs(cnfi - exp_nfi)       <= 0.01
        pass_relnfi = abs(cnfirel - exp_rel_nfi) <= 0.01
        pass_xg     = abs(cxg - exp_xg)         <= 0.01
        pass_relxg  = abs(cxgrel - exp_rel_xg)  <= 0.01
        all_pass = pass_nfi and pass_relnfi and pass_xg and pass_relxg
        flag = "✓" if all_pass else "✗"
        if all_pass: n_pass += 1
        else: n_fail += 1
        log(f"  {name:<20}  {cnfi:>8.4f} {cnfirel:>9.4f}  {cxg:>8.4f} {cxgrel:>8.4f}  {flag}")
    log(f"\n  Spot-check totals: {n_pass}/{len(RELQG_EXPECTED)} pass, {n_fail} fail (±0.01 tolerance)")
    log()

log("Done.")
_log_fh.close()
