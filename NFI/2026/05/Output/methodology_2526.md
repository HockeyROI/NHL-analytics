# May 2026 Comprehensive Build — Methodology

Run date: 2026-05-05T17:44:13
Output dir: `NFI/2026/05/output/`

## Source files

| Path | mtime |
|---|---|
| `Data/nhl_shot_events.csv` | 2026-04-30T10:37:02 |
| `NFI/Output/shots_tagged.csv` | 2026-04-30T10:37:21 |
| `Data/game_ids.csv` | 2026-04-29T13:53:25 |
| `NFI/Output/player_positions.csv` | 2026-04-21T21:18:57 |
| `NFI/Output/fully_adjusted/player_fully_adjusted.csv` | 2026-05-03T23:03:47 |
| `NFI/Geometry_post/Data/shift_data.csv` | 2026-05-05T17:02:08 |
| `NFI/Output/_player_two_way_split_join_cache.pkl` | 2026-05-05T17:22:02 |


## Variant A "broad ES"
`state == 'ES'` from `shots_tagged.csv` (5v5 + 4v4). Empty-net excluded upstream.

## Zone filter
**CNFI + MNFI only.** FNFI excluded.
- FNFI ES Fenwick events discarded (team pipeline, 2025-26): 9,644
- FNFI ES faced shots discarded (goalie pipeline, all seasons): 31,791

## Lookback windows
| Window | Seasons |
|---|---|
| 5y | 2020-21 .. 2024-25 |
| 4y | 2021-22 .. 2024-25 |
| 3y | 2022-23 .. 2024-25 |
| 2y | 2023-24 .. 2024-25 |
| current | 2025-26 only |

## Cohort selection
- Eligibility: ≥20 GP for that team in 2025-26 (per-team GP from cache `games_by_team`).
- Position from `player_positions.csv`. F = C/LW/RW; D = D.
- Sort by 2025-26 team-specific ES TOI (FA `toi_min` for season=20252026, team=X) descending.
- Top 10 = top 6 F + top 4 D ; All 18 = top 12 F + top 6 D ; Bottom 8 = F slots 7-12 + D slots 5-6.
- Sub-cohorts: all18_F = top 12 F ; all18_D = top 6 D.
- Shortfall: partial fill if eligible pool < required size; flagged with cohort_shortfall_flag.

## Traded players
Each (player, team) row independent. Player on multiple teams in 2025-26 with
≥20 GP each appears in both teams' cohorts.

## Player-level metrics
For each (player, window):
```
events_for       = sum CNFI+MNFI Fenwick FOR while on ice
events_against   = sum same AGAINST
es_toi_min       = sum ES TOI minutes
offensive_NFI_60 = events_for / es_toi_min * 60
defensive_NFI_60 = events_against / es_toi_min * 60
pool_NFI_combined = events_for / (events_for + events_against)
qualified        = es_toi_min >= 600
```

## Team aggregations
Per cohort × window:
```
combined_score = TOI-weighted mean(pool_NFI_combined) across qualified cohort players
off_rate       = TOI-weighted mean(offensive_NFI_60) across qualified cohort players
def_rate       = TOI-weighted mean(defensive_NFI_60) across qualified cohort players
```
Weights: 2025-26 team-specific ES TOI (FA `toi_min`).

## Team NFI (2025-26)
Computed directly from `shots_tagged.csv` (no shifts join needed) — uses all
82 games per team.
```
attack_count   = CNFI+MNFI Fenwick ES regular events where shooting_team = team
suppress_count = same where def_team = team
team_nfi_pct = attack / (attack + suppress)
team_attack_per_game = attack / games_played
team_suppress_per_game = suppress / games_played
```

## Goalie GSAx
Mirrors `NFI/scripts/21_goalie_gsax_by_season.py`. CNFI+MNFI only. Per-window
faced/goals/xG/GSAx aggregated from per-season totals. ES TOI prorated from
pooled `player_toi.csv` by faced share (script-21 simplification).

## Ranks
1 = best.
- combined_score, off_rate, GSAx, GSAx_per60: rank descending
- def_rate: rank ascending

## Data vintage notes (this run)
- Shifts data was incomplete on the prior build (last updated 2026-04-16,
  only 813 of 1312 2025-26 regular games). 491 missing games were ingested
  via `NFI/scripts/02_incremental_ingest.py` on 2026-05-05,
  yielding 365,007 new shift rows. Cache rebuilt from scratch.
- Six games (2025021307–2025021312, the very last of the regular season)
  returned 0 shift records from the NHL API and remain absent from the cache.
  All other 2025-26 games are now fully attributed.

## Phase 2 verification
Skipped on this run — the legacy comparison files (`team_nfi_verification_and_attack_suppress.csv`,
`roster_talent_variants_2526.csv`) were intentionally deleted during the
prior run's Phase 3, since the new outputs supersede them. With no legacy
to compare against, no verification gate applies.

## Phase 3
Not executed this run — legacy targets were already deleted previously.
