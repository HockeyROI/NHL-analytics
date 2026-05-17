# FLA Retro Validation — Top-10 Two-Way Cohort, Both Cup Years

Run date: 2026-05-06T13:18:48
Output: `NFI/2026/05/output/fla_retro_validation_2526.csv`

## Question
Did FLA build their Cup-winning rosters around two-way top-end players, more
so than peer playoff teams in BOTH 2023-24 and 2024-25?

## Source files

| Path | mtime |
|---|---|
| `Data/game_ids.csv` | 2026-04-29T13:53:25 |
| `NFI/Output/player_positions.csv` | 2026-04-21T21:18:57 |
| `NFI/Output/fully_adjusted/player_fully_adjusted.csv` | 2026-05-03T23:03:47 |
| `NFI/Output/_player_two_way_split_join_cache.pkl` | 2026-05-05T17:22:02 |

## Cohort selection

For each (playoff team × championship season):
- Top 6 forwards + top 4 defensemen by **championship-season ES TOI for that team**
  (`toi_min` from `player_fully_adjusted.csv` for season=championship, team=team)
- Eligibility: ≥20 GP for that team in the championship season
  (per-team GP from `cache.games_by_team[pid][season]`)
- Position from `player_positions.csv` (F = C/LW/RW; D = D)

## Pre-season pool (no contamination)

For each player in each Top-10 cohort, compute pre-season-only metrics:
- For 2023-24 cohort: pool 2020-21 + 2021-22 + 2022-23
- For 2024-25 cohort: pool 2020-21 + 2021-22 + 2022-23 + 2023-24

```
events_for_pre   = sum CNFI+MNFI Fenwick FOR while on ice in pre-pool
events_against_pre = sum same AGAINST in pre-pool
es_toi_pre_min   = sum ES TOI in pre-pool
offensive_NFI_60_pre = events_for_pre / es_toi_pre_min * 60
defensive_NFI_60_pre = events_against_pre / es_toi_pre_min * 60
qualified_pre  = es_toi_pre_min >= 600
```

Players who do not qualify (insufficient pre-season TOI — typically rookies or
returnees from injury) are flagged but not assigned percentiles.

## Percentile ranking (league-wide, within-position)

For each cup year, percentile-rank ALL eligible skaters (league-wide, not just
playoff teams), separately for F and D:
- `off_pct_pre` = percentile rank of `offensive_NFI_60_pre` (higher = better)
- `def_pct_pre` = percentile rank of `defensive_NFI_60_pre` ascending (lower = better suppressor)
- `two_way_min_pre` = min(off_pct_pre, def_pct_pre)

A player is "two-way" if `two_way_min_pre >= 65` and "elite two-way" if
`>= 75`. The min(off, def) requires being above-floor on **both** dimensions.

## Verdict logic
- STRONG validation: FLA top-3 of 16 playoff teams in BOTH cup years
- MODERATE validation: FLA top-5 of 16 in BOTH years
- WEAK consistent: FLA above-median in BOTH years
- INCONCLUSIVE: above-median in only one year
- NOT VALIDATED: below-median in either year
