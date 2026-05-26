# Zones / TZI Framework — Tech Debt

Items identified during audit, not blocking publication, deferred for later cleanup.

## FIXED

### 2026-05-26 — Path bug across 4 producer scripts
The May 4 "Repository reorganization for public release" reorg moved data files into `Zones/output/` and `Zones/raw/` but did not update path constants in producer scripts that resolve paths from `__file__.parent`. Four scripts were broken: `compute_iozc_iozl_dozi.py`, `compute_oze_dze_nze.py`, `compute_zone_variations.py`, `benchmark_tnzi_vs_others.py`. All four fixed by adopting the `HERE / ZONES / ROOT` three-level path convention already used in `compute_rel_tnzi.py` and `compute_tozi_tdzi.py`. All 24 path constants now resolve correctly.

### 2026-05-26 — Stale parallel ranking files (MEDIUM-1 from audit)
After path bug fix, re-ran `compute_iozc_iozl_dozi.py`. All 11 outputs in `Zones/adjusted_rankings/` now refreshed with consistent timestamps and current `DTNZI_*` column naming. Zero drift in top 10 NZI raw vs. stale data — producer logic deterministic, foundation files current.

### 2026-05-26 — Orphan top20_by_metric.csv producer restored
The May 4 reorg dropped the producer script for `top20_by_metric.csv` while leaving the file in `Zones/adjusted_rankings/`. The file was referenced as a planned Streamlit input in `Streamlit_Build_Handoff_UPDATED.md` but no consumer was ever built. Restored by adding `write_top20_by_metric()` to `compute_iozc_iozl_dozi.py` (called after `write_corr_csv()` at end of `main()`). File now self-refreshes every producer run. Schema: metric, position_group, rank, player_name, team, pos, GP, raw_score. 240 rows = 6 metrics × 2 position groups × 20.

## LOW severity

### Name+team joins instead of player_id
`tnzi_adjusted_forwards.csv`, `tnzi_adjusted_defense.csv`, `dtnzi_forwards.csv`, and `dtnzi_defense.csv` currently lack `player_id` and are joined to other files via `name+team`. No current bugs detected (audit Chunk 1, A3 confirmed zero duplicates/conflicts), but fragile to future NHL API name spelling or encoding changes. Refactor to carry `player_id` through `compute_iozc_iozl_dozi.py` rollup when convenient.

### DTNZI column naming inconsistency
`dtnzi_forwards.csv` and `dtnzi_defense.csv` use unprefixed column names (`OZI_flag`, `DZI_flag`, etc.) where docs reference `DTNZI_OZI_flag`. Values are correct; labels are inconsistent with documentation. Rename columns or update docs to match.
