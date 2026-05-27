# Zones / TZI Framework — Tech Debt

Items identified during audit, not blocking publication, deferred for later cleanup.

## RESOLVED — May 2026

### 2026-05-27 — NZI_L / DZI_L / TNZI_L removed from active framework
The team-level OLS regression that produced `r_L` for these metrics could not be identified at n=32 with multicollinear `team_mean_raw` / `team_mean_IOZL` predictors. Bootstrap analysis showed sign flips and large variance; Ridge with LOO-CV did not stabilise the estimate (see chunk B11 of the May 2026 audit). The broken methodology was removed from the production path rather than re-published with a band-aid:

- **Active script** `compute_iozc_iozl_dozi.py` now computes the IOZL regression only for OZI. For DZI / NZI / TNZI the `single_L` / `both` model entries are explicitly `None`, so `adj_raw[*]["L"]` and `adj_raw[*]["CL"]` populate as `None` for those metrics.
- **Schema split**: `OUT_COLS_OZI` keeps the full `_C / _L / _CL` quad for OZI; `OUT_COLS_RAW` (used by NZI / DZI / TNZI files) drops all `_L` / `_CL` columns AND drops the `IOZL` column from those files (no consumer left).
- **OZI_L** stays in production — it works as a modifier-scale adjustment (`r_L ≈ 1.1`, eye-tests cleanly, R² on team-level fit ≈ 0.99 in our regression of `_L` on raw + IOZL).
- **DTNZI flags**: kept. They are computed from raw per-season `norm01` values, not from `_L` adjusted scores — so they were never affected by the broken methodology. Added a mean-reversion caveat comment in the code (r ≈ −0.4 YoY between consecutive deltas).
- **top20_by_metric.csv**: kept. Its writer reads the raw metric column from each `*_adjusted_*.csv` file, so it's unaffected by removing `_L`.
- **tnzi_winning_correlation.csv**: **orphaned**. Its construction depended on the broken `r_L` for TNZI. Writer call and the whole correlation-computation block were removed from `compute_iozc_iozl_dozi.py`. The function definitions (`write_corr_csv`, `print_tnzi_corr_table`) are left in place as dead code for now; they can be deleted in a follow-up.
- **Orphan quarantine**: `Zones/_orphaned_broken_L_2026_05/` contains the pre-removal CSVs (`{nzi,dzi,tnzi}_adjusted_*.csv` with the old `_L`/`_CL` columns, `dtnzi_*.csv` from the same generation, `top20_by_metric.csv` snapshot, `tnzi_winning_correlation.csv`) plus a snapshot of the producer script as `compute_iozc_iozl_dozi.PRE_L_REMOVAL.py`. See that folder's README for the audit trail.
- **OZI files** were not orphaned. They retained the working OZI_L methodology and were untouched by the cleanup other than schema-level reorg.

If reviving:
- Player-level regression (n ≈ 2,000) instead of team-level (n = 32) would identify `r_L` reliably.
- Or hardcode a small `r_L` (0.5–1.5 range) per analytics convention and document the choice.
- Or drop the linemate adjustment entirely and lean on Rel-NZI as the on/off teammate-effect view.

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
