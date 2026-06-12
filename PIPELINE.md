# Pipeline

This document describes the HockeyROI pipeline: which scripts to run, in what order, what each one produces, and the schema/ordering constraints that have to be respected. It is the operational counterpart to `docs/METHODOLOGY.md` — methodology answers *why*, this answers *how to run it*.

If you are re-running the pipeline from scratch, follow the canonical execution order below. If you are running a single stage, check the "Schema and ordering constraints" section first — several scripts have hard prerequisites that will produce silent or loud failures if violated.

---

## Canonical execution order

The full NFI pipeline runs in this order:

```
1.  python3 NFI/scripts/03_onice_attribution_pillars.py
2.  python3 NFI/scripts/04_corsi_nfi_variants.py
3.  python3 NFI/scripts/06_qoc_qot.py
4.  python3 NFI/scripts/07_finalize_outputs.py
5.  python3 NFI/scripts/build_fa_factors.py
6.  python3 NFI/scripts/fa_linemate_without_me.py
7.  python3 NFI/scripts/stage5_7_finalize.py
8.  python3 NFI/scripts/tnfi_relatives_pp_pk.py
9.  python3 NFI/scripts/rename_and_momentum.py
10. python3 NFI/scripts/top200_article_dataset.py
11. python3 NFI/scripts/update_current_season.py
12. python3 NFI/scripts/build_playoff_data.py
```

Steps 1-10 must run sequentially. Each enforces schema on the prior, and several pickle files passed between scripts are created and consumed in strict order. Steps 11 and 12 are independent of each other and can run in any order or in parallel after step 10 completes.

Steps 1-4 (the numbered-prefix scripts) are rebuilt when `shots_tagged.csv` or upstream zone/state definitions change — they produce the foundation artifacts (`shots_tagged.csv`, `player_counts_by_state_zone.csv`, `team_counts_by_state_zone.csv`, `metrics_team.csv`, `metrics_player.csv`, `team_level_all_metrics.csv`, and the pillar CSVs) that the downstream fully-adjusted pipeline reads. In steady state — daily data updates that don't change methodology — these steps may be skipped if their inputs and outputs are already current.

Upstream of step 1, the shot database build (`NFI/Geometry_post/NF_PY/build_shot_db.py`) must have run at least once and produced the raw shot events database. This is run rarely — typically once per season ingest, not every pipeline execution — and is treated as a precondition rather than a pipeline step.

---

## May 20, 2026 audit and bug fixes

A follow-up to the May 2026 audit (the original is documented in `docs/METHODOLOGY.md`'s "Verification" section). Three independent bugs were found by gut-checking rank stability against an audit-verification CSV that had been built with a different (correct) filter set than the production pipeline.

### Bug 1 — `state == "ES"` conflation (5v5 + 4v4 + 3v3)

The upstream `state` column produced by `03_onice_attribution_pillars.py` previously labeled all even-strength play as `"ES"` — collapsing strict 5v5, 4v4 (coincidental minors), and 3v3 (regular-season OT) into a single label. Every downstream `state == "ES"` filter therefore over-included by ~1.2% of events league-wide. Fixed in `03_onice_attribution_pillars.py`'s `state_from_code` function and the parallel vectorized derivation: `"ES"` now means strict 5v5; 4v4 and 3v3 have their own labels (`"4v4"`, `"3v3"`). All downstream consumers inherit the fix without needing their own code changes — `state == "ES"` now correctly means strict 5v5.

### Bug 2 — Missing `game_type` filter at scripts that read `shots_tagged.csv` directly

`shots_tagged.csv` contains both regular-season and playoff shots because `build_playoff_data.py` needs the playoff data. Any consumer that reads `shots_tagged.csv` for regular-season aggregation must apply its own `game_type == "regular"` filter — there is no upstream filtering at the file level. Fixed at four consumer sites:

- `04_corsi_nfi_variants.py` — filter applied at line 51 (the team-level + player-level aggregator)
- `10_rerun_goalie_pillars_min300.py` — filter applied at the read site
- `21_goalie_gsax_by_season.py` — filter applied at the read site
- `03_onice_attribution_pillars.py`'s vectorized post-loop pass (added in bug-3 fix below)

Also patched: `team_nfi_verification.py` (was missing the filter; benign for 25-26 because no playoff data was ingested at runtime, but would have under-counted any prior-season run).

`fa_linemate_without_me.py` is **already protected** by an independent mechanism — it iterates over a `gids` list pre-filtered to regular games at script start, so playoff shots loaded into its in-memory DataFrame are never visited. No change needed there.

### Bug 3 — Shift-data over-filter on goalie pillars and team counters

`03_onice_attribution_pillars.py`'s per-game loop correctly gates **skater on-ice attribution** by shift-data availability — TOI and on-ice for/against counts require shifts. But the same loop also accumulated **goalie pillar counters** (`gk_faced`, `gk_goals`) and **team-level for/against counters** (`team_for`, `team_ag`, `team_goals_for`, `team_goals_ag`), neither of which depend on shift data. Goalie save-rates and team totals were therefore being under-counted by ~5% in 2024-25 (where `shift_data.csv` was missing 57 of 1312 regular-season games) and ~0.5% in 2025-26.

Fix: moved both workflows out of the gated per-game loop into a vectorized post-loop pass over the `shots` DataFrame, filtered to regular-season games (game_id digits 4-5 == `"02"`). Skater on-ice attribution stays in the gated loop — it has to, no shifts means no attribution.

### Wilson → Poisson CI migration (per-60 rate stats only)

Per-60 rate stats now use exact Poisson (chi-square / Garwood) 95% intervals on the event count, scaled to per-60. Wilson intervals were previously used here in error — Wilson is correct for **proportions** (save percent, conversion rate, NFI%), not for **rate counts over fixed exposure**. Affected scripts (one `rate_ci` helper, replicated across the pipeline): `21_rerun_corsi_with_block_fix.py`, `27_p1b_add_per60.py`, `31_pp_pk_pillars.py`. Downstream consumers picked up the corrected helper in `tnfi_relatives_pp_pk.py`, `rename_and_momentum.py`, `28_publication_outputs.py`, and `33_age_filter_publication.py`. Wilson intervals are unchanged in scripts that report proportions; they were correct there.

### New RelNFI 95% CIs

`tnfi_relatives_pp_pk.py` now emits season-level 95% CIs on `RelNFI_F_pct`, `RelNFI_A_pct`, and `RelNFI_pct` via Poisson-differential SE, using the empirical partial correlation `r(cf_cm, ca_cm | log toi)` as the covariance proxy for the combined column. Career-pool CIs are computed in `28_publication_outputs.py` and `33_age_filter_publication.py` via TOI-weighted point estimates with variance-pooled SE. New columns appear in `player_fully_adjusted.csv`, `top30_RelNFI_*.csv`, `publication_forwards_top100.csv`, and `publication_D_top100.csv`.

### Reconciliation gap (intentional)

After bug 3's fix, team and goalie counts use all regular-season games while player on-ice counts still require shift data. This introduces an intentional ~5% gap (24-25 specifically) between team totals (`team_counts_by_state_zone.csv`) and the same events summed from player on-ice counts (`player_counts_by_state_zone.csv`). Goalie save-rates use all regular-season games; player on-ice metrics use only games with shift data. This is the right tradeoff — player attribution requires shifts and can't be relaxed without losing correctness; team and goalie counts don't depend on shifts and shouldn't be artificially restricted.

### Known in-progress

`update_current_season.py` uses an approximate CNFI+MNFI zone definition (`zone == "O"` AND `|y| <= 22` AND `|x| >= 65`) that drifts from the canonical bounds (CNFI: x∈[74,89], |y|≤9; MNFI: x∈[55,73], |y|≤15). The script's docstring acknowledges this as a real-time approximation in the absence of `shots_tagged.csv`. Being fixed in a separate Streamlit-chat workstream. When the daily-update GitHub Action is re-enabled, this drift will reassert; until then, `current_season_player_fully_adjusted.csv` is regenerated by the full pipeline using canonical bounds.

---

## Script-by-script reference

### `NFI/Geometry_post/NF_PY/build_shot_db.py`

**Purpose:** Pulls raw NHL play-by-play and constructs the master shot events database used by the entire NFI pipeline.

**Inputs:** NHL API at `https://api-web.nhle.com/v1/`, cached locally.

**Outputs:** `NFI/Geometry_post/Data/shift_data.csv` (the large 133MB shot/shift events file) and supporting datasets.

**When to run:** Once per season ingest, or whenever new games need to be added to the historical shot record. Not part of the per-update pipeline.

**Note:** This script is the foundation of every NFI metric. The `Geometry_post/` folder name dates from when the geometry methodology was being developed as a post draft; the contents are now load-bearing production code that other parts of the pipeline depend on.

### `NFI/scripts/build_fa_factors.py`

**Purpose:** Computes the empirical Corsi and Fenwick zone-adjustment factors and writes them along with the locked Tulsky factor for NFI to a shared JSON.

**Reads:** Shot database from `build_shot_db.py`. Reads decision-tree intermediate pickles (`/tmp/dt_team_zone_agg.pkl`, `/tmp/s4_ppdf.pkl`) when available.

**Writes:** `/tmp/fa_factors.json` containing the three per-metric factors. Current state:

```json
{
  "Corsi": 0.13144756041078226,
  "Fenwick": 0.11911296378280883,
  "NFI": 0.035
}
```

**Important:** The NFI factor is hardcoded to 0.035 (Tulsky's published value) at line 42, overriding any empirical computation. This was set during the May 3, 2026 factor swap. See `docs/METHODOLOGY.md` for the reasoning behind the swap.

### `NFI/scripts/fa_linemate_without_me.py`

**Purpose:** Builds the canonical fully-adjusted player metrics (`player_fully_adjusted.csv`) using a linemate-without-me adjustment that excludes the focal player from their own linemate context. This is the framework's canonical FA build.

**Reads:** `/tmp/fa_factors.json`, `/tmp/s4_ppdf.pkl`.

**Writes:** All files in `NFI/Output/fully_adjusted/`.

**Replaces:** The earlier `NFI/scripts/fully_adjusted.py`, which used a naive linemate adjustment. **`fully_adjusted.py` is deprecated** and carries a deprecation header. Do not include it in regeneration sequences. Both scripts produce the same output filenames; running the deprecated one before this one is wasted compute and produces methodologically-incorrect intermediate values.

### `NFI/scripts/stage5_7_finalize.py`

**Purpose:** Final-stage player rating computation and display label formatting. Produces `stage5_r2_summary.csv` and `stage7_annual_ratings.csv` in `NFI/Output/zone_adjustment/complete_decision_tree/`.

**Note on display labels:** The script formats factor labels for output display. Labels currently say "ZA empirical" for V5/V1b rows even though the factor is now Tulsky's published value (3.5pp), not empirical. This is a stale display label that does not affect computation. Optional follow-up: change "ZA empirical" → "ZA Tulsky" for V5/V1b rows. Low priority.

### `NFI/scripts/tnfi_relatives_pp_pk.py`

**Purpose:** Computes Rel-NFI columns (RelNFI_F_pct, RelNFI_A_pct, RelNFI_pct) which compare a player's NFI to their team's NFI without the player on the ice. These are the zone-adjustment-invariant individual metrics — useful as a primary individual ranking because the factor choice (Tulsky vs empirical) doesn't affect them.

**Reads:** Output of `fa_linemate_without_me.py`.

**Writes:** Adds Rel-NFI columns to `player_fully_adjusted.csv`.

**Schema guard:** Line 50 of this script self-validates input and raises `RuntimeError` if it sees post-rename column names (i.e., if `rename_and_momentum.py` has already run). This is a deliberate guardrail to enforce the `fa_linemate → tnfi → rename` ordering. If you see a RuntimeError mentioning post-rename schema, run `fa_linemate_without_me.py` first to regenerate the input.

### `NFI/scripts/rename_and_momentum.py`

**Purpose:** Renames internal pipeline column names to their final public-facing schema and computes momentum columns (`NFI_pct_3A_MOM`, `NFI_pct_3A_MOM_3yr`, `NFI_MOM_consistency`, `CF_pct_3A_MOM`, `FF_pct_3A_MOM`).

**Critical column renames performed by this script:**
- `ZA_NFI_emp` → `NFI_pct_ZA`
- `FA_NFI_emp` → `NFI_pct_3A`

These names appear pre-rename throughout the early pipeline; downstream consumers expect the post-rename names. Confusion about column names between stages is a common source of `KeyError` issues — see "Column naming by stage" below.

**Must run after** `tnfi_relatives_pp_pk.py` (the schema guard at `tnfi_relatives_pp_pk.py:50` enforces this).

### `NFI/scripts/top200_article_dataset.py`

**Purpose:** Builds the curated top-200-player dataset used in publication artifacts. Joins multi-year derived metrics (e.g., NFI_pct_3A_MOM_3yr, NFI_MOM_consistency) onto the canonical player file by player_id.

**Convention:** When downstream consumers need multi-year derived metrics on the current-season file, they should join from the pooled `player_fully_adjusted.csv` rather than expecting those metrics to already be present on `current_season_player_fully_adjusted.csv`. See "Multi-year metrics on current-season file" below.

### `NFI/scripts/update_current_season.py`

**Purpose:** Performs a game-by-game rebuild of the current season-in-progress, producing `NFI/Output/fully_adjusted/current_season_player_fully_adjusted.csv` and the parallel team file. Independent of the steps 2-6 chain — this script writes its own current-season slice.

**Reads:** Game-by-game NHL API data plus shift data.

**Writes:** Live current-season player and team files.

**Important:** This script writes the current-season file with only the columns it produces — single-season metrics, no multi-year momentum columns. The pre-Path-C version of the pipeline had `rename_and_momentum.py` *also* writing to this file with multi-year additions, which produced an accidental column union depending on which script ran last.

After Path C, the canonical interpretation is: **the live-data version written by `update_current_season.py` is canonical**. It contains 870 rows (one per current-season player above the TOI threshold) with single-season columns only. Multi-year metrics are joined from the pooled `player_fully_adjusted.csv` by `player_id` when needed.

**Expected attrition:** `update_current_season.py` skips approximately 6 of every 1,312 current-season games due to upstream NHL API gaps (missing PBP or missing shift data for specific games). This is a ~0.5% attrition rate and is normal warn-but-continue behavior. No action required if the script reports skipped games.

### `NFI/scripts/build_playoff_data.py`

**Purpose:** Builds the playoff-specific player and team metrics, producing `NFI/Output/fully_adjusted/player_fully_adjusted_playoffs.csv` and parallel team files in `Zones/output/playoffs/`.

**Reads:** Game-by-game playoff PBP and shift data.

**Writes:** Playoff player file and zone-time playoff files.

**Note:** Contains the Fenwick filter for NFI zone determination at lines 101-108 — this is the same audit fix applied to `decision_tree_stage4.py`, `decision_tree_stage123.py`, and `factor_comparison_5metrics.py`. The filter ensures NFI zone flags only consider Fenwick events (excluding blocked shots, whose coordinates may not be in the shooter's reference frame). Without this filter, the playoff NFI metric would be corrupted by the blocked-shot coordinate inconsistencies described in `docs/METHODOLOGY.md`.

---

## Schema and ordering constraints

### Column naming by pipeline stage

The same column changes names as it moves through the pipeline. This is the single most common source of `KeyError` failures when running scripts standalone. Reference table:

| After running… | NFI%_ZA column name | NFI%_3A column name | Rel-NFI columns | Multi-year momentum |
|---|---|---|---|---|
| `fa_linemate_without_me.py` | `ZA_NFI_emp` | `FA_NFI_emp` | not present | not present |
| `+ tnfi_relatives_pp_pk.py` | `ZA_NFI_emp` (still) | `FA_NFI_emp` (still) | `RelNFI_F_pct`, `RelNFI_A_pct`, `RelNFI_pct` | not present |
| `+ rename_and_momentum.py` | renamed → `NFI_pct_ZA` | renamed → `NFI_pct_3A` | (kept) | `NFI_pct_3A_MOM`, `NFI_pct_3A_MOM_3yr`, `NFI_MOM_consistency`, `CF_pct_3A_MOM`, `FF_pct_3A_MOM` |
| `update_current_season.py` (independent path) | writes `NFI_pct_ZA` directly | writes `NFI_pct_3A` directly | writes Rel columns directly | only `NFI_pct_3A_MOM` (single-season) |

If a script errors on a column name, the most likely cause is reading a file from a stage where that column doesn't yet exist or hasn't yet been renamed.

### Multi-year metrics on current-season file

`current_season_player_fully_adjusted.csv` is produced by `update_current_season.py` with single-season columns only. It does not contain `NFI_pct_3A_MOM_3yr`, `NFI_MOM_consistency`, or other multi-year derived metrics, even though the pooled file `player_fully_adjusted.csv` does contain them.

If a downstream consumer needs multi-year metrics on the current-season file, the canonical pattern is to **join from the pooled file by `player_id`**. Do not expect those metrics to be present after running `update_current_season.py`. This is the existing pattern used by `top200_article_dataset.py`.

### The schema guard at `tnfi_relatives_pp_pk.py:50`

This script reads `player_fully_adjusted.csv` and validates that the columns are in pre-rename state (`ZA_NFI_emp`, `FA_NFI_emp`). If it finds post-rename columns (`NFI_pct_ZA`, `NFI_pct_3A`), it raises a `RuntimeError`. This catches accidental out-of-order pipeline execution where someone runs `rename_and_momentum.py` first.

Recovery: if you hit this error, re-run `fa_linemate_without_me.py` to regenerate the file with pre-rename schema, then continue from step 4.

### The pipeline pickle dependencies

Several `/tmp/` pickle files are passed between scripts:

- `/tmp/fa_factors.json` — written by `build_fa_factors.py`, read by `fa_linemate_without_me.py` and `update_current_season.py`.
- `/tmp/s4_ppdf.pkl` — written by `decision_tree_stage4.py` (legacy but still active for this purpose), read by `fa_linemate_without_me.py` and `build_fa_factors.py`.
- `/tmp/dt_team_zone_agg.pkl` — written by `decision_tree_stage123.py`, read by `build_fa_factors.py`.

These files live in `/tmp/` rather than the repo because they are intermediate artifacts that regenerate each pipeline run. They are not committed to git. If they are missing, the pipeline will produce `FileNotFoundError` at the relevant step; rerun the upstream producer.

---

## TZI / Zones pipeline

The Zone Impact metrics (DZI, NZI, OZI) are computed by `Zones/scripts/compute_iozc_iozl_dozi.py` and related scripts. The TZI pipeline is independent of the NFI pipeline; the two run on the same underlying shot/shift database but produce separate output files.

Zones outputs land in:
- `Zones/output/` — main TZI output files
- `Zones/output/playoffs/` — playoff-specific TZI files
- `Zones/adjusted_rankings/` — sorted ranking files

See `Zones/README.md` for the Zone Impact subsystem details.

---

## Referee Cache regeneration

The `Referees/Ref Cache/` directory holds approximately 7,872 cached NHL API responses (per-game play-by-play and referee assignment JSONs). This cache is gitignored — it does not ship with the repo and is regenerated locally by `pull_all_teams_penalties.py` when needed.

If you clone the repo fresh and want to run referee analyses, you will need to repopulate the cache by running the referee data-pull scripts in `Referees/`. Cache repopulation may take significant time depending on how many games need pulling.

---

## Snapshot and backup conventions

The pipeline preserves several snapshot/backup file conventions that are gitignored but kept locally:

- `*.preXxxFix.csv` — audit-cycle snapshots (May 2026 audit)
- `*.preCleanup.csv` — audit-cycle snapshots
- `*.pre_phase2.bak`, `*.pre_phase3.bak`, `*.prePhase3.json.bak2` — Path C factor swap backups
- `**/_legacy/` — directories holding deprecated/orphaned files from prior cleanups

These conventions exist so that any audit-cycle change can be rolled back if needed. They are gitignored to keep the repo clean while preserving local rollback capability.

---

## Re-running the pipeline at a later date

If you re-run the pipeline at a later date (e.g., months after the methodology version stamp in `docs/METHODOLOGY.md`), expect:

1. **Cohort sizes increase** as more games complete. The complete-seasons cohort grows by one season every June; the current cohort updates daily during the season.
2. **Spot-check values shift** for the same player. The locked values in `docs/METHODOLOGY.md` are anchored to the version stamp; later reruns will produce different values for the same player due to additional games.
3. **Methodology stays consistent** unless a methodology change is recorded in `docs/METHODOLOGY.md` and the version stamp is updated. Pipeline behavior should not silently change between runs without a corresponding methodology change.

If you re-run the pipeline and your spot-check values land significantly different from the methodology document's locked values *at the same data snapshot*, something has gone wrong. If you re-run at a later snapshot, drift is expected.

---

## Operational gotchas

### Output/output and Sizes/sizes directory casing (macOS vs Linux)

Several scripts reference output directories using a different letter case than the on-disk directory name. macOS's default case-insensitive filesystem resolves this transparently; a case-sensitive filesystem (most Linux CI runners) will not.

- On disk the directory is `NFI/Output/` (capital O); scripts reference `NFI/output/` (lowercase) in ~147 places vs ~9 using `NFI/Output/`.
- On disk the directory is `Goalies/Sizes/` (capital S); scripts reference `Goalies/sizes/` (lowercase) in 4 places vs 0 using `Goalies/Sizes/`.

Before re-enabling the GitHub Action or running any of these scripts on a case-sensitive filesystem, normalize every reference to match the on-disk case (or rename the directories). Until then, these scripts run correctly only on macOS. (Counts are occurrence counts from `grep -rho … --include='*.py'`, excluding worktrees.)

---

*This document reflects pipeline behavior as of the May 20, 2026 audit + bug fix (building on the May 2026 audit and the May 3, 2026 zone-adjustment factor swap). Future changes that affect ordering, schema, or column names will update this document alongside `docs/METHODOLOGY.md`.*
