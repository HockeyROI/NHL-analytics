# NFI — Net-Front Impact

The flagship framework. NFI redefines high-danger scoring chances based on the actual conversion geometry, narrowing the conventional high-danger trapezoid to the regions where shots genuinely behave like dangerous chances.

See `../docs/METHODOLOGY.md` for the full methodology rationale and `../PIPELINE.md` for execution order.

## What's in this folder

```
scripts/         Canonical NFI pipeline (production)
Output/          Derived metrics produced by the pipeline
Geometry_post/   Production NFI infrastructure (see note below)
```

## Geometry_post — note on the name

Despite the folder name, `Geometry_post/` holds load-bearing production code, not post drafts. Contents:

- `Geometry_post/NF_PY/` — production Python scripts including `build_shot_db.py` (the master shot-events database builder) and related shot/shift/acquisition analysis scripts.
- `Geometry_post/Data/` — production datasets including `shift_data.csv` (a large shift events file referenced by the decision-tree scripts; gitignored due to size).
- `Geometry_post/Images/` — chart files from the original Geometry post; the post itself was published in April 2026 and the working folder was archived to OneDrive afterward.

The folder name dates from when the Geometry methodology was being developed as a single post; the contents are now distributed production code that other parts of the pipeline depend on. Renaming would require updating hardcoded paths in multiple scripts and is deferred.

## scripts/

The canonical NFI pipeline. Key scripts (see `../PIPELINE.md` for the full execution order):

- `build_fa_factors.py` — computes/locks zone-adjustment factors
- `fa_linemate_without_me.py` — canonical fully-adjusted player metrics
- `stage5_7_finalize.py` — final-stage rating computation
- `tnfi_relatives_pp_pk.py` — Rel-NFI columns (zone-adjustment-invariant)
- `rename_and_momentum.py` — final schema and momentum metrics
- `top200_article_dataset.py` — curated top-200 dataset for publication
- `update_current_season.py` — game-by-game current-season rebuild
- `build_playoff_data.py` — playoff-specific NFI metrics

`fully_adjusted.py` is **deprecated** in favor of `fa_linemate_without_me.py` and is preserved with a deprecation header but should not be run.

## Output/

Derived NFI metrics at player, team, and goalie level. Notable files:

- `fully_adjusted/player_fully_adjusted.csv` — pooled multi-season player metrics (canonical)
- `fully_adjusted/current_season_player_fully_adjusted.csv` — current season-in-progress slice (live data)
- `fully_adjusted/team_fully_adjusted.csv` — team-level NFI
- `fully_adjusted/player_fully_adjusted_playoffs.csv` — playoff metrics
- `fully_adjusted/top200_article_dataset.csv` — curated top-200 set

Locked spot-check values for the player file are documented in `../docs/METHODOLOGY.md`.

## What's not here

- Historical exploratory R-squared analyses (horse-race scripts and outputs) were retired and untracked from this folder. See `../docs/METHODOLOGY.md` "On exploratory R-squared work" for context.
- Three-pillar team-construction model scripts and outputs were retired after the underlying claims could not be supported by holdout testing.
- Decision-tree variant scripts that derived the empirical zone-adjustment factor are retained for reference only. The empirical NFI factor (0.1071) they produced was evaluated against Tulsky's 0.035 and added no significant predictive value (ΔR² = +0.005, p = 0.187), so it was rejected — not merely superseded — and the framework adopted Tulsky's 0.035. See `../docs/METHODOLOGY.md` for the factor-change rationale.

---

*The NFI subsystem represents the core analytical work of HockeyROI. Methodology decisions, validation, and reasoning live in `../docs/METHODOLOGY.md`; this README orients you to the folder contents.*
