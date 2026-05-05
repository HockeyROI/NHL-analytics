# Zones — Transitional Zone Impact (TZI)

Measures how much a player can tilt the ice based on where they start their shift. TZI breaks the question into three peer metrics — DZI, NZI, OZI — rather than collapsing the three deployment contexts into a single number.

See `../METHODOLOGY.md` for the framework rationale and what TZI does and does not claim.

## The three metrics

- **DZI — Defensive Zone Impact.** A player's net ice-tilt impact during shifts that begin in the defensive zone. Captures whether the player escapes their own zone cleanly and turns DZ starts into shot-attempt advantages.
- **NZI — Neutral Zone Impact.** Net ice-tilt impact during shifts that begin in the neutral zone. Captures transition play and zone-entry effectiveness.
- **OZI — Offensive Zone Impact.** Net ice-tilt impact during shifts that begin in the offensive zone. Captures whether OZ starts get converted to sustained pressure.

Linemate-adjusted variants (DZI_L, NZI_L, OZI_L) and competition-adjusted variants are computed where the underlying linemate and competition data are available.

## What's in this folder

```
scripts/             Production scripts that compute TZI metrics
output/              Derived TZI metrics at player and team level
output/playoffs/     Playoff-specific TZI files
adjusted_rankings/   Sorted ranking files
raw/                 Raw shift and play-by-play data (gitignored)
```

## scripts/

The TZI pipeline runs independently of the NFI pipeline. Both pipelines read from the same underlying shot/shift database but produce separate output files.

Key scripts:

- `compute_iozc_iozl_dozi.py` — computes the underlying IOZC, IOZL, and DOZI metrics that feed into the TZI rollup
- `compute_oze_dze_nze.py` — per-player OZE/DZE/NZE computation
- `compute_pqr_roc_rol.py` — PQR/ROC/ROL pipeline (regular and playoff variants)
- `compute_zone_variations.py` — sweeps zone-factor variations for methodology testing

Some scripts in this folder also compute correlation-to-winning side outputs as part of historical exploratory work. Those output files are gitignored and not part of the framework's current public claims; see `../METHODOLOGY.md` "On exploratory R-squared work" for context.

## output/

Derived TZI metrics. Notable files:

- Player-level TZI files (DZI, NZI, OZI by season and pooled)
- Team-level TZI rollups
- `playoffs/` — playoff-specific TZI files including linemate-adjusted variants

## adjusted_rankings/

Sorted ranking files for forwards and defensemen by each TZI variant. Used as inputs to the Streamlit app and downstream content.

## What TZI does not claim

TZI is descriptive. It characterizes how players tilt the ice given their deployment; it does not claim any particular relationship to team winning. Exploratory R-squared analyses against standings points were run during development and did not produce findings that would support predictive claims. The framework stands as a player-evaluation descriptive tool, not a predictive one.

---

*See `../METHODOLOGY.md` for the full methodology context, including why three peer metrics rather than a single combined number, and `../PIPELINE.md` for execution details.*
