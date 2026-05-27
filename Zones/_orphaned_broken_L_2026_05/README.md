# Orphaned: Broken Linemate Adjustment (NZI_L, DZI_L, TNZI_L)

This folder contains artifacts from a linemate adjustment methodology that was removed from the active framework in May 2026.

## What's here

- Code that computed NZI_L, DZI_L, TNZI_L via OLS team-level regression
- Output files containing _L columns based on that methodology
- Trajectory flag files (DTNZI) and correlation files that depend on the broken _L values

## Why these were moved

The team-level OLS regression that recovered r_L produces dominator-scale coefficients (~20x raw signal) for NZI and DZI due to:
- n=32 (one observation per team)
- High multicollinearity between team_mean_raw and team_mean_IOZL
- Weak R squared (0.04-0.37 across the four metrics)

Ridge regression with leave-one-out cross-validation was tried as a fix; bootstrap analysis showed the data does not identify r_L for any metric at this n. Sign flips occurred for NZI under Ridge. The conclusion was: linemate adjustment cannot be recovered from this dataset's team-level standings regression.

## What's still in production

- Raw NZI, OZI, DZI metrics — three peer zone-time scores
- OZI_L — the only linemate-adjusted variant that works at modifier scale
- Rel-NZI — true on-off differential, separate construction, unaffected

## If reviving

A future revisit could:
- Use player-level regression (n=~2000 vs n=32) to identify r_L
- Hardcode r_L based on analytics convention (0.5-2.0 range)
- Use Bayesian shrinkage with a strong prior
- Drop the regression approach entirely and use a different linemate model

See main Zones/TECH_DEBT.md for the full audit trail.
