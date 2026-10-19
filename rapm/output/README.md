# RAPM dashboard exports

Published RAPM data for the Streamlit dashboard. Built by `rapm/build_rapm_exports.py`,
which reads the final RAPM study outputs **read-only** and changes nothing under
`Zones/2026-09/`.

## File

`player_rapm_5v5.csv` — one row per qualified skater (**≥ 500 pooled 5v5 minutes**;
812 skaters: 517 F, 295 D).

## Seasons

Three regular seasons pooled: **2023-24, 2024-25, 2025-26** (labelled
"2023-24 to 2025-26"). 5v5 only.

## Three families (each with net, Attack, Suppress)

| Column prefix | Metric | Source fit |
|---|---|---|
| `nfi_rapm` | **NFI RAPM** — on-ice effect on NFI-zone (CNFI ∪ MNFI) Fenwick rate | `Zones/2026-09/rapm/doorstep` |
| `iti` | **ITI** (Ice Tilt Impact) — on-ice effect on 5v5 offensive-zone time | `Zones/2026-09/iti/player_iti.csv` |
| `cg_rapm` | **cG RAPM** — on-ice effect on Chaos Created Goals rate | `Zones/2026-09/rapm/cg` |

For each family there are three measures:
- `<prefix>` = **net**, `<prefix>_a` = **Attack (A)**, `<prefix>_s` = **Suppress (S)**.
- Each carries `_lo`, `_hi` (95% interval) and `_clears_zero` (interval excludes 0).

## Conventions

- **Positive = better on every column.** Attack is the offence coefficient; Suppress
  is the negated defence coefficient (the defence coefficient is the player's effect
  on the *opponent's* rate, where lower is better, so it is sign-flipped).
- **Net = A + S, with its 95% interval summed within each of the 200 game bootstrap
  resamples** — not by adding the two separate intervals. (Same method as
  `iti/finalize_iti.py`; see `Zones/2026-09/rapm/docs_draft/methodology_rapm_DRAFT.md`.)
- Intervals are bootstrap percentiles (2.5 / 97.5) over 200 game resamples with α
  fixed at its cross-validated value. Bootstrap is the only interval method for RAPM
  coefficients (Wilson is for true proportions, Poisson for count rates).
- Units: NFI RAPM and cG RAPM are per-60 rates; ITI is offensive-zone seconds per 60
  minutes of zone-assigned 5v5 time.

## Collinearity flag

`collinearity_flag` is true when ≥ 65% of a skater's pooled 5v5 TOI is with one
same-team skater (`top_linemate`, `share_with_top_linemate`). 61 skaters flagged
(42 F, 19 D). A flagged estimate is **less certain**, not lower — ridge splits the
shared credit between the pair, so one of a pair can sit near zero even when the pair
drives play. Canonical source: `Zones/2026-09/iti/collinearity_flags.csv`.

## Not published

Special-teams (PP/PK) RAPM is **not** published: power-play units share too much ice
for the model to separate players reliably. HD-Fenwick, NFI-zone G−xG, NZ-seconds and
the shorthanded reverse model are also study-only (too little signal or not
directional). None of these appear on the dashboard.
