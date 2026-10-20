# RAPM monthly refresh — proposal (not yet automated)

How the published RAPM exports (`rapm/output/player_rapm_5v5.csv`) would be
refreshed monthly on a **rolling three-season window**. This is a plan only — no
automation is built, and the monthly run is a **local** job (the raw shift/PBP
caches and the study fits are not on CI).

## Window

Pool the **three most recent regular seasons**. Mid-season, keep the two most
recent completed seasons plus the live one; when a new season completes, drop the
oldest. Only the `--seasons` argument and the "seasons" label change — the method
is identical. (Current window: 2023-24, 2024-25, 2025-26.)

## Steps (all read the existing caches; no re-fetch of games already cached)

| # | Script | Writes | Notes |
|---|---|---|---|
| 1 | `Zones/2026-09/rapm/fetch_shifts.py` (+ the existing PBP fetch) | `Zones/raw/pbp`, shift cache | only new games |
| 2 | `Zones/2026-09/rapm/build_stints.py` | `rapm/stints.parquet` | 5v5 stints for the window |
| 3 | `Zones/2026-09/rapm/build_matrix.py` | `rapm/matrix/` | design matrix + `players.csv` |
| 4 | `Zones/2026-09/rapm/fit_rapm.py --targets doorstep,cg --boot 200` | `doorstep/`, `cg/` (`beta.npy`, `boot_betas.npy`, `player_rapm*.csv`) | the two NFI/cG fits + bootstrap |
| 5 | `Zones/2026-09/iti/build_iti.py --seasons <window> --anchor keep --boot 200` then `iti/finalize_iti.py` | `iti/player_iti.csv` | ITI (anchor kept) + net CIs |
| 6 | `Zones/2026-09/iti/collinearity_flags.py` | `iti/collinearity_flags.csv` | 65% linemate-share flag |
| 7 | `rapm/build_rapm_exports.py` | `rapm/output/player_rapm_5v5.csv`, `README.md` | the published export (read-only on steps 1–6) |

Then commit `rapm/output/` and the updated `Streamlit` season label if the window
rolled. Nothing under `Zones/2026-09/` needs committing (it stays local/untracked).

## Runtime

The bottleneck is the **200-game-resample bootstrap** in steps 4–5: each resample
solves the normal equations for a 2.68M-row × ~2,400-column sparse system, per
target. Expect roughly **tens of minutes to ~1–2 hours** on a laptop for all
fits; steps 1–3 are I/O-bound over the caches (minutes), and step 7 (the export)
is **seconds**. A monthly cadence comfortably absorbs this.

## Where outputs land

- Published, committed: `rapm/output/player_rapm_5v5.csv` (+ `README.md`).
- Study intermediates: under `Zones/2026-09/rapm/` and `…/iti/` (local, untracked).

## If this is ever automated

A single local `make`/shell wrapper running steps 1–7 in order, parameterised by
`--seasons`, is enough. It should **not** go on the weekly CI: the CI runners
don't have the raw caches, and the bootstrap is too heavy for that job. A reminder
(calendar / cron on the local machine) to run the wrapper monthly is the simplest
reliable trigger.
