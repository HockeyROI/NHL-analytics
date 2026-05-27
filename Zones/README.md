# Zones — Transitional Zone Impact (TZI)

TZI measures how much of a player's on-ice time is spent in the offensive zone after a faceoff, from three different starting positions. It is a **zone-time share** framework, not a shot-differential one.

See `../METHODOLOGY.md` for the framework rationale, construction, sample thresholds, and what TZI does / does not claim.

## The three peer metrics

- **DZI — Defensive Zone Impact.** Share of OZ time on shifts that begin with a DZ faceoff.
- **NZI — Neutral Zone Impact.** Share of OZ time on shifts that begin with an NZ faceoff.
- **OZI — Offensive Zone Impact.** Share of OZ time on shifts that begin with an OZ faceoff.

Each metric is Wilson-shrunk for sample size, then position-normalized to a 0–10 score within forwards / within defense.

## Linemate adjustment

Only **OZI** has a working linemate-adjusted variant (`OZI_L`). NZI / DZI / TNZI linemate adjustments were removed in May 2026 — the team-level OLS regression that produced them could not identify a coefficient at n=32 with multicollinear predictors. The orphaned methodology and historical CSVs sit under `_orphaned_broken_L_2026_05/` for transparency. See that folder's README for the audit trail.

Rel-NZI (`compute_rel_tnzi.py`, outputs in `adjusted_rankings/rel_tnzi_*`) provides a separate, methodologically independent on-ice minus off-ice teammate-effect view.

## Folder layout

```
scripts/             Production scripts that compute TZI metrics
output/              Derived TZI metrics at player and team level
output/playoffs/     Playoff-specific TZI files
adjusted_rankings/   Sorted publication rankings (4-year pooled headlines)
adjusted_rankings/per_season/        Per-season + 2-year-recent leaderboards
adjusted_rankings/publication_filtered/  GP-filtered rankings for post-ready usage
raw/                 Raw shift and play-by-play data (gitignored)
_orphaned_broken_L_2026_05/  Quarantined broken linemate-adjustment artifacts
```

## Key scripts

- `compute_iozc_iozl_dozi.py` — main producer. Writes OZI, DZI, NZI, TNZI raw + OZI's _C / _L / _CL adjusted variants, DTNZI flags, per-season + 2-year leaderboards, and top20 derived view.
- `compute_tozi_tdzi.py` — TOZI / TDZI peer rollup variants.
- `compute_rel_tnzi.py` — on-ice vs off-ice Rel-NZI computation (independent of the team-level OLS).
- `compute_oze_dze_nze.py` — per-player OZE / DZE / NZE foundations.
- `compute_zone_and_overlap.py` — writes `_player_meta.json`, `_zone_time.json`, `_overlap.pkl` (foundations).

## What's still actively published

- Raw NZI, OZI, DZI scores (0–10 within position group)
- OZI_L (linemate-adjusted OZI; only working linemate variant)
- IOZC-adjusted variants (`*_C`) for all four metrics
- DTNZI trajectory flags (raw deltas; mean-reverting — see methodology caveat)
- Rel-NZI (on-off)
- Per-season + 2-year-recent leaderboards in `adjusted_rankings/per_season/`
- Publication-filtered leaderboards in `adjusted_rankings/publication_filtered/`

## What TZI does not claim

TZI is descriptive. It characterizes deployment-conditioned territorial impact and does not claim a particular relationship to team winning. See `TECH_DEBT.md` for resolved items and `../METHODOLOGY.md` for the full caveat list.

---

*See `../METHODOLOGY.md` for methodology and `../PIPELINE.md` for execution details.*
