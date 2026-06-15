# Methodology

This document describes the analytical decisions underlying the HockeyROI frameworks, the reasoning behind each choice, and the verification work that supports them. It is the canonical reference for the project's methodology and is updated when methodology changes; data files reflect the methodology version stamped below.

**Methodology version:** June 12, 2026 — added Goalie Metrics section (NFI-GSAx, QNFS%, QS-GSAx) with Vollman Quality Starts disambiguation; the previous June 7 update covered the NFI-QG half-credit tie handling and documented TZI2 as exploratory single-game tooling (not part of the public framework), building on the May 20, 2026 audit + bug fix, the May 2026 audit, and the May 3, 2026 zone-adjustment factor swap.
**Data snapshot reflected in this document:** values current as of the version stamp date. Counts and player-level values shift as games are added to the dataset.

---

## NFI: Net-Front Impact

### The structural insight

Conventional NHL analytics define "high-danger" scoring chances using the home-plate-shaped trapezoid in front of the net (Natural Stat Trick's HD definition):

```
HD trapezoid (NST):
  (x ≥ 69 AND x < 85 AND |y| ≤ 22)
  OR
  (x ≥ 85 AND x ≤ 89 AND |y| ≤ 18)
```

This trapezoid lumps together two qualitatively different shot regions: the immediate net-front, where rebounds and deflections drive elite conversion rates, and the wider area to the *sides* of the net, where shots convert at unremarkable rates despite being geographically "in close." Treating these regions identically produces a high-danger metric that's noisier than it should be — strong shot locations get diluted by mediocre ones inside the same zone.

NFI narrows the high-danger zone to where the conversion geometry actually matters:

- **CNFI (Close Net-Front Impact):** the immediate doorstep area, where rebound and deflection chances live.
- **MNFI (Mid Net-Front Impact):** the high slot, where elite shooters convert rushes and clean looks.
- **TNFI (Total Net-Front Impact):** CNFI ∪ MNFI.

Zones excluded from NFI: the side-of-net regions of the conventional HD trapezoid, and the FNFI (far net-front) zone that the audit dropped from the framework after determining its noise dominated its signal.

### Why Fenwick (not Corsi) for spatial metrics

NFI metrics use Fenwick-based shot counts (shot-on-goal, missed-shot, goal — excluding blocked shots). This is the framework's choice for any metric defined by shot location, and the reasoning is defensive rather than derived from a confirmed model of the underlying data.

During the May 2026 audit, blocked-shot coordinates in the NHL play-by-play data were observed to behave inconsistently with shots-on-goal coordinates. Two anomalies surfaced:

1. The NHL API schema reports `zoneCode = D` (defensive) for blocked shots, while the same shots viewed from the shooting team's perspective should code as offensive — suggesting a possible reference-frame difference, but not one we could fully verify.
2. Empirically, blocked shots cluster approximately 8 feet closer to the net than shots-on-goal in the same nominal location, and the point-shot lobe present in shots-on-goal data is missing from blocked-shot data — suggesting the coordinates may be recorded at a different point in the shot sequence (possibly the blocker's stick, possibly something else).

We did not resolve definitively what blocked shot coordinates represent. Rather than guess, the framework excludes blocked shots from any spatial metric. Fenwick — which excludes blocks by definition — is the resulting choice for all NFI metrics at player, team, and goalie level.

This is conservative methodology: when a data property can't be verified, the right move is to exclude rather than assume. The cost is that NFI metrics ignore some real shot attempts; the benefit is that the spatial component of every NFI metric is defensible.

### Zone adjustment

NFI rates are zone-adjusted using the Tulsky linear correction:

```
ZA_pct = raw_pct - factor × (oz_ratio - 0.5)
```

Where `oz_ratio` is the share of a player's non-neutral-zone faceoffs that occurred in the offensive zone. The factor is **0.035 (3.5pp)** — Tulsky's published value, the established convention for NHL zone adjustment.

Zone adjustment's contribution to NFI's analytical content is modest. Empirical testing during the audit found that zone-adjusted NFI (NFI%_ZA) and raw NFI% rank players nearly identically. The zone adjustment is included for consistency with established NHL analytics convention rather than as a substantive methodological refinement. Raw NFI% is the recommended primary metric; NFI%_ZA is a secondary zone-corrected variant.

A note on factor selection: the project initially derived an empirical NFI factor of 0.1071 by optimizing against an outcome variable, then tested whether it earned its place. It did not: over Tulsky's published 0.035 the empirical factor added no meaningful, statistically significant predictive value (ΔR² = +0.005, p = 0.187 against standings points). The empirical 0.1071 was therefore **evaluated and rejected**, not merely deprecated, and the framework uses Tulsky's 0.035. The current data state (May 2026 onward) reflects Tulsky's factor. Goalie reports published prior to May 2026 may cite NFI%_ZA values computed with the older empirical factor; raw NFI% values cited in those reports remain unchanged. See `Goalies/README.md` for the cited-values note.

### Locked spot-check values

Anyone re-running the NFI pipeline should expect these values for reference players in the 2025-26 season, **as of the methodology version stamp at the top of this document.** These values will shift slightly as the current season progresses; the table is a verification anchor for reproducing the pipeline at this snapshot in time, not a permanent locked reference.

| Player | NFI% | NFI%_ZA |
|---|---|---|
| Auston Matthews | 0.4759 | 0.4746 |
| Connor McDavid | 0.5639 | 0.5590 |
| Brandon Hagel | 0.5881 | 0.5865 |
| Mattias Ekholm | 0.5724 | 0.5685 |
| Zach Hyman | 0.5650 | 0.5606 |

If your pipeline output for the same data snapshot differs from these values by more than rounding (±0.0005), something is wrong with your reproduction. If you're re-running the pipeline at a later date with additional games included, expect drift.

---

## NFI-QG: Per-Game Quality Game Rate

### What NFI-QG measures

NFI-QG is the per-game complement to season-aggregate NFI%. Each player's 5v5 ES regulation games are classified as a "Quality Game" if the player's on-ice NFI share for that game lands at or above the empirical position-median NFI share (F median and D median computed across all qualifying player-games over the four-season scope). Season-level NFI_QG_pct is the rate of Quality Games over a player's qualifying games. It captures **consistency of outchancing opponents in the net-front zone**, complementing season-aggregate NFI% which captures aggregate dominance.

The parallel xG-QG metric uses the same construction with MoneyPuck xGoal-weighted xG% per game in place of NFI share.

### Qualifying-game floor

A player-game enters NFI-QG only if, at 5v5 ES regulation:
- TOI on-ice ≥ 480 seconds (8 minutes), AND
- Total on-ice Fenwick attempts (for + against) ≥ 5

Below either threshold the game is excluded as low-signal.

### Handling of Exact-0.5 Tie Games

NFI per-game ratios are discrete by construction. Most qualifying games have small NFI denominators (median 5-7 NFI-zone events while a player is on-ice), so the per-game NFI share frequently lands at common fractions like 1/2, 2/4, 3/6 — all of which equal exactly 0.5000. **14.24% of all qualifying player-games (24,310 of 170,658) land at exactly 0.5000 on NFI.** This is a real consequence of the discreteness, not a data error.

xG-QG does not have this issue. xG ratios are computed from continuous MoneyPuck xGoal weights and the per-game xG share is effectively continuous — only about 0.002% of qualifying games land within ±1e-5 of 0.5000.

The framework handles this asymmetry with metric-specific rules:

- **NFI-QG (half-credit on ties):**
  ```
  is_NFI_QG = 1.0  if NFI_pct_game >  position median NFI%
            = 0.5  if NFI_pct_game == position median NFI%   (half-credit tie)
            = 0.0  if NFI_pct_game <  position median NFI%
  ```
  The 0.5-credit treatment handles indeterminate outcomes symmetrically. Prior strict-`>` rule counted ties as losses (asymmetric: flag rate fell to 0.432 vs xG's 0.500); prior `>=` rule counted them as wins (flag rate spiked to 0.574 because the entire 14.24% tie spike landed in the QG bucket). Half-credit makes NFI-QG and xG-QG comparable: NFI flag rate now ≈ 0.503 by construction.

- **xG-QG (greater-or-equal, unchanged):**
  ```
  is_xG_QG = 1.0  if xG_pct_game >= position median xG%
           = 0.0  otherwise
  ```
  No tie handling needed under a continuous distribution.

The asymmetric rule across the two metrics is not a stylistic choice — it reflects that NFI and xG have fundamentally different per-game ratio distributions. NFI is discrete-spike-heavy; xG is continuous. Either can be expressed with the other's rule, but only with the cost of either inflating QG rates by ties (xG-style `>=` on NFI) or asymmetrically penalizing ties as losses (strict `>` on NFI).

### Stability of headline findings

The June 2026 tie-handling change was verified to leave the durable-elite cohort intact. Under the strict-`>` rule, 73 players appeared on both the xG Elite Consistent High-Volume and NFI Elite Consistent High-Volume cohorts. Under the half-credit rule, 74 players appear, with 11 names changing at the cohort border — 5 dropped (rank 15-25 borderline cases whose tie shares were low enough that the credit reshuffle pushed them out of quartile 1 in one season), 6 added (similar border cases pushed in). The 68 names common to both rules contain every player at the elite-of-elite tier (McDavid, MacKinnon, Hagel, Hyman, Slavin, Bouchard, Makar, the Tkachuks, Kucherov, etc.). Within-bin rankings under the half-credit rule shuffle modestly: bin sizes can change by 1 player as borderline cases cross thresholds, but top-3 placements for elite-of-elite players are stable.

### Locked spot-check values (NFI-QG, half-credit rule, 2025-26)

| Player | Season NFI-QG_pct (25-26) | 4-yr career mean |
|---|---|---|
| Brandon Hagel | 0.7391 | 0.6808 |
| Connor McDavid | 0.6543 | 0.6861 |
| Zach Hyman | 0.6316 | 0.6890 |
| Evan Bouchard | 0.6420 | 0.6860 |
| Jaccob Slavin | 0.6154 | 0.6515 |
| Cale Makar | 0.5676 | 0.6165 |
| Miro Heiskanen | 0.5658 | 0.6193 |

If your pipeline output for the same data snapshot differs from these values by more than rounding (±0.005), something is wrong with your reproduction.

### Pipeline implementation note

NFI-QG is computed by `Quality_Games/scripts/02_quality_game_aggregation.py`. The half-credit rule produces fractional contributions per tied game; the per-(player, season, team) `NFI_QG_count` column is stored as a float (not truncated to int) so that 0.5 fractional contributions propagate correctly into the season-level aggregation. Truncating to int silently drops 0.5 of QG credit per tied game and produces a 0.003-0.008 systematic underestimate of season NFI_QG_pct — this was caught and fixed in the June 2026 revision.

---

## Zone Impact

### What Zone Impact measures

Zone Impact is a player-evaluation framework measuring how a player's on-ice deployment after each type of faceoff translates into offensive-zone time. Its three published metrics — DZI, NZI, OZI — are independent lenses, not a hierarchy. A complete player rates well on all three.

- **DZI — Defensive Zone Impact.** Share of offensive-zone time on shifts that begin with a defensive-zone faceoff. Captures whether the player escapes their own zone cleanly.
- **NZI — Neutral Zone Impact.** Share of offensive-zone time on shifts that begin with a neutral-zone faceoff. Captures transition play.
- **OZI — Offensive Zone Impact.** Share of offensive-zone time on shifts that begin with an offensive-zone faceoff. Captures whether OZ starts get converted to sustained pressure.

The framework also defines **TZI (Transitional Zone Impact)** — the net offensive tilt on neutral-zone starts: offensive-zone-time share minus defensive-zone-time share on shifts that begin with a neutral-zone faceoff (stored as `TNZI` in the data). TZI is computed but is not surfaced on the public dashboard, which leads with the three component lenses above.

Zone Impact metrics are **zone-time share** measures, not shot-differential. The earlier description in this document as "Fenwick-based shot differential" was inaccurate — corrected May 2026 to match what the code computes.

### Construction

1. For each shift, capture player on-ice time and the zone of each puck event.
2. Bucket each shift by its starting faceoff zone (NZ, OZ, DZ).
3. Within each bucket, sum seconds in each zone; compute share of time spent in the offensive zone.
4. Apply Wilson interval shrinkage to handle small-sample variance.
5. Position-normalize: rank forwards vs forwards and defense vs defense separately. Rescale to 0–10 within each position group.

### Why three metrics rather than one

Different deployment contexts produce different ice-tilt patterns for the same player. A defensively-deployed player who excels at DZ exits may have strong DZI but unremarkable OZI; a finisher who feasts on OZ starts may show the inverse profile. Collapsing these into a single "zone-impact" number loses the deployment-specific signal.

### Sample thresholds

- 50 faceoff-start shifts minimum per metric (NZ-FO for NZI, OZ-FO for OZI, DZ-FO for DZI)
- 20 games played minimum
- Stricter publication floor: 100 GP forwards / 130 GP defense (4-year pooled); 60 GP forwards / 70 GP defense (2-year recent)

### Linemate adjustment

Only **OZI** has a working linemate-adjusted variant (`OZI_L`). Linemate adjustment for NZI, DZI, and TNZI was attempted via team-level OLS regression but the regression cannot identify the coefficient at n=32 with multicollinear predictors. Bootstrap analysis showed sign flips and large variance; Ridge regression did not stabilise the estimate. Rather than publish unstable adjustments, the framework presents raw NZI / DZI plus OZI_L. The orphaned methodology and historical CSVs are preserved under `Zones/_orphaned_broken_L_2026_05/` for transparency.

### What Zone Impact is not

- **Not a complete skill rating.** Doesn't measure shooting talent, defensive engagement, faceoff ability, or specialty teams.
- **Not a value-over-replacement.**
- **Not deployment-controlled for quality of competition.** A player getting sheltered minutes may rate higher than one playing tough minutes.
- **Not predictive of standings.** Team-level correlation with standings is moderate but is explicitly not the framework's purpose. The "tnzi_winning_correlation" diagnostic that previously sat beside the framework was retired May 2026 with the broken _L methodology.

### What Zone Impact is

- A descriptive territorial-impact lens.
- A way to identify players whose zone-time output exceeds or falls short of their reputation.
- A complement to NFI (shot quality) and Rel-NZI (true on/off teammate effect).

### Caveats

- Per-season DTNZI deltas mean-revert (Pearson r ≈ −0.4 year over year). Flag a recent delta as "the most recent delta was negative", not "declining career trajectory".
- Raw scores include deployment context, not just player skill.
- Forwards and defense are normalized within their own groups; their 0–10 scores are not directly comparable across positions.
- 99.89% game coverage at the current data snapshot; a small number of postponed regular-season games are not in foundation files (same precedent as NFI; not blocking).

---

## TZI2: Share-of-Attack (exploratory, not in public framework)

TZI2 is internal commentary tooling for analyzing individual games. It has not been used in any published HockeyROI work to date, is not part of the public NFI/TZI framework, and is not featured on the public Streamlit dashboard. The section below documents the construction for future reference and for readers who encounter the related code and outputs in the repo (`Zones/scripts/compute_iozc_iozl_dozi.py`, `Zones/output/tzi2_team.csv`, `Zones/output/tzi2_player.csv`). No published claim or framework metric depends on TZI2.

### What TZI2 measures

TZI2 uses the same shift-bucket source as published TZI but applies a different aggregation formula. Where published TZI computes `oz_sec / total_sec` (NZ time included in the denominator), TZI2 computes `oz_sec / (oz_sec + dz_sec)` (NZ time excluded). The TZI2 formula yields a head-to-head share — when both teams' numbers are computed on matched shifts, they sum to 100% by construction, because one team's `oz_sec` equals the opponent's `dz_sec` by symmetry.

TZI2 operates at the team level; player-level TZI2 exists in the codebase but is not a replacement for player-level TZI. Player-level TZI2 status is covered at the end of this section.

### The five TZI2 team-level metrics

- **NZIS** (Neutral Zone Impact Share) — `oz_sec / (oz_sec + dz_sec)` on NZ-faceoff shifts
- **OZIS** (Offensive Zone Impact Share) — `oz_sec / (oz_sec + dz_sec)` on OZ-faceoff shifts
- **DZIS** (Defensive Zone Impact Share) — `oz_sec / (oz_sec + dz_sec)` on DZ-faceoff shifts
- **TZI2** (Contested Zone Share) — `oz_sec / (oz_sec + dz_sec)` on OZ-FO + DZ-FO shifts combined (the deep-zone contested territorial battle)

Note: the "2" in TZI2 means *two zones combined* (the OZ-FO + DZ-FO contested pool), not "version 2 of TZI". NZIS / OZIS / DZIS are single-zone metrics and do not use the "2" suffix.

### Naming clarification — IMPORTANT

NZIS, OZIS, DZIS, and TZI2 are **not the same metric** as published TZI's NZI, OZI, DZI, and TZI. The published versions keep NZ time in the denominator and apply Wilson shrinkage + position normalization on a 0–10 scale. The share-of-attack family (NZIS, OZIS, DZIS, TZI2) is raw share-of-attack rates (percentages summing to 100% across the two teams on matched shifts), no Wilson shrinkage, no position normalization, no 0–10 rescaling.

When citing numbers, always specify which version. "NZI = 7.4" is a published TZI score on the 0–10 scale. "NZIS = 54.4%" is the share-of-attack version. Confusing the two yields wrong conclusions.

### Purpose: complement, not replacement

TZI2 and published TZI ask different questions of the same shift buckets:

- **Published TZI** asks "how much sustained OZ pressure does this team generate?" — a volume measure of OZ time per shift.
- **TZI2** asks "when someone was attacking on this team's shifts, what share was them?" — a head-to-head efficiency measure where neutral-zone idle time drops out of the denominator.

Both are valid hockey questions. Teams can score high on one and low on the other. A team that generates lots of OZ time AND gives up lots of OZ time on the same shifts will score high on published TZI but middling on TZI2. A team that generates less OZ time but wins each contest decisively will score lower on published TZI but high on TZI2.

### Why published TZI remains primary

The audit comparing both methodologies found that removing NZ time from the denominator pushes some low-volume teams upward in ways that do not reflect actual team quality. Calgary rises from #28 published TZI 4yr composite to #4 on TZI2; Vegas falls from #3 published to #14 on NZIS; Winnipeg falls from #9 published to #18 on NZIS. The published methodology's NZ-in-denominator treatment is doing real work — distinguishing high-quality volume teams from low-volume teams that win head-to-head exchanges only because their NZ idle time isn't penalizing them.

For the season-level ranking question — "who's the territorially best team" — published TZI remains the primary metric. The share-of-attack family (NZIS, OZIS, DZIS, TZI2) is an internal lens used alongside published TZI to surface volume-vs-efficiency disagreements as interpretable analytical findings (for example, Montreal is published DZI #1 but DZIS #14 — the gap reveals MTL's defensive value is volume-driven rather than head-to-head dominant).

### Data layout

- **Team-level CSV:** `Zones/output/tzi2_team.csv`
- **Schema:** `team, season_window, gp, contested_share, contested_rank, nzis, nzis_rank, ozis, ozis_rank, dzis, dzis_rank`
- **Six season windows per team:** `2022_23`, `2023_24`, `2024_25`, `2025_26`, `2y_pool` (24-25 + 25-26), `4y_pool` (22-23 → 25-26)
- **Expected row count:** 192 (32 teams × 6 windows)
- **Historical note (June 7, 2026):** the initial schema shipped with duplicate columns — `nz_share`/`nzi_share` (same value) and `ozi_share`/`dzi_share` (legacy descriptive names). These were collapsed and renamed to single `nzis` / `ozis` / `dzis` columns on the same day, before any downstream consumers existed.

### Validation spot-checks (locked)

These values are the methodology anchor — any bucket-source change must reproduce these to ±0.05 pp or be treated as a regression:

| Team | Window | Metric | Value |
|---|---|---|---|
| CAR | 4y_pool | contested_share | 54.12% |
| CAR | 4y_pool | nzis | 54.40% |
| MTL | 4y_pool | contested_share | 48.91% |

### Player-level TZI2 status

Player-level TZI2 was computed during audit runs but is **not in production**. The audit found a median rank shift of 53 ranks on NZI forwards between TZI2 share-of-attack and published TZI, even after Wilson shrinkage was applied at the player level. This indicates a structural divergence from published TZI rather than noise — the two metrics are pointing at distinct constructs at the player level (volume of OZ time per shift vs head-to-head share-of-attack), and the disagreement is not reduced by sample-size corrections.

Player-level TZI2 will not be published until a public methodology introduction post is released that frames it as a distinct metric rather than a refinement of TZI. Until then:

- Audit and pipeline scripts MUST NOT use player-level TZI2 numbers for downstream rankings or commentary without explicit authorization.
- A computed CSV may exist at `Zones/output/tzi2_player.csv` as an audit artifact; treat its values as diagnostic-only.

---

## Goalie Metrics: NFI-GSAx, QNFS%, QS-GSAx

### Disambiguation from conventional Quality Starts

Read this before mapping any of these metrics to a conventional goalie-consistency measure. Robert Vollman's Quality Starts metric (~2009) is binary on **save percentage**: a quality start is a game where the goalie's save% exceeds league-average save% (with a small adjustment for high-shot-volume games). Anyone in hockey analytics who hears "Quality Starts" will map to that definition. The HockeyROI quality-start metrics are constructed differently on three dimensions:

1. **Threshold.** Per-game GSAx ≥ 0 — the goalie beat their expected on a danger- or xG-weighted basis — **not** save% > league average.
2. **Two parallel definitions** (QNFS% and QS-GSAx), not one.
3. **Different shot scopes.** QNFS% uses net-front (CNFI ∪ MNFI) shots only; QS-GSAx uses all shots faced.

These are not Vollman Quality Starts under a different name. Do not conflate them.

### NFI-GSAx

**What it measures:** per-(goalie, season) goals-saved-above-expected on **CNFI ∪ MNFI shots only**. The expectation baseline is an **internal per-season, per-zone league goal rate** — within each season the league's goals-per-shot-faced is computed separately for the CNFI and MNFI zones, and each goalie's expected goals is `faced_CNFI × rate_CNFI + faced_MNFI × rate_MNFI`. GSAx is that expectation minus goals allowed on net-front shots. (This is a HockeyROI-internal model, **not** the MoneyPuck xGoal model — MoneyPuck is used only by QS-GSAx, below.) The faced denominator is save-based — shots-on-goal and goals (missed shots excluded), distinct from QNFS%'s Fenwick base. Per-60 normalization allocates each goalie's pooled even-strength TOI across seasons in proportion to the share of their career net-front faced shots that fell in each season (shifts data is not season-keyed, so exposure is apportioned by faced-shot share).

**Why net-front only:** NFI's structural insight is that the immediate net-front and high slot are where shot location actually predicts conversion. Restricting GSAx to those shots puts the goalie metric on the same spatial basis as the player- and team-level NFI framework — they are commensurable in a way that all-shot GSAx and NFI% are not.

**Qualifying:** minimum 100 net-front shots faced per season for the per-season table; minimum 300 net-front shots faced across pooled seasons for the pooled table.

**Confidence intervals:** none are currently reported. NFI-GSAx per-60 is a rate count over a fixed exposure window; were CIs to be added they would use the Poisson (Garwood) interval, per the rate-vs-proportion rule established in the May 20, 2026 audit (see `docs/AUDIT_2026-05-20.md`). The present outputs carry point estimates only.

**Source:** `NFI/scripts/21_goalie_gsax_by_season.py` → `NFI/Output/goalie_nfi_gsax_by_season.csv` (per-season, 354 goalie-seasons) and `NFI/Output/goalie_nfi_gsax_pooled_v2.csv` (pooled, 84 goalies). The pooled file currently spans five seasons (2021-22 → 2025-26), one more than QNFS%/QS-GSAx.

### QNFS%: Quality Net-Front Save percentage

**What it measures:** the share of a goalie's 5v5 ES regulation appearances in which their per-game net-front (CNFI ∪ MNFI) GSAx ≥ 0. Each game yields a binary indicator (1 if NF GSAx ≥ 0, else 0); the season-level metric is the rate of 1-games over qualifying games. The per-game expectation uses the same internal per-season per-zone league goal rate as NFI-GSAx, but on a **Fenwick** base (shot-on-goal + missed-shot + goal). Wilson 95% confidence intervals are reported, because the underlying statistic is a proportion bounded in [0, 1], not a per-60 rate.

**Qualifying:** a goalie-game enters QNFS% only if the goalie faced ≥ 3 net-front shots at 5v5 ES regulation in that game. A goalie qualifies for season-level reporting with ≥ 25 qualifying games in any single season within the window.

**Interpretation:** QNFS% captures *consistency of beating expected on net-front shots*, complementing season-aggregate NFI-GSAx, which captures aggregate dominance. A goalie can post high NFI-GSAx but middling QNFS% (a few huge games against expected amid many forgettable ones), or the reverse.

**Source:** `NFI/goalie_consistency/scripts/compute_qnfs.py` and `compute_qnfs_per_season.py`; outputs `NFI/goalie_consistency/output/qnfs_2022-2026.csv` (146 goalies, of which 82 qualified) and `qnfs_per_season_2022-2026.csv` (294 goalie-seasons). Window: four seasons, 2022-23 → 2025-26.

### QS-GSAx: Quality Start GSAx percentage

**What it measures:** the share of a goalie's 5v5 ES regulation appearances in which their per-game **all-shot** GSAx ≥ 0. Same binary-indicator + season-rate construction as QNFS%, but on all shots faced rather than net-front only, and using the **MoneyPuck xGoal model** for the per-shot expectation (per-game GSAx = Σ xGoal − Σ goals). Wilson 95% confidence intervals; reported with both the point estimate and a Wilson lower-bound rank.

**Qualifying:** minimum 10 shots faced per game; minimum 25 qualifying games per season.

**Why QS-GSAx exists alongside QNFS%:** the two measure related but non-identical things. Spearman ρ between the two metrics' Wilson lower bounds (QS_GSAx_lo vs QNFS_lo) across the n = 80 qualified-goalie overlap is 0.810 — they rank goalies similarly, but ~34% of rank variance is unshared (1 − 0.810² = 0.344). Net-front-only QNFS% penalizes failures on the highest-leverage shots more sharply; all-shot QS-GSAx captures a goalie's full expected-vs-actual ledger. Reporting both keeps the framework honest about which shot scope drives which result.

**Source:** `NFI/goalie_consistency/scripts/compute_qs_gsax.py`; outputs `NFI/goalie_consistency/output/qs_gsax_2022-2026.csv` (81 goalies) and `qs_gsax_per_season_2022-2026.csv` (210 goalie-seasons). Window: four seasons, 2022-23 → 2025-26.

### Cohort overlap note

The three goalie metrics qualify at different thresholds and on different shot bases and season spans, so they produce different cohort sizes: NFI-GSAx pooled has 84 goalies (five seasons), QNFS% has 146 in-file with 82 qualified (four seasons), QS-GSAx pooled has 81 (four seasons). A goalie may appear in one metric and not another because the qualifying floors differ — this is by design, not a bug. A goalie with a thin net-front shot diet but high total volume may qualify for QS-GSAx and NFI-GSAx but not for QNFS%; an infrequent appearance-maker may qualify for none. Where the same goalie appears in multiple metrics, the metrics are commensurable *for that goalie* — bearing in mind the net-front (QNFS%, NFI-GSAx) vs all-shot (QS-GSAx) scope difference and the internal-rate (QNFS%, NFI-GSAx) vs MoneyPuck (QS-GSAx) expectation model.

### Confidence interval note

QNFS% and QS-GSAx are proportions; Wilson 95% CIs are the correct tool and are reported for both. NFI-GSAx per-60 is a rate count over a fixed exposure; per the May 20, 2026 audit migration (see `docs/AUDIT_2026-05-20.md`), Poisson CIs are the correct tool for rate metrics — but NFI-GSAx currently reports point estimates only, so no CI is emitted today. This follows the same Wilson-vs-Poisson rule the player-level framework uses.

### Locked spot-check values

Pooled values for eight reference starters, as of the methodology version stamp at the top of this document. All eight qualify in all three cohorts. QNFS% and QS-GSAx% are percentages; NFI-GSAx is GSAx per 60 ES minutes. Values shift as games are added.

| Goalie (team) | QNFS% | QS-GSAx% | NFI-GSAx /60 |
|---|---|---|---|
| Connor Hellebuyck (WPG) | 63.37 | 63.37 | 0.162 |
| Igor Shesterkin (NYR) | 62.22 | 60.89 | 0.226 |
| Ilya Sorokin (NYI) | 59.74 | 59.74 | 0.268 |
| Andrei Vasilevskiy (TBL) | 53.88 | 54.74 | 0.185 |
| Juuse Saros (NSH) | 55.10 | 53.69 | 0.061 |
| Sergei Bobrovsky (FLA) | 53.59 | 52.61 | 0.070 |
| Jacob Markström (CGY) | 49.75 | 51.02 | 0.008 |
| Adin Hill (VGK) | 52.63 | 50.38 | −0.081 |

The cohort-overlap NaN convention (show the row with NaN where a goalie qualifies for only some metrics) applies to goalies outside this eight — e.g. one clearing QS-GSAx's and NFI-GSAx's floors but under QNFS%'s 25-GP-in-a-season gate. If your pipeline output for the same data snapshot differs from these values by more than rounding (±0.01 for the percentages, ±0.005 for per-60), something is wrong with your reproduction.

---

## Single-Game Zone Reporting Convention

### Single-game zone numbers use share-of-attack

When reporting NZI, OZI, or DZI numbers for an individual game (X posts, live-game analysis, playoff game breakdowns), the numbers are reported as share-of-attack — applying the TZI2 formula to that game's data only. They are **not** the published Wilson-shrunk, position-normalized values that appear in season-level rankings.

### Formula

For each team and each faceoff zone:

```
zone_share = team_oz_sec / (team_oz_sec + team_dz_sec)
```

…on shifts that started with that zone's faceoff context (OZ-FO for OZI, DZ-FO for DZI, NZ-FO for NZI). Both teams' numbers on matched shifts sum to 100% by construction.

### Reference example

SCF Game 1 (June 2, 2026):

| Zone | CAR | VGK |
|---|---|---|
| OZI | 76.0% | 24.0% |
| DZI | 36.3% | 63.7% |
| NZI | 44.6% | 55.4% |
| Contested combined | 66.2% | 33.8% |

### Season-level rankings continue to use published TZI

Single-game share-of-attack is a **presentation choice** for live-game and playoff-game contexts, where matched-shift symmetry makes the 100%-sum framing intuitive. Season-level rankings — the canonical "best NZI defenseman" or "team OZI rank" — continue to use the published TZI methodology (NZ in denominator, Wilson-shrunk, position-normalized, 0–10 scale). Single-game share-of-attack does not feed season rankings.

---

## Data sources and standards

### Primary source

NHL API at `https://api-web.nhle.com/v1/`. Shot events, shift data, play-by-play, and referee assignment data are pulled from this source. Where MoneyPuck's xG model is referenced (in goalie analyses), values are sourced from MoneyPuck's published data files.

### Strength state

All NFI and TZI metrics are computed at **5v5 even-strength regulation time**. Power-play and penalty-kill states are excluded from the flagship metrics; PP/PK variants exist for context but are not the headline figures.

### Cohort definitions

The framework operates on two rolling cohorts:

- **Complete-seasons cohort:** all complete post-COVID seasons (2021-22 onward). Used when cross-season stability is the requirement — methodology validation, multi-year ranking comparisons, anything where mid-season data would introduce a partial-season bias.
- **Current cohort:** complete seasons plus the current season-in-progress. Used for current rankings, recent-form analyses, and anything where the most up-to-date player and team profiles matter more than cross-season consistency.

Cohort sizes update as the season progresses and as completed seasons accumulate. Spot-check values and locked numbers in this document reflect the cohort state at the version stamp at the top of this document; values for the same player will shift slightly as the current season progresses and additional games enter the dataset. Re-running the pipeline at a later date will produce different — but methodologically consistent — values.

### Sample thresholds

- **Player metrics:** minimum 200 minutes of 5v5 even-strength TOI per season for ranking inclusion.
- **Goalie metrics:** minimum 100 shots faced per filter for save-rate reporting.

These thresholds reduce noise from low-sample player-seasons and goalie filter cells. Players or goalies below threshold are excluded from ranked outputs; their data is preserved in raw files for context but not displayed in headline rankings.

### Confidence intervals

The framework uses two interval families, chosen by the underlying statistic:

- **Wilson 95% CIs** for proportions — save percent, NFI percent share, conversion rate, anything bounded in [0, 1]. Wilson is preferred over the normal approximation because it remains valid at small samples (a goalie's CNFI save rate may draw from only 50-100 shots).
- **Poisson 95% CIs** for per-60 rates — events per 60 minutes of TOI, anywhere a count is divided by a time exposure. Uses the exact Garwood (chi-square) interval on the event count, scaled to per-60. Wilson is incorrect for rates and was previously misapplied here; the May 20, 2026 bug fix migrated all per-60 helpers to Poisson.

RelNFI metrics carry 95% CIs at the season-player level (Poisson-differential SE; the combined RelNFI uses an empirical partial correlation as the covariance proxy rather than assuming independence) and at the career-pool level (TOI-weighted point with variance-pooled SE).

---

## On exploratory R-squared work

During development, the project ran a substantial volume of exploratory analyses using R-squared correlation against standings points and playoff outcomes. This work explored questions like: "which possession metric correlates best with winning?", "do specific team profiles predict playoff qualification?", and "how much of championship variation is explained by net-front offense?"

These analyses are not the basis for current claims. The reasons:

- Standings points are a noisy outcome variable in a high-variance, short-season sport. Single-season correlations against points are not durable evidence of predictive validity.
- Several of the explored claims (e.g., a three-pillar team-construction model claimed at one point to identify "92.5% of recent playoff teams") did not survive validation. Specifically, claims about playoff prediction accuracy made during the exploratory phase were not supported by holdout testing or cross-validation, and the data shapes that produced them were idiosyncratic rather than generalizable.
- The methodology insights underlying NFI and TZI — the geometric redefinition of high-danger zones, the deployment-specific zone impact metrics — stand on their analytical reasoning rather than on prediction accuracy claims. The frameworks describe player and team behavior more accurately than conventional metrics; whether that translates to winning prediction is a separate, harder question that this project does not currently make claims about.

The exploratory R-squared work and its outputs are preserved in repo history but are not framed as findings. If you encounter R-squared references in code or output files, treat them as historical exploration rather than current methodology.

---

## Verification: the May 2026 audit

The current methodology state reflects a comprehensive audit completed in May 2026. The audit identified and corrected several real bugs in the prior pipeline:

- **Block attribution corrected.** Blocked shots had been attributed to the blocking team's coordinate frame; this was corrected to the shooting team's frame for any non-spatial Corsi calculation.
- **NFI redefined as Fenwick-based** rather than Corsi-based, after the blocked-shot coordinate inconsistencies described above made the Corsi version spatially unreliable.
- **HD zone redefined as NST trapezoid** (the formula above) rather than the prior rectangular approximation.
- **FNFI dropped from the framework** after R² with team performance collapsed to 0.238 and additional analyses showed the zone's noise dominated its signal.
- **3A linemate adjustment downgraded** from default to toggleable, after team-level R² for 3A collapsed to 0.503 versus ZA's 0.583. 3A is preserved as an option but not the default for team rankings.

After the audit, a separate methodology refinement (May 3, 2026) switched the zone-adjustment factor from the empirical 0.1071 to Tulsky's published 0.035. The empirical factor had been derived by R-squared optimization, but on test it added no significant predictive value over Tulsky's 0.035 (ΔR² = +0.005, p = 0.187 against standings points — not significant). It was evaluated and not trusted, not merely set aside for convention; Tulsky's factor is established, defensible without the optimization, and gives up nothing the empirical factor provided.

### May 20, 2026 follow-up audit

A second audit, triggered by gut-checking rank stability against a verification CSV built with a different filter set, surfaced three independent pipeline bugs that survived the May 2026 audit:

1. **`state == "ES"` conflation** — the state column had collapsed 5v5, 4v4, and 3v3 into a single "ES" label, so downstream `state == "ES"` filters over-included by ~1.2% of events. Fixed by differentiating the labels at the source.
2. **Missing `game_type` filter at scripts that read `shots_tagged.csv` directly** — `shots_tagged.csv` contains playoff data (build_playoff_data needs it), so consumers that want regular-season-only aggregation must apply their own filter. Fixed at four consumer sites.
3. **Shift-data over-filter on goalie pillars and team counters** — `03_onice_attribution_pillars.py`'s per-game loop correctly gated skater on-ice attribution by shift-data availability, but the same gate was inherited by goalie pillar counters and team-level for/against counters, which don't depend on shifts. Under-counted goalie and team events by ~5% in 2024-25 (where shift_data was missing 57 of 1312 games). Fixed by moving both workflows to a vectorized post-loop pass.

The May 20 audit also corrected a Wilson-vs-Poisson misuse on per-60 rate CIs across the pipeline (see "Confidence intervals" above) and added RelNFI 95% CIs at the player level. Spot-check values in this document reflect the post-fix state.

Methodology — what NFI measures, the Fenwick choice, the zone definitions, the Tulsky 0.035 factor, the TOI thresholds — is unchanged from the May 2026 audit. Only the pipeline correctness improved.

### June 7, 2026 addition

Two related June 7, 2026 changes, documented in the "TZI2: Share-of-Attack (exploratory, not in public framework)" and "Single-Game Zone Reporting Convention" sections above:

1. **TZI2 documented as exploratory tooling.** Share-of-attack aggregation (`oz_sec / (oz_sec + dz_sec)` instead of `oz_sec / total_sec`), computed at team and player level and persisted to `Zones/output/tzi2_team.csv` (192 rows) and `Zones/output/tzi2_player.csv`. TZI2 is internal single-game commentary tooling — not used in any published work, not part of the public NFI/TZI framework, and not on the Streamlit dashboard. Player-level TZI2 is computed for individual-game commentary but is not authorized for production use. Published TZI methodology is unchanged and remains the primary season-level ranking.
2. **Single-game share-of-attack reporting convention adopted.** Single-game NZI, OZI, DZI numbers (for X posts, live-game analysis, playoff game breakdowns) are reported as share-of-attack — applying the TZI2 formula to the single game's data — rather than as the published Wilson-shrunk, position-normalized season-level values. Season-level rankings are unchanged.

---

## Pipeline reproducibility

The full pipeline can be reproduced from the NHL API given the scripts in `NFI/scripts/`, `Zones/scripts/`, and `NFI/Geometry_post/NF_PY/`. Execution order, schema dependencies, and column-naming conventions are documented in `PIPELINE.md`. Spot-check values in this document should be reproduced by anyone re-running the pipeline at the same data snapshot; deviation beyond rounding indicates a reproduction error rather than a methodology disagreement.

---

*This document reflects the methodology as of the May 2026 audit, the May 3 zone-adjustment factor swap, the June 7, 2026 single-game share-of-attack reporting convention (TZI2 itself, documented the same month, is exploratory tooling outside the public framework), and the June 12, 2026 Goalie Metrics addition (NFI-GSAx, QNFS%, QS-GSAx, with Vollman Quality Starts disambiguation). Future methodology changes will increment the version stamp at the top of this document and update the locked spot-check values accordingly.*
