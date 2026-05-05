# Methodology

This document describes the analytical decisions underlying the HockeyROI frameworks, the reasoning behind each choice, and the verification work that supports them. It is the canonical reference for the project's methodology and is updated when methodology changes; data files reflect the methodology version stamped below.

**Methodology version:** May 2026 audit + May 3, 2026 zone-adjustment factor swap.
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

A note on factor selection: the project initially derived an empirical factor of 0.1071 through optimization against an outcome variable, then switched to Tulsky's published 0.035 factor. The current data state (May 2026 onward) reflects Tulsky's factor. Goalie reports published prior to May 2026 may cite NFI%_ZA values computed with the older factor; raw NFI% values cited in those reports remain unchanged. See `Goalies/README.md` for the cited-values note.

### Locked spot-check values

Anyone re-running the NFI pipeline should expect these values for reference players in the 2025-26 season, **as of the methodology version stamp at the top of this document.** These values will shift slightly as the current season progresses; the table is a verification anchor for reproducing the pipeline at this snapshot in time, not a permanent locked reference.

| Player | NFI% | NFI%_ZA |
|---|---|---|
| Auston Matthews | 0.4824 | 0.4804 |
| Connor McDavid | 0.5782 | 0.5733 |
| Brandon Hagel | 0.5958 | 0.5939 |
| Mattias Ekholm | 0.5969 | 0.5921 |
| Zach Hyman | 0.5552 | 0.5511 |

If your pipeline output for the same data snapshot differs from these values by more than rounding (±0.0005), something is wrong with your reproduction. If you're re-running the pipeline at a later date with additional games included, expect drift.

---

## TZI: Transitional Zone Impact

### What it measures

TZI quantifies how much a player tilts the ice based on where they start their shift. Three peer metrics — DZI, NZI, OZI — measure this separately for each starting zone, rather than collapsing the three deployment contexts into a single number.

- **DZI — Defensive Zone Impact.** A player's net ice-tilt impact during shifts that begin in the defensive zone. Captures whether the player escapes their own zone cleanly and turns DZ starts into shot-attempt advantages.
- **NZI — Neutral Zone Impact.** Net ice-tilt impact during shifts that begin in the neutral zone. Captures transition play and zone-entry effectiveness.
- **OZI — Offensive Zone Impact.** Net ice-tilt impact during shifts that begin in the offensive zone. Captures whether OZ starts get converted to sustained pressure.

Each metric uses Fenwick-based shot differential while the player is on the ice, attributed to the player's starting-zone context.

### Why three metrics rather than one

Different deployment contexts produce different ice-tilt patterns for the same player. A defensively-deployed player who excels at DZ exits may have strong DZI but unremarkable OZI; a finisher who feasts on OZ starts may show the inverse profile. Collapsing these into a single "zone-impact" number — as conventional zone-start adjustments do — loses the deployment-specific signal.

Reporting DZI, NZI, and OZI as peer metrics preserves that information. Linemate-adjusted variants (DZI_L, NZI_L, OZI_L) and competition-adjusted variants are computed where the underlying linemate and competition data are available.

### What TZI does not claim

TZI is descriptive. It characterizes how players tilt the ice given their deployment; it does not claim any particular relationship to team winning. Exploratory R-squared analyses against standings points were run during development and did not produce findings that would support predictive claims. The framework stands as a player-evaluation descriptive tool, not a predictive one.

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

Player and goalie metrics report Wilson 95% confidence intervals where appropriate. The Wilson interval is preferred over the normal approximation because it remains valid at the small-sample sizes common in goalie filter analyses (e.g., a goalie's CNFI save rate may draw from only 50-100 shots in a season).

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

After the audit, a separate methodology refinement (May 3, 2026) switched the zone-adjustment factor from the empirical 0.1071 to Tulsky's published 0.035. This switch was motivated by recognition that the empirical-factor derivation rested on R-squared optimization the project no longer wants to lean on; Tulsky's factor is established convention and is methodologically defensible without that derivation. Spot-check values in this document reflect the post-factor-swap state.

---

## Pipeline reproducibility

The full pipeline can be reproduced from the NHL API given the scripts in `NFI/scripts/`, `Zones/scripts/`, and `NFI/Geometry_post/NF_PY/`. Execution order, schema dependencies, and column-naming conventions are documented in `PIPELINE.md`. Spot-check values in this document should be reproduced by anyone re-running the pipeline at the same data snapshot; deviation beyond rounding indicates a reproduction error rather than a methodology disagreement.

---

*This document reflects the methodology as of the May 2026 audit and the May 3 zone-adjustment factor swap. Future methodology changes will increment the version stamp at the top of this document and update the locked spot-check values accordingly.*
