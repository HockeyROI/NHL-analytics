# Methodology

This document describes the analytical decisions underlying the HockeyROI frameworks, the reasoning behind each choice, and the verification work that supports them. It is the canonical reference for the project's methodology and is updated when methodology changes; data files reflect the methodology version stamped below.

**Methodology version:** July 8, 2026 — added **Quality Game For / Against split** (`Quality_Games/scripts/03_quality_game_for_against.py`): the existing QG metrics use an on-ice SHARE, for/(for+against), which collapses offense and defense into one number; this step grades a game's For and Against per-60 rates separately against league position-median rates. New metrics **xG-QG-F% / xG-QG-A%** and **NFI-QG-A% / NFI-QG-S%** (share of a player's qualifying games where the on-ice offense rate/60 beat the position median, or the on-ice defense rate/60 was below it; NFI keeps the half-credit-tie rule). xG uses For/Against labels; NFI uses Attack (offense) / Suppress (defense) to match the NFI family (`NFI-QG-A%` = attack, `NFI-QG-S%` = suppress). Higher is better on all four. Same per-game qualifying floor (TOI≥480 & attempts≥5) and F/D position medians (all-season pooled) as the existing QG build; NFI uses half-credit ties, xG uses ≥ (For) / ≤ (Against). Emitted with counts + qual_GP so multi-season views pool by ratio-of-sums; team level is the TOI-weighted mean of player rates (20+ GP eligibility). Outputs `per_player_qg_fa{_playoffs}.csv`, `per_team_qg_fa{_playoffs}.csv`, `position_medians_fa{_playoffs}.csv`. Surfaced in the Streamlit Quality Games family (player + team). Builds on July 6, 2026 (d) — Streamlit display change (no data change): retired the **backup** tier from the app. The Goalies tab now surfaces only the **starter**-tier save%-based Quality Start, renamed **QG%s → sQS%** (Starter Quality Start); the starter/backup baseline toggle was removed (only the 5v5 / all-situations shot-scope toggle remains). The underlying tiered build is unchanged — it still computes both tiers (`qg_savepct*` files, `QG_pct_s` / `QG_pct_b` columns intact); only the display was simplified to the starter baseline (the "does this goalie perform like a starter?" question). Builds on July 6, 2026 (c) — Streamlit display update (no data/methodology change). (1) Renamed goalie display labels **QGx → QG** and **QNFS% → QNFG%** — labels only; the underlying `QS_GSAx_*` and `QNFS_*` columns and their construction are unchanged. (2) Split the skater **xG** metrics into their own display family, separate from Quality Games: "Quality Games" now shows only the per-game consistency `-QG%` metrics (xG-QG%, RelxG-QG%, NFI-QG%, RelNFI-QG%), while the new "xG" family shows the rate/relative metrics **xGF/60, xGA/60, RelxG%, RelxG-F%, RelxG-A%**. (3) Surfaced MoneyPuck-style on-ice **xGF/60** and **xGA/60** (on-ice expected goals for / against per 60, aggregated from `per_player_game.csv` by ratio-of-sums over each scope's seasons — `xG_for`/`xG_ag` summed, divided by summed `TOI_on_sec`, ×3600) on the player leaderboard and profile; added purple as a 4th chart line colour and a dual-y-axis combined goalie chart (NFI-GSAx/60 rate on the left axis, consistency %s on the right). Builds on July 6, 2026 (b) — added **QG%s / QG%b**, a tiered save%-based Quality Start metric: each season, goalies are ranked league-wide by GP into a top-32 "starter" tier and next-32 "backup" tier, and a volume-weighted baseline save% is computed for each tier; QG%s / QG%b are the share of a goalie's games where per-game save% clears the starter / backup baseline respectively, computed for every goalie against both baselines (a Streamlit toggle picks which one displays, default starter). Built in two shot scopes — 5v5 ES regulation and all situations — with a Streamlit toggle between them; QNFS%/QGx remain 5v5-only for now. Same date, renamed the goalie metric **GQG → QGx** (name only — same construction, per-game all-shot GSAx ≥ 0; underlying `qs_gsax` files / `QS_GSAx_*` columns unchanged) to free up "Quality Games" language for the new save%-based family and to read as "goals-saved-above-expected, expressed as a game rate." Builds on July 6, 2026 (a) — added **NFI SV%**, a raw (unadjusted) save percentage on the same CNFI ∪ MNFI shots-faced denominator as NFI-GSAx, reported alongside it as a sanity-check stat (not shot-quality adjusted). Same source files as NFI-GSAx; no new qualifying floor. Builds on June 23, 2026 — renamed the goalie metric **QS-GSAx → GQG** (Goalie Quality Games): name only, same construction (per-game all-shot GSAx ≥ 0); underlying `qs_gsax` files / `QS_GSAx_*` columns are unchanged. Builds on June 19, 2026 — added season-level RelxG per-60 rate differential columns (`RelxG_F_pct`, `RelxG_A_pct`, `RelxG_pct`) to `per_player_season.csv` and `per_player_season_team.csv`, mirroring the NFI pipeline's existing `RelNFI_F_pct`/`_A`/`_pct` methodology; same date added Playoff Quality Game build (parallel `_playoffs`-suffixed outputs covering 22-23 through 24-25 with same methodology as regular-season QG, including RelNFI-QG and RelxG-QG); builds on the June 18 addition of Relative Quality Game metrics as parallel team-relative columns; June 12 Goalie Metrics (NFI-GSAx, QNFS%, GQG) with Vollman Quality Starts disambiguation; the June 7 update covered the NFI-QG half-credit tie handling and documented TZI2 as exploratory single-game tooling (not part of the public framework), building on the May 20, 2026 audit + bug fix, the May 2026 audit, and the May 3, 2026 zone-adjustment factor swap.
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

### Relative Quality Game Metrics (RelNFI-QG, RelxG-QG)

Added June 2026 as parallel sensitivity-tested extensions. The absolute NFI-QG and xG-QG metrics above remain the **headline framework**. RelNFI-QG and RelxG-QG measure the same per-game consistency concept but net out team context: instead of comparing the player's on-ice share to a league-wide position median, they compare it to the player's own team's share during the same game *without the player on the ice*.

#### Per-game definition

For each qualifying player-game, derive the team's "without me" baseline:

```
team_NFI_for_wo = team_NFI_for_game − player.NFI_for
team_NFI_ag_wo  = team_NFI_ag_game  − player.NFI_ag
team_NFI_pct_wo = team_NFI_for_wo / (team_NFI_for_wo + team_NFI_ag_wo)

RelNFI_pct_game = NFI_pct_game − team_NFI_pct_wo
```

(`team_NFI_for_game` and `team_NFI_ag_game` are derived from per_player_game.csv via the 5× rollup: each shot at 5v5 ES is attributed to 5 on-ice skaters per side, so summing per-player NFI counters across a team's player rows in a game and dividing by 5 recovers the team's total. Same construction for xG, substituting `xGoal` sums.)

#### Per-game flag

```
RelNFI-QG (half-credit, mirrors absolute NFI rule):
  is_RelNFI_QG = 1.0  if RelNFI_pct_game >  0
               = 0.5  if RelNFI_pct_game == 0
               = 0.0  if RelNFI_pct_game <  0

RelxG-QG (strict >, mirrors absolute xG rule — continuous distribution, ties vanishingly rare):
  is_RelxG_QG = 1.0  if RelxG_pct_game > 0
              = 0.0  if RelxG_pct_game <= 0
```

Threshold is 0 (not 0.5) because the comparison is now "player share vs. team-without-player share" — break-even on Rel means the player played at the level of his linemates, not at league median.

#### Per-season aggregation

```
RelNFI_QG_pct = sum(is_RelNFI_QG) / count(qualifying games with valid RelNFI_pct_game)
RelxG_QG_pct  = sum(is_RelxG_QG)  / count(qualifying games with valid RelxG_pct_game)
```

Same qualifying-game floor as absolute QG (8+ min on-ice AND 5+ on-ice attempts). The `RelNFI_QG_count` column is stored as float (preserving half-credit fractions, same as `NFI_QG_count`); `RelxG_QG_count` can be int since strict-greater produces 0/1 only.

#### How to read Rel-QG values

A player's `RelxG_QG_pct = 0.55` means "in 55% of his qualifying games, his on-ice xG share was higher than his teammates produced during the same game when he was off the ice." Above 0.50 = drives possession beyond what his linemates do; below 0.50 = his teammates outperform him when he's off. By construction the league mean Rel-QG rate centers near 0.50.

#### Headline vs. complementary framing

Absolute QG (NFI-QG, xG-QG) remains the **headline metric** for player evaluation and the public framework. Rel-QG is a **complementary lens** for net-of-team analysis. The two are NOT interchangeable:
- **Absolute QG rewards being in a strong environment.** A median-skill player on COL benefits from Burns/Toews/Makar driving the share even when his individual contribution is small.
- **Rel-QG rewards driving share above teammates' baseline.** A median-skill player on COL looks neutral because his teammates already drive share; a strong player on a weak team (e.g., DET's Larkin, BOS's Pastrnak in 25-26) can look above-average on Rel even when their absolute QG_pct is moderate.

Where they diverge most: dominant-team stars (Burns, Slavin, Makar) lose 8-18pp on Rel because their elite linemates carry comparable share when they're off; weak-team stars (Horvat, DeBrincat) gain 4-6pp because their teammates can't sustain share without them.

Cross-framework convergence: across a 15-player sensitivity test, the Pearson correlation between (cRelNFI_QG − cNFI_QG) and (cRelxG_QG − cxG_QG) is **+0.87** — the two frameworks tell substantially the same team-context story, with material magnitude differences only on specific players (Hagel, M. Tkachuk) whose net-front and territory profiles diverge from their teammates'.

#### Locked spot-check values (Rel-QG, June 2026 sensitivity-test anchors, 4-yr career means)

| Player | cNFI_QG | cRelNFI_QG | cxG_QG | cRelxG_QG |
|---|---|---|---|---|
| Connor McDavid | 0.6861 | 0.625 | 0.6793 | 0.651 |
| Zach Hyman | 0.6890 | 0.626 | 0.6552 | 0.605 |
| Jaccob Slavin | 0.6515 | 0.537 | 0.7005 | 0.521 |
| Evan Bouchard | 0.6860 | 0.616 | 0.6987 | 0.643 |
| Brandon Hagel | 0.6839 | 0.674 | 0.6025 | 0.649 |
| Cale Makar | 0.6165 | 0.540 | 0.5963 | 0.538 |
| Nikita Kucherov | 0.6331 | 0.575 | 0.5938 | 0.579 |
| Brent Burns | 0.6235 | 0.485 | 0.6656 | 0.494 |
| David Pastrnak | 0.5036 | 0.511 | 0.5039 | 0.543 |

±0.01 tolerance for reproduction. Anything more than ±0.01 off indicates the team-totals derivation, the per-game wo computation, or the count-column type cast (RelNFI_QG_count must be float to preserve half-credit ties) didn't propagate correctly.

#### Output columns

Added to `Quality_Games/output/per_player_season.csv` and `per_player_season_team.csv`:
- `RelxG_QG_count` (int), `RelxG_qual_GP` (int), `RelxG_QG_pct` (float)
- `RelNFI_QG_count` (**float** — preserves half-credit), `RelNFI_qual_GP` (int), `RelNFI_QG_pct` (float)

No public-facing post (consistency post, Carolina post) uses Rel-QG metrics. They support future Streamlit work and ad-hoc net-of-team analysis only.

### Season-level RelxG per-60 rate differentials (`RelxG_F_pct`, `RelxG_A_pct`, `RelxG_pct`)

Added June 19, 2026 to `per_player_season.csv` and `per_player_season_team.csv` as parallel columns that mirror the NFI pipeline's existing `RelNFI_F_pct` / `RelNFI_A_pct` / `RelNFI_pct` methodology (`NFI/scripts/build_playoff_data.py:_rel()`): `on60_xG = player xG / player TOI × 3600`, `off60_xG = (team xG − player xG) / (team TOI − player TOI) × 3600`, `RelxG_F_pct = on60_F − off60_F`, `RelxG_A_pct = off60_A − on60_A` (sign flipped so suppression is positive), `RelxG_pct = RelxG_F_pct + RelxG_A_pct`; positive on all three = better. For `per_player_season_team.csv` the team baseline is that team's full-season totals across all games (5× rollup of per-player counters, then `/5`); for `per_player_season.csv` the player's pooled counters and the team baseline are summed across all team stints in the season — a 1-team player gets a single-team baseline, a traded player gets the combined A+B baseline (strict parity with how the existing RelNFI build pools at the season level). No minute floor beyond the existing `qualifying_GP` filter; emits NaN when `off_TOI <= 0` or `player_TOI <= 0`. Verified against an independent recompute from `per_player_game.csv` for McDavid (24-25: +0.80), Hyman (23-24: +0.98), Burns (23-24: −0.04), Makar (24-25: +0.33), Hagel (24-25: +1.07) — exact match to four decimals.

**Distinction from MoneyPuck's published relative-xG metrics.**

MoneyPuck publishes player-level relative xG metrics (`xGoalsForPercentageRel` and similar columns) computed using their own methodology. HockeyROI's RelxG_pct, RelxG_F_pct, and RelxG_A_pct use the same input data (MoneyPuck's per-shot xGoal values) but apply the HockeyROI methodology stack downstream:

- Same qualifying-game filter as the absolute QG framework: 8+ minutes on-ice AND 5+ on-ice shot attempts per game
- Same situation code as flagship metrics: 5v5 even-strength regulation only
- Same per-60 rate differential methodology as RelNFI: on60 minus off60 for attack, off60 minus on60 for suppression (positive on all three = better)
- Same pooled-across-stints aggregation for traded players, matching the RelNFI methodology in build_playoff_data.py
- Same season scope as all other QG metrics (22-23, 23-24, 24-25, 25-26 regular season; 22-23 through 24-25 playoffs)

MoneyPuck's published relative-xG columns use their own qualifying filters, their own aggregation methodology, and their own season scope. For these reasons, HockeyROI's RelxG values may differ from MoneyPuck's published relative-xG numbers for the same player-season. Neither is more accurate — they are calibrated to different purposes. The HockeyROI methodology is locked to ensure consistency with all other framework outputs, so cross-column comparisons within HockeyROI are methodologically valid.

**RelNFI is HockeyROI-original.**

No third-party publisher provides equivalent net-front impact relative metrics. RelNFI_pct, RelNFI_F_pct, and RelNFI_A_pct are computed in-pipeline using NHL play-by-play shot events filtered to the CNFI and MNFI zones (Fenwick-based, excluding blocked shots per the May 2026 audit decision). The same qualifying filter and per-60 rate-differential methodology apply as for RelxG.

### Playoff Quality Game Metrics

Added June 2026 as a parallel build alongside regular-season QG. Playoff QG values are written to `_playoffs`-suffixed files in `Quality_Games/output/`; regular-season files are unaffected.

#### Methodology

Playoff QG uses the **same methodology** as regular-season QG with no rule changes:
- Same per-game qualifying floor (TOI ≥ 480 sec AND on-ice attempts ≥ 5)
- Same NFI-QG half-credit tie rule (`>` = 1.0, `==` = 0.5, `<` = 0.0)
- Same xG-QG strict-greater rule (`>=` = 1.0)
- Same RelNFI-QG (half-credit) and RelxG-QG (strict-greater) team-relative flags
- Same 5× rollup to derive team-game totals from per-player on-ice counters

#### Position medians

Playoff position medians are **computed fresh from playoff data** in each run by script 02's `qual_known.NFI_pct_game.median()` and `xG_pct_game.median()` calls. Playoff medians are not inherited from the regular-season `position_medians.csv` — they're written to `position_medians_playoffs.csv` and used for that scope only. This is necessary because playoff samples have smaller denominators (median ~5-7 NFI events per game) and slightly different positional distributions.

#### Scope

The playoff build covers seasons **2022-23, 2023-24, 2024-25** (3 seasons, 262 unique games). 2025-26 playoff data is intentionally excluded from script 01's playoff scope because the canonical HR zone source (`shots_tagged.csv`) does not yet contain 2025-26 playoff records; MoneyPuck `shots_2025.csv` carries them but the MP↔HR join would fail the >= 99% match gate. When the HR pipeline ingests 25-26 playoffs, that season can be added by removing the `MP_SEASONS = [2022, 2023, 2024]` restriction in script 01's `_IS_PLAYOFF` branch.

#### Sample sizes are smaller

Playoff per-player-season GP counts are much smaller than regular season (8-25 GP per player vs. 80+):

| Season | Games | Player-game rows | Unique players |
|---|---|---|---|
| 22-23 | 88 | 3,168 | 334 |
| 23-24 | 88 | 3,168 | 339 |
| 24-25 | 86 | 3,097 | 333 |

A pooled `all_playoffs` row per player is generated in `per_player_season_playoffs.csv` (sum of counts across playoff seasons), giving a 3-season pooled view. Analogous `all_playoffs` rows are created for `per_team_season_playoffs.csv` (TOI-weighted across each team's playoff seasons).

**Minimum-game filters should be applied at the analysis layer, not the data layer.** Playoff GP is naturally short and the qualifying floor is intentionally non-stringent (no 20-GP team-eligibility floor in playoff scope; script 02 drops it via `TEAM_GP_FLOOR = 1` under `_IS_PLAYOFF`). Streamlit and ad-hoc consumers should apply their own GP minimums (e.g., 10 GP for deep-run players, all_playoffs row for cross-season pooling).

#### File locations

```
per_player_game_playoffs.csv
per_player_season_playoffs.csv
per_player_season_team_playoffs.csv
per_team_season_playoffs.csv
team_trajectories_4season_playoffs.csv
position_medians_playoffs.csv
```

All under `Quality_Games/output/`. The regular-season `_playoffs`-unsuffixed files are not affected by playoff runs.

#### Sample playoff values (sanity reference)

These five spot-checks anchor the playoff build to known deep-run cases. Replication should land within rounding (±0.005):

| Player | Season | GP | qGP | NFI_QG_pct | RelNFI_QG_pct | xG_QG_pct | RelxG_QG_pct |
|---|---|---|---|---|---|---|---|
| Connor McDavid | 22-23 | 12 | 12 | 0.7500 | 0.4583 | 0.5833 | 0.6667 |
| Sebastian Aho | 22-23 | 15 | 15 | 0.5000 | 0.3667 | 0.6667 | 0.5333 |
| Jaccob Slavin | 22-23 | 15 | 14 | 0.7143 | 0.5714 | 0.7143 | 0.5000 |
| Sam Reinhart | 23-24 | 24 | 24 | 0.6250 | 0.5000 | 0.6667 | 0.5417 |
| Connor McDavid | 24-25 | 22 | 22 | 0.5909 | 0.7500 | 0.6818 | 0.7727 |

McDavid's 24-25 playoff RelxG=0.7727 is the framework working as intended: he drove on-ice xG share 7-8pp above his teammates' baseline in 77% of games during EDM's run, even though his absolute xG_QG of 0.6818 sits well below his regular-season pace.

#### Running playoff scope

```
QG_SCOPE=playoff python3 Quality_Games/scripts/01_build_per_player_game.py
QG_SCOPE=playoff python3 Quality_Games/scripts/02_quality_game_aggregation.py
```

Both scripts default to regular-season scope when `QG_SCOPE` is unset; playoff scope is opt-in and writes only to suffixed files.

---

## Situations (all game states)

The **Situation** toggle recomputes the on-ice possession/xG/individual suite for a chosen strength state, exposed on the Player leaderboard (the `Sit …` metric family) and as a **Situation splits** table in every player drill-in and the Trade Analyzer.

### Buckets

Granular skater matchups are rolled into display buckets from each player's own-team perspective:

| Bucket | Matchups |
|---|---|
| 5v5 | 5v5 |
| PP (power play) | 5v4, 5v3, 4v3 |
| PK (penalty kill) | 4v5, 3v5, 3v4 |
| 4v4 | 4v4 |
| 3v3 | 3v3 (regular-season OT) |
| 5v3 | 5v3 |
| All situations | every matchup |

### Construction

Built by `NFI/scripts/build_situation_onice.py` (on-ice counts → `Data/player_situation_onice.csv`) and `NFI/scripts/build_situation_toi.py` (per-situation TOI → `Data/player_situation_toi.csv`) from the raw shot events + shift data:

- **Strength state** for each event/segment comes from the event `situation_code` (skater counts per team), labeled from the player's own-team perspective (a 5v4 for the power-play team is a 4v5 for the killers).
- **On-ice attribution**: every Corsi event is credited to the players whose shift intervals overlap the event time — for-side to the shooting team's skaters, against-side to the defending team's.
- **TOI per situation** is reconstructed from the same piecewise-constant strength segments intersected with shifts (including overtime, so 3v3 is captured), and reconciles exactly with the standalone TOI file.
- **Rates** are ratio-of-sums over the scope's seasons and the bucket's matchups: `sum(counts) / sum(TOI) × 60`; shares are `CF% = CF/(CF+CA)`, `xGF%`, etc. xG is the **HockeyROI model** (`build_xg.py`), not MoneyPuck.

### Columns

Per-60: `CF/CA, FF/FA, xGF/xGA, GF/GA`; shares `CF%/FF%/xGF%/GF%`; individual `iCF/ixG/iG` per-60 (+ `ixG`/`iG` totals). Two dedicated special-teams value scores (NST-style): **PP xGF+CF/60** (higher = better power-play offense) and **PK xGA+CA/60** (lower = better penalty-kill defense).

### Scope and caveats

- The bespoke 5v5-native families — **RelNFI, Quality Games, Zone Impact** — are **not** re-derived per situation. Their team-relative (on/off), per-game-median, and faceoff-anchored constructions assume even strength, so they stay on the 5v5 basis and are labeled accordingly outside 5v5.
- Per-situation **league totals** carry an on-ice roster-size multiplier (5 for-skaters vs 4 against-skaters on a power play) plus shift-reconstruction noise, so player comparison should use the per-60 **rates** and **shares**, not raw summed totals. 5v5 for/against totals are league-symmetric to <0.2% as a correctness check.

Validation (2026-07-22): 5v5 league ΣxGF≈ΣxGA and ΣGF≈ΣGA to <0.2%; McDavid 2025-26 5v5 xGF% 55.9 with individual ixG 23.9 ≈ 24 actual goals; PP xGF/60 leaders and PK lowest-xGA/60 lists match expectation.

---

## Zone Impact

### What Zone Impact measures

Zone Impact is a player-evaluation framework measuring how a player's on-ice deployment after each type of faceoff translates into offensive-zone time. Its four published metrics — OZI, DZI, NZI, TZI — are independent lenses, not a hierarchy. A complete player rates above average on all four.

- **OZI — Offensive Zone Impact.** Share of offensive-zone time on shifts that begin with an offensive-zone faceoff. Captures whether OZ starts get converted to sustained pressure.
- **DZI — Defensive Zone Impact.** Share of offensive-zone time on shifts that begin with a defensive-zone faceoff. Captures whether the player escapes their own zone cleanly.
- **NZI — Neutral Zone Impact.** Share of offensive-zone time on shifts that begin with a neutral-zone faceoff. Captures transition play.
- **TZI — Transitional Zone Impact.** The neutral-zone transition *split*: offensive-zone time **minus** defensive-zone time on shifts that begin with a neutral-zone faceoff. Reads how a player tilts play out of the neutral zone (positive = pushes toward the O-zone, negative = gets pushed back). Internally this is the metric previously labelled TNZI.

Zone Impact metrics are **zone-time share** measures, not shot-differential. The earlier description in this document as "Fenwick-based shot differential" was inaccurate — corrected May 2026 to match what the code computes.

### The 0–100 index (50 = average)

Each metric is published as a **0–100 index where 50 is the position-group league average** — above 50 means more offensive-zone time (or, for TZI, more forward tilt) than an average forward/defenseman, below 50 means less. It is built by taking each player's raw per-shift zone-time percentage and **recentring it on the league-average percentage for their position**: `index = 50 + (player% − league-average%)`, clipped to [0, 100], forwards and defense normalised separately. The natural spread of the underlying (bounded) percentage sets the spread of the index — there is **no artificial stretch** — so most players sit in a tight band around 50 and only genuine outliers approach the extremes, the same way NHL save percentages cluster between roughly .880 and .920 rather than spanning 0–100. Because each scope (single season, 2-year, 4-year pool, playoffs) is recentred on *its own* average, 50 always means "average for that scope." Built by `Zones/scripts/build_zone_index100.py` (regular seasons + pools) and `build_zone_index100_playoffs.py` (playoff pool), reusing the exact V1 event/faceoff-anchor methodology of `compute_zone_variations.py`.

### Construction

1. For each shift, capture player on-ice time and the zone of each puck event.
2. Bucket each shift by its starting faceoff zone (NZ, OZ, DZ).
3. Within each bucket, sum seconds in each zone; compute share of time spent in the offensive zone.
4. Apply Wilson interval shrinkage to handle small-sample variance.
5. Position-normalize each metric within its group (forwards vs forwards, defense vs defense) by recentring on the group's league-average percentage → the 0–100 index (50 = average) described above. (For TZI the underlying quantity is OZ%−DZ% off neutral draws rather than a single zone share.)

### Why four metrics rather than one

Different deployment contexts produce different ice-tilt patterns for the same player. A defensively-deployed player who excels at DZ exits may have strong DZI but unremarkable OZI; a finisher who feasts on OZ starts may show the inverse profile. Collapsing these into a single "zone-impact" number loses the deployment-specific signal.

### Sample thresholds

- 50 faceoff-start shifts minimum per metric (NZ-FO for NZI and TZI, OZ-FO for OZI, DZ-FO for DZI)
- 20 games played minimum (regular-season scopes; the playoff pool waives the GP floor and qualifies on the 50-shift gate alone, since playoff samples are short)
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

### D/N/O Start% (presentation layer, not a new metric)

Added 2026-07: a per-player faceoff-**start**-zone breakdown — the plain share of a player's faceoff-started 5v5 shifts that began in the defensive / neutral / offensive zone (`DZ Start%` / `NZ Start%` / `OZ Start%`). Same play-by-play data and same faceoff-started-shift basis as DZI/NZI/OZI, shown alongside them on the Player List table and in a companion scatter chart (D-zone starts vs O-zone starts, one point per player). This is descriptive only — it does not feed into, adjust, or replace DZI/NZI/OZI. Source: `Zones/output/zone_time_raw.csv`, pooled across all regular seasons (no per-season or playoff breakdown exists for this specific cut yet).

---

## PDO

Added 2026-07: a descriptive shooting%/save% luck proxy, shown as a raw column beside xG on the Player List (no relative or Quality-Games version; not used for ranking).

### Construction

`SH% = on-ice goals-for ÷ on-ice shots-on-goal-for` (5v5 or all-situations, toggle-able — the toggle affects PDO only, no other xG-group column).
`SV% = 1 − (on-ice goals-against ÷ on-ice shots-on-goal-against)`.
`PDO = (SH% + SV%) × 100`.

Shots-on-goal based (`event_type in {shot-on-goal, goal}`), **not** Fenwick or Corsi — the conventional PDO definition used across the analytics community (Natural Stat Trick, Evolving Hockey, etc.), as opposed to a Fenwick/Corsi-denominator variant some sites compute instead.

On-ice attribution is a fresh pass (`NFI/scripts/build_pdo_sog.py`) that reuses the exact validated shift-join / 5v5-state-derivation logic from `03_onice_attribution_pillars.py` (copied rather than imported, since that script executes top-to-bottom and would rewrite canonical NFI outputs as a side effect). On-ice goals-for/against and TOI are reused as-is from the existing `player_counts_by_state_zone_per_season.csv` — only the on-ice SOG-for/against counters are newly computed. Floor: ≥200 min TOI in the selected scope (5v5 or all-situations).

**Playoffs:** `NFI/scripts/build_pdo_sog_playoffs.py` — the same logic filtered to playoff games and pooled into a single all-playoffs scope per player (same ≥200-min floor, applied to the pooled multi-year total). Output: `NFI/output/player_pdo_5v5_playoffs.csv` / `player_pdo_allsit_playoffs.csv`.

### Known limitation

`NFI/Geometry_post/Data/shift_data.csv` is missing shift-chart data for a small number of games at the end of both the 2024-25 (57 games) and 2025-26 (6 games) regular seasons — confirmed as a gap in NHL's own shift-chart API (not a scraping bug on our end; the same endpoint returns complete data for every other game). This silently drops the affected game(s) from any player who played in them, for PDO and for the pre-existing Corsi/Fenwick on-ice rates alike. Sanity-checked at the team level (every 5v5 goal must be shared by exactly 5 on-ice skaters per side) — beyond the known missing games, no further gaps were found.

### PDO vs Natural Stat Trick

Spot-checks against Natural Stat Trick found meaningful player-level gaps in some cases (e.g. several percentage points on individual skaters) that the known shift-data gap does not fully explain by itself — verified via an independent from-scratch re-derivation from raw shot events + shift data, which reproduced this pipeline's own numbers exactly, and via team-level goal reconciliation, which showed no broader attribution bug. The remaining gap's source is unresolved; it may reflect a genuine difference in underlying shot/shift data between this pipeline's NHL API pull and Natural Stat Trick's source. Treat PDO as descriptive and directionally useful, not as a value guaranteed to reconcile exactly with third-party sites.

### PDOxG

PDO's luck signal, net of shot quality. Standard PDO treats every on-ice shot as an equal-quality attempt, so a player whose team consistently generates/allows better chances will run a high (or low) PDO from shot quality alone — not luck. PDOxG isolates the actual luck component by netting SH%/SV% against an xG model (shot distance/angle/type):

`PDOxG = (SH% − xSH%) + (SV% − xSV%)`, where `xSH% = on-ice xGF ÷ on-ice SOG-for` and `xSV% = 1 − (on-ice xGA ÷ on-ice SOG-against)`. Reported ×100, 0-centered (positive = finishing/goaltending running hotter than shot quality predicts, negative = colder). Same 5v5/all-situations toggle and ≥200-min floor as PDO. Descriptive, not a ranking — note it does **not** decompose PDO exactly, since PDO's SH%/SV% are Fenwick/Corsi-free (shots-on-goal-based) while the xG model underneath PDOxG is itself derived from a broader shot-attempt set; treat the two as related but not arithmetically reconcilable to the decimal.

---

## NHL EDGE

Added 2026-07: NHL's own player-tracking data (radio-frequency + camera-based, tracking player position — not puck position). Shown on the Player List table as a separate "EDGE" metric family, source-labeled and never blended into TZI/NFI.

### What it measures

- **Zone time %** — OZ/NZ/DZ time share while the player is on the ice. OZ% has a toggle-able even-strength/all-situations scope (NHL only publishes an even-strength split for the offensive-zone stat specifically; NZ%/DZ% have just the one all-situations number regardless of the toggle). The toggle **defaults to Even Strength** so EDGE OZ% (and EZI, below) sit on the same 5v5 basis as the rest of the page (OZI/DZI/NZI/TZI and OZ Start%).
- **Top skating speed** (mph) — the player's single fastest recorded moment that season.
- **Speed bursts (20+ mph)** — count of times the player exceeded 20 mph, shown as NHL's raw season total (not a per-60 rate). No finer speed bands are published for skating bursts (unlike shot speed, which NHL does band). A per-60 rate was tried and reverted (2026-07): `speed_bursts_over_20mph` is an all-situations count (PP+PK+ES), but the only ice-time this app has to divide by is ES-only, so a rate over that denominator inflates anyone with real special-teams time — there's no season-scoped all-situations TOI source in the data to build a clean matching rate.
- **Distance skated** (miles) — total for the season. Also shown as **EDGE Distance/60** — distance ÷ ES TOI minutes × 60, same ratio-of-sums construction (same all-situations-numerator/ES-only-denominator caveat as bursts above, not yet revisited).
- **Speed-Bursts-vs-Top-Speed scatter** — rendered as TWO charts, Forwards and Defense, always both shown. NHL computes its own EDGE speed/burst percentiles and league averages WITHIN position group (its published league-average top speed is 22.17 mph for F vs 21.59 mph for D), so a single mixed crosshair across both positions could show a genuinely above-average defenseman as "below average." Each chart's dashed crosshair is that position group's own average, matching NHL's basis.

### Source and scrape

`https://api-web.nhle.com/v1/edge/skater-detail/{playerId}/{season}/{gameTypeId}` — discovered via live network inspection of `www.nhl.com/nhl-edge/skaters/{slug}` (the old standalone `edge.nhl.com` site 301-redirects there under "EDGE 2.0"). Same `api-web.nhle.com` host the rest of this pipeline already uses — not Sportradar. `edge/scripts/pull_edge_stats.py` pulls all (player_id, season) pairs from `NFI/output/player_counts_by_state_zone_per_season.csv` for 2021-22 → 2025-26, regular season and playoffs. See `edge/README.md` for the full scrape write-up.

### Basis mismatch vs TZI (NZI/DZI/OZI) — read this before comparing the two

| | EDGE | TZI (NZI/DZI/OZI) |
|---|---|---|
| Tracked by | player position (chip/camera) | puck position (event/PBP-derived) |
| Scope | all-situations (or EV, OZ only) | strict 5v5 |
| Trigger | continuous TOI | faceoff-started shifts only |

EDGE zone-time% and TZI's NZI/DZI/OZI **measure genuinely different things** and must never be read as the same metric — the app's EDGE section carries a mandatory on-screen banner stating this.

### Data-quality limitation

The EDGE API only exposes pre-aggregated season totals plus a single "best game" highlight per stat — there is no per-game log to inspect or filter, so a per-game tracking-failure exclusion (originally planned) isn't possible from outside NHL's system. Season totals are taken exactly as NHL computed them.

### Display convention

Each EDGE value shows a computed **(league / team) rank**, not NHL's own percentile — matching every other ranked column in the app. Pooled/2yr views are a games-played-weighted average across the player's available seasons (including for the rank basis), since NHL doesn't expose enough to recompute a true multi-season number.

### EZI — EDGE Zone Impact

Added 2026-07: a 0–100 index (50 = position-group average) that relates EDGE's O-zone **time** to my PBP O-zone faceoff **starts**, to surface players who generate more O-zone time than their deployment alone would predict.

**Construction — a regression RESIDUAL, not a straight difference.** A first version used `raw = EDGE OZ time% − OZ Start%` (a plain percentage-point difference — chosen over `EDGE OZ time% − (DZ Start% + NZ Start%)`, since with OZ+DZ+NZ Start% always summing to 100% that alternate form is algebraically identical to `(EDGE OZ time% + OZ Start% − 100)`, which rewards a player high on *both* axes rather than isolating the mismatch of interest). That plain-difference version turned out to be badly biased: `OZ Start%` has enormous deployment-driven spread (std ≈ 6.7, range 8–53) while `OZ time%` barely moves (std ≈ 2.4, range 35–50.5) — so a straight subtraction is dominated almost entirely by the (high-variance) starts term. Verified empirically: the plain difference correlated **−0.94** with `OZ Start%` itself, and an OLS fit of `OZ time% ~ OZ Start%` gave a slope of only **≈0.23–0.25** (a 40-point gap in starts predicts only a ~9–10-point gap in time, not the full 40 the plain subtraction assumes) — so the naive version was mostly just an inverted deployment metric, and its leaderboard was dominated by low-event defensive players with essentially no offensive talent represented.

**Fix:** fit a position-group-specific OLS regression of `OZ time%` on `OZ Start%`, and use the **residual** as `raw`:

`raw = OZ time% − (intercept + slope × OZ Start%)`

fit separately for forwards and defense (least squares, `OZ Start%` as the sole predictor, fit over players clearing the ≥200 ES-min floor). By construction an OLS residual is exactly uncorrelated with the predictor — verified: correlation with `OZ Start%` is **0.0000** — so `raw` isolates "time beyond what your own starts predict" rather than restating deployment. The resulting leaderboard is a genuine mix of offensive stars and possession-driving depth players across the full range of `OZ Start%`.

- **Positive** → converts O-zone time beyond what the starts alone predict (driving play beyond deployment).
- **Negative** → O-zone starts aren't converting into O-zone time (sheltered but not producing).
- **Near zero** → time tracks the position-group's normal starts→time relationship.

Recentred exactly like OZI/DZI/NZI/TZI: `EZI = clip(50 + (raw − position-group average raw), 0, 100)`, forwards and defense normalised separately, natural spread (no artificial stretch) — though since OLS residuals already average to ≈0 within the fitted population, this step mainly keeps EZI on the same 0–100/50-average scale as the rest of the Zone Impact family rather than doing meaningful additional recentring. The position-group average (and the regression fit itself) use only players clearing a ≥200 ES-min stability floor; every player with both source columns still gets a displayed EZI value regardless of their own TOI.

**Basis:** EZI uses the **even-strength** EDGE O-zone time% (following the EDGE OZ% scope toggle, which defaults to EV) so the time term shares the 5v5 basis of `OZ Start%` — if all-situations O-zone time were used against 5v5 starts, power-play O-zone shelter would leak into the residual and read as play-driving. Switching the toggle to all-situations re-fits EZI on that basis too.

**Availability:** regular season only (single season, 2yr, 4yr pool) — playoffs has no zone-start data, so EZI isn't computed there. Team-scoped and league-wide scatters (EDGE O-zone time% vs D/N-zone Start%, the two raw ingredients) sit alongside the leaderboard column so a mismatch is visible directly, not just collapsed into the single EZI number.

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

## Goalie Metrics: NFI-GSAx, NFI SV%, QNFS%, QGx, QG%s / QG%b

### Disambiguation: QGx / QNFS% vs. QG%s / QG%b vs. conventional Quality Starts

Read this before mapping any of these metrics to a conventional goalie-consistency measure. Robert Vollman's Quality Starts metric (~2009) is binary on **save percentage**: a quality start is a game where the goalie's save% exceeds league-average save% (with a small adjustment for high-shot-volume games). Anyone in hockey analytics who hears "Quality Starts" will map to that definition. HockeyROI has **four** goalie consistency metrics, and they split into two families on exactly the dimension Vollman's definition turns on — save% vs. expected-goals:

- **QGx and QNFS% are GSAx-based, not save%-based.** A quality game is per-game GSAx ≥ 0 — the goalie beat expected on a danger- or xG-weighted basis — **not** save% > league average. QNFS% uses net-front (CNFI ∪ MNFI) shots only; QGx uses all shots faced. Neither is Vollman Quality Starts under a different name — the earlier rename (QS-GSAx → QGx, via GQG) deliberately drops the "Quality Start" label to avoid exactly this confusion. Do not conflate either with Vollman's metric.
- **QG%s / QG%b ARE save%-based, in the spirit of Vollman** — but they replace his single, fixed league-average line with two population-specific baselines recomputed every season (see below). This is the metric to reach for if you want something recognizable as a traditional Quality Start; QGx/QNFS% are not that.

Don't map QGx or QNFS% to Vollman's metric, or treat any of the four as interchangeable with each other — each is built on a different threshold, shot scope, or baseline population.

### NFI-GSAx

**What it measures:** per-(goalie, season) goals-saved-above-expected on **CNFI ∪ MNFI shots only**. The expectation baseline is an **internal per-season, per-zone league goal rate** — within each season the league's goals-per-shot-faced is computed separately for the CNFI and MNFI zones, and each goalie's expected goals is `faced_CNFI × rate_CNFI + faced_MNFI × rate_MNFI`. GSAx is that expectation minus goals allowed on net-front shots. (This is a HockeyROI-internal model, **not** the MoneyPuck xGoal model — the xGoal model is used only by QGx, below. QG%s/QG%b also read from MoneyPuck's shot log, but only the raw shot/goal outcome, not the xGoal column, since save%-based metrics don't need a shot-quality model.) The faced denominator is save-based — shots-on-goal and goals (missed shots excluded), distinct from QNFS%'s Fenwick base. Per-60 normalization allocates each goalie's pooled even-strength TOI across seasons in proportion to the share of their career net-front faced shots that fell in each season (shifts data is not season-keyed, so exposure is apportioned by faced-shot share).

**Why net-front only:** NFI's structural insight is that the immediate net-front and high slot are where shot location actually predicts conversion. Restricting GSAx to those shots puts the goalie metric on the same spatial basis as the player- and team-level NFI framework — they are commensurable in a way that all-shot GSAx and NFI% are not.

**Qualifying:** minimum 100 net-front shots faced per season for the per-season table; minimum 300 net-front shots faced across pooled seasons for the pooled table.

**Confidence intervals:** none are currently reported. NFI-GSAx per-60 is a rate count over a fixed exposure window; were CIs to be added they would use the Poisson (Garwood) interval, per the rate-vs-proportion rule established in the May 20, 2026 audit (see `docs/AUDIT_2026-05-20.md`). The present outputs carry point estimates only.

**Source:** `NFI/scripts/21_goalie_gsax_by_season.py` → `NFI/Output/goalie_nfi_gsax_by_season.csv` (per-season, 354 goalie-seasons) and `NFI/Output/goalie_nfi_gsax_pooled_v2.csv` (pooled, 84 goalies). The pooled file currently spans five seasons (2021-22 → 2025-26), one more than QNFS%/QGx.

### NFI SV%

**What it measures:** raw (unadjusted) save percentage on the same **CNFI ∪ MNFI shots-faced** set as NFI-GSAx — `(total_faced − total_goals) / total_faced` per goalie-season (and pooled/playoff equivalents). It is computed from the same `total_faced`/`total_goals` counts that feed NFI-GSAx, not a separate build.

**Why it exists:** NFI-GSAx is expectation-adjusted (goals saved relative to the league's per-zone scoring rate); NFI SV% is not — it does not account for shot difficulty within the CNFI/MNFI zones. It exists as a sanity-check/complement to NFI-GSAx, not a replacement, and is **not** the save% referenced in the Vollman Quality Starts disambiguation above (that's all-shot save%; NFI SV% is net-front-only).

**Qualifying:** same cohort and floors as NFI-GSAx (≥100 net-front shots faced per season; ≥300 pooled) — there is no separate qualifying rule.

**Source:** same producers and output files as NFI-GSAx (`21_goalie_gsax_by_season.py`, `21p_goalie_gsax_playoffs.py`, `22_pool_goalie_gsax.py`); column `NFI_save_pct` (fraction, 0–1) alongside `total_goals` in each output file.

### QNFS%: Quality Net-Front Save percentage

**What it measures:** the share of a goalie's 5v5 ES regulation appearances in which their per-game net-front (CNFI ∪ MNFI) GSAx ≥ 0. Each game yields a binary indicator (1 if NF GSAx ≥ 0, else 0); the season-level metric is the rate of 1-games over qualifying games. The per-game expectation uses the same internal per-season per-zone league goal rate as NFI-GSAx, but on a **Fenwick** base (shot-on-goal + missed-shot + goal). Wilson 95% confidence intervals are reported, because the underlying statistic is a proportion bounded in [0, 1], not a per-60 rate.

**Qualifying:** a goalie-game enters QNFS% only if the goalie faced ≥ 3 net-front shots at 5v5 ES regulation in that game. A goalie qualifies for season-level reporting with ≥ 25 qualifying games in any single season within the window.

**Interpretation:** QNFS% captures *consistency of beating expected on net-front shots*, complementing season-aggregate NFI-GSAx, which captures aggregate dominance. A goalie can post high NFI-GSAx but middling QNFS% (a few huge games against expected amid many forgettable ones), or the reverse.

**Source:** `NFI/goalie_consistency/scripts/compute_qnfs.py` and `compute_qnfs_per_season.py`; outputs `NFI/goalie_consistency/output/qnfs_2022-2026.csv` (146 goalies, of which 82 qualified) and `qnfs_per_season_2022-2026.csv` (294 goalie-seasons). Window: four seasons, 2022-23 → 2025-26.

### QGx: Goalie Quality Games (GSAx-based)

**Name:** QGx was previously labeled **GQG** (Goalie Quality Games), and before that **QS-GSAx**. It is the Quality-Start idea computed on **GSAx** (per-game GSAx ≥ 0) rather than on raw save% — same metric both times, renamed June 23, 2026 (QS-GSAx → GQG) and again July 6, 2026 (GQG → QGx, to read as "goals-saved-above-expected, as a game rate" and to free up "Quality Games" language for QG%s/QG%b below). The underlying data columns/files keep their `qs_gsax` / `QS_GSAx_*` names throughout.

**What it measures:** the share of a goalie's 5v5 ES regulation appearances in which their per-game **all-shot** GSAx ≥ 0. Same binary-indicator + season-rate construction as QNFS%, but on all shots faced rather than net-front only, and using the **MoneyPuck xGoal model** for the per-shot expectation (per-game GSAx = Σ xGoal − Σ goals). Wilson 95% confidence intervals; reported with both the point estimate and a Wilson lower-bound rank.

**Qualifying:** minimum 10 shots faced per game; minimum 25 qualifying games per season.

**Why QGx exists alongside QNFS%:** the two measure related but non-identical things. Spearman ρ between the two metrics' Wilson lower bounds (QS_GSAx_lo vs QNFS_lo) across the n = 80 qualified-goalie overlap is 0.810 — they rank goalies similarly, but ~34% of rank variance is unshared (1 − 0.810² = 0.344). Net-front-only QNFS% penalizes failures on the highest-leverage shots more sharply; all-shot QGx captures a goalie's full expected-vs-actual ledger. Reporting both keeps the framework honest about which shot scope drives which result.

**Source:** `NFI/goalie_consistency/scripts/compute_qs_gsax.py`; outputs `NFI/goalie_consistency/output/qs_gsax_2022-2026.csv` (81 goalies) and `qs_gsax_per_season_2022-2026.csv` (210 goalie-seasons). Window: four seasons, 2022-23 → 2025-26.

### QG%s / QG%b: Tiered Save%-Based Quality Starts

> **App display note (July 6, 2026 d):** the Streamlit app now surfaces only the **starter** tier, labeled **sQS%** (Starter Quality Start); the backup tier (QG%b) and the starter/backup toggle were removed from the UI. The tiered build below is unchanged and still computes both `QG_pct_s` and `QG_pct_b`; the sections below document the full build.

**Why this exists:** the standard "Quality Start" (Vollman, ~2009) grades every goalie against one league-average save% line. Two problems: the line moves every season (recent 5v5 league save% has drifted from ~91.8% in 2022-23 down to ~90.9% in 2025-26; all-situations from ~90.1% down to ~89.0%), and a single line blends two very different jobs — a goalie who starts 60+ games and a true backup starting 15-20 face different workloads and, empirically, save at different clips. Grading a backup against a workhorse #1's bar sets them up to fail. QG%s / QG%b fix both problems: two baselines, recomputed fresh every season.

**Tiering (per season, per shot scope):** rank every goalie who logged ≥1 qualifying game that season by GP, descending (ties broken by season shots faced, then goalie_id). The top 32 by GP are that season's **starter** tier; the next 32 are the **backup** tier; the rest are unused "depth." Tier assignment is a ranking exercise only — it does not gate which goalies QG%s/QG%b are computed for (see next point).

**Baselines:** for each tier, baseline save% = **total saves ÷ total shots-on-goal faced** across every goalie in that tier that season — a volume-weighted league save%, not an unweighted average of goalies' individual save%s. This matches how "league-average save%" is conventionally computed.

**The metric:** QG%s = share of a goalie's qualifying games where per-game save% ≥ that season's **starter** baseline. QG%b = share of a goalie's qualifying games where per-game save% ≥ that season's **backup** baseline. Both are computed for **every** goalie regardless of which tier they themselves fall in — a struggling starter's QG%b answers "would this goalie be a good backup?", and a hot backup's QG%s answers "is this goalie performing like a starter?" The Streamlit toggle only changes which of the two is displayed (default: starter); both are always in the data.

**Per-game save%:** saves ÷ shots-on-goal (SHOT + GOAL events), explicitly **excluding missed shots** — a missed shot never reaches the goalie, so including it in the faced denominator inflates save% (this was caught and fixed during development: including misses pushed 5v5 baselines from a realistic ~91% to an inflated ~94%). This is why QG%s/QG%b use a different shot-set convention than QGx/QNFS%, which correctly use the full Fenwick set since the xG model prices shot quality in.

**Shot scope — built twice, with a toggle:**
- **5v5 ES regulation** — matches the scope used elsewhere in the goalie consistency pipeline (QNFS%, QGx). 5v5 save% runs materially higher than all-situations (5v5 excludes PK shots-against, which are higher-danger), so 5v5 baselines read ~90.9%–91.8% across the four seasons.
- **All situations** (5v5 + PP + PK), regulation — matches the conventional meaning of "league-average save%" most people have in mind (recent seasons ~89.0%–90.9%). This is the scope closest to Vollman's original definition.

Streamlit carries a shot-scope toggle for QG%s/QG%b specifically, next to the Season/Game-type filter on the Goalies tab. **QNFS% and QGx are 5v5-only and are not affected by this toggle** — a note is shown on the Goalies tab to that effect.

**Qualifying:** minimum 10 shots faced per game (same floor as QGx); minimum 25 qualifying games per season for a goalie-season to be "qualified" for ranking (same floor as QNFS%/QGx). No half-credit ties (`>=` rule); the tie rate against either baseline is ~0% in practice since 10+ shot-count denominators rarely land on the exact same fraction as a volume-weighted league baseline.

**Source:** `NFI/goalie_consistency/scripts/compute_qg_tiered.py`; outputs per scope (suffix `""` for 5v5, `"_allsit"` for all situations): `qg_tier_baselines_by_season{suffix}.csv` (season × tier baselines), `qg_savepct_per_season_2022-2026{suffix}.csv` (per goalie-season), `qg_savepct_2022-2026{suffix}.csv` (pooled 4-season, 145 goalies, 78 qualified in both scopes). Window: four seasons, 2022-23 → 2025-26.

**Playoffs:** built via `compute_qg_tiered_playoffs.py` → `qg_savepct_playoffs{suffix}.csv` (per playoff season plus an `all_playoffs` pooled row, matching the rest of the playoff pipeline's convention — no goalie-level qualifying floor). Playoff games do **not** get a fresh playoff-only starter/backup tier split — playoff rosters are too starter-skewed for a stable top-32/next-32 GP ranking on a playoff-only sample (a true backup often plays 0-2 games). Instead, each playoff game is graded against **that season's regular-season baseline** (read from `qg_tier_baselines_by_season{suffix}.csv`), pooled the same way as the regular-season pooled table (each game judged by its own season's baseline, then summed).

### Cohort overlap note

The goalie metrics qualify at different thresholds and on different shot bases and season spans, so they produce different cohort sizes: NFI-GSAx pooled has 84 goalies (five seasons), QNFS% has 146 in-file with 82 qualified (four seasons), QGx pooled has 81 (four seasons), QG%s/QG%b pooled has 145 in-file with 78 qualified (four seasons, either shot scope). NFI SV% is the one exception — it shares NFI-GSAx's cohort and floors exactly (same source files, same denominator), so its cohort size always matches NFI-GSAx's. A goalie may appear in one metric and not another because the qualifying floors differ — this is by design, not a bug. A goalie with a thin net-front shot diet but high total volume may qualify for QGx and NFI-GSAx but not for QNFS%; an infrequent appearance-maker may qualify for none. Where the same goalie appears in multiple metrics, the metrics are commensurable *for that goalie* — bearing in mind the net-front (QNFS%, NFI-GSAx) vs all-shot (QGx, QG%s/QG%b) scope difference and the internal-rate (QNFS%, NFI-GSAx) vs MoneyPuck (QGx, QG%s/QG%b) expectation model.

### Confidence interval note

QNFS%, QGx, and QG%s/QG%b are all proportions; Wilson 95% CIs are the correct tool and are reported for all three. NFI-GSAx per-60 is a rate count over a fixed exposure; per the May 20, 2026 audit migration (see `docs/AUDIT_2026-05-20.md`), Poisson CIs are the correct tool for rate metrics — but NFI-GSAx currently reports point estimates only, so no CI is emitted today. This follows the same Wilson-vs-Poisson rule the player-level framework uses.

### Locked spot-check values

Pooled values for eight reference starters, as of the methodology version stamp at the top of this document. All eight qualify in all three cohorts. QNFS% and QGx% are percentages; NFI-GSAx is GSAx per 60 ES minutes. Values shift as games are added.

| Goalie (team) | QNFS% | QGx% | NFI-GSAx /60 |
|---|---|---|---|
| Connor Hellebuyck (WPG) | 63.37 | 63.37 | 0.162 |
| Igor Shesterkin (NYR) | 62.22 | 60.89 | 0.226 |
| Ilya Sorokin (NYI) | 59.74 | 59.74 | 0.268 |
| Andrei Vasilevskiy (TBL) | 53.88 | 54.74 | 0.185 |
| Juuse Saros (NSH) | 55.10 | 53.69 | 0.061 |
| Sergei Bobrovsky (FLA) | 53.59 | 52.61 | 0.070 |
| Jacob Markström (CGY) | 49.75 | 51.02 | 0.008 |
| Adin Hill (VGK) | 52.63 | 50.38 | −0.081 |

The cohort-overlap NaN convention (show the row with NaN where a goalie qualifies for only some metrics) applies to goalies outside this eight — e.g. one clearing QGx's and NFI-GSAx's floors but under QNFS%'s 25-GP-in-a-season gate. If your pipeline output for the same data snapshot differs from these values by more than rounding (±0.01 for the percentages, ±0.005 for per-60), something is wrong with your reproduction.

Pooled QG%s / QG%b values for six reference starters, both shot scopes (values are percentages):

| Goalie | QG%s (5v5) | QG%b (5v5) | QG%s (all-sit) | QG%b (all-sit) |
|---|---|---|---|---|
| Connor Hellebuyck | 65.02 | 67.90 | 60.91 | 66.26 |
| Jeremy Swayman | 64.17 | 66.31 | 60.32 | 62.43 |
| Ilya Sorokin | 62.01 | 62.88 | 57.14 | 61.04 |
| Logan Thompson | 58.89 | 61.11 | 60.22 | 63.54 |
| Igor Shesterkin | 58.11 | 60.81 | 58.04 | 60.71 |
| John Gibson | 55.06 | 58.43 | 47.28 | 51.09 |

Season baselines underlying these values (volume-weighted tier save%, `qg_tier_baselines_by_season{suffix}.csv`):

| Season | Starter (5v5) | Backup (5v5) | Starter (all-sit) | Backup (all-sit) |
|---|---|---|---|---|
| 2022-23 | 91.85% | 91.10% | 90.93% | 90.26% |
| 2023-24 | 91.74% | 91.39% | 90.85% | 90.31% |
| 2024-25 | 91.39% | 90.84% | 90.44% | 89.77% |
| 2025-26 | 90.89% | 90.24% | 89.89% | 89.42% |

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

### July 8–9, 2026 follow-up audit — on-ice shift/event boundary

Triggered by cross-checking on-ice goals-for/against and PDO against external references (Natural Stat Trick's full-team 5v5 tables and Hockey-Reference's per-player 5v5 career tables), a boundary bug was found in how shift intervals were matched to shot/goal events for on-ice attribution.

- **The rule.** On-ice attribution for an event at absolute time `t` must use a half-open interval `start < t <= end` (exclusive start, inclusive end). The prior code used `start <= t < end`, which **under-counted** a scorer's own shift when it ended in the same second as their goal. A naive first fix to a fully-inclusive `start <= t <= end` was then found to **double-count** at every line change (both the outgoing player, whose shift ends at `t`, and the incoming player, whose shift starts at `t`, matched the same event). The asymmetric `(start, end]` rule is the correct one.
- **Verification.** The fix was validated by the mechanical invariant that a 5v5 goal must have exactly five on-ice skaters per side (the symmetric variant produced up to ten), and by reconciling to external truth: Crosby's career 5v5 GF/GA/PDO matched Hockey-Reference exactly after the fix, and a 15-player San Jose roster's GF/GA ratios vs Natural Stat Trick moved from 1.66 / 1.89 (pre-fix) to 0.98 / 1.00 (post-fix). The small residual per-player gaps trace entirely to one game (`2025021308`) that has no shift-chart data on the NHL's own API — unrecoverable, not a pipeline error.
- **De-duplication.** `shift_data.csv` interleaves real shift rows (`type_code=517`) with zero-length goal-event marker rows (`type_code=505`) that can match the same instant; every on-ice player list is now de-duplicated so a player who has both a real shift and a marker at the same time is counted once. This was proven not to be the primary driver of the external gap (filtering the markers out gave results identical to the boundary fix alone) but is a genuine correctness measure.
- **Scope.** The fix applies to shot/goal on-ice attribution only. Faceoff-based zone classification (which shift a faceoff belongs to, used for zone-start / OZ-DZ-NZ features) uses a separately-correct convention (`start <= t_face` with `end > t_face`) and was deliberately left unchanged. All on-ice-attribution producers read by the dashboard were re-run and their outputs regenerated: `player_counts_by_state_zone_per_season.csv` and its playoff companion, `player_pdo_5v5_per_season.csv` / `_allsit_`, the `fully_adjusted` NFI/RelNFI family, `per_player_game.csv`, and the decision-tree stage outputs. Methodology — what NFI measures, the zone definitions, the thresholds — is unchanged; only on-ice event attribution became more accurate.

---

## Pipeline reproducibility

The full pipeline can be reproduced from the NHL API given the scripts in `NFI/scripts/`, `Zones/scripts/`, and `NFI/Geometry_post/NF_PY/`. Execution order, schema dependencies, and column-naming conventions are documented in `PIPELINE.md`. Spot-check values in this document should be reproduced by anyone re-running the pipeline at the same data snapshot; deviation beyond rounding indicates a reproduction error rather than a methodology disagreement.

---

*This document reflects the methodology as of the May 2026 audit, the May 3 zone-adjustment factor swap, the June 7, 2026 single-game share-of-attack reporting convention (TZI2 itself, documented the same month, is exploratory tooling outside the public framework), the June 12, 2026 Goalie Metrics addition (NFI-GSAx, QNFS%, GQG, with Vollman Quality Starts disambiguation), and the July 8–9, 2026 on-ice shift/event boundary fix (asymmetric `(start, end]` attribution, with all on-ice outputs regenerated). Future methodology changes will increment the version stamp at the top of this document and update the locked spot-check values accordingly.*
