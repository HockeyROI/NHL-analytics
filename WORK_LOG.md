# WORK_LOG.md

A running log of published HockeyROI posts. New entries added on publish.

## How to use this file

- **On publish:** add one line — date, link, one rough sentence on what the post is about.
- **Quarterly:** ask a Claude chat to "read WORK_LOG.md and update README.md based on the strongest recent items." That's the whole maintenance ritual.

Format: `YYYY-MM-DD — [link] — one-sentence summary.`

---

## 2026

- 2026-04-10 — [link](https://hockeyroi.substack.com/p/darcy-kuempers-achilles-heel-what) — Goalie analysis: how to beat Kuemper, by shot type and zone.
- 2026-04-12 — [link](https://hockeyroi.substack.com/p/i-told-you-so-but-not-for-the-reason) — Goalie analysis: how to beat Forsberg.
- 2026-04-14 — [link](https://hockeyroi.substack.com/p/the-oilers-didnt-listenagain) — Goalie analysis covering both Blackwood and Wedgewood; how to beat them.
- 2026-04-17 — [link](https://hockeyroi.substack.com/p/the-geometry-of-winning-what-921000) — First NFI post; introduction to the net-front impact framework.
- 2026-04-24 — [link](https://hockeyroi.substack.com/p/how-to-beat-lukas-dostal-a-shot-by) — Goalie analysis: shot-by-shot breakdown of Dostal.
- 2026-04-28 — [link](https://hockeyroi.substack.com/p/whats-hiding-in-the-hd-shot-map) — Second NFI post; more detailed treatment of the HD zone redefinition.
- 2026-05-03 — [link](https://hockeyroi.substack.com/p/what-net-front-impact-told-us-about) — NFI applied at team level; pre-playoff team rankings covering Round 1 and 2.
- 2026-05-08 — [link](https://hockeyroi.substack.com/p/oilers-true-goalie-problem) — Oilers' goaltending struggles framed as a team-defense problem, not just a goalie problem.
- 2026-05-11 — [link](https://hockeyroi.substack.com/p/a-deep-dive-into-the-edm-goalie-options) — Realistic goalie options for the Oilers.
- 2026-05-14 — [link](https://hockeyroi.substack.com/p/why-raw-stats-lie-the-case-for-relative) — Methodology: why RelNFI/RelCorsi beat raw stats for player evaluation.

- 2026-05-17 — [link](https://hockeyroi.substack.com/p/forget-mcmann-and-jenner-these-are) — RelNFI deep-dive identifying undervalued contributors hidden by rel production stats.

- 2026-05-20 — [link](https://hockeyroi.substack.com/p/the-cap-is-going-up-the-ufa-class) — UFA-class analysis framed against rising salary cap; talent and value framing for the 2026 free agent market.

- 2026-05-20 (in-place correction) — [link](https://hockeyroi.substack.com/p/oilers-true-goalie-problem) — Updated the 2026-05-08 Oilers post in-place with corrected NFI rankings (#8/#8/#11/#22) after the May 20 audit found a state==ES filter bug and a missing game_type filter that had previously shown EDM as near dead-last in suppression. Original framing held; specific numbers corrected.

- 2026-05-20 — Internal: May 20, 2026 audit and bug-fix session — Found and fixed three pipeline bugs (state==ES conflation of 5v5/4v4/3v3, missing game_type filter at multiple consumer scripts, shift-data over-filter affecting goalie pillars and team counters). Migrated per-60 rate CIs from Wilson to Poisson; added RelNFI 95% CIs. Five commits pushed (909a4de, 6d35519, 758f632, 5b353cb, 95e71e8). Verified EDM and DAL ranks across 4 audited seasons unchanged from pre-audit. Audit folder and trial artefacts gitignored per May 4 cleanup doctrine.

- 2026-05-23 — [link](https://hockeyroi.substack.com/p/what-can-edm-learn-from-dal-and-fla) — Lessons for Edmonton from Dallas and Florida's playoff approaches.

- 2026-05-26 — [link](https://hockeyroi.substack.com/p/the-goalie-market-is-priced-incorrectly) — First goalie market mispricing post; framework-led case for systematic over/under-valuation.

- 2026-05-28 — [link](https://hockeyroi.substack.com/p/the-goalie-market-is-priced-incorrectly-cb7) — Follow-up to the May 26 goalie market post; revisits the argument with additional context.

- 2026-05-30 — [link](https://hockeyroi.substack.com/p/the-goalie-market-is-priced-wrong) — Third goalie market post in the series; further iteration on the mispricing thesis.

- 2026-06-02 — [link](https://hockeyroi.substack.com/p/new-metric-transitional-zone-impact) — Public introduction of the Transitional Zone Impact (TZI) framework with the three peer metrics (NZI/DZI/OZI).

- 2026-06-04 — [link](https://hockeyroi.substack.com/p/what-tzi-data-says-about-five-teams) — Five-team analysis using TZI; first applied use of the framework after public introduction.

- 2026-06-10 — [link](https://hockeyroi.substack.com/p/what-our-zone-impact-found-about) — Zone Impact findings across the league; broader framework deployment.

- 2026-06-15 — [link](https://hockeyroi.substack.com/p/why-game-to-game-consistency-matters) — Case for Quality Games as a player-evaluation metric; argues consistency is undervalued vs season-aggregate dominance.
