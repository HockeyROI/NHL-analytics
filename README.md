# HockeyROI

NHL analytics centered on a redefinition of high-danger scoring chances.

## What this is

Conventional NHL analytics define "high-danger" scoring chances using a fixed geometric zone — typically the home-plate-shaped trapezoid in front of the net. HockeyROI started from a simple observation: shots taken from in close but to the *sides* of the net don't behave like dangerous chances. They convert at unremarkable rates. The conventional zone definition lumps them in with shots from the doorstep and high slot, where conversion is genuinely elite, and the result is a high-danger metric that's noisier than it should be.

This project narrows the high-danger zone to where the geometry actually matters — the immediate net-front and the high slot — and applies that narrower definition consistently across player evaluation, team evaluation, and goalie evaluation. The framework is called **NFI (Net-Front Impact)**.

## The frameworks

**NFI — Net-Front Impact (flagship).** Redefines the high-danger zone as the union of two areas: the immediate net-front (CNFI) where rebounds and deflections live, and the high slot (MNFI). Combined as TNFI = CNFI ∪ MNFI. NFI rates are computed at player, team, and goalie level using Fenwick-based shot counts (Corsi corrupts spatial metrics — see `docs/METHODOLOGY.md` for the blocked-shot coordinate finding that drives this choice). Zone-adjusted using the Tulsky linear correction with the published 3.5pp factor.

**TZI — Transitional Zone Impact.** Measures how much of a player's on-ice time is spent in the offensive zone after a faceoff, computed separately from three different starting positions: DZI (Defensive Zone Impact), NZI (Neutral Zone Impact), OZI (Offensive Zone Impact). Zone-time-share construction, not shot-differential. Reported as raw 0–10 scores within position group, plus OZI's working linemate-adjusted variant (OZI_L) and a separate Rel-NZI on-off computation. Linemate adjustments for NZI / DZI / TNZI were attempted but removed in May 2026 — see `Zones/_orphaned_broken_L_2026_05/` for the audit trail. Descriptive rather than predictive: characterizes deployment-conditioned territorial impact without making correlation-to-winning claims.

**Goalie analyses.** NFI's zone definition applied to GSAx (Goals Saved Above Expected). Goalies are evaluated by their save performance specifically within the high-danger zones the NFI framework identifies, rather than across all shots equally. Five goalies analyzed across published reports — Kuemper, Forsberg, Blackwood, Wedgewood, Dostal; more in development. See `Goalies/README.md` for the cited-values note on factor changes.

**Referee analyses.** Short-form analyses of referee penalty-call tendencies relative to team and league averages. Published as X threads and articles rather than long-form posts. Find them at [@HockeyROI](https://x.com/HockeyROI).

## What's in the repo

```
NFI/         Net-Front Impact framework — flagship
Zones/       Transitional Zone Impact framework (DZI, NZI, OZI)
Goalies/     Goalie analyses applying NFI zones to GSAx
Referees/    Referee tendency analyses
Streamlit/   Public app at hockeyroi.streamlit.app
```

For methodology details, see `docs/METHODOLOGY.md`.
For pipeline execution order, see `PIPELINE.md`.
For published work timeline, see `WORK_LOG.md`.

## Tools

**hockeyroi.streamlit.app** — public NFI rankings for players, teams, and goalies, with TZI player evaluation in development.

## Author

Ash Garg
Substack: [hockeyROI.substack.com](https://hockeyROI.substack.com)
X: [@HockeyROI](https://x.com/HockeyROI)
GitHub: [HockeyROI/NHL-analytics](https://github.com/HockeyROI/NHL-analytics)

---

*Active development. Methodology audit completed May 2026; data and framework reflect post-audit state.*
