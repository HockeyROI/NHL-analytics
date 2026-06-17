# Goalies

Goalie analyses applying the NFI framework to GSAx (Goals Saved Above Expected). Goalies are evaluated by their save performance specifically within the high-danger zones the NFI framework identifies, rather than across all shots equally.

See `../docs/METHODOLOGY.md` for the NFI methodology rationale and `../WORK_LOG.md` for the running list of published goalie posts.

## Goalie metrics

The framework defines three season-level goalie metrics. Full construction, qualifying thresholds, confidence-interval treatment, and locked spot-check values live in `../docs/METHODOLOGY.md` (Goalie Metrics section) — this is a pointer, not a duplicate.

- **NFI-GSAx** — per-(goalie, season) goals saved above expected on net-front (CNFI ∪ MNFI) shots only, against an internal per-season, per-zone league goal rate. Reported per 60 ES minutes.
- **QNFS% (Quality Net-Front Save %)** — share of a goalie's 5v5 ES regulation games in which their per-game net-front GSAx ≥ 0. A consistency measure on the highest-leverage shots.
- **QS-GSAx (Quality Start GSAx %)** — share of games in which their per-game all-shot GSAx ≥ 0, using the MoneyPuck xGoal model. The all-shot complement to QNFS%.

**Disambiguation from Vollman Quality Starts.** These are *not* Robert Vollman's Quality Starts. Vollman's QS is binary on save percentage (save% above league average). HockeyROI's quality-start metrics threshold on per-game **GSAx ≥ 0** — the goalie beat expected on a danger-/xG-weighted basis — not on save% versus league average. Do not conflate them.

Outputs:
- `../NFI/Output/goalie_nfi_gsax_by_season.csv` and `../NFI/Output/goalie_nfi_gsax_pooled_v2.csv` — NFI-GSAx (per-season and pooled).
- `../NFI/goalie_consistency/output/qnfs_2022-2026.csv` (+ `qnfs_per_season_2022-2026.csv`) — QNFS%.
- `../NFI/goalie_consistency/output/qs_gsax_2022-2026.csv` (+ `qs_gsax_per_season_2022-2026.csv`) — QS-GSAx.

These three metrics are surfaced on the Goalies tab at hockeyroi.streamlit.app.

## Note on cited values

Goalie reports published prior to May 2026 may cite NFI%_ZA values computed with the empirical 0.1071 zone-adjustment factor in effect at publication. That empirical factor was later evaluated and rejected — it added no significant predictive value over Tulsky's published 0.035 (ΔR² = +0.005, p = 0.187) — and on May 3, 2026 the framework adopted Tulsky's 3.5pp standard. The underlying methodology (NFI zone definition, raw NFI%, GSAx computation) is unchanged; only the zone-adjustment factor changed, and raw NFI% values cited in those reports remain accurate.

If you re-run the pipeline today and find NFI%_ZA values that differ from those cited in the April 2026 goalie reports, that's expected — the factor change accounts for the difference. Raw NFI% values should match.

## Published goalie analyses

Five goalies analyzed across four published reports (April 2026):

- **Darcy Kuemper** — how to beat Kuemper, by shot type and zone
- **Anton Forsberg** — how to beat Forsberg
- **Jake Allen Blackwood + Scott Wedgewood** (one combined post on the Devils' goalie tandem)
- **Lukas Dostal** — shot-by-shot breakdown

See `../WORK_LOG.md` for links and dates.

## What's in this folder

Each goalie analyzed gets a subfolder with the underlying data, working files, and any locally-saved chart sources used for the published post. Folder contents are typically:

- Raw shot data filtered to the goalie's faced shots
- NFI zone breakdowns (CNFI, MNFI, FNFI for historical reference)
- Save-percentage tables by shot type and zone
- Working notes (drafts and Word docs are gitignored — they live in OneDrive)

## Money Puck/

`Money Puck/` contains data sourced from MoneyPuck.com used for cross-validation of GSAx values. This is not a scrape pipeline — the data was pulled manually from MoneyPuck's published files for comparison purposes. See the published goalie posts for how this comparison was used.

## What's not here

- Aggregate goalie-comparison files that depended on R-squared frameworks (e.g., goalie correlation-to-winning analyses) were retired during the May 2026 audit follow-up. See `../docs/METHODOLOGY.md` "On exploratory R-squared work" for context.
- Some legacy goalie scripts (e.g., earlier versions of the spatial GSAx pipeline) were moved to `_legacy/` archives during the audit. Active goalie analysis scripts live under `../NFI/` rather than this folder: NFI-GSAx is produced by `../NFI/scripts/21_goalie_gsax_by_season.py`, and QNFS% / QS-GSAx by scripts in `../NFI/goalie_consistency/scripts/` (`compute_qnfs.py`, `compute_qs_gsax.py`).

---

*The goalie work is an applied extension of the NFI methodology. Methodology decisions live in `../docs/METHODOLOGY.md`; published findings live on Substack and are tracked in `../WORK_LOG.md`.*
