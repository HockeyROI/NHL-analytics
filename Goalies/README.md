# Goalies

Goalie analyses applying the NFI framework to GSAx (Goals Saved Above Expected). Goalies are evaluated by their save performance specifically within the high-danger zones the NFI framework identifies, rather than across all shots equally.

See `../METHODOLOGY.md` for the NFI methodology rationale and `../WORK_LOG.md` for the running list of published goalie posts.

## Note on cited values

Goalie reports published prior to May 2026 may cite NFI%_ZA values computed with the empirical 0.1071 zone-adjustment factor in effect at publication. The underlying methodology (NFI zone definition, raw NFI%, GSAx computation) is unchanged. Only the zone-adjustment factor was updated to Tulsky's published 3.5pp standard on May 3, 2026; raw NFI% values cited in those reports remain accurate.

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

- Aggregate goalie-comparison files that depended on R-squared frameworks (e.g., goalie correlation-to-winning analyses) were retired during the May 2026 audit follow-up. See `../METHODOLOGY.md` "On exploratory R-squared work" for context.
- Some legacy goalie scripts (e.g., earlier versions of the spatial GSAx pipeline) were moved to `_legacy/` archives during the audit. Active goalie analysis scripts live in `../NFI/scripts/` rather than this folder.

---

*The goalie work is an applied extension of the NFI methodology. Methodology decisions live in `../METHODOLOGY.md`; published findings live on Substack and are tracked in `../WORK_LOG.md`.*
