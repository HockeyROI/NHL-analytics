# Quality Games

Quality Games are per-game performance flags that complement the season-aggregate NFI metrics: where NFI% measures aggregate net-front dominance, Quality Games measure consistency — how often a player outchances opponents on a per-game basis.

See `../docs/METHODOLOGY.md` (NFI-QG section) for the full methodology, locked spot-check values, and the tie-handling rationale; this README orients you to the folder.

## What's in this folder

```
scripts/   Two-step pipeline (run in order)
Data/      MoneyPuck shot files used for the xG basis (Data/Money_puck/)
output/    Per-game, per-season, and per-team derived files
```

## Scripts (run in order)

1. `scripts/01_build_per_player_game.py` — builds per-(player, season, game, team) records at 5v5 ES regulation: on-ice TOI, Fenwick attempts for/against, MoneyPuck xGoal for/against, and net-front (CNFI ∪ MNFI) attempt counts. NFI zones come from HR's canonical `shots_tagged.csv` `zone` column, with a `classify_zone()` fallback for the small remainder of rows that don't join cleanly to MoneyPuck. Writes `output/per_player_game.csv`.
2. `scripts/02_quality_game_aggregation.py` — computes position-median thresholds and aggregates per-game flags into season- and team-level Quality Game rates. Writes `output/position_medians.csv`, `output/per_player_season.csv`, `output/per_player_season_team.csv`, and `output/per_team_season.csv`.

## The two metrics

Quality Games are computed on two parallel bases:

- **NFI-QG** — uses HockeyROI's internal net-front (CNFI ∪ MNFI) Fenwick share per game.
- **xG-QG** — uses MoneyPuck xGoal-weighted xG% per game.

A game is a Quality Game if the player's per-game share clears the empirical position-median (forwards and defense computed separately, four seasons pooled).

## Qualifying floor

A player-game enters Quality Games only if, at 5v5 ES regulation:

- on-ice TOI ≥ 480 seconds (8 minutes), **and**
- total on-ice Fenwick attempts (for + against) ≥ 5

Below either threshold the game is excluded as low-signal.

## NFI-QG tie handling (half-credit)

NFI per-game ratios are discrete and frequently land exactly on the position median (≈14% of qualifying games). NFI-QG handles ties symmetrically:

```
is_NFI_QG = 1.0  if NFI_pct_game >  position median
          = 0.5  if NFI_pct_game == position median   (half-credit tie)
          = 0.0  if NFI_pct_game <  position median
```

xG-QG keeps a `>=` (greater-or-equal) rule — its continuous distribution makes exact ties negligible. See `../docs/METHODOLOGY.md` for why the two metrics use different rules.

## Streamlit

Quality Games are surfaced on the **Players** and **Teams** tabs at hockeyroi.streamlit.app.

---

*Methodology decisions and locked spot-check values live in `../docs/METHODOLOGY.md`; this README orients you to the folder contents.*
