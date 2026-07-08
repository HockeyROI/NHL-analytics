# EDGE

NHL EDGE player-tracking data, scraped and displayed as a **separate, source-labeled
lens** alongside the existing TZI/NFI/xG metric system. Nothing in this folder feeds
into, adjusts, or recomputes any canonical metric (NZI/DZI/OZI, NFI, RelNFI, xG-QG,
etc.) — it is descriptive only, and it is never blended into TZI.

## Scrape source

`edge.nhl.com` was retired in the "EDGE 2.0" redesign; it now 301-redirects to
`www.nhl.com/nhl-edge/skaters/{slug}`, a React app whose data comes from an
undocumented endpoint on NHL's own `api-web.nhle.com` host (**not** Sportradar,
despite that being the original assumption — no Sportradar reference exists anywhere
in the EDGE app's JS bundle):

```
GET https://api-web.nhle.com/v1/edge/skater-detail/{playerId}/{season}/{gameTypeId}
```

`gameTypeId`: `2` = regular season, `3` = playoffs. Discovered via live network
inspection (2026-07-07). One endpoint call returns everything scraped here:
`zoneTimeDetails` (OZ/NZ/DZ time %, all-situations + an even-strength variant for OZ
only), `skatingSpeed.speedMax`, `skatingSpeed.burstsOver20` (20+ mph bursts — a single
count, no speed bands available for skating bursts, only for shot speed which isn't
scraped here), and `totalDistanceSkated`.

`edge/scripts/pull_edge_stats.py` pulls all (player_id, season) pairs from
`NFI/output/player_counts_by_state_zone_per_season.csv` (2021-22 → 2025-26, matching
the rest of the pipeline's season coverage) for both game types. Raw JSON responses
are cached under `edge/raw/` (resumable). Output:

- `edge/output/edge_skater_stats.csv` — 4,720 rows, regular season
- `edge/output/edge_skater_stats_playoffs.csv` — 1,679 rows, playoffs

Schema: `player_id, player_name, season, game_type, position, team, games_played,
oz_time_pct(+_percentile/_league_avg), oz_time_pct_ev(+_percentile/_league_avg),
nz_time_pct(+_percentile/_league_avg), dz_time_pct(+_percentile/_league_avg),
top_skating_speed_mph(+_percentile/_league_avg),
speed_bursts_over_20mph(+_percentile/_league_avg),
distance_skated_miles(+_percentile/_league_avg)`.

## Data-quality filter — not applied, and why

The original plan was to exclude games with zero/anomalously-low tracked distance
(camera dropouts, partial-game tracking failures) before rolling up to a season
total. **This isn't possible from outside NHL's system**: the EDGE API only exposes
pre-aggregated season totals plus a single "best game" highlight per stat — there is
no per-game log to inspect or filter. Season totals are taken exactly as NHL computed
them; if a tracking failure is baked into a player's season number, it's invisible
and unfixable from this side. No proxy or workaround was substituted — this is a
known, accepted limitation, not an oversight.

## Basis-mismatch caveat vs TZI (NZI/DZI/OZI)

**EDGE and TZI measure genuinely different things and must never be read as the same
metric:**

| | EDGE (`oz/nz/dz_time_pct`) | TZI (`NZI`/`DZI`/`OZI`) |
|---|---|---|
| Tracked by | player position (chip/camera tracking) | puck position (event/PBP-derived) |
| Scope | all-situations (or EV variant, OZ only) | strict 5v5 |
| Trigger | continuous TOI, no shift-start gating | faceoff-started shifts only |
| Source | NHL EDGE tracking system | this repo's own PBP/shift pipeline |

The Streamlit **EDGE tab** carries a mandatory on-screen banner stating this
explicitly, and its charts/table are physically separate from the Player List tab
where NZI/DZI/OZI live.

The Player List's **D/N/O Start%** columns (Gate 2 of this build) are a different,
compatible addition — that data **is** from this repo's own PBP pipeline
(`Zones/output/zone_time_raw.csv`, faceoff-started 5v5 shifts, same basis as
NZI/DZI/OZI), which is why it's displayed *alongside* TZI in the same table rather
than kept separate like EDGE. It is presentation only — it does not alter or feed
into the NZI/DZI/OZI values themselves. Note it's pooled-only (all regular seasons
combined); no per-season or playoff breakdown exists for it yet.

## PDO

`PDO = (5v5 on-ice SH% + 5v5 on-ice SV%) × 100`, SOG-based (not Fenwick/Corsi) — this
repo's **own** shot-events computation, not EDGE and not a third-party stat.

Built by `NFI/scripts/build_pdo_sog.py`: a fresh on-ice attribution pass that reuses
the exact validated shift-join / 5v5-state-derivation logic from
`NFI/scripts/03_onice_attribution_pillars.py` (copied rather than imported, since
that script executes top-to-bottom and would rewrite canonical NFI outputs as a side
effect if imported directly). The only new counter it computes is on-ice SOG-for/SOG
against per (player_id, season) at strict 5v5 — on-ice goals-for/against and 5v5 TOI
are reused as-is from the existing `player_counts_by_state_zone_per_season.csv`
(`onice_for_gl`/`onice_ag_gl`/`toi_min`), since those were already correct and didn't
need recomputing.

Floor: ≥200 min 5v5 TOI. Output: `NFI/output/player_pdo_5v5_per_season.csv` — a new
file; no canonical NFI output was modified to build it. Sanity-checked: distribution
centers on **100.16 mean / 100.10 median** (n=3,569 player-seasons), matching
conventional PDO behavior.

Displayed on the Player List tab as a **raw column inside the existing "xG" metric
family** (pooled via ratio-of-sums on raw SOG/goals counts, same convention as the
app's other pooled rate columns — not by averaging per-season PDO values). Regular
season only — no playoff PDO is computed. It is explicitly **not** a ranking column
(no color/rank styling), and has no relative (Rel) or Quality-Games (QG) variant —
raw luck-context only, paired with `xGF/60`/`xGA/60` as a "chance quality vs
conversion luck" pairing, visualized in the "PDO vs xG Differential" chart.
