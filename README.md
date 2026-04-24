# HockeyROI — NHL Analytics

Open-source research repository behind **HockeyROI**, a public project built to produce NHL-team-level analytics work from outside the industry. The central output is a new on-ice impact framework called **NFI (Net Fenwick Impact)** with full zone and quality-of-competition adjustments, a goalie evaluation layer that validates against MoneyPuck, a referee-effects study, and a team-construction model.

Author: **Ash Garg** — 20 years in real estate, NYU master's, building HockeyROI as the portfolio of work underlying a move into NHL analytics.

- Substack: **hockeyROI.substack.com**
- X / Twitter: **@HockeyROI**
- GitHub: **github.com/HockeyROI/nhl-analytics**

---

## Why this repo exists

Corsi and xG are the public standards for on-ice impact. Both have known gaps:

- **Corsi** counts every unblocked-or-blocked attempt equally and ignores where the shot came from.
- **xG** models shot quality but inherits whatever shot-quantity bias is baked into its training sample, and aggregates away zone and deployment context.

The **NFI** framework replaces Corsi's "attempts" numerator with **Fenwick** (unblocked attempts) and then applies a set of zone, competition, and linemate adjustments that have been missing from public models. Across 126 team-seasons it predicts goal share better than any public alternative tested.

---

## Headline findings

Predictive R² against team 5v5 goal share, 126 team-seasons, 2019-20 → 2024-25 (minus 2020-21 shortened year; playoffs held out):

| Metric | R² |
|---|---|
| **NFI%_ZA** (zone-adjusted Net Fenwick Impact) | **0.583** |
| xG% (public) | 0.538 |
| HD Fenwick% | 0.482 |
| Corsi% | 0.397 |

- **Three-pillar team-construction model** (netfront F + defensive F + two-way D): **R² = 0.655**
- **TNZI** (Team Neutral-Zone Index) correlation with current-season goal share: **r = 0.648**
- **Goalie NFI-GSAx** validation against MoneyPuck GSAx: **r = 0.858**
- **Fenwick zone-quality factor: 11.91 percentage points** between offensive- and defensive-zone shots, versus the traditional ~3.5pp assumption used in most public models — **public models are understating zone impact by ~71%**. Confirmed mechanically from 921,141 shot events, not fit to outcomes.

The zone factor finding is the load-bearing one: once zone is weighted correctly, a Fenwick-based model beats an xG-based model at predicting goals.

---

## The metric family

### Player on-ice metrics
| Metric | What it measures |
|---|---|
| **NFI%** | Net Fenwick Impact — on-ice Fenwick-for share, Fenwick-weighted |
| **NFI%_ZA** | Zone-adjusted NFI% — removes OZ/DZ deployment bias |
| **NFI%_3A** | Three-adjustment NFI% — zone + QoC + QoL |
| **RelNFI%** | Player NFI% vs team NFI% without them on the ice |
| **RelNFI_F%** | Offensive half of RelNFI% (Fenwick-for side only) |
| **RelNFI_A%** | Defensive half of RelNFI% (Fenwick-against side only) |
| **NFI%_3A_MOM** | Momentum form of NFI%_3A — rolling recent-window version |

Splitting Rel into F and A is novel; it exposes forwards whose "good Corsi" is driven entirely by offensive teammates (high RelNFI_F%, flat RelNFI_A%) versus true two-way drivers.

### Quality-of-context
| Metric | What it measures |
|---|---|
| **NFQOC** | Net Fenwick Quality of Competition — opponents' NFI, overlap-weighted |
| **NFQOL** | Net Fenwick Quality of Linemates — teammates' NFI, overlap-weighted |
| **ZQoC** | Zone Quality of Competition — zone-deployment pressure from opponents |
| **ZQoL** | Zone Quality of Linemates — zone-deployment help from teammates |

### Team and zone
| Metric | What it measures |
|---|---|
| **TNZI** | Team Neutral-Zone Index — neutral-zone possession/exit efficiency |
| **TNZI_C** | TNZI — controlled variant |
| **TNZI_L** | TNZI — loose (dump-based) variant |
| **TNZI_CL** | Combined controlled + loose |
| **DOZI** | Defensive-Zone-Origin Impact — counterpart to offensive zone-entry efficiency |

---

## Repo layout

```
NHL analysis/
├── NFI/                      Net Fenwick Impact framework
│   ├── scripts/              Build pipeline 00_* → 41_* plus stage/decision-tree scripts
│   ├── output/               Per-pillar CSVs, horse-race comparisons, publication tables
│   └── files nf/             Source reference files for the framework
│
├── Zones/                    Zone-time, PQR, ROC/ROL, TNZI/DOZI
│   ├── scripts/              pull_all_games, compute_zone_and_overlap, compute_pqr_roc_rol,
│   │                         compute_iozc_iozl_dozi, benchmark_tnzi_vs_others, update_daily
│   ├── raw/                  NHL API JSONs — shift charts + play-by-play (gitignored, ~1 GB)
│   ├── output/               PQR/OER/DER tables, zone-time diagnostics
│   └── zone_variations/      Per-team zone-factor sensitivity sweeps
│
├── Goalies/                  Goalie evaluation (NFI-GSAx layer)
│   ├── PY/                   Core goalie analysis + league benchmarks
│   ├── Money Puck/           MoneyPuck ingest + head-to-head validation scripts
│   ├── Benchmarks Goalies/   MoneyPuck cached data + league baselines
│   ├── Goalies_Height/       Height-vs-performance study
│   └── Dostal, Kuemper, Forsberg, Avs Goalie, Money Puck ...  per-goalie case studies
│
├── Referees/                 Referee-effect analysis
│   ├── scripts/              pull_all_teams_penalties, generate_ref_analysis, notebook
│   ├── Ref Cache/            Cached ref-game JSONs
│   ├── Ref Images/, output/  Charts and per-team penalty tables
│
├── Data/                     Shared inputs
│   ├── nhl_shot_events.csv   921,141 shot events across 6 seasons (133 MB)
│   ├── rebound_sequences.csv Rebound-chain reconstruction
│   ├── game_ids.csv          Master game-id index
│   ├── goalie_rankings_*.csv Published goalie rankings
│   └── qoc_qol/              Cached QoC/QoL pairwise-overlap data
│
├── Streamlit/                Public-facing app
│   ├── app_final.py          Current deployed app
│   └── requirements.txt
│
├── Brand/                    Logos and cover art for Substack posts
├── 2026 posts/               Drafts and supporting data for published + upcoming posts
│                             (Playoff_preview, Mcdavid_post, Oilers_vs_Avs, Fred_Mangi,
│                              April 2026 X replies, Future Posts)
└── .github/                  CI — daily updater workflow
```

### Key scripts

**NFI pipeline** (`NFI/scripts/`, run in numeric order):
- `00_*` — player position lookup, unknown-player check, position enrichment, 2025-26 standings
- `01_heatmap_inflection.py` — zone inflection-point heatmap and save%-by-zone
- `02_zones_and_rebound_confirm.py` — zone-factor derivation
- `03_onice_attribution_pillars.py` — splits on-ice impact into seven pillars (netfront F, defensive F, offensive D, defensive D, goalie FNFI/MNFI/CNFI)
- `04_corsi_nfi_variants.py` — NFI%, NFI%_ZA, NFI%_3A construction
- `05_horse_race.py` — univariate and multivariate R² comparisons (NFI vs xG vs Corsi vs HD Fenwick)
- `06_qoc_qot.py` — NFQOC / NFQOL
- `09_zone_conversion.py` — zone-factor empirical test
- `13_fnfi_downstream.py` — goalie-facing downstream effects
- `18_team_construction_model.py` / `19_team_archetypes_quartile.py` — 3-pillar team model, archetypes
- `28_publication_outputs.py`, `30_publication_charts.py`, `33_age_filter_publication.py` — publication artifacts
- `41_zone_factor_investigation.py` — the 11.91pp vs 3.5pp mechanical confirmation
- `rename_and_momentum.py` — NFI%_3A_MOM rolling-window form

**Zones pipeline** (`Zones/scripts/`):
- `pull_all_games.py` — resumable fetch of shift charts + play-by-play for every game
- `compute_zone_and_overlap.py` — single-pass zone-time + pairwise overlap
- `compute_pqr_roc_rol.py` — PQR normalisation, ROC/ROL, final CSVs
- `compute_iozc_iozl_dozi.py` — zone-context QoC/QoL
- `benchmark_tnzi_vs_others.py` — TNZI validation vs alternative neutral-zone metrics
- `update_daily.py` — incremental refresh triggered by GitHub Actions

**Goalies** (`Goalies/`):
- `PY/goalie_analysis_2.py`, `goalie_analysis_compare.py` — spatial save% and GSAx
- `PY/league_benchmarks_goalies.py` — league baseline by shot zone
- `Money Puck/goalie_rankings.py`, `moneypuck_goalies.py` — MoneyPuck ingest + r=0.858 validation

**Referees** (`Referees/scripts/`):
- `pull_all_teams_penalties.py` — 3-season penalty pull across all teams
- `generate_ref_analysis.py` — ref-level differential analysis
- `Ref_Analysis.ipynb` — exploratory notebook

---

## Data

All data is pulled from the **public NHL API** (`api-web.nhle.com` shift charts, play-by-play, schedule, standings). No private or paid feeds.

- **921,141 shot events** across **6 seasons** (2019-20 → 2024-25, regular + playoffs held out for model validation; 2025-26 live)
- Raw shift charts and play-by-play JSONs cached under `Zones/raw/` (gitignored, ~1 GB — regenerated by `pull_all_games.py`)
- MoneyPuck goalie data cached under `Goalies/Benchmarks Goalies/` for validation only

---

## Reproducing the work

### Requirements
- Python 3.12+
- `pip install -r Streamlit/requirements.txt` for the app
- Core analysis scripts rely on `pandas`, `numpy`, `scipy`, `requests`, `statsmodels`, `matplotlib`. Install as needed.

### Pulling data from the NHL API
```bash
# 1. Shift charts + play-by-play across seasons (resumable, writes to Zones/raw/)
python Zones/scripts/pull_all_games.py

# 2. Zone-time + pairwise overlap
python Zones/scripts/compute_zone_and_overlap.py

# 3. PQR / ROC / ROL
python Zones/scripts/compute_pqr_roc_rol.py

# 4. TNZI benchmarking
python Zones/scripts/benchmark_tnzi_vs_others.py
```

### Building the NFI framework
```bash
# Run the numbered NFI pipeline in order — each step writes CSVs to NFI/output/
for f in NFI/scripts/0*_*.py NFI/scripts/1*_*.py NFI/scripts/2*_*.py NFI/scripts/3*_*.py NFI/scripts/4*_*.py; do
    python "$f"
done
```

### Running the Streamlit app locally
```bash
cd Streamlit
pip install -r requirements.txt
streamlit run app_final.py
```

### Daily updater
`.github/workflows/update.yml` pulls newly completed games, regenerates CSVs, and commits updated outputs. Runs at 12:00 and 13:00 UTC.

---

## Status

This is an in-progress research portfolio. Results here are reproducible from the scripts and the public NHL API. Weekly write-ups land on the Substack; shorter threads on X.

Questions, corrections, and collaboration invitations welcome.
