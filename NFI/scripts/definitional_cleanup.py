#!/usr/bin/env python3
"""Definitional cleanup — pick one ES definition, audit + migrate.

Two coexisting "ES" definitions in the codebase:
  Variant A (broad ES):    shots_tagged.csv state == 'ES'
                           includes 5v5 reg + 4v4 + 3v3 OT
  Variant F (strict 5v5):  nhl_shot_events.csv situation_code == 1551
                           AND period_type == 'REG'

Set CANONICAL_DEFINITION below ('A' or 'B') and run with --dry-run first.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
CANONICAL_DEFINITION: str = "B"   # 'A' = state=='ES' broad; 'B' = strict 5v5 REG
ROOT = Path("/Users/ashgarg/Documents/HockeyROI")

# Source files
RAW_EVENTS  = ROOT / "Data" / "nhl_shot_events.csv"
TAGGED      = ROOT / "NFI" / "output" / "shots_tagged.csv"
TEAM_LEVEL  = ROOT / "NFI" / "output" / "team_level_all_metrics.csv"
METRICS_TM  = ROOT / "NFI" / "output" / "metrics_team.csv"
COMPOSITE   = ROOT / "NFI" / "output" / "team_composite_NFI.csv"
GAME_IDS    = ROOT / "Data" / "game_ids.csv"
STANDINGS   = ROOT / "NFI" / "output" / "standings_pool5.csv"

# Migration backup directory
BACKUP_DIR = ROOT / "Output" / f"migration_backup_{datetime.now().strftime('%Y%m%d')}"

# Constants
FENWICK_TYPES = {"shot-on-goal", "missed-shot", "goal"}
NFI_ZONES = {"CNFI", "MNFI", "FNFI"}

# Spot-check teams (Phase 4)
SPOT_CHECK_TEAMS = ["COL", "OTT", "EDM", "FLA", "CHI"]

# Drift threshold to surface as a blocker
TEAM_NFI_DRIFT_BLOCKER = 0.005

# =============================================================================
# UTILITIES
# =============================================================================

def banner(msg: str, width: int = 100, char: str = "=") -> None:
    print()
    print(char * width)
    print(msg)
    print(char * width)


def info(msg: str) -> None:
    print(f"  {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


def err(msg: str) -> None:
    print(f"  ❌ {msg}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def file_summary(fp: Path) -> str:
    if not fp.exists():
        return f"{fp}  [NOT FOUND]"
    sz = fp.stat().st_size
    mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fp.stat().st_mtime))
    try:
        nrows = sum(1 for _ in fp.open()) - 1
    except Exception:
        nrows = -1
    return f"{fp.relative_to(ROOT)}  size={sz:,}  mtime={mt}  rows={nrows:,}"


# =============================================================================
# PHASE 1 — AUDIT
# =============================================================================

# Patterns identifying each variant's filter usage.
# Variant A: state == 'ES' or state.isin(['ES', ...])
PAT_A = re.compile(
    r"""state\s*==\s*['\"]ES['\"]"""
    r"""|state\s*\.\s*isin\s*\(\s*[\[\{]?\s*['\"]ES['\"]"""
    r"""|"state"\s*\]\s*==\s*['\"]ES['\"]"""
    r"""|'state'\s*\]\s*==\s*['\"]ES['\"]""",
    re.MULTILINE,
)
# Variant F components
PAT_F_SIT = re.compile(
    r"""situation_code\s*==\s*1551"""
    r"""|"situation_code"\s*\]\s*==\s*1551"""
    r"""|'situation_code'\s*\]\s*==\s*1551"""
    r"""|situation_code\s*\.\s*isin\s*\(\s*[\[\{]?\s*1551""",
    re.MULTILINE,
)
PAT_F_PER = re.compile(
    r"""period_type\s*==\s*['\"]REG['\"]"""
    r"""|"period_type"\s*\]\s*==\s*['\"]REG['\"]"""
    r"""|'period_type'\s*\]\s*==\s*['\"]REG['\"]"""
    r"""|period_type\s*\.\s*isin\s*\(\s*[\[\{]?\s*['\"]REG['\"]""",
    re.MULTILINE,
)


def gather_py_files() -> list[Path]:
    out: list[Path] = []
    search_dirs = [
        ROOT / "NFI" / "scripts",
        ROOT / "NFI" / "Geometry_post",
        ROOT / "Zones" / "scripts",
        ROOT / "Streamlit",
        ROOT / "Goalies",
        ROOT / "Referees",
        ROOT / "scripts",
    ]
    # Exclude this script itself from the audit — it contains both filter patterns
    # in regex strings, which would (correctly) classify it as MIXED. That's a false
    # positive for a meta-tool that doesn't itself participate in the pipeline.
    self_path = Path(__file__).resolve()
    for d in search_dirs:
        if not d.exists():
            continue
        for fp in d.rglob("*.py"):
            if "_legacy" in str(fp) or "__pycache__" in str(fp):
                continue
            if fp.resolve() == self_path:
                continue
            out.append(fp)
    return out


def classify_file(fp: Path) -> dict | None:
    try:
        text = fp.read_text(errors="ignore")
    except Exception as e:
        warn(f"could not read {fp.relative_to(ROOT)}: {e}")
        return None

    matches_A = list(PAT_A.finditer(text))
    matches_F_sit = list(PAT_F_SIT.finditer(text))
    matches_F_per = list(PAT_F_PER.finditer(text))

    uses_A = bool(matches_A)
    uses_F_sit = bool(matches_F_sit)
    uses_F_per = bool(matches_F_per)
    uses_F = uses_F_sit and uses_F_per

    if not (uses_A or uses_F_sit or uses_F_per):
        return None

    if uses_A and uses_F:
        cat = "MIXED (uses both A and F)"
    elif uses_A:
        cat = "A — state=='ES'"
    elif uses_F:
        cat = "F — strict 5v5 REG"
    elif uses_F_sit and not uses_F_per:
        cat = "PARTIAL_F (sit_code only, no period_type=='REG')"
    elif uses_F_per and not uses_F_sit:
        cat = "PARTIAL_F (period_type only, no sit_code==1551)"
    else:
        cat = "OTHER"

    # Sample line numbers
    def line_of(m: re.Match) -> int:
        return text[: m.start()].count("\n") + 1

    line_samples = []
    for m in matches_A[:1]:
        line_samples.append(f"L{line_of(m)} (A)")
    for m in matches_F_sit[:1]:
        line_samples.append(f"L{line_of(m)} (F.sit)")
    for m in matches_F_per[:1]:
        line_samples.append(f"L{line_of(m)} (F.per)")

    return {
        "path": fp.relative_to(ROOT),
        "category": cat,
        "uses_A": uses_A,
        "uses_F_sit": uses_F_sit,
        "uses_F_per": uses_F_per,
        "uses_F": uses_F,
        "lines": line_samples,
    }


def phase1_audit() -> list[dict]:
    banner("PHASE 1 — AUDIT (codebase scan for ES filter usage)")
    py_files = gather_py_files()
    info(f"Scanning {len(py_files)} .py files (excluding _legacy and __pycache__)")
    results: list[dict] = []
    for fp in py_files:
        rec = classify_file(fp)
        if rec:
            results.append(rec)

    # Sort by category then path
    cat_order = {
        "A — state=='ES'": 0,
        "F — strict 5v5 REG": 1,
        "MIXED (uses both A and F)": 2,
        "PARTIAL_F (sit_code only, no period_type=='REG')": 3,
        "PARTIAL_F (period_type only, no sit_code==1551)": 4,
        "OTHER": 5,
    }
    results.sort(key=lambda r: (cat_order.get(r["category"], 99), str(r["path"])))

    print()
    info(f"Total files referencing any ES-related filter: {len(results)}")
    print()
    print(f"  {'category':<55} {'count':>6}")
    print("  " + "-" * 64)
    from collections import Counter
    counts = Counter(r["category"] for r in results)
    for cat in sorted(counts.keys(), key=lambda c: cat_order.get(c, 99)):
        print(f"  {cat:<55} {counts[cat]:>6}")

    print()
    print("  Detailed per-file classification:")
    print(
        f"  {'category':<35} {'file':<60} {'lines':<25}"
    )
    print("  " + "-" * 122)
    for r in results:
        cat_short = (
            r["category"]
            .replace(" (uses both A and F)", "")
            .replace(" (sit_code only, no period_type=='REG')", " (sit only)")
            .replace(" (period_type only, no sit_code==1551)", " (per only)")
        )
        print(
            f"  {cat_short:<35} {str(r['path']):<60} {', '.join(r['lines']):<25}"
        )

    # Surface any third / partial definitions as blockers
    third_def = [
        r
        for r in results
        if r["category"].startswith("PARTIAL_F")
        or r["category"] == "MIXED (uses both A and F)"
        or r["category"] == "OTHER"
    ]
    if third_def:
        print()
        warn("Files with non-canonical or mixed definitions found — review before migration:")
        for r in third_def:
            print(f"    {str(r['path'])}  [{r['category']}]")

    return results


# =============================================================================
# PHASE 2 — IMPACT ANALYSIS
# =============================================================================


def compute_team_metrics(definition: str) -> pd.DataFrame:
    """Compute per-team CNFI+MNFI counts for 2025-26 under chosen definition.

    definition ∈ {'A', 'B'}.
    Returns DataFrame with columns: team, attack, suppress, total, nfi_pct.
    """
    if definition == "A":
        # Variant A: shots_tagged.csv state == 'ES'
        sh = pd.read_csv(
            TAGGED,
            usecols=[
                "season",
                "event_type",
                "shooting_team_abbrev",
                "home_team_abbrev",
                "away_team_abbrev",
                "state",
                "zone",
            ],
        )
        sh["season"] = sh["season"].astype(str)
        sh = sh[(sh["season"] == "20252026") & (sh["state"] == "ES")]
        sh = sh[sh["event_type"].isin(FENWICK_TYPES)]
        sh = sh[sh["zone"].isin(["CNFI", "MNFI"])]
    elif definition == "B":
        # Variant F: raw with situation_code==1551 + period_type=='REG'
        raw = pd.read_csv(
            RAW_EVENTS,
            usecols=[
                "game_id",
                "event_id",
                "season",
                "event_type",
                "shooting_team_abbrev",
                "home_team_abbrev",
                "away_team_abbrev",
                "situation_code",
                "period_type",
            ],
        )
        raw["season"] = raw["season"].astype(str)
        raw = raw[
            (raw["season"] == "20252026")
            & (raw["situation_code"] == 1551)
            & (raw["period_type"] == "REG")
            & (raw["event_type"].isin(FENWICK_TYPES))
        ]
        # Merge with tagged for zone (event_id is the canonical key)
        zones = pd.read_csv(TAGGED, usecols=["game_id", "event_id", "zone"])
        sh = raw.merge(zones, on=["game_id", "event_id"], how="left")
        sh = sh[sh["zone"].isin(["CNFI", "MNFI"])]
    else:
        raise ValueError(f"Unknown definition: {definition}")

    teams = sorted(set(sh["home_team_abbrev"].unique()) | set(sh["away_team_abbrev"].unique()))
    rows = []
    for tm in teams:
        atk = (sh["shooting_team_abbrev"] == tm).sum()
        sup = (
            (sh["shooting_team_abbrev"] != tm)
            & ((sh["home_team_abbrev"] == tm) | (sh["away_team_abbrev"] == tm))
        ).sum()
        total = atk + sup
        rows.append(
            {
                "team": tm,
                "attack": int(atk),
                "suppress": int(sup),
                "total": int(total),
                "nfi_pct": float(atk / total) if total > 0 else float("nan"),
            }
        )
    df = pd.DataFrame(rows)
    df["rank"] = df["nfi_pct"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("rank").reset_index(drop=True)


def phase2_impact() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    banner("PHASE 2 — IMPACT ANALYSIS (32 teams under both definitions)")
    info("Computing under Variant A (state=='ES') ...")
    a = compute_team_metrics("A")
    info(f"  A: {len(a)} teams, league total atk={int(a['attack'].sum()):,} sup={int(a['suppress'].sum()):,}")
    info("Computing under Variant F (strict 5v5 REG) ...")
    f = compute_team_metrics("B")
    info(f"  F: {len(f)} teams, league total atk={int(f['attack'].sum()):,} sup={int(f['suppress'].sum()):,}")

    # League-balance sanity
    if a["attack"].sum() != a["suppress"].sum():
        err(f"Variant A: league attack ({a['attack'].sum()}) != suppress ({a['suppress'].sum()})")
    if f["attack"].sum() != f["suppress"].sum():
        err(f"Variant F: league attack ({f['attack'].sum()}) != suppress ({f['suppress'].sum()})")

    merged = a.merge(f, on="team", suffixes=("_A", "_F"))
    merged["delta_value"] = merged["nfi_pct_F"] - merged["nfi_pct_A"]
    merged["delta_rank"] = merged["rank_F"] - merged["rank_A"]
    merged = merged.sort_values("rank_A").reset_index(drop=True)

    print()
    print(
        f"  {'team':<5} {'A_atk':>6} {'A_sup':>6} {'A_pct':>8} {'A_rk':>5} | "
        f"{'F_atk':>6} {'F_sup':>6} {'F_pct':>8} {'F_rk':>5} | "
        f"{'Δ_atk':>6} {'Δ_sup':>6} {'Δ_pct':>8} {'Δ_rk':>5}"
    )
    print("  " + "-" * 118)
    for _, r in merged.iterrows():
        d_atk = int(r["attack_F"] - r["attack_A"])
        d_sup = int(r["suppress_F"] - r["suppress_A"])
        print(
            f"  {r['team']:<5} {int(r['attack_A']):>6} {int(r['suppress_A']):>6} "
            f"{r['nfi_pct_A']:>8.4f} {int(r['rank_A']):>5} | "
            f"{int(r['attack_F']):>6} {int(r['suppress_F']):>6} "
            f"{r['nfi_pct_F']:>8.4f} {int(r['rank_F']):>5} | "
            f"{d_atk:>+6} {d_sup:>+6} {r['delta_value']:>+8.4f} "
            f"{int(r['delta_rank']):>+5}"
        )

    # Summary stats
    print()
    info(f"max |Δ value|: {merged['delta_value'].abs().max():.4f}")
    info(f"mean |Δ value|: {merged['delta_value'].abs().mean():.4f}")
    info(f"teams shifting ≥1 ranks: {(merged['delta_rank'].abs() >= 1).sum()}")
    info(f"teams shifting ≥3 ranks: {(merged['delta_rank'].abs() >= 3).sum()}")
    if merged["delta_rank"].abs().max() > 0:
        worst = merged.iloc[merged["delta_rank"].abs().idxmax()]
        info(
            f"largest rank shift: {worst['team']} "
            f"({int(worst['rank_A'])} → {int(worst['rank_F'])}, Δ={int(worst['delta_rank']):+d})"
        )

    # Blocker check: any team where Δ exceeds drift threshold (article-level claim risk)
    blockers = []
    big_drift = merged[merged["delta_value"].abs() > TEAM_NFI_DRIFT_BLOCKER]
    if len(big_drift) > 0:
        warn(f"{len(big_drift)} team(s) with |ΔNFI%| > {TEAM_NFI_DRIFT_BLOCKER} — "
             "rankings/claims may shift materially:")
        for _, r in big_drift.iterrows():
            line = (
                f"    {r['team']}: A={r['nfi_pct_A']:.4f}({int(r['rank_A'])}) "
                f"F={r['nfi_pct_F']:.4f}({int(r['rank_F'])}) "
                f"Δ={r['delta_value']:+.4f} Δrank={int(r['delta_rank']):+d}"
            )
            print(line)
            blockers.append(line.strip())

    return a, f, blockers


# =============================================================================
# PHASE 3 — MIGRATION (--apply only)
# =============================================================================


def phase3_migrate(canonical: str, dry_run: bool) -> dict:
    banner(f"PHASE 3 — MIGRATION (canonical = Variant {canonical}; dry_run={dry_run})")

    if canonical == "A":
        info("Canonical is Variant A (state=='ES'). Current team_level_all_metrics already "
             "uses this definition; no regeneration required.")
        return {"applied": False, "reason": "Variant A already canonical in current files"}

    # Variant B (strict 5v5 REG) — regenerate metrics_team.csv and team_level_all_metrics.csv
    info("Loading raw events with strict 5v5 REG filter...")
    raw = pd.read_csv(
        RAW_EVENTS,
        usecols=[
            "game_id",
            "event_id",
            "season",
            "event_type",
            "shooting_team_abbrev",
            "home_team_abbrev",
            "away_team_abbrev",
            "situation_code",
            "period_type",
        ],
    )
    raw["season"] = raw["season"].astype(str)
    raw = raw[
        (raw["situation_code"] == 1551)
        & (raw["period_type"] == "REG")
    ]
    info(f"  raw post strict-filter: {len(raw):,}")

    info("Loading shots_tagged for zones + coords (HD geometry)...")
    tagged = pd.read_csv(
        TAGGED,
        usecols=[
            "game_id",
            "event_id",
            "zone",
            "x_coord_norm",
            "y_coord_norm",
            "score_bucket",
        ],
    )
    sh = raw.merge(tagged, on=["game_id", "event_id"], how="left")
    info(f"  after merge: {len(sh):,} (rows with zone null after merge: "
         f"{sh['zone'].isna().sum():,})")

    # ARI -> UTA normalization (matches 04_corsi_nfi_variants.py)
    sh["shooting_team_abbrev"] = sh["shooting_team_abbrev"].astype(str).replace({"ARI": "UTA"})
    sh["home_team_abbrev"] = sh["home_team_abbrev"].astype(str).replace({"ARI": "UTA"})
    sh["away_team_abbrev"] = sh["away_team_abbrev"].astype(str).replace({"ARI": "UTA"})

    # Build flags identical to 04_corsi_nfi_variants.py
    sh["is_fen"] = sh["event_type"].isin(FENWICK_TYPES).astype(int)
    sh["is_cor"] = 1  # Corsi = all attempts (everything in this filter is at minimum a Corsi attempt)
    # Wait: not strictly true. Corsi types = {SOG, missed, blocked, goal}. Need to filter.
    CORSI_TYPES = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}
    sh = sh[sh["event_type"].isin(CORSI_TYPES)]
    info(f"  after Corsi-type filter: {len(sh):,}")

    sh["is_cor"] = 1
    sh["is_TNFI"] = sh["zone"].isin(NFI_ZONES).astype(int)
    sh["is_CNFI"] = (sh["zone"] == "CNFI").astype(int)
    sh["is_MNFI"] = (sh["zone"] == "MNFI").astype(int)
    sh["is_FNFI"] = (sh["zone"] == "FNFI").astype(int)

    # Fenwick * zone parallels
    sh["is_TNFI_fen"] = sh["is_TNFI"] * sh["is_fen"]
    sh["is_CNFI_fen"] = sh["is_CNFI"] * sh["is_fen"]
    sh["is_MNFI_fen"] = sh["is_MNFI"] * sh["is_fen"]
    sh["is_FNFI_fen"] = sh["is_FNFI"] * sh["is_fen"]

    # HD definition: NST trapezoid (matches the standardized version in 04_corsi_nfi_variants.py)
    ax = sh["x_coord_norm"]
    ay = sh["y_coord_norm"].abs()
    sh["is_HD"] = (
        ((ax >= 69) & (ax < 85) & (ay <= 22))
        | ((ax >= 85) & (ax <= 89) & (ay <= 18))
    ).astype(int)
    sh["is_HD_fen"] = sh["is_HD"] * sh["is_fen"]

    # Score weights (matches 04 script)
    SCORE_W = {"trail2plus": 1.40, "trail1": 1.20, "tied": 1.00, "lead1": 0.85, "lead2plus": 0.70}
    sh["sw"] = sh["score_bucket"].map(SCORE_W).fillna(1.0)

    # Defending team
    sh["def_team"] = np.where(
        sh["shooting_team_abbrev"] == sh["home_team_abbrev"],
        sh["away_team_abbrev"],
        sh["home_team_abbrev"],
    )

    # Per-(season, team) FOR/AGAINST aggregates
    info("Building per-team aggregates...")

    # FOR side
    cols_for = {
        "CF": ("shooting_team_abbrev", np.ones(len(sh))),
        "CF_adj": ("shooting_team_abbrev", sh["sw"].values),
        "HD_CF": ("shooting_team_abbrev", sh["is_HD"].values),
        "HD_CF_adj": ("shooting_team_abbrev", (sh["is_HD"] * sh["sw"]).values),
        "TNFI_CF": ("shooting_team_abbrev", sh["is_TNFI"].values),
        "TNFI_CF_adj": ("shooting_team_abbrev", (sh["is_TNFI"] * sh["sw"]).values),
        "CNFI_CF": ("shooting_team_abbrev", sh["is_CNFI"].values),
        "MNFI_CF": ("shooting_team_abbrev", sh["is_MNFI"].values),
        "FNFI_CF": ("shooting_team_abbrev", sh["is_FNFI"].values),
        "HD_FF": ("shooting_team_abbrev", sh["is_HD_fen"].values),
        "HD_FF_adj": ("shooting_team_abbrev", (sh["is_HD_fen"] * sh["sw"]).values),
        "TNFI_FF": ("shooting_team_abbrev", sh["is_TNFI_fen"].values),
        "TNFI_FF_adj": ("shooting_team_abbrev", (sh["is_TNFI_fen"] * sh["sw"]).values),
        "CNFI_FF": ("shooting_team_abbrev", sh["is_CNFI_fen"].values),
        "MNFI_FF": ("shooting_team_abbrev", sh["is_MNFI_fen"].values),
        "FNFI_FF": ("shooting_team_abbrev", sh["is_FNFI_fen"].values),
    }
    cols_ag = {
        "CA": ("def_team", np.ones(len(sh))),
        "CA_adj": ("def_team", sh["sw"].values),
        "HD_CA": ("def_team", sh["is_HD"].values),
        "HD_CA_adj": ("def_team", (sh["is_HD"] * sh["sw"]).values),
        "TNFI_CA": ("def_team", sh["is_TNFI"].values),
        "TNFI_CA_adj": ("def_team", (sh["is_TNFI"] * sh["sw"]).values),
        "CNFI_CA": ("def_team", sh["is_CNFI"].values),
        "MNFI_CA": ("def_team", sh["is_MNFI"].values),
        "FNFI_CA": ("def_team", sh["is_FNFI"].values),
        "HD_FA": ("def_team", sh["is_HD_fen"].values),
        "HD_FA_adj": ("def_team", (sh["is_HD_fen"] * sh["sw"]).values),
        "TNFI_FA": ("def_team", sh["is_TNFI_fen"].values),
        "TNFI_FA_adj": ("def_team", (sh["is_TNFI_fen"] * sh["sw"]).values),
        "CNFI_FA": ("def_team", sh["is_CNFI_fen"].values),
        "MNFI_FA": ("def_team", sh["is_MNFI_fen"].values),
        "FNFI_FA": ("def_team", sh["is_FNFI_fen"].values),
    }
    team_dfs: dict[str, pd.DataFrame] = {}
    for col, (key, vals) in {**cols_for, **cols_ag}.items():
        tmp = sh.assign(__v=vals).groupby(["season", key])["__v"].sum().reset_index()
        tmp.columns = ["season", "team", col]
        team_dfs[col] = tmp

    team_df = team_dfs["CF"]
    for k, v in team_dfs.items():
        if k == "CF":
            continue
        team_df = team_df.merge(v, on=["season", "team"], how="outer")
    team_df = team_df.fillna(0)

    # Compute percentages (using Fenwick for NFI per the canonical framework)
    def pct(a, b):
        return np.where((a + b) > 0, a / (a + b), np.nan)

    team_df["CF_pct"] = pct(team_df["CF"], team_df["CA"])
    team_df["CF_score_adj_pct"] = pct(team_df["CF_adj"], team_df["CA_adj"])
    team_df["HD_CF_pct"] = pct(team_df["HD_CF"], team_df["HD_CA"])
    team_df["HD_CF_score_adj_pct"] = pct(team_df["HD_CF_adj"], team_df["HD_CA_adj"])
    team_df["HD_FF_pct"] = pct(team_df["HD_FF"], team_df["HD_FA"])
    team_df["HD_FF_score_adj_pct"] = pct(team_df["HD_FF_adj"], team_df["HD_FA_adj"])
    team_df["TNFI_pct"] = pct(team_df["TNFI_FF"], team_df["TNFI_FA"])
    team_df["TNFI_score_adj_pct"] = pct(team_df["TNFI_FF_adj"], team_df["TNFI_FA_adj"])
    team_df["CNFI_pct"] = pct(team_df["CNFI_FF"], team_df["CNFI_FA"])
    team_df["MNFI_pct"] = pct(team_df["MNFI_FF"], team_df["MNFI_FA"])
    team_df["FNFI_pct"] = pct(team_df["FNFI_FF"], team_df["FNFI_FA"])
    team_df["CF_zone_adj_pct"] = team_df["CF_pct"]
    team_df["CF_score_zone_adj_pct"] = team_df["CF_score_adj_pct"]
    team_df["HD_CF_score_zone_adj_pct"] = team_df["HD_CF_score_adj_pct"]
    team_df["TNFI_zone_adj_pct"] = team_df["TNFI_pct"]
    team_df["TNFI_score_zone_adj_pct"] = team_df["TNFI_score_adj_pct"]

    info(f"  built {len(team_df)} (season, team) rows for new metrics_team.csv")

    # team_df season column is currently str (from groupby on the cast season).
    # Coerce all merge inputs to str season to avoid type-mismatch errors.
    team_df["season"] = team_df["season"].astype(str)

    # Drop rows where zone is null (events present in raw but missing from shots_tagged —
    # typically out-of-rink, no-coordinate, or pre-tagging events that can't be zone-classified).
    n_before_zone_drop = len(team_df)
    # team_df is already aggregated, so this no-op (zone filter was applied per-event upstream)
    info(f"  team_df rows after aggregation: {n_before_zone_drop}")

    # Filter to the canonical season cohort (2021-22 → 2025-26), matching standings_pool5
    valid_seasons = {"20212022", "20222023", "20232024", "20242025", "20252026"}
    team_df = team_df[team_df["season"].isin(valid_seasons)].copy()
    info(f"  team_df rows after season-cohort filter (2021-22 → 2025-26): {len(team_df)}")

    # Merge with standings to get points, gp, etc. (matches 04 + 07 pipeline)
    info("Merging with standings...")
    st = pd.read_csv(STANDINGS)
    st["season"] = st["season"].astype(str)
    team_df = team_df.merge(st, on=["season", "team"], how="left")

    # Merge with composite (matches 07_finalize_outputs.py)
    info("Merging with team_composite_NFI...")
    if not COMPOSITE.exists():
        err(f"Missing required input: {COMPOSITE}. Cannot regenerate team_level_all_metrics.csv.")
        return {"applied": False, "reason": "missing team_composite_NFI.csv"}
    comp = pd.read_csv(COMPOSITE)
    comp["season"] = comp["season"].astype(str)  # coerce to str to match team_df
    pillar_cols = ["P1_NF_F", "P2_DefF", "P3_OffF", "P4_DefD", "P5_OffD", "P6_SvPct", "P7_SvPct"]
    merge_cols = ["season", "team", "NFI_composite"] + pillar_cols
    team_full = team_df.merge(comp[merge_cols], on=["season", "team"], how="left")

    # Reorder columns to match the existing team_level_all_metrics.csv schema
    if TEAM_LEVEL.exists():
        existing_cols = list(pd.read_csv(TEAM_LEVEL, nrows=1).columns)
        # Use existing column order; drop anything we don't have, keep extras at end
        ordered = [c for c in existing_cols if c in team_full.columns]
        extras = [c for c in team_full.columns if c not in ordered]
        team_full = team_full[ordered + extras]

    # 2025-26 sanity print
    print()
    info("New 2025-26 team aggregates (top 5 by TNFI_pct, post-merge):")
    s25 = team_full[team_full["season"] == "20252026"].copy()
    if len(s25):
        s25 = s25.sort_values("TNFI_pct", ascending=False).head(5)
        for _, r in s25.iterrows():
            print(
                f"    {r['team']:<5} TNFI_pct={r['TNFI_pct']:.4f} "
                f"CNFI_FF+MNFI_FF={int(r.get('CNFI_FF', 0) + r.get('MNFI_FF', 0))} "
                f"CNFI_FA+MNFI_FA={int(r.get('CNFI_FA', 0) + r.get('MNFI_FA', 0))}"
            )

    if dry_run:
        info("DRY RUN — would write the following files:")
        info(f"    backup dir would be created: {BACKUP_DIR}")
        info(f"    would back up: {METRICS_TM.relative_to(ROOT)}")
        info(f"    would back up: {TEAM_LEVEL.relative_to(ROOT)}")
        info(f"    would write new: {METRICS_TM.relative_to(ROOT)}  (rows: {len(team_df)})")
        info(f"    would write new: {TEAM_LEVEL.relative_to(ROOT)}  (rows: {len(team_full)})")
        return {
            "applied": False,
            "rows_metrics_team": len(team_df),
            "rows_team_level": len(team_full),
            "team_full_df": team_full,  # carry forward to Phase 4 verification
        }

    # APPLY mode — actually write
    info(f"Creating backup directory: {BACKUP_DIR}")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src in (METRICS_TM, TEAM_LEVEL):
        if src.exists():
            dst = BACKUP_DIR / src.name
            try:
                shutil.copy2(src, dst)
                ok(f"backed up {src.relative_to(ROOT)} → {dst.relative_to(ROOT)}")
            except Exception as e:
                err(f"failed to back up {src}: {e}")
                raise

    # Write new metrics_team.csv
    try:
        # Match the existing column order of metrics_team.csv if present
        if METRICS_TM.exists():
            existing_mt_cols = list(pd.read_csv(METRICS_TM, nrows=1).columns)
            ordered_mt = [c for c in existing_mt_cols if c in team_df.columns]
            extras_mt = [c for c in team_df.columns if c not in ordered_mt]
            team_df_out = team_df[ordered_mt + extras_mt]
        else:
            team_df_out = team_df
        team_df_out.to_csv(METRICS_TM, index=False)
        ok(f"wrote {METRICS_TM.relative_to(ROOT)} ({len(team_df_out)} rows)")
    except Exception as e:
        err(f"failed to write {METRICS_TM}: {e}")
        raise

    # Write new team_level_all_metrics.csv
    try:
        team_full.to_csv(TEAM_LEVEL, index=False)
        ok(f"wrote {TEAM_LEVEL.relative_to(ROOT)} ({len(team_full)} rows)")
    except Exception as e:
        err(f"failed to write {TEAM_LEVEL}: {e}")
        raise

    return {
        "applied": True,
        "rows_metrics_team": len(team_df),
        "rows_team_level": len(team_full),
        "team_full_df": team_full,
    }


# =============================================================================
# PHASE 4 — VERIFICATION
# =============================================================================


def phase4_verify(canonical: str, migration_result: dict, dry_run: bool) -> bool:
    banner("PHASE 4 — VERIFICATION")
    all_pass = True

    # 1. League-level sum check on canonical-definition counts
    info("Recomputing 32-team CNFI+MNFI counts directly from raw under canonical definition...")
    if canonical == "A":
        canon = compute_team_metrics("A")
    else:
        canon = compute_team_metrics("B")
    sum_a = int(canon["attack"].sum())
    sum_s = int(canon["suppress"].sum())
    if sum_a == sum_s:
        ok(f"League sum check: attack ({sum_a:,}) == suppress ({sum_s:,})  [PASS]")
    else:
        err(f"League sum check FAIL: attack ({sum_a:,}) != suppress ({sum_s:,})")
        all_pass = False

    # 2. Spot-check 5 teams
    info(f"Spot-checking {len(SPOT_CHECK_TEAMS)} teams...")
    if dry_run:
        # Compare canon (recomputed) to migration_result's in-memory team_full_df if present,
        # else to current team_level_all_metrics.csv
        ref_df = migration_result.get("team_full_df")
        ref_label = "in-memory regenerated" if ref_df is not None else "current team_level_all_metrics.csv"
        if ref_df is None:
            ref_df = pd.read_csv(TEAM_LEVEL)
            ref_df["season"] = ref_df["season"].astype(str)
    else:
        # APPLY mode: read the freshly-written file
        ref_df = pd.read_csv(TEAM_LEVEL)
        ref_df["season"] = ref_df["season"].astype(str)
        ref_label = "post-migration team_level_all_metrics.csv"

    info(f"  reference: {ref_label}")
    print(
        f"    {'team':<5} {'canon_atk':>10} {'canon_sup':>10} {'canon_pct':>10} | "
        f"{'ref_atk':>9} {'ref_sup':>9} {'ref_pct':>9} | {'match':>6}"
    )
    print("    " + "-" * 92)
    for tm in SPOT_CHECK_TEAMS:
        c = canon[canon["team"] == tm]
        if len(c) == 0:
            warn(f"    {tm}: not found in canonical recompute; skipping")
            continue
        c = c.iloc[0]
        r = ref_df[(ref_df["season"] == "20252026") & (ref_df["team"] == tm)]
        if len(r) == 0:
            warn(f"    {tm}: not found in reference; skipping")
            continue
        r = r.iloc[0]
        # Reference attack/suppress (CNFI+MNFI)
        ref_atk = int(r.get("CNFI_FF", 0) + r.get("MNFI_FF", 0))
        ref_sup = int(r.get("CNFI_FA", 0) + r.get("MNFI_FA", 0))
        ref_pct = ref_atk / (ref_atk + ref_sup) if (ref_atk + ref_sup) > 0 else float("nan")
        match = (
            int(c["attack"]) == ref_atk
            and int(c["suppress"]) == ref_sup
        )
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(
            f"    {tm:<5} {int(c['attack']):>10} {int(c['suppress']):>10} {c['nfi_pct']:>10.4f} | "
            f"{ref_atk:>9} {ref_sup:>9} {ref_pct:>9.4f} | {status:>6}"
        )

    return all_pass


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run audit + impact + verification without modifying any files")
    parser.add_argument("--apply", action="store_true",
                        help="Run the migration (Phase 3) and write new outputs")
    args = parser.parse_args()

    if args.dry_run and args.apply:
        err("Pass exactly one of --dry-run or --apply, not both.")
        return 2
    if not (args.dry_run or args.apply):
        err("Pass either --dry-run or --apply.")
        return 2

    if CANONICAL_DEFINITION not in ("A", "B"):
        err(f"CANONICAL_DEFINITION must be 'A' or 'B' (got '{CANONICAL_DEFINITION}'). Edit script.")
        return 2

    banner(
        f"DEFINITIONAL CLEANUP — canonical = Variant {CANONICAL_DEFINITION}, "
        f"mode = {'DRY-RUN' if args.dry_run else 'APPLY'}",
        char="█",
    )
    info(f"Source files (current state):")
    for fp in (RAW_EVENTS, TAGGED, TEAM_LEVEL, METRICS_TM, COMPOSITE, GAME_IDS, STANDINGS):
        info(f"  {file_summary(fp)}")

    blockers: list[str] = []

    # Phase 1
    try:
        audit = phase1_audit()
    except Exception as e:
        err(f"Phase 1 failed: {e}")
        traceback.print_exc()
        return 3
    third_def = [r for r in audit if r["category"].startswith("PARTIAL_F")
                 or r["category"] == "MIXED (uses both A and F)"]
    if third_def:
        for r in third_def:
            blockers.append(f"non-canonical/mixed definition in {r['path']} [{r['category']}]")

    # Phase 2
    try:
        a, f, p2_blockers = phase2_impact()
    except Exception as e:
        err(f"Phase 2 failed: {e}")
        traceback.print_exc()
        return 3
    blockers.extend(p2_blockers)

    # Phase 3
    try:
        migration = phase3_migrate(CANONICAL_DEFINITION, dry_run=args.dry_run)
    except Exception as e:
        err(f"Phase 3 failed: {e}")
        traceback.print_exc()
        return 3

    # Phase 4
    try:
        verify_ok = phase4_verify(CANONICAL_DEFINITION, migration, dry_run=args.dry_run)
    except Exception as e:
        err(f"Phase 4 failed: {e}")
        traceback.print_exc()
        return 3

    # Final summary
    banner("FINAL SUMMARY", char="█")
    info(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")
    info(f"Canonical definition chosen: Variant {CANONICAL_DEFINITION}"
         + (" (state=='ES')" if CANONICAL_DEFINITION == "A" else " (situation_code==1551 AND period_type=='REG')"))
    info(f"Phase 1: {len(audit)} .py files reference an ES filter "
         f"({sum(1 for r in audit if r['uses_A'] and not r['uses_F'])} use Variant A, "
         f"{sum(1 for r in audit if r['uses_F'] and not r['uses_A'])} use Variant F, "
         f"{sum(1 for r in audit if r['uses_A'] and r['uses_F'])} use both)")
    info(f"Phase 2: 32-team max |Δ NFI%| = {(a.merge(f, on='team', suffixes=('_A','_F'))['nfi_pct_F'] - a.merge(f, on='team', suffixes=('_A','_F'))['nfi_pct_A']).abs().max():.4f}")
    if args.apply:
        info(f"Phase 3: APPLIED — backup at {BACKUP_DIR}")
        info(f"  metrics_team.csv: {migration.get('rows_metrics_team', '?')} rows written")
        info(f"  team_level_all_metrics.csv: {migration.get('rows_team_level', '?')} rows written")
    else:
        info("Phase 3: would have regenerated metrics_team.csv and team_level_all_metrics.csv")
    info(f"Phase 4: {'PASS' if verify_ok else 'FAIL'}")

    print()
    if blockers:
        warn(f"Blockers / items needing attention ({len(blockers)}):")
        for b in blockers:
            print(f"    • {b}")
    else:
        ok("No blockers detected.")

    print()
    info("Downstream files NOT regenerated by this script (require their producers to be re-run "
         "after their .py source switches to canonical definition):")
    if CANONICAL_DEFINITION == "B":
        downstream_to_rebuild = [
            ("decision_tree_stage123.py", "stage1_*, stage2_*, stage3_* in zone_adjustment/complete_decision_tree/"),
            ("decision_tree_stage4.py",   "stage4_*, stage5_*, stage6_* + /tmp/s4_ppdf.pkl"),
            ("fully_adjusted.py",         "fully_adjusted/horse_race_fully_adjusted.csv + /tmp/fa_shared.pkl"),
            ("fa_linemate_without_me.py", "fully_adjusted/player_fully_adjusted.csv (linemate-corrected)"),
            ("tnfi_relatives_pp_pk.py",   "fully_adjusted/player_fully_adjusted.csv (RF/RA columns)"),
            ("rename_and_momentum.py",    "fully_adjusted/* publication-named outputs"),
            ("05_horse_race.py",          "horse_race_univariate.csv, horse_race_multivariate.csv, team_composite_NFI.csv"),
            ("07_finalize_outputs.py",    "horse_race_summary.csv, horse_race_head_to_head.csv (re-run after 05)"),
            ("18_team_construction_model.py", "team_construction_model.csv"),
            ("19_team_archetypes_quartile.py","team_construction_model_quartile.csv"),
            ("post2_horserace_consistent.py", "(this script ALREADY uses Variant F — no change needed)"),
            ("build_playoff_data.py",     "fully_adjusted/player_fully_adjusted_playoffs.csv + tnzi playoff files"),
        ]
        for prod, outs in downstream_to_rebuild:
            info(f"  {prod}: {outs}")
        info("  (After updating each producer to use Variant F filter, re-run them in pipeline order.)")
    else:
        info("  (Variant A is already canonical; downstream files match current state.)")

    return 0 if verify_ok and not blockers else 1


if __name__ == "__main__":
    sys.exit(main())
