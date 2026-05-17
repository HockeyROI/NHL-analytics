#!/usr/bin/env python3
"""
NFI-Score Step 5 — Off/Def decomposition by score state.

Reads the Step 3 wide table and the Step 2 per-state TOI file. For each of the
five score states, computes the TOI-weighted league average across the 379
forwards for Off, Def, and Net rates per 60.

TOI-weighting reduces to the pooled rate identity (Σ events_p / Σ TOI_p × 60),
so a 4th-liner is weighted by their actual ice time, not equally with McDavid.

Direction checks:
  - Hard stop if Net slope (Down2 → Up2) is not positive — would indicate a
    sign-convention bug somewhere upstream.
  - Soft warning if Off slope is positive — would contradict the standard
    score-effects pattern and warrant investigation before publishing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/ashgarg/Documents/HockeyROI")
OUT_DIR = ROOT / "NFI" / "NFI_Score_Adj" / "2026" / "05" / "Output"
WIDE = OUT_DIR / "nfi_score_player_rates.csv"
TOI = OUT_DIR / "per_state_toi.csv"
OUT_CSV = OUT_DIR / "league_avg_by_state.csv"

STATES = ["Down2", "Down1", "Tied", "Up1", "Up2"]
TYPES = ["Off", "Def", "Net"]


def stop(msg: str, code: int = 2) -> int:
    print(f"[step5] STOP — {msg}", file=sys.stderr)
    return code


def main() -> int:
    if not WIDE.exists() or not TOI.exists():
        return stop(f"missing input(s): WIDE={WIDE.exists()}, TOI={TOI.exists()}")

    wide = pd.read_csv(WIDE)
    print(f"[step5] read {WIDE}: {len(wide)} players, {len(wide.columns)} cols")

    rows = []
    for s in STATES:
        toi_col = f"TOI_{s}"
        w = wide[toi_col].fillna(0).to_numpy()
        total_toi = float(w.sum())
        rec = {"state": s, "total_TOI": round(total_toi, 2)}
        for t in TYPES:
            rate_col = f"NFI-Score-{s}-{t}"
            v = wide[rate_col].fillna(0).to_numpy()
            avg = float((v * w).sum() / total_toi) if total_toi > 0 else 0.0
            rec[f"league_avg_{t}"] = round(avg, 4)
        rows.append(rec)
    df = pd.DataFrame(rows)

    # console table
    print(f"\nLEAGUE AVERAGE BY SCORE STATE (TOI-weighted, 379-forward universe)")
    print()
    print(f"State   {'TOI(min)':>10}    {'Off':>6}    {'Def':>6}    {'Net':>6}")
    for _, r in df.iterrows():
        print(f"{r['state']:<6}  {r['total_TOI']:>10,.0f}    "
              f"{r['league_avg_Off']:>6.2f}    "
              f"{r['league_avg_Def']:>6.2f}    "
              f"{r['league_avg_Net']:>+6.2f}")

    # write CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[step5] wrote {OUT_CSV}")

    # slope = average step Down2 → Up2 (4 steps across 5 states)
    def slope(col):
        vals = df[f"league_avg_{col}"].to_numpy()
        steps = np.diff(vals)
        return float(steps.mean())

    off_slope = slope("Off")
    def_slope = slope("Def")
    net_slope = slope("Net")

    def label(s):
        if s > 0.05: return "rising"
        if s < -0.05: return "falling"
        return "flat"

    print(f"\nDIRECTION CHECK (avg per-step change Down2 → Up2, 4 steps):")
    print(f"  Off slope: {off_slope:+.3f} per state-step — {label(off_slope)} from trailing to leading")
    print(f"  Def slope: {def_slope:+.3f} per state-step — {label(def_slope)} from trailing to leading")
    print(f"  Net slope: {net_slope:+.3f} per state-step — {label(net_slope)} from trailing to leading")

    # soft flag on positive Off slope
    if off_slope > 0.05:
        print()
        print("  *** SOFT FLAG: Off slope is POSITIVE — leading teams are generating MORE")
        print("      shots than trailing teams in absolute terms. This contradicts the")
        print("      standard score-effects pattern (trailing teams push, lead teams sit")
        print("      back). Worth investigating before publishing.")

    # hard stop on non-positive Net slope
    if net_slope <= 0:
        return stop(f"Net slope is {net_slope:+.3f} (not positive) — "
                    f"sign convention or filter bug somewhere upstream")

    print(f"\n[step5] direction PASS — Net rises from trailing to leading as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
