#!/usr/bin/env python3
"""
NFI-Score Step 5 (DEFENSEMEN) — Off/Def decomposition by score state for D.

Parallel to 05_decompose_off_def_by_state.py. Reads the D wide table and D
per-state TOI file. Direction checks identical.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/ashgarg/Documents/HockeyROI")
OUT_DIR = ROOT / "NFI" / "NFI_Score_Adj" / "2026" / "05" / "Output"
WIDE = OUT_DIR / "nfi_score_player_rates_D.csv"
TOI = OUT_DIR / "per_state_toi_D.csv"
OUT_CSV = OUT_DIR / "league_avg_by_state_D.csv"

STATES = ["Down2", "Down1", "Tied", "Up1", "Up2"]
TYPES = ["Off", "Def", "Net"]


def stop(msg: str, code: int = 2) -> int:
    print(f"[step5D] STOP — {msg}", file=sys.stderr)
    return code


def main() -> int:
    if not WIDE.exists() or not TOI.exists():
        return stop(f"missing input(s): WIDE={WIDE.exists()}, TOI={TOI.exists()}")

    wide = pd.read_csv(WIDE)
    print(f"[step5D] read {WIDE}: {len(wide)} players, {len(wide.columns)} cols")

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

    print(f"\nLEAGUE AVERAGE BY SCORE STATE (TOI-weighted, 198-defenseman universe)")
    print()
    print(f"State   {'TOI(min)':>10}    {'Off':>6}    {'Def':>6}    {'Net':>6}")
    for _, r in df.iterrows():
        print(f"{r['state']:<6}  {r['total_TOI']:>10,.0f}    "
              f"{r['league_avg_Off']:>6.2f}    "
              f"{r['league_avg_Def']:>6.2f}    "
              f"{r['league_avg_Net']:>+6.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[step5D] wrote {OUT_CSV}")

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

    if off_slope > 0.05:
        print()
        print("  *** SOFT FLAG: Off slope POSITIVE — leading teams generating MORE")

    if net_slope <= 0:
        return stop(f"Net slope is {net_slope:+.3f} (not positive)")

    print(f"\n[step5D] direction PASS — Net rises from trailing to leading as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
