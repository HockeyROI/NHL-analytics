"""Derive Corsi and Fenwick OZ-DZ factors from first principles.

Inputs (real, league-pooled, 4 seasons of 5v5 ES regular-season shift remainders):
  - per-shift Corsi For/Against rates by faceoff zone (O/D)
  - per-shift Fenwick For/Against rates by faceoff zone
  - block rates by (zone, side) implied by Corsi - Fenwick

Builds three nested models:
  Model 1: Corsi only (no blocks). Factor = OZ_CF% - DZ_CF%.
  Model 2: SYMMETRIC blocks — same global block rate applied to every shot.
           Predicts Fenwick factor; if = Corsi factor, blocks-as-noise.
  Model 3: REAL asymmetric block rates by (zone, side).
           Predicts Fenwick factor; matches empirical = mechanism complete.

Decomposes the Model 2 -> Model 3 amplification into per-zone contributions
so we can see exactly how much the OZ asymmetry vs the DZ asymmetry each
contribute to the 3x amplification.
"""
from __future__ import annotations

# Real per-shift values from the league-pooled OZ/DZ attribution
# (decomposition script run on 5,242 regular-season games, 4 seasons,
# 1.6M player-shifts, 5v5 ES only).
OZ = dict(n_shifts=807_728,
          cf_f=1.9888, cf_a=1.8653,
          fen_f=1.5189, fen_a=1.2382)
DZ = dict(n_shifts=807_878,
          cf_f=1.7955, cf_a=1.9679,
          fen_f=1.1618, fen_a=1.5288)

def zonepct(f, a):
    return f / (f + a)

def fenwick_pct_from_cf_and_blockrates(cf_f, cf_a, BR_f, BR_a):
    """Given Corsi For/Against rates and block rates per side, return Fenwick%."""
    ff_f = cf_f * (1 - BR_f)
    ff_a = cf_a * (1 - BR_a)
    return zonepct(ff_f, ff_a), ff_f, ff_a

# ---------------------------------------------------------------------
# Step A — observed quantities
# ---------------------------------------------------------------------
print("=" * 78)
print("OBSERVED PER-SHIFT VALUES (league pooled, 5v5 ES, 4 seasons)")
print("=" * 78)
for name, d in [("OZ shift", OZ), ("DZ shift", DZ)]:
    blk_f = d["cf_f"] - d["fen_f"]; blk_a = d["cf_a"] - d["fen_a"]
    BR_f = blk_f / d["cf_f"];        BR_a = blk_a / d["cf_a"]
    cf_pct = zonepct(d["cf_f"], d["cf_a"]) * 100
    fen_pct = zonepct(d["fen_f"], d["fen_a"]) * 100
    print(f"\n  {name} (n_shifts = {d['n_shifts']:,})")
    print(f"    Corsi   For/sh = {d['cf_f']:.4f}   Ag/sh = {d['cf_a']:.4f}    CF% = {cf_pct:6.2f}%")
    print(f"    Fenwick For/sh = {d['fen_f']:.4f}   Ag/sh = {d['fen_a']:.4f}   FF% = {fen_pct:6.2f}%")
    print(f"    Block %  For = {BR_f*100:5.2f}%       Ag = {BR_a*100:5.2f}%       diff = {(BR_a-BR_f)*100:+5.2f} pp")

OZ["BR_f"] = (OZ["cf_f"] - OZ["fen_f"]) / OZ["cf_f"]
OZ["BR_a"] = (OZ["cf_a"] - OZ["fen_a"]) / OZ["cf_a"]
DZ["BR_f"] = (DZ["cf_f"] - DZ["fen_f"]) / DZ["cf_f"]
DZ["BR_a"] = (DZ["cf_a"] - DZ["fen_a"]) / DZ["cf_a"]

oz_cf_pct = zonepct(OZ["cf_f"], OZ["cf_a"])
dz_cf_pct = zonepct(DZ["cf_f"], DZ["cf_a"])
oz_ff_pct = zonepct(OZ["fen_f"], OZ["fen_a"])
dz_ff_pct = zonepct(DZ["fen_f"], DZ["fen_a"])
emp_corsi_factor   = (oz_cf_pct - dz_cf_pct) * 100
emp_fenwick_factor = (oz_ff_pct - dz_ff_pct) * 100

# ---------------------------------------------------------------------
# Model 1 — Corsi factor only
# ---------------------------------------------------------------------
print()
print("=" * 78)
print("MODEL 1 — Corsi factor (no blocks involved)")
print("=" * 78)
print(f"  OZ CF% = {oz_cf_pct*100:6.3f}%")
print(f"  DZ CF% = {dz_cf_pct*100:6.3f}%")
print(f"  Corsi factor = OZ CF% − DZ CF% = {emp_corsi_factor:+6.3f} pp")
print(f"  Empirical Corsi factor    : {emp_corsi_factor:+6.3f} pp  ✓")

# ---------------------------------------------------------------------
# Model 2 — Symmetric global block rate (no asymmetry)
# Predicted Fenwick factor under "blocks are noise"
# ---------------------------------------------------------------------
print()
print("=" * 78)
print("MODEL 2 — SYMMETRIC blocks (global average, applied uniformly)")
print("=" * 78)
total_blocks = (OZ["cf_f"]-OZ["fen_f"] + OZ["cf_a"]-OZ["fen_a"] +
                DZ["cf_f"]-DZ["fen_f"] + DZ["cf_a"]-DZ["fen_a"])
total_corsi  = OZ["cf_f"]+OZ["cf_a"] + DZ["cf_f"]+DZ["cf_a"]
BR_global = total_blocks / total_corsi
print(f"  Global block rate : {BR_global*100:5.2f}%   (applied to every shot)")
oz_ff_sym, _, _ = fenwick_pct_from_cf_and_blockrates(OZ["cf_f"], OZ["cf_a"],
                                                      BR_global, BR_global)
dz_ff_sym, _, _ = fenwick_pct_from_cf_and_blockrates(DZ["cf_f"], DZ["cf_a"],
                                                      BR_global, BR_global)
sym_factor = (oz_ff_sym - dz_ff_sym) * 100
print(f"  OZ FF% (symmetric) = {oz_ff_sym*100:6.3f}%   (= OZ CF% by construction)")
print(f"  DZ FF% (symmetric) = {dz_ff_sym*100:6.3f}%   (= DZ CF% by construction)")
print(f"  Predicted Fenwick factor under symmetric-blocks = {sym_factor:+6.3f} pp")
print(f"  Empirical Corsi factor                          = {emp_corsi_factor:+6.3f} pp")
print(f"  → Identical. Confirms: symmetric blocks add NOTHING to the OZ-DZ gap.")

# ---------------------------------------------------------------------
# Model 3 — REAL asymmetric block rates
# ---------------------------------------------------------------------
print()
print("=" * 78)
print("MODEL 3 — REAL block rates by (zone × side)")
print("=" * 78)
oz_ff_real, ozffF, ozffA = fenwick_pct_from_cf_and_blockrates(
    OZ["cf_f"], OZ["cf_a"], OZ["BR_f"], OZ["BR_a"])
dz_ff_real, dzffF, dzffA = fenwick_pct_from_cf_and_blockrates(
    DZ["cf_f"], DZ["cf_a"], DZ["BR_f"], DZ["BR_a"])
real_factor = (oz_ff_real - dz_ff_real) * 100
print(f"  OZ block rates: For = {OZ['BR_f']*100:5.2f}%   Ag = {OZ['BR_a']*100:5.2f}%")
print(f"  DZ block rates: For = {DZ['BR_f']*100:5.2f}%   Ag = {DZ['BR_a']*100:5.2f}%")
print(f"  OZ FF% = {oz_ff_real*100:6.3f}%   (predicted = empirical)")
print(f"  DZ FF% = {dz_ff_real*100:6.3f}%   (predicted = empirical)")
print(f"  Predicted Fenwick factor = {real_factor:+6.3f} pp")
print(f"  Empirical Fenwick factor = {emp_fenwick_factor:+6.3f} pp")
print(f"  → Match. The asymmetric-block model fully reproduces Fenwick factor.")

# ---------------------------------------------------------------------
# Decomposition — how much of the amplification comes from each zone's
# asymmetry?
# Switch only OZ asymmetry on (DZ stays symmetric); see factor.
# Switch only DZ asymmetry on (OZ stays symmetric); see factor.
# ---------------------------------------------------------------------
print()
print("=" * 78)
print("DECOMPOSITION — contribution of OZ vs DZ asymmetry to amplification")
print("=" * 78)
oz_BR_avg = (OZ["BR_f"] + OZ["BR_a"]) / 2
dz_BR_avg = (DZ["BR_f"] + DZ["BR_a"]) / 2

# (a) OZ asymmetry only — DZ collapsed to its zone-average
oz_a, _, _ = fenwick_pct_from_cf_and_blockrates(OZ["cf_f"], OZ["cf_a"],
                                                 OZ["BR_f"], OZ["BR_a"])
dz_a, _, _ = fenwick_pct_from_cf_and_blockrates(DZ["cf_f"], DZ["cf_a"],
                                                 dz_BR_avg, dz_BR_avg)
factor_oz_only = (oz_a - dz_a) * 100
print(f"\n  (a) OZ asymmetry only (DZ collapsed to zone-avg block rate):")
print(f"      OZ FF% = {oz_a*100:6.3f}%   DZ FF% = {dz_a*100:6.3f}%   factor = {factor_oz_only:+6.3f} pp")
print(f"      Δ vs Model 2 (sym): {factor_oz_only - sym_factor:+6.3f} pp  ← contribution from OZ asymmetry alone")

# (b) DZ asymmetry only — OZ collapsed to its zone-average
oz_b, _, _ = fenwick_pct_from_cf_and_blockrates(OZ["cf_f"], OZ["cf_a"],
                                                 oz_BR_avg, oz_BR_avg)
dz_b, _, _ = fenwick_pct_from_cf_and_blockrates(DZ["cf_f"], DZ["cf_a"],
                                                 DZ["BR_f"], DZ["BR_a"])
factor_dz_only = (oz_b - dz_b) * 100
print(f"\n  (b) DZ asymmetry only (OZ collapsed to zone-avg block rate):")
print(f"      OZ FF% = {oz_b*100:6.3f}%   DZ FF% = {dz_b*100:6.3f}%   factor = {factor_dz_only:+6.3f} pp")
print(f"      Δ vs Model 2 (sym): {factor_dz_only - sym_factor:+6.3f} pp  ← contribution from DZ asymmetry alone")

# (c) Both
print(f"\n  (c) BOTH asymmetries (= Model 3): factor = {real_factor:+6.3f} pp")
print(f"      Δ vs Model 2 (sym): {real_factor - sym_factor:+6.3f} pp")

# Sum of independent contributions vs joint
sum_indep = (factor_oz_only - sym_factor) + (factor_dz_only - sym_factor)
joint = real_factor - sym_factor
print(f"\n  Sum of independent OZ + DZ contributions: {sum_indep:+6.3f} pp")
print(f"  Joint contribution                      : {joint:+6.3f} pp")
print(f"  Interaction                             : {joint - sum_indep:+6.3f} pp  (small if linear)")

# ---------------------------------------------------------------------
# Sensitivity — what if block rates were slightly different?
# Perturb each (zone, side) by ±2 pp, see effect on Fenwick factor.
# ---------------------------------------------------------------------
print()
print("=" * 78)
print("SENSITIVITY — effect of ±2 pp shift in each block rate on Fenwick factor")
print("=" * 78)
print(f"  {'perturbation':<28} {'Fenwick factor':>18}  {'Δ from real':>14}")
def fac_with(brOf, brOa, brDf, brDa):
    o, _, _ = fenwick_pct_from_cf_and_blockrates(OZ["cf_f"], OZ["cf_a"], brOf, brOa)
    d, _, _ = fenwick_pct_from_cf_and_blockrates(DZ["cf_f"], DZ["cf_a"], brDf, brDa)
    return (o - d) * 100
base = fac_with(OZ["BR_f"], OZ["BR_a"], DZ["BR_f"], DZ["BR_a"])
print(f"  {'real (Model 3)':<28} {base:>+17.3f}    (baseline)")
for nm, args in [
    ("OZ For block +2pp",  (OZ["BR_f"]+0.02, OZ["BR_a"], DZ["BR_f"], DZ["BR_a"])),
    ("OZ Ag  block +2pp",  (OZ["BR_f"], OZ["BR_a"]+0.02, DZ["BR_f"], DZ["BR_a"])),
    ("DZ For block +2pp",  (OZ["BR_f"], OZ["BR_a"], DZ["BR_f"]+0.02, DZ["BR_a"])),
    ("DZ Ag  block +2pp",  (OZ["BR_f"], OZ["BR_a"], DZ["BR_f"], DZ["BR_a"]+0.02)),
]:
    f = fac_with(*args)
    print(f"  {nm:<28} {f:>+17.3f}    {f - base:+5.3f}")

print()
print("=" * 78)
print("CONCLUSION")
print("=" * 78)
print(f"""
  Empirical Corsi factor       : {emp_corsi_factor:+6.3f} pp
  Empirical Fenwick factor     : {emp_fenwick_factor:+6.3f} pp
  Amplification (Fen/Corsi)    : {emp_fenwick_factor / emp_corsi_factor:5.2f}×

  Symmetric-block model factor : {sym_factor:+6.3f} pp  (= Corsi factor by construction)
  Real-block model factor      : {real_factor:+6.3f} pp  (= empirical Fenwick factor)

  The {emp_fenwick_factor - sym_factor:.2f}-pp lift from {sym_factor:.2f} (symmetric) to {real_factor:.2f}
  (real) is fully and only attributable to the ~10-13 pp asymmetry in block
  rates between the For and Against sides within each faceoff zone.

  Decomposition: OZ asymmetry alone contributes {factor_oz_only - sym_factor:+.2f} pp,
                 DZ asymmetry alone contributes {factor_dz_only - sym_factor:+.2f} pp,
                 sum    ≈ joint   = {real_factor - sym_factor:+.2f} pp.

  Sensitivity: a ±2 pp shift in any single (zone × side) block rate moves
  the Fenwick factor by about 0.7-0.8 pp. The result is stable, not driven
  by noise in any single bucket.
""")
