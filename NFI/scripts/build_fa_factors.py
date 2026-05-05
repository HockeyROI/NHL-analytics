# Computes fa_factors.json from cached zone aggregates. Required by fully_adjusted.py and downstream FA scripts.
"""Build /tmp/fa_factors.json from cached /tmp/dt_team_zone_agg.pkl.

Replicates the empirical-factor logic in factor_comparison_5metrics.py [4]:
   factor = OZ_share - DZ_share  (league pooled, fractional pp)
Writes JSON with keys "NFI", "Corsi", "Fenwick" expected by
fully_adjusted.py / fa_linemate_without_me.py / stage5_7_finalize.py.

Prerequisite: decision_tree_stage123.py must have run first (it produces
/tmp/dt_team_zone_agg.pkl with the per-team-season-zone aggregates).

Usage:
    python3 NFI/scripts/build_fa_factors.py
"""
import json
import pandas as pd

POOLED = {"20222023", "20232024", "20242025", "20252026"}

ta = pd.read_pickle("/tmp/dt_team_zone_agg.pkl")
ta = ta[ta["season"].astype(str).isin(POOLED)]
oz = ta[ta["zone"] == "O"]
dz = ta[ta["zone"] == "D"]


def gap(f, a):
    of = oz[f].sum(); oa = oz[a].sum()
    df_ = dz[f].sum(); da = dz[a].sum()
    o = of / (of + oa)
    d = df_ / (df_ + da)
    return o, d, o - d


reg_cor = gap("cf", "ca")
reg_fen = gap("fen_f", "fen_a")
nfi     = gap("cm_f", "cm_a")  # cm_f/cm_a are Fenwick-only post-NFI-fix (stage123 line 179+)

print(f"Corsi:   OZ={reg_cor[0]*100:.3f}%  DZ={reg_cor[1]*100:.3f}%  factor={reg_cor[2]*100:+.3f}pp")
print(f"Fenwick: OZ={reg_fen[0]*100:.3f}%  DZ={reg_fen[1]*100:.3f}%  factor={reg_fen[2]*100:+.3f}pp")
print(f"NFI:     OZ={nfi[0]*100:.3f}%  DZ={nfi[1]*100:.3f}%  factor={nfi[2]*100:+.3f}pp")

factors = {"Corsi": float(reg_cor[2]), "Fenwick": float(reg_fen[2]),
           # NFI: Tulsky 2013 (3.5pp = 0.035), overrides the empirical value computed
           # at line 36. Sibling sites that must stay in sync:
           #   NFI/scripts/update_current_season.py  (NFI_ZA_FACTOR)
           #   NFI/scripts/build_playoff_data.py     (NFI_ZA_FACTOR)
           "NFI": 0.035}
with open("/tmp/fa_factors.json", "w") as f:
    json.dump(factors, f, indent=2)
print(f"\nWrote /tmp/fa_factors.json: {factors}")
