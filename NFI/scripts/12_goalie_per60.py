#!/usr/bin/env python3
"""
Add NFI-GSAx per 60 to the goalie comparison.

Uses:
  - goalie_nfi_gsax.csv (calibrated cumulative NFI-GSAx)
  - player_toi.csv (toi_ES_sec already summed from shift_data.csv,
    regulation only, ES-state intersected via event-derived state intervals)

Output columns per goalie:
  spatial_save_pct         — rate-based quality
  NFI_GSAx_cumulative      — total goals saved vs expected (calibrated)
  NFI_GSAx_per60           — per-60 rate
  total_faced_TNFI         — sample-size context

Flag goalies where |rank_per60 - rank_cumulative| > 20.
Overwrite goalie_metric_comparison.csv.
"""
import pandas as pd

OUT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/NFI/output"

gs = pd.read_csv(f"{OUT}/goalie_nfi_gsax.csv")
sp = pd.read_csv(f"{OUT}/goalie_spatial_savepct.csv")
toi = pd.read_csv(f"{OUT}/player_toi.csv")

toi_g = toi[["player_id","toi_ES_sec"]].rename(columns={"player_id":"goalie_id"})
toi_g["toi_ES_min"] = toi_g["toi_ES_sec"] / 60.0

m = gs.merge(toi_g, on="goalie_id", how="left")
m["NFI_GSAx_per60"] = m["NFI_GSAx_calibrated"] / m["toi_ES_min"] * 60.0

m = m.rename(columns={"NFI_GSAx_calibrated":"NFI_GSAx_cumulative"})

# Merge spatial save% fields
spm = sp.rename(columns={"rank_spatial":"rank_spatial_savepct"})[
    ["goalie_id","goalie_name","faced","goals","spatial_save_pct","sv_lo","sv_hi",
     "rank_spatial_savepct"]
]
m = m.merge(spm, on=["goalie_id","goalie_name"], how="outer")

# recompute ranks (min-method, descending; lowest value = worst)
m["rank_cumulative"] = m["NFI_GSAx_cumulative"].rank(method="min", ascending=False)
m["rank_per60"] = m["NFI_GSAx_per60"].rank(method="min", ascending=False)

m["rank_diff_per60_vs_cum"] = (m["rank_per60"] - m["rank_cumulative"]).abs()
m["flag_gt20_shift"] = (m["rank_diff_per60_vs_cum"] > 20).astype(int)

# Round
for c in ["NFI_GSAx_cumulative","NFI_GSAx_per60","toi_ES_min","xG","NFI_GSAx","xG_calibrated"]:
    if c in m.columns: m[c] = m[c].round(2)

# Reorder columns
cols = ["goalie_id","goalie_name",
        "total_faced","total_goals","toi_ES_min",
        "spatial_save_pct","sv_lo","sv_hi","rank_spatial_savepct",
        "NFI_GSAx_cumulative","rank_cumulative",
        "NFI_GSAx_per60","rank_per60",
        "rank_diff_per60_vs_cum","flag_gt20_shift",
        "xG_calibrated","xG","NFI_GSAx",  # spec-rate for reference
        ]
cols = [c for c in cols if c in m.columns]
m = m[cols].sort_values("rank_per60", na_position="last").reset_index(drop=True)
m.to_csv(f"{OUT}/goalie_metric_comparison.csv", index=False)

print(f"Goalies in comparison: {len(m)}")
print(f"Flagged (|rank_diff|>20): {m['flag_gt20_shift'].sum()}")

print("\n=== Top 15 by NFI-GSAx per 60 ===")
top = m.dropna(subset=["rank_per60"]).head(15)
print(top[["rank_per60","goalie_name","toi_ES_min","total_faced","spatial_save_pct",
          "NFI_GSAx_cumulative","rank_cumulative","NFI_GSAx_per60",
          "rank_diff_per60_vs_cum","flag_gt20_shift"]].to_string(index=False))

print("\n=== Bottom 5 by NFI-GSAx per 60 ===")
bot = m.dropna(subset=["rank_per60"]).sort_values("rank_per60", ascending=False).head(5)
print(bot[["rank_per60","goalie_name","toi_ES_min","total_faced","spatial_save_pct",
          "NFI_GSAx_cumulative","rank_cumulative","NFI_GSAx_per60",
          "rank_diff_per60_vs_cum","flag_gt20_shift"]].to_string(index=False))

print("\n=== All flagged goalies (|rank_diff|>20) ===")
flagged = m[m["flag_gt20_shift"]==1].sort_values("NFI_GSAx_per60", ascending=False)
print(flagged[["goalie_name","toi_ES_min","total_faced","spatial_save_pct",
               "NFI_GSAx_cumulative","rank_cumulative",
               "NFI_GSAx_per60","rank_per60","rank_diff_per60_vs_cum"]].to_string(index=False))
