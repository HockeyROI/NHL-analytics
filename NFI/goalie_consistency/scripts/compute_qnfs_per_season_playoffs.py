# Playoff QNFS% — playoff companion to compute_qnfs_per_season.py.
#
# Identical locked design (Fenwick, strict 5v5 ES situation_code 1551,
# regulation periods, net-front = CNFI∪MNFI, per-game >=3 NF shots to count a
# game, GSAx_NF >= 0 = quality save game, per-season league zone rates) but on
# PLAYOFF games (game_type == "playoff").
#
# NO goalie-level qualifying floor (regular uses GP>=10) — every goalie with at
# least one net-front game is emitted, per playoff season PLUS an `all_playoffs`
# pooled row. GP / NF_shots are kept so the Streamlit slider thresholds at
# display. The per-game >=3 NF-shots rule is the metric's own definition (it
# defines what a "net-front game" is) and is retained.
#
# Output: NFI/goalie_consistency/output/qnfs_per_season_playoffs.csv

import pandas as pd
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint

SHOT_FILE = Path("/Users/ashgarg/Documents/HockeyROI/Data/nhl_shot_events.v2.csv")
NAMES_FILE = Path("/Users/ashgarg/Documents/HockeyROI/NFI/output/player_positions.csv")
OUT_FILE = Path(__file__).parent.parent / "output" / "qnfs_per_season_playoffs.csv"

MIN_NF_SHOTS_PER_GAME = 3          # metric definition (NOT a qualifying floor)
QUALITY_THRESHOLD = 0.0
# No MIN_GP_PER_SEASON — emit every goalie; threshold on Streamlit.

print("[compute_qnfs_per_season_playoffs] playoff QNFS%, no goalie floor")
df = pd.read_csv(SHOT_FILE, low_memory=False)

df = df[df["game_type"] == "playoff"]
df = df[df["period_type"] == "REG"]
df = df[df["situation_code"] == 1551]
df = df[df["event_type"].isin(["shot-on-goal", "goal", "missed-shot"])]
df = df[df["goalie_id"].notna()].copy()
df["goalie_id"] = df["goalie_id"].astype(int)
df["season"] = df["season"].astype(str)
print(f"  playoff 5v5-ES Fenwick rows: {len(df):,} | seasons: {sorted(df['season'].unique())}")


def tag_zone(x, y):
    in_cnfi = (x >= 74) & (x <= 89) & (y.abs() <= 9)
    in_mnfi = (x >= 55) & (x < 74) & (y.abs() <= 15)
    z = pd.Series("OTHER", index=x.index)
    z[in_mnfi] = "MNFI"
    z[in_cnfi] = "CNFI"
    return z


df["zone"] = tag_zone(df["x_coord_norm"], df["y_coord_norm"])
nf = df[df["zone"].isin(["CNFI", "MNFI"])].copy()

# Per-season per-zone league Fenwick goal rates (playoff).
zr = (nf.groupby(["season", "zone"]).agg(faced=("is_goal", "size"), goals=("is_goal", "sum"))
      .assign(rate=lambda d: d["goals"] / d["faced"]).reset_index())
rates = zr.pivot(index="season", columns="zone", values="rate").reset_index()
rates.columns.name = None
rates = rates.rename(columns={"CNFI": "rate_CNFI", "MNFI": "rate_MNFI"})
for c in ("rate_CNFI", "rate_MNFI"):
    if c not in rates.columns:
        rates[c] = 0.0

# Per-game per-goalie net-front faced/goals/GSAx.
gg = (nf.groupby(["season", "game_id", "goalie_id", "zone"])
      .agg(faced=("is_goal", "size"), goals=("is_goal", "sum")).reset_index())
gg = gg.pivot_table(index=["season", "game_id", "goalie_id"], columns="zone",
                    values=["faced", "goals"], fill_value=0).reset_index()
gg.columns = ["_".join([c for c in col if c]).strip("_") for col in gg.columns]
for col in ["faced_CNFI", "faced_MNFI", "goals_CNFI", "goals_MNFI"]:
    if col not in gg.columns:
        gg[col] = 0
gg = gg.merge(rates, on="season", how="left")
gg["xG_NF"] = gg["faced_CNFI"] * gg["rate_CNFI"] + gg["faced_MNFI"] * gg["rate_MNFI"]
gg["goals_NF"] = gg["goals_CNFI"] + gg["goals_MNFI"]
gg["GSAx_NF"] = gg["xG_NF"] - gg["goals_NF"]
gg["NF_shots"] = gg["faced_CNFI"] + gg["faced_MNFI"]
gg = gg[gg["NF_shots"] >= MIN_NF_SHOTS_PER_GAME].copy()
gg["is_quality"] = (gg["GSAx_NF"] >= QUALITY_THRESHOLD).astype(int)


def aggregate(group_cols, season_label=None):
    agg = (gg.groupby(group_cols)
           .agg(GP=("game_id", "size"), quality_games=("is_quality", "sum"),
                NF_shots=("NF_shots", "sum")).reset_index())
    if season_label is not None:
        agg["season"] = season_label
    return agg


per_season = aggregate(["goalie_id", "season"])
pooled = aggregate(["goalie_id"], season_label="all_playoffs")
out = pd.concat([per_season, pooled], ignore_index=True)

out["QNFS_pct"] = out["quality_games"] / out["GP"]
lo, hi = proportion_confint(out["quality_games"].astype(int), out["GP"].astype(int),
                            alpha=0.05, method="wilson")
out["QNFS_lo"], out["QNFS_hi"] = lo, hi
for c in ("QNFS_pct", "QNFS_lo", "QNFS_hi"):
    out[c] = out[c] * 100.0

names = pd.read_csv(NAMES_FILE)[["player_id", "player_name"]].rename(
    columns={"player_id": "goalie_id", "player_name": "goalie_name"})
out = out.merge(names, on="goalie_id", how="left")
out = out[["goalie_id", "goalie_name", "season", "GP", "quality_games",
           "QNFS_pct", "QNFS_lo", "QNFS_hi", "NF_shots"]]
out = out.sort_values(["season", "QNFS_pct"], ascending=[True, False]).reset_index(drop=True)
out.to_csv(OUT_FILE, index=False)
print(f"\nWrote {OUT_FILE} — {len(out)} rows, {out['goalie_id'].nunique()} goalies")
print("\n=== all_playoffs top 6 by QNFS% (GP>=10) ===")
ap = out[(out["season"] == "all_playoffs") & (out["GP"] >= 10)]
print(ap.nlargest(6, "QNFS_pct")[["goalie_name", "GP", "quality_games",
                                  "QNFS_pct", "NF_shots"]].round(1).to_string(index=False))
