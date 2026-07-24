"""Playoff QS-GSAx — playoff companion to compute_qs_gsax.py.

Same all-shot 5v5-ES regulation methodology (per-game GSAx = xGoal − goal;
quality game = GSAx >= 0; per-game >=10 shots to count), but on PLAYOFF games
(isPlayoffGame == 1). MoneyPuck retired (2026): reads our shot events + our xG
via the MP-schema shim (NFI/scripts/_data_sources.load_shots_mp_schema).

NO goalie-level qualifying floor (regular uses GP>=25) — every goalie with a
qualifying game appears, per playoff season PLUS an `all_playoffs` pooled row.
GP is kept so the Streamlit slider thresholds at display. The per-game >=10
shots rule is the metric's definition and is retained.

Output: NFI/goalie_consistency/output/qs_gsax_per_season_playoffs.csv
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _data_sources as _ds

NAMES_FILE = Path("/Users/ashgarg/Documents/HockeyROI/NFI/output/player_positions.csv")
OUT_DIR = Path("/Users/ashgarg/Documents/HockeyROI/NFI/goalie_consistency/output")
OUT_FILE = OUT_DIR / "qs_gsax_per_season_playoffs.csv"

# int season (shim) -> the string label the app + output have always used.
SEASON_LABEL = {20222023: "20222023", 20232024: "20232024",
                20242025: "20242025", 20252026: "20252026"}
MIN_SHOTS_PER_GAME = 10   # metric definition (NOT a goalie qualifying floor)


def wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (center - margin) / denom


print("[compute_qs_gsax_playoffs] playoff QS-GSAx, no goalie floor")
allshots = _ds.load_shots_mp_schema()
df = allshots[
    allshots["season"].isin(SEASON_LABEL)
    & (allshots["homeSkatersOnIce"] == 5) & (allshots["awaySkatersOnIce"] == 5)
    & (allshots["isPlayoffGame"] == 1) & (allshots["period"] <= 3)
    & (allshots["goalieIdForShot"].notna())
].copy()
df["season"] = df["season"].map(SEASON_LABEL)
pg = (df.groupby(["game_id", "goalieIdForShot", "season"])
      .agg(shots_faced=("xGoal", "size"), xG=("xGoal", "sum"), goals=("goal", "sum")).reset_index()
      .rename(columns={"goalieIdForShot": "goalie_id"}))
pg["GSAx"] = pg["xG"] - pg["goals"]
per_game = pg[pg["shots_faced"] >= MIN_SHOTS_PER_GAME].copy()
per_game["goalie_id"] = per_game["goalie_id"].astype(int)
per_game["quality"] = (per_game["GSAx"] >= 0).astype(int)
for s in sorted(per_game["season"].unique()):
    print(f"  season {s}: {int((per_game['season']==s).sum())} qualifying game-goalie rows")

_names = pd.read_csv(NAMES_FILE)
_canon = dict(zip(_names["player_id"].astype(int), _names["player_name"].astype(str)))
canon = {g: _canon.get(g, str(g)) for g in per_game["goalie_id"].unique()}


def agg(group_cols, label=None):
    a = (per_game.groupby(group_cols)
         .agg(GP=("game_id", "size"), quality_games=("quality", "sum"),
              GSAx_total=("GSAx", "sum")).reset_index())
    if label is not None:
        a["season"] = label
    return a


out = pd.concat([agg(["goalie_id", "season"]), agg(["goalie_id"], "all_playoffs")],
                ignore_index=True)
out["goalie_name"] = out["goalie_id"].map(canon)
out["QS_GSAx_pct"] = out["quality_games"] / out["GP"] * 100
out["QS_GSAx_lo"] = out.apply(lambda r: wilson_lower(r["quality_games"], r["GP"]) * 100, axis=1)
out = out[["goalie_id", "goalie_name", "season", "GP", "quality_games",
           "GSAx_total", "QS_GSAx_pct", "QS_GSAx_lo"]]
out = out.sort_values(["season", "QS_GSAx_pct"], ascending=[True, False]).reset_index(drop=True)
out.to_csv(OUT_FILE, index=False)
print(f"\nWrote {OUT_FILE} — {len(out)} rows, {out['goalie_id'].nunique()} goalies")
ap = out[(out["season"] == "all_playoffs") & (out["GP"] >= 15)]
print("\n=== all_playoffs top 6 by QS-GSAx% (GP>=15) ===")
print(ap.nlargest(6, "QS_GSAx_pct")[["goalie_name", "GP", "quality_games",
                                     "QS_GSAx_pct", "GSAx_total"]].round(1).to_string(index=False))
