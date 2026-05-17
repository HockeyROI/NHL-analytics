# NFI/scripts/diagnostics/net_front_sample_size_diagnostic.py
#
# Purpose: Before building NFI Net-Front Save Consistency, check whether
# per-game net-front shots-against is a viable unit of analysis,
# or whether we need rolling windows.
#
# NOT part of canonical NFI pipeline. Diagnostic only.

import pandas as pd
from pathlib import Path

print("[diagnostic] NFI/scripts/diagnostics/net_front_sample_size_diagnostic.py")

# ---- CONFIG ----
SHOT_FILE = Path("/Users/ashgarg/Documents/HockeyROI/Data/nhl_shot_events.v2.csv")
SEASONS = [20222023, 20232024, 20242025]   # adjust if your season key format differs
MIN_GAMES = 30

# Goalie IDs to profile. Replace with real NHL player IDs.
# (You probably have a player_id -> name map somewhere; fine to add later.)
SAMPLE_GOALIE_IDS = {
    8479406: "Filip Gustavsson",
    8476945: "Connor Hellebuyck",
    8478048: "Igor Shesterkin",
    8478009: "Ilya Sorokin",
    8479979: "Jake Oettinger",
}

# ---- DIAGNOSTIC LOAD ----
print(f"Reading: {SHOT_FILE.name}")
df = pd.read_csv(SHOT_FILE, low_memory=False)
print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# STOP if expected columns are missing — don't guess
required = {
    "season", "game_id", "game_type", "period_type",
    "situation_code", "event_type",
    "goalie_id", "x_coord_norm", "y_coord_norm",
}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(
        f"Missing expected columns: {missing}. "
        "Check schema of nhl_shot_events.v2.csv before proceeding."
    )

# ---- FILTER TO REG SEASON, 5v5 ES (strict / Variant F), FENWICK ----
before = len(df)
df = df[df["season"].isin(SEASONS)]
df = df[df["game_type"] == "regular"]               # regular season
df = df[df["period_type"] == "REG"]                 # exclude OT/SO
df = df[df["situation_code"] == 1551]               # strict 5v5
df = df[df["event_type"].isin(["shot-on-goal", "goal", "missed-shot"])]   # Fenwick = no blocks
df = df[df["goalie_id"].notna()]                    # need a goalie on the shot
print(f"After filters (reg, 5v5 ES, Fenwick, has goalie): "
      f"{len(df):,} rows ({before-len(df):,} dropped)")

# ---- DEFINE NET-FRONT ZONE ----
# Net-front = CNFI ∪ MNFI, matching the canonical zone classifier in
# NFI/scripts/02_zones_and_rebound_confirm.py:35 and the HD proxy in
# NFI/scripts/04_corsi_nfi_variants.py:306. FNFI is excluded (it's high slot,
# not net-front).
#   CNFI: 74 <= x <= 89,  |y| <= 9   (doorstep)
#   MNFI: 55 <= x <  74,  |y| <= 15  (slot)
def is_net_front(x_norm, y_norm):
    in_cnfi = (x_norm >= 74) & (x_norm <= 89) & y_norm.between(-9, 9)
    in_mnfi = (x_norm >= 55) & (x_norm <  74) & y_norm.between(-15, 15)
    return in_cnfi | in_mnfi

df["net_front"] = is_net_front(df["x_coord_norm"], df["y_coord_norm"])
nf = df[df["net_front"]].copy()
print(f"Net-front shots: {len(nf):,} ({len(nf)/len(df):.1%} of all shots)")

# ---- PER-GAME NET-FRONT SHOTS-AGAINST PER GOALIE ----
per_game = (
    nf.groupby(["goalie_id", "season", "game_id"])
      .size()
      .reset_index(name="nf_shots_against")
)

# League-wide distribution
print("\n=== LEAGUE-WIDE: NET-FRONT SHOTS AGAINST PER GAME-GOALIE ===")
print(per_game["nf_shots_against"].describe(percentiles=[.1, .25, .5, .75, .9]))

thin = (per_game["nf_shots_against"] < 3).mean()
print(f"\nShare of game-goalie rows with <3 net-front shots: {thin:.1%}")
print("If >30-40%, per-game net-front GSAx will be too noisy — "
      "plan for rolling-window (5- or 10-game) units instead.\n")

# ---- SAMPLE GOALIE PROFILES ----
print("=== SAMPLE GOALIES: NF SHOTS-AGAINST PER GAME ===")
for gid, name in SAMPLE_GOALIE_IDS.items():
    sub = per_game[per_game["goalie_id"] == gid]
    if len(sub) < MIN_GAMES:
        print(f"  {name} ({gid}): only {len(sub)} games in window — skipping")
        continue
    print(
        f"  {name} ({gid}): n={len(sub)} games | "
        f"mean={sub['nf_shots_against'].mean():.2f} | "
        f"median={sub['nf_shots_against'].median():.0f} | "
        f"p10={sub['nf_shots_against'].quantile(.1):.0f} | "
        f"p90={sub['nf_shots_against'].quantile(.9):.0f}"
    )
