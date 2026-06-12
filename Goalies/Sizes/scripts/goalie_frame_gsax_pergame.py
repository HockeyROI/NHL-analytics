"""
GSAx-per-game frame analysis for the 2022-23..2024-25 MoneyPuck dataset.

Loads goalie_frame_gsax_2023now.csv, reconstructs total games from the
MoneyPuck source (same 25-starts filter, summed over the 3 qualifying seasons),
computes GSAx per game = total_gsax / total_games, and runs the same three
correlations (height, weight, weight/height ratio) against it.
"""

import os
import requests

from goalie_frame_2023now_analysis import (
    normalize_name, load_moneypuck, run_three, OUT_DIR, MIN_STARTS,
)
import pandas as pd

CSV = os.path.join(OUT_DIR, "goalie_frame_gsax_2023now.csv")
OUT = os.path.join(OUT_DIR, "goalie_frame_gsax_pergame_2023now.csv")


def main():
    df = pd.read_csv(CSV)
    df["name"] = df["display_name"].map(normalize_name)

    # Reconstruct total games from MoneyPuck (cache 2022-23/2023-24 + live 2024-25).
    # All goalies here have qualifying_seasons == 3, so every one of the 3 target
    # seasons cleared the 25-start filter -> summing those starts gives total games.
    session = requests.Session()
    mp = load_moneypuck(session)
    mp = mp[mp["games_started"].fillna(0) >= MIN_STARTS]
    games = mp.groupby("name")["games_started"].sum()

    df["total_games"] = df["name"].map(games)
    missing = df[df["total_games"].isna()]
    if len(missing):
        print("WARNING: no games found for:", list(missing["display_name"]))
    df = df.dropna(subset=["total_games"]).copy()

    df["gsax_per_game"] = df["total_gsax"] / df["total_games"]
    df = df.sort_values("gsax_per_game", ascending=False).reset_index(drop=True)

    run_three(df, "gsax_per_game", "display_name",
              ["total_gsax", "total_games"], "MONEYPUCK GSAx/GAME (2022-23 to 2024-25)")

    out_cols = ["display_name", "height_in", "weight_lbs", "weight_height_ratio",
                "total_gsax", "total_games", "gsax_per_game",
                "qualifying_seasons", "size_source"]
    df[out_cols].to_csv(OUT, index=False)
    print(f"\n  Saved -> {OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
