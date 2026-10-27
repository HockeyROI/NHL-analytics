"""build_rapm_exports.py -- publish the three dashboard RAPM families to rapm/output/.

Reads the final RAPM study outputs READ-ONLY from Zones/2026-09/ and writes a
single CI-portable export the Streamlit app loads:

  rapm/output/player_rapm_5v5.csv   one row per skater in the fit (everyone who
                                    played 5v5 in 2023-24..2025-26; low-minute
                                    players are shrunk toward 0 by the ridge
                                    prior and carry wide intervals). The app's
                                    sliders decide who is shown and ranked.

Three published families (special-teams RAPM is NOT published -- see README):
  NFI RAPM  <- Zones/2026-09/rapm/doorstep   (NFI zone = CNFI u MNFI Fenwick)
  ITI       <- Zones/2026-09/iti/player_iti.csv (already net/A/S + CIs)
  cG RAPM   <- Zones/2026-09/rapm/cg          (on-ice Chaos Created Goals)

For NFI RAPM and cG RAPM the net 95% interval is summed A+S WITHIN each of the
200 game bootstrap resamples (not by adding the two separate intervals), exactly
as iti/finalize_iti.py does for ITI. Sign convention: Attack (A) is the offence
coefficient; Suppress (S) is the negated defence coefficient, so POSITIVE = BETTER
on every column. clears_zero = the 95% interval excludes 0.

Nothing in Zones/2026-09/ is modified; no existing data file is changed.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STUDY = os.path.join(REPO, "Zones", "2026-09", "rapm")
ITI = os.path.join(REPO, "Zones", "2026-09", "iti")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

SEASONS_LABEL = "2023-24 to 2025-26"
QUALIFY_MIN = 500.0


def _rapm_family(folder: str, prefix: str, players: pd.DataFrame) -> pd.DataFrame:
    """Net/A/S for one RAPM fit, with per-resample bootstrap 95% intervals.

    A = offence coefficient, S = -(defence coefficient), net = A + S, all oriented
    positive = better. Mirrors iti/finalize_iti.py so ITI and these stay identical
    in method."""
    n = len(players)
    beta = np.load(os.path.join(folder, "beta.npy"))
    boot = np.load(os.path.join(folder, "boot_betas.npy"))
    pr = pd.read_csv(os.path.join(folder, "player_rapm_all.csv"))
    # column order of the fit == matrix/players.csv order; validate before trusting
    assert (pr["player_id"].to_numpy() == players["player_id"].to_numpy()).all(), \
        f"{folder}: player order mismatch vs matrix/players.csv"
    assert np.allclose(pr["rapm_off"].to_numpy(), beta[1:n + 1]) and \
        np.allclose(pr["rapm_def"].to_numpy(), beta[n + 1:2 * n + 1]), \
        f"{folder}: beta slices do not match player_rapm_all"
    a = beta[1:n + 1]
    s = -beta[n + 1:2 * n + 1]
    Ba = boot[:, 1:n + 1]
    Bs = -boot[:, n + 1:2 * n + 1]
    Bn = Ba + Bs
    out = pd.DataFrame({"player_id": players["player_id"].to_numpy()})
    for lab, point, M in ((prefix, a + s, Bn),
                          (f"{prefix}_a", a, Ba),
                          (f"{prefix}_s", s, Bs)):
        out[lab] = np.round(point, 4)
        lo = np.percentile(M, 2.5, axis=0)
        hi = np.percentile(M, 97.5, axis=0)
        out[f"{lab}_lo"] = np.round(lo, 4)
        out[f"{lab}_hi"] = np.round(hi, 4)
        out[f"{lab}_clears_zero"] = (lo > 0) | (hi < 0)
    return out


def _team_by_5v5_min() -> pd.DataFrame:
    """Each player's team = the one he logged the most pooled 5v5 minutes with,
    over the three study seasons. Built from the study stints (home_id/away_id +
    the five skater columns per side), mapped to abbrev from the shot parquet."""
    st = pd.read_parquet(os.path.join(STUDY, "stints.parquet"),
                         columns=["duration", "home_id", "away_id",
                                  "h0", "h1", "h2", "h3", "h4",
                                  "a0", "a1", "a2", "a3", "a4"])
    rows = []
    for side, tid, cols in (("h", "home_id", ["h0", "h1", "h2", "h3", "h4"]),
                            ("a", "away_id", ["a0", "a1", "a2", "a3", "a4"])):
        m = st.melt(id_vars=["duration", tid], value_vars=cols, value_name="pid")
        m = m.rename(columns={tid: "team_id"})
        rows.append(m[["pid", "team_id", "duration"]])
    allm = pd.concat(rows, ignore_index=True)
    secs = allm.groupby(["pid", "team_id"])["duration"].sum().reset_index()
    top = (secs.sort_values("duration")
           .drop_duplicates("pid", keep="last")[["pid", "team_id"]])
    # team_id -> abbrev from a committed shot parquet (has both id and abbrev)
    sp = pd.read_parquet(
        os.path.join(REPO, "Data", "shot_events_by_season", "20252026.parquet"),
        columns=["home_team_id", "home_team_abbrev", "away_team_id", "away_team_abbrev"])
    amap = pd.concat([
        sp[["home_team_id", "home_team_abbrev"]].rename(
            columns={"home_team_id": "team_id", "home_team_abbrev": "abbrev"}),
        sp[["away_team_id", "away_team_abbrev"]].rename(
            columns={"away_team_id": "team_id", "away_team_abbrev": "abbrev"}),
    ]).dropna().drop_duplicates("team_id")
    top = top.merge(amap, on="team_id", how="left")
    return top.rename(columns={"pid": "player_id", "abbrev": "team"})[["player_id", "team"]]


def main():
    players = pd.read_csv(os.path.join(STUDY, "matrix", "players.csv"))
    n = len(players)
    print(f"[rapm] {n} skaters in the fit (matrix/players.csv)")

    # --- the three families -------------------------------------------------
    nfi = _rapm_family(os.path.join(STUDY, "doorstep"), "nfi_rapm", players)
    cg = _rapm_family(os.path.join(STUDY, "cg"), "cg_rapm", players)
    iti = (pd.read_csv(os.path.join(ITI, "player_iti_all.csv"))   # every skater, net/A/S + CIs
           .drop(columns=["anchor", "source", "unit"], errors="ignore"))
    print(f"[rapm] NFI/cG families computed for {n}; ITI rows (>=500 min): {len(iti)}")

    # --- pooled 5v5 TOI + seasons (from the doorstep all-players table) ------
    base = pd.read_csv(os.path.join(STUDY, "doorstep", "player_rapm_all.csv"))[
        ["player_id", "name", "position", "toi_5v5_min"]].copy()
    base["position"] = np.where(base["position"] == "D", "D", "F")
    base["seasons"] = SEASONS_LABEL

    # --- team (most pooled 5v5 minutes) -------------------------------------
    team = _team_by_5v5_min()

    # --- collinearity flag (canonical: iti/collinearity_flags.csv) ----------
    flags = pd.read_csv(os.path.join(ITI, "collinearity_flags.csv"))[
        ["player_id", "top_linemate", "share_with_top_linemate", "collinearity_flag"]]

    # --- assemble -----------------------------------------------------------
    df = (base
          .merge(team, on="player_id", how="left")
          .merge(nfi, on="player_id", how="left")
          .merge(cg, on="player_id", how="left")
          .merge(iti.drop(columns=["name", "position", "toi_5v5_min", "seasons"]),
                 on="player_id", how="left")
          .merge(flags, on="player_id", how="left"))
    df["collinearity_flag"] = df["collinearity_flag"].fillna(False).astype(bool)

    # every skater in the fit (was >= 500 pooled 5v5 min)
    q = df.copy()
    q = q.sort_values("nfi_rapm", ascending=False).reset_index(drop=True)

    col_order = ["player_id", "name", "position", "team", "toi_5v5_min", "seasons"]
    for fam in ("nfi_rapm", "iti", "cg_rapm"):
        for suf in ("", "_a", "_s"):
            b = f"{fam}{suf}"
            col_order += [b, f"{b}_lo", f"{b}_hi", f"{b}_clears_zero"]
    col_order += ["collinearity_flag", "top_linemate", "share_with_top_linemate"]
    q = q[[c for c in col_order if c in q.columns]]

    out_path = os.path.join(OUT, "player_rapm_5v5.csv")
    q.to_csv(out_path, index=False)

    nf = int(q["collinearity_flag"].sum())
    print(f"[rapm] wrote {len(q)} qualified skaters -> {out_path}")
    print(f"[rapm]   forwards {int((q.position=='F').sum())}, "
          f"defence {int((q.position=='D').sum())}")
    print(f"[rapm]   collinearity-flagged {nf} "
          f"(F {int(((q.position=='F')&q.collinearity_flag).sum())}, "
          f"D {int(((q.position=='D')&q.collinearity_flag).sum())})")
    _spot(q)


def _spot(q: pd.DataFrame):
    print("\n[rapm] spot-check (net values, + = better):")
    cols = ["name", "team", "toi_5v5_min", "nfi_rapm", "nfi_rapm_lo", "nfi_rapm_hi",
            "iti", "cg_rapm", "collinearity_flag", "top_linemate"]
    for nm in ("McDavid", "Makar", "Hyman", "Crosby"):
        r = q[q["name"].str.contains(nm, case=False, na=False)]
        if len(r):
            print(r[cols].to_string(index=False))


if __name__ == "__main__":
    main()
