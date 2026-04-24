#!/usr/bin/env python3
"""
Individual-shooter vs on-ice forward offensive metrics — team-level
prediction of standings points.

Methodology:
  Individual shooter (1-3): for each team-season, count Corsi attempts in
    tight CNFI (74-89 |y|<=9) / MNFI (55-73 |y|<=15) where shooter is a
    forward. Per game.
  On-ice (4-6): for each forward, compute personal on-ice CNFI/MNFI Fenwick
    shots-for per 60 ES TOI (built via fresh shift-shot join — not in any
    existing file at tight-zone granularity). Then aggregate to team-season
    as the simple mean across the team's qualifying forwards (≥500 ES min,
    primary team-season by max TOI).
  Suppression: same player-aggregated approach using P2_defensive_forwards.

Two-variable models (7-8) regress points on offense + suppression.

Output: NFI/output/individual_vs_onice_comparison.csv
"""
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
SHOT_CSV = f"{ROOT}/Data/nhl_shot_events.csv"
SHIFT_CSV = f"{ROOT}/NFI/Geometry_post/Data/shift_data.csv"
GAME_CSV = f"{ROOT}/Data/game_ids.csv"

SEASONS = {"20212022","20222023","20232024","20242025","20252026"}
MIN_F_TOI = 500.0  # match existing pillar threshold

# ---- Lookups ----
print("Loading lookups ...")
pos_df = pd.read_csv(f"{OUT}/player_positions.csv", dtype={"player_id":int})
pos_grp = dict(zip(pos_df["player_id"], pos_df["pos_group"]))
team_metrics = pd.read_csv(f"{OUT}/team_level_all_metrics.csv", dtype={"season":str})
toi_df = pd.read_csv(f"{OUT}/player_toi.csv", dtype={"player_id":int})
toi_es_min = dict(zip(toi_df["player_id"], toi_df["toi_ES_sec"]/60.0))
games = pd.read_csv(GAME_CSV, dtype={"game_id":int,"season":str})
games = games[games["game_type"]=="regular"]
game_season_map = dict(zip(games["game_id"], games["season"]))

# ---- Load shots, classify zones ----
print("Loading shots ...")
sh = pd.read_csv(SHOT_CSV,
                 usecols=["game_id","season","period","situation_code","time_secs",
                          "event_type","shooting_team_abbrev","shooter_player_id",
                          "x_coord_norm","y_coord_norm",
                          "home_team_id","shooting_team_id",
                          "home_team_abbrev","away_team_abbrev"],
                 dtype={"season":str,"situation_code":str})
sh = sh[sh["season"].isin(SEASONS)]
sh = sh[sh["period"].between(1,3)]
sh = sh[sh["situation_code"].astype(str)=="1551"]
# Full Corsi for individual-shooter side (matches existing horse race);
# Fenwick for on-ice side (matches P2 methodology for orthogonal comparison)
sh = sh[sh["event_type"].isin(["shot-on-goal","missed-shot","blocked-shot","goal"])]
sh = sh.dropna(subset=["x_coord_norm","y_coord_norm","shooter_player_id"])
blk = sh["event_type"]=="blocked-shot"
sh.loc[blk,"x_coord_norm"] = sh.loc[blk,"x_coord_norm"].abs()
sh.loc[blk,"y_coord_norm"] = sh.loc[blk,"y_coord_norm"].abs()
sh["abs_y"] = sh["y_coord_norm"].abs()
sh["shooter_pos"] = sh["shooter_player_id"].astype(int).map(pos_grp)
sh["zone"] = np.where(
    (sh["x_coord_norm"].between(74,89)) & (sh["abs_y"]<=9), "CNFI",
    np.where((sh["x_coord_norm"].between(55,73)) & (sh["abs_y"]<=15), "MNFI", None))
sh = sh[sh["zone"].notna()]
sh["abs_time"] = sh["time_secs"].astype(int) + (sh["period"].astype(int)-1)*1200

# ---- (1-3) F-shooter team-season counts (Corsi) ----
fsh = sh[sh["shooter_pos"]=="F"]
agg = fsh.groupby(["season","shooting_team_abbrev","zone"]).size().reset_index(name="n")\
        .rename(columns={"shooting_team_abbrev":"team"})
piv = agg.pivot_table(index=["season","team"], columns="zone", values="n", fill_value=0).reset_index()
piv = piv.rename(columns={"CNFI":"F_CNFI_n","MNFI":"F_MNFI_n"})
piv = piv.merge(team_metrics[["season","team","gp","points"]], on=["season","team"])
piv["F_CNFI_per_gp"] = piv["F_CNFI_n"] / piv["gp"]
piv["F_MNFI_per_gp"] = piv["F_MNFI_n"] / piv["gp"]
piv["F_CNFI_MNFI_per_gp"] = piv["F_CNFI_per_gp"] + piv["F_MNFI_per_gp"]
print(f"  team-seasons (individual-shooter): {len(piv)}")

# ---- (4-6) On-ice FOR — fresh shift-shot join (Fenwick, tight zones) ----
print("On-ice FOR aggregation: fenwick subset + shift join ...")
fen = sh[sh["event_type"].isin(["shot-on-goal","missed-shot","goal"])].copy()
fen["_shoot_home"] = fen["shooting_team_id"]==fen["home_team_id"]
fen["def_team"] = np.where(fen["_shoot_home"], fen["away_team_abbrev"], fen["home_team_abbrev"])
print(f"  Fenwick CNFI/MNFI shots: {len(fen):,}")

# Per-(forward, zone) on-ice FOR counts via shot-shift join
shots_by_game = dict(tuple(fen.groupby("game_id")))
valid_gids = set(shots_by_game.keys())

print("Loading shifts ...")
parts = []
for ch in pd.read_csv(SHIFT_CSV,
                      usecols=["game_id","player_id","period","team_abbrev",
                                "abs_start_secs","abs_end_secs"],
                      chunksize=500_000):
    ch = ch.dropna()
    ch["game_id"] = ch["game_id"].astype(int); ch["player_id"] = ch["player_id"].astype(int)
    ch["period"] = ch["period"].astype(int)
    ch["abs_start_secs"] = ch["abs_start_secs"].astype(int)
    ch["abs_end_secs"]   = ch["abs_end_secs"].astype(int)
    ch = ch[ch["game_id"].isin(valid_gids) & ch["period"].between(1,3)]
    if len(ch): parts.append(ch)
shifts = pd.concat(parts, ignore_index=True); del parts
print(f"  shifts: {len(shifts):,}")
shifts_by_game = dict(tuple(shifts.groupby("game_id")))

# Per-(player, season, team) ES TOI for primary-team mapping
ts_pid = defaultdict(int)
for (gid, team), grp in shifts.groupby(["game_id","team_abbrev"]):
    season = game_season_map.get(gid)
    if season is None or season not in SEASONS: continue
    durs = (grp["abs_end_secs"] - grp["abs_start_secs"]).clip(lower=0)
    pids = grp["player_id"].values
    for pid, dur in zip(pids, durs.values):
        ts_pid[(season, team, int(pid))] += int(dur)

primary = {}
for (season, team, pid), sec in ts_pid.items():
    key = (pid, season)
    if key not in primary or primary[key][1] < sec:
        primary[key] = (team, sec)

# Now run the join: per-(forward, zone) on-ice FOR
print("Per-game shift-shot join (forward on-ice FOR) ...")
f_for = defaultdict(int)   # (pid, zone) -> count
n_games = 0; t0 = time.time()
for gid, gshots in shots_by_game.items():
    n_games += 1
    if n_games % 1000 == 0: print(f"  {n_games} games ({time.time()-t0:.1f}s)")
    gshifts = shifts_by_game.get(gid)
    if gshifts is None: continue
    shifts_by_team = {}
    for tab, tsh in gshifts.groupby("team_abbrev"):
        st = tsh["abs_start_secs"].values.astype(np.int32)
        en = tsh["abs_end_secs"].values.astype(np.int32)
        pids = tsh["player_id"].values.astype(np.int64)
        order = np.argsort(st)
        shifts_by_team[tab] = (st[order], en[order], pids[order])
    for r in gshots.itertuples(index=False):
        t = int(r.abs_time)
        zone = r.zone
        shoot_ab = r.shooting_team_abbrev
        if shoot_ab not in shifts_by_team: continue
        st, en, pids = shifts_by_team[shoot_ab]
        idx = np.searchsorted(st, t, side="right")
        if idx == 0: continue
        on = en[:idx] > t
        for pid in pids[:idx][on]:
            if pos_grp.get(int(pid)) == "F":
                f_for[(int(pid), zone)] += 1
print(f"Aggregation done in {time.time()-t0:.1f}s")

# Build per-forward on-ice for/60
fwd_rows = []
for pid in {p for (p, _) in f_for.keys()}:
    toi = toi_es_min.get(pid, 0.0)
    if toi < MIN_F_TOI: continue
    cnfi = f_for.get((pid, "CNFI"), 0)
    mnfi = f_for.get((pid, "MNFI"), 0)
    fwd_rows.append({
        "player_id": pid,
        "es_toi_min": toi,
        "F_CNFI_SF": cnfi,
        "F_MNFI_SF": mnfi,
        "F_CNFI_SF_per60": cnfi/toi*60.0,
        "F_MNFI_SF_per60": mnfi/toi*60.0,
        "F_CNFI_MNFI_SF_per60": (cnfi+mnfi)/toi*60.0,
    })
fwd_df = pd.DataFrame(fwd_rows)
print(f"  qualifying forwards: {len(fwd_df)}")

# Aggregate to team-season as simple mean across qualifying forwards (primary team)
def team_avg(player_df, col):
    rows = []
    for _, r in player_df.iterrows():
        pid = int(r["player_id"]); score = r[col]
        for season in SEASONS:
            if (pid, season) in primary:
                team, _ = primary[(pid, season)]
                rows.append({"season":season, "team":team, "score":score})
    return pd.DataFrame(rows).groupby(["season","team"])["score"].mean().reset_index()\
            .rename(columns={"score":col+"_team_avg"})

ta_cnfi = team_avg(fwd_df, "F_CNFI_SF_per60").rename(columns={"F_CNFI_SF_per60_team_avg":"F_oniceFOR_CNFI_per60"})
ta_mnfi = team_avg(fwd_df, "F_MNFI_SF_per60").rename(columns={"F_MNFI_SF_per60_team_avg":"F_oniceFOR_MNFI_per60"})
ta_comb = team_avg(fwd_df, "F_CNFI_MNFI_SF_per60").rename(columns={"F_CNFI_MNFI_SF_per60_team_avg":"F_oniceFOR_CNFI_MNFI_per60"})

# Merge with piv (team-season frame) — also bring in suppression
p2 = pd.read_csv(f"{OUT}/P2_defensive_forwards.csv")
p2["F_CNFI_MNFI_SA_per60"] = p2["CNFI_SA_per60"] + p2["MNFI_SA_per60"]
sup_rows = []
for _, r in p2.iterrows():
    pid = int(r["player_id"]); score = r["F_CNFI_MNFI_SA_per60"]
    for season in SEASONS:
        if (pid, season) in primary:
            team, _ = primary[(pid, season)]
            sup_rows.append({"season":season, "team":team, "score":score})
sup_team = pd.DataFrame(sup_rows).groupby(["season","team"])["score"].mean().reset_index()\
            .rename(columns={"score":"F_oniceAGAINST_CNFI_MNFI_per60"})

m = piv.merge(ta_cnfi, on=["season","team"]).merge(ta_mnfi, on=["season","team"])\
       .merge(ta_comb, on=["season","team"]).merge(sup_team, on=["season","team"])
print(f"  final n in horse race: {len(m)}")

# ---- OLS utility ----
def ols(X, y):
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    yhat = X @ beta; resid = y - yhat
    sigma2 = (resid**2).sum() / (n - k)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t = beta / se
    p_ = 2 * (1 - stats.t.cdf(np.abs(t), df=n-k))
    ss_res = (resid**2).sum(); ss_tot = ((y - y.mean())**2).sum()
    return beta, se, t, p_, 1 - ss_res/ss_tot

# ---- 6 univariate correlations ----
y = m["points"].values
n = len(y)
specs = [
    ("01", "Individual F CNFI shots taken /gp",      "F_CNFI_per_gp"),
    ("02", "Individual F MNFI shots taken /gp",      "F_MNFI_per_gp"),
    ("03", "Individual F CNFI+MNFI shots taken /gp", "F_CNFI_MNFI_per_gp"),
    ("04", "On-ice F CNFI SF /60 (team avg)",        "F_oniceFOR_CNFI_per60"),
    ("05", "On-ice F MNFI SF /60 (team avg)",        "F_oniceFOR_MNFI_per60"),
    ("06", "On-ice F CNFI+MNFI SF /60 (team avg)",   "F_oniceFOR_CNFI_MNFI_per60"),
]

results = []
for mid, label, col in specs:
    x = m[col].values
    r, p_ = stats.pearsonr(x, y)
    results.append({"id":mid, "metric":label, "type":"individual" if mid in ("01","02","03") else "on-ice",
                     "n":n, "pearson_r":round(r,4), "R2":round(r*r,4), "p":p_,
                     "model_type":"univariate"})

# ---- Two-variable models ----
# 7. Individual CNFI+MNFI + suppression
x_ind = m["F_CNFI_MNFI_per_gp"].values
x_oni = m["F_oniceFOR_CNFI_MNFI_per60"].values
x_sup = m["F_oniceAGAINST_CNFI_MNFI_per60"].values
X7 = np.column_stack([np.ones(n), x_ind, x_sup])
b7, se7, t7, p7_, r2_7 = ols(X7, y)
X8 = np.column_stack([np.ones(n), x_oni, x_sup])
b8, se8, t8, p8_, r2_8 = ols(X8, y)

results.append({"id":"07", "metric":"Individual CNFI+MNFI + suppression",
                 "type":"individual+sup","n":n,
                 "pearson_r":np.nan, "R2":round(r2_7,4),
                 "p":np.nan, "model_type":"two-variable"})
results.append({"id":"08", "metric":"On-ice CNFI+MNFI + suppression",
                 "type":"on-ice+sup","n":n,
                 "pearson_r":np.nan, "R2":round(r2_8,4),
                 "p":np.nan, "model_type":"two-variable"})

res_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
res_df.to_csv(f"{OUT}/individual_vs_onice_comparison.csv", index=False)
print(f"\nWrote {OUT}/individual_vs_onice_comparison.csv")

pd.options.display.float_format = lambda x: f"{x:.4f}" if not pd.isna(x) else ""
pd.options.display.width = 240; pd.options.display.max_columns = None

print("\n=== ALL 8 MODELS RANKED BY R² ===")
print(res_df[["id","metric","type","model_type","pearson_r","R2","p"]].to_string(index=False))

print("\n=== Two-variable model details ===\n")
print("Model 7 — Points ~ Individual F CNFI+MNFI /gp + Suppression")
labs = ["intercept","Individual gen","Suppression"]
for i,l in enumerate(labs):
    print(f"  {l:<22} β = {b7[i]:+9.3f}  se = {se7[i]:.3f}  t = {t7[i]:+6.3f}  p = {p7_[i]:.4e}")
print(f"  R² = {r2_7:.4f}\n")
print("Model 8 — Points ~ On-ice F CNFI+MNFI /60 + Suppression")
labs = ["intercept","On-ice gen","Suppression"]
for i,l in enumerate(labs):
    print(f"  {l:<22} β = {b8[i]:+9.3f}  se = {se8[i]:.3f}  t = {t8[i]:+6.3f}  p = {p8_[i]:.4e}")
print(f"  R² = {r2_8:.4f}")

# Correlation between individual and on-ice (informative)
print(f"\nCorrelation between individual and on-ice F CNFI+MNFI metrics: "
      f"r = {np.corrcoef(x_ind, x_oni)[0,1]:+.3f}")
print(f"Correlation between individual gen and suppression: "
      f"r = {np.corrcoef(x_ind, x_sup)[0,1]:+.3f}")
print(f"Correlation between on-ice gen and suppression: "
      f"r = {np.corrcoef(x_oni, x_sup)[0,1]:+.3f}")
