"""TNFI relatives (RF/RA), 2000-min pooled filter, current-season horse race,
PP/PK correlation check.

Naming: TNFI = CNFI+MNFI% Fenwick (aka on-ice NFI%). All metrics use this
nomenclature throughout this run.

Outputs to NFI/output/fully_adjusted/:
  player_fully_adjusted.csv        (updated with RF/RA columns)
  current_season_player_fully_adjusted.csv
  team_fully_adjusted.csv          (updated)
  top30_FA_TNFI_pooled_2000min.csv
  top200_FA_TNFI_pooled_2000min.csv
  top30_TNFI_RF_pooled.csv
  top30_TNFI_RA_pooled.csv
  top30_TNFI_combined_relative_pooled.csv
  horse_race_current_season_relatives.csv
  pp_pk_tnfi_correlation.csv
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "NFI/output/fully_adjusted"
TEAM_METRICS = ROOT / "NFI/output/team_level_all_metrics.csv"
SHOTS_CSV = ROOT / "NFI/output/shots_tagged.csv"

ABBR = {"ARI":"UTA"}
POOLED = {"20222023","20232024","20242025","20252026"}
CURRENT = "20252026"
def norm(a): return ABBR.get(a,a) if a else a

print("[0] loading ...")
# Full player table with all FA/ZA metrics
pp = pd.read_csv(OUT / "player_fully_adjusted.csv")
pp["season"] = pp["season"].astype(str)
# Raw attribution pickle has per-player cf_fen, ca_fen, cf_cor, ca_cor, cf_cm, ca_cm, toi_sec
raw = pd.read_pickle("/tmp/s4_ppdf.pkl")
raw["season"] = raw["season"].astype(str)
raw_cols = ["player_id","season","team","pos","name","toi_sec",
            "cf_fen","ca_fen","cf_cor","ca_cor","cf_cm","ca_cm"]
raw = raw[raw_cols].copy()
pp = pp.merge(raw[["player_id","season","cf_fen","ca_fen","cf_cor","ca_cor","cf_cm","ca_cm"]],
              on=["player_id","season"], how="left")

# =====================================================================
# PART 1 — TOP 30 and TOP 200 FA-TNFI% pooled with 2000 ES min TOI
# =====================================================================
print("[PART 1] FA-TNFI% with min 2000 ES min TOI (pooled) ...")
grp = pp.groupby(["player_id","player_name","position"]).agg(
    n_seasons=("season","count"),
    toi_total=("toi_sec", lambda s: s.sum()/60),
    team_recent=("team","last"),
    raw_mean=("NFI_pct","mean"),
    ZA_mean=("ZA_NFI_emp","mean"),
    IOC_mean=("IOC_NFI","mean"),
    IOL_mean=("IOL_NFI","mean"),
    FA_mean=("FA_NFI_emp","mean"),
).reset_index()
grp_2000 = grp[grp["toi_total"]>=2000].sort_values("FA_mean", ascending=False).reset_index(drop=True)
grp_2000.to_csv(OUT/"top200_FA_TNFI_pooled_2000min.csv", index=False)
top30 = grp_2000.head(30)
top30.to_csv(OUT/"top30_FA_TNFI_pooled_2000min.csv", index=False)
print(f"    {len(grp_2000)} players above 2000 ES min; saved top 30 + top 200")

print("\n=== FA-TNFI% TOP 30 POOLED (min 2000 ES min TOI) ===")
for i, r in top30.iterrows():
    print(f"{i+1:>3}  {r['player_name']:<26} {r['position']:<3} {r['team_recent']:<5}  "
          f"n={r['n_seasons']}  toi={r['toi_total']:>6.0f}  "
          f"raw={r['raw_mean']:.4f}  ZA={r['ZA_mean']:.4f}  FA={r['FA_mean']:+.4f}")

# =====================================================================
# PART 2 — Build TNFI_RF, TNFI_RA + Corsi/Fenwick relatives
# Per-team-season totals at ES. Team ES TOI = sum of player TOI / 5.
# =====================================================================
print("\n[PART 2] building RF/RA relative metrics ...")
# Per team-season totals from ppdf (from player on-ice counts).
# Team CF_F_total = sum over players of cf_fen / 5 (each for-event attributed to 5 on-ice)
# Team CF_A_total = sum ca_fen / 5  (similar)
team_totals = raw.groupby(["team","season"]).agg(
    team_toi_sec_sum=("toi_sec","sum"),
    sum_cf_fen=("cf_fen","sum"), sum_ca_fen=("ca_fen","sum"),
    sum_cf_cor=("cf_cor","sum"), sum_ca_cor=("ca_cor","sum"),
    sum_cf_cm=("cf_cm","sum"),   sum_ca_cm=("ca_cm","sum"),
).reset_index()
# Team-level aggregates (divide by 5 to remove double-counting across on-ice players)
team_totals["team_es_toi_sec"] = team_totals["team_toi_sec_sum"] / 5.0
for col in ["cf_fen","ca_fen","cf_cor","ca_cor","cf_cm","ca_cm"]:
    team_totals[f"team_{col}"] = team_totals[f"sum_{col}"] / 5.0

pp = pp.merge(team_totals[["team","season","team_es_toi_sec",
                            "team_cf_fen","team_ca_fen",
                            "team_cf_cor","team_ca_cor",
                            "team_cf_cm","team_ca_cm"]],
              on=["team","season"], how="left")

# Per-60 rates (player toi_sec is on-ice time)
def per60(num, toi_sec):
    return np.where(toi_sec > 0, num / toi_sec * 3600, np.nan)

pp["on60_cf_fen"]  = per60(pp["cf_fen"], pp["toi_sec"])
pp["on60_ca_fen"]  = per60(pp["ca_fen"], pp["toi_sec"])
pp["on60_cf_cor"]  = per60(pp["cf_cor"], pp["toi_sec"])
pp["on60_ca_cor"]  = per60(pp["ca_cor"], pp["toi_sec"])
pp["on60_cf_cm"]   = per60(pp["cf_cm"],  pp["toi_sec"])
pp["on60_ca_cm"]   = per60(pp["ca_cm"],  pp["toi_sec"])

off_toi = pp["team_es_toi_sec"] - pp["toi_sec"]
pp["off60_cf_fen"] = per60(pp["team_cf_fen"] - pp["cf_fen"], off_toi)
pp["off60_ca_fen"] = per60(pp["team_ca_fen"] - pp["ca_fen"], off_toi)
pp["off60_cf_cor"] = per60(pp["team_cf_cor"] - pp["cf_cor"], off_toi)
pp["off60_ca_cor"] = per60(pp["team_ca_cor"] - pp["ca_cor"], off_toi)
pp["off60_cf_cm"]  = per60(pp["team_cf_cm"]  - pp["cf_cm"],  off_toi)
pp["off60_ca_cm"]  = per60(pp["team_ca_cm"]  - pp["ca_cm"],  off_toi)

# RF / RA metrics
pp["TNFI_RF"] = pp["on60_cf_cm"] - pp["off60_cf_cm"]    # CNFI+MNFI for: on - off
pp["TNFI_RA"] = pp["off60_ca_cm"] - pp["on60_ca_cm"]    # against: off - on (higher = suppress more)
pp["TNFI_combined"] = pp["TNFI_RF"] + pp["TNFI_RA"]

pp["CF_RF"] = pp["on60_cf_cor"] - pp["off60_cf_cor"]
pp["CF_RA"] = pp["off60_ca_cor"] - pp["on60_ca_cor"]
pp["CF_combined"] = pp["CF_RF"] + pp["CF_RA"]

pp["FF_RF"] = pp["on60_cf_fen"] - pp["off60_cf_fen"]
pp["FF_RA"] = pp["off60_ca_fen"] - pp["on60_ca_fen"]
pp["FF_combined"] = pp["FF_RF"] + pp["FF_RA"]

# Top 30 by TNFI_RF / RA / combined — career mean, min 2000 ES min TOI pooled
def top_relative(col, n=30, min_toi=2000):
    g = pp.groupby(["player_id","player_name","position"]).agg(
        n_seasons=("season","count"),
        toi_total=("toi_sec", lambda s: s.sum()/60),
        team_recent=("team","last"),
        metric_mean=(col, "mean"),
    ).reset_index()
    g = g[g["toi_total"]>=min_toi]
    return g.sort_values("metric_mean", ascending=False).reset_index(drop=True).head(n)

top_rf   = top_relative("TNFI_RF")
top_ra   = top_relative("TNFI_RA")
top_comb = top_relative("TNFI_combined")

top_rf.to_csv(OUT/"top30_TNFI_RF_pooled.csv", index=False)
top_ra.to_csv(OUT/"top30_TNFI_RA_pooled.csv", index=False)
top_comb.to_csv(OUT/"top30_TNFI_combined_relative_pooled.csv", index=False)

def show(df, label, col):
    print(f"\n=== {label} (pooled, min 2000 ES min) ===")
    for i, r in df.iterrows():
        print(f"{i+1:>3}  {r['player_name']:<26} {r['position']:<3} {r['team_recent']:<5}  "
              f"n={r['n_seasons']}  toi={r['toi_total']:>6.0f}  {col}={r['metric_mean']:+.3f} /60")

show(top_rf,   "TOP 30 by TNFI_RF", "RF")
show(top_ra,   "TOP 30 by TNFI_RA", "RA")
show(top_comb, "TOP 30 by TNFI_RF + TNFI_RA (combined two-way)", "RF+RA")

# =====================================================================
# PART 3 — Current-season individual metric horse race (N=32)
# =====================================================================
print("\n[PART 3] current-season horse race (individual → team) ...")
tm = pd.read_csv(TEAM_METRICS)
tm["season"] = tm["season"].astype(str)
tm = tm[tm["season"]==CURRENT][["season","team","points"]]

curr = pp[pp["season"]==CURRENT].copy()

def team_agg_toi(df, col):
    d = df.dropna(subset=[col,"team"]).copy()
    d["w"] = d["toi_sec"]
    return d.groupby(["season","team"]).apply(
        lambda x: np.average(x[col], weights=x["w"])).rename(col).reset_index()

def r2(x, y):
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 5: return (np.nan, np.nan, 0)
    r = np.corrcoef(x[mask], y[mask])[0,1]
    return r*r, r, int(mask.sum())

horse = []
for label, col in [
    ("TNFI%_ZA",          "ZA_NFI_emp"),
    ("TNFI%_FA",          "FA_NFI_emp"),
    ("CF%_ZA",            "ZA_CF_emp"),
    ("CF%_FA",            "FA_CF_emp"),
    ("FF%_ZA",            "ZA_FF_emp"),
    ("FF%_FA",            "FA_FF_emp"),
    ("TNFI_RF",           "TNFI_RF"),
    ("TNFI_RA",           "TNFI_RA"),
    ("TNFI_RF+TNFI_RA",   "TNFI_combined"),
    ("CF_RF",             "CF_RF"),
    ("CF_RA",             "CF_RA"),
    ("CF_RF+CF_RA",       "CF_combined"),
    ("FF_RF",             "FF_RF"),
    ("FF_RA",             "FF_RA"),
    ("FF_RF+FF_RA",       "FF_combined"),
]:
    tb = team_agg_toi(curr, col).merge(tm, on=["season","team"], how="inner")
    r2v, rv, n = r2(tb[col].values.astype(float), tb["points"].values.astype(float))
    horse.append({"metric":label, "R2":r2v, "r":rv, "N":n})

hh = pd.DataFrame(horse).sort_values("R2", ascending=False).reset_index(drop=True)
hh.to_csv(OUT/"horse_race_current_season_relatives.csv", index=False)

print("\n=== CURRENT-SEASON HORSE RACE (individual player → team agg TOI-weighted, 2025-26 N=32) ===")
print(hh.to_string(index=False))

# =====================================================================
# PART 4 — PP / PK TNFI% correlation with actual PP%/PK%
# =====================================================================
print("\n[PART 4] PP/PK TNFI% predictiveness ...")
# Compute per team-season TNFI% at PP and PK states from shots_tagged
st = pd.read_csv(SHOTS_CSV,
    usecols=["game_id","season","event_type","shooting_team_abbrev",
             "home_team_abbrev","away_team_abbrev","zone","state","is_goal_i"])
st["season"] = st["season"].astype(str)
st = st[st["season"].isin(POOLED)]
st["shooting_team_abbrev"] = st["shooting_team_abbrev"].astype(str).map(norm)
st["home_team_abbrev"] = st["home_team_abbrev"].astype(str).map(norm)
st["away_team_abbrev"] = st["away_team_abbrev"].astype(str).map(norm)
FEN = {"shot-on-goal","missed-shot","goal"}
st = st[st["event_type"].isin(FEN)]   # Fenwick events
st["is_cm"] = st["zone"].isin(["CNFI","MNFI"])
st["opp"] = np.where(st["shooting_team_abbrev"]==st["home_team_abbrev"],
                     st["away_team_abbrev"], st["home_team_abbrev"])

def tnfi_state(state):
    s = st[st["state"]==state]
    # Team TNFI For at state
    for_df = s[s["is_cm"]].groupby(["season","shooting_team_abbrev"]).size().rename("cm_f").reset_index()
    for_df.columns = ["season","team","cm_f"]
    ag_df  = s[s["is_cm"]].groupby(["season","opp"]).size().rename("cm_a").reset_index()
    ag_df.columns = ["season","team","cm_a"]
    tot_for = s.groupby(["season","shooting_team_abbrev"]).size().rename("fen_f").reset_index()
    tot_for.columns = ["season","team","fen_f"]
    tot_ag  = s.groupby(["season","opp"]).size().rename("fen_a").reset_index()
    tot_ag.columns = ["season","team","fen_a"]
    out = for_df.merge(ag_df, on=["season","team"], how="outer").fillna(0)
    out = out.merge(tot_for, on=["season","team"], how="outer").fillna(0)
    out = out.merge(tot_ag, on=["season","team"], how="outer").fillna(0)
    out["TNFI_pct"] = out["cm_f"]/(out["cm_f"]+out["cm_a"]).replace(0,np.nan)
    out["TNFI_for_rate"] = out["cm_f"]  # shots; later divide by TOI if available
    out["TNFI_ag_rate"]  = out["cm_a"]
    return out

pp_tnfi = tnfi_state("PP")
pk_tnfi = tnfi_state("PK")

# Fetch MoneyPuck 5on4 (PP) and 4on5 (PK) team data to get goals-for/against rates
print("    fetching MoneyPuck 5on4 / 4on5 team data ...")
mp_rows = []
for yr in [2022,2023,2024,2025]:
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{yr}/regular/teams.csv"
    subprocess.run(["curl","-sL","-A","Mozilla/5.0", url, "-o", f"/tmp/mp_{yr}.csv"], check=False)
    try:
        t = pd.read_csv(f"/tmp/mp_{yr}.csv")
        t["season"] = f"{yr}{yr+1}"
        t["team"] = t["team"].map(norm)
        mp_rows.append(t[["season","team","situation","goalsFor","goalsAgainst","iceTime"]])
    except Exception: pass
mp = pd.concat(mp_rows, ignore_index=True)
# PP goals/60 = 5on4 goalsFor / iceTime × 3600
pp_perf = mp[mp["situation"]=="5on4"].copy()
pp_perf["PP_goals_per60"] = pp_perf["goalsFor"] / pp_perf["iceTime"] * 3600
# PK goals-against/60 = 4on5 goalsAgainst / iceTime × 3600
pk_perf = mp[mp["situation"]=="4on5"].copy()
pk_perf["PK_goals_against_per60"] = pk_perf["goalsAgainst"] / pk_perf["iceTime"] * 3600

# Merge
pp_merge = pp_tnfi.merge(pp_perf[["season","team","PP_goals_per60"]],
                          on=["season","team"], how="inner")
pk_merge = pk_tnfi.merge(pk_perf[["season","team","PK_goals_against_per60"]],
                          on=["season","team"], how="inner")

# PP TNFI% vs PP goals/60: positive correlation expected
r_pp = r2(pp_merge["TNFI_pct"].values, pp_merge["PP_goals_per60"].values)
# PK TNFI% = TEAM's TNFI% while on PK (low = good PK). Team is shorthanded so opp shooting. TNFI% here = own team's CNFI+MNFI share while shorthanded. Low = good PK (opponent's shots dominate).
# Actually for PK correlation: we care about opponent's NFI% when team is shorthanded → 1 - team's TNFI% at PK = opp TNFI% at PK. Let's convert.
pk_merge["Opp_TNFI_pct_against_team_PK"] = 1 - pk_merge["TNFI_pct"]
r_pk = r2(pk_merge["Opp_TNFI_pct_against_team_PK"].values, pk_merge["PK_goals_against_per60"].values)

print("\n=== PP: team TNFI%_PP vs PP goals/60 ===")
print(f"N = {r_pp[2]}, r = {r_pp[1]:+.4f}, R² = {r_pp[0]:.4f}")
# p-value via scipy
import scipy.stats as sst
mask = (~np.isnan(pp_merge["TNFI_pct"].values)) & (~np.isnan(pp_merge["PP_goals_per60"].values))
pv = sst.pearsonr(pp_merge.loc[mask,"TNFI_pct"], pp_merge.loc[mask,"PP_goals_per60"]).pvalue
print(f"p = {pv:.4g}")

print("\n=== PK: opponent TNFI% (while team shorthanded) vs team's PK goals-against/60 ===")
print(f"N = {r_pk[2]}, r = {r_pk[1]:+.4f}, R² = {r_pk[0]:.4f}")
mask2 = (~np.isnan(pk_merge["Opp_TNFI_pct_against_team_PK"].values)) & (~np.isnan(pk_merge["PK_goals_against_per60"].values))
pv2 = sst.pearsonr(pk_merge.loc[mask2,"Opp_TNFI_pct_against_team_PK"],
                   pk_merge.loc[mask2,"PK_goals_against_per60"]).pvalue
print(f"p = {pv2:.4g}")

# Save to CSV
pp_pk_rows = [
    {"test":"PP: TNFI%_PP → PP goals/60", "N":r_pp[2], "r":r_pp[1], "R2":r_pp[0], "p":pv},
    {"test":"PK: opp TNFI%_PK → PK goals-against/60", "N":r_pk[2], "r":r_pk[1], "R2":r_pk[0], "p":pv2},
]
pd.DataFrame(pp_pk_rows).to_csv(OUT/"pp_pk_tnfi_correlation.csv", index=False)

# =====================================================================
# PART 5 — Update player_fully_adjusted.csv with RF/RA columns
# =====================================================================
print("\n[PART 5] writing updated Streamlit CSVs ...")
add_cols = ["on60_cf_fen","off60_cf_fen","on60_ca_fen","off60_ca_fen",
            "on60_cf_cor","off60_cf_cor","on60_ca_cor","off60_ca_cor",
            "on60_cf_cm", "off60_cf_cm", "on60_ca_cm", "off60_ca_cm",
            "TNFI_RF","TNFI_RA","TNFI_combined",
            "CF_RF","CF_RA","CF_combined",
            "FF_RF","FF_RA","FF_combined",
            "team_es_toi_sec"]
pp_out = pp.drop(columns=[c for c in ["es_toi_min"] if c in pp.columns], errors="ignore")
pp_out["es_toi_min"] = pp_out["toi_sec"]/60
pp_out.to_csv(OUT/"player_fully_adjusted.csv", index=False)
pp_out[pp_out["season"]==CURRENT].to_csv(OUT/"current_season_player_fully_adjusted.csv", index=False)
print(f"    wrote {len(pp_out)} rows to player_fully_adjusted.csv + current-season")

# Team aggregated (team-season)
team_rows = []
for (season, team), sub in pp.groupby(["season","team"]):
    if len(sub)==0: continue
    row = {"season":season, "team":team, "n_players":len(sub)}
    for col in ["NFI_pct","ZA_NFI_emp","FA_NFI_emp",
                "CF_pct","ZA_CF_emp","ZA_CF_trad","FA_CF_emp","FA_CF_trad",
                "FF_pct","ZA_FF_emp","ZA_FF_trad","FA_FF_emp","FA_FF_trad",
                "TNFI_RF","TNFI_RA","TNFI_combined",
                "CF_RF","CF_RA","CF_combined","FF_RF","FF_RA","FF_combined"]:
        vals = sub[[col,"toi_sec"]].dropna()
        row[col] = float(np.average(vals[col], weights=vals["toi_sec"])) if len(vals) else np.nan
    team_rows.append(row)

tm_full = pd.read_csv(TEAM_METRICS)
tm_full["season"] = tm_full["season"].astype(str)
tm_full = tm_full[tm_full["season"].isin(POOLED)][["season","team","points"]]
team_out = pd.DataFrame(team_rows).merge(tm_full, on=["season","team"], how="left")
team_out.to_csv(OUT/"team_fully_adjusted.csv", index=False)
team_out[team_out["season"]==CURRENT].to_csv(OUT/"current_season_team_fully_adjusted.csv", index=False)

print(f"\n[done] all outputs in {OUT}")
