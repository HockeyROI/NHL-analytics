#!/usr/bin/env python3
"""
Filter publication_forwards_top100.csv and publication_D_top100.csv by age:
  - Remove players >25 yrs old AND <1,000 ES min
  - Remove players >40 yrs old regardless
Backfill to 100 by pulling next qualifying from full twoway_*_score.csv files.

Ages pulled from NHL API:
  https://api-web.nhle.com/v1/player/{player_id}/landing  -> birthDate
Cache to NFI/output/player_ages.csv to make re-runs fast.

Reference date: 2026-04-22 (system-supplied current date).
"""
import os, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import urllib.request
import urllib.error
import numpy as np
import pandas as pd

ROOT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis"
OUT  = f"{ROOT}/NFI/output"
AGE_CACHE = f"{OUT}/player_ages.csv"

REF_DATE = date(2026, 4, 22)

def years_between(birth_iso, ref):
    """Compute integer age (years) between birth ISO date string and ref date."""
    if not birth_iso or pd.isna(birth_iso):
        return None
    try:
        y, m, d = birth_iso.split("-")
        bd = date(int(y), int(m), int(d))
    except Exception:
        return None
    age = ref.year - bd.year - ((ref.month, ref.day) < (bd.month, bd.day))
    return age

# ---- Load source files ----
print("Loading source files ...")
twf = pd.read_csv(f"{OUT}/twoway_forward_score.csv")
twd = pd.read_csv(f"{OUT}/twoway_D_score.csv")
pub_f = pd.read_csv(f"{OUT}/publication_forwards_top100.csv")
pub_d = pd.read_csv(f"{OUT}/publication_D_top100.csv")

# Unique player_ids we need ages for
all_ids = sorted(set(twf["player_id"].astype(int).tolist()) |
                 set(twd["player_id"].astype(int).tolist()))
print(f"  unique players to age-check: {len(all_ids)}")

# ---- Load cache ----
if os.path.exists(AGE_CACHE):
    cache = pd.read_csv(AGE_CACHE, dtype={"player_id":int})
    print(f"  loaded cache: {len(cache)} cached ages")
    cached_ids = set(cache["player_id"].tolist())
else:
    cache = pd.DataFrame(columns=["player_id","birthDate","age"])
    cached_ids = set()

missing = [pid for pid in all_ids if pid not in cached_ids]
print(f"  need to pull from API: {len(missing)}")

# ---- Pull from NHL API ----
def fetch_birthdate(pid):
    url = f"https://api-web.nhle.com/v1/player/{pid}/landing"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            j = json.loads(resp.read().decode())
            return pid, j.get("birthDate")
    except urllib.error.HTTPError as e:
        return pid, f"HTTP_{e.code}"
    except Exception as e:
        return pid, f"ERR_{type(e).__name__}"

if missing:
    print(f"  pulling {len(missing)} player birth dates ...")
    new_rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=24) as ex:
        futures = {ex.submit(fetch_birthdate, pid): pid for pid in missing}
        for i, fut in enumerate(as_completed(futures), 1):
            pid, bd = fut.result()
            new_rows.append({"player_id":pid, "birthDate":bd})
            if i % 100 == 0:
                print(f"    {i}/{len(missing)} ({time.time()-t0:.1f}s)")
    new_df = pd.DataFrame(new_rows)
    cache = pd.concat([cache, new_df], ignore_index=True)
    print(f"  pulled in {time.time()-t0:.1f}s")

# Compute ages, save cache
cache["age"] = cache["birthDate"].apply(lambda b: years_between(b, REF_DATE))
cache.to_csv(AGE_CACHE, index=False)
print(f"  cache saved to {AGE_CACHE}")

bad = cache["age"].isna().sum()
print(f"  ages computed: {(cache['age'].notna()).sum()} | missing/error: {bad}")

age_map = dict(zip(cache["player_id"].astype(int), cache["age"]))

# ---- Apply filters ----
def apply_filters(df, age_col_attach=True):
    """Remove rows where (age>25 AND es_toi_min<1000) OR (age>40)."""
    df = df.copy()
    df["age"] = df["player_id"].map(age_map)
    drop_mask = ((df["age"] > 25) & (df["es_toi_min"] < 1000)) | (df["age"] > 40)
    return df[~drop_mask].copy(), df[drop_mask].copy()

# Forwards
twf_renamed = twf.rename(columns={"twoway_score":"two_way_score"})
twf_renamed["es_toi_min"] = twf_renamed["es_toi_min"]  # already this name
# Add position from publication_forwards (or player_positions)
pos = pd.read_csv(f"{OUT}/player_positions.csv", dtype={"player_id":int})
pos["pos_detail"] = pos["position"].map({"C":"C","L":"LW","R":"RW","D":"D","G":"G"})
twf_renamed = twf_renamed.merge(pos[["player_id","pos_detail"]], on="player_id", how="left")
# Pull P1b columns (P1b_pct, P1b_per60) from p1b file
p1b = pd.read_csv(f"{OUT}/P1b_rebound_arrival.csv")
twf_renamed = twf_renamed.merge(
    p1b[["player_id","P1b_pct","P1b_per60"]], on="player_id", how="left")

# Use the FULL twoway_forward_score.csv as the candidate pool, sorted by score
twf_pool = twf_renamed.sort_values("two_way_score", ascending=False).reset_index(drop=True)
twf_pool["age"] = twf_pool["player_id"].map(age_map)
# Apply filters on the full pool
mask_kept_f = ~(((twf_pool["age"] > 25) & (twf_pool["es_toi_min"] < 1000)) |
                 (twf_pool["age"] > 40))
twf_kept = twf_pool[mask_kept_f].copy()
twf_dropped = twf_pool[~mask_kept_f].copy()

# Take top 100 of kept
new_pub_f = twf_kept.head(100).copy().reset_index(drop=True)
new_pub_f["rank"] = np.arange(1, len(new_pub_f)+1)
new_pub_f["small_sample_flag"] = np.where(new_pub_f["es_toi_min"] < 2000.0, "Y", "N")
new_pub_f = new_pub_f.rename(columns={
    "pos_detail":"position",
    "z_P1a_weighted":"z_offensive",
    "z_P2_weighted": "z_defensive",
    "P1a_weighted_total": "P1a_weighted",
})
new_pub_f_cols = ["rank","player_name","position","es_toi_min","two_way_score",
                   "z_offensive","z_defensive","P1a_weighted","P1b_pct",
                   "P1b_per60","P2_weighted","small_sample_flag"]
new_pub_f = new_pub_f[new_pub_f_cols]

# D
twd_renamed = twd.rename(columns={"twoway_D_score":"two_way_D_score"})
blk = pd.read_csv(f"{OUT}/player_block_rates.csv")
twd_renamed = twd_renamed.merge(
    blk[["player_id","blocks_xG_prevented_per60"]], on="player_id", how="left")
twd_renamed["blocks_xG_prevented_per60"] = twd_renamed["blocks_xG_prevented_per60"].fillna(0.0)
twd_pool = twd_renamed.sort_values("two_way_D_score", ascending=False).reset_index(drop=True)
twd_pool["age"] = twd_pool["player_id"].map(age_map)
mask_kept_d = ~(((twd_pool["age"] > 25) & (twd_pool["es_toi_min"] < 1000)) |
                 (twd_pool["age"] > 40))
twd_kept = twd_pool[mask_kept_d].copy()
twd_dropped = twd_pool[~mask_kept_d].copy()
new_pub_d = twd_kept.head(100).copy().reset_index(drop=True)
new_pub_d["rank"] = np.arange(1, len(new_pub_d)+1)
new_pub_d["small_sample_flag"] = np.where(new_pub_d["es_toi_min"] < 2000.0, "Y", "N")
new_pub_d = new_pub_d.rename(columns={
    "z_P5_weighted":"z_offensive",
    "z_P4_weighted":"z_defensive",
})
new_pub_d_cols = ["rank","player_name","es_toi_min","two_way_D_score",
                   "z_offensive","z_defensive","P5_weighted","P4_weighted",
                   "blocks_xG_prevented_per60","small_sample_flag"]
new_pub_d = new_pub_d[new_pub_d_cols]

# Save
new_pub_f.to_csv(f"{OUT}/publication_forwards_top100.csv", index=False)
new_pub_d.to_csv(f"{OUT}/publication_D_top100.csv", index=False)

# ---- Reporting ----
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.3f}" if not pd.isna(x) else "")

# How many were in the OLD top 100 but got removed?
old_pub_f_ids = set(pub_f["player_id"]) if "player_id" in pub_f.columns else set()
# pub_f doesn't have player_id (we dropped it earlier in publication formatting). Use name matching.
old_f_names = set(pub_f["player_name"].tolist())
old_d_names = set(pub_d["player_name"].tolist())

# Map back to player_id via twf_renamed
name_to_id_f = dict(zip(twf_renamed["player_name"], twf_renamed["player_id"]))
name_to_id_d = dict(zip(twd_renamed["player_name"], twd_renamed["player_id"]))
old_f_pids = {name_to_id_f.get(n) for n in old_f_names if n in name_to_id_f}
old_d_pids = {name_to_id_d.get(n) for n in old_d_names if n in name_to_id_d}

# Removed = in old top 100 but in dropped pool
removed_f = twf_dropped[twf_dropped["player_id"].isin(old_f_pids)].copy()
removed_d = twd_dropped[twd_dropped["player_id"].isin(old_d_pids)].copy()

print(f"\n=== Removals from previous top 100 ===")
print(f"Forwards removed by age filter: {len(removed_f)}")
print(removed_f[["player_name","es_toi_min","age","two_way_score"]]
      .sort_values("two_way_score", ascending=False).to_string(index=False))
print(f"\nD removed by age filter: {len(removed_d)}")
print(removed_d[["player_name","es_toi_min","age","two_way_D_score"]]
      .sort_values("two_way_D_score", ascending=False).to_string(index=False))

# Total drops in pool (all, not just old top 100)
print(f"\n=== Pool-level filter stats ===")
print(f"Forwards pool size: {len(twf_pool)}, kept: {len(twf_kept)}, dropped: {len(twf_dropped)}")
print(f"  drop reasons: >40 yrs: {(twf_pool['age']>40).sum()}, "
      f">25 + <1000 min: {(((twf_pool['age']>25)&(twf_pool['es_toi_min']<1000))).sum()}")
print(f"D pool size: {len(twd_pool)}, kept: {len(twd_kept)}, dropped: {len(twd_dropped)}")
print(f"  drop reasons: >40 yrs: {(twd_pool['age']>40).sum()}, "
      f">25 + <1000 min: {(((twd_pool['age']>25)&(twd_pool['es_toi_min']<1000))).sum()}")

# Newcomers (in new top 100 but not in old top 100)
new_f_pids = set(new_pub_f.merge(twf_renamed[["player_name","player_id"]], on="player_name")["player_id"])
new_d_pids = set(new_pub_d.merge(twd_renamed[["player_name","player_id"]], on="player_name")["player_id"])
new_f_added = new_f_pids - old_f_pids
new_d_added = new_d_pids - old_d_pids
print(f"\n=== New entries to top 100 ===")
print(f"Forwards added: {len(new_f_added)}")
print(f"D added: {len(new_d_added)}")

print(f"\nFiles written:")
print(f"  {OUT}/publication_forwards_top100.csv")
print(f"  {OUT}/publication_D_top100.csv")
print(f"  {OUT}/player_ages.csv (cache)")
