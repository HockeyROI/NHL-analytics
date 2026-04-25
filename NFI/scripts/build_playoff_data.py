"""Build playoff data for TNZI and NFI from cached PBP/shifts.

Inputs:
  Data/game_ids.csv (game_type=playoff)
  Zones/raw/pbp/<game_id>.json
  Zones/raw/shifts/<game_id>.json
  NFI/output/shots_tagged.csv (already includes playoff shots tagged
    with CNFI/MNFI/FNFI zones)
  NFI/output/player_positions.csv

Outputs:
  Zones/output/playoffs/tnzi_adjusted_forwards_playoffs.csv
  Zones/output/playoffs/tnzi_adjusted_defense_playoffs.csv
  NFI/output/fully_adjusted/player_fully_adjusted_playoffs.csv

Notes on simplifications:
  * Playoff samples are small (~262 games / 4 seasons / ~85 per yr),
    so per-player TOI is much lower than regular season. The full 3A
    pipeline (with linemate-without-me NFQOC/NFQOL) is unstable on
    samples this small, so for playoffs we publish raw NFI% and
    NFI%_ZA only. NFI%_3A column is left blank.
  * TNZI playoff CSV omits IOZC/IOZL/_C/_L/_CL/DOZI columns for the
    same reason. Streamlit table will display the columns it has.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PBP_DIR    = ROOT / "Zones" / "raw" / "pbp"
SHIFT_CSV  = ROOT / "NFI" / "Geometry_post" / "Data" / "shift_data.csv"
SHOTS_CSV  = ROOT / "NFI" / "output" / "shots_tagged.csv"
POS_CSV    = ROOT / "NFI" / "output" / "player_positions.csv"
GAME_IDS   = ROOT / "Data" / "game_ids.csv"
OUT_TNZI   = ROOT / "Zones" / "output" / "playoffs"
OUT_NFI    = ROOT / "NFI" / "output" / "fully_adjusted"
OUT_TNZI.mkdir(parents=True, exist_ok=True)
OUT_NFI.mkdir(parents=True, exist_ok=True)

ABBR_MAP = {"ARI": "UTA"}
FLIP = {"O":"D","D":"O","N":"N"}
FENWICK_TYPES = {"shot-on-goal","missed-shot","goal"}
NFI_ZA_FACTOR = 0.10710  # empirical OZ-DZ pct-pt gap for NFI%

def norm(a): return ABBR_MAP.get(a, a) if a else a

def mmss(s):
    if not s or ":" not in s: return 0
    m, ss = s.split(":")
    try: return int(m)*60 + int(ss)
    except ValueError: return 0


# ------------------------------------------------------------------
# 1. Load playoff game IDs
# ------------------------------------------------------------------
print("[1/6] loading playoff game IDs ...")
games_df = pd.read_csv(GAME_IDS)
playoffs = games_df[games_df["game_type"] == "playoff"].copy()
playoffs["season"] = playoffs["season"].astype(str)
# Restrict to seasons we have raw PBP for
have_pbp = {p.stem for p in PBP_DIR.glob("*.json")}
playoffs = playoffs[playoffs["game_id"].astype(str).isin(have_pbp)]
g2s = dict(zip(playoffs["game_id"].astype(int), playoffs["season"].astype(str)))
gids = sorted(g2s.keys())
print(f"    playoff games with PBP on disk: {len(gids)}")
print(f"    by season: {playoffs.groupby('season').size().to_dict()}")

# ------------------------------------------------------------------
# 2. Load shifts and shots filtered to playoff scope
# ------------------------------------------------------------------
print("[2/6] loading shifts ...")
sd = pd.read_csv(SHIFT_CSV,
    usecols=["game_id","player_id","team_abbrev","abs_start_secs","abs_end_secs"])
sd = sd.dropna(subset=["player_id","abs_start_secs","abs_end_secs"])
sd["game_id"] = sd["game_id"].astype("int64")
sd = sd[sd["game_id"].isin(gids)]
sd["player_id"] = sd["player_id"].astype("int64")
sd["abs_start_secs"] = sd["abs_start_secs"].astype("int32")
sd["abs_end_secs"]   = sd["abs_end_secs"].astype("int32")
sd["team_abbrev"] = sd["team_abbrev"].astype(str).map(norm)
sd = sd.sort_values(["game_id","abs_start_secs"]).reset_index(drop=True)
print(f"    playoff shift rows: {len(sd):,}")
shifts_by_game = {gid: (g["player_id"].to_numpy(), g["team_abbrev"].to_numpy(),
                        g["abs_start_secs"].to_numpy(), g["abs_end_secs"].to_numpy())
                  for gid, g in sd.groupby("game_id", sort=False)}

print("[3/6] loading shots (ES, playoff games only) ...")
xd = pd.read_csv(SHOTS_CSV,
    usecols=["game_id","season","abs_time","event_type","shooting_team_abbrev",
             "zone","state"])
xd["game_id"] = xd["game_id"].astype("int64")
xd = xd[xd["game_id"].isin(gids)]
xd = xd[xd["state"] == "ES"]
xd["shooting_team_abbrev"] = xd["shooting_team_abbrev"].astype(str).map(norm)
xd["is_fen"] = xd["event_type"].isin(FENWICK_TYPES)
xd["is_cm"]  = xd["zone"].isin(["CNFI","MNFI"])
xd = xd.sort_values(["game_id","abs_time"]).reset_index(drop=True)
print(f"    playoff ES shots: {len(xd):,}")
shots_by_game = {gid:{"t": g["abs_time"].to_numpy(),
                      "team": g["shooting_team_abbrev"].to_numpy(),
                      "fen": g["is_fen"].to_numpy(),
                      "cm":  g["is_cm"].to_numpy()}
                 for gid, g in xd.groupby("game_id", sort=False)}

pos_df = pd.read_csv(POS_CSV)
pos_map = dict(zip(pos_df["player_id"].astype(int), pos_df["pos_group"].astype(str)))
name_map = dict(zip(pos_df["player_id"].astype(int), pos_df["player_name"].astype(str)))

# ------------------------------------------------------------------
# 4. Extract faceoffs from PBP for TNZI + per-player on-ice attribution.
#    One pass through PBP/shifts/shots produces:
#       - per-player (OZ_shifts, DZ_shifts, NZ_shifts, OZ_eventful_OZ,
#         DZ_eventful_OZ, ...) for TNZI computation
#       - per-player on-ice (cf_fen, ca_fen, cf_cm, ca_cm) + TOI for NFI
# ------------------------------------------------------------------
print("[4/6] processing playoff games (faceoffs + shot attribution) ...")

# TNZI counters: (player_id, faceoff_zone) -> dict
# faceoff_zone in {O, D, N}
tnzi_buckets = defaultdict(lambda: {"shifts":0, "oz_sec":0.0, "dz_sec":0.0, "nz_sec":0.0, "total_sec":0.0})

# NFI counters keyed by (player_id, season) so we can emit per-season rows
# AND pooled rows for the playoff CSVs.
nfi = defaultdict(lambda: {"toi_sec":0.0, "cf_fen":0, "ca_fen":0,
                            "cf_cm":0, "ca_cm":0, "fo_oz":0, "fo_dz":0,
                            "team": ""})

# Track GP per player-season
player_season_gp = defaultdict(int)

def play_abs_time(p):
    period = (p.get("periodDescriptor") or {}).get("number", 1) or 1
    return (period - 1) * 1200 + mmss(p.get("timeInPeriod", "00:00"))

for gi, gid in enumerate(gids):
    if gid not in shifts_by_game:
        continue
    season = g2s[gid]
    pbp_path = PBP_DIR / f"{gid}.json"
    if not pbp_path.exists():
        continue
    try:
        pbp = json.load(open(pbp_path))
    except Exception:
        continue
    home_id = (pbp.get("homeTeam") or {}).get("id")
    away_id = (pbp.get("awayTeam") or {}).get("id")
    home_ab = norm((pbp.get("homeTeam") or {}).get("abbrev"))
    away_ab = norm((pbp.get("awayTeam") or {}).get("abbrev"))
    if not home_id or not away_id:
        continue

    pids, teams, starts, ends = shifts_by_game[gid]

    # --- NFI per-(player, season) TOI from shifts ---
    for i in range(len(pids)):
        rec = nfi[(int(pids[i]), season)]
        rec["toi_sec"] += float(ends[i] - starts[i])
        if not rec["team"]:
            rec["team"] = teams[i]
        else:
            rec["team"] = teams[i]  # most recent

    # --- NFI on-ice shot attribution from this game's shots ---
    shots = shots_by_game.get(gid)
    if shots is not None and len(shots["t"]) > 0:
        shot_t = shots["t"]; shot_team = shots["team"]
        shot_fen = shots["fen"]; shot_cm = shots["cm"]
        for i in range(len(pids)):
            s_i, e_i = starts[i], ends[i]
            a = bisect_left(shot_t, s_i)
            b = bisect_left(shot_t, e_i)
            if b <= a: continue
            sl_team = shot_team[a:b]
            pid_i = int(pids[i]); team_i = teams[i]
            is_own = (sl_team == team_i)
            f_arr = shot_fen[a:b]; c_arr = shot_cm[a:b]
            r = nfi[(pid_i, season)]
            r["cf_fen"] += int((f_arr & is_own).sum())
            r["ca_fen"] += int((f_arr & ~is_own).sum())
            r["cf_cm"]  += int((c_arr & is_own).sum())
            r["ca_cm"]  += int((c_arr & ~is_own).sum())

    # --- GP per (player, season) ---
    for pid in set(int(p) for p in pids):
        player_season_gp[(pid, season)] += 1

    # --- TNZI: walk plays, track faceoff context, attribute eventful zone time ---
    plays = pbp.get("plays") or []
    ctx = None  # active faceoff context

    def close_and_emit(ctx, close_t):
        if ctx is None:
            return
        fo_zone_home = ctx["fo_zone_home"]
        # Build per-event zone series
        events = ctx["events"]
        # Process each shift active at fo_t
        for pid_i, team_i, shift_end in ctx["active_shifts"]:
            eff_end = min(close_t, shift_end)
            if eff_end <= ctx["fo_t"]:
                continue
            zone_p = fo_zone_home if team_i == home_id else FLIP[fo_zone_home]
            if zone_p not in ("O","D","N"):
                continue
            # OZ/DZ faceoff exposure for NFI's per-(player, season) oz_ratio
            r = nfi[(pid_i, season)]
            if zone_p == "O": r["fo_oz"] += 1
            elif zone_p == "D": r["fo_dz"] += 1
            # TNZI: count this faceoff shift, attribute time per zone of subsequent events
            key = (pid_i, season, zone_p)
            tk = tnzi_buckets[key]
            tk["shifts"] += 1
            # Time attribution from event to next event boundary
            filtered = [(t, ez_home, typ) for (t, ez_home, typ) in events
                         if ctx["fo_t"] <= t < eff_end]
            if not filtered:
                continue
            # Initial event = the faceoff itself at fo_zone_home
            for j, (t, ez_h, typ) in enumerate(filtered):
                t_next = filtered[j+1][0] if j+1 < len(filtered) else eff_end
                dt = max(0.0, t_next - t)
                ez_player = ez_h if team_i == home_id else FLIP[ez_h]
                tk["total_sec"] += dt
                if ez_player == "O": tk["oz_sec"] += dt
                elif ez_player == "D": tk["dz_sec"] += dt
                elif ez_player == "N": tk["nz_sec"] += dt

    for p in plays:
        typ = p.get("typeDescKey") or ""
        details = p.get("details") or {}
        t_abs = play_abs_time(p)
        situation = p.get("situationCode") or ""
        if typ == "faceoff":
            if ctx is not None:
                close_and_emit(ctx, t_abs)
                ctx = None
            if situation != "1551":
                continue
            zone_code = details.get("zoneCode")
            if zone_code not in ("O","D","N"):
                continue
            event_owner = details.get("eventOwnerTeamId")
            if event_owner not in (home_id, away_id):
                continue
            fo_zone_home = zone_code if event_owner == home_id else FLIP[zone_code]
            # Active shifts at faceoff
            hi = bisect_right(starts, t_abs)
            active = []
            for i in range(hi):
                if ends[i] <= t_abs: continue
                active.append((int(pids[i]), teams[i], int(ends[i])))
            ctx = {
                "fo_t": t_abs,
                "fo_zone_home": fo_zone_home,
                "active_shifts": active,
                "events": [(t_abs, fo_zone_home, "faceoff")],
            }
            continue
        if ctx is None:
            continue
        if situation and situation != "1551":
            close_and_emit(ctx, t_abs)
            ctx = None
            continue
        zc = details.get("zoneCode")
        if zc in ("O","D","N") and typ in (
            "faceoff","hit","shot-on-goal","missed-shot","blocked-shot","goal",
            "giveaway","takeaway"):
            ev_owner = details.get("eventOwnerTeamId")
            if ev_owner in (home_id, away_id):
                ez_home = zc if ev_owner == home_id else FLIP[zc]
                ctx["events"].append((t_abs, ez_home, typ))
        if typ in ("period-end","game-end"):
            close_and_emit(ctx, t_abs)
            ctx = None
    if ctx is not None:
        last_t = play_abs_time(plays[-1]) if plays else ctx["fo_t"]
        close_and_emit(ctx, last_t)

    if (gi+1) % 50 == 0 or gi+1 == len(gids):
        print(f"    {gi+1}/{len(gids)}")

# ------------------------------------------------------------------
# 5. Build TNZI playoff CSVs (forwards + defense)
# ------------------------------------------------------------------
print("[5/6] computing TNZI scores (per-season + pooled) ...")

# Tag each shift row with its season once for downstream filtering.
sd["season"] = sd["game_id"].map(g2s)

MIN_SHIFTS = 30
def norm_010(series):
    s = series.dropna()
    if len(s) == 0: return None
    lo, hi = s.min(), s.max()
    span = hi - lo if hi > lo else 1.0
    return ((series - lo) / span * 10.0).round(1)

# Two passes:
#   1) per-season aggregation — group buckets by (pid, season, zone)
#   2) pooled aggregation — group buckets by (pid, zone) ignoring season
def aggregate_tnzi_rows(group_keys, season_label):
    """group_keys: dict mapping pid -> {zone -> {shifts/oz_sec/dz_sec/total_sec}}.
    season_label: int season (e.g. 20242025) or "all_playoffs"."""
    rows = []
    for pid, b in group_keys.items():
        pos = pos_map.get(int(pid))
        if pos not in ("F","D"): continue
        name = name_map.get(int(pid), "")
        # Player team for this season slice
        if season_label == "all_playoffs":
            p_shifts = sd[sd["player_id"] == pid]
        else:
            p_shifts = sd[(sd["player_id"] == pid) & (sd["season"] == str(season_label))]
        if p_shifts.empty: continue
        team = p_shifts.sort_values("game_id")["team_abbrev"].iloc[-1]
        if season_label == "all_playoffs":
            gp = sum(g for (p, _), g in player_season_gp.items() if p == pid)
        else:
            gp = player_season_gp.get((pid, str(season_label)), 0)
        ozi  = (b["O"]["oz_sec"] / b["O"]["total_sec"]) if b["O"]["total_sec"] > 0 else None
        dzi  = (b["D"]["oz_sec"] / b["D"]["total_sec"]) if b["D"]["total_sec"] > 0 else None
        nzi  = (b["N"]["oz_sec"] / b["N"]["total_sec"]) if b["N"]["total_sec"] > 0 else None
        tnzi = (nzi - (1 - nzi)) if nzi is not None else None
        rows.append({
            "player_name": name, "team": team, "pos": pos, "GP": gp,
            "season": season_label,
            "shifts_O": b["O"]["shifts"], "shifts_D": b["D"]["shifts"], "shifts_N": b["N"]["shifts"],
            "OZI_raw_pct": ozi, "DZI_raw_pct": dzi, "NZI_raw_pct": nzi,
            "TNZI_raw":  tnzi,
        })
    return rows

# Pooled-by-player: regroup buckets to ignore season
pooled = defaultdict(lambda: {"O":{"shifts":0,"oz_sec":0.0,"dz_sec":0.0,"total_sec":0.0},
                               "D":{"shifts":0,"oz_sec":0.0,"dz_sec":0.0,"total_sec":0.0},
                               "N":{"shifts":0,"oz_sec":0.0,"dz_sec":0.0,"total_sec":0.0}})
# Per-(pid, season) regroup
per_season = defaultdict(lambda: defaultdict(lambda: {
    "O":{"shifts":0,"oz_sec":0.0,"dz_sec":0.0,"total_sec":0.0},
    "D":{"shifts":0,"oz_sec":0.0,"dz_sec":0.0,"total_sec":0.0},
    "N":{"shifts":0,"oz_sec":0.0,"dz_sec":0.0,"total_sec":0.0}}))
for (pid, season, zone), b in tnzi_buckets.items():
    for tgt in (pooled[pid][zone], per_season[season][pid][zone]):
        tgt["shifts"] += b["shifts"]
        tgt["oz_sec"] += b["oz_sec"]
        tgt["dz_sec"] += b["dz_sec"]
        tgt["total_sec"] += b["total_sec"]

all_rows = aggregate_tnzi_rows(pooled, "all_playoffs")
for season_lbl, group in per_season.items():
    all_rows.extend(aggregate_tnzi_rows(group, int(season_lbl)))

tnzi_df = pd.DataFrame(all_rows)
# Normalize 0-10 WITHIN each (season_label, pos_group) so per-season scores
# are comparable to that season's other players, and the pooled score is
# normalized over the pooled population.
for col in ["OZI_raw_pct", "DZI_raw_pct", "NZI_raw_pct", "TNZI_raw"]:
    out_col = col.replace("_raw_pct","").replace("_raw","")
    tnzi_df[out_col] = np.nan
    for (season_lbl, pos_grp), idx in tnzi_df.groupby(["season","pos"]).groups.items():
        ok = ((tnzi_df.loc[idx, "shifts_O"] >= MIN_SHIFTS) |
              (tnzi_df.loc[idx, "shifts_D"] >= MIN_SHIFTS) |
              (tnzi_df.loc[idx, "shifts_N"] >= MIN_SHIFTS))
        ok_idx = idx[ok]
        if len(ok_idx) == 0: continue
        tnzi_df.loc[ok_idx, out_col] = norm_010(tnzi_df.loc[ok_idx, col])

# Drop tiny-sample rows
tnzi_df = tnzi_df[(tnzi_df["shifts_O"] + tnzi_df["shifts_D"] + tnzi_df["shifts_N"]) >= MIN_SHIFTS]
tnzi_cols = ["player_name","team","pos","GP","season","OZI","DZI","NZI","TNZI",
             "shifts_O","shifts_D","shifts_N"]
tnzi_df = tnzi_df[[c for c in tnzi_cols if c in tnzi_df.columns]].copy()

fwd = tnzi_df[tnzi_df["pos"] == "F"].sort_values(["season","TNZI"], ascending=[True, False], na_position="last")
dfn = tnzi_df[tnzi_df["pos"] == "D"].sort_values(["season","TNZI"], ascending=[True, False], na_position="last")
fwd.to_csv(OUT_TNZI / "tnzi_adjusted_forwards_playoffs.csv", index=False)
dfn.to_csv(OUT_TNZI / "tnzi_adjusted_defense_playoffs.csv", index=False)
print(f"    TNZI playoff: {len(fwd)} forward rows, {len(dfn)} defenseman rows "
      f"(across pooled + per-season views)")
for s_lbl, sub in tnzi_df.groupby("season"):
    print(f"      season={s_lbl}: {len(sub)} rows")

# ------------------------------------------------------------------
# 6. Build NFI playoff player_fully_adjusted_playoffs.csv (raw + ZA)
# ------------------------------------------------------------------
print("[6/6] computing NFI playoff (per-season + pooled, raw + ZA) ...")

# Build per-season records from the per-(pid, season) accumulator,
# AND build pooled records by aggregating across seasons.
def _make_row(pid, r, season_label, team_override=None):
    pos = pos_map.get(int(pid))
    if pos not in ("F","D"): return None
    if r["toi_sec"] < 60*30:  # ≥ 30 min ES TOI
        return None
    fo_total = r["fo_oz"] + r["fo_dz"]
    oz_ratio = (r["fo_oz"] / fo_total) if fo_total > 0 else 0.5
    nfi_pct = (r["cf_cm"] / (r["cf_cm"] + r["ca_cm"])) if (r["cf_cm"] + r["ca_cm"]) > 0 else None
    ff_pct  = (r["cf_fen"] / (r["cf_fen"] + r["ca_fen"])) if (r["cf_fen"] + r["ca_fen"]) > 0 else None
    if nfi_pct is None: return None
    nfi_za = nfi_pct - NFI_ZA_FACTOR * (oz_ratio - 0.5)
    team = team_override if team_override else r.get("team", "")
    return {
        "player_id": int(pid),
        "player_name": name_map.get(int(pid), ""),
        "position": pos, "team": team,
        "season": season_label,
        "toi_sec": r["toi_sec"], "toi_min": r["toi_sec"] / 60,
        "oz_ratio": oz_ratio,
        "NFI_pct": nfi_pct, "NFI_pct_ZA": nfi_za, "NFI_pct_3A": np.nan,
        "NFQOC": np.nan, "NFQOL": np.nan,
        "RelNFI_F_pct": np.nan, "RelNFI_A_pct": np.nan, "RelNFI_pct": np.nan,
        "NFI_pct_3A_MOM": np.nan,
        "FF_pct": ff_pct, "FF_pct_ZA": np.nan, "FF_pct_3A": np.nan,
        "CF_pct": None, "CF_pct_ZA": np.nan, "CF_pct_3A": np.nan,
    }

# Per-season rows
nfi_rows = []
for (pid, season), r in nfi.items():
    row = _make_row(pid, r, int(season))
    if row is not None:
        nfi_rows.append(row)

# Pooled rows: sum counters across seasons per player, use most-recent team
pooled_nfi = defaultdict(lambda: {"toi_sec":0.0, "cf_fen":0, "ca_fen":0,
                                    "cf_cm":0, "ca_cm":0, "fo_oz":0, "fo_dz":0,
                                    "team":""})
for (pid, season), r in nfi.items():
    p = pooled_nfi[pid]
    p["toi_sec"] += r["toi_sec"]
    for k in ("cf_fen","ca_fen","cf_cm","ca_cm","fo_oz","fo_dz"):
        p[k] += r[k]
    if r["team"]: p["team"] = r["team"]
for pid, r in pooled_nfi.items():
    row = _make_row(pid, r, "all_playoffs")
    if row is not None:
        nfi_rows.append(row)

nfi_df = pd.DataFrame(nfi_rows).sort_values(["season","NFI_pct_ZA"],
                                             ascending=[True, False]).reset_index(drop=True)
nfi_df.to_csv(OUT_NFI / "player_fully_adjusted_playoffs.csv", index=False)
print(f"    NFI playoff: {len(nfi_df)} player-rows (≥30 min ES TOI, ≥1 row per season + pooled)")
for s_lbl, sub in nfi_df.groupby("season"):
    print(f"      season={s_lbl}: {len(sub)} rows")

print("\n[done]")
print(f"    TNZI fwd:   {len(fwd):>4}")
print(f"    TNZI def:   {len(dfn):>4}")
print(f"    NFI total:  {len(nfi_df):>4}")
