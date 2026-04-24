"""compute_pqr_roc_rol.py
--------------------------
Reads the intermediate artefacts produced by compute_zone_and_overlap.py and
emits the final end-user tables under Player_Ranking/.

METHODOLOGY (unadjusted + teammate-adjusted):

  Raw per-player faceoff-shift rates (5v5 only, shift must start with the
  player on for the faceoff; line-change shifts ignored):
    player_OZ_retention  = OZ_sec / total_5v5_sec on OZ-faceoff shifts
    player_DZ_time       = DZ_sec / total_5v5_sec on DZ-faceoff shifts
    player_DZ_escape     = 1 - player_DZ_time
    player_NZ_retention  = NZ_sec / total_5v5_sec on NZ-faceoff shifts
  Require >= 50 faceoff shifts of the corresponding type to be defined.

  League averages (per position, across qualifying players: 50+50 FO shifts
  and 200+ min 5v5 TOI):
    league_avg_OZ_retention_{F,D}
    league_avg_DZ_escape_{F,D}
    league_avg_toi_per_game_{F,D}         (5v5 TOI minutes / games played)
    league_avg_NZ_retention_{F,D}          (reference only)

  Unadjusted (vs league average):
    OUER = player_OZ_retention / league_avg_OZ_retention   (position-aware)
    DUER = player_DZ_escape    / league_avg_DZ_escape      (position-aware)
    NZ_ratio = player_NZ_retention / league_avg_NZ_retention  (reference)

  Adjusted (vs own teammates, overlap-weighted by shared 5v5 ice seconds):
    teammates_avg_OZ_retention_P = sum_T secs(P,T) * player_OZ_retention(T)
                                   / sum_T secs(P,T)
    OAER = player_OZ_retention / teammates_avg_OZ_retention
    DAER = player_DZ_escape    / teammates_avg_DZ_escape      (analogously)

  Ratings:
    PUR_raw = (OUER + DUER) * (toi_per_game / league_avg_toi_per_game)
    PAR_raw = (OAER + DAER) * (toi_per_game / league_avg_toi_per_game)
    PUR = PUR_raw / max(PUR_raw in position group)
    PAR = PAR_raw / max(PAR_raw in position group)
    PFR = PUR / PAR
          < 1.0 -> self made (outperforms context)
          > 1.0 -> system supported
          ~ 1.0 -> consistent

  Context of ice time (same-position only; opponents weight by PUR or PAR):
    ROC      = overlap-weighted mean opponent PUR
    ROL      = overlap-weighted mean teammate PUR
    ROC_adj  = overlap-weighted mean opponent PAR
    ROL_adj  = overlap-weighted mean teammate PAR

  context_flag (PAR tercile within position x PFR direction):
    Elite          | PFR<0.95 -> "[green] Elite - self made"
                   | 0.95..1.05 -> "[green] Elite - consistent"
                   | PFR>1.05 -> "[green] Elite - system supported"
    Solid          | PFR<0.95 -> "[yellow] Solid - self made"
                   | 0.95..1.05 -> "[yellow] Neutral"
                   | PFR>1.05 -> "[yellow] Solid - system supported"
    Underperforming| PFR<0.95 -> "[red] Underperforming - tough situation"
                   | 0.95..1.05 -> "[red] Underperforming - neutral"
                   | PFR>1.05 -> "[red] Underperforming - no excuse"

Outputs (Player_Ranking/):

  Primary:
    pqr_forwards_pooled.csv        qualifying forwards, pooled 4 seasons
    pqr_forwards_regular.csv       qualifying forwards, current regular season
    pqr_forwards_playoffs.csv      qualifying forwards, current playoffs
    pqr_defense_pooled.csv         qualifying defensemen, pooled 4 seasons
    pqr_defense_regular.csv        qualifying defensemen, current regular season
    pqr_defense_playoffs.csv       qualifying defensemen, current playoffs

  Supporting (kept for compatibility; regenerated each run):
    zone_time_raw.csv              per-player faceoff-shift diagnostics
    zone_time_below_threshold.csv  players who fail the 200-min 5v5 TOI gate
    player_quality_scores.csv      combined F+D pooled reference
    roc_rol_pooled.csv             position-mixed back-compat
    roc_rol_regular.csv            position-mixed back-compat
    roc_rol_playoffs.csv           position-mixed back-compat
    last_updated_regular.txt       timestamp
    last_updated_playoffs.txt      timestamp
"""
from __future__ import annotations
import csv, json, os, pickle
from collections import defaultdict

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = ROOT
os.makedirs(OUT_DIR, exist_ok=True)

MIN_QUAL_SEC      = 200 * 60          # 200 minutes 5v5 TOI
MIN_OZ_FO_SHIFTS  = 50
MIN_DZ_FO_SHIFTS  = 50
MIN_NZ_FO_SHIFTS  = 50                 # reference only
CURRENT_SEASON    = "20252026"

SCENARIOS   = ("pooled", "current_regular", "current_playoffs")
FORWARD_POS = {"C", "L", "R"}
DEFENSE_POS = {"D"}

# --------------------------------------------------------------------------
def fmt_seasons(seasons):
    return ", ".join(f"{s[:4]}-{s[6:]}" for s in sorted(seasons))

def normalize_by_max(values):
    if not values: return {}
    hi = max(values.values())
    if hi <= 0: return {p: 0.0 for p in values}
    return {p: v / hi for p, v in values.items()}

def load_corsi_reference():
    path = os.path.join(OUT_DIR, "corsi_reference.csv")
    out = {}
    if not os.path.exists(path): return out
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                pid = int(row["player_id"])
            except ValueError:
                continue
            if row["cfpct_5v5"] == "": continue
            out[pid] = float(row["cfpct_5v5"])
    return out

def position_of(pid, player_meta):
    m = player_meta.get(str(pid))
    return m["position"] if m else ""

def context_flag_for(par_val, pfr_val, par_t1, par_t2):
    """Return emoji + plain-English tag based on PAR tercile x PFR direction."""
    if par_val is None or pfr_val is None:
        return ""
    # PAR tercile within position group
    if par_val >= par_t2: tier = "elite"
    elif par_val >= par_t1: tier = "solid"
    else: tier = "under"
    # PFR bucket
    if pfr_val < 0.95: side = "self"
    elif pfr_val > 1.05: side = "system"
    else: side = "mid"

    if tier == "elite":
        if side == "self":   return "\U0001F7E2 Elite \u2014 self made"
        if side == "system": return "\U0001F7E2 Elite \u2014 system supported"
        return                      "\U0001F7E2 Elite \u2014 consistent"
    if tier == "solid":
        if side == "self":   return "\U0001F7E1 Solid \u2014 self made"
        if side == "system": return "\U0001F7E1 Solid \u2014 system supported"
        return                      "\U0001F7E1 Neutral"
    # underperforming
    if side == "self":   return "\U0001F534 Underperforming \u2014 tough situation"
    if side == "system": return "\U0001F534 Underperforming \u2014 no excuse"
    return                      "\U0001F534 Underperforming \u2014 neutral"

# --------------------------------------------------------------------------
def compute_scenario(sc, zone_map, overlap_map, player_meta, corsi_ref):
    """Compute per-scenario per-player ratings (unadjusted + adjusted).

    Returns {"forwards": {qual, below, league_avg}, "defense": {...}}.
    """
    # ---- 1) raw per-player rates --------------------------------------------
    per_player = {}
    for pid_str, c in zone_map.items():
        pid = int(pid_str)
        gp  = c["games_played"] or 0
        if gp == 0 and c["toi_sec"] == 0: continue

        oz_ret = None
        if c["oz_fo_shifts"] >= MIN_OZ_FO_SHIFTS and c["oz_fo_5v5_sec"] > 0:
            oz_ret = c["oz_fo_oz_sec"] / c["oz_fo_5v5_sec"]
        dz_time = dz_esc = None
        if c["dz_fo_shifts"] >= MIN_DZ_FO_SHIFTS and c["dz_fo_5v5_sec"] > 0:
            dz_time = c["dz_fo_dz_sec"] / c["dz_fo_5v5_sec"]
            dz_esc  = 1.0 - dz_time
        nz_ret = None
        if c["nz_fo_shifts"] >= MIN_NZ_FO_SHIFTS and c["nz_fo_5v5_sec"] > 0:
            nz_ret = c["nz_fo_nz_sec"] / c["nz_fo_5v5_sec"]

        toi_min = c["toi_sec"] / 60.0
        per_player[pid] = {
            "toi_sec": c["toi_sec"], "games_played": gp,
            "team_id": c["team_id"], "team_abbrev": c["team_abbrev"],
            "oz_fo_shifts": c["oz_fo_shifts"],
            "dz_fo_shifts": c["dz_fo_shifts"],
            "nz_fo_shifts": c["nz_fo_shifts"],
            "oz_fo_5v5_sec": c["oz_fo_5v5_sec"],
            "dz_fo_5v5_sec": c["dz_fo_5v5_sec"],
            "nz_fo_5v5_sec": c["nz_fo_5v5_sec"],
            "oz_fo_oz_sec": c["oz_fo_oz_sec"],
            "dz_fo_dz_sec": c["dz_fo_dz_sec"],
            "nz_fo_nz_sec": c["nz_fo_nz_sec"],
            "player_OZ_retention": oz_ret,
            "player_DZ_time":      dz_time,
            "player_DZ_escape":    dz_esc,
            "player_NZ_retention": nz_ret,
            "toi_per_game_min":    (toi_min / gp) if gp > 0 else 0.0,
        }

    # ---- 2) position pools --------------------------------------------------
    def _by_pos(positions):
        return {pid: p for pid, p in per_player.items()
                if position_of(pid, player_meta) in positions}
    pool_f = _by_pos(FORWARD_POS)
    pool_d = _by_pos(DEFENSE_POS)

    # qualifying = 200-min TOI + both rates defined (for league-avg pool)
    def _qual(pool):
        return {pid: p for pid, p in pool.items()
                if p["toi_sec"] >= MIN_QUAL_SEC
                and p["player_OZ_retention"] is not None
                and p["player_DZ_escape"] is not None}
    qual_f_seed = _qual(pool_f)
    qual_d_seed = _qual(pool_d)

    # ---- 3) league averages (6 headline + 2 reference NZ) ------------------
    def _avg(pool, key):
        vals = [p[key] for p in pool.values() if p[key] is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    lf_oz,  nf_oz  = _avg(qual_f_seed, "player_OZ_retention")
    lf_dze, nf_dz  = _avg(qual_f_seed, "player_DZ_escape")
    lf_tpg, nf_tg  = _avg(qual_f_seed, "toi_per_game_min")
    ld_oz,  nd_oz  = _avg(qual_d_seed, "player_OZ_retention")
    ld_dze, nd_dz  = _avg(qual_d_seed, "player_DZ_escape")
    ld_tpg, nd_tg  = _avg(qual_d_seed, "toi_per_game_min")
    lf_nz,  nf_nz  = _avg(pool_f, "player_NZ_retention")
    ld_nz,  nd_nz  = _avg(pool_d, "player_NZ_retention")

    def _ln(label, val, n):
        v = f"{val:.4f}" if isinstance(val, float) else "(no data)"
        print(f"  {label:<48} {v}   (n={n})")
    print(f"\n  === league averages for [{sc}] ===")
    _ln("FORWARDS    league_avg_OZ_retention",   lf_oz,  nf_oz)
    _ln("FORWARDS    league_avg_DZ_escape",      lf_dze, nf_dz)
    _ln("FORWARDS    league_avg_toi_per_game",   lf_tpg, nf_tg)
    _ln("DEFENSEMEN  league_avg_OZ_retention",   ld_oz,  nd_oz)
    _ln("DEFENSEMEN  league_avg_DZ_escape",      ld_dze, nd_dz)
    _ln("DEFENSEMEN  league_avg_toi_per_game",   ld_tpg, nd_tg)
    _ln("(ref) FORWARDS   league_avg_NZ_retention",   lf_nz, nf_nz)
    _ln("(ref) DEFENSEMEN league_avg_NZ_retention",   ld_nz, nd_nz)
    print()

    league_f = {"oz_retention": lf_oz, "dz_escape": lf_dze,
                "nz_retention": lf_nz, "toi_per_game_min": lf_tpg}
    league_d = {"oz_retention": ld_oz, "dz_escape": ld_dze,
                "nz_retention": ld_nz, "toi_per_game_min": ld_tpg}

    # ---- 4) unadjusted ratings OUER / DUER / NZ_ratio ---------------------
    def _apply_unadj(pool, avg):
        for p in pool.values():
            oz_r = p["player_OZ_retention"]; dz_e = p["player_DZ_escape"]
            nz_r = p["player_NZ_retention"]
            p["OUER"]     = (oz_r / avg["oz_retention"]) if (oz_r is not None and avg["oz_retention"]) else None
            p["DUER"]     = (dz_e / avg["dz_escape"])    if (dz_e is not None and avg["dz_escape"])    else None
            p["NZ_ratio"] = (nz_r / avg["nz_retention"]) if (nz_r is not None and avg["nz_retention"]) else None
    _apply_unadj(pool_f, league_f)
    _apply_unadj(pool_d, league_d)

    # ---- 5) adjusted ratings OAER / DAER via overlap-weighted teammates ---
    tm = overlap_map.get("teammate", {})
    tm_by_pid = defaultdict(list)
    for (p1, p2), secs in tm.items():
        tm_by_pid[p1].append((p2, secs))
        tm_by_pid[p2].append((p1, secs))

    def _apply_adj(pool):
        for pid, p in pool.items():
            # teammates_avg_OZ_retention (weighted by shared-ice secs)
            sw, sv = 0.0, 0.0
            for other, secs in tm_by_pid.get(pid, []):
                op = per_player.get(other)
                if op is None or op["player_OZ_retention"] is None: continue
                sw += secs; sv += secs * op["player_OZ_retention"]
            p["teammates_avg_OZ_retention"] = (sv / sw) if sw > 0 else None
            if (p["player_OZ_retention"] is not None
                and p["teammates_avg_OZ_retention"] is not None
                and p["teammates_avg_OZ_retention"] > 0):
                p["OAER"] = p["player_OZ_retention"] / p["teammates_avg_OZ_retention"]
            else:
                p["OAER"] = None
            # teammates_avg_DZ_escape
            sw, sv = 0.0, 0.0
            for other, secs in tm_by_pid.get(pid, []):
                op = per_player.get(other)
                if op is None or op["player_DZ_escape"] is None: continue
                sw += secs; sv += secs * op["player_DZ_escape"]
            p["teammates_avg_DZ_escape"] = (sv / sw) if sw > 0 else None
            if (p["player_DZ_escape"] is not None
                and p["teammates_avg_DZ_escape"] is not None
                and p["teammates_avg_DZ_escape"] > 0):
                p["DAER"] = p["player_DZ_escape"] / p["teammates_avg_DZ_escape"]
            else:
                p["DAER"] = None

    _apply_adj(pool_f); _apply_adj(pool_d)

    # ---- 6) qualifying pool = 200 min + OUER + DUER defined ---------------
    def _split(pool):
        qual, below = {}, {}
        for pid, p in pool.items():
            if (p["toi_sec"] >= MIN_QUAL_SEC
                and p["OUER"] is not None and p["DUER"] is not None):
                qual[pid] = p
            else:
                below[pid] = p
        return qual, below
    qual_f, below_f = _split(pool_f)
    qual_d, below_d = _split(pool_d)
    print(f"  [{sc}] forwards    qual={len(qual_f)}  below={len(below_f)}")
    print(f"  [{sc}] defensemen  qual={len(qual_d)}  below={len(below_d)}")

    # ---- 7) PUR, PAR, PFR --------------------------------------------------
    def _pur_raw(qual, lg_tpg):
        out = {}
        if not lg_tpg: return out
        for pid, p in qual.items():
            out[pid] = (p["OUER"] + p["DUER"]) * (p["toi_per_game_min"] / lg_tpg)
        return out
    def _par_raw(qual, lg_tpg):
        out = {}
        if not lg_tpg: return out
        for pid, p in qual.items():
            if p["OAER"] is None or p["DAER"] is None: continue
            out[pid] = (p["OAER"] + p["DAER"]) * (p["toi_per_game_min"] / lg_tpg)
        return out

    pur_raw_f = _pur_raw(qual_f, lf_tpg); pur_raw_d = _pur_raw(qual_d, ld_tpg)
    par_raw_f = _par_raw(qual_f, lf_tpg); par_raw_d = _par_raw(qual_d, ld_tpg)
    pur_f = normalize_by_max(pur_raw_f); pur_d = normalize_by_max(pur_raw_d)
    par_f = normalize_by_max(par_raw_f); par_d = normalize_by_max(par_raw_d)

    for pid, p in qual_f.items():
        p["PUR_raw"] = pur_raw_f.get(pid); p["PUR"] = pur_f.get(pid)
        p["PAR_raw"] = par_raw_f.get(pid); p["PAR"] = par_f.get(pid)
        p["PFR"] = (p["PUR"] / p["PAR"]) if (p["PUR"] is not None and p["PAR"] not in (None, 0, 0.0)) else None
    for pid, p in qual_d.items():
        p["PUR_raw"] = pur_raw_d.get(pid); p["PUR"] = pur_d.get(pid)
        p["PAR_raw"] = par_raw_d.get(pid); p["PAR"] = par_d.get(pid)
        p["PFR"] = (p["PUR"] / p["PAR"]) if (p["PUR"] is not None and p["PAR"] not in (None, 0, 0.0)) else None
    for pool in (below_f, below_d):
        for p in pool.values():
            p["PUR_raw"] = None; p["PUR"] = None
            p["PAR_raw"] = None; p["PAR"] = None; p["PFR"] = None

    # ---- 8) ROC / ROL / ROC_adj / ROL_adj (same position only) ------------
    op_map = overlap_map.get("opponent", {})

    def _roc_rol(qual_group, pur_map, par_map):
        sw_tm_u = defaultdict(float); swq_tm_u = defaultdict(float)
        sw_op_u = defaultdict(float); swq_op_u = defaultdict(float)
        sw_tm_a = defaultdict(float); swq_tm_a = defaultdict(float)
        sw_op_a = defaultdict(float); swq_op_a = defaultdict(float)

        for (p1, p2), secs in tm.items():
            if p1 in qual_group:
                if p2 in pur_map:
                    sw_tm_u[p1] += secs; swq_tm_u[p1] += secs * pur_map[p2]
                if p2 in par_map:
                    sw_tm_a[p1] += secs; swq_tm_a[p1] += secs * par_map[p2]
            if p2 in qual_group:
                if p1 in pur_map:
                    sw_tm_u[p2] += secs; swq_tm_u[p2] += secs * pur_map[p1]
                if p1 in par_map:
                    sw_tm_a[p2] += secs; swq_tm_a[p2] += secs * par_map[p1]
        for (p1, p2), secs in op_map.items():
            if p1 in qual_group:
                if p2 in pur_map:
                    sw_op_u[p1] += secs; swq_op_u[p1] += secs * pur_map[p2]
                if p2 in par_map:
                    sw_op_a[p1] += secs; swq_op_a[p1] += secs * par_map[p2]
            if p2 in qual_group:
                if p1 in pur_map:
                    sw_op_u[p2] += secs; swq_op_u[p2] += secs * pur_map[p1]
                if p1 in par_map:
                    sw_op_a[p2] += secs; swq_op_a[p2] += secs * par_map[p1]

        for pid, p in qual_group.items():
            p["ROL"]     = (swq_tm_u[pid] / sw_tm_u[pid]) if sw_tm_u[pid] > 0 else None
            p["ROC"]     = (swq_op_u[pid] / sw_op_u[pid]) if sw_op_u[pid] > 0 else None
            p["ROL_adj"] = (swq_tm_a[pid] / sw_tm_a[pid]) if sw_tm_a[pid] > 0 else None
            p["ROC_adj"] = (swq_op_a[pid] / sw_op_a[pid]) if sw_op_a[pid] > 0 else None
            p["ROL_weight_sec"] = int(sw_tm_u[pid])
            p["ROC_weight_sec"] = int(sw_op_u[pid])

    _roc_rol(qual_f, pur_f, par_f)
    _roc_rol(qual_d, pur_d, par_d)

    # ---- 9) context_flag based on PAR tercile + PFR direction --------------
    def _tercile_cuts(values):
        s = sorted(values)
        if len(s) < 3: return (0.0, 0.0)
        return (s[len(s) // 3], s[(2 * len(s)) // 3])

    par_vals_f = [p["PAR"] for p in qual_f.values() if p["PAR"] is not None]
    par_vals_d = [p["PAR"] for p in qual_d.values() if p["PAR"] is not None]
    t1_f, t2_f = _tercile_cuts(par_vals_f)
    t1_d, t2_d = _tercile_cuts(par_vals_d)

    for p in qual_f.values():
        p["context_flag"] = context_flag_for(p.get("PAR"), p.get("PFR"), t1_f, t2_f)
    for p in qual_d.values():
        p["context_flag"] = context_flag_for(p.get("PAR"), p.get("PFR"), t1_d, t2_d)
    for pool in (below_f, below_d):
        for p in pool.values():
            p["context_flag"] = ""

    return {
        "forwards": {"qual": qual_f, "below": below_f, "league_avg": league_f,
                     "par_tercile": (t1_f, t2_f)},
        "defense":  {"qual": qual_d, "below": below_d, "league_avg": league_d,
                     "par_tercile": (t1_d, t2_d)},
    }

# --------------------------------------------------------------------------
# Output schema - EXACT order from the brief, then back-compat columns after
FIELDS = [
    "player_name", "team", "pos",
    "PFR", "PUR", "PAR",
    "games_played", "total_5v5_toi_minutes", "toi_per_game",
    "oz_faceoff_shifts", "oz_faceoff_shifts_per_game",
    "dz_faceoff_shifts", "dz_faceoff_shifts_per_game",
    "player_OZ_retention_pct", "player_DZ_escape_rate_pct",
    "OUER", "DUER", "OAER", "DAER",
    "NZ_ratio",
    "ROC", "ROL", "ROC_adj", "ROL_adj",
    "context_flag",
    "CF_pct",
    # back-compat extras
    "player_id", "seasons_covered",
    "player_DZ_time_pct",
    "PUR_raw", "PAR_raw",
    "teammates_avg_OZ_retention", "teammates_avg_DZ_escape",
    "ROC_weight_sec", "ROL_weight_sec",
]

def _fmt(v, n=4):
    if v is None or v == "": return ""
    if isinstance(v, float): return round(v, n)
    return v

def _pct(x, n=2):
    if x is None or x == "": return ""
    return round(x * 100, n)

def row_for(pid, rec, meta, corsi_ref):
    gp = rec["games_played"] or 0
    toi_min = rec["toi_sec"] / 60.0
    toi_per_game = (toi_min / gp) if gp > 0 else 0.0
    oz_per_game  = (rec["oz_fo_shifts"] / gp) if gp > 0 else 0.0
    dz_per_game  = (rec["dz_fo_shifts"] / gp) if gp > 0 else 0.0
    return {
        "player_id": pid,
        "player_name": meta.get(str(pid), {}).get("name", ""),
        "pos": meta.get(str(pid), {}).get("position", ""),
        "team": rec.get("team_abbrev") or meta.get(str(pid), {}).get("team_abbrev", ""),
        "seasons_covered": fmt_seasons(meta.get(str(pid), {}).get("seasons", [])),
        "games_played": gp,
        "total_5v5_toi_minutes": round(toi_min, 2),
        "toi_per_game": round(toi_per_game, 3),
        "oz_faceoff_shifts": rec["oz_fo_shifts"],
        "oz_faceoff_shifts_per_game": round(oz_per_game, 3),
        "dz_faceoff_shifts": rec["dz_fo_shifts"],
        "dz_faceoff_shifts_per_game": round(dz_per_game, 3),
        "player_OZ_retention_pct":   _pct(rec.get("player_OZ_retention")),
        "player_DZ_time_pct":        _pct(rec.get("player_DZ_time")),
        "player_DZ_escape_rate_pct": _pct(rec.get("player_DZ_escape")),
        "OUER":      _fmt(rec.get("OUER")),
        "DUER":      _fmt(rec.get("DUER")),
        "OAER":      _fmt(rec.get("OAER")),
        "DAER":      _fmt(rec.get("DAER")),
        "NZ_ratio":  _fmt(rec.get("NZ_ratio")),
        "PUR_raw":   _fmt(rec.get("PUR_raw")),
        "PAR_raw":   _fmt(rec.get("PAR_raw")),
        "PUR":       _fmt(rec.get("PUR")),
        "PAR":       _fmt(rec.get("PAR")),
        "PFR":       _fmt(rec.get("PFR")),
        "ROC":       _fmt(rec.get("ROC")),
        "ROL":       _fmt(rec.get("ROL")),
        "ROC_adj":   _fmt(rec.get("ROC_adj")),
        "ROL_adj":   _fmt(rec.get("ROL_adj")),
        "teammates_avg_OZ_retention": _fmt(rec.get("teammates_avg_OZ_retention")),
        "teammates_avg_DZ_escape":    _fmt(rec.get("teammates_avg_DZ_escape")),
        "context_flag": rec.get("context_flag", ""),
        "ROC_weight_sec": rec.get("ROC_weight_sec", ""),
        "ROL_weight_sec": rec.get("ROL_weight_sec", ""),
        "CF_pct": round(corsi_ref[pid], 4) if pid in corsi_ref else "",
    }

def write_csv(path, rows, fields=FIELDS):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)

ZONE_RAW_FIELDS = [
    "player_id", "player_name", "pos", "team", "seasons_covered",
    "scenario", "games_played",
    "total_5v5_toi_minutes", "toi_per_game",
    "oz_faceoff_shifts", "oz_faceoff_shifts_per_game",
    "dz_faceoff_shifts", "dz_faceoff_shifts_per_game",
    "nz_faceoff_shifts",
    "player_OZ_retention_pct",
    "player_DZ_time_pct", "player_DZ_escape_rate_pct",
    "OUER", "DUER", "OAER", "DAER", "NZ_ratio",
]

def zone_raw_row(pid, rec, meta, sc):
    gp = rec["games_played"] or 0
    toi_min = rec["toi_sec"] / 60.0
    return {
        "player_id": pid,
        "player_name": meta.get(str(pid), {}).get("name", ""),
        "pos": meta.get(str(pid), {}).get("position", ""),
        "team": rec.get("team_abbrev") or meta.get(str(pid), {}).get("team_abbrev", ""),
        "seasons_covered": fmt_seasons(meta.get(str(pid), {}).get("seasons", [])),
        "scenario": sc,
        "games_played": gp,
        "total_5v5_toi_minutes": round(toi_min, 2),
        "toi_per_game": round((toi_min / gp) if gp else 0.0, 3),
        "oz_faceoff_shifts": rec["oz_fo_shifts"],
        "oz_faceoff_shifts_per_game": round((rec["oz_fo_shifts"] / gp) if gp else 0.0, 3),
        "dz_faceoff_shifts": rec["dz_fo_shifts"],
        "dz_faceoff_shifts_per_game": round((rec["dz_fo_shifts"] / gp) if gp else 0.0, 3),
        "nz_faceoff_shifts": rec["nz_fo_shifts"],
        "player_OZ_retention_pct":   _pct(rec.get("player_OZ_retention")),
        "player_DZ_time_pct":        _pct(rec.get("player_DZ_time")),
        "player_DZ_escape_rate_pct": _pct(rec.get("player_DZ_escape")),
        "OUER":      _fmt(rec.get("OUER")),
        "DUER":      _fmt(rec.get("DUER")),
        "OAER":      _fmt(rec.get("OAER")),
        "DAER":      _fmt(rec.get("DAER")),
        "NZ_ratio":  _fmt(rec.get("NZ_ratio")),
    }

# --------------------------------------------------------------------------
def main():
    meta       = json.load(open(os.path.join(OUT_DIR, "_player_meta.json")))
    zone_in    = json.load(open(os.path.join(OUT_DIR, "_zone_time.json")))
    overlap_in = pickle.load(open(os.path.join(OUT_DIR, "_overlap.pkl"), "rb"))
    corsi_ref  = load_corsi_reference()
    print(f"[load] players={len(meta)}  corsi_ref={len(corsi_ref)}")

    results = {}
    for sc in SCENARIOS:
        print(f"[{sc}] computing ...")
        results[sc] = compute_scenario(
            sc, zone_in.get(sc, {}),
            overlap_in.get(sc, {"teammate": {}, "opponent": {}}),
            meta, corsi_ref,
        )

    # primary outputs - six position-split files sorted by PFR ascending
    # (PFR < 1.0 first = self-made headliners; but the spec says PFR default
    # sort in app. In CSV we sort by PFR ascending so headers read intuitively.)
    group_file = {
        ("forwards", "pooled"):           "pqr_forwards_pooled.csv",
        ("forwards", "current_regular"):  "pqr_forwards_regular.csv",
        ("forwards", "current_playoffs"): "pqr_forwards_playoffs.csv",
        ("defense",  "pooled"):           "pqr_defense_pooled.csv",
        ("defense",  "current_regular"):  "pqr_defense_regular.csv",
        ("defense",  "current_playoffs"): "pqr_defense_playoffs.csv",
    }
    for (group, sc), fname in group_file.items():
        qual = results[sc][group]["qual"]
        rows = [row_for(pid, rec, meta, corsi_ref) for pid, rec in qual.items()]
        # default sort = PFR ascending (most self-made first)
        rows.sort(key=lambda r: (r["PFR"] if isinstance(r["PFR"], float) else 9.99))
        write_csv(os.path.join(OUT_DIR, fname), rows)
        print(f"[write] {fname} ({len(rows)})")

    # zone_time_raw + below-threshold
    pool = results["pooled"]
    qual_all  = {**pool["forwards"]["qual"],  **pool["defense"]["qual"]}
    below_all = {**pool["forwards"]["below"], **pool["defense"]["below"]}
    raw_rows = [zone_raw_row(pid, rec, meta, "pooled") for pid, rec in qual_all.items()]
    raw_rows.sort(key=lambda r: -r["total_5v5_toi_minutes"])
    write_csv(os.path.join(OUT_DIR, "zone_time_raw.csv"), raw_rows, ZONE_RAW_FIELDS)
    print(f"[write] zone_time_raw.csv ({len(raw_rows)})")
    below_rows = [zone_raw_row(pid, rec, meta, "pooled") for pid, rec in below_all.items()]
    below_rows.sort(key=lambda r: -r["total_5v5_toi_minutes"])
    write_csv(os.path.join(OUT_DIR, "zone_time_below_threshold.csv"), below_rows, ZONE_RAW_FIELDS)
    print(f"[write] zone_time_below_threshold.csv ({len(below_rows)})")

    # combined F+D quality-score reference
    pqs_fields = [
        "player_id", "player_name", "pos", "team", "seasons_covered",
        "games_played", "total_5v5_toi_minutes", "toi_per_game",
        "oz_faceoff_shifts", "oz_faceoff_shifts_per_game",
        "dz_faceoff_shifts", "dz_faceoff_shifts_per_game",
        "player_OZ_retention_pct",
        "player_DZ_time_pct", "player_DZ_escape_rate_pct",
        "OUER", "DUER", "OAER", "DAER", "NZ_ratio",
        "PUR_raw", "PAR_raw", "PUR", "PAR", "PFR",
        "context_flag",
    ]
    pqs_rows = [row_for(pid, rec, meta, corsi_ref) for pid, rec in qual_all.items()]
    pqs_rows.sort(key=lambda r: -(r["PUR"] if isinstance(r["PUR"], float) else -1))
    write_csv(os.path.join(OUT_DIR, "player_quality_scores.csv"),
              [{k: r[k] for k in pqs_fields} for r in pqs_rows], pqs_fields)
    print(f"[write] player_quality_scores.csv ({len(pqs_rows)})")

    # back-compat position-mixed roc_rol_*
    out_files = {
        "pooled":           "roc_rol_pooled.csv",
        "current_regular":  "roc_rol_regular.csv",
        "current_playoffs": "roc_rol_playoffs.csv",
    }
    for sc, fname in out_files.items():
        combined = {**results[sc]["forwards"]["qual"], **results[sc]["defense"]["qual"]}
        rows = [row_for(pid, rec, meta, corsi_ref) for pid, rec in combined.items()]
        rows.sort(key=lambda r: (r["PFR"] if isinstance(r["PFR"], float) else 9.99))
        write_csv(os.path.join(OUT_DIR, fname), rows)
        print(f"[write] {fname} ({len(rows)})  [back-compat, position-mixed]")

    import datetime as _dt
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(os.path.join(OUT_DIR, "last_updated_regular.txt"),  "w") as f: f.write(ts)
    with open(os.path.join(OUT_DIR, "last_updated_playoffs.txt"), "w") as f: f.write(ts)
    print(f"[write] last_updated_{{regular,playoffs}}.txt = {ts}")

if __name__ == "__main__":
    main()
