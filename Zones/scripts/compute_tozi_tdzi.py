"""TOZI / TDZI — two new transition zone metrics.

TOZI: on OZ faceoff shifts, Wilson lower OZ% − Wilson upper DZ%
TDZI: on DZ faceoff shifts, Wilson lower OZ% − Wilson upper DZ%

Mirrors methodology in compute_iozc_iozl_dozi.py exactly:
  - V1 event filter (faceoff/hit/shot-on-goal/missed-shot/blocked-shot/goal/giveaway/takeaway)
  - x>25 / -25..25 / x<-25 zone mapping done via play details.zoneCode
  - 5v5 only (situationCode 1551)
  - faceoff-shift identification by typeDescKey == faceoff
  - Wilson z=1.96, MIN_SHIFTS=50, MIN_GP=20
  - Linemate adjustment via team-level OLS β_IOZL (single-L), same form as TNZI_L
"""
from __future__ import annotations

import csv, json, math, pickle, subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW_PBP    = ROOT / "raw" / "pbp"
RAW_SHIFTS = ROOT / "raw" / "shifts"
GAME_IDS   = ROOT.parent / "Data" / "game_ids.csv"
OUT_DIR    = ROOT / "adjusted_rankings"
PLAYER_META = ROOT / "output" / "_player_meta.json"
OVERLAP_PKL = ROOT / "output" / "_overlap.pkl"

Z = 1.96; Z2 = Z * Z
MIN_SHIFTS = 50
MIN_GP = 20
SEASONS = ["20222023", "20232024", "20242025", "20252026"]
POOLED = "pooled"
SCENARIOS = SEASONS + [POOLED]
CURRENT = "20252026"
SEASON_END = {"20222023": "2023-04-13", "20232024": "2024-04-18",
              "20242025": "2025-04-17", "20252026": "2026-04-17"}
V1_EVENTS = {"faceoff","hit","shot-on-goal","missed-shot","blocked-shot",
             "goal","giveaway","takeaway"}
POS_F = {"C","L","R"}; POS_D = {"D"}
FLIP = {"O":"D","D":"O","N":"N"}
ABBR_MAP = {"ARI":"UTA"}
NEW_METRICS = ["TOZI", "TDZI"]
ALL_METRICS = ["OZI","DZI","NZI","TNZI","TOZI","TDZI"]

KEY_PLAYERS = [
    ("McDavid",   8478402),("MacKinnon", 8477492),("Draisaitl", 8477934),
    ("Makar",     8480069),("Hughes (Q)",8480800),("Nurse",     8477498),
    ("Bouchard",  8480803),("Ekholm",    8475218),("Bedard",    8484144),
]

# ---------- helpers ----------
def mmss(s):
    if not s or ":" not in s: return 0
    try: m,se=s.split(":"); return int(m)*60+int(se)
    except ValueError: return 0
def play_t(p):
    pd=(p.get("periodDescriptor") or {}).get("number",1) or 1
    return (pd-1)*1200 + mmss(p.get("timeInPeriod","00:00"))
def norm_team(a): return ABBR_MAP.get(a,a)
def wilson_lower(p,n):
    if n is None or n<=0: return None
    p=max(0.0,min(1.0,p))
    d=1.0+Z2/n; c=p+Z2/(2*n); m=Z*math.sqrt(max(0,p*(1-p)/n+Z2/(4*n*n)))
    return (c-m)/d
def wilson_upper(p,n):
    if n is None or n<=0: return None
    p=max(0.0,min(1.0,p))
    d=1.0+Z2/n; c=p+Z2/(2*n); m=Z*math.sqrt(max(0,p*(1-p)/n+Z2/(4*n*n)))
    return (c+m)/d
def pearson(x,y):
    n=len(x)
    if n<2: return float("nan")
    mx,my=sum(x)/n,sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y))
    sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sxx*syy) if sxx>0 and syy>0 else float("nan")
def normalize01(pid_vals):
    vs=[v for _,v in pid_vals if v is not None]
    if not vs: return {p:None for p,_ in pid_vals}
    lo,hi=min(vs),max(vs); span=hi-lo; out={}
    for p,v in pid_vals:
        out[p]=None if v is None else (0.5 if span==0 else (v-lo)/span)
    return out
def score10(pid_vals):
    n=normalize01(pid_vals)
    return {p:(None if v is None else round(v*10,1)) for p,v in n.items()}

# ---------- standings ----------
def fetch_standings(date_str):
    d0=datetime.strptime(date_str,"%Y-%m-%d")
    for off in range(0,5):
        d=(d0-timedelta(days=off)).strftime("%Y-%m-%d")
        out=subprocess.run(["curl","-s","-m","20",
             f"https://api-web.nhle.com/v1/standings/{d}"],
             capture_output=True,text=True)
        if out.returncode!=0: continue
        try: data=json.loads(out.stdout)
        except json.JSONDecodeError: continue
        rows=data.get("standings",[])
        if rows:
            res={}
            for r in rows:
                a=norm_team(r["teamAbbrev"]["default"])
                res[a]={"points":r["points"],"gp":r["gamesPlayed"]}
            if d!=date_str: print(f"    note: {date_str} empty; using {d}")
            return res
    return {}

# ---------- raw PBP processing — slim version ----------
def build_intervals(shifts_json):
    out=defaultdict(list)
    for s in shifts_json.get("data",[]):
        pid=s.get("playerId")
        if not pid: continue
        per=s.get("period") or 1
        a=(per-1)*1200+mmss(s.get("startTime") or "00:00")
        b=(per-1)*1200+mmss(s.get("endTime") or "00:00")
        if b<=a: continue
        out[pid].append((a,b,s.get("teamId")))
    return out
def on_ice(intervals,t):
    out=defaultdict(set)
    for pid,ivs in intervals.items():
        for a,b,tm in ivs:
            if a<=t<b: out[tm].add(pid); break
    return out
def shift_end(intervals,pid,t):
    for a,b,_ in intervals.get(pid,[]):
        if a<=t<b: return b
    return None

def process_game(gid,player_season_gp,bucket):
    pp=RAW_PBP/f"{gid}.json"; sp=RAW_SHIFTS/f"{gid}.json"
    if not pp.exists() or not sp.exists(): return
    try:
        pbp=json.load(open(pp)); sj=json.load(open(sp))
    except (json.JSONDecodeError,OSError): return
    season=str(pbp.get("season",""))
    home=(pbp.get("homeTeam") or {}).get("id"); away=(pbp.get("awayTeam") or {}).get("id")
    if home is None or away is None: return
    intervals=build_intervals(sj)
    for pid in intervals: player_season_gp[(pid,season)]+=1
    plays=pbp.get("plays") or []; ctx=None

    def emit(ctx,close_t):
        if ctx is None or not ctx["events"]: return
        fo_home=ctx["fo_zone_home"]
        for tid,pids in ctx["players_at_fo"].items():
            if tid not in (home,away): continue
            fz=fo_home if tid==home else FLIP[fo_home]
            for pid in pids:
                se=shift_end(intervals,pid,ctx["fo_t"])
                if se is None: continue
                eff=min(close_t,se)
                if eff<=ctx["fo_t"]: continue
                f=[(t,ez) for (t,ez,_) in ctx["events"] if ctx["fo_t"]<=t<eff]
                if not f: continue
                key=(pid,season,fz)
                b=bucket[key]
                b["shifts"]+=1
                for i,(t,ez) in enumerate(f):
                    tn=f[i+1][0] if i+1<len(f) else eff
                    dt=max(0.0,tn-t)
                    ezp=ez if tid==home else FLIP[ez]
                    b["total_sec"]+=dt
                    if ezp=="O": b["oz_sec"]+=dt
                    elif ezp=="D": b["dz_sec"]+=dt
                    else: b["nz_sec"]+=dt
    for p in plays:
        typ=p.get("typeDescKey") or ""; det=p.get("details") or {}
        ta=play_t(p); sit=p.get("situationCode") or ""
        if typ=="faceoff":
            if ctx is not None: emit(ctx,ta); ctx=None
            if sit!="1551": continue
            zc=det.get("zoneCode")
            if zc not in ("O","D","N"): continue
            owner=det.get("eventOwnerTeamId")
            if owner not in (home,away): continue
            fh=zc if owner==home else FLIP[zc]
            oi=on_ice(intervals,ta)
            ctx={"fo_t":ta,"fo_zone_home":fh,
                 "events":[(ta,fh,"faceoff")],
                 "players_at_fo":{home:set(oi.get(home,set())),
                                  away:set(oi.get(away,set()))}}
            continue
        if ctx is None: continue
        if sit and sit!="1551": emit(ctx,ta); ctx=None; continue
        zc=det.get("zoneCode")
        if zc in ("O","D","N") and typ in V1_EVENTS:
            owner=det.get("eventOwnerTeamId")
            if owner in (home,away):
                ezh=zc if owner==home else FLIP[zc]
                ctx["events"].append((ta,ezh,typ))
        if typ in ("period-end","game-end"): emit(ctx,ta); ctx=None
    if ctx is not None:
        last=play_t(plays[-1]) if plays else ctx["fo_t"]
        emit(ctx,last)

# ---------- metric computation per bucket-set ----------
def compute_metrics(b_o,b_d,b_n):
    def get(b): return b if b else {"shifts":0,"total_sec":0.0,"oz_sec":0.0,"dz_sec":0.0,"nz_sec":0.0}
    bo=get(b_o); bd=get(b_d); bn=get(b_n)
    res={}
    # OZI
    if bo["shifts"]>=MIN_SHIFTS and bo["total_sec"]>0:
        p=bo["oz_sec"]/bo["total_sec"]; res["OZI"]=(p,wilson_lower(p,bo["shifts"]),True)
    else: res["OZI"]=(None,None,False)
    # DZI
    if bd["shifts"]>=MIN_SHIFTS and bd["total_sec"]>0:
        p=bd["oz_sec"]/bd["total_sec"]; res["DZI"]=(p,wilson_lower(p,bd["shifts"]),True)
    else: res["DZI"]=(None,None,False)
    # NZI
    if bn["shifts"]>=MIN_SHIFTS and bn["total_sec"]>0:
        p=bn["oz_sec"]/bn["total_sec"]; res["NZI"]=(p,wilson_lower(p,bn["shifts"]),True)
    else: res["NZI"]=(None,None,False)
    # TNZI: NZ FO, lower OZ% - upper DZ%
    if bn["shifts"]>=MIN_SHIFTS and bn["total_sec"]>0:
        po=bn["oz_sec"]/bn["total_sec"]; pd_=bn["dz_sec"]/bn["total_sec"]
        lo=wilson_lower(po,bn["shifts"]); hi=wilson_upper(pd_,bn["shifts"])
        if lo is not None and hi is not None: res["TNZI"]=(po-pd_,lo-hi,True)
        else: res["TNZI"]=(None,None,False)
    else: res["TNZI"]=(None,None,False)
    # TOZI: OZ FO, lower OZ% - upper DZ%
    if bo["shifts"]>=MIN_SHIFTS and bo["total_sec"]>0:
        po=bo["oz_sec"]/bo["total_sec"]; pd_=bo["dz_sec"]/bo["total_sec"]
        lo=wilson_lower(po,bo["shifts"]); hi=wilson_upper(pd_,bo["shifts"])
        if lo is not None and hi is not None: res["TOZI"]=(po-pd_,lo-hi,True)
        else: res["TOZI"]=(None,None,False)
    else: res["TOZI"]=(None,None,False)
    # TDZI: DZ FO, lower OZ% - upper DZ%
    if bd["shifts"]>=MIN_SHIFTS and bd["total_sec"]>0:
        po=bd["oz_sec"]/bd["total_sec"]; pd_=bd["dz_sec"]/bd["total_sec"]
        lo=wilson_lower(po,bd["shifts"]); hi=wilson_upper(pd_,bd["shifts"])
        if lo is not None and hi is not None: res["TDZI"]=(po-pd_,lo-hi,True)
        else: res["TDZI"]=(None,None,False)
    else: res["TDZI"]=(None,None,False)
    return res

# ---------- main ----------
def main():
    OUT_DIR.mkdir(parents=True,exist_ok=True)

    print("[1/8] standings ...")
    standings={s:fetch_standings(SEASON_END[s]) for s in SEASONS}
    for s in SEASONS: print(f"    {s}: {len(standings[s])} teams")

    print("[2/8] meta + overlap ...")
    player_meta={int(k):v for k,v in json.load(open(PLAYER_META)).items()}
    overlap=pickle.load(open(OVERLAP_PKL,"rb"))
    teammate_map=overlap.get("pooled",{}).get("teammate",{})
    print(f"    meta:{len(player_meta)} teammate-pairs:{len(teammate_map)}")

    print("[3/8] game_ids ...")
    games=[]
    with open(GAME_IDS) as f:
        for r in csv.DictReader(f):
            if r["season"] in set(SEASONS) and r["game_type"]=="regular":
                games.append((int(r["game_id"]),r["season"]))
    print(f"    {len(games)} regular games")

    print("[4/8] PBP pass ...")
    bucket=defaultdict(lambda:{"shifts":0,"total_sec":0.0,"oz_sec":0.0,"dz_sec":0.0,"nz_sec":0.0})
    gp=defaultdict(int)
    n=len(games)
    for i,(gid,_) in enumerate(games):
        process_game(gid,gp,bucket)
        if (i+1)%500==0 or i+1==n: print(f"    {i+1}/{n}")

    print("[5/8] aggregate scenarios ...")
    scen_filter={s:{s} for s in SEASONS}; scen_filter[POOLED]=set(SEASONS)
    raw_p={}; wilson_a={}; shifts_p={}; gp_p={}
    for scen in SCENARIOS:
        agg=defaultdict(lambda:{"O":None,"D":None,"N":None,"gp":0})
        for (pid,season,fz),b in bucket.items():
            if season not in scen_filter[scen]: continue
            cur=agg[pid][fz] or {"shifts":0,"total_sec":0.0,"oz_sec":0.0,"dz_sec":0.0,"nz_sec":0.0}
            for k in ("shifts","total_sec","oz_sec","dz_sec","nz_sec"): cur[k]+=b[k]
            agg[pid][fz]=cur
        for (pid,season),v in gp.items():
            if season in scen_filter[scen]: agg[pid]["gp"]+=v
        scen_raw=defaultdict(dict); scen_w=defaultdict(dict); scen_sh={}
        scen_gp={}
        for pid,d in agg.items():
            if d["gp"]<MIN_GP: continue
            m=compute_metrics(d["O"],d["D"],d["N"])
            for met in ALL_METRICS:
                r,w,ok=m[met]
                if ok:
                    scen_raw[met][pid]=r; scen_w[met][pid]=w
            scen_sh[pid]={"O":(d["O"]["shifts"] if d["O"] else 0),
                          "D":(d["D"]["shifts"] if d["D"] else 0),
                          "N":(d["N"]["shifts"] if d["N"] else 0)}
            scen_gp[pid]=d["gp"]
        raw_p[scen]=scen_raw; wilson_a[scen]=scen_w
        shifts_p[scen]=scen_sh; gp_p[scen]=scen_gp

    print("[6/8] normalise + IOZL ...")
    # 0-1 normalize wilson per metric per position, per scenario
    n01=defaultdict(lambda:defaultdict(dict))
    for scen in SCENARIOS:
        for met in ALL_METRICS:
            pairs=list(wilson_a[scen][met].items())
            for pos in (POS_F,POS_D):
                sub=[(p,v) for p,v in pairs
                     if (player_meta.get(p,{}).get("position") or "").upper() in pos]
                for p,v in normalize01(sub).items():
                    n01[scen][met][p]=v

    # IOZL on pooled overlap
    def overlap_w(pid,met,pos_set):
        my=(player_meta.get(pid,{}).get("position") or "").upper()
        if my not in pos_set: return None
        sw=0.0; sv=0.0
        for (a,b),secs in teammate_map.items():
            if a!=pid and b!=pid: continue
            partner=b if a==pid else a
            pp=(player_meta.get(partner,{}).get("position") or "").upper()
            if pp not in pos_set: continue
            sc=n01[POOLED][met].get(partner)
            if sc is None: continue
            sw+=secs; sv+=sc*secs
        return (sv/sw) if sw>0 else None

    iozl={met:{} for met in ALL_METRICS}
    for met in ALL_METRICS:
        for pid in n01[POOLED][met]:
            pos=(player_meta.get(pid,{}).get("position") or "").upper()
            if pos in POS_F: iozl[met][pid]=overlap_w(pid,met,POS_F)
            elif pos in POS_D: iozl[met][pid]=overlap_w(pid,met,POS_D)

    print("[7/8] OLS βs at team level ...")
    # team points pooled = mean across seasons
    tps=defaultdict(list)
    for s in SEASONS:
        for tm,r in standings[s].items(): tps[tm].append(r["points"])
    team_points={tm:mean(v) for tm,v in tps.items() if v}

    def player_team(pid):
        return norm_team((player_meta.get(pid,{}).get("team_abbrev") or ""))
    def team_mean(pm):
        bt=defaultdict(list)
        for pid,v in pm.items():
            if v is None: continue
            tm=player_team(pid)
            if tm: bt[tm].append(v)
        return {tm:mean(vs) for tm,vs in bt.items() if vs}

    betas={}
    for met in ALL_METRICS:
        traw=team_mean(n01[POOLED][met]); tl=team_mean(iozl[met])
        common=set(team_points)&set(traw)&set(tl)
        teams=sorted(common)
        if len(teams)<5:
            betas[met]={"single_L":(None,None),"teams":0}; continue
        y=np.array([team_points[t] for t in teams])
        rv=np.array([traw[t] for t in teams]); lv=np.array([tl[t] for t in teams])
        X=np.column_stack([np.ones(len(teams)),rv,lv])
        b,*_=np.linalg.lstsq(X,y,rcond=None)
        betas[met]={"single_L":(float(b[1]),float(b[2])),"teams":len(teams)}

    # adjusted raw_L: raw - (β_iozl/β_raw) * IOZL
    adjL_raw={met:{} for met in ALL_METRICS}
    for met in ALL_METRICS:
        bR,bL=betas[met]["single_L"]
        if bR is None or bL is None or abs(bR)<1e-12:
            r=None
        else:
            r=bL/bR
        for pid,raw in n01[POOLED][met].items():
            if raw is None: continue
            l=iozl[met].get(pid)
            adjL_raw[met][pid]=(raw - r*l) if (r is not None and l is not None) else None

    # 0-10 scoring within position groups
    adjL_score={met:{} for met in ALL_METRICS}
    raw10_pool={met:{} for met in ALL_METRICS}
    for met in ALL_METRICS:
        for pos in (POS_F,POS_D):
            sub_a=[(p,v) for p,v in adjL_raw[met].items()
                   if (player_meta.get(p,{}).get("position") or "").upper() in pos]
            for p,v in score10(sub_a).items(): adjL_score[met][p]=v
            sub_r=[(p,v) for p,v in n01[POOLED][met].items()
                   if v is not None and (player_meta.get(p,{}).get("position") or "").upper() in pos]
            for p,v in score10(sub_r).items(): raw10_pool[met][p]=v

    print("[8/8] correlations + outputs ...")
    # team_points per scenario for per-season correlation
    team_points_by={"pooled":team_points}
    for s in SEASONS: team_points_by[s]={tm:r["points"] for tm,r in standings[s].items()}

    def corr(pid_map,points_map):
        t=team_mean(pid_map)
        c=set(t)&set(points_map)
        if len(c)<5: return float("nan")
        xs=[t[tm] for tm in sorted(c)]; ys=[points_map[tm] for tm in sorted(c)]
        return pearson(xs,ys)

    # raw per-season uses scenario's own raw 0-1 norm
    rows_corr=[]
    benchmark_tnzi_l_pool=None
    # adjusted pooled L: precompute dict
    for met in ALL_METRICS:
        # raw per-season
        season_r={}
        for s in SEASONS:
            season_r[s]=corr(n01[s][met], team_points_by[s])
        pool_r=corr(n01[POOLED][met], team_points_by["pooled"])
        rows_corr.append({"metric":met,"type":"raw","seasons":season_r,"pooled":pool_r})
        # adjusted L (only meaningful pooled — per-season uses pooled β + per-season raw + pooled IOZL)
        bR,bL=betas[met]["single_L"]
        r_ratio=(bL/bR) if (bR and abs(bR)>1e-12 and bL is not None) else None
        if r_ratio is not None:
            # per-season L: raw_per_season - r * iozl_pool (only IDs present in both)
            seasonL={}
            for s in SEASONS:
                pm={}
                for pid,raw_s in n01[s][met].items():
                    if raw_s is None: continue
                    l=iozl[met].get(pid)
                    if l is None: continue
                    pm[pid]=raw_s - r_ratio*l
                seasonL[s]=corr(pm, team_points_by[s])
            poolL=corr(adjL_raw[met], team_points_by["pooled"])
            rows_corr.append({"metric":met,"type":"L","seasons":seasonL,"pooled":poolL})
            if met=="TNZI": benchmark_tnzi_l_pool=poolL

    # ---------- COMPARISON TABLE FIRST ----------
    print("\n" + "=" * 100)
    print("COMPARISON TABLE — all metrics vs team points")
    print("=" * 100)
    bm = benchmark_tnzi_l_pool if benchmark_tnzi_l_pool is not None else 0.589
    header=f"{'Metric':<10} {'Type':<5} {'22/23':>7} {'23/24':>7} {'24/25':>7} {'25/26':>7} {'Pooled':>7}  Beats TNZI_L ({bm:.3f})?"
    print(header); print("-"*len(header))
    # print order matching spec
    order = [("OZI","raw"),("TOZI","raw"),("TOZI","L"),
             ("DZI","raw"),("TDZI","raw"),("TDZI","L"),
             ("NZI","raw"),("TNZI","raw"),("TNZI","L")]
    rmap={(r["metric"],r["type"]):r for r in rows_corr}
    for met,tp in order:
        r=rmap.get((met,tp))
        if not r: print(f"{met:<10} {tp:<5} (not computed)"); continue
        s=r["seasons"]
        def f(v): return f"{v:>7.3f}" if v==v else "    nan"  # nan-safe
        beats="YES" if (r["pooled"]==r["pooled"] and r["pooled"]>bm) else "no"
        if (met,tp)==("TNZI","L"): beats="benchmark"
        print(f"{met:<10} {tp:<5} {f(s.get('20222023',float('nan')))} {f(s.get('20232024',float('nan')))} "
              f"{f(s.get('20242025',float('nan')))} {f(s.get('20252026',float('nan')))} "
              f"{f(r['pooled'])}  {beats}")

    # ---------- TOP 20 RANKINGS + OUTPUT FILES ----------
    print("\n" + "=" * 100)
    print("PLAYER RANKINGS — pooled")
    print("=" * 100)

    def write_csv_for(metric):
        # produce {forwards,defense} csvs
        for pos_label,pos_set in (("forwards",POS_F),("defense",POS_D)):
            rows=[]
            for pid in n01[POOLED][metric]:
                pos=(player_meta.get(pid,{}).get("position") or "").upper()
                if pos not in pos_set: continue
                meta=player_meta.get(pid,{})
                rows.append({
                    "player_id":pid,
                    "player_name":meta.get("name",""),
                    "team":norm_team(meta.get("team_abbrev","") or ""),
                    "pos":pos,
                    "GP":gp_p[POOLED].get(pid,0),
                    "shifts_O":shifts_p[POOLED].get(pid,{}).get("O",0),
                    "shifts_D":shifts_p[POOLED].get(pid,{}).get("D",0),
                    "shifts_N":shifts_p[POOLED].get(pid,{}).get("N",0),
                    f"{metric}_raw_p":raw_p[POOLED][metric].get(pid),
                    f"{metric}_wilson":wilson_a[POOLED][metric].get(pid),
                    "IOZL":iozl[metric].get(pid),
                    metric:raw10_pool[metric].get(pid),
                    f"{metric}_L":adjL_score[metric].get(pid),
                })
            rows.sort(key=lambda r:(r[f"{metric}_L"] if r[f"{metric}_L"] is not None else -9e9), reverse=True)
            path=OUT_DIR / f"{metric.lower()}_adjusted_{pos_label}.csv"
            fields=list(rows[0].keys()) if rows else ["player_id"]
            with open(path,"w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
                for r in rows: w.writerow({k:("" if v is None else v) for k,v in r.items()})
            print(f"  wrote {path}  ({len(rows)} rows)")
            # also stash for printing top 20
            if pos_label=="forwards": all_F[metric]=rows
            else: all_D[metric]=rows

    all_F={}; all_D={}
    for met in NEW_METRICS:
        write_csv_for(met)

    def print_top(rows,metric,label,n=20):
        print(f"\n=== Top {n} {label} by {metric}_L (pooled) ===")
        print(f"{'#':>2} {'Player':<26} {'Team':<4} GP  shO  shD  shN  rawP    Wilson  IOZL    {metric}  {metric}_L")
        for i,r in enumerate(rows[:n],1):
            wilv=r[f'{metric}_wilson']
            print(f"{i:>2} {r['player_name'][:26]:<26} {r['team']:<4} "
                  f"{r['GP']:>3} {r['shifts_O']:>4} {r['shifts_D']:>4} {r['shifts_N']:>4} "
                  f"{(r[f'{metric}_raw_p'] or 0):>+6.3f} {(wilv if wilv is not None else 0):>+7.3f} "
                  f"{(r['IOZL'] or 0):>5.3f} {(r[metric] if r[metric] is not None else ''):>5} "
                  f"{(r[f'{metric}_L'] if r[f'{metric}_L'] is not None else ''):>5}")

    for met in ("TDZI","TOZI"):
        print_top(all_F[met],met,"forwards")
        print_top(all_D[met],met,"defensemen")

    # ---------- KEY PLAYER ROWS ----------
    print("\n=== Key player rows — pooled ===")
    print(f"{'Player':<14} {'Team':<4} {'Pos':<3} {'GP':>3} {'TOZI':>5} {'TOZI_L':>7} {'TDZI':>5} {'TDZI_L':>7} {'DOZI_flag'}")
    # Load DOZI flag from existing tnzi_adjusted file
    dozi_flag={}
    for fn in ("tnzi_adjusted_forwards.csv","tnzi_adjusted_defense.csv"):
        try:
            with open(OUT_DIR/fn) as f:
                for r in csv.DictReader(f):
                    nm=r["player_name"].strip()
                    if nm: dozi_flag[nm]=r.get("DOZI_flag","")
        except FileNotFoundError: pass

    def find_player(name_substr):
        for pid,m in player_meta.items():
            n=m.get("name","")
            if name_substr.lower() in n.lower(): return pid,n
        return None,None

    targets=["McDavid","MacKinnon","Draisaitl","Makar","Quinn Hughes","Nurse","Bouchard","Ekholm","Bedard"]
    for t in targets:
        pid,nm=find_player(t)
        if pid is None: print(f"{t:<14} (not found)"); continue
        m=player_meta.get(pid,{})
        pos=(m.get("position") or "").upper()
        tm=norm_team(m.get("team_abbrev") or "")
        gpv=gp_p[POOLED].get(pid,0)
        tozi=raw10_pool["TOZI"].get(pid); tozi_l=adjL_score["TOZI"].get(pid)
        tdzi=raw10_pool["TDZI"].get(pid); tdzi_l=adjL_score["TDZI"].get(pid)
        flag=dozi_flag.get(nm,"")
        print(f"{nm[:14]:<14} {tm:<4} {pos:<3} {gpv:>3} "
              f"{(tozi if tozi is not None else ''):>5} {(tozi_l if tozi_l is not None else ''):>7} "
              f"{(tdzi if tdzi is not None else ''):>5} {(tdzi_l if tdzi_l is not None else ''):>7} {flag}")

if __name__=="__main__":
    main()
