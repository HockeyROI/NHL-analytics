"""repair_shifts_from_html.py
-------------------------------
Repair empty shift chart JSONs by scraping NHL HTML TOI shift reports.

Input condition: shift JSONs under Data/qoc_qol/shifts/ whose on-disk content
is `{"data":[], "total":0}` or is < 100 bytes / unparseable. These exist
because the official stats API endpoint
(https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=X) has
been silently empty for 550 late-2024-25 and 2025-26 games, while the
public HTML TOI reports at
    https://www.nhl.com/scores/htmlreports/{season}/TV{code}.HTM
    https://www.nhl.com/scores/htmlreports/{season}/TH{code}.HTM
are fully populated.

Output: rewrites each empty shift JSON to the same schema the pipeline
expects (downstream consumer: compute_zone_and_overlap.py, which reads
`typeCode == 517` rows with `duration`, `startTime`, `endTime`, `period`,
`playerId`, `teamId`). Player IDs are resolved by looking up
(teamId, sweaterNumber) in the game's PBP rosterSpots, which are still
correct.

Does not touch non-empty shift JSONs, does not touch PBP files, and does
not run the zone-time or PQR computation. Those are kicked off by the
caller after this script finishes.

Writes Player_Ranking/html_shift_repair_log.txt with per-game status.
"""
from __future__ import annotations
import csv, json, os, re, sys, time, urllib.request

ROOT    = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
SHIFTS_DIR = os.path.join(ROOT, "raw", "shifts")
PBP_DIR    = os.path.join(ROOT, "raw", "pbp")
GAME_IDS   = os.path.join(PROJECT, "Data", "game_ids.csv")
LOG_PATH   = os.path.join(ROOT, "html_shift_repair_log.txt")

UA = {"User-Agent": "Mozilla/5.0"}
EMPTY_SIZE_THRESHOLD = 100
DRAISAITL_ID = 8477934

# ---------------------------------------------------------------------------
# parsing helpers

_PLAYER_HEADING_RE = re.compile(
    r'class="playerHeading[^"]*"[^>]*>\s*([0-9]+)\s+([^,<]+?)\s*,\s*([^<]+?)\s*</td>',
    re.IGNORECASE,
)
# A shift row is a <tr ...oddColor|evenColor...> followed by 6 <td>...</td>:
# Shift#, Per, Start (M:SS / MM:SS), End (M:SS / MM:SS), Duration (MM:SS), Event
_SHIFT_TR_RE = re.compile(
    r'<tr[^>]*(?:oddColor|evenColor)[^>]*>(.*?)</tr>',
    re.DOTALL | re.IGNORECASE,
)
_TD_RE = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL | re.IGNORECASE)
_TAGS_RE = re.compile(r'<[^>]+>')

def _strip(s):
    return _TAGS_RE.sub('', s).replace("&nbsp;", " ").strip()

def parse_html_report(html):
    """Return list of dicts: {sweater, last, first, shifts: [{shift, period, start, end, dur, event}]}."""
    out = []
    # Find all player headings with their byte offsets so we can slice between them
    heads = list(_PLAYER_HEADING_RE.finditer(html))
    for i, m in enumerate(heads):
        sweater = int(m.group(1))
        last = m.group(2).strip()
        first = m.group(3).strip()
        start_pos = m.end()
        end_pos = heads[i + 1].start() if i + 1 < len(heads) else len(html)
        segment = html[start_pos:end_pos]

        shifts = []
        for tr_m in _SHIFT_TR_RE.finditer(segment):
            tds = [_strip(x) for x in _TD_RE.findall(tr_m.group(1))]
            if len(tds) != 6:
                continue
            shift_no, per, start_cell, end_cell, dur_cell = tds[:5]
            if not shift_no.isdigit():
                continue
            per_int = int(per) if per.isdigit() else 99
            if per_int < 1 or per_int > 4:
                continue
            # Extract "M:SS" elapsed portion of "M:SS / MM:SS"
            def elapsed(cell):
                return cell.split("/", 1)[0].strip()
            try:
                shifts.append({
                    "shift": int(shift_no),
                    "period": int(per),
                    "start": elapsed(start_cell),
                    "end":   elapsed(end_cell),
                    "dur":   dur_cell,
                })
            except ValueError:
                continue
        if shifts:
            out.append({
                "sweater": sweater,
                "last": last,
                "first": first,
                "shifts": shifts,
            })
    return out


def normalize_hms(s):
    """HTML reports use 'M:SS' for start/end and 'MM:SS' for duration.
    Downstream hms() splits on ':' and int()s both halves, so both work.
    We normalise duration to 'MM:SS' to match the stats API schema. """
    if ":" not in s:
        return s
    mm, ss = s.split(":", 1)
    return f"{int(mm):02d}:{int(ss):02d}"


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# per-game repair

def build_roster_lookup(pbp):
    """(teamId, sweaterNumber) -> rosterSpot dict."""
    lut = {}
    for p in pbp.get("rosterSpots", []):
        key = (p["teamId"], int(p.get("sweaterNumber", 0)))
        lut[key] = p
    return lut


def pbp_team_meta(pbp):
    """Return dict keyed by teamId with {'abbrev': ..., 'name': ...}."""
    meta = {}
    for team_key in ("homeTeam", "awayTeam"):
        t = pbp[team_key]
        name = t.get("commonName", {}).get("default") if isinstance(t.get("commonName"), dict) else t.get("commonName", "")
        meta[t["id"]] = {
            "abbrev": t.get("abbrev", ""),
            "name": name or t.get("abbrev", ""),
        }
    return meta


def build_shift_json(gid, pbp, tv_html, th_html):
    """Turn the two HTML reports into the stats-API-style shift-chart JSON.
    Returns (dict_or_None, list_of_warnings)."""
    warnings = []
    away_id = pbp["awayTeam"]["id"]
    home_id = pbp["homeTeam"]["id"]
    roster_lut = build_roster_lookup(pbp)
    team_meta = pbp_team_meta(pbp)

    data = []
    event_num = 1
    synth_id = int(gid) * 1000   # synthetic monotonic id (the real stats DB uses autoincrement ints; exact value isn't read downstream)

    for team_id, html, side in ((away_id, tv_html, "TV"), (home_id, th_html, "TH")):
        if not html:
            warnings.append(f"missing {side} html")
            continue
        players = parse_html_report(html)
        if not players:
            warnings.append(f"{side}: no players parsed")
            continue
        for p in players:
            key = (team_id, p["sweater"])
            rs = roster_lut.get(key)
            if rs is None:
                warnings.append(f"{side}: no roster match for sweater {p['sweater']} ({p['first']} {p['last']})")
                continue
            player_id = rs["playerId"]
            first_default = rs["firstName"]["default"] if isinstance(rs.get("firstName"), dict) else rs.get("firstName", p["first"])
            last_default  = rs["lastName"]["default"]  if isinstance(rs.get("lastName"), dict)  else rs.get("lastName", p["last"])
            abbrev = team_meta.get(team_id, {}).get("abbrev", "")
            team_name = team_meta.get(team_id, {}).get("name", abbrev)
            for s in p["shifts"]:
                data.append({
                    "id": synth_id,
                    "detailCode": 0,
                    "duration": normalize_hms(s["dur"]),
                    "endTime": normalize_hms(s["end"]),
                    "eventDescription": None,
                    "eventDetails": None,
                    "eventNumber": event_num,
                    "firstName": first_default,
                    "gameId": int(gid),
                    "hexValue": None,
                    "lastName": last_default,
                    "period": s["period"],
                    "playerId": player_id,
                    "shiftNumber": s["shift"],
                    "startTime": normalize_hms(s["start"]),
                    "teamAbbrev": abbrev,
                    "teamId": team_id,
                    "teamName": team_name,
                    "typeCode": 517,
                })
                synth_id += 1
                event_num += 1
    if not data:
        return None, warnings
    return {"data": data, "total": len(data)}, warnings


def is_empty_shift_file(path):
    if not os.path.exists(path):
        return True
    try:
        if os.path.getsize(path) < EMPTY_SIZE_THRESHOLD:
            return True
        with open(path, "r") as f:
            d = json.load(f)
    except Exception:
        return True
    if isinstance(d, dict):
        if not d.get("data") or d.get("total") == 0:
            return True
    elif isinstance(d, list):
        if not d:
            return True
    return False


def repair_one(gid):
    """Fetch + parse + write for a single game. Returns (status, detail)."""
    year = int(gid[:4])
    season = f"{year}{year + 1}"
    code = gid[4:]

    pbp_path = os.path.join(PBP_DIR, f"{gid}.json")
    if not os.path.exists(pbp_path):
        return "fail_no_pbp", "pbp file missing"
    with open(pbp_path) as f:
        pbp = json.load(f)

    try:
        tv_html = fetch(f"https://www.nhl.com/scores/htmlreports/{season}/TV{code}.HTM")
    except Exception as e:
        tv_html = ""
    try:
        th_html = fetch(f"https://www.nhl.com/scores/htmlreports/{season}/TH{code}.HTM")
    except Exception as e:
        th_html = ""
    if not tv_html and not th_html:
        return "fail_no_html", "both HTML reports unreachable"

    shift_json, warnings = build_shift_json(gid, pbp, tv_html, th_html)
    if shift_json is None:
        return "fail_parse", "; ".join(warnings) or "no shifts extracted"

    out_path = os.path.join(SHIFTS_DIR, f"{gid}.json")
    with open(out_path, "w") as f:
        json.dump(shift_json, f)
    return "ok", f"{shift_json['total']} shifts" + (f"; warnings: {'; '.join(warnings)}" if warnings else "")


# ---------------------------------------------------------------------------
# orchestration

def count_draisaitl_gp():
    edm_games = set()
    with open(GAME_IDS) as f:
        for row in csv.DictReader(f):
            if row["season"] == "20252026" and row["game_type"] == "regular":
                if row["home_abbrev"] == "EDM" or row["away_abbrev"] == "EDM":
                    edm_games.add(row["game_id"])
    gp = 0
    for gid in edm_games:
        p = os.path.join(SHIFTS_DIR, f"{gid}.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        shifts = d.get("data", []) if isinstance(d, dict) else d
        if any(isinstance(s, dict) and s.get("playerId") == DRAISAITL_ID for s in shifts):
            gp += 1
    return gp, len(edm_games)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "test":
        test_gid = sys.argv[2] if len(sys.argv) > 2 else "2025020065"
        print(f"[test] repairing single game {test_gid}")
        status, detail = repair_one(test_gid)
        print(f"  status={status}  detail={detail}")
        path = os.path.join(SHIFTS_DIR, f"{test_gid}.json")
        with open(path) as f:
            d = json.load(f)
        drai_shifts = [s for s in d.get("data", []) if s.get("playerId") == DRAISAITL_ID]
        print(f"  Draisaitl shifts in {test_gid}: {len(drai_shifts)}")
        for s in drai_shifts[:3]:
            print(f"    period={s['period']}  start={s['startTime']}  end={s['endTime']}  dur={s['duration']}  shift#={s['shiftNumber']}")
        return

    # full run
    files = sorted(fn for fn in os.listdir(SHIFTS_DIR) if fn.endswith(".json"))
    empty = [fn.replace(".json", "") for fn in files
             if is_empty_shift_file(os.path.join(SHIFTS_DIR, fn))]
    print(f"[html_repair] {len(files)} shift files scanned; {len(empty)} empty")

    ok_ids, fail_rows = [], []
    for i, gid in enumerate(empty, 1):
        status, detail = repair_one(gid)
        if status == "ok":
            ok_ids.append(gid)
        else:
            fail_rows.append((gid, status, detail))
        if i % 25 == 0 or i == len(empty):
            print(f"  ... {i}/{len(empty)} ok={len(ok_ids)} fail={len(fail_rows)}")
        time.sleep(0.1)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        f.write(f"# HTML shift-report repair log\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# scanned={len(files)} empty={len(empty)} repaired={len(ok_ids)} failed={len(fail_rows)}\n\n")
        f.write("# --- SUCCESS ---\n")
        for gid in ok_ids:
            f.write(f"{gid}\tok\n")
        f.write("\n# --- FAILURE ---\n")
        for gid, status, detail in fail_rows:
            f.write(f"{gid}\t{status}\t{detail}\n")

    drai_gp, edm_total = count_draisaitl_gp()
    print()
    print("=== HTML repair summary ===")
    print(f"  empty files before:           {len(empty)}")
    print(f"  successfully repaired:        {len(ok_ids)}")
    print(f"  still failing:                {len(fail_rows)}")
    print(f"  log:                          {LOG_PATH}")
    print(f"  Draisaitl GP (EDM reg 25-26): {drai_gp} of {edm_total} scheduled")


if __name__ == "__main__":
    main()
