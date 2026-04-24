"""HockeyROI - PUR / PAR / PFR + ROC / ROL Streamlit app.

Reads CSVs produced by the daily-update pipeline (under ./Player_Ranking/).
Two-page sidebar nav (Rankings + Methodology); Forwards / Defensemen tabs;
PFR-led leaderboards with overlap-based context.
"""
from __future__ import annotations
import os
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
# CSVs live in the sibling Zones/ folder (produced by the zone pipeline).
DATA_DIR = os.path.join(os.path.dirname(ROOT), "Zones")

# ---- HockeyROI brand tokens -------------------------------------------------
BG        = "#1B3A5C"
SURFACE   = "#0B1D2E"
BLUE      = "#2E7DC4"
BLUE_LT   = "#4AB3E8"
TEXT      = "#F0F4F8"
ORANGE    = "#FF6B35"
HIGH      = "#44AA66"
MID       = "#FFB700"
LOW       = "#CC3333"
HEADLINE_FONT = "Bebas Neue, Impact, sans-serif"
BODY_FONT     = "Inter, Arial, sans-serif"
REPO_URL      = "https://github.com/HockeyROI/nhl-analytics"
SUBSTACK_URL  = "https://hockeyROI.substack.com"

TOOLTIPS = {
    "OUER": ("Offensive Unadjusted Efficiency Rating - how well a player "
             "retains the offensive zone vs league average on OZ faceoff "
             "shifts. Above 1.0 = better than league average. A HockeyROI metric."),
    "OAER": ("Offensive Adjusted Efficiency Rating - how well a player retains "
             "the offensive zone vs their own teammates. Above 1.0 = better "
             "than your own team. A HockeyROI metric."),
    "DUER": ("Defensive Unadjusted Efficiency Rating - how quickly a player "
             "escapes the defensive zone vs league average on DZ faceoff "
             "shifts. Above 1.0 = faster than league average. A HockeyROI metric."),
    "DAER": ("Defensive Adjusted Efficiency Rating - how quickly a player "
             "escapes the defensive zone vs their own teammates. Above 1.0 = "
             "faster than your own team. A HockeyROI metric."),
    "PUR": ("Player Unadjusted Rating - combines OUER and DUER weighted by "
            "TOI per game relative to league average. Measures raw zone "
            "efficiency vs the whole league. 1.0 = best in dataset at "
            "position. A HockeyROI metric."),
    "PAR": ("Player Adjusted Rating - combines OAER and DAER weighted by "
            "TOI per game relative to league average. Measures zone efficiency "
            "relative to teammates - filters out system effects. 1.0 = best "
            "in dataset at position. A HockeyROI metric."),
    "PFR": ("Player Final Rating - PUR divided by PAR. Below 1.0 = player "
            "outperforms their linemates and context, likely undervalued. "
            "Above 1.0 = player elevated by system and teammates. Near 1.0 = "
            "consistent regardless of context. The headline HockeyROI metric."),
    "ROC": ("Rank of Competition - weighted average PUR of opposing players "
            "faced at 5v5. Higher = tougher competition. A HockeyROI metric."),
    "ROL": ("Rank of Linemates - weighted average PUR of teammates at 5v5. "
            "Higher = better linemates. A HockeyROI metric."),
}

st.set_page_config(page_title="HockeyROI - PFR / PUR / PAR",
                   layout="wide", initial_sidebar_state="expanded")

# -- global styling -----------------------------------------------------------
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: {BODY_FONT}; color: {TEXT}; }}
  .stApp {{ background: {BG}; }}
  header, .stDeployButton, #MainMenu {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1700px; }}
  h1, h2, h3, .brand-title {{ font-family: {HEADLINE_FONT}; letter-spacing: 0.5px; color: {TEXT}; }}
  .brand-bar {{
    display: flex; align-items: center; justify-content: space-between;
    background: {SURFACE};
    padding: 1.0rem 1.4rem; border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 1.0rem;
  }}
  .brand-title {{ font-size: 2.4rem; color: {TEXT}; margin: 0; }}
  .brand-accent {{ color: {ORANGE}; }}
  .brand-sub   {{ color: {BLUE_LT}; font-size: 0.9rem; }}
  .metric-card {{
    background: {SURFACE}; padding: 0.6rem 0.85rem; border-radius: 8px;
    border-left: 4px solid {BLUE_LT};
  }}
  .stDataFrame {{ background: {SURFACE}; }}
  .stDataFrame thead tr th {{
    background-color: {SURFACE} !important;
    color: {BLUE_LT} !important; font-weight: 700;
  }}
  div[data-testid="stSidebar"] {{ background: {SURFACE}; }}
  label, .stMarkdown p {{ color: {TEXT}; }}
  a {{ color: {BLUE_LT}; }}
  .stTabs [role="tab"] {{
    font-family: {HEADLINE_FONT}; font-size: 1.2rem; letter-spacing: 1px;
    color: {BLUE_LT};
  }}
  .stTabs [aria-selected="true"] {{
    color: {TEXT} !important;
    border-bottom: 3px solid {ORANGE} !important;
  }}
  .section-h {{
    font-family: {HEADLINE_FONT}; color: {TEXT}; letter-spacing: 1px;
    margin-top: 1.6rem; margin-bottom: 0.4rem;
  }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_table(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def read_ts(path):
    if not os.path.exists(path): return "(not available)"
    return open(path).read().strip()

ts_reg = read_ts(os.path.join(DATA_DIR, "last_updated_regular.txt"))
ts_pof = read_ts(os.path.join(DATA_DIR, "last_updated_playoffs.txt"))

FILE_FOR = {
    ("forwards", "Pooled (4 seasons)"):        "pqr_forwards_pooled.csv",
    ("forwards", "Current season - Regular"):  "pqr_forwards_regular.csv",
    ("forwards", "Current season - Playoffs"): "pqr_forwards_playoffs.csv",
    ("defense",  "Pooled (4 seasons)"):        "pqr_defense_pooled.csv",
    ("defense",  "Current season - Regular"):  "pqr_defense_regular.csv",
    ("defense",  "Current season - Playoffs"): "pqr_defense_playoffs.csv",
}

# ----------------------------------------------------------------------------
# helpers
def mmss(v):
    if pd.isna(v): return ""
    s = int(round(float(v) * 60))
    return f"{s // 60}:{s % 60:02d}"

def pfr_color(v):
    if pd.isna(v): return ""
    if v < 0.95:  return f"background-color: {HIGH}33; color: {TEXT}; font-weight: 600;"
    if v > 1.05:  return f"background-color: {LOW}33;  color: {TEXT}; font-weight: 600;"
    return              f"background-color: {MID}33;  color: {TEXT}; font-weight: 600;"

def tertile_colors(series):
    s = series.dropna().astype(float)
    if len(s) < 3:
        return pd.Series(["" for _ in series], index=series.index)
    q33, q66 = s.quantile(0.333), s.quantile(0.667)
    def cell(v):
        if pd.isna(v): return ""
        if v >= q66:  return f"background-color: {HIGH}33; color: {TEXT};"
        if v >= q33:  return f"background-color: {MID}33;  color: {TEXT};"
        return              f"background-color: {LOW}33;  color: {TEXT};"
    return series.map(cell)

# ----------------------------------------------------------------------------
def render_table(view, label):
    """Format + color-code a leaderboard slice."""
    show = view.copy()
    if "toi_per_game" in show.columns:
        show["toi_per_game"] = show["toi_per_game"].apply(mmss)

    fmt = {
        "total_5v5_toi_minutes": "{:.1f}",
        "player_OZ_retention_pct": "{:.1f}%",
        "player_DZ_escape_rate_pct": "{:.1f}%",
        "oz_faceoff_shifts_per_game": "{:.2f}",
        "dz_faceoff_shifts_per_game": "{:.2f}",
        "OUER": "{:.3f}", "DUER": "{:.3f}",
        "OAER": "{:.3f}", "DAER": "{:.3f}",
        "NZ_ratio": "{:.3f}",
        "PUR": "{:.3f}", "PAR": "{:.3f}", "PFR": "{:.3f}",
        "ROC": "{:.3f}", "ROL": "{:.3f}",
        "ROC_adj": "{:.3f}", "ROL_adj": "{:.3f}",
        "CF_pct": lambda v: f"{v*100:.1f}%" if pd.notnull(v) else "",
    }
    fmt = {k: v for k, v in fmt.items() if k in show.columns}
    styler = show.style.format(fmt)
    if "PFR" in show.columns:
        styler = styler.map(pfr_color, subset=["PFR"])
    for col in ("PUR", "PAR", "ROC", "ROL"):
        if col in show.columns:
            styler = styler.apply(tertile_colors, subset=[col])
    st.dataframe(styler, use_container_width=True, hide_index=True, height=620)

def render_group(group_key: str, label: str):
    key = group_key
    scenario = st.selectbox(
        f"{label} - dataset",
        ["Pooled (4 seasons)", "Current season - Regular", "Current season - Playoffs"],
        index=0, key=f"{key}_scenario",
    )
    df = load_table(os.path.join(DATA_DIR, FILE_FOR[(group_key, scenario)]))
    if df.empty:
        st.warning(f"No data yet for '{label}' / '{scenario}'.")
        return

    # ---- filter row 1
    top = st.columns([2, 2, 2, 3])
    teams = sorted([t for t in df["team"].dropna().unique() if t])
    with top[0]:
        team_sel = st.multiselect(f"Team ({label})", teams, default=[], key=f"{key}_team")
    with top[1]:
        name_search = st.text_input("Search by player name", "", key=f"{key}_name")
    with top[2]:
        min_toi = st.slider(
            "Min 5v5 TOI (min)",
            min_value=200,
            max_value=int(max(200, df["total_5v5_toi_minutes"].max() // 10 * 10)),
            value=200, step=50, key=f"{key}_toi",
        )
    with top[3]:
        ctx_options = ["(all)"] + sorted(c for c in df.get("context_flag", pd.Series()).dropna().unique() if c)
        ctx_sel = st.selectbox("Context flag", ctx_options, index=0, key=f"{key}_ctx")

    # ---- toggleable columns
    tog = st.columns(5)
    show_nz       = tog[0].checkbox("Show NZ_ratio",          value=False, key=f"{key}_nz")
    show_roc_adj  = tog[1].checkbox("Show ROC_adj",           value=False, key=f"{key}_rocadj")
    show_rol_adj  = tog[2].checkbox("Show ROL_adj",           value=False, key=f"{key}_roladj")
    show_oz_fo    = tog[3].checkbox("Show oz_faceoff_shifts", value=False, key=f"{key}_ozfo")
    show_dz_fo    = tog[4].checkbox("Show dz_faceoff_shifts", value=False, key=f"{key}_dzfo")

    # ---- filter
    view = df.copy()
    if team_sel:    view = view[view["team"].isin(team_sel)]
    if name_search: view = view[view["player_name"].str.contains(name_search, case=False, na=False)]
    view = view[view["total_5v5_toi_minutes"] >= min_toi]
    if ctx_sel != "(all)" and "context_flag" in view.columns:
        view = view[view["context_flag"] == ctx_sel]

    # ---- column ordering (default sort: PFR ascending so headliners surface)
    view = view.sort_values("PFR", ascending=True, na_position="last")

    base_cols = [
        "player_name", "team", "pos",
        "PFR", "PUR", "PAR",
        "games_played", "total_5v5_toi_minutes", "toi_per_game",
        "oz_faceoff_shifts_per_game",
        "dz_faceoff_shifts_per_game",
        "player_OZ_retention_pct", "player_DZ_escape_rate_pct",
        "OUER", "DUER", "OAER", "DAER",
        "ROC", "ROL",
        "context_flag", "CF_pct",
    ]
    if show_oz_fo:   base_cols.insert(base_cols.index("oz_faceoff_shifts_per_game"), "oz_faceoff_shifts")
    if show_dz_fo:   base_cols.insert(base_cols.index("dz_faceoff_shifts_per_game"), "dz_faceoff_shifts")
    if show_nz:      base_cols.insert(base_cols.index("OUER"), "NZ_ratio")
    if show_roc_adj: base_cols.insert(base_cols.index("ROL") + 1, "ROC_adj")
    if show_rol_adj:
        idx = base_cols.index("ROC_adj") + 1 if "ROC_adj" in base_cols else base_cols.index("ROL") + 1
        base_cols.insert(idx, "ROL_adj")
    view = view[[c for c in base_cols if c in view.columns]]

    st.markdown(
        f"<h3 class='section-h'>{label} &nbsp;&middot;&nbsp; {scenario} "
        f"&nbsp;&middot;&nbsp; {len(view)} players</h3>",
        unsafe_allow_html=True,
    )
    render_table(view, label)

    # ---- Most Undervalued: PAR top half, PFR < 0.90
    st.markdown(f"<h3 class='section-h'>Most Undervalued - PAR top half, PFR &lt; 0.90</h3>",
                unsafe_allow_html=True)
    par_med = df["PAR"].median()
    under = df[(df["PAR"] >= par_med) & (df["PFR"] < 0.90)].sort_values("PFR", ascending=True).head(15)
    if under.empty:
        st.caption("No players match.")
    else:
        render_table(under[[c for c in base_cols if c in under.columns]], label + " (undervalued)")

    # ---- Most Overvalued: PAR bottom half, PFR > 1.10
    st.markdown(f"<h3 class='section-h'>Most Overvalued - PAR bottom half, PFR &gt; 1.10</h3>",
                unsafe_allow_html=True)
    over = df[(df["PAR"] < par_med) & (df["PFR"] > 1.10)].sort_values("PFR", ascending=False).head(15)
    if over.empty:
        st.caption("No players match.")
    else:
        render_table(over[[c for c in base_cols if c in over.columns]], label + " (overvalued)")

# ----------------------------------------------------------------------------
def render_methodology():
    st.markdown(f"<h2 class='section-h'>The HockeyROI Rating System</h2>", unsafe_allow_html=True)
    st.markdown(f"""
A 4-season pool (2022-23 -> 2025-26, regular + playoffs) of every NHL player's
5v5 zone-deployment efficiency, plus the competition and linemates they face.
The headline metric is **PFR (Player Final Rating)** - whether a player is
outperforming their context (`PFR < 1`) or being elevated by it (`PFR > 1`).

### What we measure (and why it isn't Corsi)
Corsi (CF%) treats every shot attempt the same. We instead track *zone time* -
how long the puck spends in each zone while a player is on the ice - which is
the upstream cause of shot differentials. Corsi is included as a reference
column only.

### Faceoff-start shifts only
A player's efficiency rating is measured **only over shifts where the player
was on the ice for the starting faceoff** (line-change shifts are ignored).
The faceoff itself must be at 5v5. This isolates how a player handles
deployment, separating zone efficiency from random line-change context.

### OUER and DUER - vs the league
- **OUER** = `player_OZ_retention% / league_avg_OZ_retention` (per position).
  Above 1.0 = retains the offensive zone better than the average forward / D.
- **DUER** = `player_DZ_escape_rate% / league_avg_DZ_escape` (per position).
  Above 1.0 = exits the defensive zone faster than average.
  *DZ_escape_rate% = 1 - DZ_time%*. Both gated to 50+ faceoff shifts.

### OAER and DAER - vs your own teammates
- **OAER** divides each player's OZ retention by an overlap-weighted average
  of their teammates' OZ retention. Above 1.0 = better than your own team.
- **DAER** does the same for DZ escape. Both filter out the system effect of
  playing on a great or weak group.

### PUR - unadjusted overall rating
`PUR_raw = (OUER + DUER) x (toi_per_game / league_avg_toi_per_game)`,
max-normalised within position so the top forward = top defenseman = 1.0.

### PAR - adjusted overall rating
`PAR_raw = (OAER + DAER) x (toi_per_game / league_avg_toi_per_game)`, max-
normalised within position. Same shape as PUR but built on the team-relative
ratings.

### PFR - the headline metric
`PFR = PUR / PAR`. Below 1.0 = the player rates higher league-wide than they
do relative to their own teammates -> they are dragging their context up.
Above 1.0 = they look better than the league because their teammates are
elevating them. Near 1.0 = consistent in either lens.

### ROC and ROL
**ROC** = overlap-weighted average **PUR** of opposing players faced at 5v5.
**ROL** = overlap-weighted average **PUR** of teammates at 5v5. Both filter
to same-position pairs only. Toggleable `ROC_adj` and `ROL_adj` columns
substitute PAR for PUR.

### Context flag
Each player gets a plain-English tag built from PAR tercile (top/middle/bottom
within position) crossed with PFR direction:

| | PFR < 0.95 | 0.95-1.05 | PFR > 1.05 |
|---|---|---|---|
| **PAR top third** | Elite - self made | Elite - consistent | Elite - system supported |
| **PAR middle third** | Solid - self made | Neutral | Solid - system supported |
| **PAR bottom third** | Underperforming - tough situation | Underperforming - neutral | Underperforming - no excuse |

### Important caveats
- **Zone tracking is an approximation.** The puck's zone between events is
  inferred from the last event with x/y coordinates; rapid zone changes
  between PBP events are not captured. `homeTeamDefendingSide` is used to
  orient OZ/DZ per period.
- **Sample thresholds.** Players need 50+ OZ faceoff shifts (for OUER/OAER),
  50+ DZ faceoff shifts (for DUER/DAER), and 200+ minutes of 5v5 TOI to enter
  the rating pool. Below-threshold players are listed in
  `Player_Ranking/zone_time_below_threshold.csv`.
- **Linemate circularity.** OAER/DAER divide by your own teammates' rates, so
  every player on a great line will look slightly worse on PAR than on PUR.
  PFR is built precisely to surface that gap - it is *the* signal here.

### Data sources
- NHL API shift charts (`api.nhle.com/stats/rest/en/shiftcharts`)
- NHL API play-by-play (`api-web.nhle.com/v1/gamecenter/{{id}}/play-by-play`)
- Pooled across 2022-23, 2023-24, 2024-25, and 2025-26-to-date
- 5v5 only (situationCode == 1551 AND both teams 5 non-goalie skaters)

### Source code & contact
- Pipeline + methodology: [{REPO_URL}]({REPO_URL})
- Writeups & contact: [hockeyROI.substack.com]({SUBSTACK_URL})
""")

# ----------------------------------------------------------------------------
# Header
st.markdown(f"""
<div class="brand-bar">
  <div>
    <div class="brand-title">HOCKEY<span class="brand-accent">ROI</span> &nbsp; / &nbsp; NHL ANALYTICS</div>
    <div class="brand-sub">PFR &middot; PUR &middot; PAR &middot; OUER &middot; DUER &middot; OAER &middot; DAER &middot; ROC &middot; ROL
        &nbsp;|&nbsp; pooled 2022-23 through 2025-26 &nbsp;|&nbsp; 5v5</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Metric tooltips strip (compact)
metric_keys = ["PFR", "PUR", "PAR", "OUER", "OAER", "DUER", "DAER", "ROC", "ROL"]
cols = st.columns(len(metric_keys))
for c, k in zip(cols, metric_keys):
    c.markdown(
        f"<div class='metric-card'><strong style='color:{ORANGE}'>{k}</strong>"
        f"<br/><span style='font-size:0.78rem'>{TOOLTIPS[k]}</span></div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Sidebar
st.sidebar.markdown(f"<h2 style='font-family:{HEADLINE_FONT};margin:0'>HockeyROI</h2>",
                    unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["Rankings", "Methodology"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<span style='font-size:0.85rem;color:{BLUE_LT}'>"
    f"Forwards and defensemen are ranked in <b>separate</b> pools.<br>"
    f"Top forward and top defenseman are both 1.0 within their own group.<br><br>"
    f"<b>PFR &lt; 0.95</b> = self made (green)<br>"
    f"<b>0.95 &le; PFR &le; 1.05</b> = consistent (yellow)<br>"
    f"<b>PFR &gt; 1.05</b> = system supported (red)"
    f"</span>", unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<span style='font-size:0.8rem;color:{BLUE_LT}'>"
    f"Last updated (regular): {ts_reg}<br>"
    f"Last updated (playoffs): {ts_pof}"
    f"</span>", unsafe_allow_html=True
)
st.sidebar.markdown(f"<a href='{REPO_URL}' target='_blank' style='color:{BLUE_LT}'>"
                    f"GitHub source &amp; methodology &rarr;</a><br>"
                    f"<a href='{SUBSTACK_URL}' target='_blank' style='color:{BLUE_LT}'>"
                    f"hockeyROI.substack.com &rarr;</a>",
                    unsafe_allow_html=True)

# ----------------------------------------------------------------------------
if page == "Rankings":
    tab_f, tab_d = st.tabs(["FORWARDS", "DEFENSEMEN"])
    with tab_f:
        render_group("forwards", "Forwards")
    with tab_d:
        render_group("defense",  "Defensemen")
else:
    render_methodology()
