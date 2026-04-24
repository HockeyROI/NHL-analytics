#!/usr/bin/env python3
"""League-wide conversion rate (goals/attempts) by zone.
ES regulation only, 5 seasons pooled."""
import math
import pandas as pd

OUT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/NFI/output"
sh = pd.read_csv(f"{OUT}/shots_tagged.csv")

# filter ES only (regulation + empty-net-dropped already applied upstream)
es = sh[sh["state"]=="ES"].copy()
print(f"ES regulation shots (5 seasons): {len(es):,}")

# Conventional HD: x>69, y in [-22,22]
es["HD_conv"] = ((es["x_coord_norm"]>69) & (es["y_coord_norm"].between(-22,22))).astype(int)

def wilson(k, n, z=1.96):
    if n == 0: return (0,0,0)
    p = k/n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0,c-h), min(1,c+h))

rows = []
# HD conventional
hd = es[es["HD_conv"]==1]
att = len(hd); gl = int(hd["is_goal_i"].sum())
p, lo, hi = wilson(gl, att)
rows.append({"zone":"HD_conventional (x>69, y in ±22)","attempts":att,"goals":gl,
             "conversion_pct":round(p*100,3),"ci_lo_pct":round(lo*100,3),"ci_hi_pct":round(hi*100,3)})

# NFI zones
for z in ["CNFI","MNFI","FNFI","Wide","lane_other"]:
    sub = es[es["zone"]==z]
    att = len(sub); gl = int(sub["is_goal_i"].sum())
    p, lo, hi = wilson(gl, att)
    rows.append({"zone":z,"attempts":att,"goals":gl,
                 "conversion_pct":round(p*100,3),"ci_lo_pct":round(lo*100,3),"ci_hi_pct":round(hi*100,3)})

# TNFI combined
tn = es[es["zone"].isin(["CNFI","MNFI","FNFI"])]
att = len(tn); gl = int(tn["is_goal_i"].sum())
p, lo, hi = wilson(gl, att)
rows.append({"zone":"TNFI (CNFI+MNFI+FNFI)","attempts":att,"goals":gl,
             "conversion_pct":round(p*100,3),"ci_lo_pct":round(lo*100,3),"ci_hi_pct":round(hi*100,3)})

# All
att = len(es); gl = int(es["is_goal_i"].sum())
p, lo, hi = wilson(gl, att)
rows.append({"zone":"ALL ES REG","attempts":att,"goals":gl,
             "conversion_pct":round(p*100,3),"ci_lo_pct":round(lo*100,3),"ci_hi_pct":round(hi*100,3)})

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/zone_conversion_rates.csv", index=False)
print(df.to_string(index=False))
