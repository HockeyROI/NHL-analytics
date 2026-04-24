#!/usr/bin/env python3
"""Build side-by-side comparison of 3-season vs 5-season results."""
import pandas as pd

OUT = "/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/NFI/output"

u3 = pd.read_csv(f"{OUT}/horse_race_univariate_3season.csv").rename(columns={"R":"R_3s","var_explained":"R2_3s","n":"n_3s","p":"p_3s"})
u5 = pd.read_csv(f"{OUT}/horse_race_univariate.csv").rename(columns={"R":"R_5s","var_explained":"R2_5s","n":"n_5s","p":"p_5s"})
u = u3.merge(u5, on="metric", how="outer")
u["dR2"] = (u["R2_5s"] - u["R2_3s"]).round(4)
u = u.sort_values("R2_5s", ascending=False)
u[["metric","R_3s","R2_3s","n_3s","R_5s","R2_5s","n_5s","dR2"]].to_csv(f"{OUT}/comparison_univariate.csv", index=False)
print("Univariate comparison (top by 5s R^2):")
print(u[["metric","R_3s","R2_3s","R_5s","R2_5s","dR2"]].head(20).to_string(index=False))

m3 = pd.read_csv(f"{OUT}/horse_race_multivariate_3season.csv").rename(columns={"beta":"beta_3s","se":"se_3s","t":"t_3s","p":"p_3s"})
m5 = pd.read_csv(f"{OUT}/horse_race_multivariate.csv").rename(columns={"beta":"beta_5s","se":"se_5s","t":"t_5s","p":"p_5s"})
m = m3.merge(m5, on="term", how="outer")
m["dbeta"] = (m["beta_5s"] - m["beta_3s"]).round(4)
m.to_csv(f"{OUT}/comparison_multivariate.csv", index=False)
print("\nMultivariate comparison:")
print(m[["term","beta_3s","beta_5s","dbeta","p_3s","p_5s"]].to_string(index=False))

c3 = pd.read_csv(f"{OUT}/pillar_ci_flagging_3season.csv").rename(columns={c:f"{c}_3s" for c in ["players","ci_clear","ci_clear_pct","below_50pct"]})
c5 = pd.read_csv(f"{OUT}/pillar_ci_flagging.csv").rename(columns={c:f"{c}_5s" for c in ["players","ci_clear","ci_clear_pct","below_50pct"]})
c = c3.merge(c5, on="pillar", how="outer")
c.to_csv(f"{OUT}/comparison_ci_flagging.csv", index=False)
print("\nCI flagging comparison:")
print(c.to_string(index=False))
