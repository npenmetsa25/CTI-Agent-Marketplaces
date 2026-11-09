import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from cti_sim.analysis import paired, mean_ci, threshold_ci, threshold_diff, label
pd.set_option("display.width", 250)
df = pd.read_csv("results/e1_all_s0-29.csv")
att = paired(df)
att.to_csv("results/e1_paired.csv", index=False)
print("rows", len(df), "paired attack rows", len(att), "capital blocked", df.capital_blocked.sum())

# Figure 1 table: defaults (K=2, T=40, p=0.05)
d = att[(att.part == "main") & (att.K == 2) & (att["T"] == 40) & (att.p_audit == 0.05)]
rows = []
for (s, a), g in d.groupby(["system", "attack"]):
    m, lo, hi = mean_ci(g.gain)
    w, wl, wh = mean_ci(g.welfare_drop)
    rows.append(dict(system=s, attack=a, gain=m, lo=lo, hi=hi, welfare_drop=w, w_lo=wl, w_hi=wh,
                     viol=g.n_viol.mean(), revoc=g.revocations.mean()))
t1 = pd.DataFrame(rows)
t1.to_csv("results/table_fig1.csv", index=False)
print("\nCoalition gain at defaults (mean [95% CI]):")
print(t1.pivot(index="system", columns="attack", values="gain").round(1).to_string())
a3 = t1.pivot(index="system", columns="attack", values="gain")
print("\nA3 - A1 (purchased history):", (a3["A3"] - a3["A1"]).round(1).to_dict())

# Figure 2: thresholds on the fine grid
print("\nDeterrence thresholds p_a* (fine grid, sustained, 95% bootstrap):")
out = []
f = att[att.part == "fine"]
for (s, a), g in f.groupby(["system", "attack"]):
    pt, lo, hi, reach, piv, bs = threshold_ci(g)
    out.append(dict(system=s, attack=a, point=pt, lo=lo, hi=hi, reach=reach, label=label(pt, lo, hi)))
th = pd.DataFrame(out)
th.to_csv("results/thresholds_fine.csv", index=False)
print(th[["system", "attack", "label", "reach"]].to_string(index=False))
print("\nCTI (R2) minus audit-only (R1) threshold; -inf = CTI reached, R1 not reached by 0.20:")
for a in sorted(f.attack.unique()):
    lo, hi, pneg = threshold_diff(f[(f.system == "R2") & (f.attack == a)], f[(f.system == "R1") & (f.attack == a)])
    print(f"  {a:4s} 95% CI [{lo}, {hi}]   share of bootstrap draws with CTI lower: {pneg:.3f}")
v = f[f.attack == "A1R"].groupby(["system", "p_audit"]).n_viol.mean().unstack(0)
print("\nA1-R mean violations by audit rate (selected):")
print(v.loc[[0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]].round(1).to_string())

# bond sweep summary at p=0.05
b = att[(att.part == "bond") & (att.p_audit == 0.05)]
print("\nBond sweep, coalition gain at p=0.05:")
print(b.pivot_table(index=["system", "attack"], columns="bond", values="gain", aggfunc="mean").round(1).to_string())
