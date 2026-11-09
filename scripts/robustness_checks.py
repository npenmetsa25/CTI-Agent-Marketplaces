"""Paired intervals for E1b, per-cell E2 results, and false revocation by exposure with seed-clustered bootstrap."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
import numpy as np, pandas as pd
from cti_sim.config import Params
from cti_sim.market import run_one
from cti_sim.analysis import mean_ci
pd.set_option("display.width", 220)

# ---- E1b: reset value and the effect of removing newcomer eligibility (paired by seed)
r = pd.read_csv("results/reset_diag.csv")
r = r[r.eligible]
rows = []
for (noisy, s), g in r.groupby(["noisy", "system"]):
    w = g.pivot(index="seed", columns="newcomer_ok", values="value_profit").dropna()
    a = mean_ci(w[True]); b = mean_ci(w[False]); d = mean_ci((w[True] - w[False]).values)
    rows.append(dict(noisy=noisy, system=s, n=len(w), reset_with_quota=a, reset_without=b, effect_of_quota=d))
t = pd.DataFrame(rows)
fmt = lambda x: f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]"
for c in ("reset_with_quota", "reset_without", "effect_of_quota"):
    t[c] = t[c].map(fmt)
t.to_csv("results/e1b_paired_intervals.csv", index=False)
print("E1b value of reset (paired, 95% CI):"); print(t.to_string(index=False))

# ---- E2: per noise cell
e = pd.read_csv("results/e2_paired.csv")
rows = []
for (a, s, p, fn, fp), g in e.groupby(["attack", "system", "p_audit", "fn", "fp"]):
    m, lo, hi = mean_ci(g.gain.values)
    status = "unprofitable" if hi < 0 else ("profitable" if lo > 0 else "unresolved")
    rows.append(dict(attack=a, system=s, p=p, fn=fn, fp=fp, gain=f"{m:.1f} [{lo:.1f}, {hi:.1f}]", status=status))
c = pd.DataFrame(rows)
c.to_csv("results/e2_per_cell.csv", index=False)
print("\nE2 per cell, CTI (R2):")
print(c[c.system == "R2"].pivot_table(index=["attack", "p"], columns=["fn", "fp"], values="status", aggfunc="first").to_string())
print("\nE2 per cell, audit-only (R1):")
print(c[c.system == "R1"].pivot_table(index=["attack", "p"], columns=["fn", "fp"], values="status", aggfunc="first").to_string())

# ---- False revocation by exposure, seed-clustered bootstrap
TH = json.load(open("results/e2_theta.json"))
def job(a):
    r_, fp, k = a
    m = run_one(Params(system="R1", attack="honest", p_audit=r_, noisy=True, fn=0.1, fp=fp,
                       theta=TH[f"R1|{r_}|{fp}"], seed=k))
    return dict(p=r_, fp=fp, seed=k, f_all=m["false_sanctions"], n_all=m["n_honest_ids"],
                f_full=m["false_full_exposure"], n_full=m["n_full_exposure"])
with Pool(4) as pool:
    fr = pd.DataFrame(pool.map(job, list(itertools.product((0.05, 0.10, 0.20), (0.02, 0.05), range(200)))))
def cluster_ratio(x, n, B=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x); n = np.asarray(n)
    idx = rng.integers(0, len(x), (B, len(x)))
    bs = x[idx].sum(1) / n[idx].sum(1)
    return x.sum() / n.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
rows = []
for (p, fp), g in fr.groupby(["p", "fp"]):
    rows.append(dict(p=p, fp=fp, all_identities=fmt(cluster_ratio(g.f_all, g.n_all)),
                     full_game_initial=fmt(tuple(100 * v for v in cluster_ratio(g.f_full, g.n_full))),
                     n_full=int(g.n_full.sum())))
fr_t = pd.DataFrame(rows)
fr_t["all_identities"] = [fmt(tuple(100 * v for v in cluster_ratio(g.f_all, g.n_all))) for _, g in fr.groupby(["p", "fp"])]
fr_t.to_csv("results/e2_false_revocation_exposure.csv", index=False)
print("\nFalse revocation (%), seed-clustered 95% CI, 200 held-out seeds:"); print(fr_t.to_string(index=False))
