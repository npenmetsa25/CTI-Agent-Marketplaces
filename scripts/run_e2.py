"""E2 noisy regime on fresh seeds with frozen theta. Attacks: A1 (r=0.3), A1S (sub-threshold)."""
import sys, os, itertools, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
import numpy as np, pandas as pd
from cti_sim.config import Params
from cti_sim.market import run_one
from cti_sim.analysis import mean_ci
TH = json.load(open("results/e2_theta.json"))

def job(a):
    s, r, fn, fp, att, seed = a
    th = TH[f"{s}|{r}|{fp}"]
    m = run_one(Params(system=s, attack=att, p_audit=r, noisy=True, fn=fn, fp=fp, theta=th, seed=seed))
    m["theta_used"] = th
    return m

if __name__ == "__main__":
    jobs = list(itertools.product(("R1", "R2"), (0.05, 0.10, 0.20), (0.1, 0.3), (0.02, 0.05),
                                  ("honest", "A1", "A1S"), range(int(sys.argv[1]) if len(sys.argv) > 1 else 30)))
    with Pool(4) as pool:
        df = pd.DataFrame(pool.map(job, jobs, chunksize=50))
    df.to_csv("results/e2_runs.csv", index=False)
    keys = ["system", "p_audit", "fn", "fp", "seed"]
    base = df[df.attack == "honest"].set_index(keys)
    att = df[df.attack != "honest"].copy()
    idx = [tuple(r) for r in att[keys].values]
    att["gain"] = att.coal_profit.values - base.loc[idx, "coal_profit"].values
    att["harm"] = att.harm_viol
    att.to_csv("results/e2_paired.csv", index=False)
    hon = df[df.attack == "honest"].groupby(["system", "p_audit", "fp"]).apply(
        lambda x: x.false_sanctions.sum() / x.n_honest_ids.sum(), include_groups=False)
    print("Test-seed false revocation rate (honest runs):"); print(hon.round(3).to_string())
    g = att.groupby(["attack", "system", "p_audit"]).agg(gain=("gain", "mean"), harm=("harm", "mean"),
            viol=("n_viol", "mean"), false_rev=("false_sanctions", "mean"))
    print(g.round(1).unstack("system").to_string())
