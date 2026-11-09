"""E2 calibration: smallest theta on a grid such that an honest identity present for the whole game
(initial identity, not retired on schedule) has a false-revocation probability <= 5%. Calibration seeds 1000-1199 (fresh seeds for testing)."""
import sys, os, itertools, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
import numpy as np, pandas as pd
from cti_sim.config import Params
from cti_sim.market import run_one

THETAS = [round(x, 2) for x in np.arange(0.0, 8.01, 0.25)]

def job(a):
    s, r, fp, th, seed = a
    m = run_one(Params(system=s, attack="honest", p_audit=r, noisy=True, fn=0.1, fp=fp, theta=th, seed=seed))
    return dict(system=s, p_audit=r, fp=fp, theta=th, seed=seed,
                false=m["false_full_exposure"], ids=m["n_full_exposure"])

if __name__ == "__main__":
    jobs = list(itertools.product(("R1",), (0.05, 0.10, 0.20), (0.02, 0.05), THETAS, range(1000, 1200)))
    with Pool(4) as pool:
        df = pd.DataFrame(pool.map(job, jobs, chunksize=200))
    g = df.groupby(["system", "p_audit", "fp", "theta"]).apply(lambda x: x.false.sum() / x.ids.sum()).rename("false_rate").reset_index()
    g.to_csv("results/e2_calibration.csv", index=False)
    chosen = {}
    for (s, r, fp), sub in g.groupby(["system", "p_audit", "fp"]):
        ok = sub[sub.false_rate <= 0.05].sort_values("theta")
        th = float(ok.theta.iloc[0]) if len(ok) else None
        chosen[f"{s}|{r}|{fp}"] = th
        print(s, r, fp, "theta =", th, " false-revocation rate =", None if th is None else round(float(ok.false_rate.iloc[0]), 4))
    for k in list(chosen):                       # honest-only runs are identical across R1 and R2
        chosen[k.replace("R1|", "R2|")] = chosen[k]
    json.dump(chosen, open("results/e2_theta.json", "w"), indent=1)
