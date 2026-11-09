"""E1b reset diagnostic: verifiable regime, all systems, defaults, A1 continuation."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
import pandas as pd, json
from cti_sim.config import Params, SYSTEMS
from cti_sim.reset_diag import run_branch_pair

def job(a):
    s, seed, nok, noisy = a
    if noisy:
        th = json.load(open("results/e2_theta.json"))[f"{s}|0.1|0.05"]
        p = Params(system=s, attack="A1", seed=seed, p_audit=0.10, noisy=True, fn=0.1, fp=0.05, theta=th)
    else:
        p = Params(system=s, attack="A1", seed=seed)
    return run_branch_pair(p, newcomer_ok=nok)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    jobs = [(s, k, nok, False) for s in SYSTEMS for k in range(n) for nok in (True, False)]
    jobs += [(s, k, nok, True) for s in ("R1", "R2") for k in range(n) for nok in (True, False)]
    t0 = time.time()
    with Pool(4) as pool:
        df = pd.DataFrame(pool.map(job, jobs))
    df.to_csv("results/reset_diag.csv", index=False)
    print(f"{len(df)} branch pairs in {time.time()-t0:.1f}s")
    g = df.groupby(["noisy", "system", "newcomer_ok"]).agg(
        eligible=("eligible", "mean"), value_profit=("value_profit", "mean"),
        se_profit=("value_profit", "sem"), value_welfare=("value_welfare", "mean"),
        se_welfare=("value_welfare", "sem"))
    print(g.round(2).to_string())
