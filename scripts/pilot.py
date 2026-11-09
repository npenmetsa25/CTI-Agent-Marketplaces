"""Checkpoint B: pilot at default settings. 30 paired seeds per system x attack."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
import pandas as pd
from cti_sim.config import Params, SYSTEMS, E1_ATTACKS
from cti_sim.market import run_one

def job(args):
    s, a, seed = args
    return run_one(Params(system=s, attack=a, seed=seed))

if __name__ == "__main__":
    seeds = range(30)
    jobs = [(s, a, k) for s in SYSTEMS for a in ("honest",) + E1_ATTACKS for k in seeds]
    t0 = time.time()
    with Pool(4) as pool:
        df = pd.DataFrame(pool.map(job, jobs))
    el = time.time() - t0
    df.to_csv("results/pilot.csv", index=False)
    base = df[df.attack == "honest"].set_index(["system", "seed"])
    df["coal_gain"] = df.coal_profit.values - base.loc[list(zip(df.system, df.seed)), "coal_profit"].values
    df["welfare_drop"] = base.loc[list(zip(df.system, df.seed)), "welfare_pre_pr"].values - df.welfare_pre_pr.values
    g = df.groupby(["system", "attack"]).agg(
        coal_gain=("coal_gain", "mean"), gain_se=("coal_gain", "sem"),
        welfare=("welfare_pre_pr", "mean"), w_drop=("welfare_drop", "mean"),
        viol=("n_viol", "mean"), revoc=("revocations", "mean"), resets=("resets", "mean"),
        outside=("outside_share", "mean"), newc_m=("newcomer_matches", "mean"),
        newc_a=("newcomer_audits", "mean"), hon_profit=("honest_profit_per_seller", "mean"),
        unexp=("unexpected", "mean"), cap=("capital_blocked", "sum"))
    pd.set_option("display.width", 250)
    print(f"{len(df)} runs in {el:.1f}s ({1000*el/len(df):.1f} ms/run on 4 cores)")
    print(g.round(2).to_string())
