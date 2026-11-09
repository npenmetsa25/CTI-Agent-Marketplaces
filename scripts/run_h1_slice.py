"""H1 reputation-sensitive slice and dilution variant.
Slice: sharper choice (tau 0.25), harder cold start (prior Beta(2,2) = 0.5), honest incumbents start with
6 clean observations while coalition identities enter new, and without audits self-dealt transactions are free fakes."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
import pandas as pd
from cti_sim.config import Params, SYSTEMS
from cti_sim.market import run_one
from cti_sim.analysis import mean_ci

SLICE = dict(tau=0.25, alpha=2.0, beta=2.0, incumbent_obs=6, fake_selfdeal=True)

def job(a):
    market, s, att, k = a
    kw = {"default": {}, "sensitive": dict(SLICE), "free_all": dict(SLICE, fake_selfdeal_all=True)}[market]
    m = run_one(Params(system=s, attack=att, seed=k, **kw))
    m["market"] = market
    return m

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    jobs = list(itertools.product(("default", "sensitive", "free_all"), SYSTEMS, ("honest", "A1", "A3", "A3d"), range(n)))
    with Pool(4) as pool:
        df = pd.DataFrame(pool.map(job, jobs))
    df.to_csv("results/h1_slice.csv", index=False)
    base = df[df.attack == "honest"].set_index(["market", "system", "seed"])
    att = df[df.attack != "honest"].copy()
    idx = list(zip(att.market, att.system, att.seed))
    att["gain"] = att.coal_profit.values - base.loc[idx, "coal_profit"].values
    rows = []
    for (mk, s), g in att.groupby(["market", "system"]):
        a1 = g[g.attack == "A1"].set_index("seed").gain
        for a in ("A3", "A3d"):
            d = g[g.attack == a].set_index("seed").gain - a1
            m, lo, hi = mean_ci(d.values)
            rows.append(dict(market=mk, system=s, comparison=f"{a}-A1", mean=m, lo=lo, hi=hi))
        if s == "R2":
            pass
    # paired difference audit-only minus CTI in the value of purchased history (same seeds)
    for mk in ("default", "sensitive", "free_all"):
        for a in ("A3", "A3d"):
            v = {}
            for s in ("R1", "R2"):
                g = att[(att.market == mk) & (att.system == s)].pivot(index="seed", columns="attack", values="gain")
                v[s] = g[a] - g["A1"]
            m, lo, hi = mean_ci((v["R1"] - v["R2"]).values)
            rows.append(dict(market=mk, system="R1-R2", comparison=f"{a}-A1", mean=m, lo=lo, hi=hi))
    out = pd.DataFrame(rows)
    out.to_csv("results/h1_value_of_purchases.csv", index=False)
    pd.set_option("display.width", 200)
    print(att.groupby(["market", "system", "attack"]).agg(gain=("gain", "mean"), coal_tx=("coal_tx_honest", "mean"),
          selfp=("self_purchases", "mean"), outside=("outside_share", "mean")).round(1).unstack("attack").to_string())
    print("\nValue of purchased history (mean [95% CI]):")
    out["ci"] = out.apply(lambda r: f"{r['mean']:.1f} [{r.lo:.1f}, {r.hi:.1f}]", axis=1)
    print(out.pivot(index=["market", "system"], columns="comparison", values="ci").to_string())
