"""E1 sweeps. Usage: python3 scripts/run_e1.py SEEDS [main|bond|fine|linked|all]
Every attack run is paired with an honest run on the same seed and settings."""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
import pandas as pd
from cti_sim.config import Params, SYSTEMS, E1_ATTACKS
from cti_sim.market import run_one

RATES = (0.02, 0.05, 0.10, 0.20)
FINE = tuple(round(0.01 * k, 2) for k in range(1, 21))


def grid(part, seeds):
    jobs = []
    att = ("honest",) + E1_ATTACKS
    if part in ("main", "all"):
        for s, a, r, K, T, k in itertools.product(SYSTEMS, att, RATES, (0, 1, 2, 3), (20, 40), seeds):
            jobs.append(("main", dict(system=s, attack=a, p_audit=r, K=K, T=T, seed=k)))
    if part in ("bond", "all"):
        for s, a, r, b, k in itertools.product(("R2", "R2_nodecl"), att, RATES, (0, 40, 100, 200), seeds):
            jobs.append(("bond", dict(system=s, attack=a, p_audit=r, bond=b, seed=k)))
    if part in ("fine", "all"):
        for s, a, r, k in itertools.product(("R1", "R2"), ("honest",) + E1_ATTACKS, FINE, seeds):
            jobs.append(("fine", dict(system=s, attack=a, p_audit=r, seed=k)))
    if part in ("linked", "all"):
        for s, a, r, k in itertools.product(SYSTEMS, ("honest", "A2"), RATES, seeds):
            jobs.append(("linked", dict(system=s, attack=a, p_audit=r, linked=True, seed=k)))
    if part in ("robust", "all"):
        for s, a, k in itertools.product(SYSTEMS, att, seeds):
            jobs.append(("robust", dict(system=s, attack=a, buyer_model="split", seed=k)))
    return jobs


def job(j):
    part, kw = j
    out = run_one(Params(**kw))
    out["part"] = part
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    part = sys.argv[2] if len(sys.argv) > 2 else "all"
    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    jobs = grid(part, range(lo, lo + n))
    t0 = time.time()
    with Pool(4) as pool:
        rows = pool.map(job, jobs, chunksize=64)
    df = pd.DataFrame(rows)
    out = f"results/e1_{part}_s{lo}-{lo+n-1}.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} runs in {time.time()-t0:.1f}s -> {out}")
