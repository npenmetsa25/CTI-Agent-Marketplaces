"""Checkpoint A: E0 terminal-round verification. Writes results/e0.csv."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
import pandas as pd
from cti_sim.e0 import cells, run_cell

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    t0 = time.time()
    with Pool(4) as pool:
        rows = pool.map(run_cell, cells(n))
    df = pd.DataFrame(rows)
    df.to_csv("results/e0.csv", index=False)
    print(f"{len(df)} cells, {n} draws each, {time.time()-t0:.1f}s")
    print("exact == closed form:", df.exact_ok.all())
    print("Monte Carlo within 4 SE:", df.mc_ok.sum(), "/", len(df))
    print("policy == formula argmax:", df.policy_ok.all())
    sub = df[(df.p == 0.05) & (df.b == 100)][["d", "g", "closed", "mc", "se", "policy_d"]]
    print(sub.round(3).to_string(index=False))
