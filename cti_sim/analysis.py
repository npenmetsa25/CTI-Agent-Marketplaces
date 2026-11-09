"""Paired gains, confidence intervals and deterrence thresholds."""
import numpy as np
import pandas as pd

KEYS = ["part", "system", "p_audit", "K", "T", "bond", "linked", "buyer_model", "seed"]


def paired(df):
    """Attack rows with coalition gain and welfare drop vs the honest run on the same seed/settings."""
    base = df[df.attack == "honest"].set_index(KEYS)
    att = df[df.attack != "honest"].copy()
    idx = [tuple(r) for r in att[KEYS].values]
    att["gain"] = att.coal_profit.values - base.loc[idx, "coal_profit"].values
    att["welfare_drop"] = base.loc[idx, "welfare_pre_pr"].values - att.welfare_pre_pr.values
    att["welfare_drop_post"] = base.loc[idx, "welfare_post_pr"].values - att.welfare_post_pr.values
    return att


def mean_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def sustained_threshold(rates, gains_by_rate):
    """Lowest tested rate from which the attack stops paying (mean gain <= 0) at every higher
    tested rate. A payoff-responsive attacker that stops cheating has gain exactly 0, which
    counts as deterred. Returns np.inf if not reached by the largest rate."""
    means = [np.mean(g) for g in gains_by_rate]
    thr = np.inf
    for r, m in zip(reversed(rates), reversed(means)):
        if m <= 1e-9:
            thr = r
        else:
            break
    return thr


def threshold_ci(sub, n_boot=1000, seed=0):
    """sub: rows for one system/attack on a rate grid, columns p_audit, seed, gain.
    Bootstrap resamples seeds (common across rates, since exogenous streams are shared)."""
    rates = sorted(sub.p_audit.unique())
    piv = sub.pivot_table(index="seed", columns="p_audit", values="gain")[rates]
    G = piv.values                       # seeds x rates
    point = sustained_threshold(rates, [G[:, j] for j in range(len(rates))])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, G.shape[0], G.shape[0])
        Gb = G[idx]
        bs.append(sustained_threshold(rates, [Gb[:, j] for j in range(len(rates))]))
    bs = np.array(bs)
    reached = np.isfinite(bs)
    srt = np.sort(bs)
    lo = srt[int(0.025 * len(srt))]
    hi = srt[min(len(srt) - 1, int(np.ceil(0.975 * len(srt))) - 1)]
    return point, lo, hi, reached.mean(), piv, bs


def label(point, lo, hi):
    if not np.isfinite(point):
        return "not reached"
    if not np.isfinite(hi):
        return f"{point:.2f} (uncertain: upper bound not reached)"
    return f"{point:.2f} [{lo:.2f}, {hi:.2f}]"


def threshold_diff(sub_a, sub_b, n_boot=1000, seed=0):
    """Bootstrap the difference p*(a) - p*(b), resampling seeds jointly (shared streams)."""
    rates = sorted(sub_a.p_audit.unique())
    A = sub_a.pivot_table(index="seed", columns="p_audit", values="gain")[rates]
    B = sub_b.pivot_table(index="seed", columns="p_audit", values="gain")[rates]
    common = A.index.intersection(B.index)
    A, B = A.loc[common].values, B.loc[common].values
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, len(common), len(common))
        ta = sustained_threshold(rates, [A[i, j] for j in range(len(rates))])
        tb = sustained_threshold(rates, [B[i, j] for j in range(len(rates))])
        ds.append(ta - tb if np.isfinite(ta) and np.isfinite(tb) else (-np.inf if np.isfinite(ta) else np.inf))
    ds = np.sort(np.array(ds))
    return ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1], np.mean(ds < 0)
