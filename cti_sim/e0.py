"""E0: terminal-round verification (an implementation check, not behavioural evidence).

A single final round (F = 0). An identity commits d violations, each gaining g, and
loses its bond b if at least one is audited. We compare:
  1. the closed form   d*g - [1-(1-p)^d] * b
  2. exact enumeration over all 2^d audit outcomes
  3. Monte Carlo through the market's own audit-and-settlement code
and check that the payoff-responsive rule cheats exactly where the formula predicts a gain.
"""
from itertools import product
import numpy as np

from .config import Params
from .market import Market
from .policies import expected_delta, choose_violations

GAINS = (1, 2, 4, 8)            # low treatment billed as high at 4, 5, 7, 11
DS = (1, 3)
BONDS = (0, 40, 100, 200)
RATES = (0.02, 0.05, 0.10, 0.20)


def exact_enumeration(d, g, b, p):
    total = 0.0
    for pattern in product((0, 1), repeat=d):
        k = sum(pattern)
        prob = (p ** k) * ((1 - p) ** (d - k))
        total += prob * (d * g - (b if k > 0 else 0.0))
    return total


class _Harness:
    """One coalition identity in a one-slot R2 market, final round."""

    def __init__(self, b, p, seed):
        self.mk = Market(Params(system="R2", attack="A1", n_slots=1, K=1, T=1, bond=b,
                                p_audit=p, seed=seed, endow_coalition=1e9))
        self.ident = self.mk.slots[0]
        self.mk.rng_audit = np.random.default_rng(seed)   # persistent stream across draws
        self.b = b

    def draw(self, d, g):
        mk, ident = self.mk, self.ident
        ident.active, ident.revoked, ident.bond = True, False, float(self.b)
        mk.slots[0] = ident
        ident.unsettled = d
        txs = [(ident, "honest", True, 1.0)] * d
        mk._end_of_round(txs)
        forfeited = self.b - ident.bond
        return d * g - forfeited


def monte_carlo(d, g, b, p, n, seed=0):
    hz = _Harness(b, p, seed)
    vals = np.fromiter((hz.draw(d, g) for _ in range(n)), float, count=n)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(n)


def run_cell(args):
    d, g, b, p, n, seed = args
    closed = expected_delta(d, g, p, b)
    exact = exact_enumeration(d, g, b, p)
    mc, se = monte_carlo(d, g, b, p, n, seed)
    best_d, best_g = choose_violations(d, (g,), p, b)
    # reference argmax with the same tie rule as the policy: no cheating at zero expected gain
    vals = [0.0] + [round(expected_delta(k, g, p, b), 9) for k in range(1, d + 1)]
    best_by_formula = max(range(d + 1), key=lambda k: (vals[k], -k)) if max(vals) > 0 else 0
    return dict(d=d, g=g, b=b, p=p, closed=closed, exact=exact, mc=mc, se=se,
                exact_ok=abs(closed - exact) < 1e-9,
                mc_ok=abs(mc - exact) <= max(4 * se, 1e-9),
                policy_d=best_d, formula_d=best_by_formula,
                policy_ok=(best_d == best_by_formula))


def cells(n=100_000):
    return [(d, g, b, p, n, 1000 + i) for i, (d, g, b, p) in
            enumerate(product(DS, GAINS, BONDS, RATES))]
