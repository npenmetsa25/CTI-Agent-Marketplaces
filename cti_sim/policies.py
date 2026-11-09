"""Seller decision rules.

Actions are tuples (given, billed, price) with treatments L=0, H=1.
"""
L, H = 0, 1


def expected_delta(d: int, g: float, p_detect: float, stake: float) -> float:
    """Sanction-channel expected gain from d violations, each gaining g, when the
    identity loses `stake` (bond + continuation value) if at least one is caught."""
    return d * g - (1.0 - (1.0 - p_detect) ** d) * stake


def choose_violations(n: int, gains, p_detect: float, stake: float):
    """Approximate payoff-responsive rule (A1-R): enumerate every feasible action,
    d = 0..n violations of each gain type, and return the (d, g) with the highest
    expected gain. Ties go to fewer violations. d = 0 means no cheating."""
    best = (0, None, 0.0)
    for g in gains:
        for d in range(1, n + 1):
            v = expected_delta(d, g, p_detect, stake)
            if v > best[2] + 1e-12:
                best = (d, g, v)
    return best[0], best[1]


def honest_action(need_big: bool, pL: float, pH: float):
    t = H if need_big else L
    return (t, t, pH if t == H else pL)
