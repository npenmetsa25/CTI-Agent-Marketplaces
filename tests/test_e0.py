import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cti_sim.e0 import cells, run_cell, exact_enumeration
from cti_sim.policies import expected_delta


def test_closed_form_matches_enumeration():
    for d, g, b, p, _, _ in cells(1):
        assert abs(expected_delta(d, g, p, b) - exact_enumeration(d, g, b, p)) < 1e-9


def test_plan_numbers():
    q = 1 - 0.95 ** 3
    assert abs(3 * 4 / q - 84.14) < 0.01 and abs(3 * 8 / q - 168.27) < 0.01
    assert abs(expected_delta(3, 4, 0.05, 100) + 2.26) < 0.01
    assert abs(expected_delta(3, 8, 0.05, 100) - 9.74) < 0.01


def test_monte_carlo_and_policy_small():
    for c in cells(4000)[::7]:
        r = run_cell(c)
        assert r["exact_ok"] and r["mc_ok"] and r["policy_ok"], r
