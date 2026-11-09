import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cti_sim.config import Params, SYSTEMS, ATTACKS
from cti_sim.market import Market


def test_all_combinations_run_and_conserve_money():
    for s in SYSTEMS:
        for a in ATTACKS:
            for seed in range(3):
                mk = Market(Params(system=s, attack=a, seed=seed))
                m = mk.run()                      # raises on any invariant violation
                assert m["capital_blocked"] == 0, (s, a, seed)


def test_honest_market_has_no_violations_or_revocations():
    for s in SYSTEMS:
        m = Market(Params(system=s, attack="honest", seed=5)).run()
        assert m["n_viol"] == 0 and m["revocations"] == 0 and m["unexpected"] == 0


def test_largest_bond_does_not_block_capital():
    for s in SYSTEMS:
        for a in ATTACKS:
            m = Market(Params(system=s, attack=a, bond=200, seed=2)).run()
            assert m["capital_blocked"] == 0, (s, a)


def test_linked_lineage_and_noisy_run():
    Market(Params(system="R2", attack="A2", linked=True, seed=3)).run()
    Market(Params(system="R1", attack="A1", noisy=True, fn=0.1, fp=0.05, theta=2.0, seed=3)).run()
