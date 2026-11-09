"""E1b: value of voluntary reset (paired continuations from a cloned state).

At the start of the final `endgame` rounds, clone the market. In one branch keep the
selected coalition identity; in the other retire it (bond settled normally) and open a
replacement from the remaining quota. Both branches then follow the same attack policy,
with identical exogenous randomness (per-round streams).
"""
import copy
from .config import Params
from .market import Market


def run_branch_pair(p: Params, newcomer_ok=True):
    mk = Market(p)
    cp = p.T - p.endgame
    while mk.t < cp:
        mk.step()
    coal = mk.coal
    cands = sorted(mk.active_identities("coalition"), key=lambda x: x.slot)
    bond = p.bond if p.system in ("R2", "R2_nodecl") else 0.0
    eligible = bool(cands) and coal.replacements_left > 0 and not coal.blocked
    out = dict(system=p.system, seed=p.seed, noisy=p.noisy, newcomer_ok=newcomer_ok,
               eligible=eligible)
    if not eligible:
        return out
    retain, reset = mk, copy.deepcopy(mk)
    target = [x for x in reset.active_identities("coalition") if x.slot == cands[0].slot][0]
    reset._retire(target)
    new = reset._create_identity(reset.coal, target.slot, reset=True)
    if new is None:                       # funds insufficient (should not happen)
        out["eligible"] = False
        return out
    new.newcomer_ok = newcomer_ok
    reset.coal.replacements_left -= 1
    w0 = mk.m["welfare_pre"]
    res = {}
    for name, m in (("retain", retain), ("reset", reset)):
        f0 = m.coal.funds + sum(x.bond for x in m.active_identities("coalition"))
        while m.t < p.T:
            m.step()
        m._finish()
        res[name] = dict(profit=m.coal.funds - f0, welfare=m.m["welfare_pre"] - w0,
                         matches=None)
    out.update(value_profit=res["reset"]["profit"] - res["retain"]["profit"],
               value_welfare=res["reset"]["welfare"] - res["retain"]["welfare"],
               retain_profit=res["retain"]["profit"], reset_profit=res["reset"]["profit"])
    return out
