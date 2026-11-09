"""Credence-goods marketplace with a trusted gateway.

One round: start-of-round identity changes -> matching -> transactions ->
hidden random audits, revocation and bond settlement. Invariants are checked
as the simulation runs (money conservation, settlement before bond return,
capacity). Welfare counts honest buyers only.
"""
import math
import numpy as np

from .config import Params, uses_audits, terms_bind, has_bond
from .policies import L, H, honest_action, choose_violations

COALITION_ATTACKS = ("A1", "A1R", "A2", "A2v", "A3", "A5", "A1S", "A3d")   # attacks with a cheating policy


class InvariantError(AssertionError):
    pass


class Controller:
    __slots__ = ("cid", "kind", "funds", "endowment", "replacements_left", "blocked")

    def __init__(self, cid, kind, endowment, M):
        self.cid = cid
        self.kind = kind            # "honest" or "coalition"
        self.funds = endowment      # spendable funds (bonds sit in escrow, not here)
        self.endowment = endowment
        self.replacements_left = M
        self.blocked = False        # linked lineage: no new identities


class Identity:
    __slots__ = ("iid", "ctrl", "slot", "birth", "pL", "pH", "bond", "active", "revoked",
                 "is_reset", "is_entrant", "a", "c", "f", "n_r", "pos_r", "bill_sum",
                 "bill_n", "served_n", "solved_n", "recent", "unsettled",
                 "early_matches", "early_audits", "newcomer_ok", "sched_retired", "pending_menu")

    def __init__(self, iid, ctrl, slot, birth, pL, pH):
        self.iid, self.ctrl, self.slot, self.birth = iid, ctrl, slot, birth
        self.pL, self.pH = pL, pH
        self.bond = 0.0
        self.active, self.revoked = True, False
        self.is_reset = False
        self.is_entrant = False
        self.a = self.c = self.f = 0            # audits, compliant audits, flags
        self.n_r = self.pos_r = 0               # ratings, positive ratings
        self.bill_sum, self.bill_n = 0.0, 0
        self.served_n = self.solved_n = 0
        self.recent = []                        # matches per round
        self.unsettled = 0
        self.early_matches = self.early_audits = 0
        self.newcomer_ok = True
        self.sched_retired = False
        self.pending_menu = None


class Market:
    def __init__(self, p: Params, seller_policy=None):
        """seller_policy: optional callable (market, identity, matched_list) -> actions,
        used by the E3 LLM harness. Coalition buyer accounts are always served honestly."""
        self.p = p
        self.seller_policy = seller_policy
        self.rng_audit = np.random.default_rng(p.seed)   # replaced each round by step()
        self.t = 0
        self._set_streams()
        self.reserve = 0.0
        self.controllers = []
        self.identities = []
        self.slots = [None] * p.n_slots
        self.slot_kind = [None] * p.n_slots
        # money accounting for the conservation invariant
        self.initial_money = 0.0
        self.honest_payments = 0.0
        self.costs = 0.0
        self.comp_honest = 0.0
        self.m = dict(welfare_pre=0.0, outside_n=0, honest_tx=0, coal_tx_honest=0,
                      harm_viol=0.0, n_viol=0, n_viol_shared=0, unexpected=0.0,
                      unexpected_loss=0.0, unexpected_new=0.0, audits=0, revocations=0,
                      false_sanctions=0, capital_blocked=0, resets=0, entrants=0,
                      self_purchases=0, rival_purchases=0, n_unexpected=0)
        self.coal = self._new_controller("coalition", p.endow_coalition)
        for s in range(p.n_slots):
            if s < p.K:
                self._create_identity(self.coal, s)
            else:
                ident = self._honest_entrant(s, initial=True)
                k = p.incumbent_obs               # established incumbents (reputation-sensitive slice)
                ident.a = ident.c = ident.n_r = ident.pos_r = k

    # ------------------------------------------------------------ helpers
    def _new_controller(self, kind, endowment):
        c = Controller(len(self.controllers), kind, endowment, self.p.M)
        self.controllers.append(c)
        self.initial_money += endowment
        return c

    def _create_identity(self, ctrl, slot, reset=False):
        p = self.p
        bond = p.bond if has_bond(p.system) else 0.0
        if ctrl.funds < bond:
            self.m["capital_blocked"] += 1
            return None
        ident = Identity(len(self.identities), ctrl, slot, self.t,
                         p.cost_L + p.markup, p.cost_H + p.markup)
        ctrl.funds -= bond
        ident.bond = bond
        ident.is_reset = reset
        self.identities.append(ident)
        self.slots[slot] = ident
        self.slot_kind[slot] = ctrl.kind
        return ident

    def _honest_entrant(self, slot, initial=False):
        ctrl = self._new_controller("honest", self.p.endow_honest)
        ident = self._create_identity(ctrl, slot)
        if ident is None:
            raise InvariantError("honest entrant could not post its bond")
        ident.is_entrant = not initial
        if not initial:
            self.m["entrants"] += 1
        return ident

    def _retire(self, ident):
        if ident.unsettled:
            raise InvariantError("bond returned before audits settled")
        ident.active = False
        ident.ctrl.funds += ident.bond
        ident.bond = 0.0
        self.slots[ident.slot] = None

    def score(self, ident):
        p = self.p
        if p.system == "R0":
            return (ident.pos_r + p.alpha) / (ident.n_r + p.alpha + p.beta)
        return (ident.c + p.alpha) / (ident.a + p.alpha + p.beta)

    def expected_bill(self, ident):
        """Same rule in every system: the posted menu is a prior worth `bill_prior_weight`
        bills, updated by the seller's observed bills."""
        p = self.p
        k0 = p.bill_prior_weight
        menu_exp = p.h * ident.pH + (1 - p.h) * ident.pL
        return (k0 * menu_exp + ident.bill_sum) / (k0 + ident.bill_n)

    def utility(self, ident):
        p = self.p
        if p.buyer_model == "split":
            s = (ident.solved_n + p.alpha) / (ident.served_n + p.alpha + p.beta)
            return s * p.V - self.expected_bill(ident)
        return self.score(ident) * p.V - self.expected_bill(ident)

    def observation(self, ident):
        """What a seller may see about itself and the rules (used by the E3 harness)."""
        p = self.p
        return dict(round=self.t, rounds_total=p.T, system=p.system,
                    terms_bind=terms_bind(p.system), bond_posted=ident.bond,
                    audit_rate=p.p_audit if uses_audits(p.system) else 0.0,
                    menu=dict(low=ident.pL, high=ident.pH), max_bill=p.max_bill,
                    costs=dict(low=p.cost_L, high=p.cost_H), score=round(self.score(ident), 3),
                    audits_so_far=ident.a, matches_last_rounds=ident.recent[-5:],
                    coalition=ident.ctrl.kind == "coalition")

    def active_identities(self, kind=None):
        return [x for x in self.slots if x is not None and x.active
                and (kind is None or x.ctrl.kind == kind)]

    # ------------------------------------------------------------ round
    def run(self):
        for _ in range(self.p.T):
            self.step()
        self._finish()
        return self.metrics()

    def _set_streams(self):
        """Per-round streams: exogenous events (buyer needs, order, reserved slots) are
        identical across paired runs and branches; choices, policies and audits have
        their own streams."""
        mk = lambda k: np.random.default_rng([self.p.seed, self.t, k])
        self.rng_exo, self.rng_choice, self.rng_policy, self.rng_audit = mk(0), mk(1), mk(2), mk(3)

    def step(self):
        self._set_streams()
        self._start_of_round()
        matches = self._match()
        txs = self._transact(matches)
        self._end_of_round(txs)
        self.t += 1

    def _can_replace(self):
        return (self.p.attack in ("A2", "A2v") and self.coal.replacements_left > 0
                and not self.coal.blocked)

    def _start_of_round(self):
        p, t = self.p, self.t
        # 0. posted prices chosen last round take effect now (terms bind per transaction at match time)
        for ident in self.active_identities():
            if ident.pending_menu is not None:
                ident.pL, ident.pH = ident.pending_menu
                ident.pending_menu = None
        # 1. fill slots emptied by revocation last round
        for s in range(p.n_slots):
            ident = self.slots[s]
            if ident is not None and ident.active:
                continue
            if ident is not None:
                self.slots[s] = None
            if self.slot_kind[s] == "coalition" and self._can_replace():
                new = self._create_identity(self.coal, s, reset=True)
                if new is not None:
                    self.coal.replacements_left -= 1
                    self.m["resets"] += 1
                    continue
            self._honest_entrant(s)
        # 2. scheduled honest newcomer replaces the longest-serving honest seller
        if t > 0 and t % p.newcomer_every == 0:
            hon = self.active_identities("honest")
            if hon:
                old = min(hon, key=lambda x: (x.birth, x.iid))
                old.sched_retired = True
                self._retire(old)
                self._honest_entrant(old.slot)
        # 3. A2-v: voluntary reset when the score falls below a fresh identity's prior
        if p.attack == "A2v":
            prior = p.alpha / (p.alpha + p.beta)
            for ident in self.active_identities("coalition"):
                if self.score(ident) < prior and self._can_replace():
                    slot = ident.slot
                    self._retire(ident)
                    new = self._create_identity(self.coal, slot, reset=True)
                    if new is not None:
                        self.coal.replacements_left -= 1
                        self.m["resets"] += 1
                    else:
                        self._honest_entrant(slot)

    def _accounts_active(self):
        p = self.p
        return (p.attack == "A3" and self.t < p.warmup) or p.attack in ("A3b", "A3d")

    def _match(self):
        p, t = self.p, self.t
        exo, rng = self.rng_exo, self.rng_choice
        n_acc = p.Q if self._accounts_active() else 0
        buyers = [("honest", i) for i in range(p.n_buyers)] + [("coal", i) for i in range(p.Q)]
        needs = exo.random(len(buyers)) < p.h
        order = [i for i in exo.permutation(len(buyers)) if i < p.n_buyers + n_acc]
        reserved = set(exo.choice(p.n_buyers, size=min(p.n_reserved, p.n_buyers), replace=False).tolist())
        active = self.active_identities()
        load = {x.iid: 0 for x in active}
        matches = {x.iid: [] for x in active}
        byid = {x.iid: x for x in active}
        for idx in order:
            kind, j = buyers[idx]
            need_big = bool(needs[idx])
            cands = [x for x in active if load[x.iid] < p.capacity]
            chosen = None
            if kind == "honest":
                if j in reserved:
                    newc = [x for x in cands if t - x.birth < p.newcomer_age and x.newcomer_ok]
                    if newc:
                        chosen = newc[rng.integers(len(newc))]
                if chosen is None:
                    us = [self.utility(x) for x in cands] + [p.outside]
                    mx = max(us)
                    w = [math.exp((u - mx) / p.tau) for u in us]
                    r = rng.random() * sum(w)
                    acc = 0.0
                    for k, wk in enumerate(w):
                        acc += wk
                        if r <= acc:
                            break
                    chosen = cands[k] if k < len(cands) else None
                if chosen is None:
                    self.m["welfare_pre"] += p.outside
                    self.m["outside_n"] += 1
                    continue
            else:
                pool = [x for x in cands if (x.ctrl.kind == "coalition") == (p.attack in ("A3", "A3d"))]
                if not pool:
                    continue
                chosen = pool[rng.integers(len(pool))]
            load[chosen.iid] += 1
            if load[chosen.iid] > p.capacity:
                raise InvariantError("capacity exceeded")
            matches[chosen.iid].append((kind, need_big))
        return [(byid[i], lst) for i, lst in matches.items()]

    # ------------------------------------------------------------ seller decisions
    def _plan(self, ident, lst):
        """Return one action per matched buyer."""
        p, t = self.p, self.t
        acts = [honest_action(nb, ident.pL, ident.pH) for _, nb in lst]
        if self.seller_policy is not None:
            ext = self.seller_policy(self, ident, lst)
            for k, (kind, _) in enumerate(lst):
                if kind == "honest":
                    acts[k] = ext[k]
            return acts
        if ident.ctrl.kind != "coalition" or p.attack not in COALITION_ATTACKS:
            return acts
        honest_idx = [k for k, (kind, _) in enumerate(lst) if kind == "honest"]
        if p.attack == "A1R":
            n = len(honest_idx)
            g4 = ident.pH - p.cost_L - p.markup          # low treatment billed as high at pH
            g8 = p.max_bill - p.cost_L - p.markup        # ... billed as high at the cap
            p_det = 0.0
            if uses_audits(p.system):
                p_det = p.p_audit * ((1 - p.fn) if p.noisy else 1.0)
            stake = (ident.bond if has_bond(p.system) else 0.0) + self._F_hat(ident)
            d, g = choose_violations(n, (g4, g8), p_det, stake)
            price = ident.pH if g == g4 else p.max_bill
            for rank, k in enumerate(honest_idx):
                if rank < d:
                    acts[k] = (L, H, price)
                elif not terms_bind(p.system) and t >= p.T - p.endgame:
                    # legal above-menu pricing, only in the end-game, where the demand
                    # it destroys no longer matters
                    gv = acts[k][0]
                    acts[k] = (gv, gv, p.max_bill)
            return acts
        if p.attack == "A1S":
            # sub-threshold attacker (noisy regime): from a1_start, cheat on as many matched
            # buyers as possible while the expected flag count stays half a flag below the
            # revocation threshold after this round's audits
            if t < p.a1_start or not p.noisy:
                return acts
            n = len(honest_idx)
            a_next = ident.a + p.p_audit * n
            allowed = p.fp * a_next + p.theta * math.sqrt(max(a_next, 1e-9) * p.fp * (1 - p.fp)) - 0.5
            d = 0
            for dd in range(n, -1, -1):
                exp_flags = ident.f + p.p_audit * ((n - dd) * p.fp + dd * (1 - p.fn))
                if exp_flags <= allowed:
                    d = dd
                    break
            for rank, k in enumerate(honest_idx):
                if rank < d:
                    acts[k] = (L, H, ident.pH)
            return acts
        # scripted A1 policy (also used by A2, A2-v and A3 after warm-up)
        for k in honest_idx:
            if t >= p.T - p.endgame:
                cheat = True
            elif t >= p.a1_start:
                cheat = self.rng_policy.random() < p.a1_r
            else:
                cheat = False
            if cheat:
                if p.attack == "A5":
                    # bait-and-switch: appropriate treatment, correctly labelled, billed at
                    # the cap (declaration-specific: legal unless terms bind)
                    gv = acts[k][0]
                    acts[k] = (gv, gv, p.max_bill)
                else:
                    acts[k] = (L, H, ident.pH)
        return acts

    def _F_hat(self, ident):
        """Heuristic continuation value for A1-R: remaining rounds x recent matches x
        honest margin. A1-R does not reset, so re-entry value is zero."""
        p = self.p
        remaining = p.T - self.t - 1
        rec = ident.recent[-5:]
        avg = (sum(rec) / len(rec)) if rec else p.capacity
        return max(0.0, remaining * avg * p.markup)

    # ------------------------------------------------------------ transactions
    def _transact(self, matches):
        p = self.p
        bind = terms_bind(p.system)
        txs = []
        for ident, lst in matches:
            if not lst:
                if self.seller_policy is not None:
                    self.seller_policy(self, ident, [])    # price-posting decision only
                continue
            acts = self._plan(ident, lst)
            for (kind, need_big), (given, billed, price) in zip(lst, acts):
                price = min(price, p.max_bill)
                cost = p.cost_H if given == H else p.cost_L
                if kind == "coal" and ident.ctrl is self.coal and ((p.fake_selfdeal and not uses_audits(p.system)) or p.fake_selfdeal_all):
                    cost = 0.0                     # pure fake: no audits, so nothing is delivered
                if kind == "coal":
                    if self.coal.funds < price:
                        self.m["capital_blocked"] += 1
                        continue
                    self.coal.funds -= price
                    if ident.ctrl is self.coal:
                        self.m["self_purchases"] += 1
                    else:
                        self.m["rival_purchases"] += 1
                else:
                    self.honest_payments += price
                ident.ctrl.funds += price - cost
                self.costs += cost
                appr = H if need_big else L
                solved = (given == H) or (not need_big)
                menu = (ident.pL, ident.pH)
                misrep = billed != given
                wrong = given != appr
                undeclared = abs(price - menu[billed]) > 1e-9
                shared = misrep or wrong
                viol = shared or (bind and undeclared)
                loss = (p.V - menu[appr]) - (p.V * solved - price)
                if kind == "honest":
                    self.m["honest_tx"] += 1
                    self.m["welfare_pre"] += p.V * solved - price
                    if ident.ctrl.kind == "coalition":
                        self.m["coal_tx_honest"] += 1
                    if viol:
                        self.m["harm_viol"] += loss
                        self.m["n_viol"] += 1
                        self.m["n_viol_shared"] += int(shared)
                    unexp = max(0.0, price - menu[billed])
                    if unexp > 0:
                        self.m["unexpected"] += unexp
                        self.m["n_unexpected"] += 1
                        if not shared:
                            self.m["unexpected_loss"] += loss
                        if ident.is_reset or (self.t - ident.birth < p.newcomer_age):
                            self.m["unexpected_new"] += unexp
                    pos = solved and not undeclared
                else:
                    pos = ident.ctrl is self.coal
                ident.n_r += 1
                ident.pos_r += int(pos)
                ident.bill_sum += price
                ident.bill_n += 1
                ident.served_n += 1
                ident.solved_n += int(solved)
                ident.unsettled += 1
                if ident.is_entrant and self.t - ident.birth < 10:
                    ident.early_matches += 1
                txs.append((ident, kind, viol, loss))
        return txs

    # ------------------------------------------------------------ audits and settlement
    def _end_of_round(self, txs):
        p, rng = self.p, self.rng_audit
        by = {}
        for tx in txs:
            by.setdefault(tx[0].iid, []).append(tx)
        for ident in self.active_identities():
            lst = by.get(ident.iid, [])
            ident.recent.append(len(lst))
            losses, found = [], False
            if uses_audits(p.system):
                for _, kind, viol, loss in lst:
                    if rng.random() < p.p_audit:
                        ident.a += 1
                        self.m["audits"] += 1
                        if ident.is_entrant and self.t - ident.birth < 10:
                            ident.early_audits += 1
                        if p.noisy:
                            flag = (rng.random() >= p.fn) if viol else (rng.random() < p.fp)
                            ident.f += int(flag)
                            ident.c += int(not flag)
                        elif viol:
                            found = True
                            losses.append((kind, loss))
                        else:
                            ident.c += 1
            ident.unsettled = 0
            if p.noisy and uses_audits(p.system):
                revoke = noisy_revoke(ident.f, ident.a, p.fp, p.theta)
            else:
                revoke = found
            if revoke:
                self._revoke(ident, losses)

    def _revoke(self, ident, losses):
        p = self.p
        self.m["revocations"] += 1
        if ident.ctrl.kind == "honest":
            self.m["false_sanctions"] += 1
        self._forfeit(ident, losses)
        if p.linked and ident.ctrl.kind == "coalition":
            ident.ctrl.blocked = True
            for other in self.active_identities("coalition"):
                self._forfeit(other, [])

    def _forfeit(self, ident, losses):
        """Revoke, forfeit the bond, compensate verified losses (verifiable regime only)."""
        ident.active = False
        ident.revoked = True
        pool, ident.bond = ident.bond, 0.0
        if pool > 0 and losses and not self.p.noisy:
            total = sum(l for _, l in losses)
            pay = min(total, pool)
            for kind, l in losses:
                amt = pay * l / total
                if kind == "honest":
                    self.comp_honest += amt
                else:
                    self.coal.funds += amt
            pool -= pay
        self.reserve += pool

    def _finish(self):
        for ident in self.active_identities():
            self._retire(ident)
        self.check_money()

    def check_money(self):
        holdings = sum(c.funds for c in self.controllers) + self.reserve + \
            sum(x.bond for x in self.identities)
        expected = self.initial_money + self.honest_payments - self.costs - self.comp_honest
        if abs(holdings - expected) > 1e-6:
            raise InvariantError(f"money not conserved: {holdings} vs {expected}")

    def metrics(self):
        p, m = self.p, dict(self.m)
        T = p.T
        m["welfare_pre_pr"] = m["welfare_pre"] / T
        m["welfare_post_pr"] = (m["welfare_pre"] + self.comp_honest) / T
        m["comp_honest"] = self.comp_honest
        m["coal_profit"] = self.coal.funds - self.coal.endowment
        m["reserve"] = self.reserve
        ents = [x for x in self.identities if x.is_entrant and x.ctrl.kind == "honest"]
        m["newcomer_matches"] = (sum(x.early_matches for x in ents) / len(ents)) if ents else float("nan")
        m["newcomer_audits"] = (sum(x.early_audits for x in ents) / len(ents)) if ents else float("nan")
        hon = [x for x in self.identities if x.ctrl.kind == "honest"]
        m["n_honest_ids"] = len(hon)
        full = [x for x in hon if not x.is_entrant and not x.sched_retired]   # initial, exposed all game
        m["n_full_exposure"] = len(full)
        m["false_full_exposure"] = sum(int(x.revoked) for x in full)
        m["honest_profit_per_seller"] = sum(c.funds - c.endowment for c in self.controllers
                                            if c.kind == "honest") / max(1, len(hon))
        m["outside_share"] = m["outside_n"] / (p.n_buyers * T)
        ct = m["coal_tx_honest"]
        m["unexpected_per_coal_tx"] = m["unexpected"] / ct if ct else float("nan")
        m["unexpected_share_coal_tx"] = m["n_unexpected"] / ct if ct else float("nan")
        return m


def noisy_revoke(f, a, fp, theta):
    """Revoke when flags exceed false-positive expectation by theta standard deviations."""
    if a == 0:
        return False
    sd = math.sqrt(a * fp * (1 - fp))
    return (f - fp * a) > theta * sd + 1e-12


def run_one(p: Params) -> dict:
    out = Market(p).run()
    out.update(p.as_dict())
    return out
