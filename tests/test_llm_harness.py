import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cti_sim.config import Params
from cti_sim.market import Market
from cti_sim.llm_harness import LLMSellerPolicy, parse
from scripts.run_e3 import HonestMock, params


class BadMock:
    def __init__(self): self.n = 0
    def complete(self, system, user, meta):
        self.n += 1
        return "not json at all"


def test_honest_mock_reproduces_honest_baseline():
    for s, enf in (("R1", "self"), ("R2", "gateway")):
        p = params(s, enf, 0)
        with tempfile.TemporaryDirectory() as d:
            pol = LLMSellerPolicy(HonestMock(), os.path.join(d, "x.jsonl"), {})
            a = Market(p, seller_policy=pol).run()
        b = Market(p).run()
        assert all(a[k] == b[k] for k in b if isinstance(b[k], (int, float)) and k != "seed"), s


def test_malformed_output_retries_once_then_falls_back_and_logs():
    p = params("R2", "gateway", 1)
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "x.jsonl")
        bad = BadMock()
        pol = LLMSellerPolicy(bad, log, {})
        m = Market(p, seller_policy=pol).run()
        lines = [json.loads(l) for l in open(log)]
    assert pol.fallbacks == len(lines) > 0 and bad.n == 2 * len(lines)
    assert all(r["fallback"] and len(r["errors"]) == 2 for r in lines)
    assert m["n_viol"] == 0                      # fallback is the honest action


def test_full_mock_run_writes_complete_log():
    p = params("R1", "gateway", 2)
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "x.jsonl")
        pol = LLMSellerPolicy(HonestMock(), log, {"run_id": "t"})
        Market(p, seller_policy=pol).run()
        lines = [json.loads(l) for l in open(log)]
    assert len(lines) == pol.calls > 0
    assert {"system", "user", "responses", "actions", "round", "slot"} <= set(lines[0])


def test_parse_validates_prices_and_lengths():
    np_ = {"low": 3, "high": 7}
    ok = json.dumps({"actions": [{"treatment": "low", "billed": "high", "price": 7}], "next_prices": np_})
    acts, prices, _, err = parse(ok, 1, 11)
    assert err is None and acts == [(0, 1, 7.0)] and prices == (3.0, 7.0)
    assert parse(ok, 2, 11)[3] is not None
    bad = json.dumps({"actions": [{"treatment": "low", "billed": "low", "price": 50}], "next_prices": np_})
    assert parse(bad, 1, 11)[3] is not None
    assert parse(json.dumps({"next_prices": {"low": 1, "high": 7}}), 0, 11, "committed")[3] is not None


class GreedyCommittedMock(HonestMock):
    """Committed sellers try to post 9/11; opportunistic sellers try to cheat. Commitment must hold."""
    def complete(self, system, user, meta):
        if meta["kind"] == "committed":
            return json.dumps({"next_prices": {"low": 9, "high": 11}, "note": "raise"})
        acts = [{"treatment": "low", "billed": "high", "price": 11} for _ in meta["buyers"]]
        return json.dumps({"actions": acts, "next_prices": {"low": 3, "high": 7}, "note": "", "message": "hi"})


def test_committed_sellers_are_bound_by_code_and_prices_apply_next_round():
    p = params("R1", "gateway", 3)
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "x.jsonl")
        pol = LLMSellerPolicy(GreedyCommittedMock(), log, {})
        mk = Market(p, seller_policy=pol)
        mk.run()
        recs = [json.loads(l) for l in open(log)]
    com = [r for r in recs if r["kind"] == "committed" and r["buyers"]]
    assert com and all(a[0] == a[1] and a[2] == (r["menu"][1] if a[1] == "high" else r["menu"][0])
                       for r in com for a in r["actions"])          # appropriate labels, posted price
    assert any(r["menu"] == [9.0, 11.0] for r in com)                # raised prices took effect later
    assert all(r["menu"] == [3.0, 7.0] for r in recs if r["round"] == 0)
    opp = [r for r in recs if r["kind"] == "opportunistic" and r["buyers"]]
    assert any(a == ["low", "high", 11.0] for r in opp for a in r["actions"])   # opportunists can cheat
    assert any(r["message"] == "hi" for r in opp)


def test_scripted_committed_reference_makes_no_calls_for_committed():
    p = params("R2", "gateway", 0)
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "x.jsonl")
        pol = LLMSellerPolicy(HonestMock(), log, {}, committed="scripted")
        Market(p, seller_policy=pol).run()
        recs = [json.loads(l) for l in open(log)]
    assert recs and all(r["kind"] == "opportunistic" for r in recs)


class TaggedMock(HonestMock):
    def __init__(self, tag): self.tag, self.name, self.n = tag, tag, 0
    def complete(self, system, user, meta):
        self.n += 1
        assert (meta["kind"] == "opportunistic") == (self.tag == "opp")
        return HonestMock.complete(self, system, user, meta)


def test_separate_clients_per_seller_type():
    p = params("R2", "gateway", 4)
    com, opp = TaggedMock("com"), TaggedMock("opp")
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "x.jsonl")
        pol = LLMSellerPolicy(com, log, {}, opp_client=opp)
        Market(p, seller_policy=pol).run()
        recs = [json.loads(l) for l in open(log)]
    assert com.n > 0 and opp.n > 0
    assert all(r["model_used"] == ("opp" if r["kind"] == "opportunistic" else "com") for r in recs)
