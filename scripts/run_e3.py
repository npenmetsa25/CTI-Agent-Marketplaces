"""E3 LLM case study. Run in YOUR terminal with your key set:
    export ANTHROPIC_API_KEY=...        (never write it into files)
    python3 scripts/run_e3.py --model <model-id> --pilot      # one run: R2, gateway-enforced, seed 0
    python3 scripts/run_e3.py --model <model-id>              # all 12 runs (skips runs already completed)
    python3 scripts/run_e3.py --model <model-id> --reference  # 6 runs: scripted committed sellers, gateway only
    python3 scripts/run_e3.py --model <claude-id> --opp-provider openai --opp-model <gpt-id> --cross
                                                              # 6 runs: GPT opportunists vs Claude committed
    python3 scripts/run_e3.py --mock                          # no API calls; checks the pipeline
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from cti_sim.config import Params
from cti_sim.market import Market
from cti_sim.llm_harness import LLMSellerPolicy, AnthropicClient, OpenAIClient
from cti_sim.policies import honest_action

BASE = dict(n_slots=6, K=2, n_buyers=12, n_reserved=2, T=20, bond=100.0, attack="honest")
CONDS = [(s, enf) for s in ("R1", "R2") for enf in ("self", "gateway")]


class HonestMock:
    """Honest actions at current prices; re-posts the same prices."""
    def complete(self, system, user, meta):
        o = meta["obs"]
        acts = [dict(zip(("treatment", "billed", "price"),
                         (("high", "high", o["menu"]["high"]) if big else ("low", "low", o["menu"]["low"]))))
                for big in meta["buyers"]]
        return json.dumps({"actions": acts, "next_prices": {"low": o["menu"]["low"], "high": o["menu"]["high"]},
                           "note": "mock", "message": "mock"})


def params(system, enf, seed):
    return Params(system=system, p_audit=0.10 if enf == "gateway" else 0.0, seed=seed, **BASE)


def run(system, enf, seed, client, out_dir, model_name, committed="llm", opp_client=None, tag=""):
    run_id = f"{system}_{enf}_s{seed}" + ("_scripted" if committed == "scripted" else "") + tag
    log = os.path.join(out_dir, f"{run_id}.jsonl")
    if os.path.exists(log):
        os.rename(log, log + f".old{int(time.time())}")
    pol = LLMSellerPolicy(client, log, dict(run_id=run_id, system=system, enforcement=enf, seed=seed, model=model_name,
                                            committed=committed), committed=committed, max_calls=1000,
                          opp_client=opp_client)
    _oc0 = opp_client or client
    _tok0 = [(c, c.__dict__.get("in_tokens", 0), c.__dict__.get("out_tokens", 0)) for c in {id(client): client, id(_oc0): _oc0}.values()]
    mk = Market(params(system, enf, seed), seller_policy=pol)
    m = mk.run()
    # profit per seller-round active (identities differ in how long they are in the market), x T
    T = mk.p.T
    coal_rounds = sum(len(x.recent) for x in mk.identities if x.ctrl.kind == "coalition")
    hon_rounds = sum(len(x.recent) for x in mk.identities if x.ctrl.kind == "honest")
    hon_profit = sum(c.funds - c.endowment for c in mk.controllers if c.kind == "honest")
    m["profit_per_opportunistic"] = T * m["coal_profit"] / max(1, coal_rounds)
    m["profit_per_committed"] = T * hon_profit / max(1, hon_rounds)
    oc = opp_client or client
    m.update(run_id=run_id, committed=committed, enforcement=enf, model=model_name,
             committed_model=getattr(client, "name", model_name), opp_model=getattr(oc, "name", model_name),
             calls=pol.calls, retries=pol.retries, fallbacks=pol.fallbacks,
             # per-run token use (clients accumulate across runs in one invocation)
             in_tokens=sum(c.__dict__.get("in_tokens", 0) - i0 for c, i0, _ in _tok0),
             out_tokens=sum(c.__dict__.get("out_tokens", 0) - o0 for c, _, o0 in _tok0))
    summ = os.path.join(out_dir, "summary.csv")
    pd.DataFrame([m]).to_csv(summ, mode="a", header=not os.path.exists(summ), index=False)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reference", action="store_true", help="scripted committed sellers, gateway-enforced only")
    ap.add_argument("--opp-provider", choices=["anthropic", "openai"], default=None)
    ap.add_argument("--opp-model", default=None)
    ap.add_argument("--opp-reasoning", default=None, help='e.g. "low" for OpenAI reasoning models')
    ap.add_argument("--cross", action="store_true", help="cross-model runs: R1, R2 gateway-enforced, 3 seeds")
    a = ap.parse_args()
    out = "results/e3_mock_v2" if a.mock else "results/e3"
    os.makedirs(out, exist_ok=True)
    if a.mock:
        client, name = HonestMock(), "mock"
    else:
        if not a.model:
            sys.exit("Pass --model <model-id> (a mid-tier model; extended thinking is off by default).")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY in your environment first (do not write it into any file).")
        client, name = AnthropicClient(a.model), a.model
    opp_client, tag = None, ""
    if a.opp_model and not a.mock:
        if a.opp_provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                sys.exit("Set OPENAI_API_KEY in your environment first (do not write it into any file).")
            opp_client = OpenAIClient(a.opp_model, reasoning_effort=a.opp_reasoning)
        else:
            opp_client = AnthropicClient(a.opp_model)
        tag = "_x-" + a.opp_model.replace("/", "-")
    committed = "scripted" if a.reference else "llm"
    if a.pilot:
        jobs = [("R2", "gateway", 0)]
    elif a.cross:
        jobs = [(s, "gateway", k) for s in ("R1", "R2") for k in range(a.seeds)]
    elif a.reference:
        jobs = [(s, "gateway", k) for s in ("R1", "R2") for k in range(a.seeds)]
    else:
        jobs = [(s, e, k) for s, e in CONDS for k in range(a.seeds)]
    done = set()
    summ = os.path.join(out, "summary.csv")
    if os.path.exists(summ) and not a.pilot:
        done = set(pd.read_csv(summ).run_id)
    for s, e, k in jobs:
        rid = f"{s}_{e}_s{k}" + ("_scripted" if committed == "scripted" else "") + tag
        if rid in done:
            print("skip (done):", rid); continue
        t0 = time.time()
        m = run(s, e, k, client, out, name, committed, opp_client, tag)
        print(f"{rid}: calls={m['calls']} fallbacks={m['fallbacks']} viol={m['n_viol']} revoc={m['revocations']} "
              f"profit/opportunistic={m['profit_per_opportunistic']:.0f} profit/committed={m['profit_per_committed']:.0f} "
              f"unexpected={m['unexpected']:.0f} "
              f"tokens in/out={m['in_tokens']}/{m['out_tokens']} ({time.time()-t0:.0f}s)")
