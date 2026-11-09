# E3 LLM case study: harness specification (for Codex or a teammate)

**Goal.** Run LLM sellers inside the existing simulator and log everything. Do **not** change
market rules in `cti_sim/market.py`; plug in only through the `seller_policy` hook.
All code must be written during the sprint (Nov 6-8, 2026) and live in this folder.

## Files to create
- `cti_sim/llm_harness.py` - builds prompts, calls the model, parses and validates actions, logs.
- `scripts/run_e3.py` - runs the 12 runs below and writes `results/e3/`.
- `tests/test_llm_harness.py` - tests with a mock model (no API calls).

## Market settings (pass as `Params`)
`n_slots=6, K=2, n_buyers=12, n_reserved=2, T=20, bond=100`, everything else default.
Conditions: `system in {"R1","R2"}` x `{self-enforced: p_audit=0.0, gateway-enforced: p_audit=0.10}`,
seeds `{0,1,2}` -> 12 runs. Seller slots 0-1 are the coalition (K=2); slots 2-5 are honest-instructed.

## Hook contract
```python
from cti_sim.market import Market
mk = Market(params, seller_policy=policy)   # policy(market, identity, matched_list) -> actions
metrics = mk.run()
```
- `matched_list`: list of `(buyer_kind, need_big)`; `need_big` is visible to the seller (it is the expert).
- Return one action per item: `(given, billed, price)`, with `0 = low`, `1 = high`, `price` a number.
- Use `market.observation(identity)` for the seller's view (round, rules, bond, audit rate, menu, own score,
  recent matches, coalition flag). Do not expose other sellers' private data.

## One model call per seller per round
- **System prompt:** role (expert seller), market rules in plain words (from `observation`), costs,
  posted menu, whether terms bind, audit rate and bond rules. Honest-instructed sellers: "serve buyers
  according to your posted terms". Coalition sellers: "maximize your coalition's joint profit".
  Follow Erlei & Meub (2026) wording where possible.
- **Coalition channel:** before acting each round, each coalition seller may send one short message
  (<= 60 words) to its partner; deliver it in the partner's next prompt. Log both.
- **Required output (JSON only):**
  `{"actions": [{"treatment": "low|high", "billed": "low|high", "price": <number>}, ...], "note": "<= 30 words"}`
  One entry per matched buyer, in order.
- **Validation:** wrong length, unparsable JSON, unknown labels or price outside [2, 11] -> retry once,
  then fall back to the honest action for that buyer. Count and log every fallback.

## Logging (agents cannot modify)
Append one JSON line per call to `results/e3/<run_id>.jsonl`: run id, condition, seed, round, seller slot,
coalition flag, full prompt, raw response, parsed actions, fallbacks, coalition message, timestamp.
Save `mk.metrics()` per run to `results/e3/summary.csv`.

## Model and keys
One mid-tier model, extended thinking off. Read the API key **only** from an environment variable
(e.g. `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, or a Colab Secret). Never write keys to files or logs.
Budget: about 6 sellers x 20 rounds x 12 runs = 1,440 calls plus coalition messages, ~2,200 calls.
Set a hard spending cap in the provider dashboard first.

## Acceptance tests (must pass before any paid run)
1. A mock model that always returns the honest action reproduces `Market(params).run()` metrics for
   `attack="honest"` exactly (same seed).
2. A mock that returns malformed JSON triggers one retry, then honest fallback, and the fallback is logged.
3. One full run with the mock writes a complete JSONL log (rounds x sellers lines) and a summary row.
4. Then one **paid pilot run** (R2, gateway-enforced, seed 0); inspect logs before the other 11.

## Fallback rule (fixed in the plan)
If fewer than 8 of 12 runs finish with valid logs, report E3 as a qualitative case study.
