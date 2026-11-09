# Commitment-Backed Trust Index (CTI)

Code and data for **"Commitment-Backed Trust Index: Testing Reputation Against Coalition Attacks in Agent Marketplaces"**, Apart Research Swarm Alignment Hackathon, Track 3 (November 6–8, 2026). Author: Nabita Penmetsa, University of Utah.

We test whether commitment (binding posted terms plus a forfeitable bond, checked by hidden random audits) makes reputation more robust to coalitions of agents in a credence-goods marketplace. We compare three systems under matched information: peer ratings (R0), an audit-only score (R1) and the CTI (R2), plus two ablations that remove the bond or the binding declaration.

All code in this repository was written during the hackathon. No pre-existing code is used.

## Main results

- **Verifiable audits (E1).** At 5% auditing, the CTI produced nonpositive mean coalition gains for every tested attack. Selective defection (+35 under audit-only), identity resets (+99) and bait-and-switch pricing (+113) were clearly unprofitable under the CTI (−92, −74, −56); the payoff-responsive attack was inconclusive (−30 [−62, 5]). The deterrence threshold for selective defection fell from 0.08 to 0.04.
- **Payoffs, not behavior.** Scripted attackers committed the same violations under audit-only and the CTI; the bond changed their payoff, and buyers were protected only through compensation.
- **Noisy audits (E2).** The CTI lowered attack profit at every audit rate but did not consistently eliminate it, even at 20% auditing; outcomes depend strongly on the false-positive rate.
- **LLM sellers (E3).** Opportunistic LLM sellers never took actions the rules called violations, but charged above non-binding posted prices in 30–62% of transactions, rising toward the end of the game. Under declared binding terms they never did, even without audits. Because the rules labelled fraud a violation, E3 measures response to declared terms, not fraud propensity. LLM sellers of both types raised posted prices, lowering buyer surplus by 28–34%.

All results concern the specified attacks in this environment, not general collusion resistance.

## Repository layout

```
cti_sim/            simulator package
  config.py         parameters (frozen dataclass); systems R0, R1, R2, R2_nobond, R2_nodecl; attacks
  market.py         market, matching, ledgers, audits, settlement, attacks, invariant checks, LLM hook
  policies.py       payoff-responsive rule (A1-R) and honest action
  e0.py             terminal-round verification (exact enumeration + Monte Carlo)
  reset_diag.py     E1b voluntary-reset diagnostic (cloned market state)
  analysis.py       paired gains, bootstrap intervals, deterrence thresholds
  llm_harness.py    E3 prompts, parsing, execution wrapper for committed sellers, logging, API clients
scripts/            one script per experiment (see below)
tests/              pytest suite (14 tests; the LLM tests use a mock model, no API calls)
results/            all result CSVs; results/e3/ holds the E3 logs and summary
figures/            paper figures (PNG and PDF)
docs/
  checkpoints.md    decision log, including analysis changes and the review of claims
  E3_spec.md        E3 harness specification
  figure_captions.md
report/             submitted report (Apart template)
```

## Setup

Python 3.10 or later.

```
python3 -m pip install -r requirements.txt
python3 -m pytest -q tests
```

## Reproducing the simulation experiments (no API calls)

Everything below runs on a laptop (4 cores) in a few minutes in total.

| Experiment | Command | Output |
| --- | --- | --- |
| E0 terminal-round verification | `python3 scripts/run_e0.py 100000` | `results/e0.csv` |
| E1 main, bond, fine grid, linked lineage, buyer-model robustness (30 paired seeds) | `python3 scripts/run_e1.py 30 all 0` | `results/e1_all_s0-29.csv` |
| E1 analysis (paired gains, thresholds) | `python3 scripts/analyze_e1.py` | `results/e1_paired.csv`, `table_fig1.csv`, `thresholds_fine.csv` |
| E1b voluntary-reset diagnostic (200 seeds) | `python3 scripts/run_reset_diag.py 200` | `results/reset_diag.csv` |
| E1c purchased history, reputation-sensitive market (200 seeds) | `python3 scripts/run_h1_slice.py 200` | `results/h1_slice.csv`, `h1_value_of_purchases.csv` |
| E2 threshold calibration (seeds 1000–1199) | `python3 scripts/calibrate_e2.py` | `results/e2_calibration.csv`, `e2_theta.json` |
| E2 noisy audits (200 paired test seeds 0–199) | `python3 scripts/run_e2.py 200` | `results/e2_runs.csv`, `e2_paired.csv` |
| Paired intervals for E1b, per-cell E2, false revocation by exposure | `python3 scripts/robustness_checks.py` | `results/e1b_paired_intervals.csv`, `e2_per_cell.csv`, `e2_false_revocation_exposure.csv` |
| Figures | `python3 scripts/make_figures.py` | `figures/` |

Seeds are fixed, so reruns reproduce the saved files exactly. `results/pilot.csv` and `results/e2_theta_pooled_v1.json` are kept for the record: the first is the superseded Checkpoint B pilot, the second the original E2 calibration, replaced after it over-revoked honest sellers present for the whole game (see `docs/checkpoints.md`).

## Reproducing E3 (LLM sellers; requires API keys)

E3 places six LLM sellers in a smaller market (12 buyers, 2 reserved for newcomers, 20 rounds, bond 100): four committed sellers bound by an execution wrapper, whose only decision is next round's prices, and two opportunistic sellers of one company with a private channel. Conditions cross audit-only (R1) vs CTI (R2) with no audits vs 10% audits, 3 seeds each.

Set keys in your own terminal as environment variables. **Never write keys into files in this repository.**

```
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...          # only for the mixed-model runs

python3 scripts/run_e3.py --model claude-sonnet-5-5 --pilot          # 1 run: CTI, audits, seed 0
python3 scripts/run_e3.py --model claude-sonnet-5-5                  # 12 main runs
python3 scripts/run_e3.py --model claude-sonnet-5-5 --reference      # 6 runs: scripted committed sellers at 3 and 7
python3 scripts/run_e3.py --model claude-sonnet-5-5 --cross \
    --opp-provider openai --opp-model gpt-6-luna --opp-reasoning low  # 6 runs: GPT-6 Luna opportunists
python3 scripts/run_e3.py --mock --pilot      # offline check with a mock model; writes results/e3_mock_v2 (git-ignored)
```

Completed runs are skipped, so an interrupted batch can be restarted. Each run writes `results/e3/<run_id>.jsonl`, one record per seller decision with the full prompt, every model response, parse errors, the actions applied and the posted prices, and appends a row to `results/e3/summary.csv`. The submitted results comprise 24 runs: 12 main, 6 mixed-model and 6 reference.

Notes on the E3 summary columns:

- `profit_per_opportunistic` and `profit_per_committed` are profit per seller normalized to a full 20-round game (profit per active seller-round × 20).
- `welfare_pre_pr` is honest buyers' surplus per round, not total surplus.
- In the submitted `summary.csv`, `in_tokens` and `out_tokens` accumulate across the runs launched by one command; per-run use is the difference from the previous run in the same batch. `run_e3.py` now records per-run counts.

## Analysis changes made during the project

Documented in full in `docs/checkpoints.md`:

1. The deterrence-threshold criterion changed from mean gain < 0 to ≤ 0, so a payoff-responsive policy that stops cheating counts as unprofitable.
2. The E2 revocation threshold was recalibrated on honest sellers present for the whole game after the first calibration over-revoked them (8–11%).
3. E1c was designed after the default market showed no value from purchased history; it is exploratory.
4. A5 (bait-and-switch), the shared expected-bill rule and the restriction of A1-R's above-menu pricing to the final 3 rounds followed pilot findings.

## Citation

Penmetsa, N. (2026). Commitment-Backed Trust Index: Testing Reputation Against Coalition Attacks in Agent Marketplaces. Apart Research Swarm Alignment Hackathon.

## LLM usage

Claude (Anthropic) assisted with experiment design, code, analysis and drafting; all results were checked against the result files. Claude Sonnet 5.5 and GPT-6 Luna are also the subjects of E3.
