# Checkpoint log

## Checkpoint A - E0 verification (passed)
- Closed form = exact enumeration in all 128 cells (d {1,3} x g {1,2,4,8} x b {0,40,100,200} x p {.02,.05,.10,.20}).
- Monte Carlo through the market's own audit/settlement code, 100,000 draws per cell: 128/128 within 4 SE.
- Payoff-responsive rule = formula argmax in all cells. Two cells sit exactly at break-even
  (4 - 0.1*40 = 0, 8 - 0.2*40 = 0); tie rule is "no cheating at zero expected gain" (check fixed, not the policy).
- Plan numbers reproduced: break-even 84.14 / 168.27; -2.26 / +9.74 at b = 100.

## Checkpoint B - pilot at defaults (30 paired seeds, 1,050 runs, ~1 s)
Sanity checks pass: newcomers get ~2.2-2.9 buyers/round in their first 10 rounds; outside option
chosen 1-7% of the time; capital never blocks (also at b = 200); money conserved in every run.

Issues needing a decision:
1. **Buyers' expected-bill rule differs by system** (menu where terms bind, past bills elsewhere).
   Overbilling therefore lowers demand under R1 but is invisible to buyers under R2. This breaks the
   matched-information principle and lets binding terms shield cheaters from market discipline.
   Proposed: one rule in every system - expected bill = posted menu as a prior blended with observed
   bills, (k0 * menu_expectation + sum of bills) / (k0 + n), k0 = 4.
2. **A2-v never triggers** (identical to A2 in all systems): under verifiable audits a score cannot
   fall without revocation, and in R0 coalition ratings stay above the 0.75 prior. Proposed: drop A2-v
   from E1 and note it in limitations.
3. **A1-R self-harm** from legal above-menu pricing (destroyed its own demand in R0/R1). Fixed:
   legal above-menu pricing only in the final 3 rounds. Plan wording needs the same change.

Early signals (pilot only, not results): bond deters A1 (R2 -58 vs R2-bond +58, R1 +36 coalition gain);
A1-R breaks even under R2 and gains +212 under R2-bond (gain-8 bursts, as predicted);
A3 and A3b lose money in every system at these parameters.

## Changes after Checkpoint B (approved)
- One expected-bill rule in every system: menu prior worth 4 bills, updated by observed bills.
- Per-round random streams (exogenous / choice / policy / audit) so paired runs and branches share buyer needs and order.
- A2-v removed from E1; replaced by the paired reset diagnostic E1b (cti_sim/reset_diag.py).
- A1-R legal above-menu pricing only in the final 3 rounds.
- E0 re-run after the changes: still 128/128 on all three checks.

## New issue found: declaration ablation was untestable (proposed fix A5, pending approval)
With one bill rule, R2-bond == R1 and R2 == R2-declaration for every scripted attack, because no
scripted attack charged an undeclared price. Proposed A5 bait-and-switch: appropriate treatment,
correctly labelled, billed at 11 (A1's schedule). Pilot: R1 +113, R2-declaration +113 (legal),
R2-bond +41 (revocation only), R2 -56 (revocation + bond); unexpected charges 156 -> 105.

## E1b preview (50 seeds, provisional)
Value of reset: R0 +29, R1 / R2-bond +18, R2 / R2-declaration +4 (noisy). Without newcomer
eligibility: about -8 everywhere -> the gain comes from the newcomer quota, not reputation clearing.

## Checkpoint C - smoke test (1 seed per cell, all parts)
1,699 runs in 2.2 s; every attack run has its honest pair; capital never blocks; no missing values.
Full E1 at 30 seeds is about 51,000 runs, roughly 70 s.

## Full runs (Sat Nov 7)
- E1: 50,970 runs in 64 s (30 paired seeds; main, bond, fine grid, linked lineage, robustness slice). Capital never blocked.
- Analysis fix: threshold criterion changed from "gain < 0" to "gain <= 0" (a deterred payoff-responsive
  attacker has gain exactly 0). Plan doc updated.
- E2: theta calibrated on seeds 1000-1199 (<= 5% whole-game false revocation); test seeds 0-29 show
  7.2% / 6.8% at p = 0.10 (out-of-sample miss, reported as is).
- E1b: 200 seeds, 2,800 branch pairs in 4.6 s.
Results summary: see chat / results/*.csv. Figures pending user review.

## E1c - H1 reputation-sensitive slice (30 paired seeds)
Default market: purchased history never pays (A3-A1 = -116 R0, -136 audit systems).
Sensitive market (tau 0.25, prior Beta(2,2), incumbents with 6 clean obs, coalition enters new, free fakes without audits):
- R0: A3-A1 = +134 [117,149]; A3d-A1 = +156 [142,168]
- Audit systems (real treatment cost): -117 to -154; dilution -533 to -600
- Free fakes everywhere (diagnostic): R1 +43 [27,61] / +68 [57,78]; R2 +7 [-15,29] / +1 [-22,25] (n.s.)
Reading: audit scarcity cuts the value of fake records to ~1/3; the bond removes the rest.

## E2 false-revocation check with 200 fresh seeds
The 7.2% / 6.8% at p = 0.10 came from only 30 test seeds (~290 honest identities). With 200 fresh seeds
(~1,900 identities per cell) the whole-game false-revocation rate is 3.7%-5.1% across cells
(p = 0.10: 4.9% [3.9, 5.8] and 5.1% [4.1, 6.0]). The calibration holds on average; because theta is the
smallest value meeting 5%, rates sit at the target rather than below it.

## E2 rerun with 200 paired seeds (14,400 runs)
Coalition gain, mean [95% CI]:
- A1: audit-only +68 / +59 / +36 at p = .05/.10/.20; CTI +7 [-2, 16] / -4 [-12, 3] / -40 [-47, -32]
- A1S: audit-only +43 / +36 / +5 [1, 8]; CTI +36 [31, 42] / -42 [-50, -33] / -79 [-86, -72]
Reading: under noise the CTI is roughly break-even for A1 at 5-10% auditing and deters clearly at 20%;
it deters the sub-threshold attacker from 10%. Audit-only stays profitable for both at every tested rate.
Honest false revocation on test seeds: 3.7-5.1% (about 0.5 honest sellers per run).

## Review of claims (after external assessment) - corrected wording and new checks
1. **E1c is exploratory** (several changes at once). Rerun at 200 seeds with the free-fakes diagnostic saved
   (results/h1_value_of_purchases.csv). Corrected claim: "Purchased history was unprofitable at the default
   settings. In an exploratory reputation-sensitive setting, free fake ratings enabled profitable exploitation
   (+137 A3, +150 A3d). Audited systems reduced this advantage. With free fakes everywhere, the CTI reduced it
   further relative to audit-only (paired difference +39 [32, 46] A3, +55 [47, 62] A3d), but did not eliminate it:
   A3 under the CTI +7 [-1, 15] (inconclusive), A3d +20 [9, 31] (still profitable)."
   A3d (dilution): buyer accounts keep buying compliant transactions from coalition sellers throughout the game,
   offsetting the bad ratings that cheating produces. (Earlier 30-seed claim "the bond removes the rest" withdrawn.)
2. **Resets** (paired, 200 seeds; results/e1b_paired_intervals.csv): "Voluntary resets were profitable with
   renewed newcomer eligibility (peer rating +29.4 [28.3, 30.6], audit-only +17.8 [15.6, 19.8]). Removing that
   eligibility made their estimated value negative (-8.0, -7.2), indicating newcomer allocation was necessary for
   profitability in these conditions." Effect of eligibility: +37.4 [36.6, 38.2] and +25.0 [23.1, 26.8].
   CTI (verifiable): reset value +2.9 [-4.0, 9.8], inconclusive. Clearing history may still contribute.
3. **E2 calibration corrected.** The original theta targeted 5% over all honest identities, including newcomers
   and sellers retired on schedule; sellers present the whole game had 8-11% false revocation. Recalibrated on
   whole-game sellers (seeds 1000-1199; original values kept in results/e2_theta_pooled_v1.json). Held-out
   whole-game false revocation 0-6.2% (seed-clustered CIs). Wording: "approximately on target; uncertainty does
   not exclude modest exceedances."
   E2 rerun (200 seeds), pooled over noise settings: A1 audit-only +86/+72/+48, CTI +57/+42/+0.4 at p=.05/.10/.20;
   A1S audit-only +89/+60/+18, CTI +51/+23/-59. Per cell (results/e2_per_cell.csv) outcomes depend strongly on the
   false-positive rate. Wording: scripted attackers -> "unprofitable", never "deterred". The earlier, more
   favourable E2 results came from a threshold that over-revoked honest sellers.
   E1b noisy cell rerun with recalibrated theta: CTI reset value +7.7 [1.5, 13.7] (was -30 under the old theta),
   effect of eligibility +9.4 [3.4, 15.2]: under noisy audits a refunded-bond reset becomes modestly profitable.
4. **Analysis history.** Threshold criterion changed from gain < 0 to gain <= 0 (a payoff-responsive policy that
   chooses zero violations has gain exactly 0). Two separate quantities: zero violations chosen (behavioural output
   of the A1-R heuristic) vs estimated mean gain <= 0 (noisy profitability estimate). A1-R's final-3-rounds
   pricing rule is a revised heuristic (pilot finding). E1c was motivated by the default-market null result for H1.
