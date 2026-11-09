# Figure captions

All error bars and bands are 95% confidence intervals (mean +/- 1.96 SE over 30 paired seeds; 200 for Fig. 5).
Coalition gain = the controller's consolidated profit under attack minus the same identities' profit when
playing honestly on the same seed (shared per-round random streams). Defaults: 8 sellers, K = 2, T = 40,
b = 100, p_a = 0.05 unless stated.

**Fig. 1.** Coalition gain by reputation system and attack at 5% auditing. Panels have independent x-axes.
R2 - bond equals R1, and R2 - declaration equals R2, for every attack that does not use undeclared prices
(all but A5): binding terms matter only for declaration-specific pricing.

**Fig. 2.** Coalition gain against audit rate on a fine grid (0.01-0.20), audit-only vs CTI. Dashed lines mark
the deterrence threshold p* (lowest rate from which the attack stops paying at every higher tested rate),
with 95% bootstrap intervals over seeds; "threshold not reached" means the attack still paid at 0.20.

**Fig. 3.** Left: coalition gain against bond size under the CTI at 5% auditing; dashed: A5 under CTI -
declaration, where bait-and-switch is legal and the bond never triggers. Right: A5 surcharge per honest-buyer
transaction with coalition sellers, with transaction volume.

**Fig. 4.** Gain from purchased history (A3 - A1: buy compliant transactions during a 10-round warm-up, then
exploit; A3d - A1: keep buying throughout, diluting the bad ratings that cheating produces), 200 paired seeds.
Default market vs an exploratory reputation-sensitive market that changes several things at once (tau 0.25,
prior Beta(2,2), honest incumbents with 6 clean observations, coalition entering new, free fakes where there are
no audits); it cannot isolate one cause. Right: diagnostic with free fakes in the audit systems too (generous to
the attacker, since an audited fake that delivered nothing would itself be a violation). Paired difference
audit-only minus CTI in the diagnostic: +39 [32, 46] (A3), +55 [47, 62] (A3d); A3d stays profitable under the
CTI (+20 [9, 31]).

**Fig. 5.** Value of a voluntary reset at the start of the final 3 rounds (paired continuations from a cloned
state, A1 policy in both branches), with and without newcomer quota for the replacement. 77% of verifiable-regime
runs in the audit systems (100% under peer rating and under noisy audits) reached the checkpoint with an eligible
identity. Noisy cell uses the recalibrated theta (4.5). Paired intervals in results/e1b_paired_intervals.csv.

**Fig. 6.** Noisy audits, 200 paired seeds, pooled over FN 0.1/0.3 and FP 0.02/0.05 (per-cell results in
results/e2_per_cell.csv; outcomes depend strongly on the false-positive rate). Theta calibrated on seeds
1000-1199 so that honest sellers present for the whole game have <= 5% false-revocation probability; held-out
estimates 0-6.2% (seed-clustered intervals, results/e2_false_revocation_exposure.csv). Scripted attackers:
the figure shows profitability, not behavioural deterrence. Dashed: CTI under verifiable audits.
