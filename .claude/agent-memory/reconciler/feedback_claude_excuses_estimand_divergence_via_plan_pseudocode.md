---
name: Claude excuses estimand divergence via plan-literal pseudocode
description: When the plan's pseudocode formula and its NAMED estimand conflict on the load-bearing cohort, Claude PASSes on "plan prescribed it verbatim"; the estimand governs.
type: feedback
---

When a plan states an estimand by name ("two-way FE residuals... residualized
on source + bystander fixed effects", "within-estimator, standard panel
practice") AND in the same breath gives a shortcut formula
(`y − src_mean − byst_mean + grand_mean`), and the formula is exact only on a
sub-regime (balanced panels) that the PRIMARY cohort violates (16×16 minus
diagonal = unbalanced), Claude code-reviewer PASSes on plan-adherence —
even after rigorously QUANTIFYING the deviation itself (|Δρ| ≤ 0.033, exact-FE
lstsq comparison) — by arguing "the plan's pseudocode prescribed this formula
verbatim" + "qualitative read survives" + persisting a CONCERN. Codex FAILs
on "not measuring the claimed estimand".

**Why Codex wins:** the named estimand governs over the pseudocode when they
provably diverge on the load-bearing cohort. Faithful-to-pseudocode but
unfaithful-to-estimand is a fix-and-disclose case for the implementer, not a
pass-through. Decisive aggravators in #539 r1: (a) the diagnostic was a
round-1 Must-Fix added by BOTH Alternatives critics specifically to remove
source main effects, and the shortcut's residual group means were materially
nonzero (max |mean| ≈ 214 vs within-sd ≈ 691) — i.e. the contamination the
control exists to remove leaked through (gauss_kl "pair affinity" 0.077→0.043
under exact FE, −44% relative); (b) the error is SYSTEMATIC, not stochastic,
and lands under the estimand's name in the JSON the analyzer consumes;
(c) fix ≈ 10 lines, zero GPU, before the production run — maximal cost
asymmetry toward bouncing.

**How to apply:** when both reviewers agree on facts and dispute only
severity of a plan-pseudocode-vs-plan-intent divergence, re-read the plan's
own estimand-naming prose (§11 decision entries, §4 inline comments). If the
formula provably fails the named estimand on the PRIMARY cohort and the fix
is cheap relative to downstream contamination, side with FAIL. "Qualitative
read survives in this instance" does not rescue a control whose purpose is
quantitative adjudication. Companions: "Claude treats predictor formula
mismatch as Nit" (#518), "Claude misses floor-vs-raise divergence" (#532).
Contrast with PASS-leaning precedents: "Codex methodology-choice as code bug"
(plan offered explicit OPTIONS — not the case here) and "#505 r6
plan-section-out-of-brief" (un-invoked path — not the case here: rho_twoway
is computed and reported on the primary cohort).

Origin: task #539 round-1 reconcile (2026-06-09),
`scripts/issue539_residual_per_cohort.py:272` `_twoway_demean`.
