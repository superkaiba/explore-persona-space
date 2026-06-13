---
name: Claude excuses estimand divergence via plan-literal pseudocode
description: When the plan's pseudocode formula and its NAMED estimand conflict on the load-bearing cohort, the estimand governs — "the plan prescribed this formula verbatim" does not rescue a control whose purpose is quantitative adjudication. FAIL when the error is systematic, lands under the estimand's name, and the fix is cheap pre-production.
type: feedback
---

**Rule:** when a plan names an estimand ("two-way FE residuals", "within-estimator") AND gives a shortcut formula exact only on a sub-regime (balanced panels) the PRIMARY cohort violates, the named estimand governs. FAIL when (a) the error is SYSTEMATIC (not stochastic) and lands under the estimand's name in the JSON the analyzer consumes; (b) the diagnostic was a Must-Fix added specifically to remove the contamination that leaks through (check residual group means vs within-sd); (c) the fix is cheap relative to downstream contamination (≈10 lines, zero GPU, pre-production). "Qualitative read survives" does not rescue; Claude even quantifying the deviation itself (|Δρ| ≤ 0.033) and persisting a CONCERN is fix-and-disclose territory for the implementer, not pass-through.

**Origin:** #539 r1 — `_twoway_demean` shortcut `y − src_mean − byst_mean + grand_mean` on the unbalanced 16×16-minus-diagonal cohort; exact-FE comparison moved gauss_kl "pair affinity" 0.077→0.043 (−44%).

Contrast with PASS-leaning precedents: [[feedback_codex_methodology_choice_as_bug]] (plan offered explicit options — not the case here) and [[feedback_codex_plan_section_in_scope]] (un-invoked path — not the case: the statistic is computed and reported on the primary cohort). Companions: [[feedback_claude_treats_predictor_formula_mismatch_as_nit]]; [[feedback_claude_misses_floor_vs_raise_divergence]].
