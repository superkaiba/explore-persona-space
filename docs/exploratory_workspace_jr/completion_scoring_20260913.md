# Complete-rollout cohort for primary paired scoring

Frozen on 2026-09-13 before any main component fit or main predictability outcome
was read. This implements the existing analysis plan's requirement that paired
primary contexts have five nonempty completed draws in both models. It changes
neither the generation recovery threshold nor the selected prompts.

The producer retains nonempty length-capped answers after the aggregate cap-hit
fraction falls below 2%. That is appropriate artifact retention, but presence of
five captured draws does not establish five completed answers. The original
comparison consumer intersected fitted IDs without checking completion. A separate
outcome-blind ledger now reads only the final uploaded generation records and
pinned checkpoint stopping policy. A context is eligible only when every seed
42–46 is present exactly once, has `finish_reason: stop`, and has a nonempty span
before the first terminal token. Earlier length-capped draws that were replaced
by successful final draws do not cause exclusion. Every excluded seed and reason
remains explicit; no prompt is replaced or backfilled.

The primary cohort is the joint two-model complete-test intersection, further
intersected with actual saved fit rows across all observed/null, sparsity,
rotation and predictor comparisons. Every main within-model and cross-model
panel uses this same cohort and paired bootstrap multiplicities. Original fit
summaries over all captured test rows remain descriptive artifacts. Generation,
capture, decomposition and fit outputs are preserved, with original and paired
coverage reported separately.

Training and validation retain their captured, nonempty K=5 targets, including
explicitly flagged censored answers. The original completion requirement applies
to paired primary scoring. Their censoring counts are a limitation, not five
completed-answer counts. Supplemental diagnostics and learning curves must state
their test cohort; any broader all-captured read remains descriptive and cannot
replace the completed joint-cohort result.
