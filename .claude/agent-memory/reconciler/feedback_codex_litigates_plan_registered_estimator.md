---
name: Codex litigates plan-registered estimator as implementation blocker
description: Codex code-reviewer FAILs an implementation for using the approved plan's registered statistical test when it differs from the original task's published estimator; check the plan's registered test before believing "wrong estimator" Criticals
type: feedback
---

Codex code-reviewer raises a CRITICAL "wrong estimator / label not isolated to the
manipulated variable" blocker against an analysis implementation, when the
implementation is faithfully executing the APPROVED PLAN's pre-registered test —
and the plan (not the implementation) is what chose a different estimator than the
original task published.

**Why:** Task #536 round 1 (2026-06-11). Codex FAILed the #478-flatness
`null-overturned (candidate rescue)` row because the label came from a
cluster-robust OLS Wald while #478's published co-primary was MixedLM (p=0.405).
Codex's factual premise was CORRECT (verified in #478 body + `issue478_analyze.py`),
but plan §3-H2 registered "the published cluster-robust Wald test at α=0.01" as the
falsification trigger, the driver computed the SAME estimator on raw AND centered
(raw p=0.022 NS at α=0.01 vs centered p=0.00075 sig — so centering-attributable at
the registered threshold), and the row carried the MixedLM caveat verbatim with
"candidate rescue pending a MixedLM refit". Codex's proposed fixes (mid-implementation
estimator swap; relabel into the sensitivity-*/partition namespaces) would themselves
violate the registered tree. Reconciled PASS; the residual risk was ledger-gated as a
CONCERN (`478-estimator-mixedlm-refit`: clean-result must carry the caveat).

**Round-2 recurrence (same task, alpha dimension + re-litigation):** #536 round 2,
Codex's only Major FAILed the round-1 fix itself — `holm_binding.binding_alpha: 0.05`
"contradicts plan H2's α=0.01". Same disease: plan §3's decision tree (which governs
regrade LABELS, names "#478 flatness null" in its originally-NULL branch, and routes
through "Holm-corrected p<0.05 within the family") is the registered rule for the
disputed object; H2's α=0.01 registers the hypothesis-FALSIFICATION verdict, a
different object the row records alongside (`holm_reject_at_h2_alpha_001`). Worse,
the r1 BINDING reconcile had already adjudicated the exact alpha point non-blocking
and its standing recommendation explicitly sanctioned the resolution the implementer
took ("tidy the comment to match the binding 0.05 §3 rule" — done, with a
"Reconciler r1: do NOT rebind post-hoc" code comment). Codex re-FAILed with no new
facts. Reconciled PASS again; persisted NIT `478-holm-alpha-plan-tension`.

**How to apply:** When Codex's blocker is "the label/statistic uses estimator X but
the original task published estimator Y": (1) open the approved plan's hypotheses /
decision-tree / Statistic sections and check whether X is the REGISTERED test — if
yes, the implementer followed the contract and the finding is at most a
plan-citation looseness + an analyzer-stage caveat duty, not a code-review FAIL;
(2) check whether the row computes X on BOTH arms (internally consistent contrast)
and records the published-Y number as a caveat — if so, the honest-reporting duty is
already met in the artifact; (3) persist the caveat-propagation duty as a CONCERN so
the analyzer/clean-result gate sees it, rather than bouncing the round; (4) read the
PRIOR round's reconcile marker — if it already ruled the point non-blocking and the
implementer followed a sanctioned option from its standing recommendations, a re-FAIL
on the same point with no new facts is repeat litigation, not a blocker. Companion
patterns: Codex litigating pre-existing trunk code in round N, and Codex stale-read
false positives on worktree files synced from main (here: a spec-freshness sync
commit made `.claude/agents/experimenter.md` look branch-edited; `git diff main --
<file>` was empty).
