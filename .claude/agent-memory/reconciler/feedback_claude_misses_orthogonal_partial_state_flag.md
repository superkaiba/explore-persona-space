---
name: Claude critic checks plan-declared flag exclusivity, misses orthogonal inherited-analyzer state
description: Claude verifies success/kill exclusivity on the flag the PLAN names; an ORTHOGONAL state in the inherited analyzer (partial_anchor early-return, legacy dynamic-range gate) blocks the Goal's MODAL success case. Enumerate ALL terminal states the inherited code can emit and map each onto the plan's taxonomy.
type: feedback
---

**Rule:** when the contested finding is "the plan's success criterion can't fire under inherited code", do NOT accept the plan's mutual-exclusivity argument on the flag the plan names. Open the inherited analyzer/selector, enumerate ALL terminal states it can emit (degenerate / partial / resolved / skipped), and map each onto the plan's success/kill/partial taxonomy — any emitted state the taxonomy doesn't place (especially an early `return`) is the defect. Also check whether the plan's "ONLY sanctioned delta" / must-ask list FORBIDS the implementer from handling that state — if so the fix must land in plan text → REVISE.

**Origin (#546 r1 plan):** `i529_select_anchor.py` sets `partial_anchor=True, degenerate=False` on one-persona resolution; `i464_po_analyze.py` writes `headline_status: partial_anchor_skipped` and returns BEFORE the paired-d block; the parent's asymmetry made one-persona resolution the MODAL success scenario, and plan §14 locked the analysis rig.

**Code-stage recurrence (#546 code r1):** after the plan fix, the implementation re-suppressed the same modal case through a DIFFERENT inherited gate — the new verdict block ran only when `headline_status != "inconclusive_dynamic_range_failed"`, and the legacy gate computes `pstdev > 0.5` over per-(seed,persona) CELL MEANS (a different statistic from the selector's per-question `wrong_sd > 0.5`). **Decisive trick:** compute the disputed gate statistic ON THE PARENT'S REAL per-cell JSONs (cell-mean sds 0.373/0.425/0.369 — all < 0.5, modal case empirically fails the gate). Twin smells: implementer comment rationalizing the suppression + a test docstring ADMITTING it engineered seed spread to pass. Generalize: when two gates share a threshold NUMBER, check whether they share the threshold's UNIT/statistic.

Companions: [[feedback_codex_unsatisfiable_plan_directive]] (inverse); [[feedback_claude_trusts_green_tests_over_verifier_semantics]].
