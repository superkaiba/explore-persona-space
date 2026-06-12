---
name: Claude misses plan-vs-parent semantic refinement when impl inherits parent code
description: The current plan REFINES a parent construct's semantics ("default-assistant-NO-KEY cell = the safety target") while the implementer carries the parent's literal cell-construction code verbatim; Claude ticks the must-fix table, Codex reads plan prose vs impl body.
type: feedback
---

**Rule:** for any cell / persona / data-construction code inherited from a parent issue, grep the CURRENT plan's prose for the construct name + surrounding "controls for" / "safety target" / "no-key" / "no-trigger" language. If the plan asserts a refined semantic, open the implementation and verify the construction matches the PLAN's claim, not the parent's code shape (the `user:` field, the `trigger:` flag, the persona key). The 2-line code vs 2-line claim divergence means the data won't measure the plan's stated target and the analyzer cannot recover (the required completions are never generated) → FAIL.

**Origin:** #506 r3 — parent #475's `NEG_default_other` = with-key on disjoint OOD questions; #506 plan §7 refines it to "default-assistant-no-key cell ... the safety target"; impl built it verbatim from `eval_issue475.py` with `trigger: True` → a second T_plus slice, not a no-key leakage control.

Companions: [[feedback_claude_misses_same_file_siblings]] (must-fix-table-walk root cause); [[feedback_claude_misses_fix_regressions]].
