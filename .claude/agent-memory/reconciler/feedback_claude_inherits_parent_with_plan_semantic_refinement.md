---
name: Claude misses plan-vs-parent semantic refinement when impl inherits parent code
description: Claude PASSes round-N when implementer copies parent issue's literal cell/data-construction code; misses that current plan's prose REFINES the semantic claim for the inherited code. Codex catches by reading the plan vs the impl docstring/comment.
type: feedback
---

When the current task's plan EXPLICITLY refines / renames a construct from a parent issue (e.g., parent's `NEG_default_other` = "different question slice WITH-key", current plan §7 says "default-assistant-NO-KEY cell controls for safety target"), and the implementer carries the parent's literal cell-construction code verbatim, Claude code-reviewer PASSes round-N by:
- ticking the round-(N-1) must-fix items in a table,
- verifying tests/lint PASS,
- not re-reading the plan's semantic claims about what each cell/construct is supposed to MEASURE.

Codex catches it by reading the plan + the impl docstring vs the impl body. Smell:
- Plan §K (often §Methodology or §Risk) describes a control cell as "<no-X> cell" / "the safety target" / "controls for Y bleeding into Z".
- Impl builds that cell with the same code shape as the parent's same-named cell, which had different semantics.
- The verbatim-from-parent code is a one-line trigger/persona/flag flip away from the plan's claimed semantics.

**Defense before believing Claude's PASS:** for any inherited cell / persona / data-construction code from a parent issue, grep the CURRENT plan's prose for the cell name + the surrounding "controls for" / "safety target" / "no-key" / "no-trigger" / "with-X" language. If the prose ASSERTS a refined semantic (e.g., "the no-key bystander"), open the impl and verify the construction matches. The 2-line fix vs the 2-line claim is the smell.

**Why:** Origin task #506 round-3. Parent #475 had `NEG_default_other` = "with-key on disjoint OOD questions". Task #506 plan §7 line 381 explicitly refines: "Default-assistant-no-key cell (NEG_default_other) controls for 'is the install bleeding into the default cell' — the safety target." Impl (`probe_issue506_install_validity.py:122-130`, `eval_issue506.py:148-156`) builds it `{"system": asst, "user": _trig(q), ..., "trigger": True}` — literal verbatim from `eval_issue475.py:225-233`. Cell becomes a second T_plus slice on different questions, NOT a no-key leakage control. Data won't measure plan §7's stated safety target; analyzer cannot recover (no-key completions never generated).

**How to apply:** When a #506-style child task's plan body uses prose like "default-assistant-no-key cell", "no-trigger control", "no-key bystander", "the safety target", "controls for X bleeding into Y" — open the implementing scripts and verify the cell construction (the `user:` field, the `trigger:` flag, the persona key) matches the plan's semantic claim, NOT the parent's code shape. Companion to "Claude misses same-file siblings" + "Claude treats round-N-1 must-fix as acceptance" + "Claude misses fix regressions" — same family: Claude over-trusts the round-N-1 must-fix table and stops there.
