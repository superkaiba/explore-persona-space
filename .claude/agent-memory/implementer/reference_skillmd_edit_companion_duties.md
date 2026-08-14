---
name: skillmd-edit-companion-duties
description: Any issue/SKILL.md content edit owes two companion checks — sweep sibling prose-pin tests for exact-string asserts on removed text, and re-check the SKILL_DOC_SIZE_GRANDFATHER byte ratchet (same-round cap raise per the landing-bytes convention)
metadata:
  type: reference
---

Two companion duties fire on ANY `.claude/skills/issue/SKILL.md` content edit
(#2240, 2026-08-12):

1. **Sibling prose-pin sweep BEFORE editing.** Many `tests/test_issue_skill_*.py`
   files pin exact SKILL.md strings. A restructure that removes/moves a pinned
   string breaks a WORKFLOW_INVARIANT test that is NOT in your plan's file list
   (#2240: `test_issue_skill_pr_state_probe.py:69` pinned the exact
   `if [ "$PR_STATE" != "OPEN" ]; then` line the fix removed by design).
   Grep `tests/` for every string your diff deletes/moves FIRST; update the
   sibling pin in the same commit as a stated forced deviation.

2. **Byte-ratchet check AFTER editing.** `workflow_lint.py`
   `SKILL_DOC_SIZE_GRANDFATHER["issue/SKILL.md"]` is a regrowth ratchet with
   only ~1-3 KB headroom at any time (headroom hygiene FAILs a cap >3,000 B
   above live size). Any net addition ≳1 KB trips the no-flags lint = the
   Step 9c gate. The SANCTIONED remedy for plan-mandated growth is a SAME-ROUND
   cap raise to `measured landing bytes + ~1-1.5 KB` with the delta documented
   in the table comment (the #1753/#1727 landing-bytes rule; precedents
   #2115/#2074/#2041 — three raises in three days). Run
   `tests/test_workflow_lint_skill_doc_size.py` (fast, ~0.4 s) after the raise.
   Do NOT butcher plan-mandated content to duck the ratchet when the arithmetic
   cannot fit.

**How to apply:** both checks are cheap (one grep, one `wc -c` + table read) and
belong in the same round as the SKILL.md edit — missing either costs a review
bounce or a fleet-wide gate FAIL discovered ~30 min later.
