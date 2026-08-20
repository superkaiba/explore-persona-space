---
name: mutation-certified-test-instrument
description: Certify a test-instrument commit by semantic mutation on HEAD when the parent-commit probe only yields AttributeError; grep-verify every sed mutation APPLIED before trusting a green run; tie-break tests keyed on dict order pass by insertion accident
metadata:
  type: feedback
---

When a fix round's TEST-INSTRUMENT commit adds tests for a defect FIXED in an
earlier commit of the SAME round, the [[fails-pre-fix-probe-parent-commit]]
swap against the round-parent often fails only with AttributeError on a
helper the fix introduced — the WEAK form (proves API binding, not that the
asserts catch the semantic defect). The strong form: surgically re-introduce
each named r1 defect into HEAD's FIXED body (drop the gen-done branch, disable
the `if uncovered:` PARTIAL guard, flip min→max / `>=`→`>` / tie polarity) and
confirm exactly the intended test fails. cp-backup to /tmp + cp-restore avoids
git verbs entirely (no guard interaction); confirm restore with an empty
`git diff --stat HEAD -- <file>`.

**Why:** #2389 R2 g5 — the b12 cap-report tests AttributeError'd on the r1
tip (`_group_by_cell` was new in the g2 fix commit), so only HEAD-mutation
probes could certify the parity/vllm-namespace and PARTIAL-branch asserts;
both mutations were caught by exactly the intended test, upgrading the
verdict from traced to measured.

**How to apply — two pitfalls measured in the same round:**
1. **A line-addressed sed that doesn't match is a SILENT no-op** and the
   green run reads as "mutation not caught". Always grep the mutated token
   immediately after the sed and before the pytest; my first tie-polarity and
   tie-drop probes were no-ops (`3908s|...|` on a pattern living at :3910).
   Prefer a python replace with `assert old in s` for multi-line mutations.
2. **A tie-break test whose fixture inserts the preferred key FIRST passes by
   dict-insertion-order accident** when the tie key is deleted outright
   (`min` returns the first minimal element). The polarity flip IS caught;
   the deletion is not. Hardening: author the tie fixture with the
   non-preferred key first. Flag as Minor when production builds the dict in
   the preferred order anyway (deletion is then behavior-preserving).

Pairs with [[decision-rule-inlined-in-gpu-phase]] (the extraction this
certified) and [[fails-pre-fix-probe-parent-commit]] (the weak/strong split).
