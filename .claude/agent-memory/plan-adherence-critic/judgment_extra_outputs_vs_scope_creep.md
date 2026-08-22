---
name: judgment-extra-outputs-vs-scope-creep
description: How to judge extra output files beyond a plan's primary_deliverable list, and parent-script seam edits under a "pure additions" plan
metadata:
  type: feedback
---

Two recurring plan-adherence judgment calls (first applied #2162 `persona-specificity-ladder` impl round 1):

1. **A plan's §6.5 `primary_deliverable` YAML is not an exhaustive output list.** Extra output files (e.g. `margin.jsonl`, `conjuncts.jsonl`) that are named as FIGURE SOURCES in the round's `planned_manifest.json` are manifest-REQUIRED, not scope creep — cross-check any "extra outputs" deviation against the manifest's `figures[].source` fields before flagging.
   **Why:** the manifest is a co-equal contract with the plan; an output the manifest needs cannot be a violation of the plan's deliverable list.
   **How to apply:** when the implementer declares "outputs beyond the §X list", grep the manifest sources first; only flag if the extra output serves nothing planned.

2. **Keyword-only seams added to PARENT scripts with byte-equivalent defaults (default `None` ⇒ prior behavior) + an in-diff comment citing the plan section that mandates the reuse satisfy the stated-reason bar under a "pure additions / never edit parent" plan** — when the plan simultaneously mandates IMPORTING (not re-implementing) a parent helper that was hard-coded to the parent's own registries, a minimal parametrization seam is structurally necessary. Judge: default-preserving? in-diff reason citing the plan? existing callers byte-equivalent? If all three, CONCERNS-level at most, not FAIL.
   **How to apply:** read the seam hunk for default values + the docstring/comment; confirm existing call sites pass no new kwarg.
