---
name: Row-merge artifact reuse needs a closed-loop regression test
description: Merged-row reuse needs (a) the consumer's actual read-path verified (input vs derived view) + (b) a strip/re-inject/reproduce-parent-numbers loop (#556)
type: feedback
---

When a plan reuses a parent's scored rows by MERGING them into the child run's input file, two things make it trustworthy (#556 v2):

1. **Input-vs-view check.** Verify the reused file is the file the consuming script actually READS, not a schema-similar derived OUTPUT view. #556 v1 planned to `cp base_headroom_judge.json` — a judge OUTPUT the analyzer never reads — which would have produced NO_PAIRED_PROMPTS on every cell; caught by code-read of the consumer (`_group(rows, kind="base")` reads only `judge_scores.json`).
2. **Closed-loop regression test (gold standard):** copy the parent's full input file to a scratch slug, STRIP the rows to be reused, re-inject via the new merge script, re-run the analysis, assert the parent's headline numbers reproduce exactly. Tests the merge path, not just the analyze path.

**Why:** schema-identical files abound in eval pipelines; a merge targeting the wrong file fails silently or floors the paired denominator, and the failure looks like a data problem.

**How to apply:** Methodology item 9 (reuse fitness). A merge without the consumer's read path named or without a reproduce-parent-numbers loop is incompletely fitness-checked — concern if trivial, Must-Fix if load-bearing for the primary DV's denominator.
