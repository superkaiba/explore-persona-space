---
name: Row-merge artifact reuse needs a closed-loop regression test
description: When a plan reuses parent rows by merging them into a new run's input file, demand (a) verification the reused file is the consumer's actual INPUT not a derived output view, (b) a strip/re-inject/reproduce-parent-numbers test
type: feedback
---

When a plan reuses a parent's scored rows (judge scores, eval rows) by MERGING them into the child run's input file before analysis, two things make the reuse trustworthy (#556 v2, 2026-06-10):

1. **Input-vs-view check.** Verify the reused file is the file the consuming script actually READS, not a schema-similar derived OUTPUT view. #556 v1 planned to `cp base_headroom_judge.json` — a judge OUTPUT the analyzer never reads — which would have produced NO_PAIRED_PROMPTS on every cell. The fact-checker caught it via code-read of the consumer (`base_index = _group(rows, kind="base")` reads only `judge_scores.json`).
2. **Closed-loop regression test as the gold standard:** copy the parent's full input file to a scratch slug, STRIP the rows to be reused, re-inject them via the new merge script, re-run the analysis, and assert the parent's headline numbers reproduce EXACTLY (to the float). This tests the merge path itself, not just the analyze path.

**Why:** schema-identical files abound in eval pipelines (input rows vs aggregated views); a merge that targets the wrong file fails silently or floors the paired denominator, and the failure looks like a data problem, not a reuse bug.

**How to apply:** Methodology lens item 9 (reuse fitness). A plan that merges reused rows WITHOUT naming the consumer's actual read path or without a reproduce-parent-numbers loop is incompletely fitness-checked — concern if the merge is trivial, Must-Fix if the merge is load-bearing for the primary DV's denominator.
