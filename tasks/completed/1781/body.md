---
title: 'workflow-fix: gotchas entry — real-corpus exact-dupe sha sampling + frozen-pinned-split
  uniqueness trap'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4ce661a0150e
created_at: '2026-07-29T01:29:50Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate: yes from #1768 crash-fix round 4
  (duplicate prompt shas; 82 dup rows in the frozen #779 pinned split)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson `gotcha_candidate: yes` raised on task #1768 (emitting agent: experiment-implementer, crash-fix round 4).

## Goal

Add a `.claude/rules/gotchas.md` entry: sha-keyed sampling from near-dupe-screened real corpora must dedup DURING the draw; measure frozen pinned splits before asserting global uniqueness over them.

## Workflow gap

- **Bug observed:** #1768 p0 crashed pod-side (`AssertionError: duplicate prompt shas in sample`, issue1768_capture.py:540) sampling 16,400 rows from the #779 n1M corpus — the corpus's near-dupe screen was vs 1,400 eval targets only, so exact duplicates survive inside the pool; additionally the frozen #779 pinned val/test itself carries 82 internal duplicate-sha rows (1,400 → 1,318 unique; val∩test overlap 13), so a GLOBAL sha-uniqueness postcondition over the frozen split is unsatisfiable by construction and would have re-crashed the relaunch.
- **Why it is a workflow gap:** gotchas.md's real-corpus entry family (streaming-filter field semantics, #1092/#1739) is silent on exact-duplicate content in near-dupe-screened corpora and on the frozen-pinned-split-uniqueness trap; every synthetic smoke passes while the real corpus crashes production (the same green-smoke/red-production class the sibling entries exist for).
- **Confidence (emitter):** high
- verified-at-filing: `grep -rln 'exact-duplicate\|duplicate prompt shas\|dedup DURING' .claude/rules/ CLAUDE.md` → 0 hits in .claude/rules/ + CLAUDE.md (absence claim — the 0-hit in-target result is the evidence; the only repo hits are experiment scripts/pycache, out of surface) (2026-07-29). Sibling-entry check: `grep -n 'Real-corpus streaming' .claude/rules/gotchas.md` → L314 (the #1092/#1739 field-semantics entry — adjacent class, does not cover exact-dupe sampling or pinned-split uniqueness).

## Proposed change (candidate diff sketch — refine in planning)

```
+ .claude/rules/gotchas.md (next to the L314 real-corpus streaming-filters entry):
+ - **Sha-keyed samples from near-dupe-screened real corpora contain exact
+   duplicates — dedup DURING the seeded draw (taken-set seeded with pinned
+   rows, top up from the continuing permutation), and MEASURE a frozen
+   pinned split's own duplicate content before writing any uniqueness
+   assert that quantifies over it (#779's val/test: 82 internal dup-sha
+   rows, val∩test overlap 13). Scope postconditions to what the sampler
+   guarantees (train-unique + train∩pinned = ∅ + exact counts), never
+   global. (#1768 p0 crash, fix 07823360; root cause confirmed.)
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'duplicate prompt shas\|dedup during' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; LESSONS.md index row untouched (gotchas.md already indexed) unless the trigger line needs the new class named.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 4ce661a0150e

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: issue1768 p0 crashed on duplicate prompt shas sampling the #779 n1M corpus; the frozen pinned val/test itself carries 82 internal duplicate-sha rows making a global-uniqueness postcondition unsatisfiable by construction
why_workflow_gap: the gotchas real-corpus entry family covers streaming-filter field semantics but not exact-duplicate content in near-dupe-screened corpora nor the frozen-pinned-split uniqueness trap; synthetic smokes stay green while real-corpus production crashes
proposed_change: add a gotchas.md entry: sha-keyed sampling from near-dupe-screened real corpora must dedup during the draw; measure frozen pinned splits before asserting global uniqueness over them
diff_sketch: |
  + gotchas.md bullet next to the real-corpus streaming-filters entry (see body)
confidence: high
related_task: #1768
<!-- /workflow-fix-candidate -->
