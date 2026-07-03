---
title: 'workflow-fix: artifact-reuse fitness check gains throughput item (i)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0102a775d4d7
created_at: '2026-07-02T16:10:26Z'
has_clean_result: false
origin_prompt: 'User (2026-07-02): "A lot of recent transcripts show that parallelization/vectorization
  is not properly being used for experiments which is leading to slowdowns -- Read
  the recent transcripts, figure out why this is happening, and propose workflow improvments
  to fix this" -> after the 7-proposal audit report: "Make all seven changes. Figure
  out the best way"'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a user-approved transcript audit (2026-07-02 PM-chat session; user: "parallelization/vectorization is not properly being used ... Make all seven changes"). Emitting agent: orchestrator, via 5 parallel transcript-mining subagents.

## Goal

Add a throughput fitness item (i) to the artifact-reuse checklist: reused fit/analysis code gets its inner loop and device routing inspected against `vectorize-many-cell-fits.md`, with fixes landing at the source module rather than the caller.

## Workflow gap

- **Bug observed:** four incidents in one week came from reusing a parent task's fit/analysis code whose serial inner loop or hardcoded device pin nobody re-inspected: #761 reused #658's `issue658_fit_predictors._ridge_predict_loco` (Python loop over 50 LOCO folds, eigh+solve per fold) → 100x over plan (0.3h → 30h/behavior projected); #667's inline analysis reused #722's serial per-cell `fit_cell` (300-epoch MLP per cell × 84 cells, MLP already known-uninformative); #763 and #812 inherited the hardcoded `DEVICE = "cpu"` module constant from `issue658_fit_predictors.py:143`, which defeated vectorization even after ops were batched (#763: ~67 min/behavior on CPU vs ~2h total on GPU after the `EPM_FIT_DEVICE=cuda` cutover).
- **Why it is a workflow gap:** `.claude/rules/artifact-reuse.md` mandates the (a)-(h) fitness check for reused trained artifacts and data, and CLAUDE.md's "Reuse existing experiment code" bullet makes code reuse the default — but NO fitness item covers throughput character. Reuse currently propagates serial loops and stale device pins silently; each downstream task rediscovers the same blowup.
- **Confidence (emitter):** high.

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/rules/artifact-reuse.md`, extend the fitness checklist:

```
+ (i) Throughput fitness (reused CODE only — fit/analysis/eval helpers): before
+   adopting a parent's fit/analysis helper, READ its inner loop and device
+   routing. Two checks: (1) the per-cell/per-fold/per-draw axis is batched
+   (vectorize-many-cell-fits signature — a serial Python loop over cells,
+   folds, draws, or rows fails); (2) the device is parametrized (a hardcoded
+   DEVICE constant or implicit CPU pin fails — #763/#812 inherited
+   issue658_fit_predictors.py's DEVICE="cpu" and burned hours batched-on-CPU).
+   On a failure: fix at the SOURCE module (parametrize/batch it there) so
+   every future reuser inherits the fix — never a caller-side workaround.
+   State the inspection result (function name, batched-or-serial, device
+   handling) in the plan's reuse map / §9 row.
```

Mirror a one-line pointer in the CLAUDE.md artifact-reuse bullet ("(a)-(h)" → "(a)-(i)") and every workflow-surface file that enumerates the (a)-(h) list (grep for "(a)-(h)").

## Scope / surfaces

- Primary target: `.claude/rules/artifact-reuse.md`
- Secondary: every file enumerating the fitness list — `grep -rln '(a)-(h)' .claude/ CLAUDE.md` and update each hit to (a)-(i).
- Cross-link: open task #869 Fix D adds a named-helper-actually-called review check (code-reviewer/consistency-checker side); this task owns the reuse-time inspection. Companion plain-infra task (filed in the same batch) parametrizes the concrete `issue658_fit_predictors.py:143` DEVICE pin.

## Constraints / invariants

- Workflow-surface only — the concrete DEVICE code fix belongs to the companion infra task, not this one.
- `scripts/workflow_lint.py` default run passes; CLAUDE.md and the rule file stay consistent.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/artifact-reuse.md
- fingerprint: 0102a775d4d7

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/artifact-reuse.md
bug_observed: four incidents (#761 100x, #667, #763, #812) reused a parent's fit/analysis code whose serial inner loop or hardcoded DEVICE=cpu nobody re-inspected; the artifact-reuse fitness check (a)-(h) has no throughput item
why_workflow_gap: code reuse is the project default but the reuse fitness checklist never inspects throughput character, so serial loops and stale device pins propagate silently to every downstream task
proposed_change: add throughput fitness item (i) to the reuse checklist: reused fit/analysis code gets its inner loop and device routing inspected against vectorize-many-cell-fits, with fixes landing at the source module rather than the caller
diff_sketch: |
  + (i) Throughput fitness (reused code): inner loop batched? device
  +     parametrized? On failure fix at the SOURCE module; record the
  +     inspection (fn name, batched-or-serial, device) in the plan reuse map.
confidence: high
related_task: n/a (user-chat transcript audit, 2026-07-02)
<!-- /workflow-fix-candidate -->
