---
title: 'workflow-fix: vectorized rewrites land on main + tombstone serial twin'
kind: infra
tags:
- wf-fix
- wf-fix-fp:75b60cab8677
created_at: '2026-07-02T16:10:45Z'
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

Require every vectorization rewrite to land the batched helper on `main` in the same round AND tombstone the superseded serial entrypoint, so reusers and same-issue follow-ups cannot silently re-run the serial original.

## Workflow gap

- **Bug observed:** three build-but-strand / duplicate events in one week: (1) #722's vectorized helper was built on the unmerged `vectorized-mlp-skill` branch (commit 452c539c31) and then merge-FAILED to main (`epm:merge-failed ... reason=new-shared-src-infra-cannot-land-via-artifact`) while the #722 session kept running the OLD serial script (~38h ETA); (2) #778's same-issue follow-up RE-RAN the serial 1000-draw null battery on 2026-07-02 because the batched `perm_null_draws` was not yet on main (confirmed in #836: "the batched perm_null_draws isn't on main, so the rule flags the serial original") — user caught it twice; (3) #834 and the #778 fix session independently vectorized the SAME module in parallel, discovering the overlap only at ~10:09 ("a parallel fix #834 independently landed the equivalent on main").
- **Why it is a workflow gap:** `.claude/rules/workflow-fix-on-bug.md` § "Built-but-stranded fixes don't help" states the lesson but prescribes no mechanism; `vectorize-many-cell-fits.md` names the canonical helper but nothing prevents a still-importable serial twin from being called by the next reuser, and nothing makes an in-session emergency fixer check for an already-open fix task on the same file.
- **Confidence (emitter):** high.

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/rules/vectorize-many-cell-fits.md`, add a "Supersede contract" section:

```
+ ## Supersede contract — land on main + tombstone the serial twin
+ A vectorization rewrite that supersedes a serial helper MUST, in the SAME
+ round: (1) land the batched helper on main (not a worktree artifact — the
+ #722 merge-fail is the incident); (2) TOMBSTONE the serial entrypoint:
+ either delete it, or make it emit a loud DeprecationWarning (or raise on
+ EPM_FORBID_SERIAL_FITS=1) naming the batched replacement, so a reuser /
+ same-issue follow-up cannot silently re-run it (#778's follow-up re-ran
+ the serial null battery; #667 called #722's old fit_cell).
+ (3) BEFORE starting an in-session vectorization fix, check for an open fix
+ task on the same file (task_workflow.is_open_workflow_fix_task + a grep of
+ proposed/running infra titles) and adopt/coordinate instead of duplicating
+ (#834 vs the #778 fix session duplicated the same rewrite).
```

Cross-reference the contract from `workflow-fix-on-bug.md` § Built-but-stranded (one pointer line).

## Scope / surfaces

- Primary targets: `.claude/rules/vectorize-many-cell-fits.md`, `.claude/rules/workflow-fix-on-bug.md`
- NOTE: open task #869 also edits `vectorize-many-cell-fits.md` (widens the trigger to dense linear-algebra fits). Distinct concern, distinct fingerprint — coordinate merges via the normal worktree rebase; if #869 lands first, rebase over it.

## Constraints / invariants

- Workflow-surface only. Actual tombstoning of specific serial helpers (e.g. `issue722_skill_over_mean.py`) is follow-on implementation the rule mandates for FUTURE rewrites; this task may tombstone at most the two known live offenders if trivially safe, else file them.
- `scripts/workflow_lint.py` default run passes; keep `.claude/rules/LESSONS.md` index row wording in sync.
- Recursion guard: this session must not auto-route its own subagents' workflow-fix candidates.

## Provenance

- workflow_fix_target: .claude/rules/vectorize-many-cell-fits.md, .claude/rules/workflow-fix-on-bug.md
- fingerprint: 75b60cab8677

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/vectorize-many-cell-fits.md, .claude/rules/workflow-fix-on-bug.md
bug_observed: #722's vectorized helper merge-failed and stranded off main; #778's same-issue follow-up re-ran the serial null battery because the batched fix was not on main; #834 and the #778 fix session independently duplicated the same vectorization
why_workflow_gap: the built-but-stranded lesson is documented but has no mechanism — nothing forces the batched helper onto main, nothing disables the serial twin, and nothing makes fixers check for an open fix task before duplicating work
proposed_change: require a vectorization rewrite to land the batched helper on main in the same round and tombstone the superseded serial entrypoint with a deprecation warn/raise pointing at the batched path, and require fixers to check for an open fix task on the same file before starting
diff_sketch: |
  + ## Supersede contract: (1) batched helper on main same round;
  + (2) serial entrypoint deleted or DeprecationWarning/raise naming the
  +     batched replacement; (3) fixers check is_open_workflow_fix_task +
  +     open infra titles before starting an in-session rewrite
confidence: high
related_task: n/a (user-chat transcript audit, 2026-07-02)
<!-- /workflow-fix-candidate -->
