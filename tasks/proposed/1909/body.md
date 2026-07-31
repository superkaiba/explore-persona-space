---
title: 'workflow-fix: gotchas entry for function-scoped import-check name shadow (UnboundLocalError
  class)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bdf38a539896
created_at: '2026-07-31T02:50:34Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1739 wcrung crash-fix round:
  an import inside a function binds that name function-wide, so an --import-check
  block importing a bare name that matches a module-level def poisons every later
  call, and no import-check can catch it because it passes by construction. Put the
  block in its own function and pin it with a co_varnames assert.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes`
failure-lesson raised on task #1739 (emitting agent: wildchat-runner teammate
session's crash-fix round; orchestrator routed).

## Goal

Add a gotchas.md entry for the function-scoped import-check name-shadow class
(an import inside a function binds the name function-wide, so an
--import-check block importing a bare module name that matches a module-level
def poisons every later call in that function; import-check smokes pass by
construction).

## Workflow gap

- **Bug observed:** issue1739_wcrung_pod.py main() --import-check block
  imported the capture module, making capture a compile-time local of main, so
  the production phase-2 call to the module-level def capture raised
  UnboundLocalError after a full generation phase.
- **Why it is a workflow gap:** the --import-check pattern is a
  gotchas-prescribed defense (the #606 lazy-import entry) whose canonical
  in-function shape SELF-DEFEATS when any imported bare name matches a
  module-level symbol the function later reads — the trap lives in the
  workflow-recommended pattern itself and no existing gotchas entry covers it
  (the #816 entry covers import-check smokes missing GPU-bound trainer
  construction; the #823 entry covers sys.path in script mode; neither covers
  the compile-time local shadow).
- **Confidence (emitter):** high
- verified-at-filing: incident verified from primary evidence, not grep-only —
  crash log workload.log lines 2428-2438 of
  issue1739_partial/att-20260731-013823-wcrung carry the exact
  UnboundLocalError traceback at issue1739_wcrung_pod.py:552; the fix commit
  resolves (`git merge-base --is-ancestor 5b78ff42d6bb95761cb453cf873948296dc50a16
  origin/issue-1739` → true, checked 2026-07-31); the fix round's pin
  test_main_phase2_reaches_module_level_capture was verified to fail pre-fix
  with the production error. Dedup greps at compose time:
  `grep -rn 'co_varnames\|import-check\|import_check\|UnboundLocalError'
  .claude/rules/gotchas.md` → 2 hits, both read in context, neither implements
  this entry (the #816 and #823 entries, distinct traps as above);
  `task_workflow.is_open_workflow_fix_task('.claude/rules/gotchas.md',
  'bdf38a539896')` → None.

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md entry (near the #606 lazy-import / #823 script-mode entries):
+ **A function-body `--import-check` block that imports a bare name matching a
+ module-level symbol SHADOWS it function-wide — the later call to the
+ module-level def raises UnboundLocalError in production, while the
+ import-check smoke passes BY CONSTRUCTION (the branch binds the name) and a
+ direct-call smoke of the target function never executes the caller's line.**
+ RULE: hoist the import-check body into its own module-level function (it
+ reads no enclosing-function state), or alias every imported name; pin with
+ `assert '<name>' not in <entrypoint>.__code__.co_varnames` (compile-time, no
+ GPU); a smoke of a multi-phase main() must execute the REAL phase-entry
+ lines (boundaries faked), not call the phase functions directly.
+ (Incident #1739 wcrung att-20260731-013823: UnboundLocalError 'capture' at
+ the capture-phase entry after a completed+persisted generation phase; one
+ A100 launch cycle; fix 5b78ff42d6 hoisted the block to _import_check() and
+ swept 7 sibling entrypoints — zero further shadows.)

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Also update `.claude/rules/LESSONS.md` gotchas.md trigger row if the new
  entry adds a fires-when clause worth indexing (likely covered by the
  existing "write/debug ... orchestration code" trigger; planner's call).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-lessons-index` passes after the edit.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST
  NOT auto-route any of its own subagents' workflow-fix candidates (recursion
  guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: bdf38a539896

Surfaced prose (verbatim from the teammate's failure-lesson): "an import
inside a function binds that name function-wide, so an `--import-check` block
importing a bare name that matches a module-level def poisons every later
call, and no import-check can catch it because it passes by construction. Put
the block in its own function and pin it with a `co_varnames` assert."
(generalizes: yes, gotcha_candidate: yes; epm:experiment-implementation v8 on
#1739 carries the full block.)
