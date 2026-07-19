---
title: 'workflow-fix: isolate root-guard self-test git state'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d6a427bd7798
- daily-auto-filed
created_at: '2026-07-19T07:08:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): 13 spurious test_check_git_recipes_root_guard
  failures because the live-hook probe read ambient shared-root git state fail-open
  during a concurrent git op, costing a baseline compare + 17-min full re-run (c3-P16).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the 2026-07-18 /daily
transcript problem sweep (c3-P16). Route-2 filing.

## Goal

Run the `test_check_git_recipes_root_guard_*` probe against an ISOLATED temp
git repo (or serialize it on the shared-root lock) so a concurrent
shared-root git op cannot make the live-hook probe read fail-open and produce
spurious failures.

## Workflow gap

- **Bug observed:** 13 spurious `test_check_git_recipes_root_guard_*` failures
  in #1506's gate — the guard probe read fail-open (blocked-probe rc=0) while
  a concurrent shared-root git op was in flight; cost a baseline compare + a
  17-min full re-run.
- **Why it is a workflow gap:** `check_git_recipes_root_guard` invokes the
  live hook (`guard_repo_root_branch.sh`) as a subprocess probe; the hook's
  git-state read resolves against the AMBIENT shared repo rather than the
  `tmp_path` fixture, so a concurrent branch/reset op on the shared root
  transiently flips the probe's verdict — a fixture-isolated test that isn't
  actually git-isolated.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'test_check_git_recipes_root_guard' tests/test_workflow_lint.py` → the fixtures pass `repo_root=tmp_path` and write the recipe fence into tmp_path, but the invoked hook resolves git state in the ambient cwd (no per-probe `GIT_DIR`/cwd pin to the fixture); a concurrent shared-root git op therefore perturbs the probe (2026-07-19). NOTE the spawned session must first REPRODUCE the fail-open mechanism (run the probe under a simulated concurrent shared-root op) before choosing isolate-git-state vs lock-serialize.

## Proposed change (candidate diff sketch — refine in planning)

```
# Pin the probe's git resolution to the fixture repo (isolate), e.g. run the
# hook with cwd=tmp_path + a tmp_path .git (git init), or GIT_DIR/GIT_WORK_TREE
# pointed at the fixture, so the ambient shared-root branch/HEAD is never read;
# OR serialize the probe on ~/.task-workflow/lock. Prefer isolation (removes
# the shared-state dependency entirely) over serialization (still contends).
```

## Scope / surfaces

- Primary target: `tests/test_workflow_lint.py, scripts/guard_repo_root_branch.sh`
- Determine whether the fix belongs in the test harness (isolate the probe's
  git env) or in the hook (honor an explicit repo-root override); the test
  side is preferred if the hook already accepts a directory argument.

## Constraints / invariants

- Workflow-surface only. Must not weaken the guard's real protection — the fix
  isolates the TEST probe's git state, not the live hook's shared-root guard.
- `scripts/workflow_lint.py --check-asks` passes; the full guard self-test
  matrix stays green AND becomes concurrency-robust.
- Recursion guard applies.

## Provenance

- workflow_fix_target: tests/test_workflow_lint.py, scripts/guard_repo_root_branch.sh
- fingerprint: 722169522df4

Surfaced problem (c3-P16): 13 spurious root-guard test failures from a
fail-open probe reading ambient shared-root git state during a concurrent git
op; a baseline compare + 17-min re-run burned.
