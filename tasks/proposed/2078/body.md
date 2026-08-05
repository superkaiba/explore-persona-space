---
title: 'workflow-fix: register eval_results/issue_1481 sparse cone (Step 9c collection
  abort)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:dbdd8c1a526b
created_at: '2026-08-05T04:29:18Z'
has_clean_result: false
origin_prompt: 'Orchestrator-observed during /issue 2054 Step 9c: pytest aborted at
  collection (rc=2, ''Interrupted: 1 error during collection'') because tests/test_issue1947_worker_dispatch.py
  hard-reads eval_results/issue_1481/analysis/verdict_manifest.json via scripts/issue1947_cells.py:109,
  and eval_results/issue_1481 is absent from tests/sparse_cones.txt on branch AND
  origin/main. #671 class.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gap the orchestrator hit
directly while running the `/issue` Step 9c test-verdict gate for task #2054
(emitting agent: orchestrator, session 7a1632b8).

## Goal

Register `eval_results/issue_1481` in `tests/sparse_cones.txt` so the Step 9c
full-suite gate collects in a sparse worktree.

## Workflow gap

- **Bug observed:** `tests/test_issue1947_worker_dispatch.py` fails at COLLECTION
  in any sparse worktree with
  `FileNotFoundError: #1481 verdict manifest missing at
  <repo>/eval_results/issue_1481/analysis/verdict_manifest.json`. Because it is a
  COLLECTION error, pytest aborts the whole run (`Interrupted: 1 error during
  collection`, rc=2) — the gate produces NO verdict at all, not a single failing
  test. On #2054 this cost a full gate attempt.
- **Why it is a workflow gap:** this is the exact #671 class CLAUDE.md documents —
  "the full pytest suite (the `/issue` Step 9c test-verdict) reads a handful of
  OTHER issues' committed `eval_results/issue_<M>/` artifacts as fixtures; those
  dirs are excluded by default, so `new_worktree.sh` pre-adds every cone in
  `tests/sparse_cones.txt` ... If a NEW test starts hard-reading
  `repo_root()/eval_results/issue_<M>/...`, add that issue's dir to the registry;
  a fresh full-suite `FileNotFoundError` in a sparse worktree is the symptom of a
  missing line (#671)." The registry line was never added when the consuming test
  landed, so EVERY sparse worktree running the full gate hits this — it is not
  specific to #2054.
- **Confidence (emitter):** high — reproduced directly, and fixed locally by
  `git sparse-checkout add eval_results/issue_1481` (5.1 MB), after which the gate
  ran to completion (7513 passed).
- verified-at-filing: `git show origin/main:tests/sparse_cones.txt | grep -nE '^\s*eval_results/issue_1481\s*$'` → **0 hits on origin/main** (absence confirmed); relocation sweep `git show origin/main:tests/sparse_cones.txt | grep -n "1481"` → 1 hit, line 45, a COMMENT about a different dir (`eval_results/issue_1434`), so no alternate spelling of this cone exists; the registry currently holds 17 `eval_results/issue_*` cones; landed-fix check `git log --oneline --since='7 days ago' origin/main -- tests/sparse_cones.txt` → no commit registering this cone (2026-08-05 UTC).

### Call-hop target tracing (rule clause (g))

The failing TEST never names `issue_1481`. The path is CONSTRUCTED at
`scripts/issue1947_cells.py:109` (`VERDICT_MANIFEST_PATH`), which the test
evaluates at import/collection time. Symptom site =
`tests/test_issue1947_worker_dispatch.py`; construction site =
`scripts/issue1947_cells.py:109`. Neither is the fix surface: per the #671 rule
above the fix belongs in the cone REGISTRY, which is why `target_file` is
`tests/sparse_cones.txt`.

Corroborating evidence that the condition is already known but unregistered:
`scripts/issue1900_prep.py:155` carries the comment "(the sparse worktree carries
no eval_results/issue_1481 copy)" — a sibling script works around the very gap
the registry should have closed.

## Proposed change (candidate diff sketch — refine in planning)

```
  tests/sparse_cones.txt
+ eval_results/issue_1481
+ #   eval_results/issue_1481 — test_issue1947_worker_dispatch.py (collection-time
+ #   read of analysis/verdict_manifest.json via scripts/issue1947_cells.py:109)
```

The planner should also consider whether a collection-time hard read is the right
shape at all: a missing fixture that aborts the ENTIRE gate is strictly worse than
one that fails a single test. A lazy read (module-level constant, resolved inside
the test body) would degrade to one failure instead of a zero-verdict abort. That
is a judgment call for the planner with the file open — the registry line is the
minimum fix.

## Scope / surfaces

- Primary target: `tests/sparse_cones.txt`
- Secondary (planner's call): `scripts/issue1947_cells.py` collection-time read shape.
- Sweep for siblings: other tests hard-reading `repo_root()/eval_results/issue_<M>/`
  whose cone is unregistered — `grep -rn "eval_results/issue_" tests/ | grep -v tmp_path`
  cross-checked against the 17 registered cones. A second unregistered cone would
  produce the identical zero-verdict abort.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Verify the fix the way the bug was found: create a FRESH sparse worktree via
  `bash scripts/new_worktree.sh` and confirm the Step 9c universe COLLECTS
  (`pytest --collect-only` on the selector output) with no FileNotFoundError.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: tests/sparse_cones.txt
- fingerprint: dbdd8c1a526b

<!-- workflow-fix-candidate v1 -->
target_file: tests/sparse_cones.txt
bug_observed: tests/test_issue1947_worker_dispatch.py fails at COLLECTION in any sparse worktree with FileNotFoundError on eval_results/issue_1481/analysis/verdict_manifest.json; the cone is absent from tests/sparse_cones.txt on branch and on origin/main.
why_workflow_gap: The #671 rule requires every hard-read eval_results/issue_<M>/ fixture dir to be registered in tests/sparse_cones.txt so new_worktree.sh pre-adds it; this one never was, so every sparse worktree running the full Step 9c gate aborts at collection with no verdict.
proposed_change: Register eval_results/issue_1481 in tests/sparse_cones.txt so the Step 9c full-suite gate collects in a sparse worktree.
diff_sketch: |
  tests/sparse_cones.txt
  + eval_results/issue_1481
confidence: high
related_task: #2054
<!-- /workflow-fix-candidate -->
