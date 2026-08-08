---
title: 'workflow-fix: scratch-eligibility for the workflow-lint scan node in step9c
  compare'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a85a7ae2235e
created_at: '2026-08-02T14:46:18Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #1932 Step 9c: compare indeterminate
  on dirty root for tests/test_workflow_lint.py::test_workflow_lint_default_exits_zero
  (scratch-ineligible node); scratch run at HEAD PASSed — make it registry-eligible'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1932 (emitting agent: orchestrator, /issue 1932 Step 9c).

## Goal

Make the `tests/test_workflow_lint.py` no-flags scan node scratch-resolvable in `scripts/step9c_baseline.py` (add to `FILE_ANCHORED_SCAN_TESTS`, or an equivalent scratch-eligibility registry entry with source-verified anchoring notes) so `compare --run-pristine` can pristine-resolve a red on it from the default scratch-worktree oracle instead of going indeterminate on a dirty shared root.

## Workflow gap

- **Bug observed:** `step9c_baseline.py compare` returns indeterminate (exit 2, MF-4c dirty-root refusal) for a red on `tests/test_workflow_lint.py::test_workflow_lint_default_exits_zero` whenever the shared root carries visible code dirt, because the node is scratch-ineligible (absent from `FILE_ANCHORED_SCAN_TESTS`) — despite the test resolving its scan root from its own `__file__` (`_REPO_ROOT = Path(__file__).resolve().parents[1]`, tests/test_workflow_lint.py:30; subprocess `cwd=_REPO_ROOT`, :110), i.e. scratch-local by the same criterion the registry's existing members satisfy.
- **Why it is a workflow gap:** the shared repo root is dirty essentially always (concurrent sessions' untracked drafts + in-flight agent-memory writes), and `test_workflow_lint_default_exits_zero` is the single most red-prone invariant node (near-cap size-ratchet files: a concurrent uncommitted `MEMORY.md` write pushed it over on #1932's gate). Every such gate run goes indeterminate and forces a manual orchestrator classification — #1932 (2026-08-02) hand-built the exact scratch run (`1 passed in 381.58s` in a detached sparse scratch worktree at HEAD) that the compare would have run itself were the node registry-eligible.
- **Confidence (emitter):** medium — the eligibility bar requires reading the WHOLE test file's live-tree anchoring (all nodes, conftest channels), not just the one node; the planner makes that deliberate call with the source open (the registry comment mandates source-verified anchoring notes + the `test_file_anchored_scan_tests_live_tree_pin` drift pin must be updated).
- verified-at-filing: `grep -n 'FILE_ANCHORED_SCAN_TESTS' scripts/step9c_baseline.py` → 5 hits (definition at :206; members: test_shared_vm_thread_caps.py, test_subprocess_env_explicit.py, test_select_step9c_tests.py per the pinned literal); `grep -c 'test_workflow_lint' scripts/step9c_baseline.py` → 2 (neither a registry membership — absence confirmed) (2026-08-02, this session). Empirical demonstration: /tmp/issue-1932-scratch-node.log (sparse scratch at HEAD, node PASSes rc=0).

## Proposed change (candidate diff sketch — refine in planning)

```
 FILE_ANCHORED_SCAN_TESTS: frozenset[str] = frozenset(
     {
         ...
+        # _REPO_ROOT = Path(__file__).resolve().parents[1] (:30); _run() subprocess-runs
+        # scripts/workflow_lint.py from the test file's own tree with cwd=_REPO_ROOT (:110);
+        # <planner: verify remaining nodes/conftest channels are scratch-local before adding>
+        "tests/test_workflow_lint.py",
     }
 )
```

Plus the matching update to `tests/test_step9c_baseline.py::test_file_anchored_scan_tests_live_tree_pin` and, if the whole-file bar fails, an alternative node-scoped eligibility mechanism.

## Scope / surfaces

- Primary target: `scripts/step9c_baseline.py`
- Secondary: `tests/test_step9c_baseline.py` (drift pin).
- Grep the workflow surface for the pattern before editing (`grep -rln 'FILE_ANCHORED_SCAN_TESTS' .claude/ CLAUDE.md scripts/ tests/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- FAIL-CLOSED doctrine preserved: a scan test absent from the registry keeps the refusal; eligibility is added only with source-verified anchoring (the registry's own curation rule).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/step9c_baseline.py
- fingerprint: a85a7ae2235e

<!-- workflow-fix-candidate v1 -->
target_file: scripts/step9c_baseline.py
bug_observed: step9c compare returns indeterminate (exit 2) for a red on tests/test_workflow_lint.py::test_workflow_lint_default_exits_zero whenever the shared root carries visible code dirt, because the node is scratch-ineligible despite resolving its scan root from its own __file__
why_workflow_gap: the always-dirty shared root makes every such gate run indeterminate, forcing manual orchestrator classification of the most red-prone invariant node (#1932, 2026-08-02: hand-built scratch run PASSed at HEAD)
proposed_change: make the tests/test_workflow_lint.py no-flags scan node scratch-resolvable in step9c_baseline.py (FILE_ANCHORED_SCAN_TESTS or an equivalent scratch-eligibility registry) so compare can pristine-resolve it on a dirty shared root
diff_sketch: |
  + "tests/test_workflow_lint.py",   # in FILE_ANCHORED_SCAN_TESTS, with source-verified
  +                                  # anchoring note (:30 __file__ root; :110 cwd=_REPO_ROOT)
  (+ update tests/test_step9c_baseline.py::test_file_anchored_scan_tests_live_tree_pin)
confidence: medium
related_task: #1932
<!-- /workflow-fix-candidate -->
