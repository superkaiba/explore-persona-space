---
title: 'workflow-fix: add eval_results/issue_1481 cone to tests/sparse_cones.txt'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d4bc5e883fd2
created_at: '2026-08-02T22:09:10Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised by the #1942 implementer (2026-08-02):
  tests/sparse_cones.txt lacks an eval_results/issue_1481 cone line while scripts/issue1947_cells.py:43
  hard-pins eval_results/issue_1481/analysis/verdict_manifest.json (imported by tests/test_issue1947_battery.py
  + tests/test_issue1947_marker_topup.py); fresh sparse worktrees hit FileNotFoundError
  at the full-suite gate.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1942 (emitting agent: implementer).

## Goal

Add an `eval_results/issue_1481` cone line to `tests/sparse_cones.txt` with the
standard provenance comment naming the `issue1947_cells.py` verdict-manifest readers.

## Workflow gap

- **Bug observed:** `scripts/issue1947_cells.py:43` hard-pins
  `REPO_ROOT/eval_results/issue_1481/analysis/verdict_manifest.json` (lazy fail-loud
  read via `verdict_lr()`; its own error message at :111 instructs
  `git sparse-checkout add eval_results/issue_1481`), and it is imported by
  `tests/test_issue1947_battery.py:23` and `tests/test_issue1947_marker_topup.py:36` —
  but `tests/sparse_cones.txt` has no `eval_results/issue_1481` cone line, so a fresh
  sparse worktree hits `FileNotFoundError` at the full-suite gate. Hit live in the
  issue-1942 worktree (2026-08-02): the implementer had to run a manual
  `git sparse-checkout add eval_results/issue_1481` (visible at line 17 of that
  worktree's `sparse-checkout list`).
- **Why it is a workflow gap:** the sparse-worktree test-gate contract (CLAUDE.md
  § sparse-worktree test gate, #671) requires every test that hard-reads another
  issue's committed `eval_results/` dir — directly or via an imported script — to have
  a cone line in the registry `new_worktree.sh` pre-adds; the line was never added when
  the #1947 verdict-LR wiring landed.
- **Confidence (emitter):** high (compose-time re-verified; the emitter's original
  block mis-named the reader as `tests/test_issue1947_worker_dispatch.py` — corrected
  per the call-hop tracing clause to `scripts/issue1947_cells.py` + its two test
  importers; `target_file` unchanged, fingerprint recomputed on the corrected claim)
- verified-at-filing: `grep -n "1481|1947" tests/sparse_cones.txt` → 1 hit, a comment
  citing `eval_results/issue_1434` for `test_issue1481_analysis.py` — NO
  `eval_results/issue_1481` cone line (absence confirmed in-target); relocation grep
  `grep -rn "issue_1481" scripts/issue1947_*.py` → `scripts/issue1947_cells.py:43`
  (`VERDICT_MANIFEST_PATH` hard-pin) + `:111` (sparse-checkout error text); importer
  grep → `tests/test_issue1947_battery.py:23`, `tests/test_issue1947_marker_topup.py:36`
  (2026-08-02)

## Proposed change (candidate diff sketch — refine in planning)

```
+ #   eval_results/issue_1481 — issue1947_cells.py VERDICT_MANIFEST_PATH (verdict_manifest.json,
+ #     the per-cell LR source) via test_issue1947_battery.py + test_issue1947_marker_topup.py
+ eval_results/issue_1481
```

## Scope / surfaces

- Primary target: `tests/sparse_cones.txt`
- Grep the workflow surface for the pattern before editing
  (`grep -rn 'issue_1481' tests/sparse_cones.txt scripts/new_worktree.sh`) and update
  every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: tests/sparse_cones.txt
- fingerprint: d4bc5e883fd2

<!-- workflow-fix-candidate v1 -->
target_file: tests/sparse_cones.txt
bug_observed: scripts/issue1947_cells.py:43 hard-pins REPO_ROOT/eval_results/issue_1481/analysis/verdict_manifest.json (lazy fail-loud read via verdict_lr), imported by tests/test_issue1947_battery.py and tests/test_issue1947_marker_topup.py, but tests/sparse_cones.txt has no eval_results/issue_1481 cone line, so a fresh sparse worktree hits FileNotFoundError at the full-suite gate (hit live in the issue-1942 worktree, 2026-08-02; manual sparse-checkout add was required).
why_workflow_gap: the sparse-worktree test-gate contract (CLAUDE.md § sparse-worktree test gate, #671) requires every test that hard-reads another issue's committed eval_results/ dir to have a cone line in the registry new_worktree.sh pre-adds; the line was never added when the #1947 verdict-LR wiring landed.
proposed_change: add an eval_results/issue_1481 cone line to tests/sparse_cones.txt with the standard provenance comment naming the issue1947_cells.py verdict-manifest readers.
diff_sketch: |
  + #   eval_results/issue_1481 — issue1947_cells.py VERDICT_MANIFEST_PATH (verdict_manifest.json,
  + #     the per-cell LR source) via test_issue1947_battery.py + test_issue1947_marker_topup.py
  + eval_results/issue_1481
confidence: high
related_task: #1942
<!-- /workflow-fix-candidate -->

(Emitter's original block named `tests/test_issue1947_worker_dispatch.py` as the reader;
compose-time verification found that file carries no such read — the block above is the
corrected form, with the original preserved in task #1942's `epm:workflow-fix-candidate`
marker for the audit trail.)
