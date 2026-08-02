---
title: 'workflow-fix: add eval_results/issue_1481 to tests/sparse_cones.txt (issue-1947
  tests collect in sparse worktrees)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:31e37654e526
created_at: '2026-08-02T17:55:48Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from #2021 unit-A implementer: test_issue1947_resume_matrix.py
  collection-fails in sparse worktrees lacking the eval_results/issue_1481 cone; sparse_cones.txt
  registry line is the documented fix (#671 class).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #2021 (emitting agent: implementer, unit A).

## Goal

Add `eval_results/issue_1481` to `tests/sparse_cones.txt` so `tests/test_issue1947_resume_matrix.py` (and the other issue-1947 tests importing `scripts/issue1947_cells.py`) collect/run in fresh sparse worktrees without a manual `git sparse-checkout add`.

## Workflow gap

- **Bug observed:** `tests/test_issue1947_resume_matrix.py` fails at collection in any sparse worktree lacking the `eval_results/issue_1481` cone; the #2021 unit-A implementer had to run `git -C "$WT" sparse-checkout add eval_results/issue_1481` by hand (16/16 passed after).
- **Why it is a workflow gap:** CLAUDE.md § sparse-worktree test gate mandates that any test hard-reading `repo_root()/eval_results/issue_<M>/...` gets its dir added to the `tests/sparse_cones.txt` registry consumed by `scripts/new_worktree.sh`; a missing line makes every fresh sparse worktree's Step 9c gate red on an untouched test (the #671 class).
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'issue_1481' tests/sparse_cones.txt` → 0 hits (absence claim — the registry line is missing; 2026-08-02). Call-hop trace (clause g): symptom site = `tests/test_issue1947_resume_matrix.py` (imports `issue1947_datagen` → `issue1947_cells`); construction site = `scripts/issue1947_cells.py:43` (`VERDICT_MANIFEST_PATH = REPO_ROOT / "eval_results/issue_1481/analysis/verdict_manifest.json"`, read at L113 with a raise naming the manual sparse-checkout remedy at L111). Landed-fix history: `git log --oneline --since='7 days ago' -- tests/sparse_cones.txt` → 1 commit (ded60e3fa8, task #1776 — adds a DIFFERENT cone, the issue1776 phase3 manifest; not this fix).

## Proposed change (candidate diff sketch — refine in planning)

```
+ eval_results/issue_1481
```
(one registry line in tests/sparse_cones.txt; planner should confirm whether sibling issue-1947 test files or `eval_results/issue_1947` need lines too, per the registry's own conventions)

## Scope / surfaces

- Primary target: `tests/sparse_cones.txt`
- Grep the workflow surface for the pattern before editing (`grep -rln 'issue_1481' tests/sparse_cones.txt scripts/new_worktree.sh`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: tests/sparse_cones.txt
- fingerprint: 31e37654e526

Surfaced prose (verbatim, from the #2021 unit-A implementer manifest): "tests/test_issue1947_resume_matrix.py fails at COLLECTION in any sparse worktree lacking the eval_results/issue_1481 cone, and eval_results/issue_1481 is absent from tests/sparse_cones.txt — a pre-existing gap unrelated to this diff (I ran git -C "$WT" sparse-checkout add eval_results/issue_1481 locally; 16/16 passed). Unit B / the gate may hit this; the registry line is the documented fix."
