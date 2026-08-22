---
title: 'Remediate 19 scripts/ drivers with repo_root()-derived sys.path inserts, then
  widen the #2181 check to scripts/'
kind: infra
tags: []
created_at: '2026-08-07T22:28:12Z'
has_clean_result: false
origin_prompt: 'Deferred from #2181 plan v3 §8: the 19 scripts/ drivers carry the
  same latent repo_root() sys.path trap; fix them, then widen the new lint check to
  scripts/.'
workflow: v1
---
# Remediate 19 `scripts/` drivers carrying `PROJECT_ROOT = repo_root()` + `sys.path.insert`, then widen the #2181 check to `scripts/`

## Goal

Fix the 19 `scripts/` drivers that derive a `sys.path` entry from `task_workflow.repo_root()` via a one-hop module constant, then extend `workflow_lint.py`'s `check_no_repo_root_syspath_in_tests` to cover `scripts/` (renaming it accordingly).

## Why

`repo_root()` branch-guards to the MAIN checkout. A driver that does `PROJECT_ROOT = repo_root()` + `sys.path.insert(0, str(PROJECT_ROOT / "scripts"))` therefore imports **main's** copy of its sibling modules when run from a worktree — so a branch-local fix to a sibling module is silently not exercised, and a foreign checkout's `scripts/` is leaked onto `sys.path` for the process.

This is the same latent trap #2164 hit on the `tests/` side, where it cost ~10 probe commands plus a ~20-minute gate re-run to diagnose (the leaked path defeated the #1296 negative control in `tests/test_backend_poll.py`). The `tests/` half is now permanently blocked by the #2181 check; this task closes the `scripts/` half.

## The 19 offenders

Enumerated by two independent AST sweeps at #2181 plan time (the plan critic's and the code reviewer's, concordant), each re-implementing the predicate and running it over the live tree — `scripts/`: **19 flagged**, `tests/`: 0, `src/`: 0.

17 × `scripts/issue1482_*.py`: `best_pc_alignment`, `continuous_predictors`, `dense_predictor_reads`, `discrete_predictors`, `footprint_moments`, `jspace_r2`, `label_reads_densesae`, `lens_reads`, `pc_dashboard`, `pc_deviation_dashboard`, `predictor_battery`, `predictor_battery_fullwidth`, `rb7_reads`, `rb_as_sae_feature`, `result2_assembly`, `shapley_blocks`, `tail_depth_sweep`

2 × `scripts/issue1738_*.py`: `deviant_band`, `pc_sae_alignment_vs_r2`

## The corrective form already exists in-repo

`scripts/issue1482_densesae_fullwidth.py:51-58` carries the fixed shape with its rationale comment:

```python
# Derived from __file__, NOT task_workflow.repo_root(): that resolver REFUSES a ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
```

Note the `parents` depth differs between the `scripts/` shape (`.parent.parent`) and the `tests/` shape (`.resolve().parents[1]`) — do not copy the tests/ form blindly.

## Why this was NOT absorbed into #2181

Landing the check with `scripts/` in scope would have put 19 immediate offenders into the **no-flags default bundle**, which gates the Step 9c compare and the Step 10d pre-push merge for every task — an instant fleet-wide red. The remediation has to land before the scope widens. #2181 pinned the `tests/`-only scope with a green test case (case 13) precisely so this widening is a deliberate, reviewed step rather than an accident.

## Acceptance criteria

1. All 19 drivers use the `__file__`-derived form; each still runs (import-level smoke at minimum).
2. `check_no_repo_root_syspath_in_tests` widened to `scripts/` (rename to drop `_in_tests`), with the existing 15 test cases preserved and a `scripts/`-scope firing case added — the current case 13 pins `scripts/` as OUT of scope and must be inverted deliberately, not deleted silently.
3. Scoped check green on the live tree; the no-flags bundle shows **no new failures vs the plan-time baseline** (the bundle is currently rc=0 / PASS / 0 FAIL / ~19-21 WARN, the WARN count varying by two transient `root-guard hook` lines that are environment- not tree-dependent).
4. `src/` is clean today (0 hits) — widening to `src/` as well is optional and should be justified separately if taken.

## Also worth folding in (surfaced during #2181, unfixed)

- **A `select_step9c_tests.py` argv dry-run in `verify_plan.py`.** `verify_plan.py` c46 already dry-parses plan-embedded `dispatch_issue.py` commands; #2181's plan §6 shipped a `--map-files` invocation that ERRORs as written (the flag takes a newline-delimited path-LIST *file*, not positional source paths). A sibling dry-parse would have caught it at plan time.
- **The pre-existing sibling `ValueError` gap at `workflow_lint.py:11846`**, noted by #2181's code review. Investigated there and found NOT reachable on CPython 3.11.15 for the null-byte case (`ast.parse` raises `SyntaxError`, already caught) — carried here only so the note is not lost, not as a known defect.

## Provenance

Deferred from #2181 (plan v3 §8) as a named follow-up. Filed WITHOUT auto-dispatch: this is a planned remediation, not a ripe bug fix, and no one has asked for it to run yet — leave at `proposed` for user/PM triage.
