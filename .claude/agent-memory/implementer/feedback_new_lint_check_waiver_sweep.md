---
name: New default-bundled lint checks need a same-commit legacy-waiver sweep
description: Adding a check to workflow_lint.py's no-flags default run requires prototyping against the live tree first, waiving every legacy offender in the same commit, and pinning a repo-tree-is-clean test
type: feedback
---

When adding a new check to `scripts/workflow_lint.py` that bundles into the no-flags default run, prototype the detection heuristic against the LIVE `scripts/` tree before finalizing it, then waive every legacy offender in the SAME commit, and add both a `test_check_<name>_repo_tree_is_clean` and a `test_workflow_lint_check_<name>_cli_exits_zero` test.

**Why:** The no-flags default runs in pre-commit, so a single unwaived legacy file breaks every commit repo-wide (the #612 stale-tables incident class). On the CVD-pin check (#578 follow-up, 2026-06-12), the candidate's proposed `scripts/*dispatch*.sh` scope would have missed half the real offenders (`launch_issue_444_*.sh` carried the violating shape under a different filename); scanning all `scripts/*.sh` with a tight per-line heuristic (backgrounded + `--gpu-id`/`+gpu_id=` + no `CUDA_VISIBLE_DEVICES=`) found 6 legacy sites, all waived with `# CVD_PIN_EXEMPT: <reason>` (≥10-char reason, same-or-preceding-non-blank-line placement — the established WANDB_INTENTIONALLY_DISABLED convention).

**How to apply:** (1) Write the heuristic as a quick standalone scan first and run it on the tree; every hit is either a true legacy offender (waive with a reason naming it pre-fix/completed) or a false positive (tighten the heuristic). (2) Don't scope by filename pattern — scope by line shape; filenames drift. (3) Waiver comments in completed-task experiment dispatchers are in-scope (behavior-preserving comments needed BY the lint), but never change those scripts' behavior. (4) `main()` in workflow_lint.py sits at the ruff C901 cap; a new flag branch pushes it over — the repo convention is an annotated `# noqa: C901 -- <reason>` on flat CLI dispatch ladders, not a refactor.
