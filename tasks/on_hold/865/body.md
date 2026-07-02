---
title: 'workflow-fix: Step 9c selector diffs main checkout, blind to worktree branches'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d2551dac39f5
created_at: '2026-07-02T11:42:07Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from /issue 861 Step 9c: select_step9c_tests.py
  computes git diff in the main checkout (HEAD==main) so worktree-branch runs degenerate
  to the invariant-only fallback with no WARN'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #861 (emitting agent: orchestrator, /issue 861 Step 9c).

## Goal

Make `scripts/select_step9c_tests.py` compute its `git diff` against the INVOKING worktree's git context (or an explicit `--worktree` arg) so a Step 9c run for a not-yet-merged worktree branch selects the branch's touched-file tests instead of silently degenerating to the invariant-only fallback.

## Workflow gap

- **Bug observed:** invoked from the issue-861 worktree (cwd = the worktree, branch `issue-861`, 5 files touched), the selector printed ONLY the 32-file WORKFLOW_INVARIANT set with zero `untested touched file` WARNs — `compute_touched` saw an empty diff.
- **Why it is a workflow gap:** `_resolve_repo_root` uses the #506-safe `--git-common-dir` pattern (correct for `tests/` path arithmetic) but `compute_touched` then runs `git diff --name-only <base>...HEAD` with `cwd=repo_root` = the MAIN checkout, where HEAD==main → the diff is ALWAYS empty for a worktree-branch invocation, and the degenerate empty-diff fallback fires with NO warning. The Step 9c gate (SKILL.md) instructs running the selector for code-change branches PRE-merge — exactly the worktree context the cwd resolution blinds. Touched-file tests are silently skipped; the gate's touched-scope contract is hollow for every /issue worktree run.
- **Confidence (emitter):** high (mechanism read from source L147-171 + `--json` repro: `n_touched: 0` from repo root AND the worktree invocation returned the identical invariant-only set).

## Proposed change (candidate diff sketch — refine in planning)

```
- def _default_runner(argv): subprocess.run(argv, cwd=str(repo_root), ...)
+ # diff cwd = the INVOKING git context (worktree-aware), NOT the main checkout:
+ diff_root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=os.getcwd(), ...).stdout.strip())
+ def _default_runner(argv): subprocess.run(argv, cwd=str(diff_root), ...)
+ # tests/-glob arithmetic in select_tests should ALSO use diff_root (new test
+ # files exist only on the branch), while WORKFLOW_INVARIANT existence checks
+ # may stay on repo_root.
+ # AND: when base...HEAD is empty but `git -C diff_root rev-parse HEAD` != `git -C diff_root rev-parse <base>`,
+ # print a LOUD stderr WARN (empty diff with divergent HEAD is a resolution bug, not a clean tree).
```

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Sibling surfaces (grep hits, update if the contract text changes): `.claude/skills/issue/SKILL.md` (Step 9c invocation prose — may need "run from the worktree" / `--worktree` note), `tests/test_select_step9c_tests.py` (add a worktree-shaped regression: injectable runner already exists via `_runner`).
- Grep the workflow surface for the pattern before editing (`grep -rln 'select_step9c' .claude/ CLAUDE.md scripts/ tests/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: d2551dac39f5

<!-- workflow-fix-candidate v1 -->
target_file: scripts/select_step9c_tests.py
bug_observed: selector resolves repo_root via --git-common-dir and diffs main...HEAD in the MAIN checkout where HEAD==main, returning the degenerate invariant-only fallback with no WARN for worktree-branch invocations
why_workflow_gap: the Step 9c gate runs the selector for code-change branches PRE-merge from the issue worktree — exactly the context the cwd resolution blinds; touched-file tests are silently skipped
proposed_change: run the Step 9c diff in the invoking worktree git context so worktree-branch runs are not empty
diff_sketch: |
  + diff_root = git rev-parse --show-toplevel of the INVOKING cwd (worktree-aware)
  + compute_touched runs git diff with cwd=diff_root (not the main checkout)
  + select_tests' tests/ globbing uses diff_root; WARN loud when base...HEAD is empty while HEAD != base
confidence: high
related_task: #861
<!-- /workflow-fix-candidate -->
