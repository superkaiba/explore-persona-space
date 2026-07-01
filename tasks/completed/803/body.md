---
title: 'workflow-fix: clean_experiment_downloads.py hangs on pod (task_workflow.repo_root
  auto-routing on MooseFS)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cedaeca24709
created_at: '2026-07-01T08:08:42Z'
has_clean_result: false
origin_prompt: 'Auto-filed by /issue 778 orchestrator: clean_experiment_downloads.py
  --incremental --apply hung 15+ min on pod-778 because task_workflow.repo_root()
  auto-routes to a managed main worktree on non-main HEAD, and git reset --hard on
  MooseFS-backed /workspace never completes. See #778 epm:progress at 2026-07-01T08:07Z.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #778 (emitting agent: /issue orchestrator, autonomous). See #778's `epm:progress` at 2026-07-01T08:07Z for the full incident log.

## Goal

Prevent `clean_experiment_downloads.py --incremental --apply` from hanging when the driver invokes it POD-SIDE on a non-`main` HEAD (the standard state for a running experiment). The current failure mode: the script imports `task_workflow.repo_root()`, whose branch-guard auto-routes to a managed `main`-pinned worktree via `git worktree add --detach --force ... main` → `git reset --hard`, which stalls indefinitely on MooseFS-backed `/workspace`.

## Workflow gap

- **Bug observed:** On task #778 (RunPod `pod-778`, HEAD=`issue-778`), the between-phase call `bash scripts/issue778_dispatch.sh` step 2 → `uv run python scripts/clean_experiment_downloads.py 778 --incremental --apply || true` hung 15+ min at 0% CPU. Process tree: `clean_experiment_downloads.py` → child git via `task_workflow.repo_root()`'s auto-routing → `git worktree add --detach --force /workspace/explore-persona-space/.claude/worktrees/_task-main-pin main` → `git reset --hard --no-recurse-submodules` (state R for 15+ min on MooseFS). The driver's `|| true` guard is inert — the process never exits, so the driver never proceeds. Manual SIGKILL of the chain unblocked the driver into `phase=monitoring`.
- **Why it is a workflow gap:** The CLAUDE.md rule "Pod-side code NEVER shells out to `scripts/task.py` for ANY subcommand" was written to prevent exactly this class of hang. `clean_experiment_downloads.py` doesn't shell out to `task.py` — but it IMPORTS from `explore_persona_space.task_workflow`, which triggers the same `_resolve_repo_root_cached` → `_ensure_managed_main_worktree` code path on a non-`main` HEAD. The between-phase disk-hygiene call is documented as pod-safe in CLAUDE.md § Disk hygiene, but the current implementation is not pod-safe — it will hang any pod-side driver that calls it on an issue branch (every experiment).
- **Confidence (emitter):** high (root cause confirmed by direct process-tree inspection: PID 5770 in state R running `git reset --hard --no-recurse-submodules` for 15+ minutes; `.git/worktrees/_task-main-pin/index.lock` present the whole time; killing the chain immediately unblocked the driver).

## Proposed change (candidate diff sketch — refine in planning)

Preferred: **short-circuit `clean_experiment_downloads.py` when it detects it is running pod-side** — so the between-phase cleanup call is a fast no-op on any pod, matching what the real script does when there are no `hf_dl`/`g*_dl` caches to reap anyway. Detection signals: `/.dockerenv` AND `/workspace/logs/` both present (both are RunPod-pod-only conditions on this project's setup). No env-var contract to add, no driver edit needed.

```python
# scripts/clean_experiment_downloads.py — at top of main(), BEFORE the
# task_workflow imports fire:
def _running_pod_side() -> bool:
    return Path("/.dockerenv").exists() and Path("/workspace/logs").is_dir()

def main() -> int:
    if _running_pod_side():
        print("clean_experiment_downloads.py: pod-side no-op (workflow-fix #<N>)")
        return 0
    # ... existing implementation ...
```

Alternative (broader but safer): teach `task_workflow.repo_root()`'s branch-guard to fail-loud rather than auto-route when running under a pod-side detector — no pod should ever be reading task workflow state, and a fail-loud is preferable to a silent 15-min hang for every script that transitively imports it. This is the "belt and suspenders" fix — the immediate fix is the script-level short-circuit above (which the planner should ship this round); the broader `task_workflow.repo_root()` fix can be a follow-up if the planner scopes it.

Also add a test: `tests/test_clean_experiment_downloads_pod_side_short_circuit.py` — assert that with `/.dockerenv` + `/workspace/logs` both present (mocked), the script returns 0 without touching `task_workflow.repo_root()`.

## Scope / surfaces

- Primary target: `scripts/clean_experiment_downloads.py` (add pod-side detection + early-return).
- Optional secondary target: `src/explore_persona_space/task_workflow.py::_resolve_repo_root_cached` (add pod-side detection + fail-loud rather than auto-route). Planner decides whether to bundle.
- Test: `tests/test_clean_experiment_downloads_pod_side_short_circuit.py`.

Grep for the pattern before planning:
```
grep -rln "from explore_persona_space.task_workflow\|from explore_persona_space import task_workflow" scripts/
```
The 30+ hits are ALL VM-side (dispatch_issue.py, task.py, poll_pipeline.py, worktree_audit.py, etc.); the pod-side hit is only `clean_experiment_downloads.py` (per the between-phase contract in CLAUDE.md § Disk hygiene). No other pod-side script triggers this today, but the broader `task_workflow.repo_root()` fix would future-proof against a new one.

## Constraints / invariants

- Workflow-surface only — `scripts/clean_experiment_downloads.py` is on the workflow-fix scope list (workflow-helper script).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `CLAUDE.md` § Disk hygiene wording changes, it stays consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).
- The fix must not change VM-side behavior. VM-side users of `clean_experiment_downloads.py` (the /issue Step 8 post-verification call; the `vm_disk_guard.py` cron's fleet-wide reap) must continue to reap `hf_dl`/`g*_dl` caches as today.

## Provenance

- workflow_fix_target: scripts/clean_experiment_downloads.py
- fingerprint: cedaeca24709

<!-- workflow-fix-candidate v1 -->
target_file: scripts/clean_experiment_downloads.py
bug_observed: clean_experiment_downloads.py --incremental --apply hung 15+ min on pod-778 (issue-778 branch); root cause task_workflow.repo_root() auto-routes to a managed main worktree via git worktree add + git reset --hard --force which never completes on MooseFS.
why_workflow_gap: The pod-side between-phase disk-hygiene contract in CLAUDE.md § Disk hygiene is not honored by the current script — it transitively depends on the VM-only task_workflow.repo_root() auto-routing, which is a hard hang on MooseFS-backed pods.
proposed_change: clean_experiment_downloads.py short-circuit on pod (HEAD != main under /workspace) to avoid task_workflow.repo_root auto-routing hang on MooseFS
diff_sketch: |
  # scripts/clean_experiment_downloads.py — at top of main():
  + def _running_pod_side() -> bool:
  +     return Path("/.dockerenv").exists() and Path("/workspace/logs").is_dir()
  +
  def main() -> int:
  +     if _running_pod_side():
  +         print("clean_experiment_downloads.py: pod-side no-op")
  +         return 0
      # ... existing implementation ...
confidence: high
related_task: #778
<!-- /workflow-fix-candidate -->
