---
title: 'workflow-fix: branch-sync already-at-tip short-circuit + recalibrated mutation
  caps (runpod.py)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0ca5c6d62d63
created_at: '2026-08-01T10:31:41Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #1895 followup round: both branch-sync
  attempts rc=124 at the 20s checkout/reset caps on a slow-but-healthy MooseFS mount
  while the tree was already at the fetched target SHA (see body candidate block)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1895 (emitting agent: orchestrator, /issue same-issue follow-up round pure-residual-path-b).

## Goal

Short-circuit `_render_branch_sync_script` to SYNC-OK when HEAD already equals the fetched tip (ref reads only), and raise the 20 s checkout/reset caps for the genuinely-divergent path.

## Workflow gap

- **Bug observed:** both branch-sync attempts died rc=124 at the 20 s checkout/reset caps on a slow-but-healthy MooseFS mount while the pod tree was already at the fetched target SHA, failing the workload start and leaving the pod billing.
- **Why it is a workflow gap:** the #1858 sync script always runs the FUSE-heavy `git checkout -f -B` + `git reset --hard` mutation ops even when they are content no-ops — and the MODAL case is exactly that: a fresh failover/bootstrap provision whose shallow clone is already at the branch tip. A pure ref-read short-circuit (`git rev-parse HEAD` vs `git rev-parse FETCH_HEAD` after the fetch) would have skipped the mutations entirely. Separately, the 20 s per-op caps are calibrated below observed healthy-mount latency (`git status` measured 59.5 s on pod-1895 while the venv import probe passed in 3.3 s — slow mount, NOT the read-wedge), so the caps convert a slow-but-recoverable pod into a hard workload-start failure with a billing diagnosis pod.
- **Confidence (emitter):** high
- verified-at-filing: `grep -rn 'timeout -k 10 20 git' src/explore_persona_space/backends/` → 2 hits in 1 file (runpod.py, the checkout + reset lines in `_render_branch_sync_script`; the sibling .pyc hit is a compiled artifact) (2026-08-01). Landed-fix history check: `git log --oneline --since='7 days ago' -- src/explore_persona_space/backends/runpod.py` → 9e3df877a6 (task #1858, the kill-and-reap + retry hardening — the change that INTRODUCED these caps; its retry cannot recover this class because the retry re-runs the same under-capped mutations), a5c9a09427 (#1698, launch-path contract), d44a52d2fc (#1669, env pins) — none implements the already-at-tip short-circuit or recalibrates the caps.

## Incident evidence (task #1895, 2026-08-01)

- GCP FLEX_START queue-vanish (#1116) failed over to RunPod pod-1895 (kdk6thqclezjk2). Bootstrap 11/11 + preflight PASS (with a `git status failed: timeout` preflight WARNING — the early signal of the slow mount).
- `_execute_workload_on_pod` branch sync: attempt 1 rc=124 (stderr shows the fetch reached FETCH_HEAD — the failure was the 20 s-capped checkout or reset); kill-and-reap REAP-OK killed=0 survivors=0 (nothing to reap — the remote `timeout` had already killed the op, exactly as designed); attempt 2 rc=124 identically. `RunPodWorkloadStartError`, pod left RUNNING for diagnosis, `terminal_runpod_workload_start_failed` poll state.
- Manual diagnosis: HEAD == origin/issue-1895 == ea2b645d99 (the bootstrap's own shallow clone had already checked out the target); venv import 3.3 s (healthy); `git status` 59.5 s (slow FUSE stat storm over the full working tree).
- Manual recovery (session 6dbf040f): sync skipped after a ref-read verification, launcher rendered via `_render_launch_script` verbatim, WRAPPER-STARTED / LAUNCH-OK pid=2166, workload ran normally — demonstrating the sync mutations were unnecessary in this (modal) case.
- unverified hypothesis — verify at plan time: the slow-mount latency class (tree-wide stat ~60 s, single-file reads fast) may be transient per-pod MooseFS load rather than a stable pod property; the short-circuit fix is valid either way, but cap recalibration sizing should not assume 59.5 s is an upper bound.

## Proposed change (candidate diff sketch — refine in planning)

diff_sketch:
```
 # _render_branch_sync_script, after the fetch line:
+ 'FETCH_SHA=$(git rev-parse FETCH_HEAD)',
+ 'HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || echo none)',
+ 'CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)',
+ 'if [ "$HEAD_SHA" = "$FETCH_SHA" ] && [ "$CUR_BRANCH" = "{branch}" ]; then',
+ '  echo "SYNC-OK $HEAD_SHA (already-at-tip short-circuit)"',
+ '  exit 0',
+ 'fi',
- f'timeout -k 10 20 git checkout -q -f -B "{branch}" FETCH_HEAD',
- "timeout -k 10 20 git reset --hard -q FETCH_HEAD",
+ f'timeout -k 10 90 git checkout -q -f -B "{branch}" FETCH_HEAD',
+ "timeout -k 10 90 git reset --hard -q FETCH_HEAD",
```
(Exact cap value + whether the short-circuit also needs a dirty-tree guard — e.g. skip the short-circuit when `git status --porcelain` is nonempty, WITHOUT paying the tree-wide stat cost on the happy path — are the planner's calls. The `SYNC-OK <sha>` stdout contract consumed by `_attempt_sync`'s regex must be preserved.)

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/runpod.py`
- Grep the workflow surface for the pattern before editing (`grep -rn 'timeout -k 10' src/explore_persona_space/backends/ scripts/`) and update every hit that shares the under-capped-mutation shape; list them in the plan. Check the tests pinning the rendered script (`tests/test_backend_*.py` / the #1858 regression tests) — the short-circuit changes the rendered script's line set.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Preserve the `SYNC-OK <sha>` stdout contract (`_attempt_sync` regex) and the pod-side HEAD==FETCH_HEAD verification semantics for the divergent path.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/runpod.py
- fingerprint: 0ca5c6d62d63

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/runpod.py
bug_observed: both branch-sync attempts died rc=124 at the 20 s checkout/reset caps on a slow-but-healthy MooseFS mount while the pod tree was already at the fetched target SHA, failing the workload start and leaving the pod billing
why_workflow_gap: the sync script has no already-at-tip short-circuit (a pure ref-read would skip the FUSE-heavy mutations, which are content no-ops in the modal fresh-provision case) and the 20 s per-op caps are calibrated below observed healthy-mount latency (git status 59.5 s on pod-1895)
proposed_change: short-circuit _render_branch_sync_script to SYNC-OK when HEAD already equals the fetched tip (ref reads only), and raise the 20 s checkout/reset caps for the divergent path
diff_sketch: |
  + after fetch: HEAD_SHA/FETCH_SHA rev-parse compare -> echo "SYNC-OK $HEAD_SHA" && exit 0 when equal (+ branch-name check)
  - timeout -k 10 20 git checkout -q -f -B "{branch}" FETCH_HEAD
  + timeout -k 10 90 git checkout -q -f -B "{branch}" FETCH_HEAD
  - timeout -k 10 20 git reset --hard -q FETCH_HEAD
  + timeout -k 10 90 git reset --hard -q FETCH_HEAD
confidence: high
related_task: #1895
<!-- /workflow-fix-candidate -->
