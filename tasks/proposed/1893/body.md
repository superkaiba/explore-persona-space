---
title: 'workflow-fix: fleet-wide pod code-sync pulls origin main over live issue-branch
  workloads (mid-run ENOENT race)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:85a20d31a52e
created_at: '2026-07-30T19:49:43Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed on /issue 1776 swap round 2026-07-30: pod-1776
  reflog shows pull --rebase=merges origin main mid-p4_capture; issue-branch-only
  driver script vanished for the rebase window; chain died rc=2 (epm:failure v11);
  respawn 1/3 consumed'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1776 (emitting agent: orchestrator, /issue 1776 same-issue follow-up round `operator_swap_success`).

## Goal

Make the fleet-wide pod code-sync live-run-safe: skip (or branch-pin) any pod whose registered workload is live, and sync a pod's OWN checked-out branch instead of pulling `origin main` over an issue-branch clone — so a sync can never transiently check main-side content out from under a running issue-branch workload.

## Workflow gap

- **Bug observed:** 2026-07-30 ~19:40Z, task #1776 swap round on pod-1776 (RunPod 8×H100): the pod clone's reflog shows `pull --rebase=merges origin main` (start: checkout `66cc523f17` main-side content → fast-forward → finish: back to `refs/heads/issue-1776` @ `c776254ce7`) executed DURING the live smoke-leg p4_capture phase. In the rebase window the working tree held main-side content, where the issue-branch-only `scripts/issue1776_swap.py` does not exist; the just-dispatched p4 shards died `[Errno 2] No such file or directory` and the whole chain FAILED rc=2 (epm:failure v11 on #1776; experimenter respawn 1/3 consumed).
- **Why it is a workflow gap:** `scripts/pod.py sync code` is documented as "Git pull on all pods" (pod.py usage text, line ~27) and the sync path targets `origin main` on EVERY pods.conf pod, regardless of (a) whether the pod's clone is checked out on an issue branch (issue-branch-only files vanish for the duration of the pull, and a `--ff-only`/rebase against main can permanently strand the checkout on main content when histories diverge) and (b) whether a registered workload is LIVE on the pod (the pid file `/workspace/logs/issue-<N>.pid` + the `epm:run-launched` marker both exist and are readable at sync time). Any concurrent session's fleet-wide sync therefore races every live issue-branch run on the fleet.
- **Confidence (emitter):** high
- verified-at-filing: pod-1776 reflog captured live (`git reflog -5` → the 5-entry pull sequence quoted above, 2026-07-30); `grep -n "pull" scripts/pod.py` → line 27 usage text `python scripts/pod.py sync code  # Git pull on all pods` (the sync-all-pods contract); `grep -n "git pull\|pull --rebase\|fetch origin" <worktree>/scripts/issue1776_swap_dispatch.sh <worktree>/scripts/issue1776_swap.py` → 0 hits (in-chain actor ruled out). Landed-fix history check: `git log --oneline --since='7 days ago' -- scripts/pod.py scripts/sync_pods.sh` shows no commit adding a live-run guard to the code-sync path.
- unverified hypothesis — verify at plan time: which exact entrypoint issued THIS pull (a concurrent session's `pod.py sync code`, `sync_pods.sh`, or another caller — the reflog records the pull form `--rebase=merges`, matching the repo-config pin bootstrap copies, but not the caller identity); the fix should guard the SYNC IMPLEMENTATION regardless of caller.

## Proposed change (candidate diff sketch — refine in planning)

```
# scripts/pod.py (sync code path) / scripts/sync_pods.sh
+ per pod, BEFORE any git operation:
+   1. live-workload probe: pod-side pid file (/workspace/logs/issue-*.pid)
+      with a live, identity-matched pid -> SKIP the pod with a loud
+      "live workload, sync skipped" line (never pull under a running job);
+   2. branch-aware target: sync `origin <current-branch>` (the pod clone's
+      checked-out branch), NEVER an unconditional `origin main`, so an
+      issue-branch clone can never transiently lose issue-branch-only files.
+ pin test: a pod fixture with a live pid file is skipped; an issue-branch
+ clone syncs its own branch.
```

## Scope / surfaces

- Primary target: `scripts/pod.py` (the `sync code` implementation)
- Sibling: `scripts/sync_pods.sh` (if it carries its own pull loop — verify at plan time)
- Grep the workflow surface for the pattern before editing (`grep -rn 'pull' scripts/pod.py scripts/sync_pods.sh scripts/bootstrap_pod.sh`) and guard every fleet-wide pull site; list them in the plan.

## Constraints / invariants

- Workflow-surface only — pod.py / sync_pods.sh are in-scope workflow-helper scripts per `.claude/rules/workflow-fix-on-bug.md`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/pod.py
- fingerprint: 85a20d31a52e

<!-- workflow-fix-candidate v1 -->
target_file: scripts/pod.py
bug_observed: a concurrent session fleet-wide code-sync ran git pull origin main on pod-1776 mid-workload, transiently checking out main-side content; the running smoke-leg p4 shards hit ENOENT on the issue-branch-only scripts/issue1776_swap.py and the chain died rc=2
why_workflow_gap: pod.py sync code pulls origin main on ALL pods.conf pods unconditionally — no live-workload skip and no branch-awareness — so any session's fleet-wide sync races every live issue-branch run on the fleet
proposed_change: make fleet-wide pod code-sync live-run-safe: skip (or branch-pin) any pod whose pods_ephemeral/pods.conf entry has a live workload pid, syncing the pod OWN checked-out branch instead of pulling origin main over an issue-branch clone
diff_sketch: |
  + live-workload probe (pod-side pid file, identity-matched) -> skip pod, loud line
  + sync `origin <current-branch>` instead of unconditional `origin main`
  + pin test: live-pid pod skipped; issue-branch clone syncs its own branch
confidence: high
related_task: #1776
<!-- /workflow-fix-candidate -->
