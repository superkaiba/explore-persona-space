---
title: 'workflow-fix: flock + additive-merge pod.py config --sync (SSH-config drop
  race)'
kind: infra
tags:
- wf-fix
created_at: '2026-07-02T02:13:54Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed on #813: concurrent config --sync dropped pod-813
  Host entry twice; see candidate block in body'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #813 (emitting agent: orchestrator).

## Goal

Serialize `pod.py config --sync`'s rewrite of `~/.ssh/config` + the MCP env block against concurrent writers, and make it additive from the canonical repo-root `pods.conf` so a sync can never drop another session's live pod entry.

## Workflow gap

- **Bug observed:** Twice within ~1.5h (2026-07-02, task #813), a concurrent session's `pod.py config --sync` dropped the `Host pod-813` entry from `~/.ssh/config` (and earlier the SSH MCP env reverted to a months-stale server list) while `scripts/pods.conf` still contained pod-813 — producing false `dead` poll verdicts on a healthy run and requiring manual re-sync mid-monitoring.
- **Why it is a workflow gap:** the sync's read-modify-write of the managed SSH-config block is not flock-serialized (unlike task.py's mutations), and a session syncing from a stale view (e.g. a worktree copy of pods.conf, or an in-memory list read before another session's provision) rewrites the managed block without entries added concurrently.
- **Confidence (emitter):** medium (mechanism inferred from two occurrences + the stale MCP-server-list symptom; the exact racing writer not identified).

## Proposed change (candidate diff sketch — refine in planning)

+ In pod_config.py sync path:
+   1. hold flock on ~/.task-workflow/pod-config.lock around the whole read-merge-write;
+   2. resolve pods.conf ONLY at the repo-root canonical path (never cwd/worktree-relative);
+   3. re-READ pods.conf after acquiring the lock (never sync from a pre-lock snapshot);
+   4. make the managed-block rewrite a MERGE: preserve any Host entry present in the
+      freshly-read pods.conf; only remove entries absent from it.

## Scope / surfaces

- Primary target: `scripts/pod_config.py`, `scripts/pod.py` (config --sync path)
- Grep the workflow surface for the managed-block rewrite before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/pod_config.py, scripts/pod.py
- fingerprint: (computed by wrapper)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/pod_config.py, scripts/pod.py
bug_observed: Concurrent `pod.py config --sync` rewrites dropped the live pod-813 Host entry from ~/.ssh/config twice within ~1.5h while pods.conf still contained it, producing false dead poll verdicts mid-run (task #813, 2026-07-02).
why_workflow_gap: The sync's read-modify-write of the managed SSH-config + MCP env block is not flock-serialized and can rewrite from a stale pods.conf view, silently removing entries added by concurrent sessions.
proposed_change: Flock-guard the whole sync, re-read the canonical repo-root pods.conf under the lock, and make the managed-block rewrite additive-merge (only remove entries absent from the freshly-read pods.conf).
diff_sketch: |
  + lock = open(os.path.expanduser("~/.task-workflow/pod-config.lock"), "w")
  + fcntl.flock(lock, fcntl.LOCK_EX)
  + conf = read_pods_conf(REPO_ROOT / "scripts/pods.conf")  # re-read under lock
  + merged = merge_managed_block(existing_ssh_config, conf)  # additive
confidence: medium
related_task: #813
<!-- /workflow-fix-candidate -->
