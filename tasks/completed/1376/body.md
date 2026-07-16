---
title: 'workflow-fix: vm_disk_guard tier for the home HF hub cache'
kind: infra
tags:
- wf-fix
- wf-fix-fp:47a604199d3a
created_at: '2026-07-16T04:38:10Z'
has_clean_result: false
origin_prompt: 'Orchestrator-observed during the 2026-07-16 root-disk CRITICAL episode:
  guard''s hub-cache tier covers only /workspace; the home cache (101 GB, 76 GB =
  one 12-revision data-repo cache) is invisible to it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed gap during the 2026-07-16 VM root-disk CRITICAL episode (emitting agent: orchestrator, user-chat session).

## Goal

Extend `vm_disk_guard.py` with a tier covering the HOME HF hub cache (`~/.cache/huggingface/hub`) — at minimum per-repo/per-revision attribution + escalation, and a safe reap arm for stale UNREFERENCED revisions of multi-revision repo caches (the project data repo accumulates one revision per pinned read; 12 revisions / 76.2 GB observed) — so the fleet's dominant root-disk consumer is no longer invisible to the guard.

## Workflow gap

- **Bug observed:** During the 2026-07-16 00:50–01:44Z `vm-disk-low` CRITICAL episode (`/` at 98%, 12 GiB free), `vm_disk_guard.py --apply` freed only 0.53 GB: its hub-cache tier (d) covers ONLY the pod-style `/workspace/.cache/huggingface` (21 GB, nothing stale), while the actual dominant consumer was `~/.cache/huggingface` at 101 GB — 76.2 GB of it a single dataset-repo cache (`superkaiba1/explore-persona-space-data`, 9,490 files, 12 cached revisions) that no guard tier attributes, escalates, or reaps.
- **Why it is a workflow gap:** the guard is the fleet's root-disk backstop; its blind spot on the home hub cache means the escalation/attribution machinery (sidecar rows, Telegram, ack sentinels) never names the biggest consumer, and every CRITICAL episode requires manual diagnosis (this one took an orchestrator scan-cache session while `/` sat in the silent-Bash-failure band, #552 class).
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "huggingface\|HF_HOME\|workspace" scripts/vm_disk_guard.py` → 8 relevant hits, all tier-(d) `/workspace/.cache/huggingface` scoped (lines 30-34, 640-659: `DEFAULT_WORKSPACE_HF_CACHE = "/workspace/.cache/huggingface"`); 0 hits reference the home cache path — absence-of-coverage claim, the 0-hit result IS the evidence (2026-07-16)

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/vm_disk_guard.py:
+ tier (e): home HF hub cache (~/.cache/huggingface/hub, env EPS_VM_HOME_HF_CACHE):
+   (1) ATTRIBUTION always: per-repo size + last_accessed + revision count via
+       huggingface_hub.scan_cache_dir; top consumers named in the report +
+       sidecar escalation rows (never silent).
+   (2) SAFE REAP arm (apply mode, both boot + threshold passes): for
+       multi-revision REPO caches, delete revisions that are (a) not
+       referenced by any ref, or referenced only by non-main refs, AND
+       (b) not accessed within EPS_VM_HOME_HF_CACHE_MAX_AGE_DAYS (default 7)
+       — via scan_cache_dir().delete_revisions() (blob-refcount safe).
+   (3) NEVER touch a repo accessed within the age window as a whole;
+       models (e.g. Qwen weights) covered by the same age gate.
+ escalation: a single repo cache > EPS_VM_HOME_HF_CACHE_REPO_ESCALATE_GB
+ (default 40) always escalates with the per-revision breakdown.
(tests: tests/test_vm_disk_guard*.py — add tier-(e) fixtures)
```

## Scope / surfaces

- Primary target: `scripts/vm_disk_guard.py`
- Secondary: `scripts/autonomous_session_watch.py` (the `vm-disk-low` sentinel's attribution already exists — cross-check it names the home cache), `.claude/rules/background-automation.md` + CLAUDE.md § Disk hygiene (document the new tier), guard tests.
- Grep the workflow surface for the pattern before editing (`grep -rln 'workspace_hf_cache\|EPS_VM_WORKSPACE_HF_CACHE' .claude/ CLAUDE.md scripts/ tests/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The reap arm must be blob-refcount safe (use the hub's delete_revisions strategy, never raw rmtree of blobs) and must fail toward KEEP on any scan error.
- `scripts/workflow_lint.py` default run passes; ruff on touched files passes; CLAUDE.md/rule-file consistency maintained.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/vm_disk_guard.py
- fingerprint: 47a604199d3a

Orchestrator-observed gap (synthesized per the workflow-fix-on-bug prose-followup rule). Evidence: #1092 events.jsonl vm-disk-low CRITICAL rows (2026-07-16T00:50Z, 01:44Z); guard apply run 2026-07-16 ~00:00Z freeing 0.53 GB with `/` at 94.8%; orchestrator `hf cache scan` at 04:35Z attributing 76.2 GB to the single data-repo cache with 12 revisions, last accessed 53 min prior.
