---
title: 'workflow-fix: janitors miss issue-keyed caches outside canonical globs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:558746fb1c07
created_at: '2026-07-03T07:01:54Z'
has_clean_result: false
origin_prompt: 'Chat-mode disk diagnosis 2026-07-02: boot disk 95% full; ~90G of issue-keyed
  re-downloadable HF-mirror caches in /tmp/i779_mirror_dl (59G), data/issue779_hfstage
  (15G), data/issue_744_dl (11G), /tmp/fact_check_823 (6G), /workspace/.cache (22G)
  escaped clean_experiment_downloads.py + vm_disk_guard.py globs; guard saw only ~80M
  reclaimable.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised during a chat-mode disk-full diagnosis (2026-07-02, boot disk at 95%; emitting agent: orchestrator).

## Goal

Widen disk-janitor coverage to issue-keyed re-downloadable caches outside the canonical `data/issue_<N>/{hf_dl,g*_dl}` layout (`/tmp/i<N>_*`, `/tmp/issue-<N>*`, `data/issue<N>_*`, `data/issue_<N>_dl`, VM `/workspace/.cache`) with terminal-status reap plus active-issue escalation, `store/` never touched.

## Workflow gap

- **Bug observed:** Boot disk hit 95% while the guard reported only ~80M reclaimable: ~90G of issue-keyed HF-mirror caches sat in non-canonical locations (`/tmp/i779_mirror_dl` 59G, `data/issue779_hfstage` 15G, `data/issue_744_dl` 11G, `/tmp/fact_check_823` 6G, `/workspace/.cache` 22G) invisible to both janitors.
- **Why it is a workflow gap:** `clean_experiment_downloads.py` and `vm_disk_guard.py` tier (b) sweep only the canonical `data/issue_<N>/{hf_dl,g*_dl}` globs, so any run that stages HF downloads under `/tmp`, a non-canonical `data/` name, or a pod-style `/workspace` path on the VM escapes both the Step-8 cleanup and the disk-guard backstop — the fleet accumulates until the disk wedges.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
clean_experiment_downloads.py:
+ CACHE_GLOB_PATTERNS += [
+   "/tmp/i{N}_*", "/tmp/issue-{N}*",          # tmp staging dirs
+   "data/issue{N}_*", "data/issue_{N}_dl",    # non-canonical data/ names
+ ]  # same safety contract: store/ + eval_results/ never touched
vm_disk_guard.py tier (b):
+ include the same issue-keyed patterns; terminal-status -> reap,
+ active-status -> escalate (sidecar + Telegram), never delete active data
+ add a VM /workspace/.cache pass (pod-style HF cache duplicated on the VM)
```

All local copies verified to be partial downloads of folders that exist on the HF data repo (`issue779_monitoring/`, `issue744_token_continuity/` confirmed via `list_repo_tree` 2026-07-02) — deletion is pure re-downloadable cache loss.

## Scope / surfaces

- Primary target: `scripts/clean_experiment_downloads.py, scripts/vm_disk_guard.py`
- Grep the workflow surface for the canonical glob pattern before editing (`grep -rln 'hf_dl' scripts/ .claude/ CLAUDE.md`) and update every janitor touchpoint; list them in the plan.
- Consider also making writers converge: the deeper fix may be a shared "issue cache root" helper so runs stop inventing `/tmp/i<N>_*` staging paths in the first place — planner's call.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- NEVER delete `store/`, `eval_results/`, or any active issue's data (escalate instead).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; CLAUDE.md § Disk hygiene stays consistent with the implemented behavior.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/clean_experiment_downloads.py, scripts/vm_disk_guard.py
- fingerprint: 558746fb1c07

Raised as a prose candidate during the 2026-07-02 disk-full diagnosis (boot disk 95%, 458G used; janitor-visible reclaim ~80M vs ~90G actually sitting in non-canonical issue-keyed caches).
