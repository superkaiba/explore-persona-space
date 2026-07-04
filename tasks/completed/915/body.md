---
title: 'workflow-fix: janitors fail rmtree on symlinked per-issue cache dirs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:43b535edca3f
created_at: '2026-07-03T07:16:28Z'
has_clean_result: false
origin_prompt: 'Disk diagnosis 2026-07-03: vm_disk_guard --apply exited 2 on ''issue
  761: FAILED to remove data/issue_761/g1_dl'' — the cache dirs are symlinks to /mnt/eps-data
  and shutil.rmtree refuses symlinks.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised during a chat-mode disk-full diagnosis (2026-07-03; emitting agent: orchestrator).

## Goal

Handle symlinked per-issue cache dirs in the janitors: unlink the symlink and reap the target only when it resolves inside the managed eps-data cache root, instead of failing rmtree on the link.

## Workflow gap

- **Bug observed:** `vm_disk_guard.py --apply` reported `issue 761: FAILED to remove data/issue_761/g1_dl` (and `hf_dl`) and exited 2, because those cache dirs are symlinks to `/mnt/eps-data/thomasjiralerspong/eps-data/issue_761/...` and `shutil.rmtree` refuses symlinks — despite zero boot-disk bytes at stake.
- **Why it is a workflow gap:** relocating a per-issue cache onto the data disk via a symlink is the natural interim pattern before the #681 cutover (and #761 already did it); every janitor that assumes cache dirs are real directories now fails loud on the healthiest configuration, and the guard's nonzero exit masks real failures.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
clean_experiment_downloads.py / vm_disk_guard.py reap path:
+ if os.path.islink(cache_dir):
+     target = os.path.realpath(cache_dir)
+     os.unlink(cache_dir)                       # link itself is boot-disk-free
+     if target.startswith(EPS_DATA_CACHE_ROOT): # managed relocation only
+         shutil.rmtree(target, ignore_errors=False)
+     else: log "symlink unlinked, external target kept: <target>"
- shutil.rmtree(cache_dir)  # raises on symlink today
```

## Scope / surfaces

- Primary target: `scripts/clean_experiment_downloads.py, scripts/vm_disk_guard.py`
- Grep both janitors for every `rmtree` call on a per-issue cache path (`grep -n 'rmtree' scripts/clean_experiment_downloads.py scripts/vm_disk_guard.py`) and cover each; add a symlinked-cache test.
- Sibling task #911 (widen janitor glob coverage) touches the same files — the planner should coordinate/sequence with it, but this is a distinct bug (rmtree-on-symlink failure, not glob coverage).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- NEVER delete `store/`, `eval_results/`, or any active issue's data; never follow a symlink outside the managed eps-data cache root.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/clean_experiment_downloads.py, scripts/vm_disk_guard.py
- fingerprint: 43b535edca3f

Raised during the 2026-07-02/03 disk-full diagnosis: a `vm_disk_guard.py --apply` run (log: `/tmp/vm_disk_guard_run.log`) freed 0.00G and exited 2 on the #761 symlinked caches.
