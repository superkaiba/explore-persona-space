---
title: 'daily-fix: vm_disk_guard tier-e nonfinite env clamp'
kind: infra
tags:
- wf-fix
- wf-fix-fp:523affe7972f
- daily-auto-filed
created_at: '2026-07-18T06:46:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-17 problem sweep (route 2): home_hf_repo_escalate_bytes()
  has the nonfinite-env-override gap #1457 fixed for the devfs knobs: EPS_VM_HOME_HF_CACHE_REPO_ESCALATE_GB=inf
  passes the val >= 0.0 clamp into int(gb * 1e9) -> OverflowError.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-17 parked-candidate sweep (Step C) from a prose-followup candidate parked on task #1457 (emitting agent: code-reviewer round-2 prose follow-up; parked under the recursion guard).

## Goal

Add a `math.isfinite` clamp (+ one pinning test) to `home_hf_repo_escalate_bytes()` in `scripts/vm_disk_guard.py` — the same nonfinite-env-override gap #1457 fixed for the devfs knobs — and sweep the sibling tier-(e) knob parsers for the same shape.

## Workflow gap

- **Bug observed:** `home_hf_repo_escalate_bytes()` (scripts/vm_disk_guard.py, the tier-(e) single-repo escalate knob) parses `EPS_VM_HOME_HF_CACHE_REPO_ESCALATE_GB` with only a `val >= 0.0` clamp — `=inf` passes the clamp into `int(gb * 1e9)` → `OverflowError`, crashing the guard run. Pre-existing on trunk, outside every branch hunk of #1457 (which fixed the identical shape for the devfs knobs).
- **Why it is a workflow gap:** `vm_disk_guard.py` is fleet infrastructure (the disk-guard cron + the watcher sub-floor reclaim arm call it); a crash on a malformed env override disables the disk backstop exactly when someone is hand-tuning it.
- **Confidence (emitter):** high
- verified-at-filing: `sed -n '1057,1070p' scripts/vm_disk_guard.py` read at compose time (2026-07-18 UTC) — `home_hf_repo_escalate_bytes()` body confirmed to clamp only `val >= 0.0` with no `isfinite` before `int(gb * 1e9)`; the adjacent `home_hf_cache_cap_bytes()` (#1450 size-cap knob) shows the same parse shape and should be swept in the same fix. `git log --oneline --since='7 days ago' -- scripts/vm_disk_guard.py` shows no later commit landing an isfinite clamp for these functions.

## Proposed change (candidate diff sketch — refine in planning)

```
-        if val >= 0.0:
+        if val >= 0.0 and math.isfinite(val):
             gb = val
```
Plus one pinning test per the #1457 round-2 fix shape, and the same clamp in `home_hf_cache_cap_bytes()` if confirmed vulnerable.

## Scope / surfaces

- Primary target: `scripts/vm_disk_guard.py`
- Grep for sibling knob parsers before editing (`grep -n 'float(raw)' scripts/vm_disk_guard.py`) and clamp every nonfinite-vulnerable one; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes; pinning test green.
- This session runs under the recursion guard — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 523affe7972f

- workflow_fix_target: scripts/vm_disk_guard.py

source candidate (verbatim, prose park on #1457, 2026-07-17T08:17:50Z): "target_file: scripts/vm_disk_guard.py — home_hf_repo_escalate_bytes() (lines ~1046-1059) has the same nonfinite-env-override gap this task just fixed for the devfs knobs: EPS_VM_HOME_HF_CACHE_REPO_ESCALATE_GB=inf passes the 'val >= 0.0' clamp into int(gb * 1e9) -> OverflowError. Pre-existing on trunk (outside every branch hunk of #1457). Proposed change: one-line math.isfinite clamp + one pinning test, same shape as #1457 round-2's fix. confidence: high. related_task: #1457."
