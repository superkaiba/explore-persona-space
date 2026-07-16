---
title: 'workflow-fix: floor GCP boot-disk at image minimum (100 GB)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d2872ceceed0
created_at: '2026-07-16T00:38:30Z'
has_clean_result: false
origin_prompt: 'orchestrator-surfaced candidate from #1336 E1 dispatch failure: GCP
  create rejects --boot-disk-gb below the 100 GB DLVM image size; clamp up at backends/gcp.py:2850
  instead of failing the rung'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1336 (emitting agent: orchestrator).

## Goal

Clamp the GCP lane's boot-disk size to the image minimum (100 GB for the DLVM image) instead of letting an undersized plan value fail the provisioning rung.

## Workflow gap

- **Bug observed:** `dispatch_issue.py launch --issue 1336 --intent cpu-mid --boot-disk-gb 60` failed the GCP rung with `Invalid value for field 'resource.disks[0].initializeParams.diskSizeGb': '60'. Requested disk size cannot be smaller than the image size (100 GB)`, then the auto chain exhausted (RunPod cpu3c-8-16 caps container disk at 50 GB) and the launch died with `cpu_fallback_infeasible_for_plan` — a healthy 0-GPU workload was blocked by a sizing formality.
- **Why it is a workflow gap:** `backends/gcp.py:2850` coerces `spec.extra["boot_disk_gb"]` straight into `--boot-disk-size` (gcp.py:2878) with NO floor against the image minimum, while the RunPod lane already floors its equivalent (`_CPU_CONTAINER_DISK_FLOOR_GB`, runpod.py:819). A plan-stated footprint below 100 GB is always satisfiable by a 100 GB disk — clamping up is strictly safe (footprint is a MINIMUM requirement) and costs pennies; failing the rung wastes a provisioning attempt + can exhaust the chain.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'boot_disk_gb' src/explore_persona_space/backends/gcp.py` → 6 hits; the unfloored coercion at gcp.py:2850 + threading at gcp.py:2878 confirmed; RunPod floor precedent at runpod.py:819 confirmed (2026-07-16). Incident evidence: #1336 events.jsonl epm:progress v40 + the failed launch's backend-selected attempts payload.

## Proposed change (candidate diff sketch — refine in planning)

```
+ _GCP_IMAGE_MIN_BOOT_DISK_GB = 100  # DLVM image size; create rejects smaller disks
...
- boot_disk_gb = int(spec.extra.get("boot_disk_gb") or config.default_boot_disk_gb)
+ boot_disk_gb = max(
+     _GCP_IMAGE_MIN_BOOT_DISK_GB,
+     int(spec.extra.get("boot_disk_gb") or config.default_boot_disk_gb),
+ )
+ (log one INFO line when the clamp raises a plan-stated value)
```

Plus a unit test pinning the clamp (a spec.extra boot_disk_gb=60 resolves to a --boot-disk-size=100GB argv).

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing (`grep -rn 'boot_disk_gb' src/explore_persona_space/backends/`) and update every hit that threads an unfloored value into a GCP create; list them in the plan.
- Consider whether `tests/test_gcp_backend.py` is the right home for the pin.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The clamp must NEVER shrink a stated footprint (only raise), and must not change the default (300 GB) path.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: d2872ceceed0

Surfaced prose (orchestrator observation, task #1336 E1 dispatch, 2026-07-16): GCP rung failed on --boot-disk-gb 60 < image size 100 GB; RunPod fallback infeasible (>50 GB); chain exhausted. Fix: floor the GCP boot disk at the image minimum, mirroring the RunPod container-disk floor.
