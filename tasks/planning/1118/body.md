---
title: 'workflow-fix: thread boot-disk requirement to RunPod GPU volume'
kind: infra
tags:
- wf-fix
- wf-fix-fp:956239fdb6c7
created_at: '2026-07-08T01:06:28Z'
has_clean_result: false
origin_prompt: '#1112 orchestrator observation 2026-07-08: RunPod pivot silently downsized
  the plan''s 575 GB disk need to the 200 GB ft-7b default volume; ENOSPC at FT rung
  24/30. Thread spec.extra boot_disk_gb to --volume-gb or refuse typed.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the #1112 orchestrator's
own observation (2026-07-08).

## Goal

Thread the launch spec's `boot_disk_gb` into the RunPod GPU provision's
`--volume-gb` (or fail loud when a declared disk requirement cannot be
satisfied on the RunPod GPU lane), so a plan-specified disk size survives a
GCP→RunPod pivot.

## Workflow gap

- **Bug observed:** #1112's RunPod pivot silently got the `ft-7b` default
  200 GB `/workspace` volume against a plan requiring ~575 GB peak (2 FT
  checkpoint ladders + marker FT + caches); ENOSPC killed the run at FT rung
  24/30 (safetensors serialization, os error 28) and cost a full pod cycle.
  `backends/runpod.py:767` documents that `boot_disk_gb` deliberately does
  NOT map to the GPU pod's persistent volume (only the CPU fallback threads
  it into containerDiskInGb, #1010), and `dispatch_issue.py launch` offers no
  RunPod-GPU volume flag — so the plan's §9 disk sizing is unenforceable on
  exactly the lane the failover policy pivots to.
- **Why it is a workflow gap:** the failover policy (GCP→RunPod) moves runs
  between lanes whose disk-sizing contracts differ silently; a declared
  requirement should either thread or refuse (the #1010
  `cpu_fallback_infeasible_for_plan` precedent), never silently downsize.
- **Confidence (emitter):** high (live incident; the line-767 comment names
  the gap explicitly).

## Proposed change (candidate diff sketch — refine in planning)

```
+ backends/runpod.py (GPU launch path): read spec.extra['boot_disk_gb']; when
+   set, pass --volume-gb max(default, value) to pod_lifecycle provision
+   (mirroring the CPU lane's containerDiskInGb threading, #1010); when the
+   requested size exceeds the provider/host cap, raise a typed refusal
+   (runpod_volume_infeasible_for_plan) instead of provisioning undersized.
+ dispatch_issue.py --boot-disk-gb help text: document the new GPU-lane
+   mapping (currently documents GCP + CPU-fallback only).
+ tests: spec-with-boot-disk → provision argv carries --volume-gb; oversize →
+   typed refusal.
```

## Scope / surfaces

- Primary targets: `src/explore_persona_space/backends/runpod.py`,
  `scripts/dispatch_issue.py` (help text), `scripts/pod_lifecycle.py` (flag
  pass-through verify only)
- Grep first: `grep -rn 'boot_disk_gb' src/explore_persona_space/backends/ scripts/dispatch_issue.py scripts/pod_lifecycle.py`

## Constraints / invariants

- Never silently downsize a declared requirement; refuse typed (the #1010
  precedent).
- CPU-fallback containerDiskInGb threading (#1010) unchanged.
- No new cost gate; volume size is a plan-declared requirement.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/runpod.py
- fingerprint: 956239fdb6c7

Origin: #1112 orchestrator observation — ENOSPC at FT rung 24/30 on the
200 GB default volume after the GCP→RunPod pivot (markers on #1112,
2026-07-08T01:0xZ).
