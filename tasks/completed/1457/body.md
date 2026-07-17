---
title: 'daily-fix: WARN on device-vs-filesystem size mismatch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f926a2a458d9
- daily-auto-filed
created_at: '2026-07-17T06:57:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the VM boot disk was resized
  to 1 TB in GCP but the filesystem was never grown — df showed 485G with ~500 GB
  unused while disk-pressure automation fired; Thomas had to notice it himself (''i
  think we increased the size to 1 TB'') before growpart+resize2fs ran'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (09f28ede ~05:04Z).

## Goal

Auto-detect a resized-but-unexpanded boot disk instead of relying on the user noticing.

## Workflow gap

- **Bug observed:** the VM boot disk was resized to 1 TB in GCP but the filesystem was never grown — df showed 485G with ~500 GB unused while disk-pressure automation fired; Thomas had to notice it himself ('i think we increased the size to 1 TB') before growpart+resize2fs ran
- **Why it is a workflow gap:** The guard's whole job is disk-pressure awareness; firing cleanup while 500 GB sits unexpanded is the automation missing the cheapest lever.
- **Confidence (emitter):** medium
- verified-at-filing: incident on transcript 09f28ede (~05:04Z; growpart+resize2fs run in-session); `grep -in 'growpart\|resize2fs\|device' scripts/vm_disk_guard.py` — no device-vs-fs reconciliation pass (absence claim)

## Proposed change (candidate diff sketch — refine in planning)

add a device-size vs mounted-fs-size reconciliation check (WARN when the block device exceeds the fs by >5%) to vm_disk_guard so an unexpanded resize is auto-detected

## Scope / surfaces

- Primary target: `scripts/vm_disk_guard.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: f926a2a458d9

