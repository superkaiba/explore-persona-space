---
title: 'daily-fix: gcp.py loud marker on boot-disk reuse + size mism'
kind: infra
tags:
- wf-fix
- wf-fix-fp:daeaa1c91693
- daily-auto-filed
created_at: '2026-07-23T06:39:04Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): a GCE create can silently
  reuse a surviving boot disk from a prior attempt with the requested --boot-disk-gb
  silently ignored, leaving no marker-visible record'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-22 parked-candidate routing pass (Step C) from a recursion-guard-parked FORMAL candidate block on task #1602 (source: plan v3 §9 delegated-judgment assessment).

## Goal

Make the GCP dispatch layer record loudly when a GCE create attaches/reuses a pre-existing boot disk (age + realized size on the `epm:backend-selected` marker extra) and WARN when the realized disk size differs from the requested `--boot-disk-gb`.

## Workflow gap

- **Bug observed:** a GCE create can silently reuse a surviving boot disk from a prior attempt (issue #779: a 20-day-old 300 GB disk was attached to a fresh create and the dispatch's `--boot-disk-gb 200` was silently ignored), leaving no marker-visible record that reuse happened or that the realized disk differs from the requested size.
- **Why it is a workflow gap:** the reuse itself is by-design (the idempotent repo-reuse else-branch exists for it), but the SILENCE is a dispatch-layer observability gap — the #1602 crash class was undiagnosable from markers alone and the size-flag mismatch is invisible until something fills the disk.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'boot_disk_reused\|boot_disk_size_mismatch\|boot disk' src/explore_persona_space/backends/gcp.py` → 0 hits for the proposed marker fields (`boot_disk_reused` / `boot_disk_size_mismatch`), 6 hits for prose mentions of "boot disk" none of which implement reuse detection (lines 408/410/1222/1754/2744/2784 — comments about sizing and DELETE-on-shutdown), 2026-07-23 UTC. Absence-of-feature claim: 0-hit in-target result is the evidence. Landed-fix history check: `git log --oneline --since='7 days ago' -- src/explore_persona_space/backends/gcp.py` → 5 commits (493693503f #1605 crash-persist sweep, 4b097d46bc #1602 branch-switch-safe repo-reuse, d7b99c2fbd #1574, 5f3dcde7ee #1547, f0fc4fb4ce #1517) — the landed #1602 fix covered the startup-script branch-switch, NOT this observability piece.

## Proposed change (candidate diff sketch — refine in planning)

```
+ after instance create resolves: describe the boot disk (creationTimestamp, sizeGb)
+ extra["boot_disk_reused"] = true/false; extra["boot_disk_age_days"]; extra["boot_disk_size_gb"]
+ if requested_boot_disk_gb and realized != requested: loud stderr WARN + extra["boot_disk_size_mismatch"] = true
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py` (in-scope: the unified backend router package is workflow surface per workflow-fix-on-bug.md)
- Grep the workflow surface for related marker-extra composition sites before editing; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `tests/test_gcp_backend.py` and the router tests pass; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: daeaa1c91693 (tag-authoritative; supersedes body-carried fingerprint: ccd20cefddf0)

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/gcp.py
bug_observed: a GCE create can silently reuse a surviving boot disk from a prior attempt (issue 779: a 20-day-old 300 GB disk was attached to a fresh create and the dispatch's --boot-disk-gb 200 was silently ignored), leaving no marker-visible record that reuse happened or that the realized disk differs from the requested size
why_workflow_gap: the reuse itself is by-design (the idempotent repo-reuse else-branch exists for it), but the SILENCE is a dispatch-layer observability gap — the #1602 crash class was undiagnosable from markers alone and the size-flag mismatch is invisible until something fills the disk
proposed_change: in the gcp.py launch/create path, detect when a create attaches or reuses a pre-existing boot disk and record it loudly on the epm:backend-selected marker extra (disk age + realized size), plus a stderr WARN when the realized disk size differs from the requested --boot-disk-gb
diff_sketch: |
  + after instance create resolves: describe the boot disk (creationTimestamp, sizeGb)
  + extra["boot_disk_reused"] = true/false; extra["boot_disk_age_days"]; extra["boot_disk_size_gb"]
  + if requested_boot_disk_gb and realized != requested: loud stderr WARN + extra["boot_disk_size_mismatch"] = true
confidence: medium
related_task: #1602
<!-- /workflow-fix-candidate -->
