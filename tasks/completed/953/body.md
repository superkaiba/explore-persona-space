---
title: 'workflow-fix: gotchas.md — GCE manual-clone requires mirroring ALL [FAIL]-guarded
  metadata secrets'
kind: infra
tags:
- wf-fix
created_at: '2026-07-04T01:21:54Z'
has_clean_result: false
origin_prompt: 'epm:failure-lesson v2 on #833: manual GCE clone died rc=78 x3 — WANDB_API_KEY
  missing from instance metadata; mirror ALL [FAIL]-guarded metadata keys'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gotcha_candidate failure-lesson raised on task #833 (emitting agent: orchestrator, r7d tail-instance launch).

## Goal

Add a gotchas.md bullet documenting that manually cloning an EPS GCE startup script requires mirroring ALL [FAIL]-guarded instance-metadata attributes, plus the live-serial-stream diagnosis recipe.

## Workflow gap

- **Bug observed:** a manually created tail instance (clone of eps-issue-833's startup script) hard-FAILed rc=78 + EXIT-trap poweroff ~4 min after boot, 3 times, because only HF_TOKEN was mirrored; the script [FAIL]-guards WANDB_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY / HF_* flags / enable-guest-attributes / eps-issue / eps-attempt-id too.
- **Why it is a workflow gap:** the GCE metadata contract is encoded only inside the generated startup script (backends/gcp.py); no rule documents it for manual instance ops, and the failure is invisible post-mortem (serial unreadable on TERMINATED; fail-guard fires before crash-persist env exists).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ .claude/rules/gotchas.md: new bullet "GCE manual instance clone: mirror ALL metadata"
+   - copy metadata via `describe --format=json` + `add-metadata` (no-echo), keys:
+     HF_TOKEN, WANDB_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
+     HF_HUB_ENABLE_HF_TRANSFER, HF_XET_HIGH_PERFORMANCE,
+     enable-guest-attributes, eps-issue, eps-attempt-id
+   - diagnose silent fast-TERMINATED boots via a LIVE
+     `gcloud compute instances tail-serial-port-output` stream on restart

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for related GCE-metadata prose before editing.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` passes.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: gce-metadata-mirror-clone
- origin: epm:failure-lesson v2 on #833 (2026-07-04T01:20:43Z)
