---
title: 'daily-fix: env-pin allowlist PYTORCH_CUDA_ALLOC_CONF + help '
kind: infra
tags:
- wf-fix
- wf-fix-fp:c44cccf4b266
- daily-auto-filed
created_at: '2026-07-30T07:02:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): PYTORCH_CUDA_ALLOC_CONF
  (the canonical CUDA-OOM remedy knob) is absent from ENV_PIN_ALLOWED_KEYS and will
  hit the same rejection WANDB keys did; the --env-pin help text enumerates only WandB-family
  examples'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: parked prose candidate on #1803 (Methodology-critic non-blocking suggestions, ts 2026-07-29T23:07:44Z)).

## Goal

A future CUDA-OOM hot-fix pin attempt should not be rejected by the env-pin allowlist; the help text should show a runtime-tuning example.

## Workflow gap

- **Bug observed:** unverified hypothesis — verify at plan time: a PYTORCH_CUDA_ALLOC_CONF pin attempt would hit the identical ENV_PIN_ALLOWED_KEYS rejection (no incident yet; the emitter marked confidence low — 'add it when it first recurs' — filed per the standing any-confidence directive; the spawned planner may deflect with a reasoned no-change report).
- **Why it is a workflow gap:** the allowlist exists to admit vetted runtime keys; the nearest same-family OOM-remedy knob is not on it, and the help text under-documents the surface.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c PYTORCH_CUDA_ALLOC_CONF src/explore_persona_space/backends/base.py` -> 0; `grep -n ENV_PIN_ALLOWED_KEYS src/explore_persona_space/backends/base.py` -> L100 (definition) (2026-07-30, this run).

## Proposed change (refine in planning)

Add the key to ENV_PIN_ALLOWED_KEYS (with the gotchas.md CUDA-OOM cross-ref) and one runtime-tuning example in the --env-pin help (~L2635 in scripts/dispatch_issue.py).

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/base.py, scripts/dispatch_issue.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/base.py, scripts/dispatch_issue.py
- fingerprint: c44cccf4b266

- Origin park: #1803 events 2026-07-29T23:07:44Z (prose park, no fingerprint).
