---
title: 'workflow-fix: GCP EXIT-trap poweroff races crash-diagnostics upload'
kind: infra
tags:
- wf-fix
created_at: '2026-07-02T09:05:56Z'
has_clean_result: false
origin_prompt: 'task #825 run-2 crash: EXIT trap logged ''uploading crash diagnostics
  + partial artifacts'' but guest poweroff ~20s later left only crash_report.json
  on HF; workload.log + partials recovered from the orphaned boot disk via diskprobe
  VM'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #825 (emitting agent: /issue orchestrator).

## Goal

Make the GCP EXIT-trap crash-diagnostics persist complete BEFORE poweroff so the workload log + partial artifacts reliably land on HF (not just crash_report.json).

## Workflow gap

- **Bug observed:** On task #825's run-2 crash (2026-07-02T08:15:59Z), the EXIT trap logged "uploading crash diagnostics + partial artifacts to ... issue825_partial/att-20260702-061417" but the guest poweroff fired ~20s later and only the tiny crash_report.json landed on HF — workload.log and the partial data files (track_s.jsonl, conversations_a1_checkpoint.jsonl) never uploaded and had to be recovered by attaching the orphaned boot disk to a diskprobe VM. Round 1's crash (same day, vLLM fork CUDA) left NO diagnostics on HF at all.
- **Why it is a workflow gap:** `_eps_persist_diagnostics` in the GCP startup-script renderer is documented as 300s-bounded and running BEFORE the poweroff that bounds billing, but in practice the poweroff races the upload (large-file uploads killed mid-flight), defeating the "GCP crash is debuggable + partial progress recoverable" contract (CLAUDE.md § Compute backends, #658).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

In src/explore_persona_space/backends/gcp.py startup-script template:
- upload crash_report.json FIRST (small, fast), then workload.log, then partials, each with a synchronous wait; only after the persist function RETURNS (or its 300s bound expires) run poweroff
- ensure the trap shell actually blocks on the upload subprocess (no backgrounded `&` / async hf upload racing `poweroff`)
- log per-file upload completion lines so the race is diagnosable from the log tail
- also emit .env/HF-token availability loudly in the trap (round-1 diagnostics landed nothing; run-2 log carried 'No .env found ... Credentialed calls will fail' warnings)

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln '_eps_persist_diagnostics' .claude/ CLAUDE.md scripts/ src/explore_persona_space/backends/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The persist must stay bounded (never delay poweroff indefinitely — billing bound wins at the 300s cap).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: (computed at filing)

Surfaced prose (task #825 crash-fix round 2, 2026-07-02): EXIT-trap poweroff raced the crash-diagnostics upload; only crash_report.json landed; workload.log + partials recovered manually from the orphaned boot disk via a diskprobe VM.
