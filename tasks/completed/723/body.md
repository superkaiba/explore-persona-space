---
title: 'workflow-fix: gotchas.md — rsLoRA parity probe needs A100 intent, fails on
  L4'
kind: infra
tags:
- wf-fix
- wf-fix-fp:rslora-l4-precision-001
created_at: '2026-06-28T23:07:29Z'
has_clean_result: false
origin_prompt: 'epm:failure-lesson v1 from #667 a36-readout-reextract-cos round 1:
  rsLoRA parity probe rc=1 on L4 (eval intent), fix via --intent lora-7b for A100-80.
  gotcha_candidate: yes.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson surfaced
on task #667 (round-1 GCE launch on `eval`-intent L4 crashed in
phase_reextract_prefetch's rsLoRA numeric parity probe; round 2 fixed by
switching to `--intent lora-7b` for A100-80).

## Goal

Add a gotcha entry to `.claude/rules/gotchas.md` warning that an inherited
rsLoRA numeric-parity probe (the `_rslora_parity_probe` / `assert_adapter_gauge`
shape) minted on A100-80 (the #537/#667 line) FAILS on L4 (`--intent eval` →
`g2-standard-4`, compute capability 8.9, bf16/TF32 precision mismatch); a plan
§9 naming "A100-80" must launch with `--intent lora-7b` (→ `a2-ultragpu-1g`),
regardless of whether the workload is forward-pass-only.

## Workflow gap

- **Bug observed:** task #667 round-1 GCE launch (attempt
  att-20260625-105641, `g2-standard-4` 1× L4 FLEX_START via `--intent eval`)
  crashed in `phase_reextract_prefetch` when the inherited
  `_rslora_parity_probe` subprocess for em/default exited rc=1; the
  diagonal-write numerical reproduction did not match #537's committed gauge
  on L4 due to bf16/TF32 precision differences. The probe correctly HALTed;
  this is a gauge/hardware precision mismatch, not a code regression.
- **Why it is a workflow gap:** `.claude/rules/gotchas.md` has no entry
  warning planners + experimenters that the `eval` intent's auto-routed L4
  machine cannot reproduce A100-80-minted gauges. A plan that names
  "A100-80" in §9 and passes `--intent eval` is internally inconsistent but
  passes verify_plan.py without warning, and the experimenter inherits the
  inconsistency at launch time.
- **Confidence (emitter):** medium-high (the failure was concretely
  reproduced; the fix landed cleanly on A100-80).

## Proposed change (candidate diff sketch — refine in planning)

Add a gotcha to `.claude/rules/gotchas.md`, in the same section as the
existing GCP-lane gotchas:

```
- **rsLoRA parity probes minted on A100 FAIL on L4 — match launch intent to
  the gauge's origin hardware, not to the workload kind.** An inherited
  `_rslora_parity_probe` / `assert_adapter_gauge` that reproduces a committed
  diagonal-write gauge from a prior task (the #537/#667 line, A100-80-trained)
  will HALT in `phase_reextract_prefetch` with subprocess rc=1 on an L4
  (`--intent eval` → `g2-standard-4`, compute capability 8.9, different
  bf16/TF32 precision than the A100-80 the gauge was minted on). Fix: launch
  with `--intent lora-7b` (→ `a2-ultragpu-1g`, 1× A100-80) regardless of
  whether the workload is forward-pass-only. Diagnosis pattern: a Python
  `RuntimeError: rsLoRA NUMERIC parity probe subprocess exited rc=1` from
  `_run_parity_probe_subprocess` on an L4 GCE instance is this exact class.
  Closed regression: #667 a36 round 1 (2026-06-28).
```

Also worth considering (not in scope for this fix; the planner can
deflect with a reasoned no-change report if it decides not):
- A `verify_plan.py` check that flags a plan §9 naming "A100-80" while
  the launch command (or plan-implied intent) uses `--intent eval`.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Out of scope for this fix: the `INTENT_TO_MACHINE` mapping itself
  (`backends/gcp.py`) — `eval` correctly routes to the cheap L4 for
  workloads that don't depend on hardware-specific gauges; the bug is
  in plan/launch routing, not in the mapping.

## Constraints / invariants

- Workflow-surface only — `.claude/rules/gotchas.md` is in scope.
- ruff on touched files passes (n/a — this is a markdown rule file).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: rslora-l4-precision-001

Related task: #667 (a36-readout-reextract-cos round 1 GCE launch failure).
