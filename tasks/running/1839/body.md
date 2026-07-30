---
title: 'workflow-fix: gotchas parity family — tensor device placement (CPU smokes
  can''t exercise CUDA branches)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b4aed79cad2b
created_at: '2026-07-29T21:30:40Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1776 crash-fix cycle 7 (epm:failure-lesson,
  2026-07-29): CUDA partial device placement — lesson block in the v11 implementation
  marker'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson `gotcha_candidate: yes` block raised on task #1776 (emitting agent: experiment-implementer, crash-fix cycle 7).

## Goal

Add a `.claude/rules/gotchas.md` entry to the smoke/production-parity family: a CUDA-divergent branch is structurally unexercisable on a CPU-only smoke host — device placement must be total-by-construction (one move site at phase entry) plus named-tensor same-device asserts before cross-tensor mm chains.

## Workflow gap

- **Bug observed:** #1776's p4 energy leg crashed on pod-1776 (`RuntimeError: Expected all tensors to be on the same device ... wrapper_CUDA_mm`) AFTER the pursuit battery and two null families ran clean — the cov family's `z @ cov_half.T` was the first cross-device mm (cpu z from `map_location="cpu"` producers vs cuda cov_half). Every CPU smoke passed by construction (all tensors cpu on the smoke host); the branch's first-ever CUDA execution was production on 8×H100, costing a launch cycle.
- **Why it is a workflow gap:** gotchas.md's smoke-parity family covers PROCESS WIDTH (#1315/#1333, ~L46) and bf16-vs-fp32 REGIME calibration (~L197), but nothing documents the DEVICE-PLACEMENT member: partial `.to(device)` wiring passes every CPU smoke and detonates only where CUDA exists — the standard experiment shape (CPU VM smokes, GPU pod production) makes the class recurrent.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n -i "same device\|device mismatch\|map_location\|cuda.*cpu.*smoke" .claude/rules/gotchas.md` → 4 hits in 1 file (L46 process-width parity, L195 logits_to_keep, L197 bf16 regime bars, L235 map_location loading — related families; NO entry covers partial device placement passing CPU smokes; absence claim) (2026-07-29)

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **Smoke/production parity includes TENSOR DEVICE PLACEMENT — a
+   CUDA-divergent branch is structurally unexercisable on a CPU-only smoke
+   host.** Partial placement (some matmul participants moved to args.device,
+   others left cpu by map_location="cpu" producers) passes EVERY CPU smoke by
+   construction and crashes only on the pod, often deep in a battery after
+   co-located sibling paths ran clean (#1776 p4 energy: pursuit + two null
+   families passed; the cov family's z @ cov_half.T was the first
+   cross-device mm; one 8xH100 launch cycle). RULES: (i) placement is TOTAL
+   at ONE move site at the phase entry (rebind every tensor set after all
+   producers, before any consumer) — never scattered incidental .to() calls;
+   (ii) named-tensor same-device asserts immediately before cross-tensor mm
+   chains (the residual mismatch then names its culprit AND doubles as the
+   CUDA-side diagnostic when validation can only happen at relaunch);
+   (iii) off-pod fix-engaged is layered: CPU smoke rc 0 + meta-device unit
+   pins that fail pre-fix + an explicit relaunch-validates note; (iv) audit
+   the leg's device FLOW once (tensor -> load site -> device at load ->
+   device at use) — torch.load(map_location="cpu") producers are the usual
+   leaks. Worked pins: tests/test_issue1776_phase4_device.py (branch
+   issue-1776; fix b38024fb63). Long-form twin:
+   .claude/agent-memory/experiment-implementer/feedback_cuda_device_partial_placement.md
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Place adjacent to the smoke/production-parity family (~L46) so the parity members cluster (width / regime / device).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; gotchas.md `paths:` frontmatter untouched unless the trigger set genuinely widens.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: b4aed79cad2b
