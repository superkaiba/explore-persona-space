---
title: 'workflow-fix: auto-degrade GCP GPU width (8->4->2) for explicit sweep-8g intents
  on capacity failure'
kind: infra
tags:
- wf-fix
- wf-fix-fp:dad49db1f308
created_at: '2026-07-16T07:12:22Z'
has_clean_result: false
origin_prompt: 'add this to workflow (8-GPU explicit sweep intents starve on capacity
  instead of narrowing; #825 turn-dynamics 2.5h stuck, #1116 queue-vanish both pools)'
workflow: v1
---
## Overview / Motivation

Auto-filed workflow-fix from the #825 turn-dynamics-5000 incident (2026-07-16): the MAIN phase requested an explicit 8-GPU sweep intent and starved for 2.5h+ on capacity, because explicit wide intents do NOT auto-degrade width on capacity failure.

## Goal

On repeated 8-GPU (wide) capacity failure, degrade the GCP GPU width (8→4→2) for EXPLICIT `sweep-8g` intents too — not only for `--gpus N` declarations — so a wide-sweep phase falls back to abundant partial-node capacity instead of starving on empty 8-GPU DWS pools.

## Workflow gap

- **Bug observed:** #825 round-11 MAIN phase (`sweep-8g-a100`, then `sweep-8g-h100`) repeatedly hit `terminal_no_compute_available` / FLEX_START queue-vanish (#1116) on BOTH us-central1 8-GPU DWS pools; it never fell back to 4×/2× (which have abundant capacity) and stayed stuck 2.5h+. An orchestrator free-text "narrow to 4×" directive was NOT honored (the width is baked into the explicit intent).
- **Why it is a workflow gap:** the width-aware auto-lane (#1121, `WIDE_A100_80_BY_WIDTH` / `_requested_wide_widths` / `_gcp_ladder_specs` in `backends/router.py`) degrades width only for `--gpus N` requests; an explicit `sweep-8g-*` intent is pinned 8-wide with no degradation path, so a capacity crunch on the single scarcest config (a whole 8-GPU node) blocks the job with no automatic route-around, even though 8-GPU width is a throughput choice (wide-parallel sweep), not a correctness requirement.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -rln -E 'sweep-8g|WIDE_A100|_requested_wide_widths|_gcp_ladder_specs' src/explore_persona_space/backends/*.py` → router.py (WIDE_A100_80_BY_WIDTH:208, _requested_wide_widths:2470, _gcp_ladder_specs:2507, _flex_start_rung:2454) + gcp.py (INTENT_TO_MACHINE, sweep-8g machine maps); confirmed the width-degradation ladder keys on `--gpus N` (`_requested_wide_widths` reads `spec.gpu_count`) and explicit sweep intents bypass it (2026-07-16 UTC).

## Proposed change (candidate diff sketch — refine in planning)

- When an explicit wide intent (`sweep-8g-a100` / `sweep-8g-h100`, or any width-eligible intent resolving to `a2-ultragpu-8g` / `a3-highgpu-8g`) exhausts its wide rungs with capacity failures (queue-vanish #1116 / queue-timeout #783 / no-capacity), have `_gcp_ladder_specs` append degraded-width rungs (8→4→2) from `WIDE_A100_80_BY_WIDTH` before the terminal `no_compute_available`, matching the existing `--gpus N` degradation.
- Gate it so genuinely width-required jobs (a stated shared-nothing 8-way memory/parallelism need) can opt out; default = degrade (throughput-only wide sweeps are the common case).
- Emit an `epm:progress` note when width is auto-degraded so the run records that it dropped from 8→N.

## Scope / surfaces
- Primary target: `src/explore_persona_space/backends/router.py` (+ `src/explore_persona_space/backends/gcp.py` for the intent→machine width maps).
- Grep the surface for `sweep-8g` / `WIDE_A100_80_BY_WIDTH` / `_requested_wide_widths` and update every hit; add a `tests/test_router*.py` case that an explicit `sweep-8g` intent degrades width on simulated wide-capacity exhaustion.

## Constraints / invariants
- Workflow-surface only (`backends/*.py` is in-scope per `.claude/rules/workflow-fix-on-bug.md`).
- Preserve the RunPod-last-rung + no-cascade invariants (`tests/test_router.py`); width-degrade stays within the GCP ladder before any RunPod/terminal fallback.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/router.py
- fingerprint: (computed by filer)

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/router.py, src/explore_persona_space/backends/gcp.py
bug_observed: explicit 8-GPU sweep intents (sweep-8g-a100/h100) do not auto-degrade width on capacity failure, so a wide-sweep phase starves on empty 8-GPU DWS pools (#825 turn-dynamics, 2.5h+ stuck, #1116 queue-vanish on both A100 and H100 pools).
why_workflow_gap: the width-aware ladder (#1121) degrades only `--gpus N` requests; explicit sweep intents are pinned 8-wide with no degradation path, and 8-GPU width is a throughput choice not a correctness need.
proposed_change: on wide-capacity exhaustion for an explicit sweep-8g intent, append degraded-width rungs (8→4→2) from WIDE_A100_80_BY_WIDTH before the terminal no_compute_available, with an opt-out for genuinely width-required jobs.
diff_sketch: |
  # router.py _gcp_ladder_specs(): after the wide rungs for an explicit sweep-8g
  # intent, if width-degradable, append 4x and 2x rungs from WIDE_A100_80_BY_WIDTH
  # (same as _requested_wide_widths does for --gpus N), then the narrow chain.
confidence: medium
related_task: #825
<!-- /workflow-fix-candidate -->
