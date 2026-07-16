---
title: 'workflow-fix: gotchas.md — bf16 determinism re-capture bar variant (#1005)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c263f7d9ebfa
created_at: '2026-07-15T19:39:13Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1005: determinism spot-check
  bar calibrated on CPU smoke refused healthy bf16 GPU capture (0.99999887 < 0.999999)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson
(`gotcha_candidate: yes`) raised on task #1005 (emitting agent: orchestrator,
crash-fix round 2).

## Goal

Extend the existing gotchas.md bf16 equivalence-gate calibration entry to name the within-run determinism re-capture variant: a parity/determinism cosine bar calibrated on the bit-exact CPU-float32 smoke fails spuriously on bf16 CUDA (~1e-6 cosine batch-composition noise).

## Workflow gap

- **Bug observed:** issue1005_run.py's within-run determinism spot-check bar 0.999999, calibrated on the CPU smoke, refused a healthy bf16 GPU capture at min cosine 0.99999887 and cost a full GCP instance cycle.
- **Why it is a workflow gap:** the existing gotchas.md entry (the #779 two-bar rule) covers batched-vs-batch-1 equivalence gates calibrated on span-means, but does not name the sibling variant — a within-run re-capture determinism check whose bar was implicitly calibrated by the CPU-float32 smoke passing bit-exactly — so a fresh implementer following the smoke-first discipline recreates the trap.
- **Confidence (emitter):** medium (the existing entry's rule arguably covers the class; this is a one-line variant extension, low-confidence deltas are filed per the 2026-06-11 directive).
- verified-at-filing: `grep -in "determinism\|parity.*cosine\|bf16.*cosine\|cosine.*bar" .claude/rules/gotchas.md` → 1 hit (line 130, the #779 entry — covers batched-vs-batch-1 equivalence gates; 0 hits for the within-run re-capture / CPU-smoke-calibration variant — the absence IS the gap) (2026-07-15).

## Proposed change (candidate diff sketch — refine in planning)

+ Extend the line-130 gotchas.md entry (or add an adjacent sibling bullet): a
+ within-run determinism RE-CAPTURE check (fresh model load, different batch
+ composition) is subject to the same bf16 kernel nondeterminism — never
+ calibrate its bar on the CPU-float32 smoke (bit-exact by construction);
+ >= 0.9999 on span-mean summaries, real failures read < 0.99. Incident #1005
+ (2026-07-15): 0.999999 bar, measured 0.99999887, one wasted GCP cycle.
+ Long-form: .claude/agent-memory/experiment-implementer/feedback_bf16_gpu_parity_gate_tolerance.md

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: c263f7d9ebfa

failure-lesson (verbatim): a capture-parity / determinism gate threshold calibrated on the CPU-float32 smoke (bit-exact) fails spuriously on bf16 CUDA — batch-composition reduction-order noise at the ~1e-6 cosine level; calibrate GPU parity bars to a bf16 margin (>= 0.9999) grounded on committed same-surface reference gates, never a bare constant. (#1005, 2026-07-15)
