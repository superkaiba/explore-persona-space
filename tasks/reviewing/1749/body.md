---
title: 'daily-fix: relaunch rounds re-run compute-character check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b4a5506635be
- daily-auto-filed
created_at: '2026-07-28T07:01:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): #1689''s r15b pod-side
  relaunch changed engine/scope (serial numpy CPU, width 1, L19-only) with no compute-character
  re-statement; 4xH100 sat at 0% for ~14h and Thomas had to order the vectorization
  (R16 measured ~10x per pair)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session 078f3709 + ca5c6f64 (#1689), 2026-07-27/28 (miners I P1, J P1).

## Goal

A crash-fix relaunch that changes recipe, engine, or scope must re-state its compute character before dispatch.

## Workflow gap

- **Bug observed:** the r15b relaunch swapped the fit to serial numpy on CPU at width 1 while the 4xH100 pod billed at ~0% GPU for ~14h (two gpu-idle escalations); the compute-deviation record reads ratio 32x (planned 4h -> projected 130h). Thomas intervened ('optimize it so it will run as much in parallel and vectorized as possible'); the R16 port measured numpy 648.5s vs torch/cuda 66s per pair.
- **Why it is a workflow gap:** the compute-character pre-launch statement duty exists for INLINE rounds (CLAUDE.md 9a-ter) but crash-fix-rounds.md's relaunch contract never requires it — a pod-side relaunch silently skips the exact review that would have caught CPU-at-width-1 on a 4-GPU pod (`grep -c 'compute-character' .claude/rules/crash-fix-rounds.md` -> 0).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'compute-character' .claude/rules/crash-fix-rounds.md` -> 0, compose time; the 32x ratio + '4xH100 idle' quoted from #1689's `epm:compute-deviation` v3 marker (durable).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/rules/crash-fix-rounds.md` (relaunch duty list): a relaunch/descope round whose fix changes engine/device/width/scope re-runs the compute-character statement — engine + device routing, MEASURED per-cell wall at the new shape, GPU-width vs pod width — posted in the relaunch marker BEFORE dispatch; a CPU-bound relaunch on a multi-GPU pod triggers the width re-eval (vectorize-many-cell-fits.md).

## Scope / surfaces

- Primary target: `.claude/rules/crash-fix-rounds.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: b4a5506635be

- workflow_fix_target: .claude/rules/crash-fix-rounds.md
