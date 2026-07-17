---
title: 'daily-fix: 429 hook output-cap pacing + message fix'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-17T06:56:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the on-stop-429-retry Stop
  hook re-wakes every ~2s with a hardcoded ''input-token cap should be higher now''
  message regardless of limiter; on 2026-07-16 org-wide OUTPUT-TPM storms it burned
  27 (#1345) / 66 (#1310) / 96 (#1005) / 32 (#1400) consecutive wasted turns because
  it neither parses input-vs-output TPM from the 429 body nor sleeps to the next minute
  boundary'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining: four sessions burned 27-96 consecutive turns each in the ~16:05-16:12Z org-wide output-TPM storm. NOTE: the target is the USER-GLOBAL hook file (outside the repo) — the /issue session edits it in place on the VM; there is no repo diff to merge for that file (tests/pins in-repo where feasible).

## Goal

Make the 429-retry Stop hook limiter-aware and minute-boundary-paced so an output-TPM storm costs 1-2 turns, not ~100.

## Workflow gap

- **Bug observed:** the on-stop-429-retry Stop hook re-wakes every ~2s with a hardcoded 'input-token cap should be higher now' message regardless of limiter; on 2026-07-16 org-wide OUTPUT-TPM storms it burned 27 (#1345) / 66 (#1310) / 96 (#1005) / 32 (#1400) / 112 (#1395) / 92 (#1397) / 68 (#825) consecutive wasted turns — and drove 99 (#1415) / 150 (#1398) instant hook re-wakes over a ~3.2h wall in the worst window because it neither parses input-vs-output TPM from the 429 body nor sleeps to the next minute boundary
- **Why it is a workflow gap:** The hook is the fleet-wide 429 recovery mechanism; re-waking instantly against a per-minute cap is a self-inflicted storm.
- **Confidence (emitter):** high (4 independent storms today)
- verified-at-filing: `grep -n 'input-token' ~/.claude/hooks/on-stop-429-retry.sh` -> L75/L77 hardcoded message; `grep -c output ~/.claude/hooks/on-stop-429-retry.sh` -> 0 (no output-TPM handling); no sleep/minute-boundary logic in the hook

## Proposed change (candidate diff sketch — refine in planning)

parse the 429 body (input vs output TPM), sleep to the next minute boundary + jitter before re-waking, cap consecutive re-wakes per storm, and correct the hardcoded message

## Scope / surfaces

- Primary target: `~/.claude/hooks/on-stop-429-retry.sh`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Non-workflow-surface fix (`wf_fix: false`): no recursion guard applies; standard /issue pipeline.

