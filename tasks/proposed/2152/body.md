---
title: Rule-26 judge pilot gate must exercise the same TRANSPORT as the wave it gates
  (sync pilot passed a batch wave that lost 34% of draws)
kind: infra
tags: []
created_at: '2026-08-06T15:15:07Z'
has_clean_result: false
parent_id: 1739
origin_prompt: 'workflow-fix candidate surfaced during #1739: rule-26 pilot routed
  SYNC at 200 draws while gating a forced-BATCH 44,310-call wave; transport-conditional
  refusal censor was structurally undetectable by the gate'
workflow: v1
---
## Goal

Require a rule-26 judge pilot gate to exercise the SAME TRANSPORT as the production wave it gates, so a transport-conditional failure mode cannot pass the gate and then destroy a third of the production wave.

## The gap

`.claude/rules/llm-judging.md` rule 26 requires every >=~5,000-call judge wave to be pilot-gated: ~100-200 draws at "the exact production instrument", gating on zero `stop_reason == "max_tokens"` and per-arm parse-fail < ~2%.

"Exact production instrument" is currently read as model + rubric + temperature + max_tokens. It does NOT include the HTTP transport. But the dispatcher routes by CALL COUNT: a ~200-draw pilot falls below the batch crossover and runs SYNC, while the production wave it gates is typically forced onto the BATCH path (`--threshold-base 0`).

So the pilot and the wave systematically run DIFFERENT transports. Any transport-conditional failure mode is invisible to the gate at any sample size or stratification — the pilot is structurally incapable of detecting it.

## Evidence (#1739, 2026-08-06)

A rule-26 pilot on the evil-OOD wave PASSED at ~1.5-4% parse-fail with zero truncation. The production wave that it gated then returned **15,091 of 44,310 draws (34.1%) with `stop_reason="refusal"`** and empty content — collapsing coverage to 4,982 items (33.7%) with ZERO valid draws.

The failure was entirely transport-conditional: re-issuing the identical instrument on the SYNC path produced **0 re-refusals in 14,887 draws**. The pilot had run SYNC. It could not have caught this.

Cost of the miss: a full 44,310-call wave spent on a censored result, plus a ~15,000-call sync re-judge to repair it, plus the diagnosis time. The censoring was also outcome-correlated (rescued items scored 1.4x higher, up to 3.7x on one corpus), so had it gone unnoticed it would have silently biased the headline DV downward on its most positive rows.

An initial diagnosis attributed the miss to the pilot's sample not being stratified over corpora/rungs. That is WRONG and the fix that follows from it would not have helped — stratification cannot surface a failure mode the pilot's transport never exercises. Recording this because the wrong lesson was briefly written into the task record (#1739 v645) before being corrected (v649).

## Deliverables

1. Rule 26 states that the pilot must run the SAME TRANSPORT as the gated wave, and that transport is part of "the exact production instrument".
2. A mechanism to force it — the pilot cannot rely on call count to route, since ~200 draws will always fall below the batch crossover. Either the pilot forces the production transport explicitly (e.g. threading the wave's `threshold_base`), or it runs a small dedicated batch-path probe alongside the sync pilot.
3. The gate reports which transport the pilot ran and which the wave will run, and FAILS on a mismatch rather than passing silently.
4. Consider adding an API-level-refusal rate to the gate's pass criteria alongside truncation and parse-fail — it is the specific failure this gap let through.

## Out of scope

Classifying the API-level `refusal` return itself (a third drop class in rules 9/24/18 plus the `judge_dispatch.py:670-681` seam) is filed separately.
