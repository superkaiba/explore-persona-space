---
title: 'workflow-fix: FT ckpt disk high-water must match implemented phase ordering
  (reap every accumulating phase)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3c81f6385eaa
created_at: '2026-07-23T03:33:58Z'
has_clean_result: false
origin_prompt: Fix all the problems in the background with happycoder (#1586 r5 train-all-then-ladder
  ENOSPC)
workflow: v1
---
## Overview / Motivation

Auto-filed from a deep-dive on #1586's crash history (2026-07-23, user-directed).
#1586 r5 died with a safetensors ENOSPC: the plan modeled a per-cell / 2-concurrent
ladder high-water (~456 GB), but the dispatcher trained ALL content cells BEFORE the
ladder phase, accumulating ~2.5 TB of full-FT checkpoints (15 GB x 15 rungs x cells)
on a 750 GB volume; stream-reap existed only inside `run_ladder_unit`, so it could not
bound the `p2_train` accumulation (plus a wave-headroom resume BLOCKER). The code was
fixed reactively (bounded-wave pipelining), but `plan-compute-sizing.md` already
modeled FT checkpoint retention (#1133, #653) and STILL did not prevent this — because
the modeling was not tied to the IMPLEMENTED phase ordering.

## Goal

Make the FT-checkpoint disk high-water a function of the implemented phase ordering:
every phase that accumulates checkpoints without an intervening reap must be
reap-bounded, and the plan's stated high-water must match the code's phase ordering.

## Workflow gap

- **Bug observed:** issue1586 r5 ENOSPC — plan modeled a per-cell ladder high-water
  (~456 GB) but the dispatcher trained all content cells before laddering, accumulating
  ~2.5 TB of full-FT checkpoints on a 750 GB volume; stream-reap lived only inside the
  ladder unit.
- **Why it is a workflow gap:** `plan-compute-sizing.md` models checkpoint-retention
  high-water + reap-between-cells (#1133, #653 r4) but does NOT require the high-water
  to be computed against the ACTUAL phase ordering, so a train-all-then-ladder ordering
  silently blows the budget the plan assumed and no plan-time critic flags it.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -niE "checkpoint|rung|retention|high-water|ENOSPC|full.?ft" .claude/rules/plan-compute-sizing.md` (2026-07-23) → the retention default (#1133) + full-FT ckpt high-water (#653 r4) exist, but the modeling is not tied to the implemented phase ordering nor requires reap on every accumulating phase.

## Proposed change (candidate diff sketch — refine in planning)

Extend `.claude/rules/plan-compute-sizing.md` (and the §9 disk lens the
efficiency-critic / critic Methodology-lens applies):

    The full-FT checkpoint disk high-water MUST be computed against the IMPLEMENTED
    phase ordering. Enumerate every phase that accumulates checkpoints without an
    intervening reap (e.g. a `train-all-cells` phase that precedes the ladder/select
    phase) and bound EACH — a reap that lives only inside the ladder read cannot bound
    an upstream train-all accumulation. The plan's stated high-water must match the
    code's phase ordering; a mismatch is a REVISE. Exemplar: #1586 r5
    (train-all-then-ladder → ~2.5 TB on a 750 GB volume → ENOSPC).

## Scope / surfaces

- Primary target: `.claude/rules/plan-compute-sizing.md`.
- If warranted, a one-line pointer in `.claude/agents/efficiency-critic.md` (or the v1
  `critic.md` Methodology lens §9/§10) so the plan-time critic checks phase-ordering vs
  the stated high-water.

## Constraints / invariants

- Workflow-surface doc/agent only; no experiment code (the #1586 code fix already
  landed on its branch as bounded-wave pipelining).
- `scripts/workflow_lint.py` passes; lessons index stays consistent if touched.
- Runs under `EPM_WORKFLOW_FIX_SESSION=1` (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/plan-compute-sizing.md
- fingerprint: 3c81f6385eaa

Origin: user chat 2026-07-23 ("Fix all the problems in the background with happycoder")
on the #1586 crash-history review. Related task: #1586 (r5 phase_ordering_rung_accumulation_enospc).
