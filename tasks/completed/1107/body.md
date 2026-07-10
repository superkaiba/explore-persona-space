---
title: 'workflow-fix: verify_task_body figure-meta vs what-is-plotted numeric reconcile'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0025bf0be5fb
created_at: '2026-07-07T09:20:47Z'
has_clean_result: false
origin_prompt: 'interp-critic prose follow-up on #825 (base-separator-control r1):
  reconcile bolded what-is-plotted numerics against figure meta.json plotted values;
  fingerprint 0025bf0be5fb'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced
by the interpretation-critic on task #825 (round `base-separator-control` r1).

## Goal

Extend verify_task_body.py's figure-sidecar staleness check to compare bolded what-is-plotted numerics in result prose against the figure meta.json bar/point values, flagging prose-vs-plotted-value mismatches.

## Workflow gap

- **Bug observed:** A result H3's prose+caption described full-n transfer fractions (0.057/0.109) while the pinned figure plotted matched-ceiling fractions (0.231 / −4.53 clipped) — the figure's own tick label disagreed with the caption. The current sidecar check compares rendered text only, so it PASSed; the mismatch was caught only by the interpretation-critic's PNG read.
- **Why it is a workflow gap:** figure meta.json sidecars already carry the plotted values (points/bars); nothing mechanically reconciles them against the bolded numerics in the adjacent what-is-plotted prose, so a figure/prose divergence survives to the LM-judgment layer.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up. Sketch: for each `### <result>` with an inline figure whose meta.json resolves at the cited SHA, extract bolded decimals from the beat-1 prose and WARN (not FAIL — matching is heuristic) when none of them appear within a tolerance among the meta's plotted values.)

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Consider WARN severity + a tolerance and a per-figure opt-out phrase; false-positive risk is high for derived/ratio numerics.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 0025bf0be5fb

Verbatim surfaced prose (interpretation-critic, #825 base-separator-control r1): "extend `verify_task_body.py`'s figure-sidecar staleness check to compare bolded what-is-plotted numerics against meta.json bar values (the request-1 mismatch class passes the current rendered-text-only check)."
