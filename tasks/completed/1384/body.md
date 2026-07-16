---
title: 'workflow-fix: implementer per-arm-class smoke clause'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d6ebd14751d4
- daily-auto-filed
created_at: '2026-07-16T07:19:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): multi-arm-class driver
  may ship single-arm smoke coverage'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1340 (emitting agent: Phase-2 Alternatives critic). #1340 itself shipped the gotchas.md per-arm-class smoke bullet; this candidate is the implementer-side checklist half it deliberately left per-PHASE-only.

## Goal

Add a one-line per-arm-class smoke-coverage clause to `.claude/agents/experiment-implementer.md` § "End-to-end smoke run PER PHASE", so a driver spanning multiple arm classes does not ship single-arm smoke coverage and rely solely on the Step 6d.0-bis gate bounce.

## Workflow gap

- **Bug observed:** the implementer-side smoke checklist is per-PHASE-only — a driver spanning multiple arm classes may still ship single-arm coverage and rely on the Step 6d.0-bis gate bounce (recoverable, not fatal — surfaced on #1340).
- **Why it is a workflow gap:** the smoke duty lives in the implementer's own checklist; catching per-arm-class blindness only at the downstream gate wastes a bounce round.
- **Confidence (emitter):** low-medium
- verified-at-filing: `grep -n 'arm.class\|arm class' .claude/agents/experiment-implementer.md` → 0 hits (absence-of-guard claim — the 0-hit in-target result IS the evidence); `grep -cn 'smoke run\|smoke-run' .claude/agents/experiment-implementer.md` → 8 hits (the § exists to extend) (2026-07-16 UTC)

## Proposed change (candidate diff sketch — refine in planning)

One clause in § End-to-end smoke run PER PHASE: when a phase's driver spans multiple ARM CLASSES (distinct code paths per arm), the smoke covers at least one cell of EACH arm class, not one arm overall; optionally keep the Step 6d.0-bis mechanical refuse-enumeration phase-keyed (unchanged).

## Scope / surfaces

- Primary target: `.claude/agents/experiment-implementer.md`
- Cross-check the just-landed gotchas.md bullet (#1340, commit f8f350fddf) for consistent vocabulary.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: d6ebd14751d4

- workflow_fix_target: .claude/agents/experiment-implementer.md

parked prose follow-up (verbatim, from #1340 events.jsonl 2026-07-15T15:08:32Z): "source: prose-followup (Phase-2 Alternatives critic, task #1340). Candidate: add a one-line per-arm-class smoke-coverage clause to .claude/agents/experiment-implementer.md § 'End-to-end smoke run PER PHASE' (implementer-side checklist stays per-PHASE-only; a driver spanning multiple arm classes may still ship single-arm coverage and rely on the Step 6d.0-bis gate bounce), optionally + the Step 6d.0-bis mechanical refuse-enumeration staying phase-keyed. Recoverable follow-up, not fatal to #[1340]."
