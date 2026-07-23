---
title: 'daily-fix: research-pm.md over 47000-byte ratchet cap, main '
kind: infra
tags:
- wf-fix
- wf-fix-fp:eead9cd66a72
- daily-auto-filed
created_at: '2026-07-23T06:39:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): research-pm.md regrew to
  47861 bytes over its 47000-byte workflow_lint ratchet cap, failing tests/test_workflow_lint.py
  on pristine origin/main fleet-wide'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-22 parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1611 (emitting agent: implementer report). FLEET-BLOCKING: the ratchet test is red on pristine origin/main.

## Goal

Bring `.claude/agents/research-pm.md` back under its 47,000-byte `workflow_lint.py` grandfather ratchet cap (per the #829 relocate-to-rules pattern), or deliberately raise the cap with a recorded justification, so `tests/test_workflow_lint.py` passes on pristine main again.

## Workflow gap

- **Bug observed:** `research-pm.md` regrew to 47,861 bytes > its 47,000-byte `workflow_lint.py` grandfather ratchet cap, failing `tests/test_workflow_lint.py` (1 failed / 527 passed per the #1611 implementer report) on pristine origin/main fleet-wide — every Step-9c gate that selects that test now sees a pre-existing red.
- **Why it is a workflow gap:** the ratchet exists to force deliberate size decisions on always-loaded agent files; an over-cap file left red poisons the shared test oracle for every concurrent session.
- **Confidence (emitter):** high
- verified-at-filing: `stat -c %s .claude/agents/research-pm.md` → 47861 bytes; `grep -n 'research-pm' scripts/workflow_lint.py` → cap entry `"research-pm.md": 47_000` at scripts/workflow_lint.py:11165 (presence claim, both targets bind), 2026-07-23 UTC.

## Proposed change (candidate diff sketch — refine in planning)

Trim `research-pm.md` back under the cap by relocating per-scenario content to `.claude/rules/` (the #829 pattern), OR raise the cap with a `Ratchet budget:` justification line. The planner decides which; the trim is preferred (the cap exists for context-load reasons).

## Scope / surfaces

- Primary target: `.claude/agents/research-pm.md`
- Secondary (only if the raise path is chosen): `scripts/workflow_lint.py` cap table + its test.

## Constraints / invariants

- Workflow-surface only. `uv run pytest tests/test_workflow_lint.py` green after the change; no PM behavioral content silently dropped (relocated content stays reachable via `.claude/rules/` pointers).
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: eead9cd66a72

- workflow_fix_target: .claude/agents/research-pm.md

Verbatim parked candidate (prose park, task #1611 events 2026-07-23T04:37:15Z):

> parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see workflow-fix-on-bug.md § Recursion guard. source: prose-followup (implementer report). target_file: .claude/agents/research-pm.md. bug_observed: research-pm.md regrew to 47861 bytes > its 47000-byte workflow_lint.py grandfather ratchet cap, failing tests/test_workflow_lint.py (1 failed / 527 passed) on pristine origin/main fleet-wide. proposed_change: trim research-pm.md back under its ratchet cap per the #829 pattern (relocate per-scenario content to .claude/rules/), or deliberately raise the cap with a 'Ratchet budget:' justification. confidence: high. related_task: #1611.
