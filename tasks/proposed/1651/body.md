---
title: 'daily-fix: implementer pin-sweep states sweep_scope'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5fecf8e6f9c1
- daily-auto-filed
created_at: '2026-07-24T06:47:49Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): the implementer report
  pin-sweep field does not state its sweep universe so a narrow sweep can read as
  exhaustive'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-23 parked-candidate routing pass (Step C). Raised as a recursion-guard-parked prose follow-up on task #1634 (code-reviewer round 1, Minor, 18:24Z).

## Goal

Extend the implementer report contract so the pin-sweep field states its sweep UNIVERSE (selector-universe vs repo-wide) — a 1-line `sweep_scope:` token in `.claude/agents/experiment-implementer.md` and `.claude/agents/implementer.md`.

## Workflow gap

- **Bug observed:** the implementer report's pin-sweep field does not state which universe was swept (selector-universe vs repo-wide), so a reviewer cannot tell a narrow sweep from an exhaustive one without re-running it.
- **Why it is a workflow gap:** the report contract is the reviewer's evidence surface; an ambiguous sweep scope forces re-derivation or lets a narrow sweep read as exhaustive (the same claim-binding discipline as the verified-at-filing per-target rule).
- **Confidence (emitter):** medium (flagged Minor by the emitting code-reviewer; mechanizable as a 1-line token check)
- verified-at-filing: `grep -n "sweep_scope" .claude/agents/experiment-implementer.md .claude/agents/implementer.md` → 0 hits in both named targets (absence claim, in-target 0-hit is the evidence) (2026-07-24 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Add a required `sweep_scope: selector-universe | repo-wide` token to the pin-sweep field spec in both agent files (optionally a code-reviewer check that the token is present).

## Scope / surfaces

- Primary target: `.claude/agents/experiment-implementer.md`, `.claude/agents/implementer.md`

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 5fecf8e6f9c1

- workflow_fix_target: .claude/agents/experiment-implementer.md, .claude/agents/implementer.md

Origin: parked candidate on #1634 (2026-07-23T18:24:03Z, item 2).
