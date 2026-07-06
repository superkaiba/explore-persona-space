---
title: 'workflow-fix: verifier WARN on stale concern-deferred markers naming addressed
  concerns'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a3f925bd4fab
created_at: '2026-07-06T16:01:05Z'
has_clean_result: false
origin_prompt: 'clean-result-critic + codex twin v3 on #833, agreed procedural item
  (mechanizable: yes)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a mechanizable finding raised on task #833 (emitting agents: clean-result-critic epm:clean-result-critique v3 + codex-clean-result-critic epm:clean-result-critique-codex v3, agreed procedural item).

## Goal

Add a verifier WARN when a `<!-- concern-deferred: <id> -->` body marker names a concern whose latest `concerns.jsonl` event is `addressed`.

## Workflow gap

- **Bug observed:** #833's body carried three stale `<!-- concern-deferred: ... -->` HTML markers after all three named concerns had transitioned to `addressed` — the markers misdescribed the resolution (deferred vs addressed) and nothing mechanical flagged them; both LM critics had to catch it as a procedural item.
- **Why it is a workflow gap:** the concerns-audit check (check 14 family) validates acknowledgment of OPEN concerns but is silent on markers that outlive their concern's resolution, so every fixed-concern round leaves stale markers behind by default.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

+ In verify_task_body.py's concerns-audit check: for each `<!-- concern-deferred: (?P<id>[a-z0-9-]+) -->` in the body, look up the id in the task's concerns ledger;
+ latest event == "addressed" (or id absent from ledger) -> WARN "stale concern-deferred marker '<id>' — concern is addressed; remove or retag";
+ latest event deferred/open -> current behavior unchanged. WARN-only (never a hard FAIL; forward-only).

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Update `tests/test_verify_task_body.py` with a stale-marker fixture.

## Constraints / invariants

- Workflow-surface only; WARN-only; `workflow_lint.py` + ruff pass.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: a3f925bd4fab

Origin: clean-result-critic + codex-clean-result-critic v3 on #833, agreed procedural fix ("mechanizable: yes — a verifier WARN when a concern-deferred marker names a concern whose latest ledger event is addressed").
