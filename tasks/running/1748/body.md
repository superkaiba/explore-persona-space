---
title: 'daily-fix: trigger-dense datagen first-pass decomposition'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0b1ae29b26b1
- daily-auto-filed
created_at: '2026-07-28T07:00:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): #1739 (multi-behavior real-corpus
  datagen) lost 9 subagents in one session — 4 usage-policy refusal kills (one at
  127 calls) + 5 autocompact-thrash deaths — ~5h of spawn attrition before converging
  on the working decomposition (data-plane code rounds with zero real-corpus streaming
  + bounded content-opaque ingestion probes the orchestrator runs, counts-only stdout)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session bae92cbd (#1739), 2026-07-28T01:14-06:08Z (miner J P2).

## Goal

Real-corpus datagen briefs should START from the decomposition #1739 converged on after ~5h of subagent attrition.

## Workflow gap

- **Bug observed:** one /issue session lost 2 planner spawns to refusal, 1 sonnet planner + consistency-checker + implementer spawn to thrash, round B to refusal (33 calls), its sonnet respawn to thrash, C2 to refusal at 127 calls, and C2b to thrash — the CLAUDE.md refusal/thrash ladders recovered it, but the working configuration (zero real-corpus text in subagent context; ingestion probes as content-opaque CLIs run by the ORCHESTRATOR with counts-only stdout; micro-scoped code rounds; orchestrator-composed report markers) was only reached as the round-4 fallback.
- **Why it is a workflow gap:** `.claude/rules/trigger-dense-review.md` covers brief vocabulary/neutralization but has no real-corpus DATAGEN decomposition default (`grep -c 'datagen\|real-corpus' .claude/rules/trigger-dense-review.md` -> 0, compose time).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'datagen\|real-corpus' .claude/rules/trigger-dense-review.md` -> 0, compose time 2026-07-28.

## Proposed change (candidate diff sketch — refine in planning)

Add a brief-composition subsection to `.claude/rules/trigger-dense-review.md`: for real-corpus / harmful-bank datagen implementer briefs, the FIRST-PASS round decomposition is (a) data-plane code rounds that stream NO real-corpus text; (b) bounded ingestion probes shipped as content-opaque CLIs the ORCHESTRATOR runs (counts-only stdout, fail-loud kept=0); (c) micro-scoped rounds sized to survive thrash; (d) report markers orchestrator-composed from durable evidence. Cite #1739's attrition as the incident.

## Scope / surfaces

- Primary target: `.claude/rules/trigger-dense-review.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 0b1ae29b26b1

- workflow_fix_target: .claude/rules/trigger-dense-review.md
