---
title: 'daily-fix: teammate idle handling branches on report channel'
kind: infra
tags:
- wf-fix
- wf-fix-fp:919be8b1f320
- daily-auto-filed
created_at: '2026-07-28T07:02:50Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): 3 of 3 fan-out Explore
  teammates in one session finished, went idle, and each ate a SendMessage nudge +
  an extra wake turn — but their spawn briefs had requested the report as FINAL TEXT
  (no SendMessage delivery), so the reports already existed as Agent results; the
  orchestrator had TaskOutput loaded and nudged instead of reading'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session 7beffce7, 2026-07-28T00:24-00:29Z (miner A P2, probed: all 3 spawn briefs contain no SendMessage-delivery request).

## Goal

Branch the idle-teammate handling on the report channel the brief actually declared.

## Workflow gap

- **Bug observed:** three background Explore teammates finished their sweeps and idled; the orchestrator sent one 'you went idle without delivering your report' nudge per teammate and each woke an extra turn to resend — ~3-9 min extra wall per report + 6 extra turns. The briefs said 'Your final text is data for the orchestrator' — the reports were already delivered as Agent results.
- **Why it is a workflow gap:** bullet (d) sequences 'nudge ONCE, then read the Agent result' unconditionally; for final-text briefs the nudge is pure waste — the rule's ordering costs a wake-turn round-trip per teammate whenever the brief used the final-text channel.
- **Confidence (emitter):** medium
- verified-at-filing: miner-probed this run: spawn prompts at transcript rows 20/22/24 contain no SendMessage-report request; `ToolSearch select:SendMessage,TaskOutput` loaded at row 36 with no subsequent TaskOutput call; nudges at rows 41/47/58. CLAUDE.md bullet (d) text confirmed at compose time (grep 'idle notification is NOT a done').

## Proposed change (candidate diff sketch — refine in planning)

In project `CLAUDE.md` § 'Orchestrator vs subagent re-invocation', teammate bullet (d): add the channel branch — final-text briefs: idle = completion, read the Agent result (TaskOutput) directly, no nudge; SendMessage briefs: current nudge-once-then-read contract unchanged.

## Scope / surfaces

- Primary target: `CLAUDE.md` (teammate-coordination bullet (d))

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 919be8b1f320

- workflow_fix_target: CLAUDE.md
