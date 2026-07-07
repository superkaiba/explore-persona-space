---
title: 'daily-fix: digest-only raw-read rule must name real-world-co'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7f54a4d27e30
- daily-auto-filed
created_at: '2026-07-07T06:49:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-06 problem sweep (route 2): The #1073 analyzer was
  refusal-killed twice (first pass 11:43, round-2 12:40, session c3655287, 2026-07-06)
  after paging raw LMSYS rollout text — real-world-corpus completions carry in-corpus
  jailbreak/explicit content. The digest-only raw-completion read rule (CLAUDE.md
  spurious-refusals item (d)) names harmful-content banks and EM completions but not
  real-world-corpus (LMSYS-class) rollout text,'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-06 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Extend the digest-only raw-completion read rule (analyzer.md + CLAUDE.md item (d)) to explicitly cover real-world-corpus rollout text (LMSYS-class), so first-pass briefs carry digest-only reads before the first kill.

## Workflow gap

- **Bug observed:** The #1073 analyzer was refusal-killed twice (first pass 11:43, round-2 12:40, session c3655287, 2026-07-06) after paging raw LMSYS rollout text — real-world-corpus completions carry in-corpus jailbreak/explicit content. The digest-only raw-completion read rule (CLAUDE.md spurious-refusals item (d)) names harmful-content banks and EM completions but not real-world-corpus (LMSYS-class) rollout text, so first-pass analyzer briefs only pick the discipline up in the post-kill retry brief.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/agents/analyzer.md, CLAUDE.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: .claude/agents/analyzer.md, CLAUDE.md
- source: /daily 2026-07-06 problem sweep (transcript-mined)
