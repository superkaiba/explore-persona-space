---
title: 'daily-held: Codex quota out to Aug 6 - pay or ride it out'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-08T07:00:22Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 3): Every codex_task.py dispatch
  since ~17:00Z 2026-07-07 fails with a hard usage-limit error (reset 2026-08-06 6:26
  AM). All Codex twins (critic x3 lenses, code-reviewer, interpretation-critic, clean-result-critic,
  follow-up-critic) no-show; every doubled review site runs Claude-only fallback.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 3) from the nightly transcript problem sweep.

## Goal

Thomas decides: top up / upgrade the OpenAI Codex plan now (restores cross-family review, costs money) or explicitly accept single-family Claude-only ensemble review until the Aug 6 reset

## Workflow gap

- **Bug observed:** Every codex_task.py dispatch since ~17:00Z 2026-07-07 fails with a hard usage-limit error (reset 2026-08-06 6:26 AM). All Codex twins (critic x3 lenses, code-reviewer, interpretation-critic, clean-result-critic, follow-up-critic) no-show; every doubled review site runs Claude-only fallback.
- **Why it is a workflow gap:** Spends money / external account decision — route-3 carve-out. CLAUDE.md calls cross-family reviewer diversity "the strongest oversight asset".

## Proposed change

Held for Thomas. The companion route-2 task (codex-quota-sentinel) stops sessions re-discovering the outage; this task is the account decision itself.

## Scope / surfaces

- Primary target: `external (OpenAI account)`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: 8+ sessions on 2026-07-07/08; first hit 96a2b6d3 (#1111) 17:02Z.
