---
title: 'daily-fix: helper-authenticated fetch in experimenter recipe'
kind: infra
tags:
- wf-fix
- wf-fix-fp:de1b08f100fd
- daily-auto-filed
created_at: '2026-07-11T06:52:22Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): experimenter.md:254 (and
  one agent-memory file) still prescribe token-in-URL git fetches (''https://x-access-token:$TOK@github.com/...'')
  for private-branch fetches; #1239 shipped the bootstrap_pod.sh credential-helper
  (no token-at-rest in the remote URL) but the experimenter recipes were outside its
  Goal'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

simplify the ephemeral tokenized-fetch recipes to helper-authenticated fetches consistent with the #1239 credential-helper contract

## Workflow gap

- **Bug observed:** experimenter.md:254 (and one agent-memory file) still prescribe token-in-URL git fetches ('https://x-access-token:$TOK@github.com/...') for private-branch fetches; #1239 shipped the bootstrap_pod.sh credential-helper (no token-at-rest in the remote URL) but the experimenter recipes were outside its Goal
- **Provenance / evidence:** Parked observation from the #1239 session (miner-01 P8.3), /daily 2026-07-10 sweep. Verified live: experimenter.md:254 still token-in-URL.

## Scope / surfaces

- Primary target: `.claude/agents/experimenter.md, .claude/agent-memory/experimenter/`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: de1b08f100fd

- workflow_fix_target: .claude/agents/experimenter.md, .claude/agent-memory/experimenter/
