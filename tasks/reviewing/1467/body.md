---
title: 'daily-fix: rev-parse SHAs cited in /daily filing bodies'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9608dfe5771a
- daily-auto-filed
created_at: '2026-07-17T06:58:29Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): a /daily-filed body shipped
  a non-resolving commit SHA (fc2b61b7 — actually a transcript basename, not a commit)
  into task #1414''s plan inputs; the fact-checker burned a round proving the real
  fix commit was 5a02359cc8 — the standing never-fabricate-SHA rule, violated by the
  nightly filer'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#1414 session 4c54094d, ~23:00Z).

## Goal

Stop the nightly filer from shipping non-resolving SHAs into task bodies.

## Workflow gap

- **Bug observed:** a /daily-filed body shipped a non-resolving commit SHA (fc2b61b7 — actually a transcript basename, not a commit) into task #1414's plan inputs; the fact-checker burned a round proving the real fix commit was 5a02359cc8 — the standing never-fabricate-SHA rule, violated by the nightly filer
- **Why it is a workflow gap:** Filed bodies seed planner/fact-checker rounds; a fabricated-looking SHA costs a verification round every time.
- **Confidence (emitter):** medium-high (incident today)
- verified-at-filing: `grep -c 'rev-parse' .claude/skills/daily/SKILL.md` -> 0 (no SHA-verify duty — absence claim); incident: #1414 fact-checker WRONG-fact verdict on the inherited SHA

## Proposed change (candidate diff sketch — refine in planning)

add to the /daily route-2 composition step: run git rev-parse --verify on every commit SHA cited in a filing body at compose time (extend the verified-at-filing duty to SHAs); an unverifiable hex string is cited as a transcript/session reference, never as a commit

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 9608dfe5771a

