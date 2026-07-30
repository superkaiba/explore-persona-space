---
title: 'daily-held: add task.py set-backend (public API change)'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-29T07:18:37Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 3): sessions hand-edit backend:
  frontmatter because no task.py subcommand exists; adding one is a public CLI contract
  change requiring architectural greenlight'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-H P10. Held under the judgment-call carve-out mapped through the workflow rules: a **task.py subcommand addition is a public API contract change** (architectural greenlight is user-only).

## The held item

A 2026-07-28 session needed to change a task's `backend:` frontmatter and, finding no subcommand, fell back to a careful hand-edit of `body.md` (worked, but the canonical-API rule says all task state mutations go through `task.py`, and a less careful session could corrupt frontmatter). Recurring foot-gun.

## Suggested action

Greenlight one of: `task.py set-backend <N> <lane>`; or a generic `set-frontmatter-key <N> <key> <value>` with an allowlist. On greenlight this becomes a normal route-2/infra filing through the full pipeline.
