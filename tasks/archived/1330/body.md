---
title: 'daily-fix: codify marker-text extraction from output file'
kind: infra
tags:
- wf-fix
- wf-fix-fp:058151c39890
- daily-auto-filed
created_at: '2026-07-15T06:52:25Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): subagent completion-notification
  bodies arrive HTML-escaped, so orchestrators must re-extract clean text from the
  agent''s output file before posting markers — the recipe is session folklore (two
  sessions independently re-derived it on 07-14), a per-round tax with latent risk
  of posting escaped text into events.jsonl'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 problem sweep (sessions 49b538f6/#1309 08:13Z and ccb16540/#1090 05:49Z — both independently noted "notification bodies arrive HTML-escaped" and worked around it).

## Goal

codify in the issue SKILL.md marker-posting guidance: always extract marker text from the agent's output file / durable artifact, never from the notification body (HTML-escaped)

## Workflow gap

- **Bug observed:** subagent completion-notification bodies arrive HTML-escaped, so orchestrators must re-extract clean text from the agent's output file before posting markers — the recipe is session folklore (two sessions independently re-derived it on 07-14), a per-round tax with latent risk of posting escaped text into events.jsonl
- **Why it is a workflow gap:** SKILL.md's marker-posting guidance never states the escaped-notification hazard for MARKER text; posting an escaped body corrupts events.jsonl rendering.
- **Confidence:** medium
- verified-at-filing: `grep -n "HTML-escaped" .claude/skills/issue/SKILL.md` -> 1 hit at :2018, a DIFFERENT context (harness escaping of `&&`/`<`/`>` in command text), not the notification-body marker-extraction recipe (presence probe + absence-of-recipe claim) (2026-07-15).

## Proposed change

One line near the marker-posting / subagent-join guidance: "Subagent completion-notification bodies arrive HTML-escaped — when posting any marker from a subagent's verdict, extract the text from its durable output file (the /tmp verdict file or artifact), never from the notification body."

## Constraints

- Workflow-surface only; recursion guard applies.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 058151c39890
