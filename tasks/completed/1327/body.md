---
title: 'daily-fix: spawn_session.py unregister subcommand'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b5946e2bdd12
- daily-auto-filed
created_at: '2026-07-15T06:52:13Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): spawn_session.py has register
  subcommands but no unregister — the #952 duplicate-session yield path had to hand-delete
  ~/.eps-autonomous/issue-952.json, and every collision-yield reimplements registry
  removal with crash-recovery risk if done wrong'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 problem sweep (session ae2aef04, #952, 05:24Z): "No such subcommand — removing the registry file directly (the watcher's own mechanism)".

## Goal

add an unregister subcommand (--issue <N> / --session-id) to scripts/spawn_session.py mirroring the registration write, for collision-yield and deliberate-stop paths

## Workflow gap

- **Bug observed:** spawn_session.py has register subcommands but no unregister — the #952 duplicate-session yield path had to hand-delete ~/.eps-autonomous/issue-952.json, and every collision-yield reimplements registry removal with crash-recovery risk if done wrong
- **Why it is a workflow gap:** the registry write has first-class affordances (`_register_autonomous_session` :495, `_register_manual_session` :583, `cmd_register_current` :2130) but removal has none, so yield paths hand-roll `rm` on watcher-owned state.
- **Confidence:** high
- verified-at-filing: `grep -n "unregister" scripts/spawn_session.py` -> 0 hits (absence claim; register functions present at :495/:583/:626/:707/:2130) (2026-07-15).

## Proposed change

`spawn_session.py unregister --issue <N>` (and/or `--session-id`) that removes the matching registry file(s) with a breadcrumb log line; wire the SKILL.md collision-yield text to use it (coordinate with the sibling filing "Step-0 guard blind to inline-chat drivers").

## Constraints

- Must be safe under the watcher's concurrent reads (atomic remove, tolerate missing); tests pin the subcommand; recursion guard applies.

## Provenance

- workflow_fix_target: scripts/spawn_session.py
- fingerprint: b5946e2bdd12
