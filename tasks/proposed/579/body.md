---
title: 'codex_task.py: add one auto-retry with backoff and verify the E2BIG file-transport
  fix merged to main'
kind: infra
tags:
- agent-ok
created_at: '2026-06-11T02:57:56Z'
has_clean_result: false
---
~10 Codex companion runtime incidents in one day (stall force-cancels at 607-730s, app-server exit 1, instant 0s failures, exit 4/5/8 probe-registry errors), all recovered via manual retry or the Claude-only fallback. Separately #540 hit E2BIG dispatching a 176KB prompt via argv; a file/stdin transport fix was dispatched in-session.
Actions: add one auto-retry with backoff inside scripts/codex_task.py and watch incident frequency; verify the E2BIG file/stdin transport fix actually merged to main.
source: logs/daily/2026-06-09.md, approved by Thomas 2026-06-10 ('Apply these')
