---
title: 'daily-held: reap stranded old-socket tmux server (~55 GiB)'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-19T07:10:59Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 3): tmux fleet-kill residuals
  — stranded old-socket server reap + Leg-3 verification confirm'
workflow: v1
---
## Overview / Motivation

Auto-filed by the 2026-07-18 /daily problem sweep (route 3 — destructive / human-gated). Sources: chunk-0 Problem 10, chunk-5 Problem 16 (logs/daily/mining-2026-07-18/).

## Goal

Reap the stranded pre-cutover shared-socket tmux server (~55 GiB RSS, left behind when mygoat-session.service moved to its dedicated `tmux -L mygoat` socket on 2026-07-18), and close out the one dangling thread from the 01:00 UTC fleet-kill: the killed interactive session's open Leg-3 delta-CI verification question (#825/#1345 work — "the report doesn't mention Leg 3 ... Verifying" at 00:58, never answered).

## Held item

- **What happened:** the mygoat-session.service 12h recycle on the #1466 shared tmux socket killed every tmux session on the VM at 01:00:48 UTC (3 live Claude sessions + Thomas's interactive tmux). Root cause was fixed the same day (dedicated `-L mygoat` socket); the dropped experiment ask was re-captured as #1489.
- **Why held (carve-out):** destructive / irreversible — killing a live (if orphaned) tmux server process; risk of taking a still-referenced session with it. Needs a human eyeball on `tmux -L <old-socket> ls` before the kill.
- **Suggested action:** verify nothing live is attached to the old server, `kill` it (frees ~55 GiB RSS), and confirm whether the Leg-3 verification was completed in a successor session or still needs a pass.

## Provenance

- source: /daily 2026-07-18 problem sweep (route 3)
- verified-at-filing: project memory `project_mygoat_tmux_socket_isolation.md` (2026-07-18) records the stranded `-L old` server pending cleanup; the fix (socket isolation) is live per the same memory + chunk-0 P10 evidence (2026-07-19)
