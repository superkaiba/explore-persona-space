---
title: 'daily-fix: move tmux sockets off /tmp (split-brain)'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-17T06:58:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the tmux server socket
  /tmp/tmux-1001/default was deleted on the 116-day-uptime VM (root cause unidentified
  — check systemd-tmpfiles age rules and the #911 non-canonical /tmp sweep''s match
  set), so the 06:00 mygoat cron silently started a SECOND tmux server and 39 sessions
  became invisible to tmux ls (user report ~17:50-18:00Z, chats c07f01a6 + 108c810d);
  recovered by rebinding the old socket as /'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (two chats hit it independently). Non-workflow-surface (system/shell config + cron wrappers). Root-cause identification is part of the task: rule out the #911 /tmp sweep as the deleter before excluding paths.

## Goal

Make tmux server sockets survive /tmp cleaning on the long-uptime VM, and identify what deleted the socket.

## Workflow gap

- **Bug observed:** the tmux server socket /tmp/tmux-1001/default was deleted on the 116-day-uptime VM (root cause unidentified — check systemd-tmpfiles age rules and the #911 non-canonical /tmp sweep's match set), so the 06:00 mygoat cron silently started a SECOND tmux server and 39 sessions became invisible to tmux ls (user report ~17:50-18:00Z, chats c07f01a6 + 108c810d); recovered by rebinding the old socket as /tmp/tmux-1001/old
- **Why it is a workflow gap:** The program orchestrator, mygoat crons, and manual sessions all assume one tmux server; a /tmp-cleaned socket silently forks the world.
- **Confidence (emitter):** high (incident today; recovery confirmed; deleter unidentified)
- verified-at-filing: incident + recovery in chats c07f01a6/108c810d (39 sessions on the rebound socket); n/a for a repo grep — target is system config + cron wrappers (`grep -rln tmux scripts/cron_*.sh` enumerates wrappers at planning time)

## Proposed change (candidate diff sketch — refine in planning)

export TMUX_TMPDIR=$HOME/.tmux-sockets in the shell profile AND every cron wrapper that touches tmux; identify + exclude the /tmp cleaner that removed the socket; document the recovery

## Scope / surfaces

- Primary target: `shell profile + scripts/cron_*.sh`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Non-workflow-surface fix (`wf_fix: false`): no recursion guard applies; standard /issue pipeline.

