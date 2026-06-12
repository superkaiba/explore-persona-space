---
title: 'Ops hygiene batch: bootstrap_pod.sh fixes, terminate pod-489, sweep zombie
  sessions, VM disk-space check cron'
kind: infra
tags:
- agent-ok
created_at: '2026-06-11T02:57:57Z'
has_clean_result: false
---
Small ops debts from June 9: bootstrap_pod.sh wants three fixes (symlink uv into /usr/local/bin for non-login shells, apt-get install rsync, -q on clone/checkout to stop 19,972-file progress dumps entering context); orphan pod-489 is stopped-not-terminated and billing storage (policy says terminate); ~40-51 zombie Happy sessions vs ~12 running issues; VM root disk pressure (43G uv cache) broke a worktree creation mid-session.
Actions: apply the three bootstrap_pod.sh fixes; terminate pod-489; sweep sessions via spawn_session.py list/stop and consider a weekly cron like the stale-pod audit; add a VM-disk check (alert under 20G free) to the daily cron set.
source: logs/daily/2026-06-09.md, approved by Thomas 2026-06-10 ('Apply these')
