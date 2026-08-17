---
title: 'gotchas.md: MooseFS preflight fallocate canary grinds ~9 min (slow-canary
  vs read-wedge discriminator) + resumed-pod uv sync flash-attn uninstall note'
kind: infra
tags: []
created_at: '2026-08-17T06:43:23Z'
has_clean_result: false
workflow: v1
---
Surfaced by the #2333 leg-A relaunch experimenter (epm:failure-lesson v1, gotcha_candidate: yes, root_cause_confirmed: yes):

1. On a MooseFS-backed RunPod pod, preflight's 1 GB posix_fallocate disk canary can grind ~9 min in glibc's per-block write emulation (FUSE without fallocate support). Signature: main thread wchan=request_wait_answer with zero sockets and NO open data fds, while spot reads/writes/statfs on /workspace stay fast. That is a SLOW CANARY, not the MooseFS read-wedge — wait it out (poll the done file) instead of killing/reprovisioning. gotchas.md's MooseFS read-wedge entry needs this discriminator added so sessions do not misroute a slow canary to the wedge runbook.

2. Companion note: `uv sync --locked` on a resumed pod uninstalls out-of-lock flash-attn; grep the driver for attn_implementation/flash refs before spending 10+ min reinstalling — default-attention loads do not need it.

Target file: .claude/rules/gotchas.md (MooseFS FUSE read-wedge entry + a resumed-pod env note). Evidence: task #2333 events (epm:run-launched v3 experimenter report, 2026-08-17T06:4xZ).
