---
title: 'daily-fix: pod-side liveness probe before LAUNCH FAILED'
kind: infra
tags:
- wf-fix
- wf-fix-fp:068a96affa5c
- daily-auto-filed
created_at: '2026-08-06T07:06:03Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): launch wrapper wrote false
  LAUNCH FAILED twice while pod-side runs were alive; heartbeat wrote UNREACHABLE
  for verified-terminated pods'
workflow: v1
---
# daily-fix: pod-side-reporting — launch wrappers and heartbeats must probe pod-side liveness before declaring failure/unreachable

## Workflow gap

Two same-day #1739 fan-out patterns wrote false negatives into durable/driver records
because the reporting layer keyed on the SSH channel instead of pod-side state:

1. **Launch-wrapper false "LAUNCH FAILED".** The 6-pod jobd/r2aug fan-out's launch ssh
   hung (~35 min for jobd-evil), was killed, returned non-zero — and the wrapper wrote
   "pod-1739-jobd-evil LAUNCH FAILED" while the detached pod-side run was alive (pid 1885
   verified). Recurred for jobd-sycophancy at 07:48Z. The teammate shipped a session-local
   v2 driver with a 60 s launch-watchdog that probes the pod-side leg before reaping hung
   sshs — the recipe exists but lives only in that session. A sibling instance the same
   day: pod-1739-a1apilot "LAUNCH FAILED (ssh_rc=124, no live pod-side proc)" needed a
   manual re-pin + relaunch (that one was a REAL failure — the probe is what tells them
   apart).
2. **Heartbeat false "UNREACHABLE — investigate".** The R5 wall-clock heartbeat flagged
   terminated-after-verified-upload pods as "UNREACHABLE (ssh rc=255) — investigate
   before assuming healthy" twice (07:45Z, 10:01Z); the 10:02Z heartbeat corrected the
   copy to "TERMINATED (verified upload) … NOT a fault" by checking the termination
   sentinel first.

verified-at-filing: all quotes are probed tool_result/driver-log readbacks from session
21e049f7 (rows 4113–4260, 4222/4556/4569) and 2f4940f0 (rows 271–328).
`grep -n 'LAUNCH FAILED\|liveness' .claude/rules/pod-side-reporting.md | head` run at
compose time — the (re)launch contract covers pid-file/log/sentinel duties but has no
"probe before declaring failure" clause for wrappers/heartbeats.

## Proposed change

Add to `.claude/rules/pod-side-reporting.md` (the (re)launch contract section): a launch
wrapper never writes LAUNCH FAILED without a pod-side pidfile/process probe (the 60 s
launch-watchdog pattern: ssh channel outcome ≠ leg outcome), and a heartbeat classifies
ssh rc=255 against the termination sentinel / `pod.py list-ephemeral` before writing
UNREACHABLE. Name the v2-driver watchdog as the reference implementation so the next
fan-out inherits it instead of re-deriving it.

## Provenance

- fingerprint: 068a96affa5c

- workflow_fix_target: .claude/rules/pod-side-reporting.md
- origin: /daily 2026-08-05 problem sweep — miner 2 P2/P21.
